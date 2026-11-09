import Foundation
import MLX
import MLXDiffusionCore

/// The Wan 2.1 video VAE (z16, 4× temporal and 8× spatial compression), as the official `WanVAE_`:
/// causal 3D convolutions, run chunk by chunk with a feature cache — the encoder takes the first frame alone and
/// then 4 frames at a time; the decoder takes one latent frame at a time, and its first frame is not temporally
/// upsampled, so `T` latent frames decode to `1 + 4·(T−1)` video frames.
///
/// Tensors are channel-last here (`[B, T, H, W, C]`), float32 throughout like the official VAE.
public final class WanVAE {
    /// Per-channel latent statistics (official `Wan2_1_VAE.mean` / `.std`).
    public static let mean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                       0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    public static let std: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                      3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    public static let zDim = 16

    let decoder: Decoder
    let encoder: Encoder?
    let conv1: CausalConv3d?  // after the encoder: 32 → 32 (mu, log_var)
    let conv2: CausalConv3d  // before the decoder: 16 → 16

    /// `url`: the VAE safetensors (official key names; kernels in PyTorch or MLX layout).
    public init(url: URL, encoder withEncoder: Bool) throws {
        let store = try WeightStore(url: url)
        // The mlx-video conversion stores kernels channel-last ([O, kt, kh, kw, I]); official files are [O, I, kt, kh, kw].
        let dc = try store.get("decoder.conv1.weight")
        let mlxLayout = dc.dim(4) == Self.zDim && dc.dim(1) != Self.zDim
        let l = Loader(store: store, mlxLayout: mlxLayout)
        decoder = try Decoder(l)
        conv2 = try l.conv3d("conv2", padding: (0, 0))
        if withEncoder {
            encoder = try Encoder(l)
            conv1 = try l.conv3d("conv1", padding: (0, 0))
        } else {
            encoder = nil
            conv1 = nil
        }
    }

    /// Normalized latents `[1, T, h, w, 16]` → video `[1, 1+4(T−1), 8h, 8w, 3]` in [-1, 1].
    /// `onFrame(i, T)` is called after each latent frame.
    public func decode(_ z: MLXArray, onFrame: ((Int, Int) throws -> Void)? = nil) rethrows -> MLXArray {
        let m = MLXArray(Self.mean), s = MLXArray(Self.std)
        let x = conv2(z.asType(.float32) * s + m, cache: nil)
        let cache = FeatCache()
        var frames: [MLXArray] = []
        let T = x.dim(1)
        for i in 0..<T {
            cache.idx = 0
            let out = decoder(x[0..., i ..< i + 1], cache)
            eval(out)
            frames.append(out)
            Memory.clearCache()
            try onFrame?(i + 1, T)
        }
        return clip(concatenated(frames, axis: 1), min: -1, max: 1)
    }

    /// Video `[1, T, H, W, 3]` in [-1, 1] (T = 1 + 4k) → normalized latent mean `[1, 1+k, H/8, W/8, 16]`.
    public func encode(_ video: MLXArray) -> MLXArray {
        guard let encoder, let conv1 else { fatalError("WanVAE was loaded without its encoder") }
        let cache = FeatCache()
        let T = video.dim(1)
        var outs: [MLXArray] = []
        for i in 0 ..< 1 + (T - 1) / 4 {
            cache.idx = 0
            let chunk = i == 0 ? video[0..., 0 ..< 1] : video[0..., (1 + 4 * (i - 1)) ..< (1 + 4 * i)]
            let o = encoder(chunk.asType(.float32), cache)
            eval(o)
            outs.append(o)
            Memory.clearCache()
        }
        let mu = conv1(concatenated(outs, axis: 1), cache: nil)[.ellipsis, 0 ..< Self.zDim]
        return (mu - MLXArray(Self.mean)) / MLXArray(Self.std)
    }
}

// MARK: - building blocks

/// Per-conv cache of the last frames of the previous chunk (`feat_cache` / `feat_idx` in the reference).
final class FeatCache {
    enum Slot { case rep, frames(MLXArray) }
    var slots: [Int: Slot] = [:]
    var idx = 0
}

struct Loader {
    let store: WeightStore
    let mlxLayout: Bool

    func conv3d(_ p: String, padding: (t: Int, s: Int), temporalStride: Int = 1) throws -> CausalConv3d {
        var w = try store.get(p + ".weight").asType(.float32)
        if !mlxLayout { w = w.transposed(0, 2, 3, 4, 1) }
        return CausalConv3d(weight: w, bias: try store.get(p + ".bias").asType(.float32), timePad: 2 * padding.t,
                            spatialPad: padding.s, temporalStride: temporalStride)
    }

    func conv2d(_ p: String) throws -> (MLXArray, MLXArray) {
        var w = try store.get(p + ".weight").asType(.float32)
        if w.ndim == 5 { w = mlxLayout ? w.squeezed(axis: 1) : w.squeezed(axis: 2) }
        if !mlxLayout { w = w.transposed(0, 2, 3, 1) }
        return (w, try store.get(p + ".bias").asType(.float32))
    }

    func gamma(_ p: String) throws -> MLXArray { try store.get(p + ".gamma").asType(.float32).reshaped(-1) }
}

/// Causal 3D convolution: zero-pads `timePad` frames in front (or prepends cached frames instead), pads space
/// symmetrically, then convolves without padding.
struct CausalConv3d {
    let weight: MLXArray  // [O, kt, kh, kw, I]
    let bias: MLXArray
    let timePad: Int
    let spatialPad: Int
    let temporalStride: Int

    func callAsFunction(_ x: MLXArray, cache: MLXArray?) -> MLXArray {
        var x = x
        var padT = timePad
        if let cache, timePad > 0 {
            x = concatenated([cache, x], axis: 1)
            padT -= cache.dim(1)
        }
        if padT > 0 || spatialPad > 0 {
            let s = spatialPad
            x = padded(x, widths: [.init(0), .init((max(padT, 0), 0)), .init(s), .init(s), .init(0)])
        }
        // One 2D convolution per temporal tap, summed: MLX's conv3d unrolls the whole kt·kh·kw·C patch at full
        // resolution (tens of GB at 832×480), while 2D convolutions stream it.
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let kt = weight.dim(1), st = temporalStride
        let tOut = (T - kt) / st + 1
        // Large frames go one output frame at a time, which also bounds the 2D convolution's own patch buffer.
        let perFrame = B * tOut > 1 && H * W * C * weight.dim(2) * weight.dim(3) > 64 << 20
        func frameOut(_ j: Int, _ count: Int) -> MLXArray {
            var y: MLXArray?
            for k in 0..<kt {
                let frames = st == 1 ? x[0..., (k + j) ..< (k + j + count)]
                    : take(x, MLXArray((0..<count).map { Int32(k + st * (j + $0)) }), axis: 1)
                let o = bandedConv2d(frames.reshaped(B * count, H, W, C), weight[0..., k])
                y = y.map { $0 + o } ?? o
            }
            let out = y! + bias
            return out.reshaped(B, count, out.dim(1), out.dim(2), out.dim(3))
        }
        if perFrame {
            return concatenated((0..<tOut).map { j in
                let o = frameOut(j, 1)
                eval(o)
                return o
            }, axis: 1)
        }
        return frameOut(0, tOut)
    }
}

/// Unpadded stride-1 2D convolution of `x` `[N, H, W, C]`, computed in horizontal bands when the unrolled patch
/// buffer would be large (each band reads the `kh − 1` rows below it, so the result is identical).
func bandedConv2d(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    let (n, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let kh = w.dim(1), kw = w.dim(2)
    let hOut = h - kh + 1
    let rowCost = n * (wd - kw + 1) * c * kh * kw  // unrolled elements per output row
    let budget = 64 << 20
    guard rowCost * hOut > budget else { return conv2d(x, w) }
    let band = max(8, budget / max(rowCost, 1))
    var parts: [MLXArray] = []
    var r = 0
    while r < hOut {
        let r1 = min(hOut, r + band)
        let o = conv2d(x[0..., r ..< (r1 + kh - 1)], w)
        eval(o)
        parts.append(o)
        r = r1
    }
    return concatenated(parts, axis: 1)
}

/// Applies `op` to `[N, H, W, C]` one image at a time when the images are large (bounds the 2D convolutions'
/// patch buffers), else to the whole batch.
func framewise(_ x: MLXArray, _ op: (MLXArray) -> MLXArray) -> MLXArray {
    let n = x.dim(0)
    guard n > 1, x.dim(1) * x.dim(2) * x.dim(3) > 8 << 20 else { return op(x) }
    return concatenated((0..<n).map { i in
        let o = op(x[i ..< i + 1])
        eval(o)
        return o
    }, axis: 0)
}

/// `F.normalize` over channels × √C × γ.
func rmsNorm(_ x: MLXArray, _ gamma: MLXArray) -> MLXArray {
    let n = sqrt((x * x).sum(axis: -1, keepDims: true))
    return x / maximum(n, MLXArray(Float(1e-12))) * Float(gamma.dim(0)).squareRoot() * gamma
}

/// Applies `conv` with the chunk cache: remembers this input's last two frames for the next chunk.
func cachedConv(_ conv: CausalConv3d, _ x: MLXArray, _ cache: FeatCache?) -> MLXArray {
    guard let cache else { return conv(x, cache: nil) }
    let i = cache.idx
    var prev: MLXArray?
    if case .frames(let p)? = cache.slots[i] { prev = p }
    let T = x.dim(1)
    var keep = x[0..., max(0, T - 2)...]
    if keep.dim(1) < 2, let prev { keep = concatenated([prev[0..., (prev.dim(1) - 1)...], keep], axis: 1) }
    let y = conv(x, cache: prev)
    cache.slots[i] = .frames(keep)
    cache.idx += 1
    return y
}

struct ResidualBlock {
    let norm1, norm2: MLXArray
    let conv1, conv2: CausalConv3d
    let shortcut: CausalConv3d?

    init(_ l: Loader, _ p: String) throws {
        norm1 = try l.gamma(p + ".residual.0")
        conv1 = try l.conv3d(p + ".residual.2", padding: (1, 1))
        norm2 = try l.gamma(p + ".residual.3")
        conv2 = try l.conv3d(p + ".residual.6", padding: (1, 1))
        shortcut = l.store.has(p + ".shortcut.weight") ? try l.conv3d(p + ".shortcut", padding: (0, 0)) : nil
    }

    func callAsFunction(_ x: MLXArray, _ cache: FeatCache?) -> MLXArray {
        let h = shortcut.map { $0(x, cache: nil) } ?? x
        var y = cachedConv(conv1, siluAct(rmsNorm(x, norm1)), cache)
        y = cachedConv(conv2, siluAct(rmsNorm(y, norm2)), cache)
        return y + h
    }
}

/// Single-head spatial self-attention, per frame.
struct AttentionBlock {
    let norm: MLXArray
    let qkvW, qkvB, projW, projB: MLXArray

    init(_ l: Loader, _ p: String) throws {
        norm = try l.gamma(p + ".norm")
        (qkvW, qkvB) = try l.conv2d(p + ".to_qkv")
        (projW, projB) = try l.conv2d(p + ".proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let f = rmsNorm(x, norm).reshaped(B * T, H, W, C)
        let qkv = (conv2d(f, qkvW) + qkvB).reshaped(B * T, 1, H * W, 3 * C)
        let q = qkv[.ellipsis, 0 ..< C], k = qkv[.ellipsis, C ..< 2 * C], v = qkv[.ellipsis, (2 * C)...]
        let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(C).squareRoot(),
                                                  mask: .none)
        let o = conv2d(a.reshaped(B * T, H, W, C), projW) + projB
        return o.reshaped(B, T, H, W, C) + x
    }
}

struct Resample {
    enum Mode { case up2d, up3d, down2d, down3d }
    let mode: Mode
    let convW, convB: MLXArray
    let timeConv: CausalConv3d?

    init(_ l: Loader, _ p: String, mode: Mode) throws {
        self.mode = mode
        (convW, convB) = try l.conv2d(p + ".resample.1")
        switch mode {
        case .up3d: timeConv = try l.conv3d(p + ".time_conv", padding: (1, 0))
        case .down3d: timeConv = try l.conv3d(p + ".time_conv", padding: (0, 0), temporalStride: 2)
        default: timeConv = nil
        }
    }

    func callAsFunction(_ x: MLXArray, _ cache: FeatCache) -> MLXArray {
        var x = x
        if mode == .up3d, let timeConv {
            let i = cache.idx
            if cache.slots[i] == nil {
                cache.slots[i] = .rep  // first chunk: no temporal upsampling
                cache.idx += 1
            } else {
                let T = x.dim(1)
                var keep = x[0..., max(0, T - 2)...]
                var prev: MLXArray?
                if case .frames(let p)? = cache.slots[i] { prev = p }
                if keep.dim(1) < 2 {
                    keep = concatenated([prev.map { $0[0..., ($0.dim(1) - 1)...] } ?? MLXArray.zeros(keep.shape, dtype: keep.dtype), keep],
                                        axis: 1)
                }
                let y = timeConv(x, cache: prev)  // [B, T, H, W, 2C]
                cache.slots[i] = .frames(keep)
                cache.idx += 1
                let (B, H, W, C) = (y.dim(0), y.dim(2), y.dim(3), y.dim(4) / 2)
                x = y.reshaped(B, T, H, W, 2, C).transposed(0, 1, 4, 2, 3, 5).reshaped(B, 2 * T, H, W, C)
            }
        }
        let (B, T, H, W, C) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        var f = x.reshaped(B * T, H, W, C)
        switch mode {
        case .up2d, .up3d:
            f = framewise(f) { repeated(repeated($0, count: 2, axis: 1), count: 2, axis: 2) }
            f = framewise(f) { bandedConv2d(padded($0, widths: [.init(0), .init(1), .init(1), .init(0)]), convW) + convB }
        case .down2d, .down3d:
            f = padded(f, widths: [.init(0), .init((0, 1)), .init((0, 1)), .init(0)])
            f = framewise(f) { conv2d($0, convW, stride: .init(2)) + convB }
        }
        x = f.reshaped(B, T, f.dim(1), f.dim(2), f.dim(3))
        if mode == .down3d, let timeConv {
            let i = cache.idx
            if case .frames(let prev)? = cache.slots[i] {
                let keep = x[0..., (x.dim(1) - 1)...]
                x = timeConv(concatenated([prev[0..., (prev.dim(1) - 1)...], x], axis: 1), cache: nil)
                cache.slots[i] = .frames(keep)
            } else {
                cache.slots[i] = .frames(x)  // first chunk: no temporal downsampling
            }
            cache.idx += 1
        }
        return x
    }
}

enum Layer {
    case res(ResidualBlock), attn(AttentionBlock), resample(Resample)

    func callAsFunction(_ x: MLXArray, _ cache: FeatCache) -> MLXArray {
        switch self {
        case .res(let r): r(x, cache)
        case .attn(let a): a(x)
        case .resample(let r): r(x, cache)
        }
    }
}

struct Decoder {
    let conv1: CausalConv3d
    let middle: [Layer]
    let upsamples: [Layer]
    let headNorm: MLXArray
    let headConv: CausalConv3d

    init(_ l: Loader, dim: Int = 96, dimMult: [Int] = [1, 2, 4, 4], resBlocks: Int = 2,
         temporalUpsample: [Bool] = [true, true, false]) throws {
        let dims = ([dimMult.last!] + dimMult.reversed()).map { dim * $0 }
        conv1 = try l.conv3d("decoder.conv1", padding: (1, 1))
        middle = [.res(try ResidualBlock(l, "decoder.middle.0")), .attn(try AttentionBlock(l, "decoder.middle.1")),
                  .res(try ResidualBlock(l, "decoder.middle.2"))]
        var ups: [Layer] = []
        for i in 0 ..< dims.count - 1 {
            for _ in 0 ... resBlocks { ups.append(.res(try ResidualBlock(l, "decoder.upsamples.\(ups.count)"))) }
            if i != dimMult.count - 1 {
                ups.append(.resample(try Resample(l, "decoder.upsamples.\(ups.count)",
                                                  mode: temporalUpsample[i] ? .up3d : .up2d)))
            }
        }
        upsamples = ups
        headNorm = try l.gamma("decoder.head.0")
        headConv = try l.conv3d("decoder.head.2", padding: (1, 1))
    }

    func callAsFunction(_ x: MLXArray, _ cache: FeatCache) -> MLXArray {
        var x = cachedConv(conv1, x, cache)
        for layer in middle { x = layer(x, cache) }
        // Up to the last temporal upsample, run the chunk as is, evaluating layer by layer.
        let split = (upsamples.lastIndex { if case .resample(let r) = $0 { r.mode == .up3d } else { false } } ?? -1) + 1
        for layer in upsamples[..<split] {
            x = layer(x, cache)
            eval(x)
        }
        // The remaining layers are per-frame or causal convolutions with a two-frame cache, so feeding the chunk's
        // frames one at a time gives the same result with a fraction of the memory at full resolution.
        let start = cache.idx
        var frames: [MLXArray] = []
        for t in 0..<x.dim(1) {
            cache.idx = start
            var f = x[0..., t ..< t + 1]
            for layer in upsamples[split...] { f = layer(f, cache) }
            f = cachedConv(headConv, siluAct(rmsNorm(f, headNorm)), cache)
            eval(f)
            frames.append(f)
        }
        return concatenated(frames, axis: 1)
    }
}

struct Encoder {
    let conv1: CausalConv3d
    let downsamples: [Layer]
    let middle: [Layer]
    let headNorm: MLXArray
    let headConv: CausalConv3d

    init(_ l: Loader, dimMult: [Int] = [1, 2, 4, 4], resBlocks: Int = 2,
         temporalDownsample: [Bool] = [false, true, true]) throws {
        conv1 = try l.conv3d("encoder.conv1", padding: (1, 1))
        var downs: [Layer] = []
        for i in 0 ..< dimMult.count {
            for _ in 0 ..< resBlocks { downs.append(.res(try ResidualBlock(l, "encoder.downsamples.\(downs.count)"))) }
            if i != dimMult.count - 1 {
                downs.append(.resample(try Resample(l, "encoder.downsamples.\(downs.count)",
                                                    mode: temporalDownsample[i] ? .down3d : .down2d)))
            }
        }
        downsamples = downs
        middle = [.res(try ResidualBlock(l, "encoder.middle.0")), .attn(try AttentionBlock(l, "encoder.middle.1")),
                  .res(try ResidualBlock(l, "encoder.middle.2"))]
        headNorm = try l.gamma("encoder.head.0")
        headConv = try l.conv3d("encoder.head.2", padding: (1, 1))
    }

    func callAsFunction(_ x: MLXArray, _ cache: FeatCache) -> MLXArray {
        var x = cachedConv(conv1, x, cache)
        for layer in downsamples {
            x = layer(x, cache)
            eval(x)
        }
        for layer in middle { x = layer(x, cache) }
        return cachedConv(headConv, siluAct(rmsNorm(x, headNorm)), cache)
    }
}
