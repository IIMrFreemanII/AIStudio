import Foundation
import MLX
import MLXDiffusionCore

/// `AutoencoderKLQwenImage21` specialised to single images (T = 1).
///
/// The reference is a causal 3D (Wan-style) VAE, but for one frame every causal conv is a plain 2D conv, the feature
/// caches never fire, and the temporal parts of the residual shortcuts reduce to fixed channel shuffles. All tensors
/// here are NHWC.
public final class QwenImage21VAE {
    public struct Config {
        public var baseDim = 96
        public var decoderBaseDim = 144
        public var zDim = 64
        public var dimMult = [1, 2, 4, 8, 8]
        public var numResBlocks = 2
        public var temporalDownsample = [false, true, true, true]
        public var inChannels = 4
        public var outChannels = 4
        public var latentsMean: [Float] = []
        public var latentsStd: [Float] = []

        public init() {}

        public init(url: URL) throws {
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            baseDim = obj["base_dim"] as? Int ?? baseDim
            decoderBaseDim = obj["decoder_base_dim"] as? Int ?? baseDim
            zDim = obj["z_dim"] as? Int ?? zDim
            dimMult = obj["dim_mult"] as? [Int] ?? dimMult
            numResBlocks = obj["num_res_blocks"] as? Int ?? numResBlocks
            temporalDownsample = obj["temperal_downsample"] as? [Bool] ?? temporalDownsample
            inChannels = obj["in_channels"] as? Int ?? inChannels
            outChannels = obj["out_channels"] as? Int ?? outChannels
            latentsMean = (obj["latents_mean"] as? [Double] ?? []).map(Float.init)
            latentsStd = (obj["latents_std"] as? [Double] ?? []).map(Float.init)
        }
    }

    public let config: Config
    public let dtype: DType
    let encoder: Encoder?
    let decoder: Decoder
    let quantConv: Conv2d?
    let postQuantConv: Conv2d
    let mean: MLXArray
    let std: MLXArray

    public init(directory: URL, dtype: DType = .bfloat16, loadEncoder: Bool = true) throws {
        self.config = try Config(url: directory.appendingPathComponent("config.json"))
        self.dtype = dtype
        let store = try WeightStore(directory: directory)
        let c = config
        self.decoder = try Decoder(store: store, prefix: "decoder", config: c, dtype: dtype)
        self.postQuantConv = try Conv2d(store: store, prefix: "post_quant_conv", dtype: dtype)
        if loadEncoder {
            self.encoder = try Encoder(store: store, prefix: "encoder", config: c, dtype: dtype)
            self.quantConv = try Conv2d(store: store, prefix: "quant_conv", dtype: dtype)
        } else {
            self.encoder = nil
            self.quantConv = nil
        }
        self.mean = MLXArray(c.latentsMean)
        self.std = MLXArray(c.latentsStd)
    }

    /// Normalised latents `[B, h, w, 64]` → image `[B, H, W, 4]` in `[-1, 1]`.
    public func decode(normalizedLatents z: MLXArray) -> MLXArray {
        let raw = z.asType(.float32) * std + mean
        var x = postQuantConv(raw.asType(dtype))
        x = decoder(x)
        return clip(x, min: -1, max: 1)
    }

    /// ``decode(normalizedLatents:)`` over overlapping latent tiles (`tile`/`overlap` in latent cells), blended
    /// linearly. Bounds peak memory for large images; batch size must be 1.
    public func decodeTiled(normalizedLatents z: MLXArray, tile: Int = 64, overlap: Int = 8) -> MLXArray {
        Tiling.run(z, cell: 1, outPerCell: 16, tile: tile, overlap: overlap) { decode(normalizedLatents: $0) }
    }

    /// ``encode(_:)`` over overlapping pixel tiles (`tile`/`overlap` in latent cells of 16 px).
    public func encodeTiled(_ image: MLXArray, tile: Int = 64, overlap: Int = 8) -> MLXArray {
        Tiling.run(image, cell: 16, outPerCell: 1, tile: tile, overlap: overlap) { encode($0) }
    }

    /// Image `[B, H, W, 4]` in `[-1, 1]` → normalised latents `[B, H/16, W/16, 64]` (posterior mode).
    public func encode(_ image: MLXArray) -> MLXArray {
        guard let encoder, let quantConv else { fatalError("VAE was loaded without its encoder") }
        var x = encoder(image.asType(dtype))
        x = quantConv(x)
        let mu = x[.ellipsis, ..<config.zDim]
        return (mu.asType(.float32) - mean) / std
    }
}

// MARK: - building blocks

/// `F.normalize(x, dim=channels) * sqrt(C) * gamma`, i.e. an RMS norm over channels.
struct ChannelRMSNorm {
    let gamma: MLXArray

    init(store: WeightStore, prefix: String, dtype: DType) throws {
        gamma = try store.get(prefix + ".gamma").reshaped(-1).asType(dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // ||x|| clamped at 1e-12 ⇔ rms clamped at 1e-12/sqrt(C); the eps below is negligible either way.
        rmsNorm(x, weight: gamma, eps: 1e-12)
    }
}

struct ResidualBlock {
    let norm1: ChannelRMSNorm
    let conv1: Conv2d
    let norm2: ChannelRMSNorm
    let conv2: Conv2d
    let shortcut: Conv2d?

    init(store: WeightStore, prefix: String, dtype: DType) throws {
        norm1 = try ChannelRMSNorm(store: store, prefix: prefix + ".norm1", dtype: dtype)
        conv1 = try Conv2d(store: store, prefix: prefix + ".conv1", padding: 1, dtype: dtype)
        norm2 = try ChannelRMSNorm(store: store, prefix: prefix + ".norm2", dtype: dtype)
        conv2 = try Conv2d(store: store, prefix: prefix + ".conv2", padding: 1, dtype: dtype)
        shortcut = store.has(prefix + ".conv_shortcut.weight")
            ? try Conv2d(store: store, prefix: prefix + ".conv_shortcut", dtype: dtype) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = shortcut.map { $0(x) } ?? x
        var y = conv1(siluAct(norm1(x)))
        y = conv2(siluAct(norm2(y)))
        return y + h
    }
}

/// Single-head spatial self-attention.
struct AttentionBlock {
    let norm: ChannelRMSNorm
    let toQKV: Conv2d
    let proj: Conv2d

    init(store: WeightStore, prefix: String, dtype: DType) throws {
        norm = try ChannelRMSNorm(store: store, prefix: prefix + ".norm", dtype: dtype)
        toQKV = try Conv2d(store: store, prefix: prefix + ".to_qkv", dtype: dtype)
        proj = try Conv2d(store: store, prefix: prefix + ".proj", dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let qkv = toQKV(norm(x)).reshaped(b, 1, h * w, 3 * c)
        let q = qkv[.ellipsis, 0 ..< c]
        let k = qkv[.ellipsis, c ..< 2 * c]
        let v = qkv[.ellipsis, (2 * c)...]
        let o = scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / sqrt(Float(c)), mask: nil)
        return proj(o.reshaped(b, h, w, c)) + x
    }
}

struct MidBlock {
    let res0: ResidualBlock
    let attn: AttentionBlock
    let res1: ResidualBlock

    init(store: WeightStore, prefix: String, dtype: DType) throws {
        res0 = try ResidualBlock(store: store, prefix: prefix + ".resnets.0", dtype: dtype)
        attn = try AttentionBlock(store: store, prefix: prefix + ".attentions.0", dtype: dtype)
        res1 = try ResidualBlock(store: store, prefix: prefix + ".resnets.1", dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { res1(attn(res0(x))) }
}

/// `AvgDown3D` for one frame: the frame is front-padded with `factorT - 1` zero frames, space-to-channel folded and
/// averaged in groups.
func avgDown(_ x: MLXArray, outChannels: Int, factorT: Int, factorS: Int) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let hh = h / factorS, ww = w / factorS
    // [B, H', fs(i), W', fs(j), C] → [B, H', W', C, i, j]
    var y = x.reshaped(b, hh, factorS, ww, factorS, c).transposed(0, 1, 3, 5, 2, 4)
    // channel order is (c, t, i, j) with the real frame at t = factorT - 1
    y = y.expandedDimensions(axis: 4)
    if factorT > 1 {
        let zeros = MLXArray.zeros([b, hh, ww, c, factorT - 1, factorS, factorS], dtype: y.dtype)
        y = concatenated([zeros, y], axis: 4)
    }
    let total = c * factorT * factorS * factorS
    let group = total / outChannels
    y = y.reshaped(b, hh, ww, outChannels, group)
    return y.mean(axis: -1)
}

/// `DupUp3D` for the first chunk of a single frame: channel repeat, channel-to-space unfold, keep the last frame.
func dupUp(_ x: MLXArray, outChannels: Int, factorT: Int, factorS: Int) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let factor = factorT * factorS * factorS
    let repeats = outChannels * factor / c
    // repeat_interleave over channels
    var y = repeated(x.expandedDimensions(axis: 4), count: repeats, axis: 4).reshaped(b, h, w, c * repeats)
    y = y.reshaped(b, h, w, outChannels, factorT, factorS, factorS)
    y = y[0..., 0..., 0..., 0..., factorT - 1, 0..., 0...]  // [B, H, W, out, i, j]
    y = y.transposed(0, 1, 4, 2, 5, 3)  // [B, H, i, W, j, out]
    return y.reshaped(b, h * factorS, w * factorS, outChannels)
}

func upsampleNearest2x(_ x: MLXArray) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let y = broadcast(x.reshaped(b, h, 1, w, 1, c), to: [b, h, 2, w, 2, c])
    return y.reshaped(b, h * 2, w * 2, c)
}

struct ResidualDownBlock {
    let resnets: [ResidualBlock]
    let downsampler: Conv2d?
    let outDim: Int
    let factorT: Int
    let factorS: Int

    init(store: WeightStore, prefix: String, outDim: Int, numRes: Int, temporal: Bool, down: Bool,
         dtype: DType) throws {
        resnets = try (0..<numRes).map {
            try ResidualBlock(store: store, prefix: "\(prefix).resnets.\($0)", dtype: dtype)
        }
        downsampler = down
            ? try Conv2d(store: store, prefix: prefix + ".downsampler.resample.1", stride: 2, dtype: dtype) : nil
        self.outDim = outDim
        self.factorT = temporal ? 2 : 1
        self.factorS = down ? 2 : 1
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for r in resnets { y = r(y) }
        if let downsampler {
            // ZeroPad2d((0, 1, 0, 1)) then a stride-2 3x3 conv
            y = padded(y, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])
            y = downsampler(y)
        }
        return y + avgDown(x, outChannels: outDim, factorT: factorT, factorS: factorS)
    }
}

struct ResidualUpBlock {
    let resnets: [ResidualBlock]
    let upsampler: Conv2d?
    let outDim: Int
    let factorT: Int

    init(store: WeightStore, prefix: String, outDim: Int, numRes: Int, temporal: Bool, up: Bool,
         dtype: DType) throws {
        resnets = try (0...numRes).map {
            try ResidualBlock(store: store, prefix: "\(prefix).resnets.\($0)", dtype: dtype)
        }
        upsampler = up
            ? try Conv2d(store: store, prefix: prefix + ".upsampler.resample.1", padding: 1, dtype: dtype) : nil
        self.outDim = outDim
        self.factorT = temporal ? 2 : 1
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for r in resnets { y = r(y) }
        if let upsampler {
            // nearest-exact ×2 equals plain nearest for integer factors
            y = upsampler(upsampleNearest2x(y))
            y = y + dupUp(x, outChannels: outDim, factorT: factorT, factorS: 2)
        }
        return y
    }
}

final class Encoder {
    let convIn: Conv2d
    let downBlocks: [ResidualDownBlock]
    let mid: MidBlock
    let normOut: ChannelRMSNorm
    let convOut: Conv2d

    init(store: WeightStore, prefix: String, config c: QwenImage21VAE.Config, dtype: DType) throws {
        let dims = ([1] + c.dimMult).map { $0 * c.baseDim }
        convIn = try Conv2d(store: store, prefix: prefix + ".conv_in", padding: 1, dtype: dtype)
        var blocks: [ResidualDownBlock] = []
        for i in 0..<(dims.count - 1) {
            let last = i == c.dimMult.count - 1
            blocks.append(try ResidualDownBlock(
                store: store, prefix: "\(prefix).down_blocks.\(i)", outDim: dims[i + 1], numRes: c.numResBlocks,
                temporal: last ? false : c.temporalDownsample[i], down: !last, dtype: dtype))
        }
        downBlocks = blocks
        mid = try MidBlock(store: store, prefix: prefix + ".mid_block", dtype: dtype)
        normOut = try ChannelRMSNorm(store: store, prefix: prefix + ".norm_out", dtype: dtype)
        convOut = try Conv2d(store: store, prefix: prefix + ".conv_out", padding: 1, dtype: dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convIn(x)
        for b in downBlocks { y = b(y) }
        y = mid(y)
        return convOut(siluAct(normOut(y)))
    }
}

final class Decoder {
    let convIn: Conv2d
    let mid: MidBlock
    let upBlocks: [ResidualUpBlock]
    let normOut: ChannelRMSNorm
    let convOut: Conv2d

    init(store: WeightStore, prefix: String, config c: QwenImage21VAE.Config, dtype: DType) throws {
        let mult = [c.dimMult.last!] + c.dimMult.reversed()
        let dims = mult.map { $0 * c.decoderBaseDim }
        let temporalUp = Array(c.temporalDownsample.reversed())
        convIn = try Conv2d(store: store, prefix: prefix + ".conv_in", padding: 1, dtype: dtype)
        mid = try MidBlock(store: store, prefix: prefix + ".mid_block", dtype: dtype)
        var blocks: [ResidualUpBlock] = []
        for i in 0..<(dims.count - 1) {
            let up = i != c.dimMult.count - 1
            blocks.append(try ResidualUpBlock(
                store: store, prefix: "\(prefix).up_blocks.\(i)", outDim: dims[i + 1], numRes: c.numResBlocks,
                temporal: up ? temporalUp[i] : false, up: up, dtype: dtype))
        }
        upBlocks = blocks
        normOut = try ChannelRMSNorm(store: store, prefix: prefix + ".norm_out", dtype: dtype)
        convOut = try Conv2d(store: store, prefix: prefix + ".conv_out", padding: 1, dtype: dtype)
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var y = mid(convIn(z))
        for b in upBlocks {
            y = b(y)
            eval(y)  // keep the peak graph small at high resolution
        }
        return convOut(siluAct(normOut(y)))
    }
}
