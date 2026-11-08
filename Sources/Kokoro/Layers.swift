import Foundation
import MLX
import MLXDiffusionCore

// Building blocks shared by the Kokoro modules. Everything is channels-last: sequences are `[1, T, C]`.

/// Flat checkpoint view with the `module.` prefix that `kokoro-v1_0.pth` carries under each top-level key removed
/// (`bert.module.embeddings…` → `bert.embeddings…`, as `KModel.__init__` does when its first load attempt fails).
final class KWeights {
    let arrays: [String: MLXArray]
    /// Compute dtype for convolutions/linears (norms, LSTMs and the iSTFT always run in float32).
    let dtype: DType

    init(_ raw: [String: MLXArray], dtype: DType) {
        var a: [String: MLXArray] = [:]
        for (k, v) in raw {
            if let dot = k.firstIndex(of: "."), k[k.index(after: dot)...].hasPrefix("module.") {
                a[String(k[..<dot]) + "." + String(k[k.index(dot, offsetBy: 8)...])] = v
            } else {
                a[k] = v
            }
        }
        arrays = a
        self.dtype = dtype
    }

    func get(_ k: String) throws -> MLXArray {
        guard let a = arrays[k] else { throw WeightError.missing(k) }
        return a.asType(.float32)
    }

    func has(_ k: String) -> Bool { arrays[k] != nil }

    func linear(_ k: String, dtype: DType? = nil) throws -> Lin {
        let dt = dtype ?? self.dtype
        return Lin(weight: try get(k + ".weight").asType(dt), bias: arrays[k + ".bias"].map { $0.asType(dt) })
    }

    /// Torch-layout conv weight with weight norm folded (`g · v / ‖v‖`, norm over all axes but the first).
    func torchConvWeight(_ k: String) throws -> MLXArray {
        if has(k + ".weight_g") {
            let g = try get(k + ".weight_g"), v = try get(k + ".weight_v")
            return g * v / sqrt((v * v).sum(axes: [1, 2], keepDims: true))
        }
        return try get(k + ".weight")
    }

    func conv(_ k: String, stride: Int = 1, padding: Int = 0, dilation: Int = 1, dtype: DType? = nil) throws -> KConv {
        let dt = dtype ?? self.dtype
        let w = try torchConvWeight(k).transposed(0, 2, 1)  // [out, in, k] → [out, k, in]
        return KConv(weight: w.asType(dt), bias: arrays[k + ".bias"].map { $0.asType(dt) },
                     stride: stride, padding: padding, dilation: dilation)
    }

    func convTransposed(_ k: String, stride: Int, padding: Int) throws -> KConvT {
        let w = try torchConvWeight(k).transposed(1, 2, 0)  // [in, out, k] → [out, k, in]
        return KConvT(weight: w.asType(dtype), bias: arrays[k + ".bias"].map { $0.asType(dtype) },
                      stride: stride, padding: padding)
    }
}

struct KConv {
    let weight: MLXArray  // [out, k, in]
    let bias: MLXArray?
    let stride: Int, padding: Int, dilation: Int

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x.asType(weight.dtype), weight, stride: stride, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

struct KConvT {
    let weight: MLXArray  // [out, k, in]
    let bias: MLXArray?
    let stride: Int, padding: Int
    /// `[in, k·out]` for the polyphase path (kernel = 2 · stride, the iSTFTNet upsamplers).
    let packed: MLXArray?

    init(weight: MLXArray, bias: MLXArray?, stride: Int, padding: Int) {
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        let k = weight.dim(1)
        packed = k == 2 * stride && padding == stride / 2 && stride % 2 == 0
            ? weight.transposed(2, 1, 0).reshaped(weight.dim(2), -1) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = x.asType(weight.dtype)
        var y: MLXArray
        if let packed {
            // Polyphase transposed conv: one [T, in] × [in, k·out] matmul, then overlap-add of the two kernel halves
            // (MLX's convTransposed1d dilates the input and unfolds it: gigabytes at the generator's lengths).
            let T = x.dim(1), s = stride, out = weight.dim(0)
            let z = matmul(x[0], packed).reshaped(T, 2, s, out)
            let zero = MLXArray.zeros([1, s, out], dtype: z.dtype)
            let full = concatenated([z[0..., 0], zero], axis: 0) + concatenated([zero, z[0..., 1]], axis: 0)
            y = full.reshaped(1, (T + 1) * s, out)[0..., padding ..< padding + T * s]
        } else {
            y = convTransposed1d(x, weight, stride: stride, padding: padding)
        }
        if let bias { y = y + bias }
        return y
    }
}

@inline(__always) func leaky(_ x: MLXArray, _ slope: Float) -> MLXArray { maximum(x, slope * x) }

/// `F.layer_norm` over the channel axis (float32 statistics).
func channelLayerNorm(_ x: MLXArray, weight: MLXArray? = nil, bias: MLXArray? = nil, eps: Float = 1e-5) -> MLXArray {
    MLXFast.layerNorm(x.asType(.float32), weight: weight, bias: bias, eps: eps).asType(x.dtype)
}

/// StyleTTS2 `AdaIN1d`: instance norm over time (biased variance, eps 1e-5; the affine params are absent from the
/// checkpoint so they stay at identity), then `(1 + γ(s)) · x̂ + β(s)`.
struct AdaIN1d {
    let fc: Lin

    init(_ w: KWeights, _ k: String) throws { fc = try w.linear(k + ".fc", dtype: .float32) }

    /// `x`: `[1, T, C]`, `s`: `[1, style]`.
    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let h = fc(s)  // [1, 2C]
        let c = h.dim(1) / 2
        let gamma = h[0..., ..<c].expandedDimensions(axis: 1), beta = h[0..., c...].expandedDimensions(axis: 1)
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: 1, keepDims: true)
        let d = xf - mean
        let v = (d * d).mean(axis: 1, keepDims: true)
        let n = d * rsqrt(v + 1e-5)
        return ((1 + gamma) * n + beta).asType(x.dtype)
    }
}

/// StyleTTS2 `AdaLayerNorm`: layer norm over channels without affine, then `(1 + γ(s)) · x̂ + β(s)`.
struct AdaLayerNorm {
    let fc: Lin

    init(_ w: KWeights, _ k: String) throws { fc = try w.linear(k + ".fc", dtype: .float32) }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        let h = fc(s)
        let c = h.dim(1) / 2
        let gamma = h[0..., ..<c].expandedDimensions(axis: 1), beta = h[0..., c...].expandedDimensions(axis: 1)
        return (1 + gamma) * MLXFast.layerNorm(x.asType(.float32), eps: 1e-5) + beta
    }
}

/// StyleTTS2 `AdainResBlk1d` (LeakyReLU 0.2), optionally upsampling ×2 (nearest on the shortcut, a depthwise
/// `ConvTranspose1d(k=3, s=2, p=1, output_padding=1)` on the residual path).
struct AdainResBlk1d {
    let norm1: AdaIN1d, norm2: AdaIN1d
    let conv1: KConv, conv2: KConv
    let conv1x1: KConv?
    let pool: (weight: MLXArray, bias: MLXArray)?  // depthwise taps [3, C], bias [C]
    let upsample: Bool

    init(_ w: KWeights, _ k: String, upsample: Bool = false) throws {
        norm1 = try AdaIN1d(w, k + ".norm1")
        norm2 = try AdaIN1d(w, k + ".norm2")
        conv1 = try w.conv(k + ".conv1", padding: 1)
        conv2 = try w.conv(k + ".conv2", padding: 1)
        conv1x1 = w.has(k + ".conv1x1.weight_v") || w.has(k + ".conv1x1.weight") ? try w.conv(k + ".conv1x1") : nil
        self.upsample = upsample
        if upsample {
            let pw = try w.torchConvWeight(k + ".pool")  // [C, 1, 3]
            pool = (pw.squeezed(axis: 1).transposed(1, 0).asType(w.dtype), try w.get(k + ".pool.bias").asType(w.dtype))
        } else {
            pool = nil
        }
    }

    /// Depthwise transposed conv, stride 2: y[2m] = x[m]·w1, y[2m+1] = x[m]·w2 + x[m+1]·w0 (+ bias).
    private func poolUp(_ x: MLXArray) -> MLXArray {
        guard let pool else { return x }
        let T = x.dim(1), C = x.dim(2)
        let xn = concatenated([x[0..., 1...], MLXArray.zeros([1, 1, C], dtype: x.dtype)], axis: 1)
        let even = x * pool.weight[1]
        let odd = x * pool.weight[2] + xn * pool.weight[0]
        return stacked([even, odd], axis: 2).reshaped(1, 2 * T, C) + pool.bias
    }

    private func nearest2(_ x: MLXArray) -> MLXArray {
        let T = x.dim(1), C = x.dim(2)
        return broadcast(x.expandedDimensions(axis: 2), to: [1, T, 2, C]).reshaped(1, 2 * T, C)
    }

    func callAsFunction(_ x: MLXArray, _ s: MLXArray) -> MLXArray {
        var r = leaky(norm1(x, s), 0.2)
        if upsample { r = poolUp(r) }
        r = conv1(r)
        r = conv2(leaky(norm2(r, s), 0.2))
        var sc = upsample ? nearest2(x) : x
        if let conv1x1 { sc = conv1x1(sc) }
        return (r + sc) * Float(0.70710677)  // torch.rsqrt(torch.tensor(2)) in float32
    }
}
