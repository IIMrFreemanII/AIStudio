import Foundation
import MLX
import MLXDiffusionCore

/// Hyper-parameters of one BigVGAN-v2 generator (the `vocoder` / `bwe` sections of `embedded_config.json`).
/// Kernel sizes and channel counts are read from the weights; only strides and the residual-block layout live here.
public struct LTXBigVGANConfig: Sendable {
    public var upsampleRates: [Int]
    public var resblockKernelSizes: [Int] = [3, 7, 11]
    public var resblockDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    /// The reference applies `tanh` after `conv_post` on the base vocoder and nothing on the BWE generator.
    public var finalTanh: Bool

    public init(upsampleRates: [Int], finalTanh: Bool) {
        self.upsampleRates = upsampleRates
        self.finalTanh = finalTanh
    }

    /// LTX-2.3 base vocoder: 100 mel frames/s → 16 kHz (rates 5·2·2·2·2·2 = 160).
    public static let ltxBase = LTXBigVGANConfig(upsampleRates: [5, 2, 2, 2, 2, 2], finalTanh: true)
    /// LTX-2.3 bandwidth-extension generator: 200 mel frames/s → 48 kHz (rates 6·5·2·2·2 = 240).
    public static let ltxBWE = LTXBigVGANConfig(upsampleRates: [6, 5, 2, 2, 2], finalTanh: false)

    init(json: [String: Any]?, fallback: LTXBigVGANConfig) {
        self = fallback
        guard let json else { return }
        if let r = json["upsample_rates"] as? [Int] { upsampleRates = r }
        if let k = json["resblock_kernel_sizes"] as? [Int] { resblockKernelSizes = k }
        if let d = json["resblock_dilation_sizes"] as? [[Int]] { resblockDilations = d }
    }
}

/// Conv1d in MLX layout `[O, K, I]` (the LTX MLX checkpoints are already converted). `x`: `[B, L, C]`.
struct LTXConv1d {
    let weight: MLXArray
    let bias: MLXArray?
    let padding: Int
    let dilation: Int

    init(store: WeightStore, prefix: String, padding: Int? = nil, dilation: Int = 1) throws {
        let w = try store.get(prefix + ".weight")
        guard w.ndim == 3 else { throw WeightError.badShape(prefix + ".weight", w.shape) }
        weight = w.asType(.float32)
        bias = store.optional(prefix + ".bias")?.asType(.float32)
        self.dilation = dilation
        self.padding = padding ?? (w.dim(1) * dilation - dilation) / 2
    }

    var arrays: [MLXArray] { [weight] + (bias.map { [$0] } ?? []) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, padding: padding, dilation: dilation)
        if let bias { y = y + bias }
        return y
    }
}

/// ConvTranspose1d in MLX layout `[O, K, I]`, padding `(K − stride) / 2` as in BigVGAN.
struct LTXConvTranspose1d {
    let weight: MLXArray
    let bias: MLXArray?
    let stride: Int
    let padding: Int

    init(store: WeightStore, prefix: String, stride: Int) throws {
        let w = try store.get(prefix + ".weight")
        guard w.ndim == 3 else { throw WeightError.badShape(prefix + ".weight", w.shape) }
        weight = w.asType(.float32)
        bias = store.optional(prefix + ".bias")?.asType(.float32)
        self.stride = stride
        padding = (w.dim(1) - stride) / 2
    }

    var arrays: [MLXArray] { [weight] + (bias.map { [$0] } ?? []) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed1d(x, weight, stride: stride, padding: padding)
        if let bias { y = y + bias }
        return y
    }
}

/// Anti-aliased SnakeBeta (`Activation1d`): 2× zero-insert upsample + 12-tap low-pass (×2), SnakeBeta
/// `x + 1/β · sin²(αx)` with log-scale α/β, then replicate-padded 12-tap low-pass with stride 2.
///
/// The reference runs the depthwise filters as a 1-channel conv over `[B·C, T, 1]`; here they run channels-last
/// as a grouped conv (`groups = C`), which gives bit-identical results without the transposes.
struct LTXActivation1d {
    let alpha: MLXArray     // exp(log α), [C]
    let invBeta: MLXArray   // 1 / (exp(log β) + 1e-9), [C]
    let upFilter: MLXArray  // [C, K, 1]
    let downFilter: MLXArray
    let kernel: Int

    init(store: WeightStore, prefix: String) throws {
        let a = try store.get(prefix + ".act.alpha").asType(.float32)
        let b = try store.get(prefix + ".act.beta").asType(.float32)
        let c = a.dim(0)
        alpha = exp(a)
        invBeta = 1.0 / (exp(b) + Float(1e-9))
        let fu = try store.get(prefix + ".upsample.filter").asType(.float32)            // [1, K, 1]
        let fd = try store.get(prefix + ".downsample.lowpass.filter").asType(.float32)  // [1, K, 1]
        kernel = fu.dim(1)
        upFilter = MLX.contiguous(broadcast(fu, to: [c, fu.dim(1), 1]))
        downFilter = MLX.contiguous(broadcast(fd, to: [c, fd.dim(1), 1]))
    }

    var arrays: [MLXArray] { [alpha, invBeta, upFilter, downFilter] }

    /// `x`: `[B, T, C]` float32 → `[B, T, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let k = kernel
        // UpSample1d: zero-insert, replicate-pad the zero-inserted signal (K/2 left, K/2 − 1 right), filter, ×2.
        var z = stacked([x, MLXArray.zeros(like: x)], axis: 2).reshaped(b, 2 * t, c)
        let padL = k / 2, padR = k / 2 - 1
        z = concatenated([repeated(z[0..., ..<1, 0...], count: padL, axis: 1), z,
                          repeated(z[0..., (2 * t - 1)..., 0...], count: padR, axis: 1)], axis: 1)
        var u = conv1d(z, upFilter, groups: c) * Float(2.0)
        // SnakeBeta
        u = u + invBeta * pow(sin(alpha * u), Float(2.0))
        // DownSample1d: replicate pad (K/2 − even, K/2), stride-2 filter.
        let even = k % 2 == 0 ? 1 : 0
        let dl = k / 2 - even, dr = k / 2
        let n = u.dim(1)
        u = concatenated([repeated(u[0..., ..<1, 0...], count: dl, axis: 1), u,
                          repeated(u[0..., (n - 1)..., 0...], count: dr, axis: 1)], axis: 1)
        return conv1d(u, downFilter, stride: 2, groups: c)
    }
}

/// `AMPBlock1`: three (act → dilated conv → act → conv) residual units.
struct LTXAMPBlock1 {
    let acts1: [LTXActivation1d]
    let convs1: [LTXConv1d]
    let acts2: [LTXActivation1d]
    let convs2: [LTXConv1d]

    init(store: WeightStore, prefix: String, dilations: [Int]) throws {
        var a1: [LTXActivation1d] = [], c1: [LTXConv1d] = [], a2: [LTXActivation1d] = [], c2: [LTXConv1d] = []
        for (i, d) in dilations.enumerated() {
            a1.append(try LTXActivation1d(store: store, prefix: "\(prefix).acts1.\(i)"))
            c1.append(try LTXConv1d(store: store, prefix: "\(prefix).convs1.\(i)", dilation: d))
            a2.append(try LTXActivation1d(store: store, prefix: "\(prefix).acts2.\(i)"))
            c2.append(try LTXConv1d(store: store, prefix: "\(prefix).convs2.\(i)"))
        }
        acts1 = a1; convs1 = c1; acts2 = a2; convs2 = c2
    }

    var arrays: [MLXArray] {
        acts1.flatMap(\.arrays) + convs1.flatMap(\.arrays) + acts2.flatMap(\.arrays) + convs2.flatMap(\.arrays)
    }

    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        var x = x0
        for i in 0 ..< convs1.count {
            let r = x
            x = convs2[i](acts2[i](convs1[i](acts1[i](x))))
            x = x + r
        }
        return x
    }
}

/// BigVGAN-v2 generator (`BigVGANVocoder`): `[B, T, C_mel]` → `[B, T · ∏rates, C_out]`, float32 throughout.
public struct LTXBigVGAN {
    let convPre: LTXConv1d
    let ups: [LTXConvTranspose1d]
    let resblocks: [LTXAMPBlock1]
    let actPost: LTXActivation1d
    let convPost: LTXConv1d
    public let config: LTXBigVGANConfig
    /// Evaluate after every upsampling layer and residual block: bounds the lazy graph (lower peak memory), same numerics.
    public var materializeStages = true

    /// Loads `<prefix>conv_pre`, `ups.*`, `resblocks.*`, `act_post`, `conv_post` (weights upcast to float32).
    public init(store: WeightStore, prefix: String, config: LTXBigVGANConfig) throws {
        self.config = config
        convPre = try LTXConv1d(store: store, prefix: prefix + "conv_pre")
        var ups: [LTXConvTranspose1d] = []
        var rbs: [LTXAMPBlock1] = []
        let nk = config.resblockKernelSizes.count
        for (i, r) in config.upsampleRates.enumerated() {
            ups.append(try LTXConvTranspose1d(store: store, prefix: prefix + "ups.\(i)", stride: r))
            for j in 0 ..< nk {
                rbs.append(try LTXAMPBlock1(store: store, prefix: prefix + "resblocks.\(i * nk + j)",
                                            dilations: config.resblockDilations[j]))
            }
        }
        self.ups = ups
        resblocks = rbs
        actPost = try LTXActivation1d(store: store, prefix: prefix + "act_post")
        convPost = try LTXConv1d(store: store, prefix: prefix + "conv_post")
    }

    /// Every parameter tensor (for materialising after load).
    public var arrays: [MLXArray] {
        convPre.arrays + ups.flatMap(\.arrays) + resblocks.flatMap(\.arrays) + actPost.arrays + convPost.arrays
    }

    /// Total upsampling factor (output samples per input frame).
    public var hopLength: Int { config.upsampleRates.reduce(1, *) }

    /// `mel`: `[B, T, C_mel]` (any float dtype; computed in float32) → waveform `[B, T · hop, C_out]`.
    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel.asType(.float32))
        let nk = config.resblockKernelSizes.count
        for i in 0 ..< ups.count {
            x = ups[i](x)
            if materializeStages { eval(x) }
            var xs = resblocks[i * nk](x)
            if materializeStages { eval(xs) }
            for j in 1 ..< nk {
                xs = xs + resblocks[i * nk + j](x)
                if materializeStages { eval(xs) }
            }
            x = xs / Float(nk)
        }
        x = convPost(actPost(x))
        return config.finalTanh ? tanh(x) : x
    }
}
