import Foundation
import MLX
import MLXDiffusionCore

/// Converts between transformer audio tokens `[B, T, 128]` and audio-VAE latents `[B, 8, T, 16]`
/// (`AudioPatchifier` in the reference).
public enum LTXAudioPatchifier {
    /// `[B, 8, T, 16]` → `[B, T, 128]`.
    public static func patchify(_ latent: MLXArray) -> MLXArray {
        let (b, c1, t, c2) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        return latent.transposed(0, 2, 1, 3).reshaped(b, t, c1 * c2)
    }

    /// `[B, T, 128]` → `[B, 8, T, 16]`.
    public static func unpatchify(_ tokens: MLXArray) -> MLXArray {
        let (b, t) = (tokens.dim(0), tokens.dim(1))
        return tokens.reshaped(b, t, 8, 16).transposed(0, 2, 1, 3)
    }
}

/// Conv2d already stored in MLX layout `[O, kh, kw, I]`, with optional causal padding along height (time):
/// `kh - 1` rows on top, none at the bottom, symmetric width padding (`WrappedConv2d(causal=True)`).
struct LTXCausalConv2d {
    let weight: MLXArray
    let bias: MLXArray?
    let causal: Bool

    init(weight: MLXArray, bias: MLXArray?, causal: Bool) {
        self.weight = weight
        self.bias = bias
        self.causal = causal
    }

    init(store: WeightStore, prefix: String, causal: Bool) throws {
        let w = try store.get(prefix + ".weight")
        guard w.ndim == 4 else { throw WeightError.badShape(prefix + ".weight", w.shape) }
        weight = w
        bias = store.optional(prefix + ".bias")
        self.causal = causal
    }

    var kernel: Int { weight.dim(1) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        let k = kernel
        if k > 1 {
            let pw = k / 2
            if causal {
                let xp = padded(x, widths: [0, .init((k - 1, 0)), .init(pw), 0])
                y = conv2d(xp, weight)
            } else {
                y = conv2d(x, weight, padding: .init(pw))
            }
        } else {
            y = conv2d(x, weight)
        }
        if let bias { y = y + bias }
        return y
    }
}

/// The LTX-2 audio VAE decoder (`AudioVAEDecoder`): latent `[B, 8, T, 16]` → stereo log-mel `[B, 2, 4T-3, 64]`.
///
/// The latent is treated as an NHWC image `[B, T, 16, 8]` (time = height, frequency = width). Convolutions are
/// causal along time, normalisation is PixelNorm (RMS over channels), and the per-channel latent statistics
/// (`per_channel_statistics`) are applied first. No attention (LTX-2.3: `attn_resolutions = []`,
/// `mid_block_add_attention = false`).
public final class LTXAudioVAEDecoder {
    struct ResBlock {
        let conv1: LTXCausalConv2d
        let conv2: LTXCausalConv2d
        let shortcut: LTXCausalConv2d?
    }

    struct UpStage {
        let blocks: [ResBlock]
        let upsample: LTXCausalConv2d?
    }

    public let dtype: DType
    let convIn: LTXCausalConv2d
    let mid: [ResBlock]
    /// Indexed like the checkpoint (`up.0` … `up.N`); executed in reverse order.
    let up: [UpStage]
    let convOut: LTXCausalConv2d
    let latentMean: MLXArray
    let latentStd: MLXArray

    /// Loads `audio_vae.decoder.*` and `audio_vae.per_channel_statistics.*` from `audio_vae.safetensors`.
    /// `dtype` is the compute/weight dtype (the reference runs the decoder in the checkpoint's bfloat16).
    public init(store: WeightStore, prefix: String = "audio_vae.", dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        let d = prefix + "decoder."
        func conv(_ p: String, causal: Bool = true) throws -> LTXCausalConv2d {
            var c = try LTXCausalConv2d(store: store, prefix: d + p, causal: causal)
            if c.weight.dtype != dtype || (c.bias.map { $0.dtype != dtype } ?? false) {
                c = LTXCausalConv2d(weight: c.weight.asType(dtype), bias: c.bias?.asType(dtype), causal: causal)
            }
            return c
        }
        func res(_ p: String) throws -> ResBlock {
            let sc = store.has(d + p + ".nin_shortcut.conv.weight") ? try conv(p + ".nin_shortcut.conv", causal: false) : nil
            return ResBlock(conv1: try conv(p + ".conv1.conv"), conv2: try conv(p + ".conv2.conv"), shortcut: sc)
        }
        convIn = try conv("conv_in.conv")
        mid = [try res("mid.block_1"), try res("mid.block_2")]
        var stages: [UpStage] = []
        var i = 0
        while store.has(d + "up.\(i).block.0.conv1.conv.weight") {
            var blocks: [ResBlock] = []
            var j = 0
            while store.has(d + "up.\(i).block.\(j).conv1.conv.weight") {
                blocks.append(try res("up.\(i).block.\(j)"))
                j += 1
            }
            let upw = d + "up.\(i).upsample.conv.conv.weight"
            stages.append(UpStage(blocks: blocks,
                                  upsample: store.has(upw) ? try conv("up.\(i).upsample.conv.conv") : nil))
            i += 1
        }
        up = stages
        convOut = try conv("conv_out.conv")
        latentMean = try store.get(prefix + "per_channel_statistics._mean_of_means").asType(dtype)
        latentStd = try store.get(prefix + "per_channel_statistics._std_of_means").asType(dtype)
    }

    static func pixelNorm(_ x: MLXArray) -> MLXArray {
        rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: 1e-6)
    }

    static func resBlock(_ b: ResBlock, _ x: MLXArray) -> MLXArray {
        var h = b.conv1(siluAct(pixelNorm(x)))
        h = b.conv2(siluAct(pixelNorm(h)))
        return h + (b.shortcut.map { $0(x) } ?? x)
    }

    /// Runs `conv_in` → mid → up stages (reverse index order). `x`: NHWC `[B, T, 16, 8]` (already denormalised).
    /// `taps` receives intermediate activations (`conv_in`, `mid`, `up<i>`) for testing.
    func trunk(_ x0: MLXArray, taps: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        var x = convIn(x0)
        taps?("dec_conv_in", x)
        for b in mid { x = Self.resBlock(b, x) }
        taps?("dec_mid", x)
        for i in stride(from: up.count - 1, through: 0, by: -1) {
            for b in up[i].blocks { x = Self.resBlock(b, x) }
            if let u = up[i].upsample {
                x = repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2)
                x = u(x)
                x = x[0..., 1..., 0..., 0...]  // causal: drop the first time row
            }
            taps?("dec_up\(i)", x)
        }
        return x
    }

    /// Latent `[B, 8, T, 16]` → mel `[B, 2, T', 64]` (T' = 4T − 3), in ``dtype``.
    public func decode(latent: MLXArray, taps: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let (b, c1, t, c2) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
        var flat = latent.asType(dtype).transposed(0, 2, 1, 3).reshaped(b, t, c1 * c2)
        flat = flat * latentStd + latentMean
        let x = flat.reshaped(b, t, c1, c2).transposed(0, 1, 3, 2)  // [B, T, 16, 8]
        var h = trunk(x, taps: taps)
        h = convOut(siluAct(Self.pixelNorm(h)))  // [B, T', 64, 2]
        return h.transposed(0, 3, 1, 2)
    }

    /// Mel frames produced for `tokens` latent frames (each causal 2× upsample maps T → 2T − 1).
    public func melFrames(forTokens tokens: Int) -> Int {
        var t = tokens
        for s in up where s.upsample != nil { t = 2 * t - 1 }
        return t
    }
}
