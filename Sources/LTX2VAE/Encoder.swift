import Foundation
import MLX
import MLXDiffusionCore

/// LTX-2.3 video VAE encoder (`vae_encoder.safetensors`, reference `VideoEncoder`).
///
/// Pixels `[B, F, H, W, 3]` in `[-1, 1]` (F = 1 + 8k, H and W multiples of 32) → **normalized** latent
/// `[B, 1 + (F−1)/8, H/32, W/32, 128]` (posterior mean, normalized with ``mean`` / ``std`` =
/// `per_channel_statistics._mean_of_means / _std_of_means`). Causal in time. Like the reference, the input dtype is
/// not changed (the pipeline feeds bf16, so the whole encode runs in bf16).
public final class LTXVideoEncoder {
    /// Per-channel latent statistics, `[128]`.
    public let mean: MLXArray
    public let std: MLXArray

    struct Down {
        let conv: LTXCausalConv3d
        let stride: (Int, Int, Int)
        let groupSize: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = x
            if stride.0 == 2 { x = concatenated([x[0..., 0 ..< 1], x], axis: 1) }
            var skip = ltxSpaceToDepth(x, st: stride.0, sh: stride.1, sw: stride.2)
            if groupSize > 1 {
                let s = skip.shape
                skip = skip.reshaped(s[0], s[1], s[2], s[3], s[4] / groupSize, groupSize).mean(axis: -1)
            }
            let y = ltxSpaceToDepth(conv(x), st: stride.0, sh: stride.1, sw: stride.2)
            return y + skip
        }
    }

    let convIn: LTXCausalConv3d
    let stages: [LTXResStage]  // down_blocks 0, 2, 4, 6, 8
    let downs: [Down]  // down_blocks 1, 3, 5, 7
    let convOut: LTXCausalConv3d

    /// Loads `vae_encoder.safetensors` (keys may carry the `vae_encoder.` file-stem prefix).
    public convenience init(url: URL) throws {
        try self.init(store: WeightStore(url: url))
    }

    public init(store raw: WeightStore) throws {
        let store = ltxStripPrefix(raw, "vae_encoder.")
        convIn = try LTXCausalConv3d(store: store, prefix: "conv_in.conv", causal: true)
        stages = try [0, 2, 4, 6, 8].map { try LTXResStage(store: store, prefix: "down_blocks.\($0)", causal: true) }
        // encoder_blocks: compress_space_res, compress_time_res, compress_all_res ×2
        let strides = [(1, 2, 2), (2, 1, 1), (2, 2, 2), (2, 2, 2)]
        var ds: [Down] = []
        for (i, idx) in [1, 3, 5, 7].enumerated() {
            let conv = try LTXCausalConv3d(store: store, prefix: "down_blocks.\(idx).conv.conv", causal: true)
            let s = strides[i]
            let prod = s.0 * s.1 * s.2
            let inCh = conv.weight.dim(4)
            let outCh = conv.weight.dim(0) * prod
            ds.append(Down(conv: conv, stride: s, groupSize: inCh * prod / outCh))
        }
        downs = ds
        convOut = try LTXCausalConv3d(store: store, prefix: "conv_out.conv", causal: true)
        mean = try store.get("per_channel_statistics._mean_of_means")
        std = try store.get("per_channel_statistics._std_of_means")
    }

    /// `(x - mean) / std` over the channel axis (reference `normalize_latent`).
    public func normalizeLatent(_ latent: MLXArray) -> MLXArray { (latent - mean) / std }

    /// `x * std + mean` over the channel axis (reference `denormalize_latent`).
    public func denormalizeLatent(_ latent: MLXArray) -> MLXArray { latent * std + mean }

    /// Encodes pixels `[B, F, H, W, 3]` in `[-1, 1]` to a normalized latent `[B, F', H/32, W/32, 128]`.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        var x = ltxPatchify(pixels, patch: 4)
        x = convIn(x)
        for i in 0..<4 {
            x = stages[i](x)
            x = downs[i](x)
        }
        x = stages[4](x)
        x = convOut(siluAct(ltxPixelNorm(x)))
        x = x[.ellipsis, 0 ..< 128]
        return normalizeLatent(x)
    }
}
