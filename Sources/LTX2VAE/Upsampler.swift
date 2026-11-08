import Foundation
import MLX
import MLXDiffusionCore

/// LTX-2 latent upsampler, spatial ×2 variant (`spatial_upscaler_x2_v1_1.safetensors`, reference `LatentUpsampler`
/// with `spatial_upsample = true`, `rational_resampler = false`).
///
/// Works on **denormalized** latents `[B, F, H, W, 128]` → `[B, F, 2H, 2W, 128]`. Use
/// ``upsampleNormalized(_:encoder:)`` for the pipeline's normalized latents. Like the reference, the input dtype is
/// kept (bf16 in the pipeline; a float32 input runs in float32 against the bf16 weights).
public final class LTXLatentUpsampler {
    struct Conv {
        let weight: MLXArray
        let bias: MLXArray

        init(store: WeightStore, prefix: String) throws {
            weight = try store.get(prefix + ".weight")
            bias = try store.get(prefix + ".bias")
        }

        /// Zero-padded 3-D (5-D input) or 2-D (4-D input) conv, padding 1.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            if weight.ndim == 5 { return conv3d(x, weight, padding: 1) + bias }
            return conv2d(x, weight, padding: 1) + bias
        }
    }

    struct Norm {
        let weight: MLXArray
        let bias: MLXArray

        init(store: WeightStore, prefix: String) throws {
            weight = try store.get(prefix + ".weight")
            bias = try store.get(prefix + ".bias")
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { ltxGroupNorm(x, groups: 32, weight: weight, bias: bias) }
    }

    struct ResBlock {
        let conv1: Conv, norm1: Norm, conv2: Conv, norm2: Norm

        init(store: WeightStore, prefix: String) throws {
            conv1 = try Conv(store: store, prefix: prefix + ".conv1")
            norm1 = try Norm(store: store, prefix: prefix + ".norm1")
            conv2 = try Conv(store: store, prefix: prefix + ".conv2")
            norm2 = try Norm(store: store, prefix: prefix + ".norm2")
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = siluAct(norm1(conv1(x)))
            y = norm2(conv2(y))
            return siluAct(y + x)
        }
    }

    let initialConv: Conv
    let initialNorm: Norm
    let resBlocks: [ResBlock]
    let upsampler: Conv
    let postBlocks: [ResBlock]
    let finalConv: Conv
    /// Debug hook: receives named intermediates during ``callAsFunction(_:)``.
    public var debugTrace: ((String, MLXArray) -> Void)?

    /// Loads the upsampler weights. `configURL` (`<stem>_config.json`) is checked when given: only the
    /// `spatial_upsample` ×2 non-rational variant is ported.
    public convenience init(url: URL, configURL: URL? = nil) throws {
        if let configURL, let data = try? Data(contentsOf: configURL),
           let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            let c = obj["config"] as? [String: Any] ?? obj
            let spatial = c["spatial_upsample"] as? Bool ?? true
            let temporal = c["temporal_upsample"] as? Bool ?? false
            let rational = c["rational_resampler"] as? Bool ?? false
            let scale = c["spatial_scale"] as? Double ?? 2
            guard spatial, !temporal, !rational, scale == 2 else {
                throw WeightError.missing("unsupported LatentUpsampler variant in \(configURL.lastPathComponent)")
            }
        }
        let stem = url.deletingPathExtension().lastPathComponent
        try self.init(store: ltxStripPrefix(WeightStore(url: url), stem + "."))
    }

    public init(store: WeightStore) throws {
        initialConv = try Conv(store: store, prefix: "initial_conv")
        initialNorm = try Norm(store: store, prefix: "initial_norm")
        var rb: [ResBlock] = []
        while store.has("res_blocks.\(rb.count).conv1.weight") {
            rb.append(try ResBlock(store: store, prefix: "res_blocks.\(rb.count)"))
        }
        var pb: [ResBlock] = []
        while store.has("post_upsample_res_blocks.\(pb.count).conv1.weight") {
            pb.append(try ResBlock(store: store, prefix: "post_upsample_res_blocks.\(pb.count)"))
        }
        resBlocks = rb
        postBlocks = pb
        upsampler = try Conv(store: store, prefix: "upsampler.0")
        guard upsampler.weight.ndim == 4 else { throw WeightError.badShape("upsampler.0.weight", upsampler.weight.shape) }
        finalConv = try Conv(store: store, prefix: "final_conv")
    }

    /// Denormalized latent `[B, F, H, W, 128]` → `[B, F, 2H, 2W, 128]`.
    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        var x = initialConv(latent)
        debugTrace?("iconv", x)
        x = initialNorm(x)
        debugTrace?("inorm", x)
        x = siluAct(x)
        debugTrace?("isilu", x)
        if let t = debugTrace, let r = resBlocks.first {
            var y = r.conv1(x); t("rb0_c1", y)
            y = r.norm1(y); t("rb0_n1", y)
            y = siluAct(y); t("rb0_s1", y)
            y = r.conv2(y); t("rb0_c2", y)
            y = r.norm2(y); t("rb0_n2", y)
        }
        for (i, b) in resBlocks.enumerated() {
            x = b(x)
            debugTrace?("rb\(i)", x)
        }
        let (b, f, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        x = ltxPixelShuffle2d(upsampler(x.reshaped(b * f, h, w, c)), factor: 2)
        x = x.reshaped(b, f, 2 * h, 2 * w, x.dim(-1))
        debugTrace?("ups", x)
        if let t = debugTrace, let r = postBlocks.first {
            var y = r.conv1(x); t("pb0_c1", y)
            y = r.norm1(y); t("pb0_n1", y)
            y = siluAct(y); t("pb0_s1", y)
            y = r.conv2(y); t("pb0_c2", y)
            y = r.norm2(y); t("pb0_n2", y)
            t("pb0_add", y + x)
        }
        for (i, b) in postBlocks.enumerated() {
            x = b(x)
            debugTrace?("pb\(i)", x)
        }
        return finalConv(x)
    }

    /// Pipeline `_upsample_latent`: denormalize with the encoder statistics, upsample, renormalize, evaluate.
    public func upsampleNormalized(_ latent: MLXArray, encoder: LTXVideoEncoder) -> MLXArray {
        let out = encoder.normalizeLatent(self(encoder.denormalizeLatent(latent)))
        eval(out)
        return out
    }
}
