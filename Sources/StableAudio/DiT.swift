import Foundation
import MLX
import MLXDiffusionCore

/// The Stable Audio diffusion transformer (stable-audio-tools `DiffusionTransformer` with a `ContinuousTransformer`):
/// latents `[1, T, 64]` plus a prepended global token (timestep + seconds), T5 cross-attention, partial rotary
/// embeddings, GLU feed-forward. Predicts the rectified-flow velocity.
public final class StableAudioDiT {
    public struct Config {
        public var ioChannels = 64, embedDim = 1024, depth = 16, numHeads = 8
        public var condTokenDim = 768, globalCondDim = 768
        public var normEps: Float = 1e-5
        /// stable-audio-tools `timestep_features_logsnr`.
        public var timestepLogSNR = false

        public init() {}

        /// From `model_config.json` → `model.diffusion.config`.
        public init(modelConfig: [String: Any]) {
            let d = ((modelConfig["model"] as? [String: Any])?["diffusion"] as? [String: Any])?["config"] as? [String: Any] ?? [:]
            ioChannels = d["io_channels"] as? Int ?? ioChannels
            embedDim = d["embed_dim"] as? Int ?? embedDim
            depth = d["depth"] as? Int ?? depth
            numHeads = d["num_heads"] as? Int ?? numHeads
            condTokenDim = d["cond_token_dim"] as? Int ?? condTokenDim
            globalCondDim = d["global_cond_dim"] as? Int ?? globalCondDim
            timestepLogSNR = d["timestep_features_logsnr"] as? Bool ?? false
        }
    }

    struct Norm {
        let gamma: MLXArray, beta: MLXArray?
        let eps: Float
        func callAsFunction(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: gamma, bias: beta, eps: eps) }
    }

    struct Block {
        let preNorm: Norm, qkv: Lin, selfOut: Lin, selfQNorm: Norm?, selfKNorm: Norm?
        let crossNorm: Norm, crossQ: Lin, crossKV: Lin, crossOut: Lin, crossQNorm: Norm?, crossKNorm: Norm?
        let ffNorm: Norm, ffIn: Lin, ffOut: Lin
    }

    public let config: Config
    let timestepFreqs: MLXArray  // [128, 1]
    let tEmbed0, tEmbed2, gEmbed0, gEmbed2, cEmbed0, cEmbed2: Lin
    let preConv, postConv: MLXArray  // 1×1 convs as [out, in]
    let projectIn, projectOut: Lin
    let invFreq: MLXArray
    let blocks: [Block]
    /// Development hook: receives named intermediates.
    public var trace: ((String, MLXArray) -> Void)?

    public init(store s: WeightStore, config: Config, prefix p: String = "model.model.") throws {
        self.config = config
        let eps = config.normEps
        func lin(_ k: String) throws -> Lin { try s.linear(p + k, dtype: .float32) }
        func norm(_ k: String) throws -> Norm {
            Norm(gamma: try s.get(p + k + ".gamma").asType(.float32), beta: s.optional(p + k + ".beta")?.asType(.float32),
                 eps: eps)
        }
        // q/k norms are plain nn.LayerNorm (weight/bias, eps 1e-5); block norms use gamma/beta.
        func optNorm(_ k: String) throws -> Norm? {
            if s.has(p + k + ".gamma") { return try norm(k) }
            guard let w = s.optional(p + k + ".weight") else { return nil }
            return Norm(gamma: w.asType(.float32), beta: s.optional(p + k + ".bias")?.asType(.float32), eps: 1e-5)
        }
        func conv1x1(_ k: String) throws -> MLXArray { try s.get(p + k + ".weight").asType(.float32).squeezed(axis: -1) }
        timestepFreqs = try s.get(p + "timestep_features.weight").asType(.float32)
        tEmbed0 = try lin("to_timestep_embed.0"); tEmbed2 = try lin("to_timestep_embed.2")
        gEmbed0 = try lin("to_global_embed.0"); gEmbed2 = try lin("to_global_embed.2")
        cEmbed0 = try lin("to_cond_embed.0"); cEmbed2 = try lin("to_cond_embed.2")
        preConv = try conv1x1("preprocess_conv"); postConv = try conv1x1("postprocess_conv")
        projectIn = try lin("transformer.project_in"); projectOut = try lin("transformer.project_out")
        let headDim = config.embedDim / config.numHeads
        let rotDim = max(headDim / 2, 32)
        if let f = s.optional(p + "transformer.rotary_pos_emb.inv_freq") {
            invFreq = f.asType(.float32)
        } else {
            invFreq = 1 / pow(MLXArray(Float(10_000)), MLXArray(stride(from: 0, to: rotDim, by: 2).map { Float($0) / Float(rotDim) }))
        }
        blocks = try (0..<config.depth).map { i in
            let b = "transformer.layers.\(i)."
            return Block(
                preNorm: try norm(b + "pre_norm"), qkv: try lin(b + "self_attn.to_qkv"), selfOut: try lin(b + "self_attn.to_out"),
                selfQNorm: try optNorm(b + "self_attn.q_norm"), selfKNorm: try optNorm(b + "self_attn.k_norm"),
                crossNorm: try norm(b + "cross_attend_norm"), crossQ: try lin(b + "cross_attn.to_q"),
                crossKV: try lin(b + "cross_attn.to_kv"), crossOut: try lin(b + "cross_attn.to_out"),
                crossQNorm: try optNorm(b + "cross_attn.q_norm"), crossKNorm: try optNorm(b + "cross_attn.k_norm"),
                ffNorm: try norm(b + "ff_norm"), ffIn: try lin(b + "ff.ff.0.proj"), ffOut: try lin(b + "ff.ff.2"))
        }
    }

    /// `[cos, sin]` of `2π · t · W` (stable-audio-tools `FourierFeatures`).
    func timestepFeatures(_ t: MLXArray) -> MLXArray {
        let f = 2 * Float.pi * matmul(t.reshaped(-1, 1), timestepFreqs.T)
        return concatenated([cos(f), sin(f)], axis: -1)
    }

    /// stable-audio-tools `_t_to_logsnr_cond`.
    func logSNRCond(_ t: MLXArray) -> MLXArray {
        let tc = clip(t, min: 1e-7, max: 1 - 1e-7)
        let logsnr = clip(log((1 - tc) / tc), min: -12, max: 5)
        return (5 - logsnr) / 17
    }

    func rope(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let r = c.dim(-1), half = r / 2
        let xr = x[.ellipsis, 0..<r], rest = x[.ellipsis, r...]
        let rotated = concatenated([-xr[.ellipsis, half...], xr[.ellipsis, 0..<half]], axis: -1)
        return concatenated([xr * c + rotated * s, rest], axis: -1)
    }

    /// Velocity for latents `x` `[B, T, 64]` at time `t` `[B]`, with cross-attention tokens `cond` `[B, S, 768]` and
    /// the global (seconds) embedding `global` `[B, 768]`.
    public func callAsFunction(_ x: MLXArray, t: MLXArray, cond: MLXArray, global: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1), H = config.numHeads, D = config.embedDim / H
        let ctx = cEmbed2(siluAct(cEmbed0(cond)))
        let tIn = config.timestepLogSNR ? logSNRCond(t) : t
        let tEmb = tEmbed2(siluAct(tEmbed0(timestepFeatures(tIn))))
        let g = gEmbed2(siluAct(gEmbed0(global))) + tEmb
        trace?("ctx", ctx); trace?("temb", tEmb); trace?("global", g)
        var h = x + matmul(x, preConv.T)
        h = concatenated([g.expandedDimensions(axis: 1), projectIn(h)], axis: 1)  // prepend the global token
        trace?("in", h)
        let L = T + 1
        let freqs = MLXArray(0..<L).asType(.float32).reshaped(L, 1) * invFreq.reshaped(1, -1)
        let f2 = concatenated([freqs, freqs], axis: -1)
        let (c, s) = (cos(f2), sin(f2))
        func heads(_ a: MLXArray, _ n: Int) -> MLXArray { a.reshaped(B, n, H, D).transposed(0, 2, 1, 3) }
        let S = ctx.dim(1)
        let scale = 1 / Float(D).squareRoot()
        for b in blocks {
            let qkv = b.qkv(b.preNorm(h)).split(parts: 3, axis: -1)
            var q = heads(qkv[0], L), k = heads(qkv[1], L)
            if let n = b.selfQNorm { q = n(q) }
            if let n = b.selfKNorm { k = n(k) }
            q = rope(q, cos: c, sin: s); k = rope(k, cos: c, sin: s)
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: heads(qkv[2], L), scale: scale, mask: .none)
            h = h + b.selfOut(a.transposed(0, 2, 1, 3).reshaped(B, L, H * D))

            var cq = heads(b.crossQ(b.crossNorm(h)), L)
            let kv = b.crossKV(ctx).split(parts: 2, axis: -1)
            var ck = heads(kv[0], S)
            if let n = b.crossQNorm { cq = n(cq) }
            if let n = b.crossKNorm { ck = n(ck) }
            let ca = MLXFast.scaledDotProductAttention(queries: cq, keys: ck, values: heads(kv[1], S), scale: scale, mask: .none)
            h = h + b.crossOut(ca.transposed(0, 2, 1, 3).reshaped(B, L, H * D))

            let ff = b.ffIn(b.ffNorm(h)).split(parts: 2, axis: -1)
            h = h + b.ffOut(ff[0] * siluAct(ff[1]))
            trace?("block", h)
        }
        var out = projectOut(h)[0..., 1..., 0...]
        out = out + matmul(out, postConv.T)
        return out
    }
}
