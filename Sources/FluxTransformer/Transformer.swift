import Foundation
import MLX
import MLXDiffusionCore

/// The FLUX.1 MMDiT (port of diffusers' `FluxTransformer2DModel`, loaded from its own key names).
///
/// 19 double-stream blocks (separate text and image weights, joint attention), then 38 single-stream blocks
/// (parallel attention + MLP over the joined sequence). Matmuls run on the stored (quantized or bf16) weights with
/// bf16 activations; the residual streams, norms, modulation and rotary embedding are float32.
public final class FluxTransformer {
    public struct Config: Sendable {
        public var inChannels = 64, numLayers = 19, numSingleLayers = 38, headDim = 128, numHeads = 24
        public var jointDim = 4096, pooledDim = 768
        public var guidanceEmbeds = true
        public var axesDims = [16, 56, 56]
        public var theta: Double = 10_000

        public var dim: Int { headDim * numHeads }

        public init() {}

        /// Reads diffusers' `transformer/config.json`.
        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            if let v = o["in_channels"] as? Int { inChannels = v }
            if let v = o["num_layers"] as? Int { numLayers = v }
            if let v = o["num_single_layers"] as? Int { numSingleLayers = v }
            if let v = o["attention_head_dim"] as? Int { headDim = v }
            if let v = o["num_attention_heads"] as? Int { numHeads = v }
            if let v = o["joint_attention_dim"] as? Int { jointDim = v }
            if let v = o["pooled_projection_dim"] as? Int { pooledDim = v }
            if let v = o["guidance_embeds"] as? Bool { guidanceEmbeds = v }
            if let v = o["axes_dims_rope"] as? [Int] { axesDims = v }
        }
    }

    struct DoubleBlock {
        var norm1, norm1Context: Lin
        var q, k, v, out, addQ, addK, addV, addOut: Lin
        let normQ, normK, normAddQ, normAddK: MLXArray
        var ff0, ff2, ffc0, ffc2: Lin

        mutating func mapLinears(_ f: (Lin) -> Lin) {
            norm1 = f(norm1); norm1Context = f(norm1Context)
            q = f(q); k = f(k); v = f(v); out = f(out); addQ = f(addQ); addK = f(addK); addV = f(addV)
            addOut = f(addOut); ff0 = f(ff0); ff2 = f(ff2); ffc0 = f(ffc0); ffc2 = f(ffc2)
        }
        var linears: [Lin] { [norm1, norm1Context, q, k, v, out, addQ, addK, addV, addOut, ff0, ff2, ffc0, ffc2] }
    }

    struct SingleBlock {
        var norm, q, k, v, projMLP, projOut: Lin
        let normQ, normK: MLXArray

        mutating func mapLinears(_ f: (Lin) -> Lin) {
            norm = f(norm); q = f(q); k = f(k); v = f(v); projMLP = f(projMLP); projOut = f(projOut)
        }
        var linears: [Lin] { [norm, q, k, v, projMLP, projOut] }
    }

    public let config: Config
    var xEmbedder, contextEmbedder: Lin
    var time1, time2, text1, text2: Lin
    var guidance1, guidance2: Lin?
    var doubleBlocks: [DoubleBlock]
    var singleBlocks: [SingleBlock]
    var normOut, projOut: Lin
    /// Matmul activation dtype: bf16 (as the reference runs), or float32 for exact module checks.
    let dtype: DType

    public init(store: WeightStore, config: Config, activations: DType = .bfloat16) throws {
        self.config = config
        dtype = activations
        func lin(_ k: String) throws -> Lin { try store.linear(k, dtype: .bfloat16) }
        func w(_ k: String) throws -> MLXArray { try store.get(k).asType(.float32) }
        xEmbedder = try lin("x_embedder")
        contextEmbedder = try lin("context_embedder")
        time1 = try lin("time_text_embed.timestep_embedder.linear_1")
        time2 = try lin("time_text_embed.timestep_embedder.linear_2")
        text1 = try lin("time_text_embed.text_embedder.linear_1")
        text2 = try lin("time_text_embed.text_embedder.linear_2")
        if config.guidanceEmbeds {
            guidance1 = try lin("time_text_embed.guidance_embedder.linear_1")
            guidance2 = try lin("time_text_embed.guidance_embedder.linear_2")
        }
        doubleBlocks = try (0..<config.numLayers).map { i in
            let p = "transformer_blocks.\(i)."
            return DoubleBlock(
                norm1: try lin(p + "norm1.linear"), norm1Context: try lin(p + "norm1_context.linear"),
                q: try lin(p + "attn.to_q"), k: try lin(p + "attn.to_k"), v: try lin(p + "attn.to_v"),
                out: try lin(p + "attn.to_out.0"), addQ: try lin(p + "attn.add_q_proj"),
                addK: try lin(p + "attn.add_k_proj"), addV: try lin(p + "attn.add_v_proj"),
                addOut: try lin(p + "attn.to_add_out"),
                normQ: try w(p + "attn.norm_q.weight"), normK: try w(p + "attn.norm_k.weight"),
                normAddQ: try w(p + "attn.norm_added_q.weight"), normAddK: try w(p + "attn.norm_added_k.weight"),
                ff0: try lin(p + "ff.net.0.proj"), ff2: try lin(p + "ff.net.2"),
                ffc0: try lin(p + "ff_context.net.0.proj"), ffc2: try lin(p + "ff_context.net.2"))
        }
        singleBlocks = try (0..<config.numSingleLayers).map { i in
            let p = "single_transformer_blocks.\(i)."
            return SingleBlock(norm: try lin(p + "norm.linear"), q: try lin(p + "attn.to_q"),
                               k: try lin(p + "attn.to_k"), v: try lin(p + "attn.to_v"),
                               projMLP: try lin(p + "proj_mlp"), projOut: try lin(p + "proj_out"),
                               normQ: try w(p + "attn.norm_q.weight"), normK: try w(p + "attn.norm_k.weight"))
        }
        normOut = try lin("norm_out.linear")
        projOut = try lin("proj_out")
    }

    private init(copying o: FluxTransformer) {
        config = o.config; dtype = o.dtype; xEmbedder = o.xEmbedder; contextEmbedder = o.contextEmbedder
        time1 = o.time1; time2 = o.time2; text1 = o.text1; text2 = o.text2
        guidance1 = o.guidance1; guidance2 = o.guidance2
        doubleBlocks = o.doubleBlocks; singleBlocks = o.singleBlocks; normOut = o.normOut; projOut = o.projOut
    }

    var topLinears: [Lin] {
        [xEmbedder, contextEmbedder, time1, time2, text1, text2, normOut, projOut] + [guidance1, guidance2].compactMap { $0 }
    }

    /// Names of every linear layer (diffusers key prefixes), for matching LoRA files.
    public var linearNames: Set<String> {
        var names = topLinears.map(\.name)
        for b in doubleBlocks { names += b.linears.map(\.name) }
        for b in singleBlocks { names += b.linears.map(\.name) }
        return Set(names)
    }

    /// Output widths by layer name (for splitting fused LoRA projections).
    public var linearOutFeatures: [String: Int] {
        var out: [String: Int] = [:]
        for l in topLinears { out[l.name] = l.outFeatures }
        for b in doubleBlocks { for l in b.linears { out[l.name] = l.outFeatures } }
        for b in singleBlocks { for l in b.linears { out[l.name] = l.outFeatures } }
        return out
    }

    /// A transformer sharing these weights with `adapters` added to the matching layers; returns the match count.
    public func adding(_ adapters: [String: LoRAAdapter]) -> (FluxTransformer, Int) {
        let t = FluxTransformer(copying: self)
        var matched = 0
        func f(_ l: Lin) -> Lin {
            guard let a = adapters[l.name] else { return l }
            matched += 1
            return l.adding(a)
        }
        t.xEmbedder = f(t.xEmbedder); t.contextEmbedder = f(t.contextEmbedder)
        t.time1 = f(t.time1); t.time2 = f(t.time2); t.text1 = f(t.text1); t.text2 = f(t.text2)
        t.guidance1 = t.guidance1.map(f); t.guidance2 = t.guidance2.map(f)
        t.normOut = f(t.normOut); t.projOut = f(t.projOut)
        for i in t.doubleBlocks.indices { t.doubleBlocks[i].mapLinears(f) }
        for i in t.singleBlocks.indices { t.singleBlocks[i].mapLinears(f) }
        return (t, matched)
    }

    // MARK: embeddings

    /// diffusers `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)`: `[cos, sin]` of `t · e^(−ln(10⁴)·i/128)`.
    static func sinusoid(_ t: Float, dim: Int = 256) -> MLXArray {
        let half = dim / 2
        let freqs = (0..<half).map { i in Float(exp(-log(10_000.0) * Double(i) / Double(half))) }
        let args = MLXArray(freqs) * t
        return concatenated([cos(args), sin(args)]).reshaped(1, dim)
    }

    /// Rounds to bfloat16.
    static func bf16(_ x: Float) -> Float { MLXArray(x).asType(.bfloat16).asType(.float32).item(Float.self) }

    /// The embedded time value as the reference (and BFL's original code) computes it in the model dtype: the value
    /// is cast to bf16 and multiplied by 1000 in bf16, so σ 0.7 embeds as t = 700 and guidance 3.5 as 3504.
    static func bf16Times1000(_ x: Float) -> Float { bf16(bf16(x) * 1000) }

    /// The conditioning vector `[1, dim]` (float32) from the sigma, the distilled guidance and the pooled CLIP output.
    public func conditioningVector(sigma: Float, guidance: Float, pooled: MLXArray) -> MLXArray {
        func mlp(_ a: Lin, _ b: Lin, _ x: MLXArray) -> MLXArray {
            b(siluAct(a(x.asType(dtype)).asType(.float32)).asType(dtype)).asType(.float32)
        }
        let scaled: (Float) -> Float = dtype == .bfloat16 ? Self.bf16Times1000 : { $0 * 1000 }
        var v = mlp(time1, time2, Self.sinusoid(scaled(sigma)))
        if let g1 = guidance1, let g2 = guidance2 { v = v + mlp(g1, g2, Self.sinusoid(scaled(guidance))) }
        return v + mlp(text1, text2, pooled.reshaped(1, -1))
    }

    /// Rotary tables `(cos, sin)` `[S, headDim/2]` (float32) for `textLength` text tokens (ids 0) followed by an
    /// `h × w` grid of image tokens (ids `(0, row, col)`), per axis `θ^(−2i/d)` in double precision.
    public func rope(textLength: Int, height h: Int, width w: Int) -> (MLXArray, MLXArray) {
        let axes = config.axesDims
        let freqs = axes.map { d in (0 ..< d / 2).map { i in 1 / pow(config.theta, Double(2 * i) / Double(d)) } }
        let half = config.headDim / 2
        let S = textLength + h * w
        var c = [Float](repeating: 1, count: S * half), s = [Float](repeating: 0, count: S * half)
        for r in 0..<h {
            for col in 0..<w {
                let base = (textLength + r * w + col) * half
                var j = 0
                for (axis, f) in freqs.enumerated() {
                    let pos = Double(axis == 0 ? 0 : axis == 1 ? r : col)
                    for fr in f {
                        c[base + j] = Float(cos(pos * fr)); s[base + j] = Float(sin(pos * fr))
                        j += 1
                    }
                }
            }
        }
        return (MLXArray(c, [S, half]), MLXArray(s, [S, half]))
    }

    /// Interleaved-pair rotation of `x` `[B, H, S, D]` (float32).
    static func applyRope(_ x: MLXArray, _ cs: (MLXArray, MLXArray)) -> MLXArray {
        let (B, H, S, D) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let p = x.reshaped(B, H, S, D / 2, 2)
        let re = p[.ellipsis, 0], im = p[.ellipsis, 1]
        let (c, s) = cs
        return stacked([re * c - im * s, re * s + im * c], axis: -1).reshaped(B, H, S, D)
    }

    // MARK: forward

    func layerNorm(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) }

    /// `x · (1 + scale) + shift`, cast to the matmul dtype.
    func modulate(_ x: MLXArray, _ shift: MLXArray, _ scale: MLXArray) -> MLXArray {
        (layerNorm(x) * (1 + scale) + shift).asType(dtype)
    }

    /// Projects to heads `[1, H, S, D]` with the per-head RMSNorm (float32).
    func heads(_ lin: Lin, _ x: MLXArray, norm: MLXArray?) -> MLXArray {
        let S = x.dim(1)
        let h = lin(x).reshaped(1, S, config.numHeads, config.headDim).transposed(0, 2, 1, 3).asType(.float32)
        guard let norm else { return h }
        return MLXFast.rmsNorm(h, weight: norm, eps: 1e-6)
    }

    func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rope: (MLXArray, MLXArray)) -> MLXArray {
        let qr = Self.applyRope(q, rope).asType(dtype), kr = Self.applyRope(k, rope).asType(dtype)
        let o = MLXFast.scaledDotProductAttention(queries: qr, keys: kr, values: v.asType(dtype),
                                                  scale: 1 / Float(config.headDim).squareRoot(), mask: .none)
        return o.transposed(0, 2, 1, 3).reshaped(1, q.dim(2), config.dim)  // [1, S, dim] bf16
    }

    func chunks(_ lin: Lin, _ vec: MLXArray, _ n: Int) -> [MLXArray] {
        let m = lin(siluAct(vec).asType(dtype)).asType(.float32)  // [1, n·dim]
        return split(m, parts: n, axis: -1).map { $0.expandedDimensions(axis: 1) }  // each [1, 1, dim]
    }

    /// Velocity `[1, S_img, 64]` (float32) for packed latents `latents` `[1, S_img, 64]`.
    ///
    /// - Parameters:
    ///   - context: T5 hidden states `[1, L, 4096]`.
    ///   - vec: ``conditioningVector(sigma:guidance:pooled:)``.
    ///   - rope: ``rope(textLength:height:width:)`` for `L` text tokens and the image grid.
    public func callAsFunction(_ latents: MLXArray, context: MLXArray, vec: MLXArray,
                               rope: (MLXArray, MLXArray)) -> MLXArray {
        let L = context.dim(1)
        var img = xEmbedder(latents.asType(dtype)).asType(.float32)
        var txt = contextEmbedder(context.asType(dtype)).asType(.float32)

        for b in doubleBlocks {
            let m = chunks(b.norm1, vec, 6), mc = chunks(b.norm1Context, vec, 6)
            let ni = modulate(img, m[0], m[1]), nt = modulate(txt, mc[0], mc[1])
            let q = concatenated([heads(b.addQ, nt, norm: b.normAddQ), heads(b.q, ni, norm: b.normQ)], axis: 2)
            let k = concatenated([heads(b.addK, nt, norm: b.normAddK), heads(b.k, ni, norm: b.normK)], axis: 2)
            let v = concatenated([heads(b.addV, nt, norm: nil), heads(b.v, ni, norm: nil)], axis: 2)
            let a = attention(q, k, v, rope: rope)
            let at = a[0..., ..<L], ai = a[0..., L...]
            img = img + m[2] * b.out(ai).asType(.float32)
            img = img + m[5] * b.ff2(geluTanh(b.ff0(modulate(img, m[3], m[4])))).asType(.float32)
            txt = txt + mc[2] * b.addOut(at).asType(.float32)
            txt = txt + mc[5] * b.ffc2(geluTanh(b.ffc0(modulate(txt, mc[3], mc[4])))).asType(.float32)
        }

        var x = concatenated([txt, img], axis: 1)
        for b in singleBlocks {
            let m = chunks(b.norm, vec, 3)
            let n = modulate(x, m[0], m[1])
            let a = attention(heads(b.q, n, norm: b.normQ), heads(b.k, n, norm: b.normK), heads(b.v, n, norm: nil),
                              rope: rope)
            let mlp = geluTanh(b.projMLP(n))
            x = x + m[2] * b.projOut(concatenated([a, mlp], axis: -1)).asType(.float32)
        }
        x = x[0..., L...]
        // AdaLayerNormContinuous: (scale, shift) order.
        let m = chunks(normOut, vec, 2)
        return projOut(modulate(x, m[1], m[0])).asType(.float32)
    }
}
