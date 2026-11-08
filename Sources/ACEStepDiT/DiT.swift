import Foundation
import MLX
import MLXDiffusionCore

/// ACE-Step 1.5 diffusion transformer (`AceStepDiTModel`, the `decoder.` part of the turbo checkpoint).
///
/// Latents `[B, T, 64]` at 25 Hz are concatenated after the context latents `[B, T, 128]` (source latents + chunk
/// mask), patchified by 2 (`T/2` tokens of width 2048), run through 24 AdaLN DiT layers (self-attention with RoPE,
/// alternating bidirectional sliding-window (|i-j| <= 128 tokens) / full attention; cross-attention to the condition
/// encoder states; SwiGLU MLP), then de-patchified back to `[B, T, 64]`. Predicts the flow-matching velocity.
///
/// Attention masks follow the torch reference exactly: `AceStepDiTModel.forward` sets `attention_mask` and
/// `encoder_attention_mask` to `None` before building its 4D masks, so the only mask ever applied is the geometric
/// sliding-window one (no key padding mask on either the latent or the encoder side).
///
/// Precision: `dtype` is the activation dtype of the matmuls/attention, `weightDType` the storage dtype of the big
/// linears (default: as in the checkpoint, bf16), `residualDType` the residual stream (default float32). Timestep
/// embeddings and AdaLN modulation are always float32; the output velocity is float32.
/// Recommended: `dtype: .float32` on the bf16 weights (fp32-exact, 3.3 GB, ~90 ms/step at T = 500 on M4 Max).
/// `dtype: .bfloat16` is ~25% faster but the model is very bf16-sensitive at high t (velocity rel-L2 ~7e-2 at t = 1;
/// torch's own bf16 run is 2.4e-1), ~1.2e-1 rel-L2 on the final latent.
public final class ACEDiT {
    public struct Config {
        public var hiddenSize = 2048, intermediateSize = 6144, numLayers = 24
        public var numHeads = 16, numKVHeads = 8, headDim = 128
        public var inChannels = 192, outChannels = 64, patchSize = 2
        public var ropeTheta: Float = 1_000_000, rmsNormEps: Float = 1e-6
        public var slidingWindow = 128, useSlidingWindow = true
        public var layerTypes: [String] = (0..<24).map { $0 % 2 == 0 ? "sliding_attention" : "full_attention" }

        public init() {}

        /// From the checkpoint's `config.json` (`AceStepConfig`).
        public init(json d: [String: Any]) {
            func int(_ k: String, _ def: Int) -> Int { (d[k] as? NSNumber)?.intValue ?? def }
            func flt(_ k: String, _ def: Float) -> Float { (d[k] as? NSNumber)?.floatValue ?? def }
            hiddenSize = int("hidden_size", hiddenSize)
            intermediateSize = int("intermediate_size", intermediateSize)
            numLayers = int("num_hidden_layers", numLayers)
            numHeads = int("num_attention_heads", numHeads)
            numKVHeads = int("num_key_value_heads", numKVHeads)
            headDim = int("head_dim", hiddenSize / numHeads)
            inChannels = int("in_channels", inChannels)
            outChannels = int("audio_acoustic_hidden_dim", outChannels)
            patchSize = int("patch_size", patchSize)
            ropeTheta = flt("rope_theta", ropeTheta)
            rmsNormEps = flt("rms_norm_eps", rmsNormEps)
            slidingWindow = int("sliding_window", slidingWindow)
            useSlidingWindow = (d["use_sliding_window"] as? Bool) ?? useSlidingWindow
            layerTypes = (d["layer_types"] as? [String])
                ?? (0..<numLayers).map { $0 % 2 == 0 ? "sliding_attention" : "full_attention" }
        }

        public init(url: URL) throws {
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
            self.init(json: obj as? [String: Any] ?? [:])
        }
    }

    /// Per-layer cross-attention keys/values (`[B, KVH, S, headDim]`), computed once from the encoder states and
    /// reused for every sampler step (the reference caches them in an `EncoderDecoderCache` the same way).
    public struct CrossCache {
        public let kv: [(k: MLXArray, v: MLXArray)]
    }

    struct TimeEmbed {
        let l1, l2, proj: Lin  // float32
    }

    struct Layer {
        let selfNorm: MLXArray
        let qkv: Lin, o: Lin
        let qNorm, kNorm: MLXArray
        let crossNorm: MLXArray
        let cq: Lin, ckv: Lin, co: Lin
        let cqNorm, ckNorm: MLXArray
        let mlpNorm: MLXArray
        let gateUp: Lin, down: Lin
        let scaleShift: MLXArray  // [1, 6, D] float32
        let sliding: Bool
    }

    public let config: Config
    /// Activation dtype of the transformer body (matmul inputs, attention).
    public let dtype: DType
    /// Storage dtype of the large linear weights (the checkpoint's own dtype unless overridden).
    public private(set) var weightDType: DType = .bfloat16
    /// Dtype of the residual stream and AdaLN math (block norms, modulation, gated residual adds).
    public let residualDType: DType
    let timeEmbed, timeEmbedR: TimeEmbed
    let conditionEmbedder: Lin
    let projIn: Lin   // [D, patch*inC]
    let projOut: Lin  // [patch*outC, D]
    let normOut: MLXArray
    let outScaleShift: MLXArray  // [1, 2, D] float32
    let layers: [Layer]
    /// Development hook: receives named intermediates.
    public var trace: ((String, MLXArray) -> Void)?

    /// - Parameters:
    ///   - dtype: activation dtype for matmuls and attention (`.float32` or `.bfloat16`).
    ///   - weightDType: storage dtype of the big linear weights; nil keeps the checkpoint dtype (bf16 for the
    ///     official turbo weights). Upcasting bf16 weights is exact, so `dtype: .float32` with bf16 weights gives
    ///     the same float32 results as fp32 weights at half the resident memory (a bit slower: ~90 vs ~75 ms/step).
    ///   - residualDType: dtype of the residual stream between layers. Float32 by default: the residual
    ///     reaches |h| ~ 1e4, and keeping it in bf16 (what torch does with a bf16 model) costs ~10x accuracy for
    ///     almost no speed (see the probe numbers in the module docs).
    public init(store s: WeightStore, config: Config, dtype: DType = .float32, weightDType: DType? = nil,
                residualDType: DType = .float32, prefix p: String = "decoder.") throws {
        self.config = config
        self.dtype = dtype
        let wdt = weightDType
        self.residualDType = residualDType
        let rdt = residualDType
        let D = config.hiddenSize
        func w(_ k: String, _ dt: DType? = nil) throws -> MLXArray { try s.get(p + k).asType(dt ?? dtype) }
        func lin(_ k: String, _ dt: DType? = nil) throws -> Lin {
            let l = try s.linear(p + k, dtype: dt ?? wdt)  // nil: as stored
            return Lin(weight: l.weight, bias: l.bias?.asType(dt ?? dtype))
        }
        func fused(_ ks: [String]) throws -> Lin {
            let ls = try ks.map { try lin($0) }
            let bias: MLXArray? = ls.allSatisfy({ $0.bias != nil }) ? concatenated(ls.map { $0.bias! }, axis: 0) : nil
            return Lin(weight: concatenated(ls.map(\.weight), axis: 0), bias: bias)
        }
        func te(_ k: String) throws -> TimeEmbed {
            TimeEmbed(l1: try lin(k + ".linear_1", .float32), l2: try lin(k + ".linear_2", .float32),
                      proj: try lin(k + ".time_proj", .float32))
        }
        timeEmbed = try te("time_embed")
        timeEmbedR = try te("time_embed_r")
        conditionEmbedder = try lin("condition_embedder")

        // Conv1d(k = stride = patch) == linear over [patch, inC]-flattened frames. torch weight [out, in, k].
        let wIn = try w("proj_in.1.weight")
        guard wIn.ndim == 3, wIn.dim(2) == config.patchSize else { throw WeightError.badShape(p + "proj_in.1.weight", wIn.shape) }
        projIn = Lin(weight: wIn.transposed(0, 2, 1).reshaped(D, -1), bias: try w("proj_in.1.bias"))
        // ConvTranspose1d(k = stride = patch) == linear to [patch, outC]. torch weight [in, out, k].
        let wOut = try w("proj_out.1.weight")
        guard wOut.ndim == 3, wOut.dim(2) == config.patchSize else { throw WeightError.badShape(p + "proj_out.1.weight", wOut.shape) }
        let bOut = try w("proj_out.1.bias")
        projOut = Lin(weight: wOut.transposed(2, 1, 0).reshaped(-1, D),
                      bias: concatenated(Array(repeating: bOut, count: config.patchSize), axis: 0))
        normOut = try w("norm_out.weight", rdt)
        outScaleShift = try w("scale_shift_table", .float32)

        layers = try (0..<config.numLayers).map { i in
            let b = "layers.\(i)."
            return Layer(
                selfNorm: try w(b + "self_attn_norm.weight", rdt),
                qkv: try fused([b + "self_attn.q_proj", b + "self_attn.k_proj", b + "self_attn.v_proj"]),
                o: try lin(b + "self_attn.o_proj"),
                qNorm: try w(b + "self_attn.q_norm.weight"), kNorm: try w(b + "self_attn.k_norm.weight"),
                crossNorm: try w(b + "cross_attn_norm.weight", rdt),
                cq: try lin(b + "cross_attn.q_proj"),
                ckv: try fused([b + "cross_attn.k_proj", b + "cross_attn.v_proj"]),
                co: try lin(b + "cross_attn.o_proj"),
                cqNorm: try w(b + "cross_attn.q_norm.weight"), ckNorm: try w(b + "cross_attn.k_norm.weight"),
                mlpNorm: try w(b + "mlp_norm.weight", rdt),
                gateUp: try fused([b + "mlp.gate_proj", b + "mlp.up_proj"]),
                down: try lin(b + "mlp.down_proj"),
                scaleShift: try w(b + "scale_shift_table", .float32),
                sliding: config.useSlidingWindow && i < config.layerTypes.count && config.layerTypes[i] == "sliding_attention")
        }

        self.weightDType = layers.first?.qkv.weight.dtype ?? dtype
        var all: [MLXArray] = [conditionEmbedder.weight, projIn.weight, projOut.weight, normOut, outScaleShift]
        for t in [timeEmbed, timeEmbedR] { all += [t.l1.weight, t.l2.weight, t.proj.weight] }
        for l in layers { all += [l.qkv.weight, l.o.weight, l.cq.weight, l.ckv.weight, l.co.weight, l.gateUp.weight, l.down.weight] }
        eval(all)
    }

    /// Convenience: loads `config.json` + the `decoder.` tensors of `model.safetensors` from a checkpoint directory.
    public convenience init(checkpoint dir: URL, dtype: DType = .float32) throws {
        let store = try WeightStore(url: dir.appendingPathComponent("model.safetensors"))
        try self.init(store: store, config: Config(url: dir.appendingPathComponent("config.json")), dtype: dtype)
    }

    // MARK: - pieces

    /// `TimestepEmbedding`: sinusoid of `t·1000` (256 = [cos, sin]) → linear → SiLU → linear (= temb);
    /// SiLU → time_proj → `[B, 6, D]`.
    func embed(_ te: TimeEmbed, _ t: MLXArray) -> (MLXArray, MLXArray) {
        let f = sinusoid(t)
        let temb = te.l2(siluAct(te.l1(f)))
        let proj = te.proj(siluAct(temb)).reshaped(t.dim(0), 6, -1)
        return (temb, proj)
    }

    /// The 256-wide `[cos, sin]` features (`freqs = exp(−ln(1e4)·i/128)`), computed on the CPU like torch does:
    /// float32 argument `t·1000·freq` (up to 1000 rad), correctly rounded `cos`/`sin` of it. Only 256 values.
    func sinusoid(_ t: MLXArray) -> MLXArray {
        let ts = t.asType(.float32).asArray(Float.self)
        let half = 128
        let freqs = (0..<half).map { Float(exp(Double(-Float(log(10_000.0)) * Float($0) / Float(half)))) }
        var out = [Float](); out.reserveCapacity(ts.count * 2 * half)
        for tv in ts {
            let args = freqs.map { (tv * 1000) * $0 }
            out += args.map { Float(cos(Double($0))) }
            out += args.map { Float(sin(Double($0))) }
        }
        return MLXArray(out, [ts.count, 2 * half])
    }

    func rms(_ x: MLXArray, _ w: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: w, eps: config.rmsNormEps) }

    /// Cross-attention K/V for every layer from encoder hidden states `[B, S, 2048]` (after `condition_embedder`).
    public func crossCache(encoder: MLXArray) -> CrossCache {
        let B = encoder.dim(0), S = encoder.dim(1), KVH = config.numKVHeads, Dh = config.headDim
        let enc = conditionEmbedder(encoder.asType(dtype))
        let kv = layers.map { l -> (k: MLXArray, v: MLXArray) in
            let p = l.ckv(enc).split(parts: 2, axis: -1)
            let k = rms(p[0].reshaped(B, S, KVH, Dh), l.ckNorm).transposed(0, 2, 1, 3)
            let v = p[1].reshaped(B, S, KVH, Dh).transposed(0, 2, 1, 3)
            return (k, v)
        }
        eval(kv.flatMap { [$0.k, $0.v] })
        return CrossCache(kv: kv)
    }

    /// Bidirectional sliding-window keep-mask (`|i - j| <= window`), or nil when it would keep everything.
    func slidingMask(_ L: Int) -> MLXArray? {
        guard L > config.slidingWindow + 1 else { return nil }
        let i = MLXArray(0..<L).asType(.int32)
        return abs(i.reshaped(L, 1) - i.reshaped(1, L)) .<= Int32(config.slidingWindow)
    }

    // MARK: - forward

    /// Velocity `[B, T, 64]` (float32) for noisy latents `x` `[B, T, 64]` at flow time `t` (1 = noise),
    /// with `context` `[B, T, 128]` and encoder states `encoder` `[B, S, 2048]`. Pass a ``CrossCache`` built from the
    /// same `encoder` to skip recomputing cross-attention K/V (then `encoder` is not read).
    public func callAsFunction(_ x: MLXArray, t: Float, context: MLXArray, encoder: MLXArray,
                               cache: CrossCache? = nil) -> MLXArray {
        let cache = cache ?? crossCache(encoder: encoder)
        let B = x.dim(0), T = x.dim(1), D = config.hiddenSize
        let H = config.numHeads, KVH = config.numKVHeads, Dh = config.headDim, P = config.patchSize

        // Timestep conditioning: time_embed(t) + time_embed_r(t - r), with r = t in the sampler.
        let tv = MLXArray(Array(repeating: t, count: B))
        let (tembT, projT) = embed(timeEmbed, tv)
        let (tembR, projR) = embed(timeEmbedR, tv - tv)
        let temb = tembT + tembR       // [B, D]
        let tproj = projT + projR      // [B, 6, D]

        var h = concatenated([context.asType(dtype), x.asType(dtype)], axis: -1)
        var Tp = T
        if T % P != 0 {
            let pad = P - T % P
            h = concatenated([h, zeros([B, pad, h.dim(2)], dtype: dtype)], axis: 1)
            Tp += pad
        }
        let L = Tp / P
        h = projIn(h.reshaped(B, L, -1)).asType(residualDType)
        trace?("proj_in", h)

        let mask = slidingMask(L)
        let scale = 1 / Float(Dh).squareRoot()
        func rope(_ a: MLXArray) -> MLXArray {
            MLXFast.RoPE(a, dimensions: Dh, traditional: false, base: config.ropeTheta, scale: 1, offset: 0)
        }
        for (li, l) in layers.enumerated() {
            let m = (l.scaleShift + tproj).asType(residualDType).split(parts: 6, axis: 1)  // each [B, 1, D]
            let (shiftMSA, scaleMSA, gateMSA, cShift, cScale, cGate) = (m[0], m[1], m[2], m[3], m[4], m[5])

            // self-attention
            var n = (rms(h, l.selfNorm) * (1 + scaleMSA) + shiftMSA).asType(dtype)
            let qkv = l.qkv(n).split(indices: [H * Dh, (H + KVH) * Dh], axis: -1)
            let q = rope(rms(qkv[0].reshaped(B, L, H, Dh), l.qNorm).transposed(0, 2, 1, 3))
            let k = rope(rms(qkv[1].reshaped(B, L, KVH, Dh), l.kNorm).transposed(0, 2, 1, 3))
            let v = qkv[2].reshaped(B, L, KVH, Dh).transposed(0, 2, 1, 3)
            let a = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale,
                mask: (l.sliding ? mask : nil).map { .array($0) } ?? .none)
            h = h + l.o(a.transposed(0, 2, 1, 3).reshaped(B, L, H * Dh)).asType(residualDType) * gateMSA

            // cross-attention (plain residual, no modulation)
            let cq = rms(l.cq(rms(h, l.crossNorm).asType(dtype)).reshaped(B, L, H, Dh), l.cqNorm).transposed(0, 2, 1, 3)
            let ckv = cache.kv[li]
            let ca = MLXFast.scaledDotProductAttention(queries: cq, keys: ckv.k, values: ckv.v, scale: scale, mask: .none)
            h = h + l.co(ca.transposed(0, 2, 1, 3).reshaped(B, L, H * Dh)).asType(residualDType)

            // MLP
            n = (rms(h, l.mlpNorm) * (1 + cScale) + cShift).asType(dtype)
            let gu = l.gateUp(n).split(parts: 2, axis: -1)
            h = h + l.down(siluAct(gu[0]) * gu[1]).asType(residualDType) * cGate
            trace?("layer\(li)", h)
        }

        let ss = (outScaleShift + temb.reshaped(B, 1, D)).asType(residualDType).split(parts: 2, axis: 1)
        h = (rms(h, normOut) * (1 + ss[1]) + ss[0]).asType(dtype)
        h = projOut(h).reshaped(B, Tp, config.outChannels)
        if Tp != T { h = h[0..., 0..<T, 0...] }
        return h.asType(.float32)
    }
}
