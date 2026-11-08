import Foundation
import MLX
import MLXDiffusionCore

/// The Gemma 3 text decoder (mlx-lm `gemma3_text`), used as a feature extractor: it returns the hidden states of
/// every layer (embedding output + one per decoder layer) instead of logits.
///
/// Weights are the mlx-community 4-bit checkpoint (`language_model.model.*`, affine quantization with the bits and
/// group size of `config.json` → `quantization`). The vision tower and the LM head are not loaded.
public final class LTXGemma3Model {
    /// Text-model hyper-parameters, with mlx-lm's `gemma3_text.ModelArgs` defaults for fields the checkpoint omits.
    public struct Config {
        public var hiddenSize = 1152
        public var numLayers = 26
        public var intermediateSize = 6912
        public var numHeads = 4
        public var headDim = 256
        public var rmsEps: Float = 1e-6
        public var numKVHeads = 1
        public var ropeTheta: Float = 1_000_000
        public var ropeLocalBase: Float = 10_000
        public var queryPreAttnScalar: Float = 256
        public var slidingWindow = 512
        public var slidingWindowPattern = 6
        /// `1 / factor` for `rope_scaling.rope_type == "linear"` (global layers only), else 1.
        public var globalRopeScale: Float = 1
        public var quantBits = 4
        public var quantGroupSize = 64

        public init(url: URL) throws {
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            // mlx-lm `gemma3.ModelArgs.__post_init__` defaults heads to 8 / kv heads to 4 when text_config lacks them.
            let isWrapper = obj["text_config"] != nil
            let t = obj["text_config"] as? [String: Any] ?? obj
            func int(_ k: String, _ d: Int) -> Int { (t[k] as? NSNumber)?.intValue ?? d }
            func flt(_ k: String, _ d: Float) -> Float { (t[k] as? NSNumber)?.floatValue ?? d }
            hiddenSize = int("hidden_size", hiddenSize)
            numLayers = int("num_hidden_layers", numLayers)
            intermediateSize = int("intermediate_size", intermediateSize)
            numHeads = int("num_attention_heads", isWrapper ? 8 : numHeads)
            numKVHeads = int("num_key_value_heads", isWrapper ? 4 : numKVHeads)
            headDim = int("head_dim", headDim)
            rmsEps = flt("rms_norm_eps", rmsEps)
            ropeTheta = flt("rope_theta", ropeTheta)
            ropeLocalBase = flt("rope_local_base_freq", ropeLocalBase)
            queryPreAttnScalar = flt("query_pre_attn_scalar", queryPreAttnScalar)
            slidingWindow = int("sliding_window", slidingWindow)
            slidingWindowPattern = int("sliding_window_pattern", slidingWindowPattern)
            if let rs = t["rope_scaling"] as? [String: Any] {
                let type = (rs["type"] as? String) ?? (rs["rope_type"] as? String) ?? "default"
                if type == "linear", let f = (rs["factor"] as? NSNumber)?.floatValue { globalRopeScale = 1 / f }
            }
            let q = (obj["quantization"] as? [String: Any]) ?? (obj["quantization_config"] as? [String: Any]) ?? [:]
            quantBits = (q["bits"] as? NSNumber)?.intValue ?? quantBits
            quantGroupSize = (q["group_size"] as? NSNumber)?.intValue ?? quantGroupSize
        }

        /// Layer `i` uses sliding-window attention with the local RoPE base (mlx-lm `is_sliding`).
        public func isSliding(_ i: Int) -> Bool { (i + 1) % slidingWindowPattern != 0 }
    }

    struct Layer {
        let inputNorm, postAttnNorm, preFFNorm, postFFNorm: MLXArray  // already `1 + weight`
        let q, k, v, o: Lin
        let qNorm, kNorm: MLXArray  // already `1 + weight`
        let gate, up, down: Lin
    }

    /// Quantized (or dense) token embedding table.
    struct Embedding {
        let weight: MLXArray
        let scales: MLXArray?
        let biases: MLXArray?
        let bits: Int
        let groupSize: Int

        func callAsFunction(_ ids: MLXArray) -> MLXArray {
            guard let scales else { return weight[ids] }
            return dequantized(weight[ids], scales: scales[ids], biases: biases?[ids], groupSize: groupSize, bits: bits)
        }
    }

    public let config: Config
    let embed: Embedding
    let layers: [Layer]

    /// Loads the text decoder from an mlx-community Gemma 3 folder (`config.json` + `*.safetensors`).
    public init(directory: URL, dtype: DType = .bfloat16) throws {
        config = try Config(url: directory.appendingPathComponent("config.json"))
        let store = try WeightStore(directory: directory)
        store.bitsOverride = config.quantBits
        for p in ["vision_tower.", "multi_modal_projector.", "language_model.lm_head.", "lm_head."] {
            store.remove(prefix: p)
        }
        let candidates = ["language_model.model.", "model.language_model.", "model.", ""]
        guard let p = candidates.first(where: { store.has($0 + "embed_tokens.weight") }) else {
            throw WeightError.missing("embed_tokens.weight")
        }
        let c = config
        func norm(_ k: String) throws -> MLXArray {
            let w = try store.get(k)
            return (1 + w.asType(dtype))
        }
        func lin(_ k: String) throws -> Lin {
            let l = try store.linear(k, dtype: dtype)
            if l.isQuantized { precondition(l.bits == c.quantBits, "unexpected bits for \(k)") }
            return l
        }
        let ew = try store.get(p + "embed_tokens.weight")
        if let s = store.optional(p + "embed_tokens.scales") {
            let inDim = ew.dim(1) * 32 / c.quantBits
            embed = Embedding(weight: ew, scales: s, biases: store.optional(p + "embed_tokens.biases"),
                              bits: c.quantBits, groupSize: inDim / s.dim(1))
        } else {
            embed = Embedding(weight: ew.asType(dtype), scales: nil, biases: nil, bits: 0, groupSize: 0)
        }
        layers = try (0..<c.numLayers).map { i in
            let l = "\(p)layers.\(i)."
            return Layer(
                inputNorm: try norm(l + "input_layernorm.weight"),
                postAttnNorm: try norm(l + "post_attention_layernorm.weight"),
                preFFNorm: try norm(l + "pre_feedforward_layernorm.weight"),
                postFFNorm: try norm(l + "post_feedforward_layernorm.weight"),
                q: try lin(l + "self_attn.q_proj"), k: try lin(l + "self_attn.k_proj"),
                v: try lin(l + "self_attn.v_proj"), o: try lin(l + "self_attn.o_proj"),
                qNorm: try norm(l + "self_attn.q_norm.weight"), kNorm: try norm(l + "self_attn.k_norm.weight"),
                gate: try lin(l + "mlp.gate_proj"), up: try lin(l + "mlp.up_proj"), down: try lin(l + "mlp.down_proj"))
        }
        // Materialize the derived `1 + w` norm weights now so they don't keep the raw arrays alive.
        eval(layers.flatMap { [$0.inputNorm, $0.postAttnNorm, $0.preFFNorm, $0.postFFNorm, $0.qNorm, $0.kNorm] })
    }

    /// Hidden states of every layer for `tokens` (`[1, L]` int32 ids): `numLayers + 1` arrays `[1, L, hidden]`
    /// (scaled embeddings first, then each decoder layer's output; the final norm is not applied).
    ///
    /// - Parameters:
    ///   - positionOffset: RoPE position of the first token (tokens are at consecutive positions from there).
    ///   - mask: additive attention mask broadcastable to `[1, heads, L, L]`; the same mask is used by every layer
    ///     (as in the LTX reference, which calls the layers directly with one causal + padding mask).
    ///   - evalEvery: materialize the graph after every `evalEvery` layers (0 = never), like the reference.
    public func allHiddenStates(tokens: MLXArray, positionOffset: Int = 0, mask: MLXArray?,
                                evalEvery: Int = 1) -> [MLXArray] {
        let c = config
        let L = tokens.dim(1)
        var h = embed(tokens)
        h = h * MLXArray(Float(c.hiddenSize).squareRoot()).asType(.bfloat16).asType(h.dtype)
        var states = [h]
        states.reserveCapacity(c.numLayers + 1)
        let scale = 1 / c.queryPreAttnScalar.squareRoot()
        for (i, layer) in layers.enumerated() {
            let sliding = c.isSliding(i)
            let base = sliding ? c.ropeLocalBase : c.ropeTheta
            let ropeScale: Float = sliding ? 1 : c.globalRopeScale

            let x = rmsNorm(h, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(x).reshaped(1, L, c.numHeads, c.headDim).transposed(0, 2, 1, 3)
            var k = layer.k(x).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            let v = layer.v(x).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps)
            k = rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps)
            q = RoPE(q, dimensions: c.headDim, traditional: false, base: base, scale: ropeScale, offset: positionOffset)
            k = RoPE(k, dimensions: c.headDim, traditional: false, base: base, scale: ropeScale, offset: positionOffset)
            let a = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
            let r = layer.o(a.transposed(0, 2, 1, 3).reshaped(1, L, -1))
            let hh = h + rmsNorm(r, weight: layer.postAttnNorm, eps: c.rmsEps)
            let f = rmsNorm(hh, weight: layer.preFFNorm, eps: c.rmsEps)
            let m = layer.down(geluApprox(layer.gate(f)) * layer.up(f))
            h = hh + rmsNorm(m, weight: layer.postFFNorm, eps: c.rmsEps)
            states.append(h)
            if evalEvery > 0 && (i + 1) % evalEvery == 0 { eval(h) }
        }
        return states
    }
}

extension LTXGemma3Model {
    /// Debugging: runs decoder layer `i` on `h` `[1, L, hidden]` and returns its intermediates by name
    /// (`xn`, `attn`, `h`, `f`, `gate`, `up`, `gelu`, `mlp`, `out`).
    public func debugLayer(_ i: Int, _ h: MLXArray, positionOffset: Int = 0, mask: MLXArray?) -> [String: MLXArray] {
        let c = config, layer = layers[i], L = h.dim(1)
        let sliding = c.isSliding(i)
        let base = sliding ? c.ropeLocalBase : c.ropeTheta
        let ropeScale: Float = sliding ? 1 : c.globalRopeScale
        let x = rmsNorm(h, weight: layer.inputNorm, eps: c.rmsEps)
        var q = layer.q(x).reshaped(1, L, c.numHeads, c.headDim).transposed(0, 2, 1, 3)
        var k = layer.k(x).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
        let v = layer.v(x).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
        q = RoPE(rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps), dimensions: c.headDim, traditional: false,
                 base: base, scale: ropeScale, offset: positionOffset)
        k = RoPE(rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps), dimensions: c.headDim, traditional: false,
                 base: base, scale: ropeScale, offset: positionOffset)
        let a = scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / c.queryPreAttnScalar.squareRoot(),
                                          mask: mask)
        let r = layer.o(a.transposed(0, 2, 1, 3).reshaped(1, L, -1))
        let hh = h + rmsNorm(r, weight: layer.postAttnNorm, eps: c.rmsEps)
        let f = rmsNorm(hh, weight: layer.preFFNorm, eps: c.rmsEps)
        let g = layer.gate(f), u = layer.up(f), ge = geluApprox(g)
        let m = layer.down(ge * u)
        let out = hh + rmsNorm(m, weight: layer.postFFNorm, eps: c.rmsEps)
        return ["xn": x, "attn": r, "h": hh, "f": f, "gate": g, "up": u, "gelu": ge, "mlp": m, "out": out]
    }
}

/// `mlx.nn.gelu_approx`: `0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))`, compiled shapeless like the original.
let geluApprox: @Sendable (MLXArray) -> MLXArray = compile(shapeless: true) { x in
    let c = Float((2 / Double.pi).squareRoot())
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * pow(x, 3))))
}

/// The LTX reference's additive Gemma mask (bfloat16): `-1e9` above the diagonal plus `-1e9` on padded keys.
/// `valid`: 1 for real tokens, 0 for padding (length `L`). Returns `[1, 1, L, L]`.
public func ltxGemmaMask(valid: [Int32]) -> MLXArray {
    let L = valid.count
    let idx = MLXArray(Array(Int32(0) ..< Int32(L)))
    let future = idx.expandedDimensions(axis: 0) .> idx.expandedDimensions(axis: 1)  // [L, L], j > i
    let causal = MLX.where(future, MLXArray(Float(-1e9)), MLXArray(Float(0))).asType(.bfloat16)
    let m = MLXArray(valid).asType(.bfloat16).reshaped(1, 1, 1, L)
    let pad = (1 - m) * MLXArray(Float(-1e9)).asType(.bfloat16)
    return causal.reshaped(1, 1, L, L) + pad
}
