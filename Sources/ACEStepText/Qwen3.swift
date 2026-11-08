import Foundation
import MLX
import MLXDiffusionCore

/// A plain Qwen3 decoder (`Qwen3Model` / `Qwen3ForCausalLM`): RMSNorm, GQA with per-head q/k RMSNorm, rotate-half RoPE,
/// SwiGLU MLP, causal attention. ACE-Step uses it twice: Qwen3-Embedding-0.6B as the caption encoder (final hidden
/// states) and the 5 Hz planner LM (tied LM head, KV-cached generation).
public final class Qwen3Model {
    public struct Config {
        public var hiddenSize = 1024, numLayers = 28, numHeads = 16, numKVHeads = 8, headDim = 128
        public var vocabSize = 151_669
        public var rmsEps: Float = 1e-6
        public var ropeTheta: Float = 1_000_000
        public var tieWordEmbeddings = true

        public init() {}

        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            hiddenSize = o["hidden_size"] as? Int ?? hiddenSize
            numLayers = o["num_hidden_layers"] as? Int ?? numLayers
            numHeads = o["num_attention_heads"] as? Int ?? numHeads
            numKVHeads = o["num_key_value_heads"] as? Int ?? numKVHeads
            headDim = o["head_dim"] as? Int ?? headDim
            vocabSize = o["vocab_size"] as? Int ?? vocabSize
            rmsEps = Float(o["rms_norm_eps"] as? Double ?? Double(rmsEps))
            ropeTheta = Float(o["rope_theta"] as? Double ?? Double(ropeTheta))
            tieWordEmbeddings = o["tie_word_embeddings"] as? Bool ?? tieWordEmbeddings
        }
    }

    struct Layer {
        let inputNorm, postNorm: MLXArray
        let q, k, v, o: Lin
        let qNorm, kNorm: MLXArray
        let gate, up, down: Lin
    }

    /// Keys and values of every layer for the tokens seen so far. Batched: `[B, kvHeads, T, headDim]`.
    ///
    /// Storage is preallocated in chunks of `step` positions and written in place, so long generations don't pay
    /// for re-concatenating the whole cache every token.
    public final class KVCache {
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        /// Tokens held in the cache.
        public internal(set) var length = 0
        /// Growth granularity (positions).
        public var step = 256
        public init() {}

        /// Appends `k`/`v` `[B, H, L, D]` for layer `i` at `length` and returns the cached keys/values up to it.
        func update(layer i: Int, keys k: MLXArray, values v: MLXArray) -> (MLXArray, MLXArray) {
            let start = length, L = k.dim(2), end = start + L
            if i >= keys.count || end > keys[i].dim(2) {
                let cap = ((end + step - 1) / step) * step
                func grow(_ old: MLXArray?, _ new: MLXArray) -> MLXArray {
                    let kept = old == nil ? 0 : start
                    let z = MLXArray.zeros([new.dim(0), new.dim(1), cap - kept, new.dim(3)], dtype: new.dtype)
                    guard let old else { return z }
                    return concatenated([old[0..., 0..., ..<kept, 0...], z], axis: 2)
                }
                if i >= keys.count {
                    keys.append(grow(nil, k))
                    values.append(grow(nil, v))
                } else {
                    keys[i] = grow(keys[i], k)
                    values[i] = grow(values[i], v)
                }
            }
            keys[i][0..., 0..., start ..< end, 0...] = k
            values[i][0..., 0..., start ..< end, 0...] = v
            return (keys[i][0..., 0..., ..<end, 0...], values[i][0..., 0..., ..<end, 0...])
        }
    }

    public let config: Config
    public let dtype: DType
    /// Token embeddings `[vocab, hidden]`.
    public let embedTokens: MLXArray
    let layers: [Layer]
    let finalNorm: MLXArray
    let lmHead: Lin?

    /// `directory`: a Hugging Face Qwen3 folder (`config.json`, `model*.safetensors`). Keys may or may not carry the
    /// `model.` prefix (Qwen3Model vs Qwen3ForCausalLM checkpoints).
    public init(directory: URL, dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        config = try Config(url: directory.appendingPathComponent("config.json"))
        let s = try WeightStore(directory: directory)
        let p = s.has("model.embed_tokens.weight") ? "model." : ""
        embedTokens = try s.get(p + "embed_tokens.weight").asType(dtype)
        layers = try (0..<config.numLayers).map { i in
            let l = "\(p)layers.\(i)."
            return Layer(
                inputNorm: try s.get(l + "input_layernorm.weight").asType(dtype),
                postNorm: try s.get(l + "post_attention_layernorm.weight").asType(dtype),
                q: try s.linear(l + "self_attn.q_proj", dtype: dtype),
                k: try s.linear(l + "self_attn.k_proj", dtype: dtype),
                v: try s.linear(l + "self_attn.v_proj", dtype: dtype),
                o: try s.linear(l + "self_attn.o_proj", dtype: dtype),
                qNorm: try s.get(l + "self_attn.q_norm.weight").asType(dtype),
                kNorm: try s.get(l + "self_attn.k_norm.weight").asType(dtype),
                gate: try s.linear(l + "mlp.gate_proj", dtype: dtype),
                up: try s.linear(l + "mlp.up_proj", dtype: dtype),
                down: try s.linear(l + "mlp.down_proj", dtype: dtype))
        }
        finalNorm = try s.get(p + "norm.weight").asType(dtype)
        lmHead = s.has("lm_head.weight") && !config.tieWordEmbeddings ? try s.linear("lm_head", dtype: dtype) : nil
    }

    /// Token embeddings `[B, L, hidden]` for `ids` `[B, L]`.
    public func embed(_ ids: MLXArray) -> MLXArray { embedTokens[ids] }

    /// The decoder stack plus the final norm: `last_hidden_state` `[B, L, hidden]`.
    ///
    /// `mask`: optional additive mask broadcastable to `[B, heads, L, T]` (T = cached + L). Without one, a causal mask
    /// is used when `L > 1`. With a `cache`, keys/values are appended and positions continue from `cache.length`
    /// (or from `positionOffset` when given, for left-padded batches).
    public func callAsFunction(_ ids: MLXArray, cache: KVCache? = nil, mask: MLXArray? = nil) -> MLXArray {
        forward(embed(ids), cache: cache, mask: mask)
    }

    public func forward(_ input: MLXArray, cache: KVCache? = nil, mask: MLXArray? = nil) -> MLXArray {
        let c = config
        let B = input.dim(0), L = input.dim(1)
        let offset = cache?.length ?? 0
        let scale = 1 / Float(c.headDim).squareRoot()
        var attnMask: MLXFast.ScaledDotProductAttentionMaskMode = .none
        if let mask {
            attnMask = .array(mask.asType(dtype))
        } else if L > 1 {
            attnMask = offset == 0 ? .causal : .array(Self.causalMask(L, offset: offset, dtype: dtype))
        }
        var x = input.asType(dtype)
        for (i, layer) in layers.enumerated() {
            let h = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(h).reshaped(B, L, c.numHeads, c.headDim)
            var k = layer.k(h).reshaped(B, L, c.numKVHeads, c.headDim)
            var v = layer.v(h).reshaped(B, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = MLXFast.rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            k = MLXFast.rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
            k = MLXFast.RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
            if let cache { (k, v) = cache.update(layer: i, keys: k, values: v) }
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: attnMask)
            x = x + layer.o(a.transposed(0, 2, 1, 3).reshaped(B, L, -1))
            let h2 = MLXFast.rmsNorm(x, weight: layer.postNorm, eps: c.rmsEps)
            x = x + layer.down(siluAct(layer.gate(h2)) * layer.up(h2))
        }
        if let cache { cache.length += L }
        return MLXFast.rmsNorm(x, weight: finalNorm, eps: c.rmsEps)
    }

    /// Next-token logits `[B, L, vocab]` (float32) from final hidden states.
    public func logits(_ hidden: MLXArray) -> MLXArray {
        if let lmHead { return lmHead(hidden).asType(.float32) }
        return matmul(hidden, embedTokens.T).asType(.float32)
    }

    /// Additive causal mask `[L, offset + L]` for `L` new tokens after `offset` cached ones.
    public static func causalMask(_ n: Int, offset: Int, dtype: DType) -> MLXArray {
        let rows = MLXArray(Array(Int32(offset) ..< Int32(offset + n))).expandedDimensions(axis: 1)
        let cols = MLXArray(Array(Int32(0) ..< Int32(offset + n))).expandedDimensions(axis: 0)
        return MLX.where(cols .<= rows, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
    }
}
