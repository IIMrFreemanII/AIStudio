import Foundation
import MLX
import MLXDiffusionCore

/// The language model of `Qwen3VLForConditionalGeneration`, used as a feature extractor.
///
/// Text-only prompts give every token the same position on the three mRoPE axes, so interleaved mRoPE reduces to
/// ordinary (non-traditional) RoPE. The vision tower is not needed for text-to-image and is not loaded yet.
public final class Qwen3TextEncoder {
    public struct Config {
        public var hiddenSize = 4096
        public var numLayers = 36
        public var numHeads = 32
        public var numKVHeads = 8
        public var headDim = 128
        public var rmsEps: Float = 1e-6
        public var ropeTheta: Float = 5_000_000

        public init(url: URL) throws {
            let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            let t = obj["text_config"] as? [String: Any] ?? obj
            hiddenSize = t["hidden_size"] as? Int ?? hiddenSize
            numLayers = t["num_hidden_layers"] as? Int ?? numLayers
            numHeads = t["num_attention_heads"] as? Int ?? numHeads
            numKVHeads = t["num_key_value_heads"] as? Int ?? numKVHeads
            headDim = t["head_dim"] as? Int ?? headDim
            rmsEps = Float(t["rms_norm_eps"] as? Double ?? Double(rmsEps))
            ropeTheta = Float(t["rope_theta"] as? Double ?? Double(ropeTheta))
        }
    }

    struct Layer {
        let inputNorm: MLXArray
        let postNorm: MLXArray
        let q, k, v, o: Lin
        let qNorm, kNorm: MLXArray
        let gate, up, down: Lin
    }

    public let config: Config
    let embed: MLXArray
    let layers: [Layer]

    public init(directory: URL, dtype: DType = .bfloat16) throws {
        config = try Config(url: directory.appendingPathComponent("config.json"))
        let store = try WeightStore(directory: directory)
        // Free the vision tower and LM head right away; only the decoder stack is used.
        store.remove(prefix: "model.visual.")
        store.remove(prefix: "lm_head.")
        let p = "model.language_model."
        embed = try store.get(p + "embed_tokens.weight").asType(dtype)
        layers = try (0..<config.numLayers).map { i in
            let l = "\(p)layers.\(i)."
            return Layer(
                inputNorm: try store.get(l + "input_layernorm.weight").asType(dtype),
                postNorm: try store.get(l + "post_attention_layernorm.weight").asType(dtype),
                q: try store.linear(l + "self_attn.q_proj", dtype: dtype),
                k: try store.linear(l + "self_attn.k_proj", dtype: dtype),
                v: try store.linear(l + "self_attn.v_proj", dtype: dtype),
                o: try store.linear(l + "self_attn.o_proj", dtype: dtype),
                qNorm: try store.get(l + "self_attn.q_norm.weight").asType(dtype),
                kNorm: try store.get(l + "self_attn.k_norm.weight").asType(dtype),
                gate: try store.linear(l + "mlp.gate_proj", dtype: dtype),
                up: try store.linear(l + "mlp.up_proj", dtype: dtype),
                down: try store.linear(l + "mlp.down_proj", dtype: dtype))
        }
    }

    /// Returns the output of the last decoder layer *before* the final RMSNorm, `[1, L, hidden]`.
    /// This is the feature the Qwen-Image 2.1 transformer was trained on.
    public func hiddenStates(tokens: [Int]) -> MLXArray {
        let c = config
        let L = tokens.count
        var x = embed[MLXArray(tokens.map { Int32($0) })].expandedDimensions(axis: 0)
        let mask: MLXArray? = L > 1 ? causalMask(L, dtype: x.dtype) : nil
        let scale = 1 / sqrt(Float(c.headDim))
        for layer in layers {
            let h = rmsNorm(x, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(h).reshaped(1, L, c.numHeads, c.headDim)
            var k = layer.k(h).reshaped(1, L, c.numKVHeads, c.headDim)
            let v = layer.v(h).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            k = rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            q = RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            k = RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            let a = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
            x = x + layer.o(a.transposed(0, 2, 1, 3).reshaped(1, L, -1))
            let h2 = rmsNorm(x, weight: layer.postNorm, eps: c.rmsEps)
            x = x + layer.down(siluAct(layer.gate(h2)) * layer.up(h2))
        }
        return x
    }
}

/// Additive causal mask `[L, L]` (0 on and below the diagonal, -inf above).
func causalMask(_ n: Int, dtype: DType) -> MLXArray {
    let idx = MLXArray(Array(Int32(0) ..< Int32(n)))
    let allowed = idx.expandedDimensions(axis: 1) .>= idx.expandedDimensions(axis: 0)
    return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
}
