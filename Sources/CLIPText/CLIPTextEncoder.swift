import Foundation
import MLX
import MLXDiffusionCore

/// The CLIP text transformer (Hugging Face `CLIPTextModel`, e.g. CLIP ViT-L/14's text tower): learned absolute
/// positions, pre-norm causal self-attention, quick-GELU MLP, final LayerNorm. Computed in float32.
public final class CLIPTextEncoder {
    public struct Config: Sendable {
        public var hiddenSize = 768, intermediateSize = 3072, numLayers = 12, numHeads = 12, maxPositions = 77
        public var eps: Float = 1e-5
        public var quickGELU = true

        public init() {}

        /// Reads a transformers `config.json` (`CLIPTextConfig`).
        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            if let v = o["hidden_size"] as? Int { hiddenSize = v }
            if let v = o["intermediate_size"] as? Int { intermediateSize = v }
            if let v = o["num_hidden_layers"] as? Int { numLayers = v }
            if let v = o["num_attention_heads"] as? Int { numHeads = v }
            if let v = o["max_position_embeddings"] as? Int { maxPositions = v }
            if let v = o["layer_norm_eps"] as? Double { eps = Float(v) }
            quickGELU = (o["hidden_act"] as? String ?? "quick_gelu") == "quick_gelu"
        }
    }

    struct Layer {
        let ln1w, ln1b, ln2w, ln2b: MLXArray
        let q, k, v, o, fc1, fc2: Lin
    }

    public let config: Config
    let tokenEmbedding, positionEmbedding: MLXArray
    let layers: [Layer]
    let finalW, finalB: MLXArray

    /// `store` holds `text_model.…` keys (the `text_model.` prefix is optional).
    public init(store: WeightStore, config: Config = Config()) throws {
        self.config = config
        let p = store.has("text_model.embeddings.token_embedding.weight") ? "text_model." : ""
        func f32(_ k: String) throws -> MLXArray { try store.get(p + k).asType(.float32) }
        func lin(_ k: String) throws -> Lin { try store.linear(p + k, dtype: .float32) }
        tokenEmbedding = try f32("embeddings.token_embedding.weight")
        positionEmbedding = try f32("embeddings.position_embedding.weight")
        layers = try (0..<config.numLayers).map { i in
            let l = "encoder.layers.\(i)."
            return Layer(ln1w: try f32(l + "layer_norm1.weight"), ln1b: try f32(l + "layer_norm1.bias"),
                         ln2w: try f32(l + "layer_norm2.weight"), ln2b: try f32(l + "layer_norm2.bias"),
                         q: try lin(l + "self_attn.q_proj"), k: try lin(l + "self_attn.k_proj"),
                         v: try lin(l + "self_attn.v_proj"), o: try lin(l + "self_attn.out_proj"),
                         fc1: try lin(l + "mlp.fc1"), fc2: try lin(l + "mlp.fc2"))
        }
        finalW = try f32("final_layer_norm.weight")
        finalB = try f32("final_layer_norm.bias")
    }

    /// Final-norm hidden states `[1, L, hidden]` for `ids` (at most `maxPositions` tokens).
    public func hiddenStates(_ ids: [Int]) -> MLXArray {
        let c = config
        let L = ids.count, H = c.numHeads, D = c.hiddenSize / c.numHeads
        var x = tokenEmbedding[MLXArray(ids.map { Int32($0) })] + positionEmbedding[0 ..< L]
        x = x.expandedDimensions(axis: 0)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = L > 1 ? .causal : .none
        for l in layers {
            let h = MLXFast.layerNorm(x, weight: l.ln1w, bias: l.ln1b, eps: c.eps)
            func heads(_ t: MLXArray) -> MLXArray { t.reshaped(1, L, H, D).transposed(0, 2, 1, 3) }
            let a = MLXFast.scaledDotProductAttention(queries: heads(l.q(h)), keys: heads(l.k(h)), values: heads(l.v(h)),
                                                      scale: 1 / Float(D).squareRoot(), mask: mask)
            x = x + l.o(a.transposed(0, 2, 1, 3).reshaped(1, L, H * D))
            let h2 = MLXFast.layerNorm(x, weight: l.ln2w, bias: l.ln2b, eps: c.eps)
            let m = l.fc1(h2)
            x = x + l.fc2(c.quickGELU ? m * sigmoid(1.702 * m) : 0.5 * m * (1 + erf(m / Float(2).squareRoot())))
        }
        return MLXFast.layerNorm(x, weight: finalW, bias: finalB, eps: c.eps)
    }

    /// The pooled output `[1, hidden]`: the final hidden state at the first end-of-text token (`eos`).
    public func pooled(_ ids: [Int], eos: Int) -> MLXArray {
        let h = hiddenStates(ids)
        let i = ids.firstIndex(of: eos) ?? ids.count - 1
        return h[0..., i, 0...]
    }
}
