import Foundation
import MLX
import MLXDiffusionCore

/// The T5 v1.1 / umT5 encoder stack (XXL size by default): RMS norms, unscaled attention with a bucketed relative
/// position bias, and a gated-GELU feed-forward.
///
/// umT5 (Wan) gives every layer its own relative-position bias table; T5 v1.1 (Flux) has one table in layer 0 that
/// all layers share. Activations are float32 (bf16 weights are used as stored, promoted in each matmul) — the
/// precision of the mlx-video reference, a little above the official bf16 run.
public final class T5XXLEncoder {
    public struct Config: Sendable {
        public var vocabSize = 256_384, dim = 4096, dimAttn = 4096, dimFFN = 10_240
        public var numHeads = 64, numLayers = 24, numBuckets = 32, maxDistance = 128
        /// One bias table shared by all layers (T5 v1.1) instead of one per layer (umT5).
        public var sharedPositionBias = false
        public var eps: Float = 1e-6
        public init() {}
    }

    /// Checkpoint key names.
    public enum Layout: Sendable {
        /// The official Wan / mlx-video layout: `token_embedding`, `blocks.N.{attn,ffn,norm1,norm2,pos_embedding}`, `norm`.
        case wan
        /// Hugging Face `T5EncoderModel`: `shared` / `encoder.block.N.layer.{0,1}…`, `encoder.final_layer_norm`.
        case huggingFace
    }

    struct Block {
        let norm1, norm2: MLXArray
        let q, k, v, o: Lin
        let gate, fc1, fc2: Lin
        let positionBias: MLXArray?
    }

    public let config: Config
    let embedding: MLXArray
    let blocks: [Block]
    let finalNorm: MLXArray
    let sharedBias: MLXArray?

    public init(store: WeightStore, config: Config = Config(), layout: Layout = .wan) throws {
        self.config = config
        func w(_ k: String) throws -> MLXArray { try store.get(k) }
        func lin(_ k: String) throws -> Lin { try store.linear(k) }
        switch layout {
        case .wan:
            embedding = try w("token_embedding.weight")
            blocks = try (0..<config.numLayers).map { i in
                let p = "blocks.\(i)."
                let gateKey = store.has(p + "ffn.gate_proj.weight") ? p + "ffn.gate_proj" : p + "ffn.gate.0"
                return Block(norm1: try w(p + "norm1.weight"), norm2: try w(p + "norm2.weight"),
                             q: try lin(p + "attn.q"), k: try lin(p + "attn.k"), v: try lin(p + "attn.v"),
                             o: try lin(p + "attn.o"), gate: try lin(gateKey), fc1: try lin(p + "ffn.fc1"),
                             fc2: try lin(p + "ffn.fc2"),
                             positionBias: config.sharedPositionBias ? nil
                                 : try store.get(p + "pos_embedding.embedding.weight").asType(.float32))
            }
            finalNorm = try w("norm.weight")
            sharedBias = config.sharedPositionBias ? try store.get("pos_embedding.embedding.weight").asType(.float32) : nil
        case .huggingFace:
            embedding = try w(store.has("shared.weight") ? "shared.weight" : "encoder.embed_tokens.weight")
            blocks = try (0..<config.numLayers).map { i in
                let p = "encoder.block.\(i).layer."
                let bias = store.optional(p + "0.SelfAttention.relative_attention_bias.weight")?.asType(.float32)
                return Block(norm1: try w(p + "0.layer_norm.weight"), norm2: try w(p + "1.layer_norm.weight"),
                             q: try lin(p + "0.SelfAttention.q"), k: try lin(p + "0.SelfAttention.k"),
                             v: try lin(p + "0.SelfAttention.v"), o: try lin(p + "0.SelfAttention.o"),
                             gate: try lin(p + "1.DenseReluDense.wi_0"), fc1: try lin(p + "1.DenseReluDense.wi_1"),
                             fc2: try lin(p + "1.DenseReluDense.wo"),
                             positionBias: config.sharedPositionBias ? nil : bias)
            }
            finalNorm = try w("encoder.final_layer_norm.weight")
            sharedBias = config.sharedPositionBias
                ? try store.get("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight").asType(.float32)
                : nil
        }
    }

    /// Bidirectional relative-position buckets `[L, L]` (T5's `_relative_position_bucket`).
    static func buckets(_ n: Int, numBuckets: Int, maxDistance: Int) -> MLXArray {
        let half = numBuckets / 2, maxExact = half / 2
        var out = [Int32](repeating: 0, count: n * n)
        for qi in 0..<n {
            for ki in 0..<n {
                let rel = ki - qi
                var bucket = rel > 0 ? half : 0
                let a = abs(rel)
                if a < maxExact {
                    bucket += a
                } else {
                    // float32 like the reference: log(a / maxExact) / log(maxDistance / maxExact) * (half - maxExact)
                    let f = Float(log(Float(a) / Float(maxExact)) / Float(log(Double(maxDistance) / Double(maxExact)))
                        * Float(half - maxExact))
                    bucket += min(maxExact + Int(f), half - 1)
                }
                out[qi * n + ki] = Int32(bucket)
            }
        }
        return MLXArray(out, [n, n])
    }

    /// Final hidden states `[B, L, dim]` for token ids `[B, L]` (no padding: every token attends to every token).
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let c = config
        let B = ids.dim(0), L = ids.dim(1), H = c.numHeads, D = c.dimAttn / c.numHeads
        let bucketIdx = Self.buckets(L, numBuckets: c.numBuckets, maxDistance: c.maxDistance)
        func bias(_ table: MLXArray) -> MLXArray { table[bucketIdx].transposed(2, 0, 1).expandedDimensions(axis: 0) }
        let shared = sharedBias.map(bias)
        var x = embedding[ids].asType(.float32)
        for b in blocks {
            let h = MLXFast.rmsNorm(x, weight: b.norm1.asType(.float32), eps: c.eps)
            let q = b.q(h).reshaped(B, L, H, D).transposed(0, 2, 1, 3).asType(.float32)
            let k = b.k(h).reshaped(B, L, H, D).transposed(0, 2, 1, 3).asType(.float32)
            let v = b.v(h).reshaped(B, L, H, D).transposed(0, 2, 1, 3)
            var logits = matmul(q, k.transposed(0, 1, 3, 2))
            if let pb = shared ?? b.positionBias.map(bias) { logits = logits + pb }
            let attn = softmax(logits, axis: -1).asType(v.dtype)
            x = x + b.o(matmul(attn, v).transposed(0, 2, 1, 3).reshaped(B, L, H * D))
            let h2 = MLXFast.rmsNorm(x, weight: b.norm2.asType(.float32), eps: c.eps)
            x = x + b.fc2(b.fc1(h2) * geluTanh(b.gate(h2)))
        }
        return MLXFast.rmsNorm(x, weight: finalNorm.asType(.float32), eps: c.eps)
    }
}
