import Foundation
import MLX
import MLXDiffusionCore

/// The T5 encoder (`T5EncoderModel`, t5-base) that turns the prompt into cross-attention tokens.
///
/// Pre-norm blocks with T5's scale-free RMS norm, unscaled attention with a learned relative-position bias (computed
/// in block 0 and shared), and a ReLU feed-forward. Runs in float32.
public final class T5Encoder {
    public struct Config {
        public var dModel = 768, numLayers = 12, numHeads = 12, dKV = 64
        public var buckets = 32, maxDistance = 128
        public var eps: Float = 1e-6

        public init(url: URL?) {
            guard let url, let data = try? Data(contentsOf: url),
                  let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            dModel = o["d_model"] as? Int ?? dModel
            numLayers = o["num_layers"] as? Int ?? numLayers
            numHeads = o["num_heads"] as? Int ?? numHeads
            dKV = o["d_kv"] as? Int ?? dKV
            buckets = o["relative_attention_num_buckets"] as? Int ?? buckets
            maxDistance = o["relative_attention_max_distance"] as? Int ?? maxDistance
            eps = Float(o["layer_norm_epsilon"] as? Double ?? Double(eps))
        }
    }

    struct Block {
        let attnNorm: MLXArray
        let q, k, v, o: Lin
        let ffNorm: MLXArray
        let wi, wo: Lin
    }

    public let config: Config
    let embed: MLXArray
    let relativeBias: MLXArray  // [buckets, heads]
    let blocks: [Block]
    let finalNorm: MLXArray

    /// `directory`: a t5-base folder (`config.json`, `model.safetensors`); decoder weights are ignored.
    public init(directory: URL) throws {
        config = Config(url: directory.appendingPathComponent("config.json"))
        let s = try WeightStore(url: directory.appendingPathComponent("model.safetensors"))
        s.remove(prefix: "decoder.")
        func f32(_ k: String) throws -> MLXArray { try s.get(k).asType(.float32) }
        embed = try f32(s.has("encoder.embed_tokens.weight") ? "encoder.embed_tokens.weight" : "shared.weight")
        relativeBias = try f32("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
        blocks = try (0..<config.numLayers).map { i in
            let p = "encoder.block.\(i).layer."
            return Block(
                attnNorm: try f32(p + "0.layer_norm.weight"),
                q: try s.linear(p + "0.SelfAttention.q", dtype: .float32),
                k: try s.linear(p + "0.SelfAttention.k", dtype: .float32),
                v: try s.linear(p + "0.SelfAttention.v", dtype: .float32),
                o: try s.linear(p + "0.SelfAttention.o", dtype: .float32),
                ffNorm: try f32(p + "1.layer_norm.weight"),
                wi: try s.linear(p + "1.DenseReluDense.wi", dtype: .float32),
                wo: try s.linear(p + "1.DenseReluDense.wo", dtype: .float32))
        }
        finalNorm = try f32("encoder.final_layer_norm.weight")
    }

    /// T5's layer norm: no mean subtraction and no bias.
    func norm(_ x: MLXArray, _ w: MLXArray) -> MLXArray { MLXFast.rmsNorm(x, weight: w, eps: config.eps) }

    /// Bidirectional relative-position buckets (HF `_relative_position_bucket`), `[L, L]` int32.
    func buckets(_ length: Int) -> MLXArray {
        let n = config.buckets / 2, maxExact = n / 2
        var out = [Int32](repeating: 0, count: length * length)
        for q in 0..<length {
            for k in 0..<length {
                let rel = k - q
                var bucket = rel > 0 ? n : 0
                let d = abs(rel)
                if d < maxExact {
                    bucket += d
                } else {
                    // float32 like torch: log(d / maxExact) / log(maxDistance / maxExact) * (n - maxExact), truncated.
                    let r = logf(Float(d) / Float(maxExact)) / Float(log(Double(config.maxDistance) / Double(maxExact)))
                    let large = maxExact + Int(r * Float(n - maxExact))
                    bucket += min(large, n - 1)
                }
                out[q * length + k] = Int32(bucket)
            }
        }
        return MLXArray(out, [length, length])
    }

    /// `ids`: `[1, L]` token ids; `mask`: `[1, L]` 1 for real tokens, 0 for padding. Returns `[1, L, d_model]` with
    /// padding rows zeroed (the conditioner's `padding_mode` "zero").
    public func callAsFunction(ids: MLXArray, mask: MLXArray) -> MLXArray {
        let L = ids.dim(1), H = config.numHeads, D = config.dKV
        var h = embed[ids]
        // Position bias [1, H, L, L] plus the padding mask on keys.
        let bias = relativeBias[buckets(L)].transposed(2, 0, 1).expandedDimensions(axis: 0)
        let keyMask = (1 - mask.asType(.float32)).reshaped(1, 1, 1, L) * Float(-1e9)
        let attnBias = bias + keyMask
        for b in blocks {
            let x = norm(h, b.attnNorm)
            func heads(_ t: MLXArray) -> MLXArray { t.reshaped(1, L, H, D).transposed(0, 2, 1, 3) }
            let a = MLXFast.scaledDotProductAttention(queries: heads(b.q(x)), keys: heads(b.k(x)), values: heads(b.v(x)),
                                                      scale: 1, mask: .array(attnBias))
            h = h + b.o(a.transposed(0, 2, 1, 3).reshaped(1, L, H * D))
            h = h + b.wo(maximum(b.wi(norm(h, b.ffNorm)), 0))
        }
        return norm(h, finalNorm) * mask.asType(.float32).expandedDimensions(axis: -1)
    }
}
