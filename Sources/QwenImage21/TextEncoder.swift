import Foundation
import MLX
import MLXDiffusionCore

/// The language model of `Qwen3VLForConditionalGeneration`, used as a feature extractor.
///
/// Text-only prompts give every token the same position on the three mRoPE axes, so interleaved mRoPE reduces to
/// ordinary (non-traditional) RoPE. Prompts with images use the vision tower, interleaved mRoPE and deepstack.
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
    public let directory: URL
    public let dtype: DType
    let embed: MLXArray
    let layers: [Layer]
    /// The vision tower, used when prompts carry condition images.
    public let vision: Qwen3VisionTower
    /// LM head for text generation, loaded on first use (TextGenerator.swift).
    var lmHeadStorage: LMHead?
    let lmHeadLock = NSLock()
    public var mropeSection = [24, 20, 20]

    public init(directory: URL, dtype: DType = .bfloat16) throws {
        self.directory = directory
        self.dtype = dtype
        config = try Config(url: directory.appendingPathComponent("config.json"))
        let store = try WeightStore(directory: directory)
        // The LM head isn't needed for features; text generation loads it lazily (see TextGenerator.swift).
        store.remove(prefix: "lm_head.")
        vision = try Qwen3VisionTower(
            store: store, config: Qwen3VisionTower.Config(textEncoderConfig: directory.appendingPathComponent("config.json")),
            dtype: dtype)
        if let obj = try? JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json")))
            as? [String: Any], let t = obj["text_config"] as? [String: Any],
            let rs = t["rope_scaling"] as? [String: Any], let sec = rs["mrope_section"] as? [Int] {
            mropeSection = sec
        }
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

    /// Visual inputs for a multimodal prompt.
    public struct VisualInputs {
        /// Contiguous runs of image-token positions in the token sequence, one per image, in order.
        public var runs: [(start: Int, count: Int)]
        /// Merged visual embeddings for all runs, concatenated `[sum(count), hidden]`.
        public var embeddings: MLXArray
        /// Deepstack features (same rows), added after the first `deepstack.count` decoder layers.
        public var deepstack: [MLXArray]
        /// mRoPE positions `(t, h, w)` per token.
        public var positions: [(Int32, Int32, Int32)]
    }

    /// Returns the output of the last decoder layer *before* the final RMSNorm, `[1, L, hidden]`.
    /// This is the feature the Qwen-Image 2.1 transformer was trained on.
    public func hiddenStates(tokens: [Int], visual: VisualInputs? = nil) -> MLXArray {
        let L = tokens.count
        // Text-only prompts have equal positions on every mRoPE axis: plain RoPE. Otherwise build interleaved mRoPE.
        let positions: RotaryPositions = visual.map { .mrope(mropeTables($0.positions)) } ?? .sequential(offset: 0)
        return forward(embeddings(for: tokens, visual: visual), positions: positions,
                       mask: L > 1 ? causalMask(L, dtype: embed.dtype) : nil, cache: nil, visual: visual)
    }

    /// Token embeddings `[1, L, hidden]` with the visual embeddings scattered into the image runs.
    func embeddings(for tokens: [Int], visual: VisualInputs?) -> MLXArray {
        var x = embed[MLXArray(tokens.map { Int32($0) })]
        if let visual {
            let rows = Self.scatterRows(visual.embeddings.asType(x.dtype), runs: visual.runs, length: tokens.count)
            x = MLX.where(Self.rowMask(tokens.count, runs: visual.runs), rows, x)
        }
        return x.expandedDimensions(axis: 0)
    }

    /// Rotary positions for a forward pass: consecutive positions from `offset` (text), or precomputed mRoPE tables.
    enum RotaryPositions {
        case sequential(offset: Int)
        case mrope((MLXArray, MLXArray))
    }

    /// Keys and values of every layer for the tokens seen so far (text generation).
    public final class KVCache {
        var keys: [MLXArray] = []
        var values: [MLXArray] = []
        /// Tokens held in the cache.
        public internal(set) var length = 0
        /// RoPE position of the next token. After an image this is ahead of `length` (mRoPE skips positions).
        public internal(set) var nextPosition = 0
        public init() {}
    }

    /// The decoder stack on `x` `[1, L, hidden]`; returns the last layer's output before the final norm.
    /// With a `cache`, the new keys/values are appended to it and attention covers the cached tokens too.
    func forward(_ input: MLXArray, positions: RotaryPositions, mask: MLXArray?, cache: KVCache?,
                 visual: VisualInputs?) -> MLXArray {
        let c = config
        let L = input.dim(1)
        var x = input
        let scale = 1 / sqrt(Float(c.headDim))
        for (i, layer) in layers.enumerated() {
            let h = rmsNorm(x, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(h).reshaped(1, L, c.numHeads, c.headDim)
            var k = layer.k(h).reshaped(1, L, c.numKVHeads, c.headDim)
            var v = layer.v(h).reshaped(1, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            k = rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            switch positions {
            case .mrope(let tables):
                q = applyRotateHalf(q, tables)
                k = applyRotateHalf(k, tables)
            case .sequential(let offset):
                q = RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
                k = RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: offset)
            }
            if let cache {
                if i < cache.keys.count {
                    k = concatenated([cache.keys[i], k], axis: 2)
                    v = concatenated([cache.values[i], v], axis: 2)
                    cache.keys[i] = k
                    cache.values[i] = v
                } else {
                    cache.keys.append(k)
                    cache.values.append(v)
                }
            }
            let a = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
            x = x + layer.o(a.transposed(0, 2, 1, 3).reshaped(1, L, -1))
            let h2 = rmsNorm(x, weight: layer.postNorm, eps: c.rmsEps)
            x = x + layer.down(siluAct(layer.gate(h2)) * layer.up(h2))
            if let visual, i < visual.deepstack.count {
                x = x + Self.scatterRows(visual.deepstack[i].asType(x.dtype), runs: visual.runs, length: L)
                    .expandedDimensions(axis: 0)
            }
        }
        if let cache { cache.length += L }
        return x
    }

    /// Interleaved mRoPE: frequency `d` uses the h position when `d % 3 == 1`, w when `d % 3 == 2` (within the
    /// first `3 * section` frequencies), t otherwise; then duplicated for the rotate-half form. `[L, headDim]`, which
    /// broadcasts over `[1, heads, L, headDim]`.
    func mropeTables(_ positions: [(Int32, Int32, Int32)]) -> (MLXArray, MLXArray) {
        let half = config.headDim / 2
        let inv = (0..<half).map { 1 / pow(config.ropeTheta, Float(2 * $0) / Float(config.headDim)) }
        var axis = [Int](repeating: 0, count: half)
        for d in 0..<half {
            if d % 3 == 1 && d < mropeSection[1] * 3 { axis[d] = 1 }
            if d % 3 == 2 && d < mropeSection[2] * 3 { axis[d] = 2 }
        }
        var angles = [Float]()
        angles.reserveCapacity(positions.count * half)
        for (t, h, w) in positions {
            for d in 0..<half {
                let p = axis[d] == 0 ? t : (axis[d] == 1 ? h : w)
                angles.append(Float(p) * inv[d])
            }
        }
        let a = MLXArray(angles, [positions.count, half])
        let full = concatenated([a, a], axis: 1)
        return (cos(full), sin(full))
    }

    /// `[L, 1]` boolean mask of the rows covered by `runs`.
    static func rowMask(_ length: Int, runs: [(start: Int, count: Int)]) -> MLXArray {
        var m = [Bool](repeating: false, count: length)
        for r in runs { for i in r.start ..< r.start + r.count { m[i] = true } }
        return MLXArray(m).reshaped(length, 1)
    }

    /// Places `values` (rows in run order) at the run positions of a zero `[length, D]` array.
    static func scatterRows(_ values: MLXArray, runs: [(start: Int, count: Int)], length: Int) -> MLXArray {
        let d = values.dim(1)
        var parts: [MLXArray] = []
        var cursor = 0, offset = 0
        for r in runs {
            if r.start > cursor { parts.append(MLXArray.zeros([r.start - cursor, d], dtype: values.dtype)) }
            parts.append(values[offset ..< offset + r.count])
            offset += r.count
            cursor = r.start + r.count
        }
        if cursor < length { parts.append(MLXArray.zeros([length - cursor, d], dtype: values.dtype)) }
        return concatenated(parts, axis: 0)
    }
}

/// Additive causal mask `[L, L]` (0 on and below the diagonal, -inf above).
func causalMask(_ n: Int, dtype: DType) -> MLXArray {
    let idx = MLXArray(Array(Int32(0) ..< Int32(n)))
    let allowed = idx.expandedDimensions(axis: 1) .>= idx.expandedDimensions(axis: 0)
    return MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
}
