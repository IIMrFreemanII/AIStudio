import Foundation
import MLX
import MLXDiffusionCore

/// Turns 5 Hz planner audio codes into the DiT's 25 Hz "LM hints" `[1, 5·N, 64]`.
///
/// Port of `AceStepConditionGenerationModel.tokenizer.quantizer.get_output_from_indices` (a single-level
/// `ResidualFSQ`, levels [8, 8, 8, 5, 5, 5] → 64000 codes, then `project_out` 6 → 2048) followed by
/// `AudioTokenDetokenizer` (each code embedded, expanded into `pool_window_size` = 5 patches with learned special
/// tokens, two bidirectional Qwen3-style encoder layers over the 5 patches, norm, `proj_out` 2048 → 64).
/// Weights live under the `tokenizer.quantizer.` / `detokenizer.` prefixes of `acestep-v15-turbo/model.safetensors`.
public final class ACEAudioCodeDecoder {
    public struct Config {
        public var hiddenSize = 2048, numHeads = 16, numKVHeads = 8, headDim = 128
        public var poolWindow = 5, acousticDim = 64
        public var rmsEps: Float = 1e-6
        public var ropeTheta: Float = 1_000_000
        public var levels = [8, 8, 8, 5, 5, 5]

        public init() {}

        /// Reads the fields it needs from the DiT checkpoint's `config.json`.
        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            hiddenSize = o["hidden_size"] as? Int ?? hiddenSize
            numHeads = o["num_attention_heads"] as? Int ?? numHeads
            numKVHeads = o["num_key_value_heads"] as? Int ?? numKVHeads
            headDim = o["head_dim"] as? Int ?? headDim
            poolWindow = o["pool_window_size"] as? Int ?? poolWindow
            acousticDim = o["audio_acoustic_hidden_dim"] as? Int ?? acousticDim
            rmsEps = Float(o["rms_norm_eps"] as? Double ?? Double(rmsEps))
            ropeTheta = Float(o["rope_theta"] as? Double ?? Double(ropeTheta))
            levels = o["fsq_input_levels"] as? [Int] ?? levels
        }

        /// Number of distinct codes (product of the FSQ levels): 64000.
        public var codebookSize: Int { levels.reduce(1, *) }
    }

    struct Layer {
        let inputNorm, postNorm, qNorm, kNorm: MLXArray
        let q, k, v, o, gate, up, down: Lin
    }

    public let config: Config
    public let dtype: DType
    let projectOut: Lin
    let embedTokens: Lin
    let specialTokens: MLXArray
    let layers: [Layer]
    let norm: MLXArray
    let projOut: Lin

    /// `store`: the turbo DiT checkpoint (`acestep-v15-turbo/model.safetensors`). Compute runs in `dtype`
    /// (float32 matches the reference to ~1e-6; the weights themselves are stored as bf16).
    public init(store: WeightStore, config: Config = Config(), dtype: DType = .float32) throws {
        self.config = config
        self.dtype = dtype
        projectOut = try store.linear("tokenizer.quantizer.project_out", dtype: dtype)
        let p = "detokenizer."
        embedTokens = try store.linear(p + "embed_tokens", dtype: dtype)
        specialTokens = try store.get(p + "special_tokens").asType(dtype)
        var ls: [Layer] = []
        var i = 0
        while store.has("\(p)layers.\(i).input_layernorm.weight") {
            let l = "\(p)layers.\(i)."
            ls.append(Layer(
                inputNorm: try store.get(l + "input_layernorm.weight").asType(dtype),
                postNorm: try store.get(l + "post_attention_layernorm.weight").asType(dtype),
                qNorm: try store.get(l + "self_attn.q_norm.weight").asType(dtype),
                kNorm: try store.get(l + "self_attn.k_norm.weight").asType(dtype),
                q: try store.linear(l + "self_attn.q_proj", dtype: dtype),
                k: try store.linear(l + "self_attn.k_proj", dtype: dtype),
                v: try store.linear(l + "self_attn.v_proj", dtype: dtype),
                o: try store.linear(l + "self_attn.o_proj", dtype: dtype),
                gate: try store.linear(l + "mlp.gate_proj", dtype: dtype),
                up: try store.linear(l + "mlp.up_proj", dtype: dtype),
                down: try store.linear(l + "mlp.down_proj", dtype: dtype)))
            i += 1
        }
        guard !ls.isEmpty else { throw WeightError.missing(p + "layers.0.input_layernorm.weight") }
        layers = ls
        norm = try store.get(p + "norm.weight").asType(dtype)
        projOut = try store.linear(p + "proj_out", dtype: dtype)
    }

    /// FSQ codes `[N, 6]` in [-1, 1] for code indices (`ResidualFSQ.get_codes_from_indices`, one quantizer, scale 1).
    public func fsqCodes(_ codes: [Int]) -> MLXArray {
        let levels = config.levels
        var basis: [Int] = []
        var b = 1
        for l in levels { basis.append(b); b *= l }
        var out = [Float](repeating: 0, count: codes.count * levels.count)
        for (n, c) in codes.enumerated() {
            let c = max(0, min(c, config.codebookSize - 1))
            for (j, l) in levels.enumerated() {
                let lvl = (c / basis[j]) % l
                out[n * levels.count + j] = Float(lvl) * (2 / Float(l - 1)) - 1
            }
        }
        return MLXArray(out, [codes.count, levels.count])
    }

    /// 25 Hz hints `[1, 5·N, 64]` (float32) for `codes` (values 0..<64000, clamped like the reference).
    public func hints(codes: [Int]) -> MLXArray {
        let c = config
        let n = codes.count
        let P = c.poolWindow
        precondition(n > 0, "no audio codes")
        var x = projectOut(fsqCodes(codes).asType(dtype))  // [N, hidden]
        x = embedTokens(x).reshaped(n, 1, c.hiddenSize) + specialTokens.reshaped(1, P, c.hiddenSize)  // [N, P, D]
        let scale = 1 / Float(c.headDim).squareRoot()
        for layer in layers {
            let h = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(h).reshaped(n, P, c.numHeads, c.headDim)
            var k = layer.k(h).reshaped(n, P, c.numKVHeads, c.headDim)
            let v = layer.v(h).reshaped(n, P, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = MLXFast.rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            k = MLXFast.rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            x = x + layer.o(a.transposed(0, 2, 1, 3).reshaped(n, P, -1))
            let h2 = MLXFast.rmsNorm(x, weight: layer.postNorm, eps: c.rmsEps)
            x = x + layer.down(siluAct(layer.gate(h2)) * layer.up(h2))
        }
        x = projOut(MLXFast.rmsNorm(x, weight: norm, eps: c.rmsEps))  // [N, P, 64]
        return x.reshaped(1, n * P, c.acousticDim).asType(.float32)
    }
}
