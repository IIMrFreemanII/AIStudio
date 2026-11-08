import Foundation
import MLX
import MLXDiffusionCore

/// Dense `nn.Linear` (`addmm(bias, x, Wᵀ)`, i.e. the bias is fused into the matmul like MLX's Python layer).
struct LTXDense {
    let weight: MLXArray
    let bias: MLXArray?

    init(store: WeightStore, prefix: String, dtype: DType) throws {
        weight = try store.get(prefix + ".weight").asType(dtype)
        bias = store.optional(prefix + ".bias")?.asType(dtype)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if let bias { return addMM(bias, x, weight.T) }
        return matmul(x, weight.T)
    }
}

/// `apply_rope_split` (compiled, like the reference): rotates the two halves of the head dimension.
/// `cos`/`sin`: float32 `[1, H, N, D/2]`, cast to the activation dtype first.
let ltxApplyRopeSplit: @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray = compile { x, cosF, sinF in
    let c = cosF.asType(x.dtype)
    let s = sinF.asType(x.dtype)
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1)
}

/// SPLIT RoPE tables of the embeddings connector (`precompute_rope_freqs` with 1D positions `0..<length`,
/// `max_pos = [maxPos]`, theta 10000, `frequencies_precision: float64`).
///
/// The frequency grid is built exactly like the reference: MLX's float32 `linspace(0, 1, dim/2)`, then
/// `theta ** t * π/2` in Double on the CPU, rounded to float32. The per-position angles and cos/sin are float32.
/// Returns `(cos, sin)`, each `[1, heads, length, headDim/2]` float32.
public func ltxConnectorRope(length: Int, dim: Int, heads: Int, theta: Double = 10_000, maxPos: Int = 4096,
                             doublePrecision: Bool = true) -> (MLXArray, MLXArray) {
    let numFreqs = dim / 2
    let t0 = log(1.0) / log(theta), t1 = log(theta) / log(theta)
    let grid: MLXArray
    if doublePrecision {
        let t = MLXArray.linspace(Float(t0), Float(t1), count: numFreqs, stream: .cpu).asArray(Float.self)
        let g = t.map { Float(pow(theta, Double($0)) * (Double.pi / 2)) }
        grid = MLXArray(g)
    } else {
        grid = pow(MLXArray(Float(theta)), MLXArray.linspace(Float(t0), Float(t1), count: numFreqs)) * Float(Double.pi / 2)
    }
    let pos = MLXArray(Array(0 ..< length).map { Float($0) })
    let frac = pos / Float(maxPos)
    let scaled = grid.reshaped(1, numFreqs) * (frac.reshaped(length, 1) * 2 - 1)  // [N, numFreqs]
    let hd2 = dim / (2 * heads)
    let c = cos(scaled).reshaped(1, length, heads, hd2).transposed(0, 2, 1, 3)
    let s = sin(scaled).reshaped(1, length, heads, hd2).transposed(0, 2, 1, 3)
    return (c, s)
}

/// `Embeddings1DConnector`: learnable registers fill the padding, then 8 pre-norm transformer blocks
/// (affine-free RMSNorm, gated self-attention with q/k RMSNorm and SPLIT RoPE, GELU-tanh feed-forward),
/// and an affine-free output RMSNorm.
public final class LTXEmbeddingsConnector {
    struct Block {
        let q, k, v, out, gate: LTXDense
        let qNorm, kNorm: MLXArray
        let ff1, ff2: LTXDense
    }

    public let dim: Int
    public let heads: Int
    public let headDim: Int
    public let maxPos: Int
    public let doublePrecisionRope: Bool
    /// `[numRegisters, dim]`.
    public let registers: MLXArray
    let blocks: [Block]
    let ones: MLXArray
    private var ropeCache: (Int, MLXArray, MLXArray)?

    init(store: WeightStore, prefix: String, heads: Int, headDim: Int, maxPos: Int, doublePrecisionRope: Bool,
         dtype: DType) throws {
        self.heads = heads
        self.headDim = headDim
        self.maxPos = maxPos
        self.doublePrecisionRope = doublePrecisionRope
        registers = try store.get(prefix + "learnable_registers").asType(dtype)
        dim = registers.dim(1)
        var bs: [Block] = []
        var i = 0
        while store.has("\(prefix)transformer_1d_blocks.\(i).attn1.to_q.weight") {
            let b = "\(prefix)transformer_1d_blocks.\(i)."
            bs.append(Block(
                q: try LTXDense(store: store, prefix: b + "attn1.to_q", dtype: dtype),
                k: try LTXDense(store: store, prefix: b + "attn1.to_k", dtype: dtype),
                v: try LTXDense(store: store, prefix: b + "attn1.to_v", dtype: dtype),
                out: try LTXDense(store: store, prefix: b + "attn1.to_out.0", dtype: dtype),
                gate: try LTXDense(store: store, prefix: b + "attn1.to_gate_logits", dtype: dtype),
                qNorm: try store.get(b + "attn1.q_norm.weight").asType(dtype),
                kNorm: try store.get(b + "attn1.k_norm.weight").asType(dtype),
                ff1: try LTXDense(store: store, prefix: b + "ff.net.0.proj", dtype: dtype),
                ff2: try LTXDense(store: store, prefix: b + "ff.net.2", dtype: dtype)))
            i += 1
        }
        blocks = bs
        ones = MLXArray.ones([dim], dtype: dtype)
    }

    /// The reference's `_replace_padding_with_registers` for left-padded input: the `validCount` valid rows
    /// `validRows` `[1, n, dim]` go first, then the tiled registers at positions `n..<length`.
    public func fillWithRegisters(_ validRows: MLXArray, length: Int) -> MLXArray {
        let n = validRows.dim(1)
        let r = registers.dim(0)
        precondition(length % r == 0, "sequence length must be a multiple of the register count")
        guard n < length else { return validRows }
        let tiledRegs = tiled(registers, repetitions: [length / r, 1])[n ..< length]
        return concatenated([validRows, tiledRegs.expandedDimensions(axis: 0).asType(validRows.dtype)], axis: 1)
    }

    /// Runs the blocks and the output norm on the register-filled sequence `x` `[1, N, dim]` (no attention mask).
    /// `blockLimit` (debugging) stops after that many blocks and skips the output norm.
    public func callAsFunction(_ input: MLXArray, evalEachBlock: Bool = true, blockLimit: Int? = nil) -> MLXArray {
        let N = input.dim(1)
        let (cosF, sinF) = rope(N)
        let scale = Float(1 / Double(headDim).squareRoot())
        var x = input
        for b in blocks.prefix(blockLimit ?? blocks.count) {
            let h = rmsNorm(x, weight: ones, eps: 1e-6)
            var q = rmsNorm(b.q(h), weight: b.qNorm, eps: 1e-5)
            var k = rmsNorm(b.k(h), weight: b.kNorm, eps: 1e-5)
            q = q.reshaped(1, N, heads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(1, N, heads, headDim).transposed(0, 2, 1, 3)
            let v = b.v(h).reshaped(1, N, heads, headDim).transposed(0, 2, 1, 3)
            q = ltxApplyRopeSplit(q, cosF, sinF)
            k = ltxApplyRopeSplit(k, cosF, sinF)
            let w = softmax(matmul(q, k.transposed(0, 1, 3, 2)) * scale, axis: -1)
            var o = matmul(w, v)
            let gate = 2.0 * sigmoid(b.gate(h))  // [1, N, H]
            o = o * gate.transposed(0, 2, 1).expandedDimensions(axis: 3)
            o = o.transposed(0, 2, 1, 3).reshaped(1, N, heads * headDim)
            x = x + b.out(o)
            let h2 = rmsNorm(x, weight: ones, eps: 1e-6)
            x = x + b.ff2(geluApprox(b.ff1(h2)))
            if evalEachBlock { eval(x) }
        }
        return blockLimit == nil ? rmsNorm(x, weight: ones, eps: 1e-6) : x
    }

    func rope(_ n: Int) -> (MLXArray, MLXArray) {
        if let c = ropeCache, c.0 == n { return (c.1, c.2) }
        let (c, s) = ltxConnectorRope(length: n, dim: dim, heads: dim / headDim, maxPos: maxPos,
                                      doublePrecision: doublePrecisionRope)
        eval(c, s)
        ropeCache = (n, c, s)
        return (c, s)
    }
}

/// `TextEncoderConnector` (`connector.safetensors`, prefix `connector.`): the multi-layer projection
/// (`text_embedding_projection.{video,audio}_aggregate_embed`) plus the video and audio embeddings connectors.
public final class LTXTextConnector {
    let videoProj: LTXDense
    let audioProj: LTXDense
    /// Gemma hidden size the projection's rescale refers to (3840).
    public let embeddingDim: Int
    public let video: LTXEmbeddingsConnector
    public let audio: LTXEmbeddingsConnector

    /// - Parameters:
    ///   - url: `connector.safetensors`.
    ///   - ltxConfig: `embedded_config.json` (its `transformer` section gives the connector heads / max position /
    ///     `frequencies_precision`); defaults match LTX-2.3 when absent.
    public init(url: URL, ltxConfig: URL?, embeddingDim: Int = 3840, dtype: DType = .bfloat16) throws {
        let store = try WeightStore(url: url)
        var t: [String: Any] = [:]
        if let ltxConfig, let d = try? Data(contentsOf: ltxConfig),
           let obj = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            t = obj["transformer"] as? [String: Any] ?? obj
        }
        func int(_ k: String, _ d: Int) -> Int { (t[k] as? NSNumber)?.intValue ?? d }
        let maxPos = (t["connector_positional_embedding_max_pos"] as? [NSNumber])?.first?.intValue ?? 4096
        let f64 = (t["frequencies_precision"] as? String) == "float64"
        let p = "connector."
        self.embeddingDim = embeddingDim
        videoProj = try LTXDense(store: store, prefix: p + "text_embedding_projection.video_aggregate_embed", dtype: dtype)
        audioProj = try LTXDense(store: store, prefix: p + "text_embedding_projection.audio_aggregate_embed", dtype: dtype)
        video = try LTXEmbeddingsConnector(
            store: store, prefix: p + "video_embeddings_connector.",
            heads: int("connector_num_attention_heads", 32), headDim: int("connector_attention_head_dim", 128),
            maxPos: maxPos, doublePrecisionRope: f64, dtype: dtype)
        audio = try LTXEmbeddingsConnector(
            store: store, prefix: p + "audio_embeddings_connector.",
            heads: int("audio_connector_num_attention_heads", 32), headDim: int("audio_connector_attention_head_dim", 64),
            maxPos: maxPos, doublePrecisionRope: f64, dtype: dtype)
    }

    /// `TextEmbeddingProjection`: `x` `[1, n, layers*hidden]` → video `[1, n, 4096]`, audio `[1, n, 2048]`,
    /// each with the `sqrt(target_dim / embedding_dim)` input rescale.
    public func project(_ x: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let vs = Float((Double(videoProj.weight.dim(0)) / Double(embeddingDim)).squareRoot())
        let v = videoProj(x * vs)
        eval(v)
        let asc = Float((Double(audioProj.weight.dim(0)) / Double(embeddingDim)).squareRoot())
        let a = audioProj(x * asc)
        eval(a)
        return (v, a)
    }
}

/// Per-token RMS normalization of the stacked Gemma hidden states (`text_encoder_norm_type: per_token_rms`):
/// stack on a new last axis `[1, n, D, L]`, normalize over `D`, flatten D-interleaved to `[1, n, D*L]`.
///
/// `paddedLength`: the reference normalizes the whole left-padded sequence, and MLX's strided mean picks its
/// accumulation order from the full shape, which moves the bfloat16 variance by an ulp for some tokens. Passing the
/// padded length reproduces that exactly: the rows are left-padded with zeros, normalized, and the real rows returned
/// (or all `paddedLength` rows with `keepPadding`, the padded rows being zero as in the reference).
public func ltxPerTokenRMSStack(_ states: [MLXArray], paddedLength: Int? = nil, keepPadding: Bool = false) -> MLXArray {
    let n = states[0].dim(1)
    var input = states
    if let L = paddedLength, L > n {
        let z = MLXArray.zeros([1, L - n, states[0].dim(2)], dtype: states[0].dtype)
        input = states.map { concatenated([z, $0], axis: 1) }
    }
    let encoded = stacked(input, axis: -1)
    let variance = mean(encoded * encoded, axis: 2, keepDims: true)
    let normed = encoded * rsqrt(variance + 1e-6)
    let (b, t, d, l) = (normed.dim(0), normed.dim(1), normed.dim(2), normed.dim(3))
    let flat = normed.reshaped(b, t, d * l)
    return t > n && !keepPadding ? flat[0..., (t - n)...] : flat
}
