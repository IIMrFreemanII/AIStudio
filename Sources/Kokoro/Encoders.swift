import Foundation
import MLX
import MLXDiffusionCore

/// PL-BERT: transformers' `AlbertModel` (one shared layer applied `layers` times, gelu_new, LayerNorm eps 1e-12),
/// returning `last_hidden_state`. Batch 1 without padding, so the attention mask is all ones.
struct Albert {
    let word: MLXArray, pos: MLXArray, type0: MLXArray
    let embLN: (MLXArray, MLXArray)
    let mapIn: Lin
    let q: Lin, k: Lin, v: Lin, dense: Lin, ffn: Lin, ffnOut: Lin
    let attnLN: (MLXArray, MLXArray), fullLN: (MLXArray, MLXArray)
    let heads: Int, layers: Int

    init(_ w: KWeights, heads: Int, layers: Int) throws {
        let p = "bert.", l = "bert.encoder.albert_layer_groups.0.albert_layers.0."
        let dt = w.dtype
        word = try w.get(p + "embeddings.word_embeddings.weight").asType(dt)
        pos = try w.get(p + "embeddings.position_embeddings.weight").asType(dt)
        type0 = try w.get(p + "embeddings.token_type_embeddings.weight")[0].asType(dt)
        embLN = (try w.get(p + "embeddings.LayerNorm.weight"), try w.get(p + "embeddings.LayerNorm.bias"))
        mapIn = try w.linear(p + "encoder.embedding_hidden_mapping_in")
        q = try w.linear(l + "attention.query"); k = try w.linear(l + "attention.key")
        v = try w.linear(l + "attention.value"); dense = try w.linear(l + "attention.dense")
        ffn = try w.linear(l + "ffn"); ffnOut = try w.linear(l + "ffn_output")
        attnLN = (try w.get(l + "attention.LayerNorm.weight"), try w.get(l + "attention.LayerNorm.bias"))
        fullLN = (try w.get(l + "full_layer_layer_norm.weight"), try w.get(l + "full_layer_layer_norm.bias"))
        self.heads = heads
        self.layers = layers
    }

    private func ln(_ x: MLXArray, _ p: (MLXArray, MLXArray), dtype: DType) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: p.0, bias: p.1, eps: 1e-12).asType(dtype)
    }

    /// `ids`: `[1, T]` int32 → `[1, T, hidden]`.
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let T = ids.dim(1), dt = word.dtype
        var e = word[ids[0]] + pos[0 ..< T] + type0
        e = ln(e.expandedDimensions(axis: 0), embLN, dtype: dt)
        var h = mapIn(e)
        let D = h.dim(2), hd = D / heads
        for _ in 0 ..< layers {
            func split(_ t: MLXArray) -> MLXArray { t.reshaped(1, T, heads, hd).transposed(0, 2, 1, 3) }
            let a = MLXFast.scaledDotProductAttention(
                queries: split(q(h)), keys: split(k(h)), values: split(v(h)),
                scale: 1 / Float(hd).squareRoot(), mask: nil)
            let ctx = a.transposed(0, 2, 1, 3).reshaped(1, T, D)
            let attn = ln(h + dense(ctx), attnLN, dtype: dt)
            h = ln(ffnOut(geluTanh(ffn(attn))) + attn, fullLN, dtype: dt)
        }
        return h
    }
}

/// `TextEncoder`: embedding → 3 × (weight-normed Conv1d k=5, LayerNorm, LeakyReLU 0.2) → BiLSTM. `[1, T, 512]`.
struct TextEncoder {
    let embedding: MLXArray
    let cnn: [(conv: KConv, gamma: MLXArray, beta: MLXArray)]
    let lstm: BiLSTM

    init(_ w: KWeights, depth: Int, kernel: Int) throws {
        embedding = try w.get("text_encoder.embedding.weight").asType(w.dtype)
        cnn = try (0 ..< depth).map { i in
            let k = "text_encoder.cnn.\(i)."
            return (try w.conv(k + "0", padding: (kernel - 1) / 2), try w.get(k + "1.gamma"), try w.get(k + "1.beta"))
        }
        lstm = try BiLSTM(w, "text_encoder.lstm")
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var x = embedding[ids[0]].expandedDimensions(axis: 0)
        for c in cnn {
            x = c.conv(x)
            x = leaky(MLXFast.layerNorm(x.asType(.float32), weight: c.gamma, bias: c.beta, eps: 1e-5), 0.2)
        }
        return lstm(x)
    }
}

/// `ProsodyPredictor`: the `DurationEncoder` (BiLSTM + AdaLayerNorm stack with the style re-appended after each
/// norm), the duration LSTM/projection and `F0Ntrain`.
struct ProsodyPredictor {
    let encLSTMs: [BiLSTM]
    let encNorms: [AdaLayerNorm]
    let lstm: BiLSTM
    let durationProj: Lin
    let shared: BiLSTM
    let f0: [AdainResBlk1d], n: [AdainResBlk1d]
    let f0Proj: KConv, nProj: KConv

    init(_ w: KWeights, layers: Int) throws {
        let p = "predictor."
        encLSTMs = try (0 ..< layers).map { try BiLSTM(w, p + "text_encoder.lstms.\(2 * $0)") }
        encNorms = try (0 ..< layers).map { try AdaLayerNorm(w, p + "text_encoder.lstms.\(2 * $0 + 1)") }
        lstm = try BiLSTM(w, p + "lstm")
        durationProj = try w.linear(p + "duration_proj.linear_layer", dtype: .float32)
        shared = try BiLSTM(w, p + "shared")
        f0 = try [0, 1, 2].map { try AdainResBlk1d(w, p + "F0.\($0)", upsample: $0 == 1) }
        n = try [0, 1, 2].map { try AdainResBlk1d(w, p + "N.\($0)", upsample: $0 == 1) }
        f0Proj = try w.conv(p + "F0_proj", dtype: .float32)
        nProj = try w.conv(p + "N_proj", dtype: .float32)
    }

    /// `dEn`: `[1, T, 512]`, `s`: `[1, 128]` → `d`: `[1, T, 640]` (last 128 channels are the style).
    func durationEncoder(_ dEn: MLXArray, _ s: MLXArray) -> MLXArray {
        let T = dEn.dim(1)
        let sT = broadcast(s.expandedDimensions(axis: 1), to: [1, T, s.dim(1)])
        var x = concatenated([dEn.asType(.float32), sT], axis: 2)
        for (l, nrm) in zip(encLSTMs, encNorms) {
            x = concatenated([nrm(l(x), s), sT], axis: 2)
        }
        return x
    }

    /// Float durations (frames per token, before rounding): `sigmoid(proj(lstm(d))).sum(-1) / speed`. `[T]`.
    func durations(_ d: MLXArray, speed: Float) -> MLXArray {
        sigmoid(durationProj(lstm(d))).sum(axis: -1)[0] / speed
    }

    /// `en`: `[1, F, 640]` → (F0, N) curves `[1, 2F]`.
    func f0n(_ en: MLXArray, _ s: MLXArray) -> (MLXArray, MLXArray) {
        let x = shared(en)
        var f = x, nn = x
        for b in f0 { f = b(f, s) }
        for b in n { nn = b(nn, s) }
        return (f0Proj(f.asType(.float32))[0..., 0..., 0], nProj(nn.asType(.float32))[0..., 0..., 0])
    }
}
