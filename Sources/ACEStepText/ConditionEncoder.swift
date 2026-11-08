import Foundation
import MLX
import MLXDiffusionCore

/// ACE-Step 1.5 condition encoder (`AceStepConditionEncoder`, weights under `encoder.` in the DiT checkpoint).
///
/// Builds the DiT cross-attention sequence from three conditions:
/// - caption: Qwen3-Embedding `last_hidden_state` `[B, Lt, 1024]` → `text_projector` (linear, no bias) → `[B, Lt, 2048]`;
/// - lyrics: Qwen3-Embedding token embeddings (`Qwen3Model.embed`) `[B, Ll, 1024]` → lyric encoder (linear + 8
///   bidirectional encoder layers + RMSNorm) → `[B, Ll, 2048]`;
/// - timbre: reference acoustic latents `[N, T, 64]` (T = 750) → timbre encoder (linear + 4 bidirectional encoder layers
///   + RMSNorm), frame 0 of each reference taken as its embedding `[N, 2048]`, then grouped per batch item by
///   `timbreOrder` → `[B, maxRefs, 2048]`.
/// The three are packed as `pack(pack(lyrics, timbre), text)`: per batch item, valid tokens (mask 1) first in their
/// original order, padding after. Output length S = Ll + maxRefs + Lt.
///
/// Encoder layers are Qwen3-style (pre-RMSNorm, GQA 16/8 heads × 128 with per-head q/k RMSNorm, rotate-half RoPE,
/// SwiGLU MLP) but non-causal. Layers alternate `layer_types` from the config (even = sliding window |i-j| ≤ 128, odd =
/// full). The lyric encoder masks padded keys with `lyricMask`; the timbre encoder uses no padding mask.
public final class ACEConditionEncoder {
    public struct Config {
        public var hiddenSize = 2048, intermediateSize = 6144
        public var textHiddenDim = 1024, timbreHiddenDim = 64
        public var numHeads = 16, numKVHeads = 8, headDim = 128
        public var numLyricLayers = 8, numTimbreLayers = 4
        public var rmsEps: Float = 1e-6
        public var ropeTheta: Float = 1_000_000
        public var slidingWindow = 128
        public var useSlidingWindow = true
        /// Per layer index: `true` = sliding-window attention, `false` = full attention.
        public var slidingLayers: [Bool] = (0..<24).map { $0 % 2 == 0 }

        public init() {}

        /// Parses `acestep-v15-turbo/config.json`.
        public init(url: URL) throws {
            let o = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] ?? [:]
            hiddenSize = o["hidden_size"] as? Int ?? hiddenSize
            intermediateSize = o["intermediate_size"] as? Int ?? intermediateSize
            textHiddenDim = o["text_hidden_dim"] as? Int ?? textHiddenDim
            timbreHiddenDim = o["timbre_hidden_dim"] as? Int ?? timbreHiddenDim
            numHeads = o["num_attention_heads"] as? Int ?? numHeads
            numKVHeads = o["num_key_value_heads"] as? Int ?? numKVHeads
            headDim = o["head_dim"] as? Int ?? hiddenSize / numHeads
            numLyricLayers = o["num_lyric_encoder_hidden_layers"] as? Int ?? numLyricLayers
            numTimbreLayers = o["num_timbre_encoder_hidden_layers"] as? Int ?? numTimbreLayers
            rmsEps = Float(o["rms_norm_eps"] as? Double ?? Double(rmsEps))
            ropeTheta = Float(o["rope_theta"] as? Double ?? Double(ropeTheta))
            slidingWindow = o["sliding_window"] as? Int ?? slidingWindow
            useSlidingWindow = o["use_sliding_window"] as? Bool ?? useSlidingWindow
            if let t = o["layer_types"] as? [String] { slidingLayers = t.map { $0 == "sliding_attention" } }
        }
    }

    struct Layer {
        let inputNorm, postNorm: MLXArray
        let q, k, v, o: Lin
        let qNorm, kNorm: MLXArray
        let gate, up, down: Lin
        let sliding: Bool
    }

    struct Encoder {
        let embed: Lin
        let layers: [Layer]
        let norm: MLXArray
    }

    public let config: Config
    public let dtype: DType
    let textProjector: Lin
    let lyric: Encoder
    let timbreEnc: Encoder
    /// `null_condition_emb` `[1, 1, 2048]` (CFG unconditional embedding; unused by turbo), if present.
    public let nullConditionEmb: MLXArray?
    /// `timbre_encoder.special_token` `[1, 1, 2048]`. Present in the checkpoint but unused by the reference forward
    /// (the prepend is commented out); kept for completeness.
    public let timbreSpecialToken: MLXArray?

    /// `store`: the DiT checkpoint (`acestep-v15-turbo/model.safetensors`); keys under `encoder.` (and top-level
    /// `null_condition_emb`). `prefix` lets a pre-stripped store be used.
    public init(store: WeightStore, config: Config, dtype: DType = .float32, prefix: String = "encoder.") throws {
        self.config = config
        self.dtype = dtype
        func layers(_ p: String, _ n: Int) throws -> [Layer] {
            try (0..<n).map { i in
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
                    down: try store.linear(l + "mlp.down_proj", dtype: dtype),
                    sliding: config.useSlidingWindow && i < config.slidingLayers.count && config.slidingLayers[i])
            }
        }
        func encoder(_ p: String, _ n: Int) throws -> Encoder {
            Encoder(embed: try store.linear(p + "embed_tokens", dtype: dtype), layers: try layers(p, n),
                    norm: try store.get(p + "norm.weight").asType(dtype))
        }
        textProjector = try store.linear(prefix + "text_projector", dtype: dtype)
        lyric = try encoder(prefix + "lyric_encoder.", config.numLyricLayers)
        timbreEnc = try encoder(prefix + "timbre_encoder.", config.numTimbreLayers)
        nullConditionEmb = store.optional("null_condition_emb")?.asType(dtype)
        timbreSpecialToken = store.optional(prefix + "timbre_encoder.special_token")?.asType(dtype)
    }

    /// Encodes all conditions and packs them.
    ///
    /// - `text`: caption hidden states `[B, Lt, textHiddenDim]`; `textMask` `[B, Lt]` (1 = valid, 0 = pad).
    /// - `lyrics`: lyric token embeddings `[B, Ll, textHiddenDim]`; `lyricMask` `[B, Ll]`.
    /// - `timbre`: packed reference latents `[N, T, timbreHiddenDim]`; `timbreOrder[n]` = batch item of reference `n`
    ///   (text-to-music: `silence_latent[:, :750, :]` with `[0]`). Every batch item 0...max(order) gets a slot.
    /// - Returns `hidden` `[B, S, hiddenSize]` in `dtype` and `mask` `[B, S]` float32 (1 = valid; valid tokens first).
    public func callAsFunction(text: MLXArray, textMask: MLXArray, lyrics: MLXArray, lyricMask: MLXArray,
                               timbre: MLXArray, timbreOrder: [Int]) -> (hidden: MLXArray, mask: MLXArray) {
        let t = projectText(text)
        let l = encodeLyrics(lyrics, mask: lyricMask)
        let (tb, tbMask) = encodeTimbre(timbre, order: timbreOrder)
        let (h1, m1) = Self.pack(l, tb, lyricMask, tbMask)
        return Self.pack(h1, t, m1, textMask)
    }

    /// `text_projector(text)`: `[B, Lt, hiddenSize]`.
    public func projectText(_ text: MLXArray) -> MLXArray { textProjector(text.asType(dtype)) }

    /// Lyric encoder `last_hidden_state` `[B, Ll, hiddenSize]`.
    public func encodeLyrics(_ lyrics: MLXArray, mask: MLXArray) -> MLXArray {
        run(lyric, lyrics, keyMask: mask)
    }

    /// Timbre embeddings grouped per batch item: `[B, maxRefs, hiddenSize]` (zeros in empty slots) and mask
    /// `[B, maxRefs]` float32.
    public func encodeTimbre(_ refs: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        precondition(order.count == refs.dim(0), "timbreOrder must have one entry per reference")
        let emb = run(timbreEnc, refs, keyMask: nil)[0..., 0, 0...]  // [N, D]
        let B = (order.max() ?? -1) + 1
        var slots = [[Int]](repeating: [], count: B)
        for (n, b) in order.enumerated() { slots[b].append(n) }
        let maxCount = slots.map(\.count).max() ?? 0
        let D = emb.dim(1)
        let zero = MLXArray.zeros([1, D], dtype: emb.dtype)
        var rows: [MLXArray] = []
        var maskVals: [Float] = []
        for s in slots {
            for j in 0..<maxCount {
                if j < s.count {
                    rows.append(emb[s[j] ..< s[j] + 1])
                    maskVals.append(1)
                } else {
                    rows.append(zero)
                    maskVals.append(0)
                }
            }
        }
        let out = concatenated(rows, axis: 0).reshaped(B, maxCount, D)
        return (out, MLXArray(maskVals, [B, maxCount]))
    }

    // MARK: - internals

    private func run(_ enc: Encoder, _ input: MLXArray, keyMask: MLXArray?) -> MLXArray {
        let c = config
        let B = input.dim(0), L = input.dim(1)
        let scale = 1 / Float(c.headDim).squareRoot()
        var x = enc.embed(input.asType(dtype))
        let fullMask = attentionMask(L: L, window: nil, keyMask: keyMask)
        let slideMask = attentionMask(L: L, window: c.slidingWindow, keyMask: keyMask)
        for layer in enc.layers {
            let h = MLXFast.rmsNorm(x, weight: layer.inputNorm, eps: c.rmsEps)
            var q = layer.q(h).reshaped(B, L, c.numHeads, c.headDim)
            var k = layer.k(h).reshaped(B, L, c.numKVHeads, c.headDim)
            let v = layer.v(h).reshaped(B, L, c.numKVHeads, c.headDim).transposed(0, 2, 1, 3)
            q = MLXFast.rmsNorm(q, weight: layer.qNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            k = MLXFast.rmsNorm(k, weight: layer.kNorm, eps: c.rmsEps).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: c.headDim, traditional: false, base: c.ropeTheta, scale: 1, offset: 0)
            let a = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale,
                                                      mask: layer.sliding ? slideMask : fullMask)
            x = x + layer.o(a.transposed(0, 2, 1, 3).reshaped(B, L, -1))
            let h2 = MLXFast.rmsNorm(x, weight: layer.postNorm, eps: c.rmsEps)
            x = x + layer.down(siluAct(layer.gate(h2)) * layer.up(h2))
        }
        return MLXFast.rmsNorm(x, weight: enc.norm, eps: c.rmsEps)
    }

    /// `create_4d_mask(is_causal: false)`: bidirectional, optional |i-j| ≤ window band, padded keys masked.
    /// Additive with a large finite negative (as torch's `finfo.min`) so an all-padding row stays finite.
    private func attentionMask(L: Int, window: Int?, keyMask: MLXArray?) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let needBand = window.map { L - 1 > $0 } ?? false
        var keep: MLXArray?
        if let w = window, needBand {
            let i = MLXArray(Array(Int32(0) ..< Int32(L)))
            let diff = abs(i.expandedDimensions(axis: 1) - i.expandedDimensions(axis: 0))
            keep = (diff .<= Int32(w)).expandedDimensions(axes: [0, 1])  // [1, 1, L, L]
        }
        if let keyMask {
            let km = (keyMask.asType(.float32) .> 0).reshaped(keyMask.dim(0), 1, 1, L)  // [B, 1, 1, L]
            if km.all().item(Bool.self) == false {
                keep = keep.map { $0 .&& km } ?? km
            }
        }
        guard let keep else { return .none }
        let neg = MLXArray(dtype == .float16 ? Float(-65504) : Float(-1e30))
        return .array(MLX.where(keep, MLXArray(Float(0)), neg).asType(dtype))
    }

    /// `pack_sequences`: concatenates along the sequence axis and stably moves valid (mask > 0) tokens to the front
    /// of each batch row. Returns the packed hidden states and a float32 mask `[B, L1 + L2]` (1 for the first
    /// `sum(mask)` positions).
    public static func pack(_ h1: MLXArray, _ h2: MLXArray, _ m1: MLXArray, _ m2: MLXArray) -> (MLXArray, MLXArray) {
        let h = concatenated([h1, h2.asType(h1.dtype)], axis: 1)
        let m = concatenated([m1.asType(.float32), m2.asType(.float32)], axis: 1)
        let B = h.dim(0), L = h.dim(1)
        let vals = m.asArray(Float.self)
        var idx: [Int32] = []
        var newMask: [Float] = []
        idx.reserveCapacity(B * L)
        for b in 0..<B {
            let row = Array(vals[b * L ..< (b + 1) * L])
            // argsort(descending, stable)
            let order = (0..<L).sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }
            idx += order.map(Int32.init)
            let n = Int(row.reduce(0, +))
            newMask += (0..<L).map { $0 < n ? 1 : 0 }
        }
        let gather = MLXArray(idx, [B, L, 1])
        let packed = takeAlong(h, gather, axis: 1)
        return (packed, MLXArray(newMask, [B, L]))
    }
}
