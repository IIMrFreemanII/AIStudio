import ACEStepText
import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

public enum ACEPlannerError: Error, CustomStringConvertible {
    case tokenizer(String)
    case generation(String)

    public var description: String {
        switch self {
        case .tokenizer(let s): return "ACE planner tokenizer: \(s)"
        case .generation(let s): return "ACE planner: \(s)"
        }
    }
}

/// Inputs of one planner run (`LLMHandler.generate_with_stop_condition` with `infer_type="llm_dit"` as called by
/// `acestep/inference.py generate_music`; defaults are the `GenerationParams` ones).
public struct ACEPlanRequest {
    public var caption: String
    public var lyrics: String
    /// Target length in seconds: forced into the CoT (`duration: N`) and fixes the code count to `Int(duration·5)`.
    public var duration: Double
    /// Optional user metadata; when given it is injected into the CoT instead of generated.
    public var bpm: Int?
    public var keyscale: String?
    public var timeSignature: String?
    /// Vocal language for the DiT's lyric prompt. Not shown to the LM (matching the official pipeline).
    public var language: String?
    public var seed: UInt64
    public var temperature: Float = 0.85
    public var cfgScale: Float = 2.0
    public var topP: Float = 0.9
    /// 0 = off.
    public var topK: Int = 0
    public var negativePrompt = ACEPlannerText.defaultNegativePrompt
    /// `use_cot_caption`: let the LM rewrite the caption (the DiT then sees the rewrite).
    public var useCotCaption = true
    /// `use_cot_language`: let the LM write a `language:` field.
    public var useCotLanguage = true
    /// Safety cap on CoT tokens (the reference allows ~3500; the caption alone is capped at 512).
    public var maxCoTTokens = 2048

    public init(caption: String, lyrics: String, duration: Double, bpm: Int? = nil, keyscale: String? = nil,
                timeSignature: String? = nil, language: String? = nil, seed: UInt64) {
        self.caption = caption
        self.lyrics = lyrics
        self.duration = duration
        self.bpm = bpm
        self.keyscale = keyscale
        self.timeSignature = timeSignature
        self.language = language
        self.seed = seed
    }

    /// `user_metadata` of the official call: bpm / keyscale / timesignature / duration as strings.
    var userMetadata: [String: String] {
        var m: [String: String] = [:]
        if let bpm, bpm > 0 { m["bpm"] = String(bpm) }
        if let k = keyscale?.trimmingCharacters(in: .whitespacesAndNewlines), !k.isEmpty, k.lowercased() != "n/a" {
            m["keyscale"] = k
        }
        if let t = timeSignature?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty, t.lowercased() != "n/a" {
            m["timesignature"] = t
        }
        if duration > 0 { m["duration"] = String(Int(duration)) }
        return m
    }
}

/// Result of a planner run.
public struct ACEPlan {
    /// Metadata parsed from the generated CoT (or the user's when all of bpm/key/time signature/duration were
    /// given, in which case the CoT phase is skipped like in the reference).
    public var metadata: ACEPlanMetadata
    /// Raw phase-1 output (`<think>\nbpm: …\n<|im_end|>`), empty when phase 1 was skipped.
    public var cotText: String
    /// The `<think>…</think>` block (YAML) that conditioned phase 2.
    public var cot: String
    /// Audio codes 0..<64000, exactly `Int(duration·5)` of them.
    public var codes: [Int]
    public var cotTokens = 0
    public var cotSeconds = 0.0
    public var codesSeconds = 0.0

    public init(metadata: ACEPlanMetadata, cotText: String, cot: String, codes: [Int], cotTokens: Int = 0,
                cotSeconds: Double = 0, codesSeconds: Double = 0) {
        self.metadata = metadata
        self.cotText = cotText
        self.cot = cot
        self.codes = codes
        self.cotTokens = cotTokens
        self.cotSeconds = cotSeconds
        self.codesSeconds = codesSeconds
    }

    public var bpm: Int? { metadata.bpm }
    public var caption: String? { metadata.caption }
    public var keyscale: String? { metadata.keyscale }
    public var language: String? { metadata.language }
    public var timeSignature: String? { metadata.timeSignature }
}

public enum ACEPlanPhase {
    /// Phase 1; `total` is the token cap, the run usually ends far earlier.
    case thinking
    /// Phase 2; `total` is the exact code count.
    case codes
}

/// The ACE-Step 1.5 5 Hz planner LM (Qwen3-1.7B fine-tune, `acestep-5Hz-lm-1.7B`).
public final class ACEPlanner {
    public let tokenizer: Tokenizer
    public let model: Qwen3Model
    let fsm: ACEConstrainedDecoder
    let pad: Int
    let captionMask: MLXArray
    public private(set) var tokenizerLoadSeconds = 0.0
    public private(set) var modelLoadSeconds = 0.0

    public init(directory: URL, dtype: DType = .bfloat16) async throws {
        var t = Date()
        tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        tokenizerLoadSeconds = Date().timeIntervalSince(t)
        t = Date()
        model = try Qwen3Model(directory: directory, dtype: dtype)
        // Weights are memory-mapped lazily; one tiny forward pulls them all in (and compiles the kernels).
        eval(model.logits(model(MLXArray([Int32(0)]).reshaped(1, 1))))
        modelLoadSeconds = Date().timeIntervalSince(t)
        fsm = try ACEConstrainedDecoder(tokenizer: tokenizer)
        pad = tokenizer.convertTokenToId("<|endoftext|>") ?? 151_643
        let v = model.config.vocabSize
        var m = [Float](repeating: 0, count: v)
        for i in fsm.audioCodes { m[i] = -.infinity }
        if fsm.backtick >= 0 { m[fsm.backtick] = -.infinity }
        captionMask = MLXArray(m)
    }

    /// Diagnostics: the constrained decoder's prefix trees ("bpm", "duration", "timesig", "keyscale", "language";
    /// token-id prefix → sorted allowed tokens) and the token sequence it forces for each fixed string.
    public static func constraintTables(tokenizer: Tokenizer) throws -> (trees: [String: [[Int]: [Int]]], fixed: [String: [Int]]) {
        let d = try ACEConstrainedDecoder(tokenizer: tokenizer)
        func conv(_ t: [[Int]: Set<Int>]) -> [[Int]: [Int]] { t.mapValues { $0.sorted() } }
        let trees = ["bpm": conv(d.bpmTree), "duration": conv(d.durationTree), "timesig": conv(d.timesigTree),
                     "keyscale": conv(d.keyscaleTree), "language": conv(d.languageTree)]
        var fixed: [String: [Int]] = [:]
        for s in ["<think>", "\n", "bpm:", "caption:", "duration:", "keyscale:", "language:", "timesignature:", "</think>"] {
            fixed[s] = d.forcedTokens(for: s)
        }
        return (trees, fixed)
    }

    public func encode(_ text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// Text for token ids (audio codes rendered as `<|audio_code_N|>`).
    public func decode(_ ids: [Int]) -> String {
        var out = ""
        var run: [Int] = []
        for id in ids {
            if fsm.audioCodes.contains(id) {
                if !run.isEmpty { out += tokenizer.decode(tokens: run); run = [] }
                out += fsm.decode(id)
            } else {
                run.append(id)
            }
        }
        if !run.isEmpty { out += tokenizer.decode(tokens: run) }
        return out
    }

    // MARK: - Sampling

    /// `_apply_top_k_filter` + `_apply_top_p_filter` (on untempered logits) + temperature sampling (argmax when
    /// `temperature <= 0`). `logits`: `[n]` float32. Returns an index into `logits` (lazy scalar).
    static func sample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float, key: MLXArray) -> MLXArray {
        if temperature <= 0 { return logits.argMax() }
        var l = logits
        if topK > 0 && topK < l.dim(0) {
            let kth = sorted(l)[l.dim(0) - topK]
            l = MLX.where(l .< kth, MLXArray(-Float.infinity), l)
        }
        if topP > 0 && topP < 1 {
            let order = argSort(-l)
            let s = l[order]
            let probs = softmax(s, axis: -1, precise: true)
            let before = probs.cumsum(axis: -1, inclusive: false)
            let kept = MLX.where(before .> topP, MLXArray(-Float.infinity), s / temperature)
            let j = MLXRandom.categorical(kept, key: key)
            return order[j]
        }
        return MLXRandom.categorical(l / temperature, key: key)
    }

    // MARK: - Planning

    public func plan(_ r: ACEPlanRequest, onProgress: ((ACEPlanPhase, Int, Int) throws -> Void)? = nil) throws -> ACEPlan {
        let user = r.userMetadata
        let keys = MLXRandom.split(key: MLXRandom.key(r.seed), into: 2)
        var metadata: ACEPlanMetadata
        var cotText = ""
        var cotTokens = 0
        var t = Date()
        let hasAllMetas = user["bpm"] != nil && user["keyscale"] != nil && user["timesignature"] != nil
            && user["duration"] != nil
        if !hasAllMetas {
            let ids = try generateCoT(r, user: user, key: keys[0], onProgress: onProgress)
            cotTokens = ids.count
            cotText = decode(ids)
            metadata = ACEPlannerText.parse(cotText)
        } else {
            metadata = ACEPlanMetadata(bpm: r.bpm, duration: Int(r.duration), keyscale: user["keyscale"],
                                       timeSignature: user["timesignature"])
        }
        let cotSeconds = Date().timeIntervalSince(t)
        var target = r.duration
        if target <= 0, let d = metadata.duration { target = Double(d) }
        let cot = ACEPlannerText.formatCoT(metadata)
        t = Date()
        let codes = try generateCodes(r, cot: cot, targetDuration: target, key: keys[1], onProgress: onProgress)
        return ACEPlan(metadata: metadata, cotText: cotText, cot: cot, codes: codes, cotTokens: cotTokens,
                       cotSeconds: cotSeconds, codesSeconds: Date().timeIntervalSince(t))
    }

    /// Phase 2 only: codes for given metadata (e.g. edited by the user), conditioned on `formatCoT(metadata)`.
    public func codes(_ r: ACEPlanRequest, metadata: ACEPlanMetadata,
                      onProgress: ((ACEPlanPhase, Int, Int) throws -> Void)? = nil) throws -> [Int] {
        let keys = MLXRandom.split(key: MLXRandom.key(r.seed), into: 2)
        return try generateCodes(r, cot: ACEPlannerText.formatCoT(metadata), targetDuration: r.duration, key: keys[1],
                                 onProgress: onProgress)
    }

    /// Phase 1: constrained CoT generation without CFG (the reference forces `cfg_scale = 1` here). Returns the
    /// generated token ids (ending with EOS when the FSM stops at `</think>`).
    func generateCoT(_ r: ACEPlanRequest, user: [String: String], key: MLXArray,
                     onProgress: ((ACEPlanPhase, Int, Int) throws -> Void)?) throws -> [Int] {
        let fsm = self.fsm
        fsm.reset()
        fsm.phase = .cot
        fsm.stopAtReasoning = true
        fsm.skipCaption = !r.useCotCaption
        fsm.skipLanguage = !r.useCotLanguage
        fsm.setTargetDuration(nil)
        fsm.user = user
        fsm.promptHasThinkEnd = false

        let cache = Qwen3Model.KVCache()
        var pending = encode(ACEPlannerText.cotPrompt(caption: r.caption, lyrics: r.lyrics))
        var lastHidden: MLXArray?
        var stepLogits: MLXArray?
        func logits() -> MLXArray {
            if let stepLogits { return stepLogits }
            if !pending.isEmpty {
                let h = model(MLXArray(pending.map(Int32.init)).reshaped(1, -1), cache: cache)
                lastHidden = h[0, -1]
                pending = []
            }
            let l = model.logits(lastHidden!)
            eval(l)
            stepLogits = l
            return l
        }
        var key = key
        var out: [Int] = []
        for _ in 0..<r.maxCoTTokens {
            stepLogits = nil
            let c = fsm.constraint(
                top: { logits().argMax().item(Int.self) },
                best: { cands in cands[logits()[MLXArray(cands.map(Int32.init))].argMax().item(Int.self)] })
            let (k1, k2) = MLXRandom.split(key: key)
            key = k1
            let tok: Int
            switch c {
            case .only(let a) where a.count <= 1:
                tok = a.first ?? fsm.newline
            case .only(let a):
                let idx = MLXArray(a.map(Int32.init))
                tok = a[Self.sample(logits()[idx], temperature: r.temperature, topK: r.topK, topP: r.topP, key: k2)
                    .item(Int.self)]
            case .free:
                tok = Self.sample(logits(), temperature: r.temperature, topK: r.topK, topP: r.topP, key: k2)
                    .item(Int.self)
            case .caption:
                tok = Self.sample(logits() + captionMask, temperature: r.temperature, topK: r.topK, topP: r.topP,
                                  key: k2).item(Int.self)
            case .codes:
                throw ACEPlannerError.generation("audio codes requested during the CoT phase")
            }
            fsm.update(tok)
            out.append(tok)
            try onProgress?(.thinking, out.count, r.maxCoTTokens)
            if tok == fsm.eos || tok == pad { break }
            pending.append(tok)
        }
        return out
    }

    /// Phase 2: exactly `Int(targetDuration·5)` audio codes with CFG (cond + uncond batch, left-padded), codes-only
    /// vocabulary, EOS forced at the target.
    func generateCodes(_ r: ACEPlanRequest, cot: String, targetDuration: Double, key: MLXArray,
                       onProgress: ((ACEPlanPhase, Int, Int) throws -> Void)?) throws -> [Int] {
        let target = Int(targetDuration * 5)
        guard target > 0 else { return [] }
        let cond = encode(ACEPlannerText.codesPrompt(caption: r.caption, lyrics: r.lyrics, cot: cot))
        let useCFG = r.cfgScale > 1
        let rows: [[Int]] = useCFG ? [cond, encode(ACEPlannerText.codesUnconditionalPrompt(negativePrompt: r.negativePrompt))]
            : [cond]
        let L = rows.map(\.count).max()!
        let pads = rows.map { L - $0.count }
        let B = rows.count
        var ids = [Int32]()
        for (row, p) in zip(rows, pads) { ids += [Int32](repeating: Int32(pad), count: p) + row.map(Int32.init) }

        // Additive masks: causal + padded keys hidden (padded queries see only themselves so rows stay finite).
        let total = L + target + 1
        var keyMask = [Float](repeating: 0, count: B * total)
        for b in 0..<B { for j in 0..<pads[b] { keyMask[b * total + j] = -.infinity } }
        let padRow = MLXArray(keyMask, [B, 1, 1, total])
        let ar = MLXArray(Array(Int32(0)..<Int32(L)))
        let causal = ar.expandedDimensions(axis: 0) .<= ar.expandedDimensions(axis: 1)  // [L(q), L(k)]
        let notPad = MLXArray(Array(0..<B).flatMap { b in (0..<L).map { $0 >= pads[b] } }, [B, 1, 1, L])
        let eye = ar.expandedDimensions(axis: 0) .== ar.expandedDimensions(axis: 1)
        let allowed = causal.reshaped(1, 1, L, L) .&& (notPad .|| eye.reshaped(1, 1, L, L))
        let prefillMask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))

        let lo = fsm.audioCodes.lowerBound
        let codeEmb = model.embedTokens[lo ..< lo + 64_000]
        let scale = r.cfgScale
        let cache = Qwen3Model.KVCache()
        func codeLogits(_ hidden: MLXArray) -> MLXArray {  // [B, D] → [64000] float32 (after CFG)
            let l = matmul(hidden, codeEmb.T).asType(.float32)
            if !useCFG { return l[0] }
            let c = l[0], u = l[1]
            return u + scale * (c - u)
        }
        var key = key
        var h = model(MLXArray(ids, [B, L]), cache: cache, mask: prefillMask)
        var codes: [Int] = []
        codes.reserveCapacity(target)
        var prev: MLXArray?
        for step in 0..<target {
            let (k1, k2) = MLXRandom.split(key: key)
            key = k1
            let tok = Self.sample(codeLogits(h[0..., -1]), temperature: r.temperature, topK: r.topK, topP: r.topP,
                                  key: k2).asType(.int32)
            if step + 1 < target {
                let input = broadcast(tok.reshaped(1, 1), to: [B, 1]) + Int32(lo)
                let T = cache.length + 1
                h = model(input, cache: cache, mask: padRow[0..., 0..., 0..., ..<T])
                asyncEval(h, tok)
            } else {
                asyncEval(tok)
            }
            if let p = prev {
                codes.append(p.item(Int.self))
                try onProgress?(.codes, codes.count, target)
            }
            prev = tok
        }
        if let p = prev { codes.append(p.item(Int.self)) }
        try onProgress?(.codes, codes.count, target)
        return codes
    }
}
