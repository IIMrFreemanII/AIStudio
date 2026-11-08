import Foundation
import Tokenizers

/// What the sampler may pick for the next token.
enum ACETokenConstraint: Equatable {
    /// No constraint (full vocabulary).
    case free
    /// Only these tokens (whitelist; scores of the others become -inf).
    case only([Int])
    /// Caption text: everything except valid audio-code tokens and the backtick.
    case caption
    /// Audio codes only (valid range 0..<64000), with EOS also allowed when `allowEOS`.
    case codes(allowEOS: Bool)
}

/// Port of `acestep/constrained_logits_processor.py` `MetadataConstrainedLogitsProcessor` for the single-sequence
/// planner path (`skip_genres = True`). It forces the `<think>` scaffold and field names, restricts numeric / key /
/// language / time-signature values to valid token sequences (prefix trees built from the actual tokenization),
/// injects user-provided fields, lets the caption run free (no audio codes, no backticks, ≤ 512 tokens), stops at
/// `</think>` with EOS in the CoT phase, and only allows audio codes (EOS gated by the target count) in the codes
/// phase.
///
/// Known simplification: if the model spontaneously writes a `genres:` field after the caption, the reference
/// constrains it with `genres_vocab.txt`; here that value is generated freely up to the newline.
final class ACEConstrainedDecoder {
    enum State {
        case thinkTag, newlineAfterThink
        case bpmName, bpmValue, captionName, captionValue, durationName, durationValue
        case genresValue, keyscaleName, keyscaleValue, languageName, languageValue, timesigName, timesigValue
        case thinkEndTag, codesGeneration, completed
    }

    enum Phase { case cot, codes }

    static let validLanguages = [
        "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu",
        "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr",
        "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
    ]
    static let bpmRange = 30...300
    static let durationRange = 10...600
    static let timeSignatures = [2, 3, 4, 6]

    let tokenizer: Tokenizer
    let newline: Int
    let backtick: Int
    let eos: Int
    /// Valid audio-code token ids (`<|audio_code_0|>` … `<|audio_code_63999|>`, contiguous).
    let audioCodes: Range<Int>
    var digitTokens: [Int] = []
    var bpmTree: [[Int]: Set<Int>] = [:]
    var durationTree: [[Int]: Set<Int>] = [:]
    var timesigTree: [[Int]: Set<Int>] = [:]
    var keyscaleTree: [[Int]: Set<Int>] = [:]
    var languageTree: [[Int]: Set<Int>] = [:]
    private var fixedCache: [String: [Int]] = [:]
    private var decodeCache: [Int: String] = [:]

    // Per-generation settings.
    var phase: Phase = .cot
    var stopAtReasoning = false
    var skipCaption = false
    var skipLanguage = false
    var targetDuration: Double?
    var targetCodes: Int?
    var user: [String: String] = [:]
    var promptHasThinkEnd = false

    // Per-generation state.
    private(set) var state: State = .thinkTag
    private var position = 0
    private var accValue = ""
    private var accIDs: [Int] = []
    private(set) var codesCount = 0
    private var captionAfterNewline = false
    private var captionTokenCount = 0
    private var captionEnding = false
    private var pendingFieldName = ""
    private var queue: [Int] = []
    private var currentUserField: String?

    init(tokenizer: Tokenizer) throws {
        self.tokenizer = tokenizer
        func last(_ s: String) -> Int { tokenizer.encode(text: s, addSpecialTokens: false).last ?? -1 }
        newline = last("\n")
        backtick = last("`")
        eos = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<|im_end|>") ?? 151_645
        guard let c0 = tokenizer.convertTokenToId("<|audio_code_0|>"),
              tokenizer.convertTokenToId("<|audio_code_63999|>") == c0 + 63_999 else {
            throw ACEPlannerError.tokenizer("audio-code tokens missing or not contiguous")
        }
        audioCodes = c0 ..< c0 + 64_000
        digitTokens = (0...9).map { last(String($0)) }
        bpmTree = buildTree(Self.bpmRange.map(String.init), field: "bpm")
        durationTree = buildTree(Self.durationRange.map(String.init), field: "duration")
        timesigTree = buildTree(Self.timeSignatures.map(String.init), field: "timesignature")
        languageTree = buildTree(Self.validLanguages, field: "language")
        var keys: [String] = []
        for n in ["A", "B", "C", "D", "E", "F", "G"] {
            for a in ["", "#", "b", "♯", "♭"] { for m in ["major", "minor"] { keys.append("\(n)\(a) \(m)") } }
        }
        keyscaleTree = buildTree(keys, field: "keyscale") { [tokenizer] ids in
            let first = tokenizer.decode(tokens: [ids[0]]).drop(while: { $0.isWhitespace })
            return first.first.map { "ABCDEFG".contains($0.uppercased()) } ?? false
        }
    }

    /// `_build_*_prefix_tree`: token-id prefix → allowed next tokens, newline after a complete value.
    private func buildTree(_ values: [String], field: String, accept: (([Int]) -> Bool)? = nil) -> [[Int]: Set<Int>] {
        let ctx = tokenizer.encode(text: field + ":", addSpecialTokens: false)
        var tree: [[Int]: Set<Int>] = [:]
        for v in values {
            let full = tokenizer.encode(text: field + ": " + v, addSpecialTokens: false)
            guard full.count >= ctx.count, Array(full[..<ctx.count]) == ctx else { continue }
            let ids = Array(full[ctx.count...])
            guard !ids.isEmpty else { continue }
            if let accept, !accept(ids) { continue }
            for i in 0...ids.count {
                let p = Array(ids[..<i])
                tree[p, default: []].insert(i < ids.count ? ids[i] : newline)
            }
        }
        return tree
    }

    func decode(_ id: Int) -> String {
        if let s = decodeCache[id] { return s }
        let s = audioCodes.contains(id) ? "<|audio_code_\(id - audioCodes.lowerBound)|>" : tokenizer.decode(tokens: [id])
        decodeCache[id] = s
        return s
    }

    func reset() {
        state = .thinkTag
        position = 0
        accValue = ""
        accIDs = []
        codesCount = 0
        queue = []
        currentUserField = nil
        captionAfterNewline = false
        captionTokenCount = 0
        captionEnding = false
        pendingFieldName = ""
    }

    func setTargetDuration(_ d: Double?) {
        targetDuration = d
        targetCodes = d.flatMap { $0 > 0 ? Int($0 * 5) : nil }
    }

    // MARK: - Transitions

    private static let fixedStrings: [State: String] = [
        .thinkTag: "<think>", .newlineAfterThink: "\n", .bpmName: "bpm:", .captionName: "caption:",
        .durationName: "duration:", .keyscaleName: "keyscale:", .languageName: "language:",
        .timesigName: "timesignature:", .thinkEndTag: "</think>",
    ]

    private func nextFieldState(after field: String) -> State {
        let order = ["bpm", "caption", "duration", "genres", "keyscale", "language", "timesignature"]
        let toState: [String: State] = [
            "bpm": .bpmName, "caption": .captionName, "duration": .durationName, "keyscale": .keyscaleName,
            "language": .languageName, "timesignature": .timesigName,
        ]
        guard let i = order.firstIndex(of: field) else { return .thinkEndTag }
        for f in order[(i + 1)...] {
            if f == "genres" { continue }  // skip_genres = True
            if f == "caption" && skipCaption { continue }
            if f == "language" && skipLanguage { continue }
            return toState[f]!
        }
        return .thinkEndTag
    }

    private func nextState(_ s: State) -> State? {
        switch s {
        case .thinkTag: return .newlineAfterThink
        case .newlineAfterThink: return .bpmName
        case .bpmName: return .bpmValue
        case .bpmValue: return nextFieldState(after: "bpm")
        case .captionName: return skipCaption ? nil : .captionValue
        case .captionValue: return skipCaption ? nil : nextFieldState(after: "caption")
        case .durationName: return .durationValue
        case .durationValue: return nextFieldState(after: "duration")
        case .genresValue: return nextFieldState(after: "genres")
        case .keyscaleName: return .keyscaleValue
        case .keyscaleValue: return nextFieldState(after: "keyscale")
        case .languageName: return skipLanguage ? nil : .languageValue
        case .languageValue: return skipLanguage ? nil : nextFieldState(after: "language")
        case .timesigName: return .timesigValue
        case .timesigValue: return .thinkEndTag
        case .thinkEndTag: return .codesGeneration
        case .codesGeneration: return .completed
        case .completed: return nil
        }
    }

    private func enter(_ s: State) {
        state = s
        position = 0
        accValue = ""
        accIDs = []
    }

    private func transition() {
        guard let n = nextState(state) else { return }
        enter(n)
        captionAfterNewline = false
        captionTokenCount = 0
        captionEnding = false
        pendingFieldName = ""
    }

    /// `_get_allowed_tokens_for_fixed_string`: the longest prefix of the remaining text that is one token.
    private func fixedTokens(_ fixed: String) -> [Int] {
        let rem = Array(fixed.unicodeScalars.dropFirst(position))
        if rem.isEmpty { return [] }
        let key = String(String.UnicodeScalarView(rem))
        if let c = fixedCache[key] { return c }
        var result: [Int] = []
        for end in stride(from: rem.count, through: 1, by: -1) {
            let t = tokenizer.encode(text: String(String.UnicodeScalarView(rem[..<end])), addSpecialTokens: false)
            if t.count == 1 { result = t; break }
        }
        if result.isEmpty, let f = tokenizer.encode(text: key, addSpecialTokens: false).first { result = [f] }
        fixedCache[key] = result
        return result
    }

    /// The tokens a fixed-string state emits from start to end (diagnostics).
    func forcedTokens(for fixed: String) -> [Int] {
        let saved = position
        defer { position = saved }
        position = 0
        var out: [Int] = []
        while position < fixed.unicodeScalars.count, let t = fixedTokens(fixed).first {
            out.append(t)
            position += decode(t).unicodeScalars.count
        }
        return out
    }

    // MARK: - Constraint for the next token

    /// The constraint for the next token. `top` returns the argmax of the raw scores; `best` the highest-scoring
    /// token among candidates (both only evaluated when the FSM needs them).
    func constraint(top: () -> Int, best: ([Int]) -> Int) -> ACETokenConstraint {
        if state == .completed { return .free }
        if phase == .codes && state == .thinkTag && promptHasThinkEnd {
            state = .codesGeneration
            codesCount = 0
        }
        if state == .codesGeneration {
            guard let target = targetCodes else { return .codes(allowEOS: true) }
            return codesCount < target ? .codes(allowEOS: false) : .only([eos])
        }
        return single(top: top, best: best)
    }

    private func inject(_ field: String) -> ACETokenConstraint? {
        guard let value = user[field], queue.isEmpty else { return nil }
        let toks = tokenizer.encode(text: " \(value)\n", addSpecialTokens: false)
        guard let first = toks.first else { return nil }
        queue = toks
        currentUserField = field
        return .only([first])
    }

    private func treeConstraint(_ tree: [[Int]: Set<Int>]) -> ACETokenConstraint {
        let next = tree[accIDs] ?? []
        if next.contains(newline) { return .only([newline]) }
        return .only(next.isEmpty ? [newline] : next.sorted())
    }

    private func single(top: () -> Int, best: ([Int]) -> Int) -> ACETokenConstraint {
        if let q = queue.first { return .only([q]) }
        if let fixed = Self.fixedStrings[state] {
            let allowed = fixedTokens(fixed)
            if state == .thinkEndTag && stopAtReasoning && (allowed.isEmpty || fixed.unicodeScalars.count - position <= 10) {
                return .only([eos])
            }
            if !allowed.isEmpty { return .only(allowed) }
            transition()
            if Self.fixedStrings[state] != nil { return .free }
            return single(top: top, best: best)
        }
        switch state {
        case .bpmValue, .durationValue:
            let field = state == .bpmValue ? "bpm" : "duration"
            if accIDs.isEmpty, let c = inject(field) { return c }
            if state == .durationValue, let d = targetDuration {
                let target = Array(String(Int(d)))
                if accValue.count < target.count, let digit = Int(String(target[accValue.count])) {
                    return .only([digitTokens[digit]])
                }
                return .only([newline])
            }
            let tree = state == .bpmValue ? bpmTree : durationTree
            var allowed = tree[accIDs] ?? []
            if allowed.contains(newline) { allowed.insert(newline) }
            return .only(allowed.sorted())
        case .captionValue:
            if accValue.isEmpty, let c = inject("caption") { return c }
            if captionAfterNewline {
                let t = decode(top())
                captionAfterNewline = false
                if let f = t.unicodeScalars.first, f != " " && f != "\t" {
                    captionEnding = true
                    pendingFieldName = ""
                    return .free
                }
            }
            if captionEnding { return .free }
            if captionTokenCount >= 512 { return .only([newline]) }
            return .caption
        case .genresValue:
            return .free
        case .keyscaleValue:
            if accIDs.isEmpty, let c = inject("keyscale") { return c }
            return treeConstraint(keyscaleTree)
        case .languageValue:
            if accIDs.isEmpty, let c = inject("language") { return c }
            if accIDs.isEmpty {
                let cands = (languageTree[[]] ?? []).sorted()
                return cands.isEmpty ? .only([newline]) : .only([best(cands)])
            }
            return treeConstraint(languageTree)
        case .timesigValue:
            if accIDs.isEmpty, let c = inject("timesignature") { return c }
            let next = timesigTree[accIDs] ?? []
            return next.contains(newline) ? .only([newline]) : .only(next.sorted())
        default:
            return .free
        }
    }

    // MARK: - State update after sampling

    func update(_ token: Int) {
        if state == .completed { return }
        if state == .codesGeneration { codesCount += 1; return }
        if !queue.isEmpty {
            queue.removeFirst()
            if queue.isEmpty, let field = currentUserField {
                currentUserField = nil
                enter(nextFieldState(after: field))
            }
            return
        }
        let s = decode(token)
        if let fixed = Self.fixedStrings[state] {
            position += s.unicodeScalars.count
            if position >= fixed.unicodeScalars.count {
                if state == .thinkEndTag && stopAtReasoning { enter(.completed) } else { transition() }
            }
            return
        }
        switch state {
        case .bpmValue, .durationValue, .timesigValue:
            if token == newline { transition(); return }
            accIDs.append(token)
            let d = s.trimmingCharacters(in: .whitespacesAndNewlines)
            if !d.isEmpty, d.allSatisfy(\.isNumber) { accValue += d }
        case .genresValue:
            if token == newline { transition(); return }
            accValue += s
        case .captionValue:
            captionTokenCount += 1
            accValue += s
            captionAfterNewline = s.contains("\n")
            if captionEnding {
                pendingFieldName += s
                if s.contains(":") {
                    var name = pendingFieldName.trimmingCharacters(in: .whitespacesAndNewlines)
                    while name.hasSuffix(":") { name.removeLast() }
                    name = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    let map: [String: State] = [
                        "duration": .durationValue, "genres": .genresValue, "keyscale": .keyscaleValue,
                        "language": .languageValue, "timesignature": .timesigValue,
                    ]
                    captionEnding = false
                    pendingFieldName = ""
                    if let st = map[name] { enter(st) } else { transition() }
                }
            }
        case .keyscaleValue, .languageValue:
            if token == newline { transition(); return }
            accIDs.append(token)
            accValue += s
        default:
            break
        }
    }
}
