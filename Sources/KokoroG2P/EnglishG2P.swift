// Pure-Swift port of misaki's English G2P (`misaki.en.G2P`, Apache-2.0, https://github.com/hexgrad/misaki)
// for Kokoro-82M. spaCy is replaced by a port of its tokenizer plus NaturalLanguage-based POS tags,
// and misaki's (espeak / neural) out-of-vocabulary fallback by a rule-based letter-to-sound engine.
import Foundation

/// English text → Kokoro phonemes, matching misaki `en.G2P(british:)` as used by `KPipeline` for
/// lang codes "a" (American, voices `a*`) and "b" (British, voices `b*`).
///
/// Thread-safety: an instance keeps mutable state (an `NLTagger`, lazily loaded lexicon parts), so
/// use it from one thread at a time (or one instance per thread). Parsed lexicons are cached
/// process-wide and shared between instances; creating a second instance is cheap.
public final class EnglishG2P {
    public let british: Bool
    /// String used for words that cannot be pronounced (only possible with `useFallback == false`).
    /// Kokoro's pipeline uses "" (misaki's default is "❓").
    public var unknownMarker: String
    /// Use the rule-based letter-to-sound fallback for out-of-vocabulary words (default true).
    /// When false, unknown words get `unknownMarker`, exactly like misaki without a fallback.
    public var useFallback: Bool

    let lexicon: Lexicon
    private let tagger = POSTagger()
    private lazy var fallback = LetterToSound(lexicon: lexicon, british: british)

    /// Loads the gold lexicon (silver loads lazily on first miss; call `preload()` to load it now).
    public init(british: Bool = false, unknownMarker: String = "", useFallback: Bool = true) throws {
        self.british = british
        self.unknownMarker = unknownMarker
        self.useFallback = useFallback
        lexicon = try Lexicon(british: british)
        let golds = lexicon.golds
        tagger.isAbbreviation = { golds[$0] != nil }
    }

    /// Loads everything that is otherwise loaded lazily (silver lexicon, tokenizer tables).
    public func preload() {
        lexicon.preloadSilver()
        _ = SpacyTokenizer.shared
        _ = fallback
    }

    /// Phonemizes one text segment. Returns misaki's output string (tokens' phonemes joined with
    /// their original whitespace) and the tokens. Like `KPipeline`, callers should split long input
    /// on newlines first and phonemize each segment.
    public func phonemize(_ text: String) -> (phonemes: String, tokens: [Token]) {
        let tokens = run(text)
        var out = ""
        for tk in tokens { out += (tk.phonemes ?? unknownMarker) + tk.whitespace }
        return (out, tokens.map { $0.publicToken })
    }

    /// Phonemes joined the way `KPipeline.tokens_to_ps` does (single spaces, trimmed); this is the
    /// string Kokoro is actually fed.
    public static func pipelinePhonemes(_ tokens: [Token]) -> String {
        var s = ""
        for t in tokens { s += (t.phonemes ?? "") + (t.whitespace.isEmpty ? "" : " ") }
        return s.trimmingCharacters(in: .whitespaces)
    }

    /// How `fallbackPhonemes(for:mode:)` treats the lexicon (diagnostics / evaluation).
    public enum FallbackMode: Sendable {
        /// What `phonemize` does for an unknown word: spelling, morphology with the lexicon, then rules.
        case full
        /// Like `.full`, but the word itself is hidden from the lexicon (to evaluate on known words).
        case excludingWord
        /// Letter-to-sound rules only (no lexicon decomposition, no spelling of initialisms).
        case rulesOnly
    }

    /// The rule-based fallback pronunciation of a single word (bypassing the lexicon lookup).
    public func fallbackPhonemes(for word: String, mode: FallbackMode = .full) -> String? {
        let ps: String?
        switch mode {
        case .full: ps = fallback.phonemize(word)
        case .excludingWord:
            fallback.excludedWord = word.lowercased()
            ps = fallback.phonemize(word)
            fallback.excludedWord = nil
        case .rulesOnly: ps = fallback.rules(word.lowercased())
        }
        return ps.map { $0.replacingOccurrences(of: "ɾ", with: "T").replacingOccurrences(of: "ʔ", with: "t") }
    }

    // MARK: - Preprocessing (markdown-like links)

    enum Feature { case stress(Double); case phonemes(String); case numFlags(String) }

    private static let linkRE = PyRegex(#"\[([^\]]+)\]\(([^\)]*)\)"#)

    /// misaki `G2P.preprocess`: `[text](/phonemes/)`, `[text](-1)` stress, `[text](#flags#)` number flags.
    static func preprocess(_ text0: String) -> (String, [String], [Int: Feature]) {
        let text = String(text0.drop(while: { $0.unicodeScalars.allSatisfy { $0.pyIsSpace } }))
        let ns = text as NSString
        var result = ""
        var tokens: [String] = []
        var features: [Int: Feature] = [:]
        var lastEnd = 0
        func splitWS(_ s: String) -> [String] {
            s.split(whereSeparator: { $0.unicodeScalars.allSatisfy { $0.pyIsSpace } }).map(String.init)
        }
        for m in linkRE.re.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) {
            let before = ns.substring(with: NSRange(location: lastEnd, length: m.range.location - lastEnd))
            result += before
            tokens += splitWS(before)
            let label = ns.substring(with: m.range(at: 1))
            let f = ns.substring(with: m.range(at: 2))
            var feature: Feature? = nil
            let unsigned = (f.hasPrefix("-") || f.hasPrefix("+")) ? String(f.dropFirst()) : f
            if unsigned.isASCIIDigits, let v = Int(f) {
                feature = .stress(Double(v))
            } else if f == "0.5" || f == "+0.5" {
                feature = .stress(0.5)
            } else if f == "-0.5" {
                feature = .stress(-0.5)
            } else if f.pyCount > 1 && f.hasPrefix("/") && f.hasSuffix("/") {
                feature = .phonemes(("/" + String(f.dropFirst()).pyRStrip(["/"])).pyLStrip(["/"]))
            } else if f.pyCount > 1 && f.hasPrefix("#") && f.hasSuffix("#") {
                feature = .numFlags(String(f.dropFirst()).pyRStrip(["#"]))
            }
            if let feature { features[tokens.count] = feature }
            result += label
            tokens.append(label)
            lastEnd = m.range.location + m.range.length
        }
        if lastEnd < ns.length {
            let rest = ns.substring(from: lastEnd)
            result += rest
            tokens += splitWS(rest)
        }
        return (result, tokens, features)
    }

    // MARK: - Tokenization

    func tokenize(_ text: String, _ words: [String], _ features: [Int: Feature]) -> [MToken] {
        let raw = SpacyTokenizer.shared.tokenize(text)
        let tags = raw.isEmpty ? [] : tagger.tags(text: text, tokens: raw)
        let tokens = zip(raw, tags).map { MToken(text: $0.text, tag: $1, whitespace: $0.whitespace) }
        guard !features.isEmpty else { return tokens }
        // spaCy Alignment.from_strings(words, tokens): character alignment ignoring whitespace.
        // Map every non-space character of `text` to the index of the preprocess word it belongs to.
        var charToWord: [Int] = []  // per UTF-16 unit of `text` (whitespace → -1)
        let ns = text as NSString
        charToWord.reserveCapacity(ns.length)
        var w = 0
        var posInWord = 0
        let wordLens = words.map { ($0.filter { !$0.isWhitespace } as NSString).length }
        let units = Array(text.utf16)
        var idx = 0
        while idx < units.count {
            let u = units[idx]
            let isSpace = u == 0x20 || u == 0x0A || u == 0x09 || u == 0x0D || u == 0xA0 || u == 0x0B || u == 0x0C
            if isSpace { charToWord.append(-1); idx += 1; continue }
            while w < wordLens.count && posInWord >= wordLens[w] { w += 1; posInWord = 0 }
            charToWord.append(w < wordLens.count ? w : -1)
            posInWord += 1
            idx += 1
        }
        var data: [Int] = []  // flattened y2x
        for t in raw {
            let len = (t.text as NSString).length
            var xs: [Int] = []
            for c in t.start..<min(t.start + len, charToWord.count) where charToWord[c] >= 0 {
                if !xs.contains(charToWord[c]) { xs.append(charToWord[c]) }
            }
            data += xs.sorted()
        }
        for (k, v) in features.sorted(by: { $0.key < $1.key }) {
            var i = 0
            for (j, x) in data.enumerated() where x == k {
                defer { i += 1 }
                if j >= tokens.count { continue }
                switch v {
                case .stress(let s): tokens[j].stress = s
                case .phonemes(let p):
                    tokens[j].isHead = i == 0
                    tokens[j].phonemes = i == 0 ? p : ""
                    tokens[j].rating = 5
                case .numFlags(let f): tokens[j].numFlags = f
                }
            }
        }
        return tokens
    }

    func foldLeft(_ tokens: [MToken]) -> [MToken] {
        var result: [MToken] = []
        for tk in tokens {
            if !result.isEmpty && !tk.isHead {
                let prev = result.removeLast()
                result.append(mergeTokens([prev, tk], unk: unknownMarkerForMerge))
            } else {
                result.append(tk)
            }
        }
        return result
    }

    private var unknownMarkerForMerge: String { unknownMarker }

    private static let subtokenRE = PyRegex(#"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$"#)

    static let punctTags: Set<String> = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"]
    static let punctTagPhonemes: [String: String] = ["-LRB-": "(", "-RRB-": ")", "``": "\u{201C}", "\"\"": "\u{201D}", "''": "\u{201D}"]

    enum Word { case single(MToken); case group([MToken]) }

    /// misaki `G2P.retokenize`.
    func retokenize(_ tokens: [MToken]) -> [Word] {
        var words: [Word] = []
        var currency: String? = nil
        for (i, token) in tokens.enumerated() {
            var tks: [MToken]
            if token.alias == nil && token.phonemes == nil {
                tks = EnglishG2P.subtokenRE.findAll(token.text).map { t in
                    let x = MToken(text: t, tag: token.tag, whitespace: "")
                    x.isHead = true
                    x.numFlags = token.numFlags
                    x.stress = token.stress
                    x.prespace = false
                    return x
                }
                if tks.isEmpty { tks = [] }
            } else {
                tks = [token]
            }
            if tks.isEmpty { continue }  // (misaki would crash on tks[-1]; drop the token)
            tks[tks.count - 1].whitespace = token.whitespace
            for (j, tk) in tks.enumerated() {
                if tk.alias != nil || tk.phonemes != nil {
                } else if tk.tag == "$" && currencies[tk.text] != nil {
                    currency = tk.text
                    tk.phonemes = ""
                    tk.rating = 4
                } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                    tk.phonemes = "—"
                    tk.rating = 3
                } else if EnglishG2P.punctTags.contains(tk.tag) && !tk.text.unicodeScalars.allSatisfy({
                    let v = String($0).lowercased().unicodeScalars.first!.value
                    return v >= 97 && v <= 122
                }) {
                    tk.phonemes = EnglishG2P.punctTagPhonemes[tk.tag] ?? String(String.UnicodeScalarView(tk.text.unicodeScalars.filter { Phon.puncts.contains($0) }))
                    tk.rating = 4
                } else if currency != nil {
                    if tk.tag != "CD" {
                        currency = nil
                    } else if j + 1 == tks.count && (i + 1 == tokens.count || tokens[i + 1].tag != "CD") {
                        tk.currency = currency
                    }
                } else if j > 0 && j < tks.count - 1 && tk.text == "2" {
                    let a = tks[j - 1].text.unicodeScalars.last, b = tks[j + 1].text.unicodeScalars.first
                    if let a, let b, a.pyIsAlpha, b.pyIsAlpha { tk.alias = "to" }
                }
                if tk.alias != nil || tk.phonemes != nil {
                    words.append(.single(tk))
                } else if case .group(var g)? = words.last, g.last!.whitespace.isEmpty {
                    tk.isHead = false
                    g.append(tk)
                    words[words.count - 1] = .group(g)
                } else {
                    words.append(tk.whitespace.isEmpty ? .group([tk]) : .single(tk))
                }
            }
        }
        return words.map { if case .group(let g) = $0, g.count == 1 { return .single(g[0]) }; return $0 }
    }

    static func tokenContext(_ ctx: TokenContext, _ ps: String?, _ token: MToken) -> TokenContext {
        var vowel = ctx.futureVowel
        if let ps, !ps.isEmpty {
            for c in ps.unicodeScalars where Phon.vowels.contains(c) || Phon.consonants.contains(c) || Phon.nonQuotePuncts.contains(c) {
                vowel = Phon.nonQuotePuncts.contains(c) ? nil : Phon.vowels.contains(c)
                break
            }
        }
        let futureTo = token.text == "to" || token.text == "To" || (token.text == "TO" && (token.tag == "TO" || token.tag == "IN"))
        return TokenContext(futureVowel: vowel, futureTo: futureTo)
    }

    /// misaki `G2P.resolve_tokens`: spacing and stress inside a group of glued sub-tokens.
    static func resolveTokens(_ tokens: [MToken]) {
        var text = ""
        for tk in tokens.dropLast() { text += tk.text + tk.whitespace }
        text += tokens.last!.text
        var kinds = Set<Int>()
        for c in text.unicodeScalars where !Phon.subtokenJunks.contains(c) {
            kinds.insert(c.pyIsAlpha ? 0 : (c.isASCIIDigit ? 1 : 2))
        }
        let prespace = text.contains(" ") || text.contains("/") || kinds.count > 1
        for (i, tk) in tokens.enumerated() {
            if tk.phonemes == nil {
                if i == tokens.count - 1, tk.text.pyCount == 1, Phon.nonQuotePuncts.contains(tk.text.unicodeScalars.first!) {
                    tk.phonemes = tk.text
                    tk.rating = 3
                } else if tk.text.unicodeScalars.allSatisfy({ Phon.subtokenJunks.contains($0) }) {
                    tk.phonemes = ""
                    tk.rating = 3
                }
            } else if i > 0 {
                tk.prespace = prespace
            }
        }
        if prespace { return }
        var indices: [(Bool, Int, Int)] = []
        for (i, tk) in tokens.enumerated() {
            if let p = tk.phonemes, !p.isEmpty {
                indices.append((p.containsScalar(Phon.primary), Phon.stressWeight(p), i))
            }
        }
        if indices.count == 2 && tokens[indices[0].2].text.pyCount == 1 {
            let i = indices[1].2
            tokens[i].phonemes = Phon.applyStress(tokens[i].phonemes, -0.5)
            return
        } else if indices.count < 2 || indices.filter({ $0.0 }).count <= (indices.count + 1) / 2 {
            return
        }
        let sorted = indices.sorted { a, b in
            if a.0 != b.0 { return !a.0 }
            if a.1 != b.1 { return a.1 < b.1 }
            return a.2 < b.2
        }
        for (_, _, i) in sorted.prefix(indices.count / 2) {
            tokens[i].phonemes = Phon.applyStress(tokens[i].phonemes, -0.5)
        }
    }

    private func applyFallback(_ tk: MToken) -> (String?, Int?) {
        guard useFallback else { return (nil, nil) }
        let ps = fallback.phonemize(tk.alias ?? tk.text)
        return (ps, ps == nil ? nil : 1)
    }

    /// misaki `G2P.__call__`.
    func run(_ text0: String) -> [MToken] {
        let (text, words, features) = EnglishG2P.preprocess(text0)
        var tokens = tokenize(text, words, features)
        tokens = foldLeft(tokens)
        var ws = retokenize(tokens)
        var ctx = TokenContext()
        for wi in stride(from: ws.count - 1, through: 0, by: -1) {
            switch ws[wi] {
            case .single(let w):
                if w.phonemes == nil {
                    let (p, r) = lexicon(w.copy(), ctx)
                    w.phonemes = p
                    w.rating = r
                }
                if w.phonemes == nil {
                    let (p, r) = applyFallback(w)
                    w.phonemes = p
                    w.rating = r
                    w.isFallback = p != nil
                }
                ctx = EnglishG2P.tokenContext(ctx, w.phonemes, w)
            case .group(let w):
                var left = 0, right = w.count
                var shouldFallback = false
                while left < right {
                    let slice = Array(w[left..<right])
                    let tk: MToken? = slice.contains(where: { $0.alias != nil || $0.phonemes != nil }) ? nil : mergeTokens(slice, unk: nil)
                    let (ps, rating) = tk.map { lexicon($0, ctx) } ?? (nil, nil)
                    if let ps {
                        w[left].phonemes = ps
                        w[left].rating = rating
                        for x in w[(left + 1)..<right] {
                            x.phonemes = ""
                            x.rating = rating
                        }
                        ctx = EnglishG2P.tokenContext(ctx, ps, tk!)
                        right = left
                        left = 0
                    } else if left + 1 < right {
                        left += 1
                    } else {
                        right -= 1
                        let t = w[right]
                        if t.phonemes == nil {
                            if t.text.unicodeScalars.allSatisfy({ Phon.subtokenJunks.contains($0) }) {
                                t.phonemes = ""
                                t.rating = 3
                            } else {
                                // misaki always has a fallback here (unk-only mode = a fallback returning None).
                                shouldFallback = true
                                break
                            }
                        }
                        left = 0
                    }
                }
                if shouldFallback {
                    let merged = mergeTokens(w, unk: nil)
                    let (p, r) = applyFallback(merged)
                    w[0].phonemes = p
                    w[0].rating = r
                    w[0].isFallback = p != nil
                    for x in w.dropFirst() {
                        x.phonemes = ""
                        x.rating = r
                    }
                    // misaki doesn't update the context after a group fallback.
                } else {
                    EnglishG2P.resolveTokens(w)
                }
            }
        }
        var out: [MToken] = ws.map {
            switch $0 {
            case .single(let t): return t
            case .group(let g): return mergeTokens(g, unk: unknownMarker)
            }
        }
        ws = []
        for tk in out {
            if let p = tk.phonemes, !p.isEmpty {
                tk.phonemes = p.replacingOccurrences(of: "ɾ", with: "T").replacingOccurrences(of: "ʔ", with: "t")
            }
        }
        out = out.map { $0 }
        return out
    }
}
