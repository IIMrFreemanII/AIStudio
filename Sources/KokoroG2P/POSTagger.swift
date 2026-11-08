// Part-of-speech tags for misaki without spaCy: Apple's NaturalLanguage lexical classes mapped to
// the fine-grained Penn Treebank tags misaki branches on, plus rules for the cases that change a
// pronunciation (read/wound/used, "in"/"by" as particles, "that" as determiner, punctuation, …).
import Foundation
import NaturalLanguage

final class POSTagger {
    private let tagger = NLTagger(tagSchemes: [.lexicalClass])
    /// Returns true for abbreviations spelled with a final dot that the lexicon knows ("etc.").
    var isAbbreviation: ((String) -> Bool)? = nil

    static let modals: Set<String> = ["will", "would", "can", "could", "shall", "should", "may", "might", "must",
                                      "'ll", "’ll", "ca", "wo", "sha", "ought", "'d", "’d"]
    static let doForms: Set<String> = ["do", "does", "did"]
    static let haveForms: Set<String> = ["have", "has", "had", "having", "'ve", "’ve"]
    static let beForms: Set<String> = ["be", "is", "are", "was", "were", "been", "being", "am", "'s", "'re", "'m",
                                       "’s", "’re", "’m", "get", "gets", "got", "getting", "gotten", "become", "became"]
    static let skipAdverbs: Set<String> = ["not", "n't", "n’t", "never", "also", "always", "just", "really", "often",
                                           "already", "ever", "still", "even", "only", "probably", "actually", "all",
                                           "both", "usually", "sometimes", "finally", "recently", "once", "rarely",
                                           "seldom", "hardly", "barely", "certainly", "definitely", "simply", "carefully",
                                           "quickly", "slowly", "first", "then", "now", "nt"]
    static let subjectPronouns: Set<String> = ["i", "you", "we", "they"]
    static let possessivePronouns: Set<String> = ["my", "your", "his", "her", "its", "our", "their", "thy"]
    static let whPronouns: Set<String> = ["who", "whom", "what", "whoever", "whatever"]
    static let whDeterminers: Set<String> = ["which", "whichever", "whose"]
    static let whAdverbs: Set<String> = ["how", "when", "where", "why", "whenever", "wherever"]
    static let coordinators: Set<String> = ["and", "or", "but", "nor", "yet", "plus", "&"]
    static let numberWords: Set<String> = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                                           "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                                           "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
                                           "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
                                           "million", "billion", "trillion", "dozen"]
    static let irregularPast: Set<String> = ["was", "were", "had", "did", "went", "came", "saw", "said", "made", "took",
                                             "got", "gave", "found", "thought", "told", "became", "left", "felt",
                                             "brought", "began", "kept", "held", "wrote", "stood", "heard", "let",
                                             "meant", "set", "met", "ran", "paid", "sat", "spoke", "lay", "led",
                                             "grew", "lost", "fell", "sent", "built", "understood", "drew", "broke",
                                             "spent", "cut", "rose", "drove", "bought", "wore", "chose", "sang",
                                             "won", "ate", "knew", "flew", "threw", "caught", "taught", "fought",
                                             "sold", "slept", "swam", "woke", "hid", "shook", "rode", "forgot",
                                             "read", "wound", "put", "hit", "hurt", "beat", "bit", "dug", "fed",
                                             "fled", "hung", "lit", "shot", "shut", "slid", "spun", "stuck", "struck",
                                             "swept", "swung", "tore", "wept", "bent", "bled", "bound", "bred",
                                             "clung", "crept", "dealt", "dove", "dwelt", "forbade", "froze", "ground",
                                             "knelt", "leapt", "lent", "sought", "sank", "shone", "sprang", "stank",
                                             "stole", "stung", "strove", "swore", "thrust", "trod", "wove", "wrung"]
    static let irregularParticiples: Set<String> = ["been", "done", "gone", "seen", "taken", "given", "known", "written",
                                                    "spoken", "broken", "chosen", "driven", "eaten", "fallen",
                                                    "forgotten", "frozen", "hidden", "ridden", "risen", "shaken",
                                                    "stolen", "sworn", "thrown", "torn", "worn", "woken", "grown",
                                                    "drawn", "flown", "shown", "begun", "sung", "swum", "drunk",
                                                    "run", "become", "come", "gotten", "proven", "beaten", "bitten"]

    private struct NLTok { let start: Int; let end: Int; let cls: String }

    /// Returns one Penn tag per raw token (`text` is the string the tokens were cut from).
    func tags(text: String, tokens: [RawToken]) -> [String] {
        tagger.string = text
        var nl: [NLTok] = []
        let ns = text as NSString
        tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .word, scheme: .lexicalClass,
                             options: [.omitWhitespace]) { tag, r in
            let nr = NSRange(r, in: text)
            nl.append(NLTok(start: nr.location, end: nr.location + nr.length, cls: tag?.rawValue ?? ""))
            return true
        }
        _ = ns
        // Map each raw token to the NL token covering its first character (fallback: nearest overlap).
        var classes: [String] = []
        var k = 0
        for t in tokens {
            while k < nl.count && nl[k].end <= t.start { k += 1 }
            if k < nl.count && nl[k].start <= t.start {
                classes.append(nl[k].cls)
            } else {
                classes.append("")
            }
        }
        var tags = [String](repeating: "", count: tokens.count)
        for i in tokens.indices {
            tags[i] = baseTag(tokens, classes, i)
        }
        contextualFixes(tokens, classes, &tags)
        return tags
    }

    private static func isOpenQuoteContext(_ tokens: [RawToken], _ i: Int) -> Bool {
        if i == 0 { return true }
        let p = tokens[i - 1]
        if !p.whitespace.isEmpty || p.isSpace { return true }
        if let c = p.text.unicodeScalars.last, "([{“‘«—–-".unicodeScalars.contains(c) { return true }
        // A quote glued to the following word but not the previous one is an opening quote.
        if i + 1 < tokens.count && tokens[i].whitespace.isEmpty {
            if let n = tokens[i + 1].text.unicodeScalars.first, n.pyIsAlpha || n.isASCIIDigit {
                return p.text.unicodeScalars.last.map { !($0.pyIsAlpha || $0.isASCIIDigit) } ?? true
            }
        }
        return false
    }

    private func baseTag(_ tokens: [RawToken], _ classes: [String], _ i: Int) -> String {
        let text = tokens[i].text
        let lower = text.lowercased()
        let sc = text.scalars
        if tokens[i].isSpace { return "_SP" }

        // Punctuation and symbols.
        switch text {
        case ".", "!", "?", "?!", "!?", "!!", "??", "!!!", "???": return "."
        case ",": return ","
        case ":", ";", "...", "…", "—", "–", "--", "---", "……": return ":"
        case "(", "[", "{": return "-LRB-"
        case ")", "]", "}": return "-RRB-"
        case "“", "«", "„": return "``"
        case "”", "»": return "''"
        case "\"", "'", "‘", "’", "`", "´", "''", "``":
            if text == "''" { return "''" }
            if text == "``" || text == "‘" && POSTagger.isOpenQuoteContext(tokens, i) { return "``" }
            return POSTagger.isOpenQuoteContext(tokens, i) ? "``" : "''"
        case "$", "£", "€", "¥", "US$", "C$", "A$", "₹", "₽", "₩", "฿": return "$"
        case "#": return "#"
        case "%": return "NN"
        case "&": return "CC"
        case "-":
            let prevGlued = i > 0 && tokens[i - 1].whitespace.isEmpty && !tokens[i - 1].isSpace
            let nextGlued = tokens[i].whitespace.isEmpty && i + 1 < tokens.count
            if prevGlued && nextGlued {
                let p = tokens[i - 1].text.unicodeScalars.last!, n = tokens[i + 1].text.unicodeScalars.first!
                if p.isASCIIDigit && n.isASCIIDigit { return "SYM" }
                return "HYPH"
            }
            return ":"
        case "+", "=", "<", ">", "^", "|", "~", "/", "\\", "@", "°": return "SYM"
        case "*", "**", "***", "•", "·": return "NFP"
        default: break
        }
        if sc.allSatisfy({ !$0.pyIsAlpha && !$0.isASCIIDigit }) {
            if sc.allSatisfy({ ".!?".unicodeScalars.contains($0) }) { return "." }
            if sc.allSatisfy({ Phon.puncts.contains($0) || "-–—".unicodeScalars.contains($0) }) { return ":" }
            return "NFP"
        }
        // Numbers.
        if sc.first!.isASCIIDigit || (sc.first == "." && sc.count > 1 && sc[1].isASCIIDigit) {
            if sc.allSatisfy({ $0.isASCIIDigit || $0 == "," || $0 == "." || $0 == ":" }) { return "CD" }
            if lower.hasSuffix("s") && sc.dropLast().allSatisfy({ $0.isASCIIDigit || $0 == "'" || $0 == "’" }) { return "NNS" }
            if ["st", "nd", "rd", "th"].contains(where: { lower.hasSuffix($0) }) { return "JJ" }
            if sc.allSatisfy({ $0.isASCIIDigit || ",.:'’\"/-".unicodeScalars.contains($0) }) { return "CD" }
            return "NN"
        }
        // Contraction pieces produced by the tokenizer exceptions.
        switch lower {
        case "n't", "n’t", "nt": return "RB"
        case "'ll", "’ll": return "MD"
        case "'d", "’d": return "MD"
        case "'re", "’re", "'m", "’m": return "VBP"
        case "'ve", "’ve": return "VB"
        case "'s", "’s":
            let prevCls = i > 0 ? classes[i - 1] : ""
            if prevCls == "Pronoun" || (i > 0 && ["it", "he", "she", "that", "what", "who", "there", "here", "where", "let", "how"].contains(tokens[i - 1].text.lowercased())) {
                return (i > 0 && tokens[i - 1].text.lowercased() == "let") ? "PRP" : "VBZ"
            }
            return "POS"
        case "'", "’": return "POS"
        case "ca", "wo", "sha": if classes[i] == "Verb" || (i + 1 < tokens.count && tokens[i + 1].text.lowercased().hasPrefix("n")) { return "MD" }
        case "gon": return "VBG"
        case "na", "ta": if i > 0 && ["gon", "got", "wan"].contains(tokens[i - 1].text.lowercased()) { return "TO" }
        case "'em", "’em": return "PRP"
        default: break
        }
        if lower == "to" { return "TO" }
        if sc.count == 2 && sc[1] == "." && sc[0].pyIsAlpha { return "NNP" }
        let isCap = sc.first!.isUppercaseLetter
        let allCaps = sc.count >= 2 && text == text.uppercased() && sc.contains(where: { $0.pyIsAlpha })
        let cls = classes[i]
        switch cls {
        case "Noun":
            if text == "I" { return "PRP" }
            if allCaps || (isCap && !POSTagger.isSentenceStart(tokens, i)) {
                return lower.hasSuffix("s") && !allCaps && sc.count > 3 ? "NNPS" : "NNP"
            }
            if POSTagger.numberWords.contains(lower) { return "CD" }
            return lower.hasSuffix("s") && !lower.hasSuffix("ss") && sc.count > 3 ? "NNS" : "NN"
        case "Verb": return verbTag(tokens, classes, i)
        case "Adjective":
            if lower.hasSuffix("est") && sc.count > 5 { return "JJS" }
            return "JJ"
        case "Adverb":
            if POSTagger.whAdverbs.contains(lower) { return "WRB" }
            return "RB"
        case "Pronoun":
            if POSTagger.possessivePronouns.contains(lower) {
                // "her" is PRP$ before a noun phrase, PRP otherwise.
                if lower == "her" || lower == "his" {
                    let n = i + 1 < tokens.count ? classes[i + 1] : ""
                    return ["Noun", "Adjective", "Number"].contains(n) ? "PRP$" : (lower == "his" ? "PRP$" : "PRP")
                }
                return "PRP$"
            }
            if POSTagger.whPronouns.contains(lower) { return "WP" }
            if POSTagger.whDeterminers.contains(lower) { return "WDT" }
            if ["that", "this", "these", "those"].contains(lower) { return "DT" }
            if ["everyone", "everything", "someone", "something", "anyone", "anything", "nobody", "nothing",
                "everybody", "somebody", "anybody", "none"].contains(lower) { return "NN" }
            return "PRP"
        case "Determiner":
            if POSTagger.whDeterminers.contains(lower) { return "WDT" }
            if lower == "all" || lower == "both" || lower == "half" {
                if i + 1 < tokens.count && ["Determiner", "Pronoun"].contains(classes[i + 1]) { return "PDT" }
                return "DT"
            }
            if POSTagger.possessivePronouns.contains(lower) { return "PRP$" }
            return "DT"
        case "Particle":
            return "RP"
        case "Preposition":
            if lower == "that" { return "IN" }
            return "IN"
        case "Conjunction":
            if POSTagger.coordinators.contains(lower) { return "CC" }
            if lower == "so" { return "RB" }
            return "IN"
        case "Interjection": return "UH"
        case "Number": return "CD"
        case "Classifier", "Idiom": return "NN"
        default:
            if allCaps || isCap { return "NNP" }
            return "NN"
        }
    }

    static func isSentenceStart(_ tokens: [RawToken], _ i: Int) -> Bool {
        var j = i - 1
        while j >= 0 {
            let t = tokens[j].text
            if tokens[j].isSpace { j -= 1; continue }
            if ["\"", "“", "‘", "'", "(", "[", "«", "``", "-", "—", "–"].contains(t) { j -= 1; continue }
            return [".", "!", "?", ":", ";", "…", "..."].contains(t)
        }
        return true
    }

    /// Index of the previous word, skipping light adverbs, or nil at a clause boundary.
    private func previousWord(_ tokens: [RawToken], _ i: Int, skipAdverbs: Bool = true) -> Int? {
        var j = i - 1
        while j >= 0 {
            let t = tokens[j].text.lowercased()
            if tokens[j].isSpace { j -= 1; continue }
            if skipAdverbs && POSTagger.skipAdverbs.contains(t) { j -= 1; continue }
            if let c = t.unicodeScalars.first, !c.pyIsAlpha && !c.isASCIIDigit && c != "'" && c != "’" { return nil }
            return j
        }
        return nil
    }

    private func verbTag(_ tokens: [RawToken], _ classes: [String], _ i: Int) -> String {
        let lower = tokens[i].text.lowercased()
        if POSTagger.modals.contains(lower) { return "MD" }
        let prevIdx = previousWord(tokens, i)
        let prev = prevIdx.map { tokens[$0].text.lowercased() }
        if lower.hasSuffix("ing") && lower.count > 4 { return "VBG" }
        if ["is", "has", "does", "'s", "’s"].contains(lower) { return "VBZ" }
        if ["am", "are", "'re", "'m", "’re", "’m"].contains(lower) { return "VBP" }
        if lower == "be" { return "VB" }
        let pastLike = lower.hasSuffix("ed") || POSTagger.irregularPast.contains(lower) || POSTagger.irregularParticiples.contains(lower)
        let baseContext: Bool = {
            guard let prev else { return true }  // sentence start: imperative
            return POSTagger.modals.contains(prev) || prev == "to" || POSTagger.doForms.contains(prev)
                || prev == "please" || prev == "let" || prev == "let's" || prev == "'s" && prevIdx! > 0 && tokens[prevIdx! - 1].text.lowercased() == "let"
                || prev == "ca" || prev == "wo"
        }()
        if pastLike && !(lower == "read" || lower == "wound" || lower == "set" || lower == "put" || lower == "cut"
                         || lower == "let" || lower == "hit" || lower == "hurt" || lower == "run" || lower == "come"
                         || lower == "become" || lower == "beat" || lower == "shut" || lower == "spread" || lower == "cost") || ((lower == "read" || lower == "wound") && !baseContext) {
            if let prev, POSTagger.haveForms.contains(prev) || POSTagger.beForms.contains(prev) { return "VBN" }
            if POSTagger.irregularParticiples.contains(lower) && !POSTagger.irregularPast.contains(lower) { return "VBN" }
            if lower == "read" || lower == "wound" {
                if let prev, let pi = prevIdx, pi > 0, POSTagger.subjectPronouns.contains(prev) || prev == "he" || prev == "she" || prev == "it" {
                    let pp = tokens[pi - 1].text.lowercased()
                    if POSTagger.doForms.contains(pp) || POSTagger.modals.contains(pp) || POSTagger.haveForms.contains(pp) { return "VB" }
                }
                if let prev, POSTagger.subjectPronouns.contains(prev) { return "VBP" }
            }
            return "VBD"
        }
        if baseContext { return "VB" }
        if lower.hasSuffix("s") && !lower.hasSuffix("ss") && lower.count > 2 { return "VBZ" }
        if let prev, POSTagger.subjectPronouns.contains(prev) { return "VBP" }
        return "VBP"
    }

    private func contextualFixes(_ tokens: [RawToken], _ classes: [String], _ tags: inout [String]) {
        let n = tokens.count
        func isWordTok(_ j: Int) -> Bool {
            guard let c = tokens[j].text.unicodeScalars.first else { return false }
            return c.pyIsAlpha || c.isASCIIDigit || ((c == "'" || c == "’") && tokens[j].text.count > 1)
        }
        func nextIdx(_ i: Int) -> Int? {
            var j = i + 1
            while j < n { if !tokens[j].isSpace { return j }; j += 1 }
            return nil
        }
        func prevIdx(_ i: Int, skipAdverbs: Bool = false) -> Int? {
            var j = i - 1
            while j >= 0 {
                if tokens[j].isSpace || (skipAdverbs && POSTagger.skipAdverbs.contains(tokens[j].text.lowercased())) { j -= 1; continue }
                return j
            }
            return nil
        }
        func lowerAt(_ j: Int?) -> String { j.map { tokens[$0].text.lowercased() } ?? "" }
        func tagAt(_ j: Int?) -> String { j.map { tags[$0] } ?? "" }
        func isPunctAt(_ j: Int?) -> Bool { guard let j else { return true }; return !isWordTok(j) }
        let objectPronouns: Set<String> = ["it", "me", "him", "them", "us", "her", "you", "this", "that", "these", "those"]
        let npStartTags: Set<String> = ["DT", "PDT", "PRP$", "CD", "WP", "WDT"]
        let particleNext: Set<String> = ["here", "there", "now", "then", "again", "too", "also", "together", "back",
                                         "anymore", "yet", "already", "soon", "later"]
        var openSingle = false
        for i in 0..<n {
            let text = tokens[i].text
            let lower = text.lowercased()
            let p = prevIdx(i), nx = nextIdx(i)
            // Apostrophes: closing single quote vs possessive ("dogs' owners").
            if text == "'" || text == "’" || text == "‘" {
                if tags[i] == "``" { openSingle = true; continue }
                if tags[i] == "''" {
                    if openSingle { openSingle = false; continue }
                    if let p, tokens[p].whitespace.isEmpty, let c = tokens[p].text.unicodeScalars.last, c.pyIsAlpha {
                        tags[i] = "POS"
                    }
                }
                continue
            }
            // "etc." / "approx." glued to a following lowercase word: spaCy keeps the dot in the word.
            if text == ".", let p, tokens[p].whitespace.isEmpty, isWordTok(p), let nx, !tokens[i].whitespace.isEmpty,
               let c = tokens[nx].text.unicodeScalars.first, c.isLowercaseLetter || c.isASCIIDigit,
               let isAbbreviation, isAbbreviation(tokens[p].text + ".") {
                tags[i] = "FW"
                continue
            }
            guard isWordTok(i) else { continue }
            let cls = classes[i]
            let sentStart = POSTagger.isSentenceStart(tokens, i)
            switch lower {
            case "a" where text == "a":
                tags[i] = "DT"
            case "a":
                let fn: Set<String> = ["to", "and", "or", "is", "was", "of", "in", "on", "at", "for", "with", "by", "from", "are", "were"]
                if let nx, isWordTok(nx), !fn.contains(lowerAt(nx)) { tags[i] = "DT" } else { tags[i] = "NNP" }
            case "that":
                let pl = lowerAt(p), nl = lowerAt(nx), nt = tagAt(nx)
                if isPunctAt(nx) || ["which", "who", "whom"].contains(nl) {
                    tags[i] = "DT"
                } else if ["i", "we", "they", "he", "she", "you"].contains(nl) {
                    tags[i] = "IN"
                } else if ["IN", "TO"].contains(tagAt(p)) || ["like", "of", "about"].contains(pl) {
                    tags[i] = "DT"
                } else if let p, ["NN", "NNS", "NNP", "NNPS", "JJS", "RBS"].contains(tags[p]) || ["those", "one", "ones", "all"].contains(pl) {
                    tags[i] = "WDT"
                } else if ["PRP", "PRP$", "DT", "EX"].contains(nt) || nl == "there" || ["IN", "VBN", "VBG", "CD"].contains(nt) {
                    tags[i] = sentStart && ["DT", "CD"].contains(nt) ? "DT" : "IN"
                    if nl == "it" || nl == "this" { tags[i] = "IN" }
                } else if nt.hasPrefix("VB") || nt == "MD" {
                    tags[i] = "DT"
                } else if sentStart {
                    tags[i] = "DT"
                } else {
                    // "know that man." (DT) vs "know that inner wisdom is …" (IN): look for a verb ahead.
                    var j = nx
                    var foundVerb = false
                    var steps = 0
                    while let jj = j, steps < 5, isWordTok(jj) {
                        if jj != nx && (tags[jj].hasPrefix("VB") || tags[jj] == "MD") { foundVerb = true; break }
                        j = nextIdx(jj)
                        steps += 1
                    }
                    tags[i] = foundVerb ? "IN" : "DT"
                }
            case "in":
                if text != "IN" {
                    if isPunctAt(nx) || ["CC", "TO"].contains(tagAt(nx)) || particleNext.contains(lowerAt(nx)) {
                        tags[i] = "RP"
                    } else {
                        tags[i] = "IN"
                    }
                }
            case "by":
                if text != "BY" {
                    tags[i] = (isPunctAt(nx) || tagAt(nx) == "CC" || (tagAt(nx) == "IN" && lowerAt(nx) != "that")) ? "RB" : "IN"
                }
            case "used":
                if lowerAt(nx) == "to" {
                    let pv = prevIdx(i, skipAdverbs: true)
                    if POSTagger.beForms.contains(lowerAt(pv)) {
                        let k = nx.flatMap { nextIdx($0) }
                        let after = k.map { classes[$0] } ?? ""
                        tags[i] = after == "Verb" && !lowerAt(k).hasSuffix("ing") ? "VBN" : "JJ"
                    } else {
                        tags[i] = "VBD"
                    }
                }
            case "live":
                if cls == "Adjective" || tags[i] == "JJ" {
                    let pv = prevIdx(i, skipAdverbs: true)
                    if let nx, classes[nx] == "Noun" { tags[i] = "JJ" }
                    else if POSTagger.beForms.contains(lowerAt(pv)) || ["go", "went", "goes", "going"].contains(lowerAt(pv)) { tags[i] = "JJ" }
                    else { tags[i] = "VBP" }
                }
            case "close":
                if tags[i].hasPrefix("NN") && !(lowerAt(nx) == "of" && tagAt(p) == "DT") { tags[i] = "JJ" }
            default:
                break
            }
            // Content-word verb/noun corrections (heteronyms such as record/present/object/use).
            guard ["Noun", "Adjective", "Adverb", "Verb"].contains(cls), !["that", "a", "in", "by", "used", "live"].contains(lower) else { continue }
            let pv = prevIdx(i, skipAdverbs: true)
            let pvl = lowerAt(pv)
            if cls != "Verb" && !tags[i].hasPrefix("VB") {
                if sentStart, i == 0 || tokens[i].text.unicodeScalars.first!.isUppercaseLetter,
                   !(lower.hasSuffix("s") && !lower.hasSuffix("ss")), !lower.hasSuffix("ing"), !lower.hasSuffix("ed"),
                   let nx, !["that", "which", "who", "whose"].contains(lowerAt(nx)), npStartTags.contains(tags[nx]) || objectPronouns.contains(lowerAt(nx)) || tagAt(nx) == "PRP" {
                    tags[i] = "VB"  // imperative: "Use the tool", "Close the door"
                } else if let pv, POSTagger.modals.contains(pvl) && tags[pv] == "MD" {
                    tags[i] = "VB"  // "can separate", "will record"
                } else if pvl == "to", let nx, npStartTags.contains(tags[nx]) || ["PRP"].contains(tags[nx]) {
                    tags[i] = "VB"  // "to present the award"
                }
            } else if cls == "Verb", let p = prevIdx(i), ["DT", "PRP$", "POS"].contains(tags[p]), !["all", "both", "each", "that", "this"].contains(lowerAt(p)) {
                tags[i] = "NN"  // "the record", "a permit"
            }
        }
    }
}
