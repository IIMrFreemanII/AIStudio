// Port of misaki `en.Lexicon` (Apache-2.0, https://github.com/hexgrad/misaki): dictionary lookup with
// tags, capitalization/stress rules, special cases, -s/-ed/-ing morphology and number reading.
import Foundation

enum Phon {
    static let primary: Unicode.Scalar = "ˈ"
    static let secondary: Unicode.Scalar = "ˌ"
    static let stresses: Set<Unicode.Scalar> = ["ˌ", "ˈ"]
    static let vowels: Set<Unicode.Scalar> = Set("AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ".unicodeScalars)
    static let consonants: Set<Unicode.Scalar> = Set("bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ".unicodeScalars)
    static let usTaus: Set<Unicode.Scalar> = Set("AIOWYiuæɑəɛɪɹʊʌ".unicodeScalars)
    static let diphthongs: Set<Unicode.Scalar> = Set("AIOQWYʤʧ".unicodeScalars)
    static let puncts: Set<Unicode.Scalar> = Set(";:,.!?—…\"“”".unicodeScalars)
    static let nonQuotePuncts: Set<Unicode.Scalar> = Set(";:,.!?—…".unicodeScalars)
    static let subtokenJunks: Set<Unicode.Scalar> = Set("',-._‘’/".unicodeScalars)

    static func stressWeight(_ ps: String?) -> Int {
        guard let ps else { return 0 }
        return ps.unicodeScalars.reduce(0) { $0 + (diphthongs.contains($1) ? 2 : 1) }
    }

    /// misaki `apply_stress`.
    static func applyStress(_ ps: String?, _ stress: Double?) -> String? {
        guard let ps else { return nil }
        guard let stress else { return ps }
        let hasPrimary = ps.containsScalar(primary)
        let hasAny = ps.unicodeScalars.contains(where: { stresses.contains($0) })
        let hasVowel = ps.unicodeScalars.contains(where: { vowels.contains($0) })
        if stress < -1 {
            return ps.replacingOccurrences(of: "ˈ", with: "").replacingOccurrences(of: "ˌ", with: "")
        } else if stress == -1 || ((stress == 0 || stress == -0.5) && hasPrimary) {
            return ps.replacingOccurrences(of: "ˌ", with: "").replacingOccurrences(of: "ˈ", with: "ˌ")
        } else if (stress == 0 || stress == 0.5 || stress == 1) && !hasAny {
            if !hasVowel { return ps }
            return restress("ˌ" + ps)
        } else if stress >= 1 && !hasPrimary && ps.containsScalar(secondary) {
            return ps.replacingOccurrences(of: "ˌ", with: "ˈ")
        } else if stress > 1 && !hasAny {
            if !hasVowel { return ps }
            return restress("ˈ" + ps)
        }
        return ps
    }

    /// Moves every stress mark to just before the next vowel.
    static func restress(_ ps: String) -> String {
        let sc = ps.scalars
        var items: [(Double, Unicode.Scalar)] = []
        for (i, p) in sc.enumerated() {
            if stresses.contains(p) {
                if let j = sc[i...].firstIndex(where: { vowels.contains($0) }) {
                    items.append((Double(j) - 0.5, p))
                } else {
                    items.append((Double(i), p))
                }
            } else {
                items.append((Double(i), p))
            }
        }
        items.sort { $0.0 != $1.0 ? $0.0 < $1.0 : $0.1.value < $1.1.value }
        return String(scalars: items.map { $0.1 })
    }
}

let lexiconOrds: Set<UInt32> = Set([39, 45] + Array(65...90) + Array(97...122))

let currencies: [String: (String, String)] = ["$": ("dollar", "cent"), "£": ("pound", "pence"), "€": ("euro", "cent")]
let ordinalSuffixes: Set<String> = ["st", "nd", "rd", "th"]
let addSymbols: [String: String] = [".": "dot", "/": "slash"]
let symbolWords: [String: String] = ["%": "percent", "&": "and", "+": "plus", "@": "at"]

final class Lexicon {
    let british: Bool
    let capStresses: (Double, Double) = (0.5, 2)
    let golds: [String: LexValue]
    private var silversStorage: [String: LexValue]?
    private let silverName: String

    init(british: Bool) throws {
        self.british = british
        let prefix = british ? "gb" : "us"
        golds = try LexiconStore.shared.dictionary("\(prefix)_gold")
        silverName = "\(prefix)_silver"
    }

    /// The silver dictionary is only needed for words missing from gold, so it loads lazily.
    var silvers: [String: LexValue] {
        if let s = silversStorage { return s }
        let s = (try? LexiconStore.shared.dictionary(silverName)) ?? [:]
        silversStorage = s
        return s
    }

    func preloadSilver() { _ = silvers }

    private func goldString(_ key: String) -> String? {
        if case .word(let s)? = golds[key] { return s }
        if case .tagged(let t)? = golds[key] { return t["DEFAULT"] ?? nil }
        return nil
    }

    private func goldTagged(_ key: String, _ tag: String) -> String? {
        if case .tagged(let t)? = golds[key] { return (t[tag] ?? nil) ?? (t["DEFAULT"] ?? nil) }
        return goldString(key)
    }

    static func parentTag(_ tag: String?) -> String? {
        guard let tag else { return nil }
        if tag.hasPrefix("VB") { return "VERB" }
        if tag.hasPrefix("NN") { return "NOUN" }
        if tag.hasPrefix("ADV") || tag.hasPrefix("RB") { return "ADV" }
        if tag.hasPrefix("ADJ") || tag.hasPrefix("JJ") { return "ADJ" }
        return tag
    }

    func getNNP(_ word: String) -> (String?, Int?) {
        var parts: [String] = []
        for c in word.unicodeScalars where c.pyIsAlpha {
            guard let p = goldString(String(c).uppercased()) else { return (nil, nil) }
            parts.append(p)
        }
        guard let joined = Phon.applyStress(parts.joined(), 0) else { return (nil, nil) }
        // ps.rsplit('ˌ', 1) joined with 'ˈ': the last secondary stress becomes primary.
        if let r = joined.range(of: "ˌ", options: .backwards) {
            return (joined.replacingCharacters(in: r, with: "ˈ"), 3)
        }
        return (joined, 3)
    }

    func getSpecialCase(_ word: String, _ tag: String?, _ stress: Double?, _ ctx: TokenContext) -> (String?, Int?) {
        let tag = tag ?? ""
        if tag == "ADD", let w = addSymbols[word] {
            return lookup(w, nil, -0.5, ctx)
        } else if let w = symbolWords[word] {
            return lookup(w, nil, nil, ctx)
        } else if word.pyStrip(["."]).containsScalar("."),
                  word.replacingOccurrences(of: ".", with: "").pyIsAlpha,
                  (word.components(separatedBy: ".").map { $0.pyCount }.max() ?? 0) < 3 {
            return getNNP(word)
        } else if word == "a" || word == "A" {
            return (tag == "DT" ? "ɐ" : "ˈA", 4)
        } else if word == "am" || word == "Am" || word == "AM" {
            if tag.hasPrefix("NN") { return getNNP(word) }
            if ctx.futureVowel == nil || word != "am" || (stress != nil && stress! > 0) {
                return (goldString("am"), 4)
            }
            return ("ɐm", 4)
        } else if word == "an" || word == "An" || word == "AN" {
            if word == "AN" && tag.hasPrefix("NN") { return getNNP(word) }
            return ("ɐn", 4)
        } else if word == "I" && tag == "PRP" {
            return ("ˌI", 4)
        } else if (word == "by" || word == "By" || word == "BY") && Lexicon.parentTag(tag) == "ADV" {
            return ("bˈI", 4)
        } else if word == "to" || word == "To" || (word == "TO" && (tag == "TO" || tag == "IN")) {
            switch ctx.futureVowel {
            case nil: return (goldString("to"), 4)
            case false?: return ("tə", 4)
            case true?: return ("tʊ", 4)
            }
        } else if word == "in" || word == "In" || (word == "IN" && tag != "NNP") {
            let s = (ctx.futureVowel == nil || tag != "IN") ? "ˈ" : ""
            return (s + "ɪn", 4)
        } else if word == "the" || word == "The" || (word == "THE" && tag == "DT") {
            return (ctx.futureVowel == true ? "ði" : "ðə", 4)
        } else if tag == "IN" && (word.lowercased() == "vs" || word.lowercased() == "vs.") {
            return lookup("versus", nil, nil, ctx)
        } else if word == "used" || word == "Used" || word == "USED" {
            if (tag == "VBD" || tag == "JJ") && ctx.futureTo {
                return (goldTagged("used", "VBD"), 4)
            }
            return (goldTagged("used", "DEFAULT"), 4)
        }
        return (nil, nil)
    }

    func isKnown(_ word: String, _ tag: String?) -> Bool {
        if golds[word] != nil || symbolWords[word] != nil || silvers[word] != nil { return true }
        if !word.pyIsAlpha || !word.unicodeScalars.allSatisfy({ lexiconOrds.contains($0.value) }) { return false }
        if word.pyCount == 1 { return true }
        if word == word.uppercased() && golds[word.lowercased()] != nil { return true }
        let rest = word.pySlice(1)
        return rest == rest.uppercased()
    }

    func lookup(_ word0: String, _ tag0: String?, _ stress: Double?, _ ctx: TokenContext?) -> (String?, Int?) {
        var word = word0
        var tag = tag0
        var isNNP: Bool? = nil
        if word == word.uppercased() && golds[word] == nil {
            word = word.lowercased()
            isNNP = tag == "NNP"
        }
        var entry = golds[word]
        var rating = 4
        if entry == nil && isNNP != true {
            entry = silvers[word]
            rating = 3
        }
        var ps: String? = nil
        switch entry {
        case .word(let s)?: ps = s
        case .tagged(let t)?:
            if let ctx, ctx.futureVowel == nil, t["None"] != nil {
                tag = "None"
            } else if tag == nil || t[tag!] == nil {
                tag = Lexicon.parentTag(tag)
            }
            if let tag, let v = t[tag] { ps = v } else { ps = t["DEFAULT"] ?? nil }
        case nil: ps = nil
        }
        if ps == nil || (isNNP == true && !ps!.containsScalar(Phon.primary)) {
            let (p, r) = getNNP(word)
            if p != nil { return (p, r) }
        }
        return (Phon.applyStress(ps, stress), rating)
    }

    func suffixS(_ stem: String?) -> String? {
        guard let stem, let last = stem.lastScalar else { return nil }
        if "ptkfθ".unicodeScalars.contains(last) { return stem + "s" }
        if "szʃʒʧʤ".unicodeScalars.contains(last) { return stem + (british ? "ɪ" : "ᵻ") + "z" }
        return stem + "z"
    }

    func stemS(_ word: String, _ tag: String?, _ stress: Double?, _ ctx: TokenContext?) -> (String?, Int?) {
        let n = word.pyCount
        guard n >= 3, word.hasSuffix("s") else { return (nil, nil) }
        let stem: String
        if !word.hasSuffix("ss") && isKnown(word.pySlice(nil, -1), tag) {
            stem = word.pySlice(nil, -1)
        } else if (word.hasSuffix("'s") || (n > 4 && word.hasSuffix("es") && !word.hasSuffix("ies"))) && isKnown(word.pySlice(nil, -2), tag) {
            stem = word.pySlice(nil, -2)
        } else if n > 4 && word.hasSuffix("ies") && isKnown(word.pySlice(nil, -3) + "y", tag) {
            stem = word.pySlice(nil, -3) + "y"
        } else {
            return (nil, nil)
        }
        let (ps, rating) = lookup(stem, tag, stress, ctx)
        return (suffixS(ps), rating)
    }

    func suffixED(_ stem: String?) -> String? {
        guard let stem, let last = stem.lastScalar else { return nil }
        let sc = stem.scalars
        if "pkfθʃsʧ".unicodeScalars.contains(last) { return stem + "t" }
        if last == "d" { return stem + (british ? "ɪ" : "ᵻ") + "d" }
        if last != "t" { return stem + "d" }
        if british || sc.count < 2 { return stem + "ɪd" }
        if Phon.usTaus.contains(sc[sc.count - 2]) { return String(scalars: sc.dropLast()) + "ɾᵻd" }
        return stem + "ᵻd"
    }

    func stemED(_ word: String, _ tag: String?, _ stress: Double?, _ ctx: TokenContext?) -> (String?, Int?) {
        let n = word.pyCount
        guard n >= 4, word.hasSuffix("d") else { return (nil, nil) }
        let stem: String
        if !word.hasSuffix("dd") && isKnown(word.pySlice(nil, -1), tag) {
            stem = word.pySlice(nil, -1)
        } else if n > 4 && word.hasSuffix("ed") && !word.hasSuffix("eed") && isKnown(word.pySlice(nil, -2), tag) {
            stem = word.pySlice(nil, -2)
        } else {
            return (nil, nil)
        }
        let (ps, rating) = lookup(stem, tag, stress, ctx)
        return (suffixED(ps), rating)
    }

    func suffixING(_ stem: String?) -> String? {
        guard let stem, let last = stem.lastScalar else { return nil }
        let sc = stem.scalars
        if british {
            if last == "ə" || last == "ː" { return nil }
        } else if sc.count > 1 && last == "t" && Phon.usTaus.contains(sc[sc.count - 2]) {
            return String(scalars: sc.dropLast()) + "ɾɪŋ"
        }
        return stem + "ɪŋ"
    }

    private static let doubledING = PyRegex("([bcdgklmnprstvxz])\\1ing$|cking$")

    func stemING(_ word: String, _ tag: String?, _ stress: Double?, _ ctx: TokenContext?) -> (String?, Int?) {
        let n = word.pyCount
        guard n >= 5, word.hasSuffix("ing") else { return (nil, nil) }
        let stem: String
        if n > 5 && isKnown(word.pySlice(nil, -3), tag) {
            stem = word.pySlice(nil, -3)
        } else if isKnown(word.pySlice(nil, -3) + "e", tag) {
            stem = word.pySlice(nil, -3) + "e"
        } else if n > 5 && Lexicon.doubledING.matches(word) && isKnown(word.pySlice(nil, -4), tag) {
            stem = word.pySlice(nil, -4)
        } else {
            return (nil, nil)
        }
        let (ps, rating) = lookup(stem, tag, stress, ctx)
        return (suffixING(ps), rating)
    }

    func getWord(_ word0: String, _ tag: String?, _ stress: Double?, _ ctx: TokenContext) -> (String?, Int?) {
        let special = getSpecialCase(word0, tag, stress, ctx)
        if special.0 != nil { return special }
        var word = word0
        let wl = word.lowercased()
        if word.pyCount > 1 && word.replacingOccurrences(of: "'", with: "").pyIsAlpha && word != wl
            && (tag != "NNP" || word.pyCount > 7)
            && golds[word] == nil && silvers[word] == nil
            && (word == word.uppercased() || word.pySlice(1) == word.pySlice(1).lowercased())
            && (golds[wl] != nil || silvers[wl] != nil
                || stemS(wl, tag, stress, ctx).0 != nil || stemED(wl, tag, stress, ctx).0 != nil
                || stemING(wl, tag, stress, ctx).0 != nil) {
            word = wl
        }
        if isKnown(word, tag) {
            return lookup(word, tag, stress, ctx)
        } else if word.hasSuffix("s'") && isKnown(word.pySlice(nil, -2) + "'s", tag) {
            return lookup(word.pySlice(nil, -2) + "'s", tag, stress, ctx)
        } else if word.hasSuffix("'") && isKnown(word.pySlice(nil, -1), tag) {
            return lookup(word.pySlice(nil, -1), tag, stress, ctx)
        }
        let s = stemS(word, tag, stress, ctx)
        if s.0 != nil { return s }
        let ed = stemED(word, tag, stress, ctx)
        if ed.0 != nil { return ed }
        let ing = stemING(word, tag, stress ?? 0.5, ctx)
        if ing.0 != nil { return ing }
        return (nil, nil)
    }

    static func isCurrency(_ word: String) -> Bool {
        if !word.contains(".") { return true }
        if word.filter({ $0 == "." }).count > 1 { return false }
        let cents = word.components(separatedBy: ".")[1]
        // `set(cents) == {0}` compares characters with the int 0 in Python: never true.
        return cents.pyCount < 3
    }

    private static let numSuffixRE = PyRegex("[a-z']+$")
    private static let nonAlphaRE = PyRegex("[^a-z]+")

    func getNumber(_ word0: String, _ currency: String?, _ isHead: Bool, _ numFlags: String) -> (String?, Int?) {
        var word = word0
        var suffix: String? = nil
        if let m = Lexicon.numSuffixRE.firstMatch(word) {
            suffix = (word as NSString).substring(with: m.range)
            word = (word as NSString).substring(to: m.range.location)
        }
        var result: [(String?, Int?)] = []
        if word.hasPrefix("-") {
            result.append(lookup("minus", nil, nil, nil))
            word = String(word.dropFirst())
        }
        func extendNum(_ num: String, first: Bool = true, escape: Bool = false) {
            let text = escape ? num : NumberWords.cardinal(digits: num)
            let splits = Lexicon.nonAlphaRE.split(text)
            for (i, w) in splits.enumerated() {
                if w != "and" || numFlags.contains("&") {
                    if first && i == 0 && splits.count > 1 && w == "one" && numFlags.contains("a") {
                        result.append(("ə", 4))
                    } else {
                        result.append(lookup(w, nil, w == "point" ? -2 : nil, nil))
                    }
                } else if w == "and" && numFlags.contains("n") && !result.isEmpty {
                    let last = result[result.count - 1]
                    result[result.count - 1] = ((last.0 ?? "") + "ən", last.1)
                }
            }
        }
        if word.isASCIIDigits, let suffix, ordinalSuffixes.contains(suffix) {
            extendNum(NumberWords.ordinal(digits: word), escape: true)
        } else if result.isEmpty && word.pyCount == 4 && (currency == nil || currencies[currency!] == nil) && word.isASCIIDigits {
            extendNum(NumberWords.year(Int(word)!), escape: true)
        } else if !isHead && !word.contains(".") {
            let num = word.replacingOccurrences(of: ",", with: "")
            let ns = num.scalars
            if ns.first == "0" || ns.count > 3 {
                for n in ns { extendNum(String(n), first: false) }
            } else if ns.count == 3 && !num.hasSuffix("00") {
                extendNum(String(ns[0]))
                if ns[1] == "0" {
                    result.append(lookup("O", nil, -2, nil))
                    extendNum(String(ns[2]), first: false)
                } else {
                    extendNum(String(scalars: ns[1...]), first: false)
                }
            } else {
                extendNum(num)
            }
        } else if word.filter({ $0 == "." }).count > 1 || !isHead {
            var first = true
            for num in word.replacingOccurrences(of: ",", with: "").components(separatedBy: ".") {
                let ns = num.scalars
                if num.isEmpty {
                } else if ns[0] == "0" || (ns.count != 2 && ns.dropFirst().contains(where: { $0 != "0" })) {
                    for n in ns { extendNum(String(n), first: false) }
                } else {
                    extendNum(num, first: first)
                }
                first = false
            }
        } else if let currency, let units = currencies[currency], Lexicon.isCurrency(word) {
            var pairs: [(Int, String)] = []
            let comps = word.replacingOccurrences(of: ",", with: "").components(separatedBy: ".")
            for (num, unit) in zip(comps, [units.0, units.1]) {
                pairs.append((num.isEmpty ? 0 : (Int(num) ?? 0), unit))
            }
            if pairs.count > 1 {
                if pairs[1].0 == 0 { pairs = Array(pairs.prefix(1)) } else if pairs[0].0 == 0 { pairs = Array(pairs.dropFirst()) }
            }
            for (i, (num, unit)) in pairs.enumerated() {
                if i > 0 { result.append(lookup("and", nil, nil, nil)) }
                extendNum(String(num), first: i == 0)
                result.append(abs(num) != 1 && unit != "pence" ? stemS(unit + "s", nil, nil, nil) : lookup(unit, nil, nil, nil))
            }
        } else {
            var text: String
            if word.isASCIIDigits {
                text = NumberWords.cardinal(digits: word)
            } else if !word.contains(".") {
                let digits = word.replacingOccurrences(of: ",", with: "")
                guard digits.isASCIIDigits else { return (nil, nil) }
                text = (suffix != nil && ordinalSuffixes.contains(suffix!)) ? NumberWords.ordinal(digits: digits) : NumberWords.cardinal(digits: digits)
            } else {
                let w = word.replacingOccurrences(of: ",", with: "")
                if w.hasPrefix(".") {
                    text = "point " + w.dropFirst().compactMap { $0.wholeNumberValue }.map { NumberWords.cardinal($0) }.joined(separator: " ")
                } else {
                    guard let f = NumberWords.float(w) else { return (nil, nil) }
                    text = f
                }
            }
            extendNum(text, escape: true)
        }
        if result.isEmpty { return (nil, nil) }
        if result.contains(where: { $0.0 == nil }) { return (nil, nil) }
        let joined = result.map { $0.0! }.joined(separator: " ")
        let rating = result.compactMap { $0.1 }.min()
        switch suffix {
        case "s"?, "'s"?: return (suffixS(joined), rating)
        case "ed"?, "'d"?: return (suffixED(joined), rating)
        case "ing"?: return (suffixING(joined), rating)
        default: return (joined, rating)
        }
    }

    func appendCurrency(_ ps: String, _ currency: String?) -> String {
        guard let currency else { return ps }
        guard let c = currencies[currency], let s = stemS(c.0 + "s", nil, nil, nil).0 else { return ps }
        return "\(ps) \(s)"
    }

    static func numericIfNeeded(_ c: Unicode.Scalar) -> String {
        let p = c.properties
        guard p.numericType == .decimal || p.numericType == .digit, let v = p.numericValue else { return String(c) }
        return v == v.rounded() ? String(Int(v)) : String(c)
    }

    static func isNumber(_ word0: String, _ isHead: Bool) -> Bool {
        if !word0.unicodeScalars.contains(where: { $0.isASCIIDigit }) { return false }
        var word = word0
        for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] where word.hasSuffix(s) {
            word = String(word.dropLast(s.count))
            break
        }
        for (i, c) in word.unicodeScalars.enumerated() {
            if !(c.isASCIIDigit || c == "," || c == "." || (isHead && i == 0 && c == "-")) { return false }
        }
        return true
    }

    /// misaki `Lexicon.__call__`.
    func callAsFunction(_ tk: MToken, _ ctx: TokenContext) -> (String?, Int?) {
        var word = (tk.alias ?? tk.text).replacingOccurrences(of: "\u{2018}", with: "'").replacingOccurrences(of: "\u{2019}", with: "'")
        word = word.precomposedStringWithCompatibilityMapping
        word = word.unicodeScalars.map { Lexicon.numericIfNeeded($0) }.joined()
        let stress: Double? = word == word.lowercased() ? nil : (word == word.uppercased() ? capStresses.1 : capStresses.0)
        let (ps, rating) = getWord(word, tk.tag, stress, ctx)
        if let ps {
            return (Phon.applyStress(appendCurrency(ps, tk.currency), tk.stress), rating)
        } else if Lexicon.isNumber(word, tk.isHead) {
            let (ps, rating) = getNumber(word, tk.currency, tk.isHead, tk.numFlags)
            return (Phon.applyStress(ps, tk.stress), rating)
        }
        return (nil, nil)
    }
}
