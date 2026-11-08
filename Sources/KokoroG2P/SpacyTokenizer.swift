// A port of spaCy's English rule-based tokenizer (prefix/suffix/infix rules, special cases, URL
// matching), which misaki uses through `en_core_web_sm`. spaCy is MIT-licensed
// (https://github.com/explosion/spaCy); the regular expressions are simplified to Unicode
// properties (e.g. spaCy's huge ALPHA class → \p{L}, its symbol class → \p{So}).
import Foundation

struct RawToken {
    var text: String
    var whitespace: String
    /// UTF-16 offset of the token in the tokenized string.
    var start: Int
    var isSpace: Bool { text.unicodeScalars.allSatisfy { $0.pyIsSpace } }
}

final class SpacyTokenizer: @unchecked Sendable {
    static let shared = SpacyTokenizer()

    let specials: [String: [String]]

    // LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY, LIST_ICONS from spacy/lang/char_classes.py
    static let punct = #"…|……|,|:|;|!|\?|¿|؟|¡|\(|\)|\[|\]|\{|\}|<|>|_|#|\*|&|。|？|！|，|、|；|：|～|·|।|،|۔|؛|٪"#
    static let quotes = #"'|"|”|“|`|‘|´|’|‚|,|„|»|«|「|」|『|』|（|）|〔|〕|【|】|《|》|〈|〉|〈|〉|⟦|⟧"#
    static let quoteChars = #"'"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉⟦⟧"#
    static let currency = #"\$|£|€|¥|฿|US\$|C\$|A\$|₽|﷼|[₠-₿]"#
    static let units = "km|km²|km³|m|m²|m³|dm|dm²|dm³|cm|cm²|cm³|mm|mm²|mm³|ha|µm|nm|yd|in|ft|kg|g|mg|µg|t|lb|oz|m/s|km/h|kmh|mph|hPa|Pa|mbar|mb|MB|kb|KB|gb|GB|tb|TB|T|G|M|K|%"
    static let icons = #"\p{So}"#
    static let hyphens = "-|–|—|--|---|——|~"

    let prefixRE: PyRegex
    let suffixRE: PyRegex
    let infixRE: PyRegex
    let urlRE: PyRegex

    init() {
        var sp: [String: [String]] = [:]
        for line in spacyTokenizerExceptionsData.split(separator: "\n", omittingEmptySubsequences: true) {
            let parts = line.split(separator: "\t", maxSplits: 1)
            let key = String(parts[0])
            sp[key] = parts.count > 1 ? parts[1].split(separator: " ").map(String.init) : [key]
        }
        specials = sp

        let P = SpacyTokenizer.self
        prefixRE = PyRegex("^(?:§|%|=|—|–|\\+(?![0-9])|" + P.punct + "|\\.\\.+|…|" + P.quotes + "|" + P.currency + "|" + P.icons + ")")
        let puncChars = #"…,:;!?¿؟¡()\[\]{}<>_#*\&。？！，、；：～·।،۔؛٪|"#
        suffixRE = PyRegex("(?:" + P.punct + "|\\.\\.+|…|" + P.quotes + "|" + P.icons
            + "|'s|'S|’s|’S|—|–"
            + "|(?<=[0-9])\\+"
            + "|(?<=°[FfCcKk])\\."
            + "|(?<=[0-9])(?:" + P.currency + ")"
            + "|(?<=[0-9])(?:" + P.units + ")"
            + "|(?<=[0-9\\p{Ll}%²\\-+" + puncChars + P.quoteChars + "])\\."
            + "|(?<=\\p{Lu}\\p{Lu})\\."
            + ")$")
        infixRE = PyRegex("\\.\\.+|…|" + P.icons
            + "|(?<=[0-9])[+\\-*^](?=[0-9\\-])"
            + "|(?<=[\\p{Ll}" + P.quoteChars + "])\\.(?=[\\p{Lu}" + P.quoteChars + "])"
            + "|(?<=\\p{L}),(?=\\p{L})"
            + "|(?<=[\\p{L}0-9])(?:" + P.hyphens + ")(?=\\p{L})"
            + "|(?<=[\\p{L}0-9])[:<>=/](?=\\p{L})")
        urlRE = PyRegex(#"^(?:(?:[\w+\-.]{2,})://)?(?:\S+(?::\S*)?@)?(?:(?!(?:10|127)(?:\.\d{1,3}){3})(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[A-Za-z0-9\x{a1}-\x{ffff}][A-Za-z0-9\x{a1}-\x{ffff}_\-]{0,62})?[A-Za-z0-9\x{a1}-\x{ffff}]\.)+(?:\p{Ll}{2,63}))(?::\d{2,5})?(?:[/?#]\S*)?$"#)
    }

    /// Tokenizes `text` like spaCy: returns tokens with their trailing whitespace (" " or "").
    /// Extra whitespace (double spaces, newlines) becomes separate whitespace tokens.
    func tokenize(_ text: String) -> [RawToken] {
        var out: [RawToken] = []
        let sc = text.scalars
        guard !sc.isEmpty else { return out }
        // UTF-16 offsets of each scalar.
        var offs: [Int] = []
        offs.reserveCapacity(sc.count + 1)
        var o = 0
        for s in sc { offs.append(o); o += s.utf16.count }
        offs.append(o)

        var inWS = sc[0].pyIsSpace
        var start = 0
        func emit(_ a: Int, _ b: Int) {
            let span = String(scalars: sc[a..<b])
            for (t, rel) in tokenizeSpan(span) {
                out.append(RawToken(text: t, whitespace: "", start: offs[a] + rel))
            }
        }
        for i in 0..<sc.count {
            let uc = sc[i]
            if uc.pyIsSpace != inWS {
                if start < i { emit(start, i) }
                if uc == " " {
                    if !out.isEmpty { out[out.count - 1].whitespace = " " }
                    start = i + 1
                } else {
                    start = i
                }
                inWS.toggle()
            }
        }
        if start < sc.count {
            emit(start, sc.count)
            if sc.last == " " && !inWS { out[out.count - 1].whitespace = " " }
        }
        return out
    }

    private func prefixLen(_ s: String) -> Int? {
        guard let m = prefixRE.firstMatch(s) else { return nil }
        return m.range.length
    }

    private func suffixStart(_ s: String) -> Int? {
        guard let m = suffixRE.firstMatch(s) else { return nil }
        return m.range.location
    }

    /// spaCy `Tokenizer.explain` algorithm for one whitespace-free span.
    /// Returns pieces with their UTF-16 offsets relative to the span.
    func tokenizeSpan(_ span: String) -> [(String, Int)] {
        if span.unicodeScalars.allSatisfy({ $0.pyIsSpace }) { return [(span, 0)] }
        var tokens: [String] = []
        var suffixes: [String] = []
        var substring = span as NSString
        func special(_ s: NSString) -> [String]? { specials[s as String] }
        outer: while substring.length > 0 {
            if let sp = special(substring) {
                tokens += sp
                substring = ""
                continue
            }
            while prefixLen(substring as String) != nil || suffixStart(substring as String) != nil {
                if let sp = special(substring) {
                    tokens += sp
                    substring = ""
                    break
                }
                if let split = prefixLen(substring as String) {
                    if split == 0 { break }
                    tokens.append(substring.substring(to: split))
                    substring = substring.substring(from: split) as NSString
                    if special(substring) != nil { continue }
                }
                if let split = suffixStart(substring as String) {
                    if split == substring.length { break }
                    suffixes.append(substring.substring(from: split))
                    substring = substring.substring(to: split) as NSString
                }
            }
            if substring.length == 0 { continue outer }
            let s = substring as String
            if urlRE.matches(s) {
                tokens.append(s)
            } else if let sp = special(substring) {
                tokens += sp
            } else {
                let ms = infixRE.re.matches(in: s, options: [], range: NSRange(location: 0, length: substring.length))
                if ms.isEmpty {
                    tokens.append(s)
                } else {
                    var offset = 0
                    for m in ms {
                        if offset == 0 && m.range.location == 0 { continue }
                        if m.range.location > offset {
                            tokens.append(substring.substring(with: NSRange(location: offset, length: m.range.location - offset)))
                        }
                        if m.range.length > 0 { tokens.append(substring.substring(with: m.range)) }
                        offset = m.range.location + m.range.length
                    }
                    if offset < substring.length { tokens.append(substring.substring(from: offset)) }
                }
            }
            substring = ""
        }
        tokens += suffixes.reversed()
        // Attach offsets (pieces concatenate back to the span).
        var res: [(String, Int)] = []
        var off = 0
        for t in tokens {
            res.append((t, off))
            off += (t as NSString).length
        }
        return res
    }
}
