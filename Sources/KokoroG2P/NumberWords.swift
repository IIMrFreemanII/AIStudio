// English number spelling that reproduces the Python `num2words` package (lang "en"), which
// misaki uses: cardinals ("one hundred and twenty-three"), ordinals, years and floats.
// Foundation's NumberFormatter `.spellOut` differs in details misaki depends on (the British
// "and" after hundreds, commas between groups, "oh-five" years), so this is a direct port.
import Foundation

enum NumberWords {
    static let ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                       "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                       "eighteen", "nineteen"]
    static let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    static let scales = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
                         "sextillion", "septillion", "octillion", "nonillion", "decillion", "undecillion",
                         "duodecillion", "tredecillion", "quattuordecillion", "quindecillion", "sexdecillion",
                         "septdecillion", "octodecillion", "novemdecillion", "vigintillion"]
    static let ords = ["one": "first", "two": "second", "three": "third", "four": "fourth", "five": "fifth",
                       "six": "sixth", "seven": "seventh", "eight": "eighth", "nine": "ninth", "ten": "tenth",
                       "eleven": "eleventh", "twelve": "twelfth"]

    private static func below100(_ n: Int) -> String {
        if n < 20 { return ones[n] }
        let t = tens[n / 10], u = n % 10
        return u == 0 ? t : "\(t)-\(ones[u])"
    }

    private static func below1000(_ n: Int) -> String {
        let h = n / 100, r = n % 100
        if h == 0 { return below100(r) }
        return r == 0 ? "\(ones[h]) hundred" : "\(ones[h]) hundred and \(below100(r))"
    }

    /// `num2words(int(digits))` for a non-negative decimal digit string (any length up to 10^66).
    static func cardinal(digits raw: String) -> String {
        var digits = Substring(raw.drop(while: { $0 == "0" }))
        if digits.isEmpty { return "zero" }
        // Split into groups of three from the right.
        var groups: [Int] = []
        while !digits.isEmpty {
            let take = min(3, digits.count)
            let g = digits.suffix(take)
            groups.append(Int(g)!)
            digits = digits.dropLast(take)
        }
        if groups.count > scales.count {  // beyond num2words' range: read digit by digit
            return raw.compactMap { $0.wholeNumberValue }.map { ones[$0] }.joined(separator: " ")
        }
        var parts: [(String, Bool)] = []  // (text, isSmallTail: value < 100)
        for i in stride(from: groups.count - 1, through: 0, by: -1) where groups[i] != 0 {
            let text = i == 0 ? below1000(groups[i]) : "\(below1000(groups[i])) \(scales[i])"
            parts.append((text, i == 0 && groups[i] < 100))
        }
        var out = parts[0].0
        for (k, p) in parts.enumerated().dropFirst() {
            out += (k == parts.count - 1 && p.1) ? " and " : ", "
            out += p.0
        }
        return out
    }

    static func cardinal(_ n: Int) -> String {
        n < 0 ? "minus " + cardinal(digits: String(-n)) : cardinal(digits: String(n))
    }

    static func ordinal(digits: String) -> String {
        var outwords = cardinal(digits: digits).components(separatedBy: " ")
        var lastwords = outwords[outwords.count - 1].components(separatedBy: "-")
        var last = lastwords[lastwords.count - 1].lowercased()
        if let o = ords[last] {
            last = o
        } else {
            if last.hasSuffix("y") { last = String(last.dropLast()) + "ie" }
            last += "th"
        }
        lastwords[lastwords.count - 1] = last
        outwords[outwords.count - 1] = lastwords.joined(separator: "-")
        return outwords.joined(separator: " ")
    }

    /// `num2words(n, to='year')` for non-negative `n`.
    static func year(_ val: Int) -> String {
        let high = val / 100, low = val % 100
        if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
            return cardinal(val)
        }
        let hightext = cardinal(high)
        let lowtext: String
        if low == 0 { lowtext = "hundred" } else if low < 10 { lowtext = "oh-\(cardinal(low))" } else { lowtext = cardinal(low) }
        return "\(hightext) \(lowtext)"
    }

    /// `num2words(float(s))` for a plain decimal string such as "3.14" (no sign, no commas).
    static func float(_ s: String) -> String? {
        guard let v = Double(s), v.isFinite else { return nil }
        if v == v.rounded(.towardZero), abs(v) < 1e18 {
            return cardinal(Int(v))
        }
        // Python's repr(float) and Swift's description are both the shortest round-trip form.
        var repr = v.description
        if repr.contains("e") {  // exotic magnitudes: keep the digits as typed
            repr = s
        }
        let comps = repr.split(separator: ".", omittingEmptySubsequences: false)
        let pre = comps[0].isEmpty ? "0" : String(comps[0])
        let post = comps.count > 1 ? String(comps[1]) : ""
        var out = [cardinal(digits: pre), "point"]
        for ch in post { if let d = ch.wholeNumberValue { out.append(ones[d]) } }
        return out.joined(separator: " ")
    }
}
