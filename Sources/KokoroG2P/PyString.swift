// Small helpers that reproduce the Python `str` semantics misaki relies on
// (code-point based slicing, `isalpha`, `lower`, `capitalize`, …).
import Foundation

extension Unicode.Scalar {
    /// Python `str.isalpha()` for one code point (categories Lu, Ll, Lt, Lm, Lo).
    @inline(__always) var pyIsAlpha: Bool {
        switch properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter: return true
        default: return false
        }
    }

    @inline(__always) var isASCIIDigit: Bool { value >= 48 && value <= 57 }

    /// Python `str.isspace()`.
    @inline(__always) var pyIsSpace: Bool {
        switch value {
        case 0x09...0x0D, 0x1C...0x1F, 0x20, 0x85, 0xA0: return true
        default: return value > 0x7F && properties.isWhitespace
        }
    }

    @inline(__always) var isUppercaseLetter: Bool { properties.generalCategory == .uppercaseLetter }
    @inline(__always) var isLowercaseLetter: Bool { properties.generalCategory == .lowercaseLetter }
}

extension String {
    /// Code points as an array (Python indexing semantics).
    @inline(__always) var scalars: [Unicode.Scalar] { Array(unicodeScalars) }

    init(scalars: some Sequence<Unicode.Scalar>) {
        var s = String.UnicodeScalarView()
        s.append(contentsOf: scalars)
        self = String(s)
    }

    @inline(__always) var pyLower: String { lowercased() }
    @inline(__always) var pyUpper: String { uppercased() }
    @inline(__always) var isPyLowerEqual: Bool { self == lowercased() }
    @inline(__always) var isPyUpperEqual: Bool { self == uppercased() }

    /// Python `str.isalpha()`.
    var pyIsAlpha: Bool {
        var any = false
        for s in unicodeScalars {
            if !s.pyIsAlpha { return false }
            any = true
        }
        return any
    }

    /// Python `str.capitalize()`: first character upper-cased, the rest lower-cased.
    var pyCapitalize: String {
        guard let first = unicodeScalars.first else { return self }
        let rest = String(scalars: unicodeScalars.dropFirst())
        return String(first).uppercased() + rest.lowercased()
    }

    /// Number of code points (Python `len`).
    @inline(__always) var pyCount: Int { unicodeScalars.count }

    /// Python slice `s[from:]` / `s[:to]` with code-point indices (negative = from end).
    func pySlice(_ from: Int? = nil, _ to: Int? = nil) -> String {
        let sc = scalars
        let n = sc.count
        func norm(_ i: Int) -> Int { i < 0 ? max(0, n + i) : min(i, n) }
        let a = from.map(norm) ?? 0
        let b = to.map(norm) ?? n
        if a >= b { return "" }
        return String(scalars: sc[a..<b])
    }

    var lastScalar: Unicode.Scalar? { unicodeScalars.last }
    var firstScalar: Unicode.Scalar? { unicodeScalars.first }

    /// Python `str.strip(chars)`.
    func pyStrip(_ chars: Set<Unicode.Scalar>) -> String {
        let sc = scalars
        var a = 0, b = sc.count
        while a < b && chars.contains(sc[a]) { a += 1 }
        while b > a && chars.contains(sc[b - 1]) { b -= 1 }
        return String(scalars: sc[a..<b])
    }

    func pyRStrip(_ chars: Set<Unicode.Scalar>) -> String {
        let sc = scalars
        var b = sc.count
        while b > 0 && chars.contains(sc[b - 1]) { b -= 1 }
        return String(scalars: sc[0..<b])
    }

    func pyLStrip(_ chars: Set<Unicode.Scalar>) -> String {
        let sc = scalars
        var a = 0
        while a < sc.count && chars.contains(sc[a]) { a += 1 }
        return String(scalars: sc[a...])
    }

    /// `^[0-9]+$` (misaki's `is_digit`).
    var isASCIIDigits: Bool {
        var any = false
        for s in unicodeScalars {
            if !s.isASCIIDigit { return false }
            any = true
        }
        return any
    }

    func containsScalar(_ c: Unicode.Scalar) -> Bool { unicodeScalars.contains(c) }
}

/// Thin wrapper over NSRegularExpression with Python-like helpers.
struct PyRegex: @unchecked Sendable {
    let re: NSRegularExpression
    init(_ pattern: String, options: NSRegularExpression.Options = []) {
        // Patterns are compile-time constants; a failure is a programming error.
        re = try! NSRegularExpression(pattern: pattern, options: options)
    }

    func firstMatch(_ s: String) -> NSTextCheckingResult? {
        re.firstMatch(in: s, options: [], range: NSRange(s.startIndex..., in: s))
    }

    func matches(_ s: String) -> Bool { firstMatch(s) != nil }

    /// Python `re.findall` for patterns without groups.
    func findAll(_ s: String) -> [String] {
        let ns = s as NSString
        return re.matches(in: s, options: [], range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
    }

    func split(_ s: String) -> [String] {
        let ns = s as NSString
        var out: [String] = []
        var last = 0
        for m in re.matches(in: s, options: [], range: NSRange(location: 0, length: ns.length)) {
            out.append(ns.substring(with: NSRange(location: last, length: m.range.location - last)))
            last = m.range.location + m.range.length
        }
        out.append(ns.substring(from: last))
        return out
    }
}
