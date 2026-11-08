// Loading and caching of misaki's bundled pronunciation dictionaries.
//
// The JSON files are flat objects: `"word": "phonemes"` or `"word": {"DEFAULT": "…", "VERB": "…", "NOUN": null}`.
// A small hand-written parser reads them ~10× faster than JSONSerialization + bridging.
// Parsed (and "grown", see `growDictionary`) dictionaries are cached process-wide per file, so
// creating several `EnglishG2P` instances costs nothing after the first.
import Foundation

enum LexValue: Sendable {
    case word(String)
    /// Tag → phonemes (nil = "no pronunciation for this tag": spell it). Always has "DEFAULT".
    case tagged([String: String?])
}

public enum KokoroG2PError: Error, CustomStringConvertible {
    case missingResource(String)
    case malformedLexicon(String)

    public var description: String {
        switch self {
        case .missingResource(let n): return "KokoroG2P: missing bundled lexicon \(n)"
        case .malformedLexicon(let n): return "KokoroG2P: malformed lexicon \(n)"
        }
    }
}

final class LexiconStore: @unchecked Sendable {
    static let shared = LexiconStore()
    private let lock = NSLock()
    private var cache: [String: [String: LexValue]] = [:]

    /// Returns the grown dictionary for e.g. "us_gold". Thread-safe; parsed once per process.
    func dictionary(_ name: String) throws -> [String: LexValue] {
        lock.lock()
        defer { lock.unlock() }
        if let d = cache[name] { return d }
        let d = try Self.load(name)
        cache[name] = d
        return d
    }

    static func resourceURL(_ name: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Lexicons")
            ?? Bundle.module.url(forResource: name, withExtension: "json")
    }

    static func load(_ name: String) throws -> [String: LexValue] {
        guard let url = resourceURL(name) else { throw KokoroG2PError.missingResource(name) }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard let raw = data.withUnsafeBytes({ buf -> [String: LexValue]? in
            var p = FlatJSONParser(buf.bindMemory(to: UInt8.self))
            return p.parseObject()
        }) else { throw KokoroG2PError.malformedLexicon(name) }
        return growDictionary(raw)
    }

    /// misaki `Lexicon.grow_dictionary`: add "Word" for every lowercase "word" and vice versa
    /// (original entries win).
    static func growDictionary(_ d: [String: LexValue]) -> [String: LexValue] {
        var out = d
        out.reserveCapacity(d.count * 2)
        for (k, v) in d {
            guard k.pyCount >= 2 else { continue }
            let lower = k.lowercased()
            if k == lower {
                let cap = k.pyCapitalize
                if k != cap, d[cap] == nil { out[cap] = v }
            } else if k == lower.pyCapitalize {
                if d[lower] == nil { out[lower] = v }
            }
        }
        return out
    }
}

/// Minimal JSON reader for `{ "k": "v" | { "k": "v" | null, … }, … }` with standard string escapes.
struct FlatJSONParser {
    let b: UnsafeBufferPointer<UInt8>
    var i = 0
    init(_ b: UnsafeBufferPointer<UInt8>) { self.b = b }

    mutating func skipWS() {
        while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 }
    }

    mutating func expect(_ c: UInt8) -> Bool {
        skipWS()
        guard i < b.count, b[i] == c else { return false }
        i += 1
        return true
    }

    mutating func parseString() -> String? {
        skipWS()
        guard i < b.count, b[i] == 0x22 else { return nil }
        i += 1
        let start = i
        var hasEscape = false
        while i < b.count, b[i] != 0x22 {
            if b[i] == 0x5C { hasEscape = true; i += 2 } else { i += 1 }
        }
        guard i < b.count else { return nil }
        let slice = UnsafeBufferPointer(rebasing: b[start..<i])
        i += 1
        if !hasEscape { return String(decoding: slice, as: UTF8.self) }
        // Slow path with escapes.
        var bytes: [UInt8] = []
        var j = 0
        var pendingHigh: UInt32? = nil
        func appendScalar(_ v: UInt32) {
            if let s = Unicode.Scalar(v) { bytes.append(contentsOf: Array(String(Character(s)).utf8)) }
        }
        while j < slice.count {
            let c = slice[j]
            if c != 0x5C { bytes.append(c); j += 1; continue }
            let e = slice[j + 1]
            j += 2
            switch e {
            case 0x6E: bytes.append(0x0A)
            case 0x74: bytes.append(0x09)
            case 0x72: bytes.append(0x0D)
            case 0x62: bytes.append(0x08)
            case 0x66: bytes.append(0x0C)
            case 0x75:
                let hex = String(decoding: UnsafeBufferPointer(rebasing: slice[j..<min(j + 4, slice.count)]), as: UTF8.self)
                j += 4
                let v = UInt32(hex, radix: 16) ?? 0xFFFD
                if (0xD800..<0xDC00).contains(v) { pendingHigh = v; continue }
                if (0xDC00..<0xE000).contains(v), let h = pendingHigh {
                    appendScalar(0x10000 + ((h - 0xD800) << 10) + (v - 0xDC00))
                    pendingHigh = nil
                    continue
                }
                appendScalar(v)
            default: bytes.append(e)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    mutating func parseNull() -> Bool {
        skipWS()
        guard i + 4 <= b.count, b[i] == 0x6E, b[i + 1] == 0x75, b[i + 2] == 0x6C, b[i + 3] == 0x6C else { return false }
        i += 4
        return true
    }

    mutating func parseTagged() -> [String: String?]? {
        guard expect(0x7B) else { return nil }
        var out: [String: String?] = [:]
        if expect(0x7D) { return out }
        while true {
            guard let k = parseString(), expect(0x3A) else { return nil }
            if parseNull() { out[k] = .some(nil) } else {
                guard let v = parseString() else { return nil }
                out[k] = v
            }
            if expect(0x2C) { continue }
            guard expect(0x7D) else { return nil }
            return out
        }
    }

    mutating func parseObject() -> [String: LexValue]? {
        guard expect(0x7B) else { return nil }
        var out: [String: LexValue] = [:]
        out.reserveCapacity(b.count / 28)
        if expect(0x7D) { return out }
        while true {
            guard let k = parseString(), expect(0x3A) else { return nil }
            skipWS()
            guard i < b.count else { return nil }
            if b[i] == 0x7B {
                guard let t = parseTagged() else { return nil }
                out[k] = .tagged(t)
            } else {
                guard let v = parseString() else { return nil }
                out[k] = .word(v)
            }
            if expect(0x2C) { continue }
            guard expect(0x7D) else { return nil }
            return out
        }
    }
}
