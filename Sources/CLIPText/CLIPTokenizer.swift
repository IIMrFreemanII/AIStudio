import Foundation

/// The CLIP byte-level BPE tokenizer (Hugging Face `CLIPTokenizer`), read from `vocab.json` and `merges.txt`.
///
/// Text is cleaned as the slow tokenizer does without `ftfy` (see ``clean(_:)``), split
/// with CLIP's pattern, mapped to byte characters, and merged with an end-of-word `</w>` marker.
public final class CLIPTokenizer {
    public let vocab: [String: Int]
    let ranks: [String: Int]  // "first second" → merge rank
    public let bos: Int
    public let eos: Int
    public let maxLength: Int
    let byteChars: [Character]
    let pattern: NSRegularExpression
    private var cache: [String: [String]] = [:]
    private let lock = NSLock()

    public init(directory: URL, maxLength: Int = 77) throws {
        let vocabData = try Data(contentsOf: directory.appendingPathComponent("vocab.json"))
        vocab = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] ?? [:]
        let merges = try String(contentsOf: directory.appendingPathComponent("merges.txt"), encoding: .utf8)
        // As `CLIPTokenizer`: skip the `#version` line and use at most 49152 − 256 − 2 merges.
        let lines = merges.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: "\n")
        var r: [String: Int] = [:]
        for (i, line) in lines.dropFirst().prefix(49_152 - 256 - 2).enumerated() where r[line] == nil { r[line] = i }
        ranks = r
        bos = vocab["<|startoftext|>"] ?? 49_406
        eos = vocab["<|endoftext|>"] ?? 49_407
        self.maxLength = maxLength
        byteChars = Self.bytesToUnicode()
        pattern = try NSRegularExpression(
            pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
            options: [.caseInsensitive])
    }

    /// GPT-2's reversible byte → printable character table.
    public static func bytesToUnicode() -> [Character] {
        var bs = Array(33...126) + Array(161...172) + Array(174...255)
        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }
        var table = [Character](repeating: " ", count: 256)
        for (b, c) in zip(bs, cs) { table[b] = Character(Unicode.Scalar(UInt32(c))!) }
        return table
    }

    /// What `CLIPTokenizer` does without `ftfy` (BERT's `BasicTokenizer`): drop control characters, put spaces
    /// around CJK ideographs (each becomes its own word), NFC, lower-case, collapse whitespace.
    public static func clean(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        for u in text.unicodeScalars {
            let v = u.value
            if v == 0 || v == 0xFFFD { continue }
            if u.properties.isWhitespace { out.append(" "); continue }
            if [.control, .format].contains(u.properties.generalCategory) { continue }
            if isCJK(v) { out.append(" "); out.append(u); out.append(" ") } else { out.append(u) }
        }
        return String(out).precomposedStringWithCanonicalMapping.lowercased()
            .split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    /// BERT's `_is_chinese_char` ranges (CJK Unified Ideographs and their extensions, compatibility ideographs).
    static func isCJK(_ v: UInt32) -> Bool {
        (0x4E00...0x9FFF).contains(v) || (0x3400...0x4DBF).contains(v) || (0x20000...0x2A6DF).contains(v)
            || (0x2A700...0x2B73F).contains(v) || (0x2B740...0x2B81F).contains(v) || (0x2B820...0x2CEAF).contains(v)
            || (0xF900...0xFAFF).contains(v) || (0x2F800...0x2FA1F).contains(v)
    }

    func bpe(_ token: String) -> [String] {
        lock.lock()
        if let c = cache[token] { lock.unlock(); return c }
        lock.unlock()
        var word = token.map { String($0) }
        guard !word.isEmpty else { return [] }
        word[word.count - 1] += "</w>"
        while word.count > 1 {
            var best: (rank: Int, i: Int)?
            for i in 0 ..< word.count - 1 {
                if let rk = ranks[word[i] + " " + word[i + 1]], rk < (best?.rank ?? .max) { best = (rk, i) }
            }
            guard let best else { break }
            let first = word[best.i], second = word[best.i + 1]
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == first, word[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
        }
        lock.lock(); cache[token] = word; lock.unlock()
        return word
    }

    /// Token ids without the start/end tokens.
    public func encodeBody(_ text: String) -> [Int] {
        let s = Self.clean(text)
        let ns = s as NSString
        var ids: [Int] = []
        for m in pattern.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            let piece = ns.substring(with: m.range)
            let mapped = String(piece.utf8.map { byteChars[Int($0)] })
            for t in bpe(mapped) { if let id = vocab[t] { ids.append(id) } }
        }
        return ids
    }

    /// `<|startoftext|>`, the text (truncated to fit `maxLength`) and `<|endoftext|>`, unpadded.
    public func encode(_ text: String) -> [Int] {
        [bos] + encodeBody(text).prefix(maxLength - 2) + [eos]
    }
}
