import Foundation
import Hub
import MLX
import MLXDiffusionCore
import T5Text
import Tokenizers

/// Prompt preprocessing of the official Wan tokenizer wrapper (`clean='whitespace'`): `ftfy.fix_text`, double HTML
/// unescape, whitespace collapsing. The ftfy part covers what matters for prompts — curly quotes, Latin
/// ligatures, full-width characters (the Chinese default negative prompt uses full-width commas) and NFC.
public enum WanPrompt {
    public static func clean(_ text: String) -> String {
        var s = fixText(text)
        s = htmlUnescape(htmlUnescape(s))
        return s.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
    }

    static func fixText(_ text: String) -> String {
        var out = String.UnicodeScalarView()
        let ligatures: [UInt32: String] = [0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl",
                                           0xFB05: "st", 0xFB06: "st", 0x0132: "IJ", 0x0133: "ij", 0x01C7: "LJ",
                                           0x01C8: "Lj", 0x01C9: "lj", 0x01CA: "NJ", 0x01CB: "Nj", 0x01CC: "nj"]
        for u in text.unicodeScalars {
            switch u.value {
            case 0x2018, 0x2019, 0x201A, 0x201B: out.append("'")
            case 0x201C, 0x201D, 0x201E, 0x201F: out.append("\"")
            case 0xFF01...0xFF5E: out.append(Unicode.Scalar(u.value - 0xFF01 + 0x21)!)
            case 0x3000: out.append(" ")
            default:
                if let l = ligatures[u.value] { out.append(contentsOf: l.unicodeScalars) } else { out.append(u) }
            }
        }
        return String(out).precomposedStringWithCanonicalMapping
    }

    static func htmlUnescape(_ s: String) -> String {
        guard s.contains("&") else { return s }
        let named = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": "\u{00A0}"]
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            if s[i] == "&", let semi = s[i...].prefix(12).firstIndex(of: ";") {
                let name = String(s[s.index(after: i)..<semi])
                var rep: String?
                if name.hasPrefix("#x") || name.hasPrefix("#X") {
                    rep = UInt32(name.dropFirst(2), radix: 16).flatMap(Unicode.Scalar.init).map { String($0) }
                } else if name.hasPrefix("#") {
                    rep = UInt32(name.dropFirst()).flatMap(Unicode.Scalar.init).map { String($0) }
                } else {
                    rep = named[name]
                }
                if let rep {
                    out += rep
                    i = s.index(after: semi)
                    continue
                }
            }
            out.append(s[i])
            i = s.index(after: i)
        }
        return out
    }
}

/// umT5-XXL prompt encoder for Wan: tokenizer + encoder, loaded only while encoding.
public final class WanTextEncoder {
    let weightsURL: URL
    let tokenizer: Tokenizer
    let maxTokens: Int

    /// `weights`: `t5_encoder.safetensors`; `tokenizerFile`: the umT5 `tokenizer.json`.
    public init(weights: URL, tokenizerFile: URL, maxTokens: Int = 512) throws {
        weightsURL = weights
        self.maxTokens = maxTokens
        var data = try JSONSerialization.jsonObject(with: Data(contentsOf: tokenizerFile)) as? [NSString: Any] ?? [:]
        data = Self.dedupingVocab(data)
        let config: [NSString: Any] = ["tokenizer_class": "T5Tokenizer", "eos_token": "</s>", "pad_token": "<pad>",
                                       "unk_token": "<unk>", "model_max_length": maxTokens]
        tokenizer = try AutoTokenizer.from(tokenizerConfig: Config(config), tokenizerData: Config(data))
    }

    /// The umT5 vocabulary lists a few pieces twice (e.g. `[`). Hugging Face's tokenizer maps such a piece to its last
    /// id; swift-transformers refuses duplicates. Earlier copies get an unmatchable name, so ids stay aligned and
    /// lookups agree with Hugging Face.
    static func dedupingVocab(_ data: [NSString: Any]) -> [NSString: Any] {
        guard var model = data["model"] as? [NSString: Any], var vocab = model["vocab"] as? [[Any]] else { return data }
        // Exact UTF-16 comparison, as swift-transformers keys its vocabulary (Swift `String` equality would merge
        // canonically equivalent pieces such as ";" and the Greek question mark).
        var last: [[UInt16]: Int] = [:]
        for (i, e) in vocab.enumerated() { if let t = e.first as? String { last[Array(t.utf16)] = i } }
        var changed = false
        for (i, e) in vocab.enumerated() {
            guard let t = e.first as? String, last[Array(t.utf16)] != i else { continue }
            vocab[i][0] = "\u{0}duplicate\(i)\u{0}"
            changed = true
        }
        guard changed else { return data }
        model["vocab"] = vocab
        var out = data
        out["model"] = model
        return out
    }

    /// Token ids of the cleaned prompt (with the closing `</s>`), truncated to the text length.
    public func tokens(_ prompt: String) -> [Int] {
        var ids = tokenizer.encode(text: WanPrompt.clean(prompt))
        if ids.count > maxTokens { ids = Array(ids.prefix(maxTokens - 1)) + [ids.last!] }
        return ids
    }

    /// Hidden states `[L, 4096]` for each prompt (unpadded). Loads the encoder for the call and frees it afterwards.
    public func encode(_ prompts: [String]) throws -> [MLXArray] {
        let encoder = try T5XXLEncoder(store: WeightStore(url: weightsURL))
        defer { Memory.clearCache() }
        return prompts.map { p in
            let ids = tokens(p)
            let h = encoder(MLXArray(ids.map { Int32($0) }, [1, ids.count]))[0]
            eval(h)
            return h
        }
    }
}
