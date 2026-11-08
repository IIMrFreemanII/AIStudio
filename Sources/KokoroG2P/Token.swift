import Foundation

/// One output token of `EnglishG2P.phonemize`, mirroring misaki's `MToken`.
public struct Token: Sendable, Equatable, CustomStringConvertible {
    /// Source text of the token (after misaki's sub-tokenization; e.g. "don't", "$", "5.50").
    public var text: String
    /// Penn Treebank-style tag (spaCy `tag_` equivalent, approximated with NaturalLanguage + rules).
    public var tag: String
    /// Whitespace that followed the token in the input ("" or " ").
    public var whitespace: String
    /// Phonemes in Kokoro's alphabet; nil when the word is unknown and the fallback is disabled.
    public var phonemes: String?
    /// True when `phonemes` came from the rule-based letter-to-sound fallback (not a lexicon).
    public var isFallback: Bool
    /// misaki's confidence rating: 5 user-supplied, 4 gold lexicon, 3 silver/derived, 1 fallback.
    public var rating: Int?

    public var description: String { "\(text)/\(tag)→\(phonemes ?? "nil")" }
}

/// misaki `TokenContext`: what follows the current word (we walk the sentence right-to-left).
struct TokenContext {
    /// nil = followed by punctuation/end, true = next sound is a vowel, false = a consonant.
    var futureVowel: Bool? = nil
    var futureTo: Bool = false
}

/// Mutable working token (misaki `MToken` + its `_` underscore attributes).
final class MToken {
    var text: String
    var tag: String
    var whitespace: String
    var phonemes: String?
    // underscore attributes
    var isHead: Bool = true
    var alias: String? = nil
    var stress: Double? = nil
    var currency: String? = nil
    var numFlags: String = ""
    var prespace: Bool = false
    var rating: Int? = nil
    var isFallback: Bool = false

    init(text: String, tag: String, whitespace: String, phonemes: String? = nil) {
        self.text = text
        self.tag = tag
        self.whitespace = whitespace
        self.phonemes = phonemes
    }

    func copy() -> MToken {
        let t = MToken(text: text, tag: tag, whitespace: whitespace, phonemes: phonemes)
        t.isHead = isHead; t.alias = alias; t.stress = stress; t.currency = currency
        t.numFlags = numFlags; t.prespace = prespace; t.rating = rating; t.isFallback = isFallback
        return t
    }

    var publicToken: Token {
        Token(text: text, tag: tag, whitespace: whitespace, phonemes: phonemes, isFallback: isFallback, rating: rating)
    }
}

/// misaki `merge_tokens`.
func mergeTokens(_ tokens: [MToken], unk: String?) -> MToken {
    let stresses = Set(tokens.compactMap { $0.stress })
    let currencies = tokens.compactMap { $0.currency }
    var phonemes: String? = nil
    if let unk {
        var ps = ""
        for tk in tokens {
            if tk.prespace, let last = ps.unicodeScalars.last, !last.pyIsSpace, let p = tk.phonemes, !p.isEmpty {
                ps += " "
            }
            ps += tk.phonemes ?? unk
        }
        phonemes = ps
    }
    var text = ""
    for tk in tokens.dropLast() { text += tk.text + tk.whitespace }
    text += tokens.last!.text
    // Python max() returns the first maximal element.
    var best = tokens[0]
    var bestW = -1
    for tk in tokens {
        let w = tk.text.unicodeScalars.reduce(0) { $0 + (String($1) == String($1).lowercased() ? 1 : 2) }
        if w > bestW { bestW = w; best = tk }
    }
    let m = MToken(text: text, tag: best.tag, whitespace: tokens.last!.whitespace, phonemes: phonemes)
    m.isHead = tokens[0].isHead
    m.alias = nil
    m.stress = stresses.count == 1 ? stresses.first : nil
    m.currency = currencies.max()
    m.numFlags = String(Set(tokens.flatMap { Array($0.numFlags) }).sorted())
    m.prespace = tokens[0].prespace
    m.rating = tokens.contains(where: { $0.rating == nil }) ? nil : tokens.compactMap { $0.rating }.min()
    m.isFallback = tokens.contains(where: { $0.isFallback })
    return m
}
