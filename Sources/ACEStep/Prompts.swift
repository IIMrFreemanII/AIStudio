import Foundation

/// The text ACE-Step's DiT is conditioned on, formatted exactly like the official handler
/// (`SFT_GEN_PROMPT`, `_dict_to_meta_string`, `_format_lyrics` in ace-step/ACE-Step-1.5).
public enum ACEPrompts {
    /// Text-to-music instruction (`DEFAULT_DIT_INSTRUCTION`).
    public static let ditInstruction = "Fill the audio semantic mask based on the given conditions:"

    /// The caption prompt: instruction, caption and the "# Metas" block, ending in `<|endoftext|>` (the tokenizer adds
    /// its own EOS after it, as in the reference).
    public static func caption(instruction: String = ditInstruction, caption: String, metas: String) -> String {
        let instr = instruction.hasSuffix(":") ? instruction : instruction + ":"
        return "# Instruction\n\(instr)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// The "# Metas" block; nil or empty fields print as N/A. Duration is in whole seconds.
    public static func metas(bpm: Int?, timeSignature: String?, keyScale: String?, seconds: Double) -> String {
        func field(_ s: String?) -> String { (s?.isEmpty ?? true) ? "N/A" : s! }
        return "- bpm: \(bpm.map(String.init) ?? "N/A")\n- timesignature: \(field(timeSignature))\n"
            + "- keyscale: \(field(keyScale))\n- duration: \(Int(seconds)) seconds\n"
    }

    /// The lyric prompt with its language header ("unknown" when not given).
    public static func lyrics(_ lyrics: String, language: String?) -> String {
        let lang = (language?.isEmpty ?? true) ? "unknown" : language!
        return "# Languages\n\(lang)\n\n# Lyric\n\(lyrics)<|endoftext|>"
    }

    /// Lyrics as the model expects them: trimmed, "[Instrumental]" when empty.
    public static func normalizedLyrics(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? "[Instrumental]" : t
    }
}
