import Foundation

/// Metadata the planner writes into its `<think>` block (all optional; `parse` keeps what it finds).
public struct ACEPlanMetadata: Equatable {
    public var bpm: Int?
    public var caption: String?
    public var duration: Int?
    public var keyscale: String?
    public var language: String?
    /// As generated (e.g. "4"); the DiT metas use it verbatim.
    public var timeSignature: String?

    public init(bpm: Int? = nil, caption: String? = nil, duration: Int? = nil, keyscale: String? = nil,
                language: String? = nil, timeSignature: String? = nil) {
        self.bpm = bpm
        self.caption = caption
        self.duration = duration
        self.keyscale = keyscale
        self.language = language
        self.timeSignature = timeSignature
    }
}

/// Prompt strings and CoT text handling of `acestep/llm_inference.py` (`build_formatted_prompt`,
/// `build_formatted_prompt_with_cot`, `_build_unconditional_prompt`, `_format_metadata_as_cot`, `parse_lm_output`).
public enum ACEPlannerText {
    public static let lmInstruction = "Generate audio semantic tokens based on the given conditions:"
    public static let defaultNegativePrompt = "NO USER INPUT"

    /// The Qwen3 chat template with a system + user message and `add_generation_prompt=True`.
    public static func chat(user: String) -> String {
        "<|im_start|>system\n# Instruction\n\(lmInstruction)\n\n<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
    }

    static func hasMeaningfulNegative(_ s: String) -> Bool {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return !t.isEmpty && t != defaultNegativePrompt
    }

    /// Phase 1 (CoT) prompt.
    public static func cotPrompt(caption: String, lyrics: String) -> String {
        chat(user: "# Caption\n\(caption)\n\n# Lyric\n\(lyrics)\n")
    }

    /// Phase 1 unconditional prompt (unused: the official code runs the CoT phase without CFG).
    public static func cotUnconditionalPrompt(lyrics: String, negativePrompt: String = defaultNegativePrompt) -> String {
        hasMeaningfulNegative(negativePrompt)
            ? chat(user: "# Caption\n\(negativePrompt)\n\n# Lyric\n\(lyrics)\n")
            : chat(user: "# Lyric\n\(lyrics)\n")
    }

    /// Phase 2 (codes) prompt: the CoT prompt plus the formatted `<think>…</think>` block and `\n\n`.
    public static func codesPrompt(caption: String, lyrics: String, cot: String) -> String {
        cotPrompt(caption: caption, lyrics: lyrics) + cot + "\n\n"
    }

    /// Phase 2 unconditional (CFG) prompt: raw negative prompt as the user turn and an empty think block.
    public static func codesUnconditionalPrompt(negativePrompt: String = defaultNegativePrompt) -> String {
        let user = hasMeaningfulNegative(negativePrompt) ? negativePrompt : defaultNegativePrompt
        return chat(user: user) + "<think>\n\n</think>" + "\n\n"
    }

    // MARK: - CoT parsing / formatting

    /// `MetadataConstrainedLogitsProcessor.postprocess_caption`: strip each line, drop empty ones, join with spaces.
    public static func postprocessCaption(_ s: String) -> String {
        s.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// The metadata part of `parse_lm_output`.
    public static func parse(_ output: String) -> ACEPlanMetadata {
        var reasoning: String?
        if let a = output.range(of: "<think>"), let b = output.range(of: "</think>", range: a.upperBound..<output.endIndex) {
            reasoning = String(output[a.upperBound..<b.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if reasoning == nil || reasoning!.isEmpty {
            let before = output.components(separatedBy: "<|audio_code_").first ?? output
            reasoning = before.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var m = ACEPlanMetadata()
        guard let text = reasoning, !text.isEmpty else { return m }
        var key: String?
        var lines: [String] = []
        func save() {
            defer { key = nil; lines = [] }
            guard let k = key, !lines.isEmpty else { return }
            let value = lines.joined(separator: "\n")
            let v = value.trimmingCharacters(in: .whitespacesAndNewlines)
            switch k {
            case "bpm": m.bpm = Int(v)
            case "caption": m.caption = postprocessCaption(value)
            case "duration": m.duration = Int(v)
            case "keyscale": m.keyscale = v
            case "language": m.language = v
            case "timesignature": m.timeSignature = v
            default: break
            }
        }
        for line in text.components(separatedBy: "\n") {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("<") { continue }
            if let f = line.unicodeScalars.first, !CharacterSet.whitespacesAndNewlines.contains(f), line.contains(":") {
                save()
                let parts = line.split(separator: ":", maxSplits: 1, omittingEmptySubsequences: false)
                if parts.count == 2 {
                    key = parts[0].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    let first = String(parts[1])
                    if !first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lines.append(first) }
                }
            } else if line.hasPrefix(" ") || line.hasPrefix("\t") {
                if key != nil { lines.append(line) }
            }
        }
        save()
        return m
    }

    /// `_format_metadata_as_cot`: `<think>\n{yaml.dump(sorted, allow_unicode)}\n</think>`.
    public static func formatCoT(_ m: ACEPlanMetadata) -> String {
        var items: [(String, YAMLValue)] = []
        if let v = m.bpm { items.append(("bpm", .int(v))) }
        if let v = m.caption { items.append(("caption", .str(v))) }
        if let v = m.duration { items.append(("duration", .int(v))) }
        if let v = m.keyscale { items.append(("keyscale", strOrInt(v))) }
        if let v = m.language { items.append(("language", strOrInt(v))) }
        if var v = m.timeSignature {
            if v.hasSuffix("/4") { v = String(v.split(separator: "/", omittingEmptySubsequences: false)[0]) }
            items.append(("timesignature", strOrInt(v)))
        }
        let yaml = items.isEmpty ? "" : PyYAML.dumpMapping(items).trimmingCharacters(in: .whitespacesAndNewlines)
        return "<think>\n\(yaml)\n</think>"
    }

    /// Python `str.isdigit()` → int conversion done before dumping.
    static func strOrInt(_ s: String) -> YAMLValue {
        if !s.isEmpty, s.unicodeScalars.allSatisfy({ $0.properties.numericType == .decimal || $0.properties.numericType == .digit }),
           let i = Int(s) {
            return .int(i)
        }
        return .str(s)
    }
}

enum YAMLValue {
    case int(Int)
    case str(String)
}

/// The slice of PyYAML's emitter that `yaml.dump({...}, allow_unicode=True, sort_keys=True)` exercises for a flat
/// mapping of ints and single-line strings: plain / single-quoted / double-quoted scalar choice and line folding at
/// width 80 with a 2-space continuation indent.
enum PyYAML {
    static let bestWidth = 80
    static let indent = 2

    static func dumpMapping(_ items: [(String, YAMLValue)]) -> String {
        var out = ""
        for (k, v) in items.sorted(by: { $0.0 < $1.0 }) {
            out += k + ":"
            switch v {
            case .int(let i): out += " \(i)\n"
            case .str(let s): out += scalar(s, column: k.unicodeScalars.count + 1) + "\n"
            }
        }
        return out
    }

    struct Analysis {
        var empty = false, multiline = false
        var allowBlockPlain = true, allowSingleQuoted = true
    }

    static let breaks: Set<Unicode.Scalar> = ["\n", "\u{85}", "\u{2028}", "\u{2029}"]

    static func analyze(_ s: [Unicode.Scalar]) -> Analysis {
        var a = Analysis()
        if s.isEmpty { a.empty = true; return a }
        var blockIndicators = false, lineBreaks = false, special = false
        var leadingSpace = false, leadingBreak = false, trailingSpace = false, trailingBreak = false
        var breakSpace = false, spaceBreak = false
        let str = String(String.UnicodeScalarView(s))
        if str.hasPrefix("---") || str.hasPrefix("...") { blockIndicators = true }
        let ws: Set<Unicode.Scalar> = ["\0", " ", "\t", "\r", "\n", "\u{85}", "\u{2028}", "\u{2029}"]
        var precededByWS = true
        var followedByWS = s.count == 1 || ws.contains(s[1])
        var prevSpace = false, prevBreak = false
        for (i, ch) in s.enumerated() {
            if i == 0 {
                if "#,[]{}&*!|>'\"%@`".unicodeScalars.contains(ch) { blockIndicators = true }
                if "?:".unicodeScalars.contains(ch), followedByWS { blockIndicators = true }
                if ch == "-", followedByWS { blockIndicators = true }
            } else {
                if ch == ":", followedByWS { blockIndicators = true }
                if ch == "#", precededByWS { blockIndicators = true }
            }
            if breaks.contains(ch) { lineBreaks = true }
            let v = ch.value
            if !(ch == "\n" || (0x20...0x7E).contains(v)) {
                let uni = (v == 0x85 || (0xA0...0xD7FF).contains(v) || (0xE000...0xFFFD).contains(v)
                    || (0x10000..<0x10FFFF).contains(v)) && v != 0xFEFF
                if !uni { special = true }
            }
            if ch == " " {
                if i == 0 { leadingSpace = true }
                if i == s.count - 1 { trailingSpace = true }
                if prevBreak { breakSpace = true }
                prevSpace = true; prevBreak = false
            } else if breaks.contains(ch) {
                if i == 0 { leadingBreak = true }
                if i == s.count - 1 { trailingBreak = true }
                if prevSpace { spaceBreak = true }
                prevSpace = false; prevBreak = true
            } else {
                prevSpace = false; prevBreak = false
            }
            precededByWS = ws.contains(ch)
            followedByWS = i + 2 >= s.count || ws.contains(s[i + 2])
        }
        if leadingSpace || leadingBreak || trailingSpace || trailingBreak { a.allowBlockPlain = false }
        if breakSpace { a.allowBlockPlain = false; a.allowSingleQuoted = false }
        if spaceBreak || special { a.allowBlockPlain = false; a.allowSingleQuoted = false }
        if lineBreaks { a.allowBlockPlain = false; a.multiline = true }
        if blockIndicators { a.allowBlockPlain = false }
        return a
    }

    /// Whether PyYAML's resolver would read the plain scalar back as something other than a string.
    static func resolvesImplicitly(_ s: String) -> Bool {
        let patterns = [
            "^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$",
            "^(?:[-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+][0-9]+)?|\\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*|[-+]?\\.(?:inf|Inf|INF)|\\.(?:nan|NaN|NAN))$",
            "^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
            "^(?:<<)$",
            "^(?:~|null|Null|NULL|)$",
            "^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \\t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\\.[0-9]*)?(?:[ \\t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$",
            "^(?:=)$",
        ]
        return patterns.contains { s.range(of: $0, options: .regularExpression) != nil }
    }

    /// A mapping value scalar starting at `column` (just after the colon), including the leading space.
    static func scalar(_ text: String, column: Int) -> String {
        let s = Array(text.unicodeScalars)
        let a = analyze(s)
        if !resolvesImplicitly(text), a.allowBlockPlain { return plain(s, column: column) }
        if a.allowSingleQuoted { return singleQuoted(s, column: column) }
        return doubleQuoted(text)
    }

    static func width(_ s: ArraySlice<Unicode.Scalar>) -> Int { s.count }
    static func str(_ s: ArraySlice<Unicode.Scalar>) -> String { String(String.UnicodeScalarView(s)) }

    static func plain(_ s: [Unicode.Scalar], column startColumn: Int) -> String {
        var out = " "
        var column = startColumn + 1
        var spaces = false
        var start = 0, end = 0
        while end <= s.count {
            let ch: Unicode.Scalar? = end < s.count ? s[end] : nil
            if spaces {
                if ch != " " {
                    if start + 1 == end && column > bestWidth {
                        out += "\n" + String(repeating: " ", count: indent)
                        column = indent
                    } else {
                        out += str(s[start..<end]); column += width(s[start..<end])
                    }
                    start = end
                }
            } else if ch == nil || ch == " " {
                out += str(s[start..<end]); column += width(s[start..<end])
                start = end
            }
            if let ch { spaces = ch == " " }
            end += 1
        }
        return out
    }

    static func singleQuoted(_ s: [Unicode.Scalar], column startColumn: Int) -> String {
        var out = " '"
        var column = startColumn + 2
        var spaces = false, inBreaks = false
        var start = 0, end = 0
        while end <= s.count {
            let ch: Unicode.Scalar? = end < s.count ? s[end] : nil
            if spaces {
                if ch == nil || ch != " " {
                    if start + 1 == end && column > bestWidth && start != 0 && end != s.count {
                        out += "\n" + String(repeating: " ", count: indent)
                        column = indent
                    } else {
                        out += str(s[start..<end]); column += width(s[start..<end])
                    }
                    start = end
                }
            } else if inBreaks {
                if ch == nil || !breaks.contains(ch!) {
                    if s[start] == "\n" { out += "\n" }
                    for br in s[start..<end] { out.unicodeScalars.append(br) }
                    out += String(repeating: " ", count: indent)
                    column = indent
                    start = end
                }
            } else if ch == nil || ch == " " || ch == "'" || breaks.contains(ch!) {
                if start < end {
                    out += str(s[start..<end]); column += width(s[start..<end])
                    start = end
                }
            }
            if ch == "'" {
                out += "''"; column += 2
                start = end + 1
            }
            if let ch { spaces = ch == " "; inBreaks = breaks.contains(ch) }
            end += 1
        }
        return out + "'"
    }

    /// Rare fallback (non-printable characters): JSON-style escaping without folding.
    static func doubleQuoted(_ text: String) -> String {
        var out = " \""
        for u in text.unicodeScalars {
            switch u {
            case "\"": out += "\\\""
            case "\\": out += "\\\\"
            case "\n": out += "\\n"
            case "\t": out += "\\t"
            case "\r": out += "\\r"
            case "\0": out += "\\0"
            default:
                if u.value < 0x20 || u.value == 0x7F || u.value == 0xFEFF {
                    out += String(format: u.value <= 0xFF ? "\\x%02X" : "\\u%04X", u.value)
                } else {
                    out.unicodeScalars.append(u)
                }
            }
        }
        return out + "\""
    }
}
