import Foundation

/// Follow-up suggestions at the end of a reply, which the system prompt asks for in this form:
///
///     <followups>
///     - Try it at strength 1.0
///     - What's CFG?
///     </followups>
///
/// The app takes the block out of the text and shows the items as buttons.
public enum FollowUps {
    public static let maxItems = 4
    public static let maxLength = 80

    /// The text without the block, and its items. An unfinished block (from `<followups` on) is hidden too and gives
    /// no items; while a reply is `streaming`, so is a trailing `<`, `<fol`, … that may start one.
    public static func split(_ text: String, streaming: Bool = false) -> (text: String, items: [String]) {
        guard let open = text.range(of: "<followups", options: [.caseInsensitive, .backwards]) else {
            return (streaming ? trimPartialTag(text) : text, [])
        }
        let before = String(text[..<open.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let rest = text[open.upperBound...]
        guard let gt = rest.firstIndex(of: ">") else { return (before, []) }
        let inner = rest[rest.index(after: gt)...]
        guard let close = inner.range(of: "</followups>", options: .caseInsensitive) else { return (before, []) }
        let items = inner[..<close.lowerBound].split(whereSeparator: \.isNewline).compactMap { line -> String? in
            var l = line.trimmingCharacters(in: .whitespaces)
            for bullet in ["- ", "* ", "• "] where l.hasPrefix(bullet) { l = String(l.dropFirst(bullet.count)) }
            if let dot = l.firstIndex(of: "."), l[..<dot].allSatisfy(\.isNumber), !l[..<dot].isEmpty {
                l = String(l[l.index(after: dot)...])
            }
            l = l.trimmingCharacters(in: .whitespaces)
            guard !l.isEmpty else { return nil }
            return l.count > maxLength ? String(l.prefix(maxLength - 1)) + "…" : l
        }
        let after = String(inner[close.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        let kept = after.isEmpty ? before : before + "\n\n" + after
        return (kept, Array(items.prefix(maxItems)))
    }

    /// A trailing "<", "<fol", … that may be the start of the block, while it streams in.
    static func trimPartialTag(_ text: String) -> String {
        guard let lt = text.lastIndex(of: "<") else { return text }
        let tail = text[lt...].lowercased()
        if tail.count < "<followups".count, "<followups".hasPrefix(tail) {
            return String(text[..<lt]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return text
    }
}
