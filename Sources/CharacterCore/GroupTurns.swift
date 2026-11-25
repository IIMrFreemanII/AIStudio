import Foundation

/// Who answers in a group chat, and how the others' messages read.
public enum GroupTurns {
    public static let maxReplies = 3

    /// Where `word` first appears in `text` as a whole word (case-insensitive).
    private static func firstMatch(_ word: String, in text: String) -> Int? {
        let w = word.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !w.isEmpty,
              let re = try? NSRegularExpression(pattern: "(?<![\\p{L}\\p{N}_])\(NSRegularExpression.escapedPattern(for: w))(?![\\p{L}\\p{N}_])",
                                                options: .caseInsensitive) else { return nil }
        let ns = text as NSString
        return re.firstMatch(in: text, range: NSRange(location: 0, length: ns.length))?.range.location
    }

    /// Who answers the user's message, in order: the characters it names (whole words, by full or first name), in
    /// the order it names them; else everyone in turn, starting after the one who answered first last time
    /// (`lastFirst`). At most `maxReplies`.
    public static func speakers(for text: String, cast: [Character], lastFirst: UUID?) -> [UUID] {
        if cast.isEmpty { return [] }
        if cast.count == 1 { return [cast[0].id] }
        var named: [(pos: Int, id: UUID)] = []
        for c in cast {
            let name = c.name.trimmingCharacters(in: .whitespacesAndNewlines)
            var hits: [Int] = []
            if let p = firstMatch(name, in: text) { hits.append(p) }
            if let first = name.split(whereSeparator: \.isWhitespace).first, String(first) != name,
               let p = firstMatch(String(first), in: text) { hits.append(p) }
            if let p = hits.min() { named.append((p, c.id)) }
        }
        if !named.isEmpty {
            return Array(named.sorted { $0.pos < $1.pos }.map(\.id).prefix(maxReplies))
        }
        let start = lastFirst.flatMap { id in cast.firstIndex { $0.id == id } }.map { ($0 + 1) % cast.count } ?? 0
        let rotated = (0..<cast.count).map { cast[(start + $0) % cast.count].id }
        return Array(rotated.prefix(maxReplies))
    }

    /// Removes one leading `Name:`, `**Name:**`, `**Name**:`, `[Name]:` or `[Name]` and the whitespace after it.
    public static func stripName(_ text: String, name: String) -> String {
        let n = NSRegularExpression.escapedPattern(for: name)
        let pattern = "^\\s*(?:\\*\\*\(n):\\*\\*|\\*\\*\(n)\\*\\*:|\\[\(n)\\]:?|\(n):)\\s*"
        guard let re = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
              let m = re.firstMatch(in: text, range: NSRange(location: 0, length: (text as NSString).length)) else { return text }
        return (text as NSString).substring(from: m.range.upperBound)
    }

    /// How another character's message reads in this character's history.
    public static func line(_ name: String, _ text: String) -> String {
        "\(name): \(MediaTags.plain(text))"
    }
}
