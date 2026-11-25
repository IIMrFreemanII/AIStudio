import Foundation

/// Every character must be an adult: this refuses youth descriptors and ages under 18.
public enum AgeGuard {
    public static let minimumAge = 18

    /// Youth descriptors matched as whole words, case-insensitive (ages under 18 are matched separately).
    public static let youthTerms: [String] = [
        "child", "children", "kid", "kids", "minor", "minors", "teen", "teens", "teenage", "teenager", "underage",
        "preteen", "pre-teen", "loli", "lolita", "shota", "toddler", "infant", "schoolgirl", "schoolboy",
        "middle school", "elementary school", "high school student",
    ]

    /// Terms for image negatives.
    public static let negative = "child, minor, teen, underage, young-looking, childlike"

    private static let termRegex: NSRegularExpression = {
        let alternation = youthTerms.map { NSRegularExpression.escapedPattern(for: $0).replacingOccurrences(of: " ", with: "\\s+") }
            .joined(separator: "|")
        return try! NSRegularExpression(pattern: "(?<![\\p{L}\\p{N}])(?:\(alternation))(?![\\p{L}\\p{N}])",
                                        options: .caseInsensitive)
    }()

    private static let ageRegex = try! NSRegularExpression(
        pattern: "(?<![\\p{N}])(1[0-7]|[1-9])\\s*(?:yo|y/o|-\\s*year-old|\\s+years?\\s+old|\\s+year\\s+old)(?![\\p{L}])",
        options: .caseInsensitive)

    /// The first youth descriptor in the text, nil when there is none.
    public static func youthTerm(in text: String) -> String? {
        let range = NSRange(text.startIndex..., in: text)
        var best: (Int, String)?
        for re in [termRegex, ageRegex] {
            if let m = re.firstMatch(in: text, range: range), let r = Range(m.range, in: text) {
                if best == nil || m.range.location < best!.0 { best = (m.range.location, String(text[r])) }
            }
        }
        return best?.1
    }

    /// What stops a character from being saved.
    public static func problems(_ c: Character) -> [String] {
        var out: [String] = []
        if c.age < minimumAge { out.append("Age must be \(minimumAge) or older.") }
        let fields: [(String, String)] = [("Name", c.name), ("Look", c.look), ("Personality", c.personality),
                                          ("Backstory", c.backstory), ("Speaking style", c.speakingStyle),
                                          ("Likes", c.likes), ("Greeting", c.greeting)]
        for (field, text) in fields {
            if let term = youthTerm(in: text) { out.append("\(field) describes someone underage (\"\(term)\").") }
        }
        return out
    }
}
