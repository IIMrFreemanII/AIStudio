import Foundation

/// Subtitles as SubRip text, and long lines cut into cards of at most two rows.
public enum MovieSubtitles {
    /// How long a row may be in the .srt (the usual limit for two-row subtitles).
    public static let rowLength = 42

    /// One subtitle as it goes on screen: one or two rows.
    public struct Card: Sendable, Equatable {
        public var start: Double
        public var end: Double
        public var rows: [String]
        public var text: String { rows.joined(separator: "\n") }
    }

    /// `speakers`: each line starts with who says it ("MARA: …"). Long lines are cut into cards of two rows of at
    /// most `rowLength` characters, one after another over the line's time.
    public static func srt(_ subtitles: [MovieTimeline.Subtitle], speakers: Bool = true,
                           rowLength: Int = rowLength) -> String {
        subtitles.sorted { $0.start < $1.start }
            .flatMap { s in cards(speakers ? s.labelled : s.shown, start: s.start, end: s.end, rowLength: rowLength) }
            .enumerated().map { i, c in "\(i + 1)\n\(stamp(c.start)) --> \(stamp(c.end))\n\(c.text)\n\n" }
            .joined()
    }

    /// `text` as cards of one or two rows of at most `rowLength` characters (a single longer word gets a row of its
    /// own), the two rows of a card as even as they can be. A line too long for one card is shown a card at a time,
    /// each for its share of the characters.
    public static func cards(_ text: String, start: Double, end: Double, rowLength: Int = rowLength) -> [Card] {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !words.isEmpty else { return [] }
        // Rows filled a word at a time, then paired into cards and evened out.
        var rows: [[String]] = [[]]
        for w in words {
            let row = rows[rows.count - 1]
            if !row.isEmpty, length(row) + 1 + w.count > rowLength { rows.append([w]) } else { rows[rows.count - 1].append(w) }
        }
        var groups: [[String]] = stride(from: 0, to: rows.count, by: 2).map { i in
            rows[i..<min(i + 2, rows.count)].flatMap { $0 }
        }
        // A last card of a word or two reads badly: take words from the card before while both still fit.
        if groups.count > 1 {
            let k = groups.count - 1
            while groups[k - 1].count > 1, length(groups[k]) < rowLength / 2,
                  length(groups[k]) + 1 + groups[k - 1].last!.count <= rowLength * 2 {
                groups[k].insert(groups[k - 1].removeLast(), at: 0)
            }
        }
        let total = Double(groups.reduce(0) { $0 + length($1) })
        var t = start
        return groups.enumerated().map { i, g in
            let d = (end - start) * Double(length(g)) / total
            let card = Card(start: t, end: i == groups.count - 1 ? end : t + d, rows: balanced(g, rowLength: rowLength))
            t += d
            return card
        }
    }

    /// Words as one row, or two rows that fit as evenly as they can.
    static func balanced(_ words: [String], rowLength: Int) -> [String] {
        guard length(words) > rowLength, words.count > 1 else { return [words.joined(separator: " ")] }
        var best: (cut: Int, score: Int)?
        for cut in 1..<words.count {
            let a = length(Array(words[..<cut])), b = length(Array(words[cut...]))
            // Rows that fit beat rows that don't; then the longer row as short as it can be.
            let score = (max(a, b) > rowLength ? 10_000 : 0) + max(a, b)
            if best == nil || score < best!.score { best = (cut, score) }
        }
        let cut = best!.cut
        return [words[..<cut].joined(separator: " "), words[cut...].joined(separator: " ")]
    }

    private static func length(_ words: [String]) -> Int {
        words.reduce(0) { $0 + $1.count } + max(0, words.count - 1)
    }

    static func stamp(_ t: Double) -> String {
        let ms = Int((max(0, t) * 1000).rounded())
        return String(format: "%02d:%02d:%02d,%03d", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
    }
}
