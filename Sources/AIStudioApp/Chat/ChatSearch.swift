import Foundation
import LLMClient

extension ChatStore {
    /// Saved chats whose text contains `query` (case-insensitive): id → a snippet of about 120 characters around the
    /// first match. Reads the files directly, so it is safe off the main actor. At most 50 results.
    nonisolated func search(_ query: String) -> [UUID: String] {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return [:] }
        let files = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var found: [UUID: String] = [:]
        for u in files where u.pathExtension == "json" {
            if found.count >= 50 { break }
            guard let data = try? Data(contentsOf: u), let t = try? decoder.decode(ChatTranscript.self, from: data) else { continue }
            for turn in t.nodes {
                guard let r = turn.text.range(of: q, options: [.caseInsensitive, .diacriticInsensitive]) else { continue }
                found[t.id] = Self.snippet(turn.text, around: r)
                break
            }
        }
        return found
    }

    private nonisolated static func snippet(_ text: String, around r: Range<String.Index>, length: Int = 120) -> String {
        let matchLen = text.distance(from: r.lowerBound, to: r.upperBound)
        let before = max(0, (length - matchLen) / 3)
        let start = text.index(r.lowerBound, offsetBy: -before, limitedBy: text.startIndex) ?? text.startIndex
        let end = text.index(start, offsetBy: length, limitedBy: text.endIndex) ?? text.endIndex
        var s = String(text[start..<end])
            .replacingOccurrences(of: "\r\n", with: " ").replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        s = s.trimmingCharacters(in: .whitespaces)
        return (start > text.startIndex ? "…" : "") + s + (end < text.endIndex ? "…" : "")
    }
}
