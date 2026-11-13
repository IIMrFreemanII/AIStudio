import Foundation

/// A failed reply's error in one line for the chat's error card; the server's own words stay behind Details.
public enum ReplyError {
    /// The one-line summary, or nil when the error already is one short line (show it as it is).
    public static func summary(_ error: String) -> String? {
        let lower = error.lowercased()
        if lower.contains("maximum context length") || lower.contains("context length") && lower.contains("exceed") {
            if let n = number(after: "maximum context length is", in: lower) {
                return "The chat is longer than this model's \(tokens(n)) context"
            }
            return "The chat is longer than this model's context"
        }
        if lower.contains("timed out") || lower.contains("timeout") { return "The model server took too long to answer" }
        if lower.contains("could not connect") || lower.contains("couldn't connect") || lower.contains("connection refused")
            || lower.contains("network connection was lost") || lower.contains("offline") {
            return "Couldn't reach the model server"
        }
        if let code = number(after: "http ", in: lower), (500...599).contains(code) {
            return "The model server failed (HTTP \(code))"
        }
        if lower.contains("401") || lower.contains("403") || lower.contains("unauthorized") || lower.contains("forbidden") {
            return "The model server refused the token"
        }
        // Short enough already: one sentence, one line.
        let firstLine = error.split(separator: "\n").first.map(String.init) ?? error
        if firstLine.count <= 90 && !error.contains("\n") { return nil }
        let sentence = firstLine.split(separator: ".", maxSplits: 1).first.map(String.init) ?? firstLine
        return sentence.count <= 90 ? sentence : String(sentence.prefix(88)) + "…"
    }

    /// "32768" → "32K", "131072" → "128K", "8000" → "8K".
    static func tokens(_ n: Int) -> String {
        n >= 1024 && n % 1024 == 0 ? "\(n / 1024)K" : n >= 1000 ? "\(Int((Double(n) / 1000).rounded()))K" : "\(n)"
    }

    static func number(after marker: String, in text: String) -> Int? {
        guard let r = text.range(of: marker) else { return nil }
        let digits = text[r.upperBound...].drop { $0 == " " }.prefix { $0.isNumber }
        return Int(digits)
    }
}
