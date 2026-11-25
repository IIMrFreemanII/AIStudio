import Foundation

/// A photo, video or voice note the model asked for inside a message.
public struct MediaRequest: Equatable, Sendable {
    public var kind: MediaKind
    public var text: String
    /// 0-based order in the message.
    public var index: Int
    public init(kind: MediaKind, text: String, index: Int) {
        self.kind = kind
        self.text = text
        self.index = index
    }
}

/// Tags the model writes inline: `<photo>…</photo>`, `<video>…</video>`, `<voice>…</voice>`, `<talk>…</talk>`.
public enum MediaTags {
    public enum Segment: Equatable, Sendable {
        case text(String)
        case media(MediaRequest)
        case pending(MediaKind)
    }

    private enum Piece {
        case text(String)
        case media(MediaKind, String)
        case pending(MediaKind)
    }

    /// Splits `text` into pieces. An unclosed open tag becomes `.pending` (streaming) and swallows the rest.
    private static func scan(_ text: String) -> [Piece] {
        var pieces: [Piece] = []
        var buffer = ""
        var i = text.startIndex
        while i < text.endIndex {
            if text[i] == "<", let (kind, afterOpen) = openTag(in: text, at: i) {
                let closeTag = "</\(kind.rawValue)>"
                if let r = text.range(of: closeTag, options: .caseInsensitive, range: afterOpen..<text.endIndex) {
                    if !buffer.isEmpty { pieces.append(.text(buffer)); buffer = "" }
                    pieces.append(.media(kind, String(text[afterOpen..<r.lowerBound])
                        .trimmingCharacters(in: .whitespacesAndNewlines)))
                    i = r.upperBound
                    continue
                } else {
                    if !buffer.isEmpty { pieces.append(.text(buffer)); buffer = "" }
                    pieces.append(.pending(kind))
                    return pieces
                }
            }
            buffer.append(text[i])
            i = text.index(after: i)
        }
        if !buffer.isEmpty { pieces.append(.text(buffer)) }
        return pieces
    }

    private static func openTag(in text: String, at i: String.Index) -> (MediaKind, String.Index)? {
        for kind in MediaKind.allCases {
            let tag = "<\(kind.rawValue)>"
            if let r = text.range(of: tag, options: [.caseInsensitive, .anchored], range: i..<text.endIndex) {
                return (kind, r.upperBound)
            }
        }
        return nil
    }

    private static func tidy(_ s: String) -> String {
        var t = s.replacingOccurrences(of: "[ \\t]{2,}", with: " ", options: .regularExpression)
        t = t.replacingOccurrences(of: "[ \\t]*\\n[ \\t]*", with: "\n", options: .regularExpression)
        t = t.replacingOccurrences(of: "\\n{3,}", with: "\n\n", options: .regularExpression)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A finished message: the text without complete tags, and the requests in order. An unclosed tag stays text.
    public static func parse(_ text: String) -> (text: String, requests: [MediaRequest]) {
        var out = ""
        var requests: [MediaRequest] = []
        for p in scan(text) {
            switch p {
            case .text(let t): out += t
            case .media(let k, let body):
                requests.append(MediaRequest(kind: k, text: body, index: requests.count))
            case .pending:
                // Unclosed: keep the rest of the message as text.
                if let r = unclosedStart(text) { out += String(text[r...]) }
            }
        }
        return (tidy(out), requests)
    }

    /// Where the trailing unclosed open tag starts.
    private static func unclosedStart(_ text: String) -> String.Index? {
        var i = text.startIndex
        while i < text.endIndex {
            if text[i] == "<", let (kind, after) = openTag(in: text, at: i) {
                if let r = text.range(of: "</\(kind.rawValue)>", options: .caseInsensitive, range: after..<text.endIndex) {
                    i = r.upperBound
                    continue
                }
                return i
            }
            i = text.index(after: i)
        }
        return nil
    }

    /// For showing a message, also while it streams.
    public static func segments(_ text: String) -> [Segment] {
        var out: [Segment] = []
        var index = 0
        var pieces = scan(text)
        // Hide a trailing partial opening tag ("<", "<ph", "<photo").
        if case .text(let t)? = pieces.last, let cut = partialOpenStart(t) {
            pieces.removeLast()
            pieces.append(.text(String(t[..<cut])))
        }
        for p in pieces {
            switch p {
            case .text(let t):
                let s = tidy(t)
                if !s.isEmpty { out.append(.text(s)) }
            case .media(let k, let body):
                out.append(.media(MediaRequest(kind: k, text: body, index: index)))
                index += 1
            case .pending(let k):
                out.append(.pending(k))
            }
        }
        return out
    }

    /// The index of a trailing "<…" that could still become an opening tag.
    private static func partialOpenStart(_ t: String) -> String.Index? {
        guard let lt = t.lastIndex(of: "<") else { return nil }
        let tail = t[lt...].lowercased()
        if tail.contains(">") { return nil }
        for kind in MediaKind.allCases where "<\(kind.rawValue)".hasPrefix(tail) { return lt }
        return nil
    }

    /// The tags' instructions for the system prompt, only for `kinds` (none: "").
    public static func instructions(_ kinds: [MediaKind]) -> String {
        let wanted = MediaKind.allCases.filter { kinds.contains($0) }
        if wanted.isEmpty { return "" }
        var lines = ["You can send media by writing a tag inside your message, where it should appear:"]
        for k in wanted {
            switch k {
            case .photo:
                lines.append("- <photo>what the photo shows: your outfit, pose, expression and the place</photo> sends a photo of you. "
                    + "Write it as a short caption without \"you\", \"I\" or your name, starting with the pose or the place.")
            case .video:
                lines.append("- <video>what happens in a few seconds of video of you</video> sends a short video of you.")
            case .voice:
                lines.append("- <voice>the exact words you say, a sentence or two</voice> sends a voice message.")
            case .talk:
                lines.append("- <talk>the exact words you say on camera, one or two short sentences</talk> sends a video of you saying them.")
            }
        }
        lines.append("Use at most one tag per message, and never describe or mention the tag itself.")
        return lines.joined(separator: "\n")
    }
}
