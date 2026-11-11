import Foundation

/// One Server-Sent Event: its `event:` name (nil for the default "message") and its `data:` lines joined by "\n".
public struct SSEEvent: Equatable, Sendable {
    public var event: String?
    public var data: String
    public init(event: String? = nil, data: String) {
        self.event = event
        self.data = data
    }
}

/// Turns the lines of a `text/event-stream` into events, as the HTML spec says: `field: value` lines, a blank line
/// ends an event, lines starting with ":" are comments (keep-alive pings). Feed it lines without their newline.
public struct SSEParser: Sendable {
    private var event: String?
    private var data: [String] = []
    private var sawField = false
    /// Comments seen (pings), for callers that care.
    public private(set) var comments = 0

    public init() {}

    /// The event this line completes, if any.
    public mutating func feed(line raw: String) -> SSEEvent? {
        let line = raw.hasSuffix("\r") ? String(raw.dropLast()) : raw
        if line.isEmpty { return flush() }
        if line.hasPrefix(":") {
            comments += 1
            return nil
        }
        let field: Substring
        var value: Substring
        if let colon = line.firstIndex(of: ":") {
            field = line[..<colon]
            value = line[line.index(after: colon)...]
            if value.first == " " { value = value.dropFirst() }
        } else {
            field = Substring(line)
            value = ""
        }
        switch field {
        case "event": event = String(value); sawField = true
        case "data": data.append(String(value)); sawField = true
        default: break  // id, retry: unused here
        }
        return nil
    }

    /// The event in progress at the end of the stream (servers may omit the last blank line).
    public mutating func flush() -> SSEEvent? {
        defer { event = nil; data = []; sawField = false }
        guard sawField, !data.isEmpty else { return nil }
        return SSEEvent(event: event, data: data.joined(separator: "\n"))
    }
}
