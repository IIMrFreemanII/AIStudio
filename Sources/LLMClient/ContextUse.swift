import Foundation

/// Rough token counts, when the server hasn't counted: characters / 4, and 1,000 per image or video.
public enum TokenEstimate {
    public static let perMedia = 1_000

    public static func text(_ s: String) -> Int { s.count / 4 }

    /// Text and tool-call arguments / 4, plus 1,000 per image or video.
    public static func of(_ messages: [ChatMessage]) -> Int {
        var chars = 0
        var media = 0
        for m in messages {
            for p in m.parts {
                switch p {
                case .text(let t): chars += t.count
                case .image, .video: media += 1
                }
            }
            for c in m.toolCalls { chars += c.arguments.count }
        }
        return chars / 4 + media * perMedia
    }
}

/// How full a chat's context is: the next request's prompt, as far as it's known.
public struct ContextUse: Equatable, Sendable {
    /// Tokens in use.
    public var tokens: Int
    /// Counted by the server (the last reply's prompt and completion) with nothing estimated on top.
    public var exact: Bool
    /// What it's made of (estimates; only with `breakdown: true`). The system prompt and tool definitions are what's
    /// left of `tokens` after the rest.
    public var breakdown: Breakdown?

    public struct Breakdown: Equatable, Sendable {
        public var system: Int
        public var messages: Int
        /// Tool calls and their results.
        public var tools: Int
        /// Images and videos.
        public var media: Int
    }
}

extension ChatTranscript {
    /// How full the context is, for the meter under the composer. The last reply on the shown branch (after the latest
    /// summary) that the server counted is exact; what came after it (the message just sent, a reply still streaming,
    /// tool results) is estimated. Without a counted reply: `overhead` (the system prompt and tool definitions, as the
    /// last run estimated them) plus the estimate of the messages.
    public func contextUse(overhead: Int? = nil, breakdown: Bool = false) -> ContextUse {
        let shown = turns
        let from = shown.lastIndex { $0.summary == true } ?? shown.startIndex
        let sent = Array(shown[from...]).filter { $0.error == nil || !$0.text.isEmpty || $0.sessionEvent != nil || $0.flowEvent != nil }
        // Only the last two tool results with screenshots send them (`messages`).
        let keepShots = Set(sent.filter { $0.role == .tool && !$0.images.isEmpty }.map(\.id).suffix(2))

        func cost(_ t: ChatTurn) -> (messages: Int, tools: Int, media: Int) {
            if let e = t.flowEvent { return (TokenEstimate.text(e.textForModel), 0, 0) }
            if let e = t.sessionEvent { return (TokenEstimate.text(e.textForModel), 0, 0) }
            if t.summary == true { return (TokenEstimate.text(ChatTurn.summaryPrefix + t.text), 0, 0) }
            switch t.role {
            case .user:
                return (TokenEstimate.text(t.textForModel), 0, t.images.count * TokenEstimate.perMedia)
            case .assistant:
                return (TokenEstimate.text(t.text), t.toolCalls.reduce(0) { $0 + $1.arguments.count } / 4, 0)
            case .tool:
                let shots = keepShots.contains(t.id) ? t.images.count : 0
                return (0, TokenEstimate.text(t.text), shots * TokenEstimate.perMedia)
            }
        }
        func total(_ ts: ArraySlice<ChatTurn>) -> Int { ts.reduce(0) { let c = cost($1); return $0 + c.messages + c.tools + c.media } }

        let tokens: Int
        let exact: Bool
        if let i = sent.lastIndex(where: { $0.stats?.promptTokens != nil }), let stats = sent[i].stats,
           let prompt = stats.promptTokens {
            let tail = sent[(i + 1)...]
            tokens = prompt + (stats.completionTokens ?? 0) + total(tail)
            exact = tail.isEmpty
        } else {
            tokens = (overhead ?? 0) + total(sent[...])
            exact = false
        }
        guard breakdown else { return ContextUse(tokens: tokens, exact: exact) }
        var b = ContextUse.Breakdown(system: 0, messages: 0, tools: 0, media: 0)
        for t in sent {
            let c = cost(t)
            b.messages += c.messages
            b.tools += c.tools
            b.media += c.media
        }
        b.system = max(0, tokens - b.messages - b.tools - b.media)
        return ContextUse(tokens: tokens, exact: exact, breakdown: b)
    }
}
