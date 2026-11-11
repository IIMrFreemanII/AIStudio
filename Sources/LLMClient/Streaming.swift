import Foundation

/// Splits `<think>…</think>` out of streamed text, for servers that send a thinking model's reasoning inline instead
/// of in `reasoning` / `reasoning_content`. Tags may arrive split across chunks; text that could be the start of a tag
/// waits for the next chunk. A reply that starts inside its thinking (templates that open `<think>` themselves) is
/// handled with `startsThinking`.
public struct ThinkSplitter: Sendable {
    public enum Piece: Equatable, Sendable {
        case content(String)
        case reasoning(String)
    }

    private var thinking: Bool
    private var pending = ""
    static let open = "<think>", close = "</think>"

    public init(startsThinking: Bool = false) { thinking = startsThinking }

    public mutating func feed(_ text: String) -> [Piece] {
        var out: [Piece] = []
        var s = pending + text
        pending = ""
        while !s.isEmpty {
            let tag = thinking ? Self.close : Self.open
            if let r = s.range(of: tag) {
                emit(String(s[..<r.lowerBound]), into: &out)
                thinking.toggle()
                s = String(s[r.upperBound...])
                if !thinking, s.hasPrefix("\n") { s = String(s.drop { $0 == "\n" }) }
                continue
            }
            // Keep back a tail that may be the start of the tag.
            var keep = 0
            for n in stride(from: min(tag.count - 1, s.count), through: 1, by: -1) where tag.hasPrefix(s.suffix(n)) {
                keep = n
                break
            }
            emit(String(s.dropLast(keep)), into: &out)
            pending = String(s.suffix(keep))
            s = ""
        }
        return out
    }

    /// Whatever was held back, at the end of the stream.
    public mutating func finish() -> [Piece] {
        var out: [Piece] = []
        emit(pending, into: &out)
        pending = ""
        return out
    }

    private func emit(_ text: String, into out: inout [Piece]) {
        guard !text.isEmpty else { return }
        out.append(thinking ? .reasoning(text) : .content(text))
    }
}

/// Puts streamed tool-call pieces back together: the first piece of each call (by index) has its id and name, the
/// rest add to its arguments.
public struct ToolCallAssembler: Sendable {
    private var byIndex: [Int: ToolCall] = [:]

    public init() {}

    public mutating func add(index: Int, id: String?, name: String?, arguments: String) {
        var call = byIndex[index] ?? ToolCall(id: "", name: "", arguments: "")
        if let id, !id.isEmpty { call.id = id }
        if let name, !name.isEmpty, call.name.isEmpty { call.name = name }
        call.arguments += arguments
        byIndex[index] = call
    }

    public var isEmpty: Bool { byIndex.isEmpty }

    /// The calls in order, with ids filled in where the server sent none.
    public var calls: [ToolCall] {
        byIndex.keys.sorted().map { i in
            var c = byIndex[i]!
            if c.id.isEmpty { c.id = "call_\(i)" }
            return c
        }
    }
}

/// Decodes one `data:` chunk of an OpenAI chat stream into events.
enum ChunkDecoder {
    static func events(_ json: [String: Any]) throws -> [ChatEvent] {
        if let error = json["error"] {
            let message = (error as? [String: Any])?["message"] as? String ?? (error as? String) ?? "\(error)"
            throw LLMError.server(message)
        }
        var out: [ChatEvent] = []
        for choice in json["choices"] as? [[String: Any]] ?? [] {
            let delta = choice["delta"] as? [String: Any] ?? choice["message"] as? [String: Any] ?? [:]
            let reasoning = delta["reasoning"] as? String ?? delta["reasoning_content"] as? String
            if let reasoning, !reasoning.isEmpty { out.append(.reasoning(reasoning)) }
            if let content = delta["content"] as? String, !content.isEmpty { out.append(.content(content)) }
            for call in delta["tool_calls"] as? [[String: Any]] ?? [] {
                let f = call["function"] as? [String: Any] ?? [:]
                out.append(.toolCall(index: call["index"] as? Int ?? 0, id: call["id"] as? String,
                                     name: f["name"] as? String, arguments: f["arguments"] as? String ?? ""))
            }
            if let reason = choice["finish_reason"] as? String { out.append(.finish(reason: reason)) }
        }
        if let usage = json["usage"] as? [String: Any] {
            out.append(.usage(promptTokens: usage["prompt_tokens"] as? Int ?? 0,
                              completionTokens: usage["completion_tokens"] as? Int ?? 0))
        }
        return out
    }
}
