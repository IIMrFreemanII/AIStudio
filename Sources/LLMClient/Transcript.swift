import Foundation

/// One message of a saved chat, with what the app shows around it (thinking, stats, errors).
public struct ChatTurn: Codable, Identifiable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable { case user, assistant, tool }

    public var id: UUID
    public var role: Role
    public var text: String
    /// The model's thinking (assistant turns of thinking models).
    public var reasoning: String?
    /// How long it thought, in seconds.
    public var thinkingSeconds: Double?
    /// Attached images: file names in the chat's files folder.
    public var images: [String]
    public var toolCalls: [ToolCall]
    public var toolCallID: String?
    /// The model that wrote it (assistant turns).
    public var model: String?
    public var stats: RequestStats?
    /// Why the reply stopped early (a failed request, a cancel). The turn isn't sent back to the model.
    public var error: String?
    public var date: Date

    public init(id: UUID = UUID(), role: Role, text: String, reasoning: String? = nil, images: [String] = [],
                toolCalls: [ToolCall] = [], toolCallID: String? = nil, model: String? = nil, date: Date = Date()) {
        self.id = id
        self.role = role
        self.text = text
        self.reasoning = reasoning
        self.images = images
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.model = model
        self.date = date
    }
}

/// A saved conversation: which server and model it talks to, and its turns.
public struct ChatTranscript: Codable, Identifiable, Equatable, Sendable {
    public var version = 1
    public var id: UUID
    public var title: String
    public var created: Date
    public var updated: Date
    /// The GPU server connection and the model id on it.
    public var serverID: String?
    public var modelID: String?
    public var systemPrompt: String?
    /// "low", "medium" or "high" (nil: the model's default).
    public var reasoningEffort: String?
    public var turns: [ChatTurn]

    public init(id: UUID = UUID(), title: String = "New Chat", serverID: String? = nil, modelID: String? = nil,
                turns: [ChatTurn] = [], date: Date = Date()) {
        self.id = id
        self.title = title
        self.created = date
        self.updated = date
        self.serverID = serverID
        self.modelID = modelID
        self.turns = turns
    }

    /// What the chat has cost so far (its requests' GPU time).
    public var costUSD: Double { turns.compactMap { $0.stats?.costUSD }.reduce(0, +) }

    /// The turns as OpenAI messages, without failed replies. `image(name)` gives an attached image's data URL.
    public func messages(image: (String) -> String? = { _ in nil }) -> [ChatMessage] {
        var out: [ChatMessage] = []
        if let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out.append(.system(systemPrompt))
        }
        for t in turns where t.error == nil || !t.text.isEmpty {
            switch t.role {
            case .user:
                var parts: [ContentPart] = t.images.compactMap { image($0) }.map { .image(url: $0) }
                parts.append(.text(t.text))
                out.append(ChatMessage(role: .user, parts: parts))
            case .assistant:
                out.append(ChatMessage(role: .assistant, parts: [.text(t.text)], toolCalls: t.toolCalls))
            case .tool:
                out.append(ChatMessage(role: .tool, parts: [.text(t.text)], toolCallID: t.toolCallID))
            }
        }
        return out
    }

    /// A title from the first user message: its first line, at most 48 characters.
    public static func title(from text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty } ?? ""
        guard !line.isEmpty else { return "New Chat" }
        return line.count > 48 ? String(line.prefix(47)) + "…" : line
    }
}
