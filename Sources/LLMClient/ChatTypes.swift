import Foundation

/// Any JSON value, for the free-form parts of requests and replies (tool schemas, template options).
public enum JSONValue: Codable, Equatable, Sendable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else if let a = try? c.decode([JSONValue].self) { self = .array(a) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let b): try c.encode(b)
        case .number(let n):
            if n.rounded() == n, abs(n) < 1e15 { try c.encode(Int64(n)) } else { try c.encode(n) }
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        case .object(let o): try c.encode(o)
        }
    }

    public var string: String? { if case .string(let s) = self { return s }; return nil }
    public var number: Double? { if case .number(let n) = self { return n }; return nil }
}

/// A part of a user message: text, an image (a `data:` URL or an http one) or a video (`upload:<name>`, a file
/// uploaded to the GPU server first, or an http URL).
public enum ContentPart: Codable, Equatable, Sendable, Hashable {
    case text(String)
    case image(url: String)
    case video(url: String)

    enum CodingKeys: String, CodingKey { case type, text, image_url, video_url }
    struct ImageURL: Codable { var url: String }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        switch try c.decode(String.self, forKey: .type) {
        case "image_url": self = .image(url: try c.decode(ImageURL.self, forKey: .image_url).url)
        case "video_url": self = .video(url: try c.decode(ImageURL.self, forKey: .video_url).url)
        default: self = .text(try c.decodeIfPresent(String.self, forKey: .text) ?? "")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let t):
            try c.encode("text", forKey: .type)
            try c.encode(t, forKey: .text)
        case .image(let url):
            try c.encode("image_url", forKey: .type)
            try c.encode(ImageURL(url: url), forKey: .image_url)
        case .video(let url):
            try c.encode("video_url", forKey: .type)
            try c.encode(ImageURL(url: url), forKey: .video_url)
        }
    }
}

/// A tool call the model made: the function's name and its JSON arguments (as the model wrote them).
public struct ToolCall: Codable, Equatable, Sendable, Hashable {
    public var id: String
    public var name: String
    public var arguments: String
    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    enum CodingKeys: String, CodingKey { case id, type, function }
    struct Function: Codable { var name: String; var arguments: String }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        let f = try c.decode(Function.self, forKey: .function)
        name = f.name
        arguments = f.arguments
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode("function", forKey: .type)
        try c.encode(Function(name: name, arguments: arguments), forKey: .function)
    }
}

/// A message in OpenAI's chat format.
public struct ChatMessage: Encodable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable { case system, user, assistant, tool }

    public var role: Role
    public var parts: [ContentPart]
    public var toolCalls: [ToolCall] = []
    public var toolCallID: String?
    /// The model's thinking, sent back only when the model wants it (`reasoning_content`).
    public var reasoning: String?

    public init(role: Role, parts: [ContentPart], toolCalls: [ToolCall] = [], toolCallID: String? = nil,
                reasoning: String? = nil) {
        self.role = role
        self.parts = parts
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.reasoning = reasoning
    }

    public static func system(_ text: String) -> ChatMessage { ChatMessage(role: .system, parts: [.text(text)]) }
    public static func user(_ text: String) -> ChatMessage { ChatMessage(role: .user, parts: [.text(text)]) }
    public static func assistant(_ text: String) -> ChatMessage { ChatMessage(role: .assistant, parts: [.text(text)]) }

    public var text: String {
        parts.compactMap { if case .text(let t) = $0 { return t }; return nil }.joined(separator: "\n")
    }

    enum CodingKeys: String, CodingKey { case role, content, tool_calls, tool_call_id, reasoning_content }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(role, forKey: .role)
        // Plain text as a string (every server takes that); images and videos need the list form.
        if parts.contains(where: { if case .text = $0 { return false }; return true }) {
            try c.encode(parts, forKey: .content)
        } else {
            try c.encode(text, forKey: .content)
        }
        if !toolCalls.isEmpty { try c.encode(toolCalls, forKey: .tool_calls) }
        if let toolCallID { try c.encode(toolCallID, forKey: .tool_call_id) }
        if let reasoning, !reasoning.isEmpty { try c.encode(reasoning, forKey: .reasoning_content) }
    }
}

/// A function the model may call.
public struct ToolSpec: Encodable, Equatable, Sendable {
    public var name: String
    public var description: String
    public var parameters: JSONValue
    public init(name: String, description: String, parameters: JSONValue) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }

    enum CodingKeys: String, CodingKey { case type, function }
    struct Function: Encodable { var name: String; var description: String; var parameters: JSONValue }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode("function", forKey: .type)
        try c.encode(Function(name: name, description: description, parameters: parameters), forKey: .function)
    }
}

/// A chat completions request. Unset sampling fields keep the model's own defaults (its generation_config).
public struct ChatRequest: Encodable, Sendable {
    public var model: String
    public var messages: [ChatMessage]
    public var stream = true
    public var tools: [ToolSpec] = []
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var maxTokens: Int?
    /// "low", "medium", "high"… for models that take it (Qwen3.8, gpt-oss).
    public var reasoningEffort: String?
    /// Chat template switches, such as `["enable_thinking": .bool(false)]`.
    public var chatTemplateKwargs: [String: JSONValue] = [:]

    public init(model: String, messages: [ChatMessage]) {
        self.model = model
        self.messages = messages
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, tools, temperature, top_p, top_k, max_tokens, reasoning_effort
        case chat_template_kwargs, stream_options
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(model, forKey: .model)
        try c.encode(messages, forKey: .messages)
        try c.encode(stream, forKey: .stream)
        if stream { try c.encode(["include_usage": true], forKey: .stream_options) }
        if !tools.isEmpty { try c.encode(tools, forKey: .tools) }
        try c.encodeIfPresent(temperature, forKey: .temperature)
        try c.encodeIfPresent(topP, forKey: .top_p)
        try c.encodeIfPresent(topK, forKey: .top_k)
        try c.encodeIfPresent(maxTokens, forKey: .max_tokens)
        try c.encodeIfPresent(reasoningEffort, forKey: .reasoning_effort)
        if !chatTemplateKwargs.isEmpty { try c.encode(chatTemplateKwargs, forKey: .chat_template_kwargs) }
    }
}

/// The engine's state while a request waits for it (`event: aistudio.status`).
public struct EngineStatus: Codable, Equatable, Sendable {
    public var state: String
    public var phase: String?
    public var progress: Double?
    public var gpu: String?
    public var count: Int?

    public init(state: String, phase: String? = nil, progress: Double? = nil, gpu: String? = nil, count: Int? = nil) {
        self.state = state
        self.phase = phase
        self.progress = progress
        self.gpu = gpu
        self.count = count
    }
}

/// What a request took (`event: aistudio.usage`, sent before `[DONE]`).
public struct RequestStats: Codable, Equatable, Sendable {
    public var seconds: Double
    public var waitSeconds: Double?
    public var ttftMs: Double?
    public var tokPerSec: Double?
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var usdPerSecond: Double?
    public var costUSD: Double?

    public init(seconds: Double, waitSeconds: Double? = nil, ttftMs: Double? = nil, tokPerSec: Double? = nil,
                promptTokens: Int? = nil, completionTokens: Int? = nil, usdPerSecond: Double? = nil,
                costUSD: Double? = nil) {
        self.seconds = seconds
        self.waitSeconds = waitSeconds
        self.ttftMs = ttftMs
        self.tokPerSec = tokPerSec
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.usdPerSecond = usdPerSecond
        self.costUSD = costUSD
    }
}

/// What a streamed reply delivers, in order.
public enum ChatEvent: Equatable, Sendable {
    case content(String)
    case reasoning(String)
    /// A piece of a tool call: the first piece has the id and name, later ones more of the arguments.
    case toolCall(index: Int, id: String?, name: String?, arguments: String)
    case finish(reason: String)
    case usage(promptTokens: Int, completionTokens: Int)
    case status(EngineStatus)
    case stats(RequestStats)
}

public enum LLMError: LocalizedError, Equatable {
    case http(Int, String)
    case server(String)
    case badReply(String)

    public var errorDescription: String? {
        switch self {
        case .http(let code, let text):
            switch code {
            case 401: return "The server refused the token (401). Check it in Servers."
            case 404: return text.isEmpty ? "Not found on the server (404)" : text
            default: return text.isEmpty ? "HTTP \(code)" : "\(text) (HTTP \(code))"
            }
        case .server(let text): return text
        case .badReply(let text): return "The server sent something unexpected: \(text)"
        }
    }
}
