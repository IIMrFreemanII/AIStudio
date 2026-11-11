import AIStudioProtocol
import Foundation

/// An installed model on a GPU server (`GET v1/llm/models`).
public struct LLMModel: Codable, Identifiable, Equatable, Sendable {
    public var id: String
    public var name: String
    public var url: String?
    public var repo: String
    public var gpu: String
    public var count: Int
    public var context: Int
    public var nativeContext: Int?
    public var vision: Bool?
    public var weightBytes: Int?
    public var toolParser: String?
    public var reasoningParser: String?
    public var engineArgs: [String]?
    public var installed: Bool?
    public var status: LLMStatus?

    public var gpuLabel: String { count > 1 ? "\(count)× \(gpu)" : gpu }
}

/// A model's engine: cold (no GPU), starting (waiting for one), loading, ready or failed, and what it has cost since
/// it started.
public struct LLMStatus: Codable, Equatable, Sendable {
    public var id: String?
    public var state: String
    public var phase: String?
    public var progress: Double?
    public var error: String?
    public var gpu: String?
    public var count: Int?
    public var uptime: Double?
    public var usdPerSecond: Double?
    public var costUSD: Double?
    public var idleSeconds: Double?
    public var leaseSeconds: Double?
    /// Set on a keepalive the server ignored (past the lease: no chat for a long time).
    public var ignored: Bool?

    public init(state: String, phase: String? = nil, progress: Double? = nil) {
        self.state = state
        self.phase = phase
        self.progress = progress
    }

    public var isActive: Bool { ["starting", "loading", "ready"].contains(state) }
}

/// An install in progress.
public struct LLMInstallState: Codable, Equatable, Sendable {
    public var id: String?
    public var state: String
    public var name: String?
    public var bytes: Int?
    public var total: Int?
    public var error: String?
}

public struct LLMListing: Codable, Equatable, Sendable {
    public var models: [LLMModel]
    public var installs: [String: LLMInstallState]
}

/// One way to run a model: a GPU kind and count, what it costs, and the longest context that fits.
public struct GPUOption: Codable, Equatable, Hashable, Sendable {
    public var gpu: String
    public var count: Int
    public var fits: Bool
    public var usdPerHour: Double
    public var maxContext: Int
    public var weightsPerGPUGB: Double?
    public var kvPerTokenKB: Double?
    public var note: String?

    public var label: String { count > 1 ? "\(count)× \(gpu)" : gpu }
}

/// What a link points at, and how the server would run it (`POST v1/llm/inspect`).
public struct LLMInspection: Codable, Equatable, Sendable {
    public struct Facts: Codable, Equatable, Sendable {
        public var repo: String
        public var revision: String?
        public var architecture: String
        public var modelType: String
        public var weightBytes: Int
        public var nativeContext: Int
        public var vision: Bool
        public var experts: Int?
        public var quant: [String: JSONValue]?
        public var params: Int?
        public var gated: Bool?
        public var notes: [String]?
    }

    public struct Profile: Codable, Equatable, Sendable {
        public var toolParser: String?
        public var reasoningParser: String?
        public var engineArgs: [String]?
    }

    public struct Plan: Codable, Equatable, Sendable {
        public var context: Int
        public var recommended: GPUOption?
        public var options: [GPUOption]
    }

    public var facts: Facts
    public var profile: Profile
    public var plan: Plan
    public var suggestedId: String
    public var name: String
}

/// `POST v1/llm/install`.
public struct LLMInstallRequest: Encodable, Sendable {
    public var url: String
    public var id: String?
    public var name: String?
    public var gpu: String?
    public var count: Int?
    public var context: Int?
    public var toolParser: String?
    public var reasoningParser: String?
    /// The user's Hugging Face token, for gated models (sent over TLS, used once, never stored by the server).
    public var hfToken: String?

    public init(url: String, id: String? = nil, name: String? = nil, gpu: String? = nil, count: Int? = nil,
                context: Int? = nil, hfToken: String? = nil) {
        self.url = url
        self.id = id
        self.name = name
        self.gpu = gpu
        self.count = count
        self.context = context
        self.hfToken = hfToken
    }
}

/// The GPU server's LLM routes (`v1/llm/…`): installs, warmth and status. Chats go through ``OpenAIChatClient``.
public struct LLMServerClient: Sendable {
    public let http: HTTPService

    public init(http: HTTPService) { self.http = http }
    public init(baseURL: URL, token: String) { http = HTTPService(baseURL: baseURL, token: token) }

    public var chat: OpenAIChatClient { OpenAIChatClient(http: http) }

    public func inspect(url: String, hfToken: String? = nil, context: Int? = nil) async throws -> LLMInspection {
        var body: [String: JSONValue] = ["url": .string(url)]
        if let hfToken, !hfToken.isEmpty { body["hfToken"] = .string(hfToken) }
        if let context { body["context"] = .number(Double(context)) }
        return try await send("v1/llm/inspect", method: "POST", body: body)
    }

    public func install(_ request: LLMInstallRequest) async throws -> LLMInstallState {
        try await send("v1/llm/install", method: "POST", body: request)
    }

    public func models() async throws -> LLMListing { try await send("v1/llm/models", method: "GET", body: nil as String?) }

    public func update(_ id: String, _ changes: [String: JSONValue]) async throws -> LLMModel {
        try await send("v1/llm/models/\(id)", method: "PATCH", body: changes)
    }

    public func remove(_ id: String) async throws {
        let (data, response) = try await http.session.data(for: http.request("v1/llm/models/\(id)", method: "DELETE"))
        try Self.check(response, data)
    }

    public func warm(_ id: String) async throws -> LLMStatus { try await send("v1/llm/models/\(id)/warm", method: "POST", body: nil as String?) }
    public func keepalive(_ id: String) async throws -> LLMStatus {
        try await send("v1/llm/models/\(id)/keepalive", method: "POST", body: nil as String?)
    }
    public func stop(_ id: String) async throws -> LLMStatus { try await send("v1/llm/models/\(id)/stop", method: "POST", body: nil as String?) }
    public func status(_ id: String) async throws -> LLMStatus { try await send("v1/llm/models/\(id)/status", method: "GET", body: nil as String?) }

    static func check(_ response: URLResponse, _ data: Data) throws {
        guard let status = (response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(status) else { return }
        let text = OpenAIChatClient.errorText(data)
        if status == 404, text == "Not Found" {  // the route itself is missing: a server from before M24
            throw LLMError.http(404, "This GPU server has no LLM support yet: redeploy it (docs/remote-server.md)")
        }
        throw LLMError.http(status, text)
    }

    private func send<B: Encodable, T: Decodable>(_ path: String, method: String, body: B?) async throws -> T {
        var r = http.request(path, method: method)
        if let body {
            r.httpBody = try JSONEncoder().encode(body)
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        let (data, response) = try await http.session.data(for: r)
        try Self.check(response, data)
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw LLMError.badReply("\(path): \(error)")
        }
    }
}
