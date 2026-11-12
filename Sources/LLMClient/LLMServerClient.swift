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
    /// "vllm" or "llamacpp" (GGUF).
    public var engine: String?
    /// "huggingface" or "civitai".
    public var source: String?
    /// How vLLM reads the weights: "auto" (its loader), "processed" (fast start: a copy saved after vLLM's processing,
    /// made by the first start) or "runai" (Run:ai's streamer, for object storage).
    public var loadFormat: String?
    /// Starts from a GPU memory snapshot (experimental; vLLM on one GPU).
    public var snapshot: Bool?
    /// Multi-token-prediction layers the checkpoint ships (config and tensors); 0 or nil: none.
    public var mtpLayers: Int?
    /// Drafts tokens with those layers (speculative decoding; nil means on when the model has them).
    public var mtp: Bool?
    /// Thinking levels the model takes ("low"/"medium"/"high", or "off"/"on"); [] can't be controlled; nil: older server.
    public var efforts: [String]?

    public var hasMTP: Bool { (mtpLayers ?? 0) > 0 && engine != "llamacpp" }
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
    /// Seconds since the start was asked for, while starting or loading.
    public var bootSeconds: Double?
    /// The model's last cold start and how long one usually takes.
    public var coldStart: ColdStartSummary?
    /// Tokens the running engine's KV cache holds, and how many full-context requests fit at once.
    public var kvTokens: Int?
    public var maxConcurrency: Double?

    public init(state: String, phase: String? = nil, progress: Double? = nil) {
        self.state = state
        self.phase = phase
        self.progress = progress
    }

    public var isActive: Bool { ["starting", "loading", "ready"].contains(state) }
    /// Asked to start but not ready yet (waiting for a GPU, or loading): Stop cancels the start.
    public var isStarting: Bool { state == "starting" || state == "loading" }
}

/// How long a model takes to start: the last start (split into phases) and the typical one.
public struct ColdStartSummary: Codable, Equatable, Sendable {
    public var last: ColdStartRecord
    public var typicalSeconds: Double
    public var count: Int

    public init(last: ColdStartRecord, typicalSeconds: Double, count: Int) {
        self.last = last
        self.typicalSeconds = typicalSeconds
        self.count = count
    }
}

/// One start: when it was asked for, how long it took, the GPU wait, and each phase.
public struct ColdStartRecord: Codable, Equatable, Sendable {
    public struct Phase: Codable, Equatable, Sendable {
        public var name: String
        public var seconds: Double
        public init(name: String, seconds: Double) { self.name = name; self.seconds = seconds }
    }

    public var at: Double?
    public var seconds: Double
    public var waitSeconds: Double?
    /// "cold", or "snapshot" (restored from a GPU memory snapshot).
    public var how: String?
    public var phases: [Phase]
    public var gpu: String?
    public var count: Int?
    public var loadFormat: String?
    public var snapshot: Bool?

    public init(seconds: Double, phases: [Phase], how: String? = "cold") {
        self.seconds = seconds
        self.phases = phases
        self.how = how
    }

    /// "3 min 12 s", "48 s".
    public static func duration(_ s: Double) -> String {
        let t = Int(s.rounded())
        if t < 60 { return "\(t) s" }
        return t % 60 == 0 ? "\(t / 60) min" : "\(t / 60) min \(t % 60) s"
    }

    /// "Waiting for a GPU 20 s · Loading weights 2 min 10 s · …"
    public var breakdown: String {
        phases.filter { $0.seconds >= 0.5 }.map { "\($0.name) \(Self.duration($0.seconds))" }.joined(separator: " · ")
    }
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
    /// Tokens this option's KV cache holds (not capped at the model's window).
    public var kvTokens: Int?
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
        public var engine: String?
        public var source: String?
        public var mmproj: String?
        /// A GGUF repo's quantizations (`quant`, `bytes`), smallest-good first.
        public var ggufOptions: [GGUFOption]?
    }

    public struct GGUFOption: Codable, Equatable, Hashable, Sendable {
        public var quant: String
        public var bytes: Int
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

/// An installed model's GPU options (`GET v1/llm/models/{id}/plan`).
public struct LLMPlan: Codable, Equatable, Sendable {
    public struct Current: Codable, Equatable, Sendable {
        public var gpu: String
        public var count: Int
        public var context: Int
    }

    public var context: Int
    /// The model's window, capped at 1,048,576.
    public var nativeContext: Int?
    public var recommended: GPUOption?
    public var options: [GPUOption]
    public var current: Current
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
    /// A GGUF repo's quantization ("Q4_K_M"…).
    public var quant: String?
    /// The user's Hugging Face token, for gated models (sent over TLS, used once, never stored by the server).
    public var hfToken: String?
    /// The user's Civitai API key, for Civitai links that need a login.
    public var civitaiToken: String?

    public init(url: String, id: String? = nil, name: String? = nil, gpu: String? = nil, count: Int? = nil,
                context: Int? = nil, quant: String? = nil, hfToken: String? = nil, civitaiToken: String? = nil) {
        self.quant = quant
        self.civitaiToken = civitaiToken
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

    /// Uploads a chat's video unless the server has it (resumable; the file's name must be `<sha256>.<ext>`, as the
    /// app saves them) and returns the URL a message uses for it: `upload:<name>`.
    public func uploadMedia(_ file: URL,
                            progress: (@Sendable (_ sent: Int64, _ total: Int64, _ start: Int64) -> Void)? = nil)
        async throws -> String {
        let bytes = (try file.resourceValues(forKeys: [.fileSizeKey])).fileSize ?? 0
        _ = try await http.upload(name: file.lastPathComponent, file: file, bytes: bytes, progress: progress)
        return "upload:" + file.lastPathComponent
    }

    public func inspect(url: String, hfToken: String? = nil, civitaiToken: String? = nil, context: Int? = nil,
                        quant: String? = nil) async throws -> LLMInspection {
        var body: [String: JSONValue] = ["url": .string(url)]
        if let hfToken, !hfToken.isEmpty { body["hfToken"] = .string(hfToken) }
        if let civitaiToken, !civitaiToken.isEmpty { body["civitaiToken"] = .string(civitaiToken) }
        if let context { body["context"] = .number(Double(context)) }
        if let quant { body["quant"] = .string(quant) }
        return try await send("v1/llm/inspect", method: "POST", body: body)
    }

    public func install(_ request: LLMInstallRequest) async throws -> LLMInstallState {
        try await send("v1/llm/install", method: "POST", body: request)
    }

    public func models() async throws -> LLMListing { try await send("v1/llm/models", method: "GET", body: nil as String?) }

    public func update(_ id: String, _ changes: [String: JSONValue]) async throws -> LLMModel {
        try await send("v1/llm/models/\(id)", method: "PATCH", body: changes)
    }

    /// An installed model's GPU options again, and its current choice (to change them).
    /// With `context`, the server re-plans for that many tokens (`recommended` is the cheapest option that fits).
    public func plan(_ id: String, context: Int? = nil) async throws -> LLMPlan {
        let query = context.map { "?context=\($0)" } ?? ""
        return try await send("v1/llm/models/\(id)/plan\(query)", method: "GET", body: nil as String?)
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
        // A query string can't ride in `appendingPathComponent` (it would be escaped), so it is split off and put back.
        let parts = path.split(separator: "?", maxSplits: 1).map(String.init)
        var r = http.request(parts[0], method: method)
        if parts.count > 1, let url = r.url, var c = URLComponents(url: url, resolvingAgainstBaseURL: false) {
            c.percentEncodedQuery = parts[1]
            r.url = c.url ?? url
        }
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
