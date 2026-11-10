import Foundation
import GraphRuntime

// Cold-path messages: JSON text frames on the WebSocket and JSON bodies of the REST routes. They are rare (a few per
// run), so readability wins over size here.

public enum ClientRole: String, Codable, Sendable {
    /// The MLXComfyUI app: gets the events of its own runs.
    case app
    /// The web dashboard: gets every run's events, metrics, state and logs.
    case dashboard
    /// `mlxcomfy-cli bench-remote`.
    case bench
}

public struct ClientHello: Codable, Equatable, Sendable {
    public var protocolVersion: Int
    public var role: ClientRole
    /// Shown in the dashboard's client list ("MLXComfyUI on MacBook Pro").
    public var name: String
    /// Preview frames per second wanted (0 = none) and their longest side in pixels.
    public var previewHz: Double
    public var previewMaxSide: Int

    public init(protocolVersion: Int = RemoteProtocol.version, role: ClientRole, name: String,
                previewHz: Double = 4, previewMaxSide: Int = 512) {
        self.protocolVersion = protocolVersion
        self.role = role
        self.name = name
        self.previewHz = previewHz
        self.previewMaxSide = previewMaxSide
    }
}

public struct SubmitRequest: Codable, Equatable, Sendable {
    /// Chosen by the client to match the `submitted`/`rejected` reply.
    public var requestID: String
    /// The graph to run: flattened, with server-side file names (see ``GraphTransport``).
    public var graph: Graph
    /// The client's original graph, embedded in the outputs instead of `graph`.
    public var metadataGraph: Graph?
    public var workflow: String
    /// Uploaded input files the graph needs (`<sha256>.<ext>`). The server waits for missing ones while it runs
    /// the nodes that don't need them.
    public var files: [String]

    public init(requestID: String, graph: Graph, metadataGraph: Graph?, workflow: String, files: [String]) {
        self.requestID = requestID
        self.graph = graph
        self.metadataGraph = metadataGraph
        self.workflow = workflow
        self.files = files
    }
}

public struct TextRequest: Codable, Equatable, Sendable {
    public var requestID: String
    /// Model folder name on the server.
    public var model: String
    public var system: String
    public var user: String
    /// Uploaded images (`<sha256>.<ext>`).
    public var images: [String]
    public var maxTokens: Int
    public var temperature: Double

    public init(requestID: String, model: String, system: String, user: String, images: [String],
                maxTokens: Int, temperature: Double) {
        self.requestID = requestID
        self.model = model
        self.system = system
        self.user = user
        self.images = images
        self.maxTokens = maxTokens
        self.temperature = temperature
    }
}

/// Client → server.
public enum ClientMessage: Equatable, Sendable {
    case hello(ClientHello)
    case submit(SubmitRequest)
    case cancel(run: UInt32)
    /// Re-attach to a run after reconnecting; the server replays its events after `lastSeq`.
    case subscribe(run: UInt32, lastSeq: UInt32)
    case setPreview(hz: Double, maxSide: Int)
    case ping(id: Int)
    case text(TextRequest)
    case cancelText(requestID: String)
}

/// Server → client.
public enum ServerMessage: Equatable, Sendable {
    case welcome(protocolVersion: Int, serverVersion: String, session: String)
    /// `nodes` is the node table: record `node` fields index into it.
    case submitted(requestID: String, run: UInt32, nodes: [String])
    case rejected(requestID: String, message: String)
    case pong(id: Int)
    /// The full text generated so far.
    case textDelta(requestID: String, text: String)
    case textDone(requestID: String, text: String?, error: String?)
    case state(DashboardState)
    case log(LogLine)
    case error(message: String)
}

// MARK: - coding (`{"type": "...", ...fields}`)

private struct TypeKey: CodingKey {
    var stringValue: String
    init(_ s: String) { stringValue = s }
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { nil }
}

extension ClientMessage: Codable {
    private enum K: String, CodingKey { case type, hello, submit, run, lastSeq, hz, maxSide, id, text, requestID }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: K.self)
        switch try c.decode(String.self, forKey: .type) {
        case "hello": self = .hello(try c.decode(ClientHello.self, forKey: .hello))
        case "submit": self = .submit(try c.decode(SubmitRequest.self, forKey: .submit))
        case "cancel": self = .cancel(run: try c.decode(UInt32.self, forKey: .run))
        case "subscribe": self = .subscribe(run: try c.decode(UInt32.self, forKey: .run),
                                            lastSeq: try c.decode(UInt32.self, forKey: .lastSeq))
        case "setPreview": self = .setPreview(hz: try c.decode(Double.self, forKey: .hz),
                                              maxSide: try c.decode(Int.self, forKey: .maxSide))
        case "ping": self = .ping(id: try c.decode(Int.self, forKey: .id))
        case "text": self = .text(try c.decode(TextRequest.self, forKey: .text))
        case "cancelText": self = .cancelText(requestID: try c.decode(String.self, forKey: .requestID))
        case let t: throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "unknown type \(t)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: K.self)
        switch self {
        case .hello(let h): try c.encode("hello", forKey: .type); try c.encode(h, forKey: .hello)
        case .submit(let s): try c.encode("submit", forKey: .type); try c.encode(s, forKey: .submit)
        case .cancel(let r): try c.encode("cancel", forKey: .type); try c.encode(r, forKey: .run)
        case .subscribe(let r, let s):
            try c.encode("subscribe", forKey: .type); try c.encode(r, forKey: .run); try c.encode(s, forKey: .lastSeq)
        case .setPreview(let hz, let side):
            try c.encode("setPreview", forKey: .type); try c.encode(hz, forKey: .hz); try c.encode(side, forKey: .maxSide)
        case .ping(let id): try c.encode("ping", forKey: .type); try c.encode(id, forKey: .id)
        case .text(let t): try c.encode("text", forKey: .type); try c.encode(t, forKey: .text)
        case .cancelText(let r): try c.encode("cancelText", forKey: .type); try c.encode(r, forKey: .requestID)
        }
    }
}

extension ServerMessage: Codable {
    private enum K: String, CodingKey {
        case type, protocolVersion, serverVersion, session, requestID, run, nodes, message, id, text, error, state, log
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: K.self)
        switch try c.decode(String.self, forKey: .type) {
        case "welcome":
            self = .welcome(protocolVersion: try c.decode(Int.self, forKey: .protocolVersion),
                            serverVersion: try c.decode(String.self, forKey: .serverVersion),
                            session: try c.decode(String.self, forKey: .session))
        case "submitted":
            self = .submitted(requestID: try c.decode(String.self, forKey: .requestID),
                              run: try c.decode(UInt32.self, forKey: .run),
                              nodes: try c.decode([String].self, forKey: .nodes))
        case "rejected":
            self = .rejected(requestID: try c.decode(String.self, forKey: .requestID),
                             message: try c.decode(String.self, forKey: .message))
        case "pong": self = .pong(id: try c.decode(Int.self, forKey: .id))
        case "textDelta":
            self = .textDelta(requestID: try c.decode(String.self, forKey: .requestID),
                              text: try c.decode(String.self, forKey: .text))
        case "textDone":
            self = .textDone(requestID: try c.decode(String.self, forKey: .requestID),
                             text: try c.decodeIfPresent(String.self, forKey: .text),
                             error: try c.decodeIfPresent(String.self, forKey: .error))
        case "state": self = .state(try c.decode(DashboardState.self, forKey: .state))
        case "log": self = .log(try c.decode(LogLine.self, forKey: .log))
        case "error": self = .error(message: try c.decode(String.self, forKey: .message))
        case let t: throw DecodingError.dataCorruptedError(forKey: .type, in: c, debugDescription: "unknown type \(t)")
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: K.self)
        switch self {
        case .welcome(let v, let sv, let s):
            try c.encode("welcome", forKey: .type)
            try c.encode(v, forKey: .protocolVersion); try c.encode(sv, forKey: .serverVersion); try c.encode(s, forKey: .session)
        case .submitted(let r, let run, let nodes):
            try c.encode("submitted", forKey: .type)
            try c.encode(r, forKey: .requestID); try c.encode(run, forKey: .run); try c.encode(nodes, forKey: .nodes)
        case .rejected(let r, let m):
            try c.encode("rejected", forKey: .type); try c.encode(r, forKey: .requestID); try c.encode(m, forKey: .message)
        case .pong(let id): try c.encode("pong", forKey: .type); try c.encode(id, forKey: .id)
        case .textDelta(let r, let t):
            try c.encode("textDelta", forKey: .type); try c.encode(r, forKey: .requestID); try c.encode(t, forKey: .text)
        case .textDone(let r, let t, let e):
            try c.encode("textDone", forKey: .type); try c.encode(r, forKey: .requestID)
            try c.encodeIfPresent(t, forKey: .text); try c.encodeIfPresent(e, forKey: .error)
        case .state(let s): try c.encode("state", forKey: .type); try c.encode(s, forKey: .state)
        case .log(let l): try c.encode("log", forKey: .type); try c.encode(l, forKey: .log)
        case .error(let m): try c.encode("error", forKey: .type); try c.encode(m, forKey: .message)
        }
    }
}

extension ClientMessage {
    public func jsonString() throws -> String { String(decoding: try JSONEncoder().encode(self), as: UTF8.self) }
    public static func decode(_ text: String) throws -> ClientMessage {
        try JSONDecoder().decode(ClientMessage.self, from: Data(text.utf8))
    }
}

extension ServerMessage {
    public func jsonString() throws -> String { String(decoding: try JSONEncoder().encode(self), as: UTF8.self) }
    public static func decode(_ text: String) throws -> ServerMessage {
        try JSONDecoder().decode(ServerMessage.self, from: Data(text.utf8))
    }
}

// MARK: - REST bodies

public struct MemoryInfo: Codable, Equatable, Sendable {
    public var total: Int
    public var active: Int
    public var cache: Int
    public var peak: Int
    public var budget: Int
    public init(total: Int, active: Int, cache: Int, peak: Int, budget: Int) {
        self.total = total; self.active = active; self.cache = cache; self.peak = peak; self.budget = budget
    }
}

public struct InstalledModel: Codable, Equatable, Sendable, Identifiable {
    /// Folder name in the server's models folder (what Load Model nodes use).
    public var name: String
    /// Size of the weights on disk.
    public var bytes: Int
    public var loaded: Bool
    public var id: String { name }
    public init(name: String, bytes: Int, loaded: Bool) { self.name = name; self.bytes = bytes; self.loaded = loaded }
}

public struct LoadedModelInfo: Codable, Equatable, Sendable {
    public var key: String
    public var name: String
    public var bytes: Int
    public var pinned: Bool
    public var lastUsed: Double
    public init(key: String, name: String, bytes: Int, pinned: Bool, lastUsed: Double) {
        self.key = key; self.name = name; self.bytes = bytes; self.pinned = pinned; self.lastUsed = lastUsed
    }
}

/// `GET /v1/info`.
public struct ServerInfo: Codable, Equatable, Sendable {
    public var protocolVersion: Int
    public var serverVersion: String
    public var hostName: String
    public var chip: String
    public var osVersion: String
    public var gpu: String
    public var memory: MemoryInfo
    /// Node types this server can run.
    public var nodeTypes: [String]
    public var models: [InstalledModel]

    public init(protocolVersion: Int, serverVersion: String, hostName: String, chip: String, osVersion: String,
                gpu: String, memory: MemoryInfo, nodeTypes: [String], models: [InstalledModel]) {
        self.protocolVersion = protocolVersion; self.serverVersion = serverVersion; self.hostName = hostName
        self.chip = chip; self.osVersion = osVersion; self.gpu = gpu; self.memory = memory
        self.nodeTypes = nodeTypes; self.models = models
    }
}

// MARK: - dashboard

public enum RunStatus: String, Codable, Sendable {
    case waiting, running, done, failed, cancelled
}

public struct RunSummary: Codable, Equatable, Sendable, Identifiable {
    public var run: UInt32
    public var workflow: String
    public var client: String
    public var status: RunStatus
    public var created: Double
    public var started: Double?
    public var finished: Double?
    public var seconds: Double?
    public var peakMemory: Int?
    public var error: String?
    /// The node table (ids) and each node's type, so dashboards can name `node` indices.
    public var nodes: [String]
    public var nodeTypes: [String]
    public var outputs: [OutputDescriptor]
    public var id: UInt32 { run }

    public init(run: UInt32, workflow: String, client: String, status: RunStatus, created: Double,
                started: Double? = nil, finished: Double? = nil, seconds: Double? = nil, peakMemory: Int? = nil,
                error: String? = nil, nodes: [String], nodeTypes: [String], outputs: [OutputDescriptor] = []) {
        self.run = run; self.workflow = workflow; self.client = client; self.status = status
        self.created = created; self.started = started; self.finished = finished; self.seconds = seconds
        self.peakMemory = peakMemory; self.error = error; self.nodes = nodes; self.nodeTypes = nodeTypes
        self.outputs = outputs
    }
}

public struct ClientInfo: Codable, Equatable, Sendable {
    public var session: String
    public var name: String
    public var role: ClientRole
    public var address: String
    public var connected: Double
    public var bytesIn: Int
    public var bytesOut: Int
    public var previewHz: Double
    public init(session: String, name: String, role: ClientRole, address: String, connected: Double,
                bytesIn: Int, bytesOut: Int, previewHz: Double) {
        self.session = session; self.name = name; self.role = role; self.address = address
        self.connected = connected; self.bytesIn = bytesIn; self.bytesOut = bytesOut; self.previewHz = previewHz
    }
}

public struct TransferInfo: Codable, Equatable, Sendable {
    public enum Direction: String, Codable, Sendable { case upload, download }
    public var id: String
    public var name: String
    public var direction: Direction
    public var bytes: Int
    public var total: Int?
    public var bytesPerSecond: Double
    public init(id: String, name: String, direction: Direction, bytes: Int, total: Int?, bytesPerSecond: Double) {
        self.id = id; self.name = name; self.direction = direction; self.bytes = bytes; self.total = total
        self.bytesPerSecond = bytesPerSecond
    }
}

public struct ServerSettings: Codable, Equatable, Sendable {
    public var budgetFraction: Double
    public var previewHz: Double
    public var previewMaxSide: Int
    public var host: String
    public var port: Int
    public var dataDirectory: String
    public var modelsDirectory: String
    public init(budgetFraction: Double, previewHz: Double, previewMaxSide: Int, host: String, port: Int,
                dataDirectory: String, modelsDirectory: String) {
        self.budgetFraction = budgetFraction; self.previewHz = previewHz; self.previewMaxSide = previewMaxSide
        self.host = host; self.port = port; self.dataDirectory = dataDirectory; self.modelsDirectory = modelsDirectory
    }
}

public struct DiskInfo: Codable, Equatable, Sendable {
    public var path: String
    public var free: Int
    public var total: Int
    public init(path: String, free: Int, total: Int) { self.path = path; self.free = free; self.total = total }
}

/// Everything the dashboard shows that isn't a time series; pushed on change, at most a few times per second.
public struct DashboardState: Codable, Equatable, Sendable {
    public var info: ServerInfo
    public var started: Double
    public var thermalState: String
    public var queue: [RunSummary]
    public var history: [RunSummary]
    public var loadedModels: [LoadedModelInfo]
    public var clients: [ClientInfo]
    public var transfers: [TransferInfo]
    public var inputCacheBytes: Int
    public var disks: [DiskInfo]
    public var settings: ServerSettings

    public init(info: ServerInfo, started: Double, thermalState: String, queue: [RunSummary], history: [RunSummary],
                loadedModels: [LoadedModelInfo], clients: [ClientInfo], transfers: [TransferInfo],
                inputCacheBytes: Int, disks: [DiskInfo], settings: ServerSettings) {
        self.info = info; self.started = started; self.thermalState = thermalState; self.queue = queue
        self.history = history; self.loadedModels = loadedModels; self.clients = clients
        self.transfers = transfers; self.inputCacheBytes = inputCacheBytes; self.disks = disks
        self.settings = settings
    }
}

public struct LogLine: Codable, Equatable, Sendable {
    public enum Level: String, Codable, Sendable { case debug, info, warning, error }
    public var time: Double
    public var level: Level
    public var message: String
    public init(time: Double, level: Level, message: String) { self.time = time; self.level = level; self.message = message }
}
