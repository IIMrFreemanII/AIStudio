import Foundation

/// Socket type of a node input/output. Scalars can be literals (widgets); everything else must be linked.
public struct SocketType: RawRepresentable, Hashable, Codable, Sendable, CustomStringConvertible {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ raw: String) { self.rawValue = raw }
    public var description: String { rawValue }

    public static let int = SocketType("INT")
    public static let float = SocketType("FLOAT")
    public static let string = SocketType("STRING")
    public static let bool = SocketType("BOOL")
    public static let model = SocketType("MODEL")
    public static let conditioning = SocketType("CONDITIONING")
    public static let latent = SocketType("LATENT")
    public static let image = SocketType("IMAGE")
    public static let mask = SocketType("MASK")
    public static let video = SocketType("VIDEO")
    public static let audio = SocketType("AUDIO")
    /// Accepts any type (a Reroute's sockets); resolved to the real type before execution.
    public static let any = SocketType("*")

    public var isScalar: Bool { [.int, .float, .string, .bool].contains(self) }

    /// Whether a link of type `other` may connect to a socket of this type.
    public func accepts(_ other: SocketType) -> Bool { self == other || self == .any || other == .any }
}

/// How the UI should present a scalar input.
public enum Widget: Hashable, Sendable {
    case number(min: Double? = nil, max: Double? = nil, step: Double? = nil)
    case text(multiline: Bool = false)
    case choice([String])
    case toggle
    case seed
    case filePath
    /// A pick from the app's library (`"lora"`): the value is the item's id; the app draws a menu of what's installed.
    case library(String)
}

public struct InputSpec: Hashable, Sendable {
    public var name: String
    public var type: SocketType
    public var defaultValue: JSONValue?
    public var widget: Widget?
    public var optional: Bool

    public init(_ name: String, _ type: SocketType, default defaultValue: JSONValue? = nil, widget: Widget? = nil,
                optional: Bool = false) {
        self.name = name
        self.type = type
        self.defaultValue = defaultValue
        self.widget = widget
        self.optional = optional
    }
}

public struct OutputSpec: Hashable, Sendable {
    public var name: String
    public var type: SocketType
    public init(_ name: String, _ type: SocketType) { self.name = name; self.type = type }
}

public struct NodeSchema: Sendable {
    public var type: String
    public var displayName: String
    public var category: String
    public var inputs: [InputSpec]
    public var outputs: [OutputSpec]
    /// Output nodes are the roots the executor works back from (SaveImage, PreviewImage…).
    public var isOutput: Bool
    /// Re-run every time even when inputs are unchanged (side effects such as writing files).
    public var alwaysExecute: Bool

    public init(type: String, displayName: String? = nil, category: String, inputs: [InputSpec],
                outputs: [OutputSpec], isOutput: Bool = false, alwaysExecute: Bool = false) {
        self.type = type
        self.displayName = displayName ?? type
        self.category = category
        self.inputs = inputs
        self.outputs = outputs
        self.isOutput = isOutput
        self.alwaysExecute = alwaysExecute
    }

    public func input(_ name: String) -> InputSpec? { inputs.first { $0.name == name } }
    public func output(_ name: String) -> OutputSpec? { outputs.first { $0.name == name } }
}

/// A runtime value flowing along a link.
public enum Value {
    case int(Int)
    case float(Double)
    case string(String)
    case bool(Bool)
    case object(Any)

    init(literal: JSONValue, as type: SocketType) throws {
        switch (type, literal) {
        case (.int, .int(let i)): self = .int(i)
        case (.int, .double(let d)) where d == d.rounded(): self = .int(Int(d))
        case (.float, .double(let d)): self = .float(d)
        case (.float, .int(let i)): self = .float(Double(i))
        case (.string, .string(let s)): self = .string(s)
        case (.string, let v): self = .string(v.description)
        case (.bool, .bool(let b)): self = .bool(b)
        default: throw GraphError.literalMismatch(expected: type, got: literal)
        }
    }
}

/// Resolved inputs handed to a node's `execute`.
public struct NodeInputs {
    public let nodeID: String
    let values: [String: Value]

    public init(nodeID: String, values: [String: Value]) {
        self.nodeID = nodeID
        self.values = values
    }

    public func has(_ name: String) -> Bool { values[name] != nil }

    public func int(_ name: String) throws -> Int {
        guard case .int(let v) = values[name] else { throw missing(name) }
        return v
    }

    public func float(_ name: String) throws -> Double {
        switch values[name] {
        case .float(let v): return v
        case .int(let v): return Double(v)
        default: throw missing(name)
        }
    }

    public func string(_ name: String) throws -> String {
        guard case .string(let v) = values[name] else { throw missing(name) }
        return v
    }

    public func bool(_ name: String) throws -> Bool {
        guard case .bool(let v) = values[name] else { throw missing(name) }
        return v
    }

    public func object<T>(_ name: String, as: T.Type = T.self) throws -> T {
        guard case .object(let o) = values[name] else { throw missing(name) }
        guard let t = o as? T else {
            // E.g. a video model wired into an image sampler.
            throw GraphError.nodeFailed(node: nodeID, message: "input '\(name)' is a \(Swift.type(of: o)), which can't be used as \(T.self)")
        }
        return t
    }

    public func optionalObject<T>(_ name: String, as: T.Type = T.self) -> T? {
        guard case .object(let o) = values[name] else { return nil }
        return o as? T
    }

    func missing(_ name: String) -> GraphError { .missingInput(node: nodeID, input: name) }
}

/// Services a node can use while executing.
public final class ExecutionContext {
    public let nodeID: String
    public let graph: Graph
    public let resources: ResourceCache
    let emit: (ExecutionEvent) -> Void

    init(nodeID: String, graph: Graph, resources: ResourceCache, emit: @escaping (ExecutionEvent) -> Void) {
        self.nodeID = nodeID
        self.graph = graph
        self.resources = resources
        self.emit = emit
    }

    /// Reports progress (e.g. sampler steps). `preview` is model/UI-specific (latents, an image…).
    public func progress(step: Int, total: Int, preview: Any? = nil) {
        emit(.nodeProgress(node: nodeID, step: step, total: total, preview: preview))
    }

    /// Shows what the node is doing when that isn't a step count (waiting for a server, downloading); nil clears it.
    public func status(_ text: String?) {
        emit(.nodeStatus(node: nodeID, text: text))
    }

    /// Throws `CancellationError` when the run has been cancelled.
    public func checkCancellation() throws { try Task.checkCancellation() }
}

public protocol NodeImplementation {
    static var schema: NodeSchema { get }
    init()
    /// Returns values keyed by output name. Output nodes may return UI results (e.g. `"images"`).
    func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value]
}

/// Known node types.
public final class NodeRegistry {
    private var types: [String: NodeImplementation.Type] = [:]
    public init() {}

    public func register(_ t: NodeImplementation.Type) { types[t.schema.type] = t }
    public func register(_ ts: [NodeImplementation.Type]) { ts.forEach(register) }

    public func implementation(_ type: String) -> NodeImplementation.Type? { types[type] }
    public func schema(_ type: String) -> NodeSchema? { types[type]?.schema }
    public var schemas: [NodeSchema] { types.values.map { $0.schema }.sorted { $0.type < $1.type } }
}

/// Long-lived objects shared across runs (loaded models), keyed by a caller-chosen string.
///
/// With a `budgetBytes`, it keeps resources only while they fit: before creating a new one it evicts the least
/// recently used entries that aren't pinned (everything the current run has touched is pinned) until the resident
/// size plus the new one's estimate is within budget. `onEvict` hears about evictions so callers can drop other
/// references (cached outputs) and return memory.
public final class ResourceCache {
    public struct EntryInfo: Sendable, Hashable {
        public let key: String
        public let bytes: Int
        public let lastUsed: Date
        public let pinned: Bool
    }

    private struct Entry {
        var value: Any
        var bytes: Int
        var tick: UInt64
        var lastUsed: Date
    }

    private var items: [String: Entry] = [:]
    /// Creations in progress, so concurrent callers share one load instead of building the resource twice.
    private var pending: [String: Task<Any, Error>] = [:]
    private var pinCounts: [String: Int] = [:]
    /// Keys used by the run in progress (pinned until it ends).
    private var runKeys: Set<String>?
    private var tick: UInt64 = 0
    private let lock = NSLock()

    /// Upper bound for the summed `estimatedBytes` of resident entries; nil = unlimited.
    public var budgetBytes: Int?
    /// Called with the evicted keys, outside the lock, before the new resource is created.
    public var onEvict: (([String]) -> Void)?

    public init(budgetBytes: Int? = nil) { self.budgetBytes = budgetBytes }

    /// Returns the cached value for `key`, creating it once. Callers that arrive while it is being created wait
    /// for that same creation (loading a model twice at once would double its memory and race on the GPU).
    public func get<T>(_ key: String, estimatedBytes: Int = 0, create: @escaping () async throws -> T) async throws -> T {
        let task: Task<Any, Error>
        var evicted: [String] = []
        lock.lock()
        runKeys?.insert(key)
        if var e = items[key], let v = e.value as? T {
            tick += 1
            e.tick = tick
            e.lastUsed = Date()
            items[key] = e
            lock.unlock()
            return v
        }
        if let p = pending[key] {
            task = p
            lock.unlock()
        } else {
            evicted = evictionPlan(adding: estimatedBytes, excluding: key)
            for k in evicted { items[k] = nil }
            task = Task { try await create() }
            pending[key] = task
            lock.unlock()
            if !evicted.isEmpty { onEvict?(evicted) }
        }
        do {
            let v = try await task.value
            lock.lock()
            if pending[key] == task {
                pending[key] = nil
                tick += 1
                items[key] = Entry(value: v, bytes: estimatedBytes, tick: tick, lastUsed: Date())
            }
            lock.unlock()
            guard let t = v as? T else { throw ResourceError.typeMismatch(key) }
            return t
        } catch {
            lock.lock()
            if pending[key] == task { pending[key] = nil }
            lock.unlock()
            throw error
        }
    }

    public enum ResourceError: Error { case typeMismatch(String) }

    /// Least-recently-used unpinned keys to drop so `bytes` more fits the budget (caller holds the lock).
    func evictionPlan(adding bytes: Int, excluding key: String) -> [String] {
        guard let budget = budgetBytes else { return [] }
        var resident = items.values.reduce(0) { $0 + $1.bytes }
        var plan: [String] = []
        let candidates = items.filter { k, _ in k != key && !isPinned(k) }.sorted { $0.value.tick < $1.value.tick }
        for (k, e) in candidates where resident + bytes > budget {
            plan.append(k)
            resident -= e.bytes
        }
        return plan
    }

    private func isPinned(_ key: String) -> Bool { (pinCounts[key] ?? 0) > 0 || runKeys?.contains(key) == true }

    /// Starts pinning every key the run touches; `endRun` releases them.
    public func beginRun() { lock.lock(); runKeys = []; lock.unlock() }
    public func endRun() { lock.lock(); runKeys = nil; lock.unlock() }

    public func pin(_ key: String) { lock.lock(); pinCounts[key, default: 0] += 1; lock.unlock() }
    public func unpin(_ key: String) {
        lock.lock()
        pinCounts[key] = max(0, (pinCounts[key] ?? 0) - 1)
        lock.unlock()
    }

    public func contains(_ key: String) -> Bool { lock.lock(); defer { lock.unlock() }; return items[key] != nil }

    /// Drops `key` (and tells `onEvict`).
    public func remove(_ key: String) {
        lock.lock()
        let had = items.removeValue(forKey: key) != nil
        lock.unlock()
        if had { onEvict?([key]) }
    }

    public func removeAll() {
        lock.lock()
        let keys = Array(items.keys)
        items.removeAll()
        lock.unlock()
        if !keys.isEmpty { onEvict?(keys) }
    }

    public var keys: [String] { lock.lock(); defer { lock.unlock() }; return Array(items.keys) }

    /// Resident entries, most recently used first.
    public var entries: [EntryInfo] {
        lock.lock(); defer { lock.unlock() }
        return items.sorted { $0.value.tick > $1.value.tick }.map {
            EntryInfo(key: $0.key, bytes: $0.value.bytes, lastUsed: $0.value.lastUsed, pinned: isPinned($0.key))
        }
    }

    public var residentBytes: Int { lock.lock(); defer { lock.unlock() }; return items.values.reduce(0) { $0 + $1.bytes } }
}
