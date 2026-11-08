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
        guard case .object(let o) = values[name], let t = o as? T else { throw missing(name) }
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
public final class ResourceCache {
    private var items: [String: Any] = [:]
    /// Creations in progress, so concurrent callers share one load instead of building the resource twice.
    private var pending: [String: Task<Any, Error>] = [:]
    private let lock = NSLock()
    public init() {}

    /// Returns the cached value for `key`, creating it once. Callers that arrive while it is being created wait
    /// for that same creation (loading a model twice at once would double its memory and race on the GPU).
    public func get<T>(_ key: String, create: @escaping () async throws -> T) async throws -> T {
        let task: Task<Any, Error>
        lock.lock()
        if let v = items[key] as? T { lock.unlock(); return v }
        if let p = pending[key] {
            task = p
        } else {
            task = Task { try await create() }
            pending[key] = task
        }
        lock.unlock()
        do {
            let v = try await task.value
            lock.lock()
            if pending[key] == task { pending[key] = nil; items[key] = v }
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

    public func contains(_ key: String) -> Bool { lock.lock(); defer { lock.unlock() }; return items[key] != nil }
    public func remove(_ key: String) { lock.lock(); items[key] = nil; lock.unlock() }
    public func removeAll() { lock.lock(); items.removeAll(); lock.unlock() }
    public var keys: [String] { lock.lock(); defer { lock.unlock() }; return Array(items.keys) }
}
