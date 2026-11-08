import Foundation

/// A literal widget value as stored in graph JSON.
public enum JSONValue: Codable, Hashable, Sendable, CustomStringConvertible {
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let i = try? c.decode(Int.self) { self = .int(i) }
        else if let d = try? c.decode(Double.self) { self = .double(d) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else { throw DecodingError.dataCorruptedError(in: c, debugDescription: "unsupported literal") }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .bool(let b): try c.encode(b)
        case .int(let i): try c.encode(i)
        case .double(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        }
    }

    public var description: String {
        switch self {
        case .bool(let b): return String(b)
        case .int(let i): return String(i)
        case .double(let d): return String(d)
        case .string(let s): return s
        }
    }

    /// Parses a command-line style override (`5`, `0.5`, `true`, anything else → string).
    public static func parse(_ s: String) -> JSONValue {
        if let i = Int(s) { return .int(i) }
        if let d = Double(s) { return .double(d) }
        if s == "true" || s == "false" { return .bool(s == "true") }
        return .string(s)
    }
}

/// One input of a node instance: either a literal or a link to another node's named output.
public enum NodeInput: Codable, Hashable, Sendable {
    case value(JSONValue)
    case link(node: String, output: String)

    private enum Keys: String, CodingKey { case node, output }

    public init(from decoder: Decoder) throws {
        if let c = try? decoder.container(keyedBy: Keys.self), let node = try? c.decode(String.self, forKey: .node) {
            self = .link(node: node, output: try c.decode(String.self, forKey: .output))
        } else {
            self = .value(try JSONValue(from: decoder))
        }
    }

    public func encode(to encoder: Encoder) throws {
        switch self {
        case .value(let v): try v.encode(to: encoder)
        case .link(let node, let output):
            var c = encoder.container(keyedBy: Keys.self)
            try c.encode(node, forKey: .node)
            try c.encode(output, forKey: .output)
        }
    }
}

public struct GraphPoint: Codable, Hashable, Sendable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
}

public struct GraphNode: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var type: String
    public var title: String?
    public var inputs: [String: NodeInput]
    public var position: GraphPoint?

    public init(id: String, type: String, title: String? = nil, inputs: [String: NodeInput] = [:],
                position: GraphPoint? = nil) {
        self.id = id
        self.type = type
        self.title = title
        self.inputs = inputs
        self.position = position
    }
}

/// The document: a set of node instances wired by links stored on their inputs.
public struct Graph: Codable, Hashable, Sendable {
    public static let currentVersion = 1

    public var version: Int
    public var nodes: [GraphNode]
    /// Free-form notes that travel with the graph (and into saved PNGs), e.g. the prompt template a queued prompt
    /// was expanded from. Optional, so graphs without it still load.
    public var metadata: [String: String]?
    /// Editor groups (visual only).
    public var groups: [GraphGroup]?
    /// Subgraph definitions by id, shared by all instances (including ones nested inside other definitions).
    public var subgraphs: [String: SubgraphDef]?

    public init(version: Int = Graph.currentVersion, nodes: [GraphNode] = [], metadata: [String: String]? = nil,
                groups: [GraphGroup]? = nil, subgraphs: [String: SubgraphDef]? = nil) {
        self.version = version
        self.nodes = nodes
        self.metadata = metadata
        self.groups = groups
        self.subgraphs = subgraphs
    }

    public subscript(id: String) -> GraphNode? {
        get { nodes.first { $0.id == id } }
        set {
            if let i = nodes.firstIndex(where: { $0.id == id }) {
                if let newValue { nodes[i] = newValue } else { nodes.remove(at: i) }
            } else if let newValue {
                nodes.append(newValue)
            }
        }
    }

    /// Sets a literal input, e.g. from `--set 3.seed=5`. `instance/inner` paths reach into subgraphs (changing the
    /// definition, which every instance of it shares).
    public mutating func setValue(_ value: JSONValue, node: String, input: String) throws {
        if self[node] == nil, node.contains("/") {
            var parts = node.split(separator: "/").map(String.init)
            let innerID = parts.removeLast()
            var level = self
            var defID: String?
            for id in parts {
                guard let n = level[id], let (d, def) = subgraphDef(for: n) else { throw GraphError.unknownNode(node) }
                defID = d
                level = def.graph
            }
            guard let defID, var inner = subgraphs?[defID]?.graph else { throw GraphError.unknownNode(node) }
            try inner.setValue(value, node: innerID, input: input)
            subgraphs?[defID]?.graph = inner
            return
        }
        guard var n = self[node] else { throw GraphError.unknownNode(node) }
        n.inputs[input] = .value(value)
        self[node] = n
    }

    public static func load(_ url: URL) throws -> Graph {
        try JSONDecoder().decode(Graph.self, from: Data(contentsOf: url))
    }

    public func jsonData(pretty: Bool = true) throws -> Data {
        let e = JSONEncoder()
        e.outputFormatting = pretty ? [.prettyPrinted, .sortedKeys] : [.sortedKeys]
        return try e.encode(self)
    }

    public func jsonString(pretty: Bool = false) -> String {
        (try? jsonData(pretty: pretty)).flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }
}
