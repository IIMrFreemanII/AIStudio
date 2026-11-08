import CryptoKit
import Foundation

public enum GraphError: Error, CustomStringConvertible, Equatable {
    case unknownNode(String)
    case unknownNodeType(node: String, type: String)
    case unknownInput(node: String, input: String)
    case unknownOutput(node: String, output: String)
    case missingInput(node: String, input: String)
    case typeMismatch(node: String, input: String, expected: SocketType, got: SocketType)
    case literalMismatch(expected: SocketType, got: JSONValue)
    case literalForSocket(node: String, input: String, type: SocketType)
    case cycle([String])
    case noOutputs
    case nodeFailed(node: String, message: String)

    public var description: String {
        switch self {
        case .unknownNode(let n): return "no node with id '\(n)'"
        case .unknownNodeType(let n, let t): return "node \(n): unknown type '\(t)'"
        case .unknownInput(let n, let i): return "node \(n): no input named '\(i)'"
        case .unknownOutput(let n, let o): return "node \(n) has no output '\(o)'"
        case .missingInput(let n, let i): return "node \(n): input '\(i)' is required"
        case .typeMismatch(let n, let i, let e, let g): return "node \(n).\(i): expected \(e), got \(g)"
        case .literalMismatch(let e, let g): return "value \(g) is not a valid \(e)"
        case .literalForSocket(let n, let i, let t): return "node \(n).\(i): \(t) inputs must be linked"
        case .cycle(let ids): return "graph has a cycle through \(ids.joined(separator: ", "))"
        case .noOutputs: return "graph has no output nodes (SaveImage, PreviewImage…)"
        case .nodeFailed(let n, let m): return "node \(n) failed: \(m)"
        }
    }
}

public enum ExecutionEvent {
    case started(nodes: [String])
    case nodeStarted(node: String)
    case nodeCached(node: String)
    case nodeProgress(node: String, step: Int, total: Int, preview: Any?)
    case nodeFinished(node: String, seconds: Double, peakMemory: Int?)
    case nodeFailed(node: String, error: Error)
    case finished(seconds: Double)
    case cancelled
}

public struct RunResult {
    /// Values returned by output nodes, keyed by node id.
    public var outputs: [String: [String: Value]] = [:]
    public var executed: [String] = []
    public var cached: [String] = []
    public var seconds: Double = 0
}

/// Hooks for metrics the runtime cannot measure itself (it does not depend on MLX).
public struct ExecutorHooks {
    public var resetPeakMemory: () -> Void = {}
    public var peakMemory: () -> Int? = { nil }
    public init(resetPeakMemory: @escaping () -> Void = {}, peakMemory: @escaping () -> Int? = { nil }) {
        self.resetPeakMemory = resetPeakMemory
        self.peakMemory = peakMemory
    }
}

/// Validates and runs graphs, caching node outputs between runs by a signature of their type, literal inputs and
/// upstream signatures — so changing one widget only re-runs that node and what depends on it.
///
/// Not thread-safe: run one graph at a time (the app serialises runs through its queue).
public final class GraphExecutor {
    public let registry: NodeRegistry
    public let resources: ResourceCache
    public var hooks: ExecutorHooks

    private var cache: [String: [String: Value]] = [:]

    public init(registry: NodeRegistry, resources: ResourceCache = ResourceCache(), hooks: ExecutorHooks = .init()) {
        self.registry = registry
        self.resources = resources
        self.hooks = hooks
    }

    public func clearCache() { cache.removeAll() }
    public var cachedEntryCount: Int { cache.count }

    // MARK: validation

    /// Checks node types, link targets, socket types and required inputs; returns nodes in topological order.
    public func validate(_ graph: Graph) throws -> [GraphNode] {
        var byID: [String: GraphNode] = [:]
        for n in graph.nodes { byID[n.id] = n }

        for node in graph.nodes {
            guard let schema = registry.schema(node.type) else {
                throw GraphError.unknownNodeType(node: node.id, type: node.type)
            }
            for (name, input) in node.inputs {
                guard let spec = schema.input(name) else { throw GraphError.unknownInput(node: node.id, input: name) }
                switch input {
                case .value(let v):
                    guard spec.type.isScalar else {
                        throw GraphError.literalForSocket(node: node.id, input: name, type: spec.type)
                    }
                    _ = try Value(literal: v, as: spec.type)
                case .link(let src, let out):
                    guard let srcNode = byID[src] else { throw GraphError.unknownNode(src) }
                    guard let srcSchema = registry.schema(srcNode.type) else {
                        throw GraphError.unknownNodeType(node: src, type: srcNode.type)
                    }
                    guard let o = srcSchema.output(out) else { throw GraphError.unknownOutput(node: src, output: out) }
                    guard o.type == spec.type else {
                        throw GraphError.typeMismatch(node: node.id, input: name, expected: spec.type, got: o.type)
                    }
                }
            }
            for spec in schema.inputs where !spec.optional && node.inputs[spec.name] == nil && spec.defaultValue == nil {
                throw GraphError.missingInput(node: node.id, input: spec.name)
            }
        }
        return try topologicalOrder(graph)
    }

    func topologicalOrder(_ graph: Graph) throws -> [GraphNode] {
        var indegree: [String: Int] = [:]
        var dependents: [String: [String]] = [:]
        for n in graph.nodes {
            indegree[n.id, default: 0] += 0
            for case .link(let src, _) in n.inputs.values {
                indegree[n.id, default: 0] += 1
                dependents[src, default: []].append(n.id)
            }
        }
        // Stable: start from nodes in document order.
        var ready = graph.nodes.map(\.id).filter { indegree[$0] == 0 }
        var order: [String] = []
        while !ready.isEmpty {
            let id = ready.removeFirst()
            order.append(id)
            for d in dependents[id] ?? [] {
                indegree[d]! -= 1
                if indegree[d] == 0 { ready.append(d) }
            }
        }
        guard order.count == graph.nodes.count else {
            throw GraphError.cycle(graph.nodes.map(\.id).filter { !order.contains($0) })
        }
        let byID = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        return order.map { byID[$0]! }
    }

    /// Nodes that the requested outputs depend on (inclusive).
    func requiredNodes(_ graph: Graph, outputs: [String]) -> Set<String> {
        let byID = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        var needed = Set<String>()
        var stack = outputs
        while let id = stack.popLast() {
            guard needed.insert(id).inserted, let n = byID[id] else { continue }
            for case .link(let src, _) in n.inputs.values { stack.append(src) }
        }
        return needed
    }

    /// Cache key: node type + resolved literals + upstream signatures. Positions/titles don't matter.
    func signatures(_ order: [GraphNode]) -> [String: String] {
        var sigs: [String: String] = [:]
        for node in order {
            let schema = registry.schema(node.type)!
            var parts = ["type=\(node.type)"]
            for spec in schema.inputs {
                switch node.inputs[spec.name] {
                case .value(let v): parts.append("\(spec.name)=v:\(v.description)")
                case .link(let src, let out): parts.append("\(spec.name)=l:\(sigs[src]!).\(out)")
                case nil: parts.append("\(spec.name)=d:\(spec.defaultValue?.description ?? "nil")")
                }
            }
            let digest = SHA256.hash(data: Data(parts.joined(separator: "|").utf8))
            sigs[node.id] = digest.map { String(format: "%02x", $0) }.joined()
        }
        return sigs
    }

    // MARK: execution

    /// Runs every output node (or just `targets`) and what they depend on.
    @discardableResult
    public func run(_ graph: Graph, targets: [String]? = nil,
                    onEvent: @escaping (ExecutionEvent) -> Void = { _ in }) async throws -> RunResult {
        let start = Date()
        let order = try validate(graph)
        let outputs = targets ?? order.filter { registry.schema($0.type)!.isOutput }.map(\.id)
        guard !outputs.isEmpty else { throw GraphError.noOutputs }
        let needed = requiredNodes(graph, outputs: outputs)
        let plan = order.filter { needed.contains($0.id) }
        let sigs = signatures(plan)
        onEvent(.started(nodes: plan.map(\.id)))

        var values: [String: [String: Value]] = [:]
        var result = RunResult()
        var used = Set<String>()

        do {
            for node in plan {
                try Task.checkCancellation()
                let impl = registry.implementation(node.type)!
                let schema = impl.schema
                let sig = sigs[node.id]!
                used.insert(sig)

                if !schema.alwaysExecute, let hit = cache[sig] {
                    values[node.id] = hit
                    result.cached.append(node.id)
                    onEvent(.nodeCached(node: node.id))
                    if schema.isOutput { result.outputs[node.id] = hit }
                    continue
                }

                let inputs = try resolveInputs(node, schema: schema, values: values)
                onEvent(.nodeStarted(node: node.id))
                hooks.resetPeakMemory()
                let t = Date()
                let ctx = ExecutionContext(nodeID: node.id, graph: graph, resources: resources, emit: onEvent)
                let out: [String: Value]
                do {
                    out = try await impl.init().execute(inputs, context: ctx)
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    onEvent(.nodeFailed(node: node.id, error: error))
                    throw GraphError.nodeFailed(node: node.id, message: String(describing: error))
                }
                for o in schema.outputs where out[o.name] == nil {
                    throw GraphError.nodeFailed(node: node.id, message: "did not produce output '\(o.name)'")
                }
                values[node.id] = out
                cache[sig] = out
                result.executed.append(node.id)
                if schema.isOutput { result.outputs[node.id] = out }
                onEvent(.nodeFinished(node: node.id, seconds: Date().timeIntervalSince(t),
                                      peakMemory: hooks.peakMemory()))
            }
        } catch is CancellationError {
            onEvent(.cancelled)
            throw CancellationError()
        }

        // Keep only what this run touched, so stale branches don't pin memory.
        cache = cache.filter { used.contains($0.key) }
        result.seconds = Date().timeIntervalSince(start)
        onEvent(.finished(seconds: result.seconds))
        return result
    }

    func resolveInputs(_ node: GraphNode, schema: NodeSchema, values: [String: [String: Value]]) throws
        -> NodeInputs
    {
        var resolved: [String: Value] = [:]
        for spec in schema.inputs {
            switch node.inputs[spec.name] {
            case .value(let v):
                resolved[spec.name] = try Value(literal: v, as: spec.type)
            case .link(let src, let out):
                guard let v = values[src]?[out] else { throw GraphError.missingInput(node: node.id, input: spec.name) }
                resolved[spec.name] = v
            case nil:
                if let d = spec.defaultValue { resolved[spec.name] = try Value(literal: d, as: spec.type) }
            }
        }
        return NodeInputs(nodeID: node.id, values: resolved)
    }
}
