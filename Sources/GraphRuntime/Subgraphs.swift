import Foundation

/// A labelled rectangle behind nodes in the editor. Purely visual: it never affects execution.
public struct GraphGroup: Codable, Hashable, Sendable, Identifiable {
    public var id: String
    public var title: String
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double
    /// One of a few named tints (`gray`, `blue`, `green`, `yellow`, `red`, `purple`).
    public var color: String

    public init(id: String, title: String, x: Double, y: Double, width: Double, height: Double, color: String = "blue") {
        self.id = id
        self.title = title
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
    }

    public func contains(x px: Double, y py: Double) -> Bool {
        px >= x && px <= x + width && py >= y && py <= y + height
    }
}

/// One socket of a subgraph instance, mapped to the inner sockets it stands for.
public struct SubgraphPort: Codable, Hashable, Sendable {
    public struct Target: Codable, Hashable, Sendable {
        public var node: String
        public var socket: String
        public init(node: String, socket: String) { self.node = node; self.socket = socket }
    }

    public var name: String
    public var type: SocketType
    /// Inputs: every inner input this port feeds. Outputs: exactly one inner output.
    public var targets: [Target]

    public init(name: String, type: SocketType, targets: [Target]) {
        self.name = name
        self.type = type
        self.targets = targets
    }
}

/// A reusable piece of graph. Instances are nodes of type ``Graph/subgraphType`` whose `def` input names it.
public struct SubgraphDef: Codable, Hashable, Sendable {
    public var name: String
    public var graph: Graph
    public var inputs: [SubgraphPort]
    public var outputs: [SubgraphPort]

    public init(name: String, graph: Graph, inputs: [SubgraphPort], outputs: [SubgraphPort]) {
        self.name = name
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs
    }
}

extension GraphError {
    static func unknownSubgraph(_ node: String) -> GraphError {
        .nodeFailed(node: node, message: "subgraph definition is missing")
    }
}

extension Graph {
    /// Node type of a subgraph instance.
    public static let subgraphType = "Subgraph"
    /// Node type that passes a link through unchanged (spliced out before execution).
    public static let rerouteType = "Reroute"
    /// Node type of a sticky note (never runs).
    public static let noteType = "Note"

    /// The definition an instance node refers to.
    public func subgraphDef(for node: GraphNode) -> (id: String, def: SubgraphDef)? {
        guard node.type == Self.subgraphType, case .value(.string(let id))? = node.inputs["def"],
              let def = subgraphs?[id] else { return nil }
        return (id, def)
    }

    /// A schema for an instance node, built from its definition's ports (for the editor and the inspector).
    public func subgraphSchema(_ node: GraphNode) -> NodeSchema? {
        guard let (_, def) = subgraphDef(for: node) else { return nil }
        return NodeSchema(
            type: Self.subgraphType, displayName: def.name, category: "subgraph",
            inputs: def.inputs.map { InputSpec($0.name, $0.type, optional: true) },
            outputs: def.outputs.map { OutputSpec($0.name, $0.type) })
    }

    // MARK: flattening

    /// The graph as it executes: subgraph instances inlined (inner ids become `instance/inner`), reroutes spliced
    /// out, notes dropped. Links through removed nodes are redirected to their real source.
    public func flattened() throws -> Graph {
        var g = self
        g.nodes = try flatten(depth: 0).nodes
        g.groups = nil
        g.subgraphs = nil
        return g
    }

    /// Nodes of this level, flattened, and where each removed node's outputs really come from.
    func flatten(depth: Int) throws -> (nodes: [GraphNode], redirect: [String: [String: NodeInput]]) {
        let defs = subgraphs ?? [:]
        var out: [GraphNode] = []
        var redirect: [String: [String: NodeInput]] = [:]
        for n in self.nodes {
            switch n.type {
            case Self.noteType:
                continue
            case Self.rerouteType:
                redirect[n.id] = n.inputs["in"].map { ["out": $0] } ?? [:]
            case Self.subgraphType:
                guard depth < 8, let (_, def) = subgraphDef(for: n) else { throw GraphError.unknownSubgraph(n.id) }
                var inner = def.graph.prefixed(n.id + "/")
                inner.subgraphs = defs
                // Bind the instance's inputs (links or literals in this level's namespace) to the inner sockets.
                for port in def.inputs {
                    guard let value = n.inputs[port.name] else { continue }
                    for t in port.targets {
                        let id = n.id + "/" + t.node
                        guard var target = inner[id] else { continue }
                        target.inputs[t.socket] = value
                        inner[id] = target
                    }
                }
                let flat = try inner.flatten(depth: depth + 1)
                out += flat.nodes
                var outs: [String: NodeInput] = [:]
                for port in def.outputs {
                    guard let t = port.targets.first else { continue }
                    let link = NodeInput.link(node: n.id + "/" + t.node, output: t.socket)
                    outs[port.name] = Self.resolve(link, flat.redirect, depth: 0) ?? link
                }
                redirect[n.id] = outs
                for (k, v) in flat.redirect { redirect[k] = v }
            default:
                out.append(n)
            }
        }
        // Point every link at its real source; links into removed nodes with no source are dropped.
        for i in out.indices {
            for (k, v) in out[i].inputs {
                guard case .link(let src, _) = v, redirect[src] != nil else { continue }
                out[i].inputs[k] = Self.resolve(v, redirect, depth: 0)
            }
        }
        return (out, redirect)
    }

    static func resolve(_ input: NodeInput, _ redirect: [String: [String: NodeInput]], depth: Int) -> NodeInput? {
        guard depth < 64, case .link(let src, let out) = input, let map = redirect[src] else { return input }
        guard let next = map[out] else { return nil }
        return resolve(next, redirect, depth: depth + 1)
    }

    /// Copy with every node id (and the links between them) prefixed.
    func prefixed(_ p: String) -> Graph {
        var g = self
        let ids = Set(nodes.map(\.id))
        g.nodes = nodes.map { n in
            var c = n
            c.id = p + n.id
            for (k, v) in c.inputs {
                if case .link(let src, let out) = v, ids.contains(src) { c.inputs[k] = .link(node: p + src, output: out) }
            }
            return c
        }
        return g
    }

    // MARK: collapse & expand

    /// Moves the nodes `ids` into a new subgraph and puts one instance node in their place.
    ///
    /// Links from outside into the selection become input ports (one per distinct source), links from the selection
    /// to the outside become output ports. `typeOf` gives socket types: `(node, socket, isInput)`.
    public func collapsing(_ ids: Set<String>, name: String,
                           typeOf: (GraphNode, String, Bool) -> SocketType?) -> (graph: Graph, instance: String)? {
        let inside = nodes.filter { ids.contains($0.id) }
        guard !inside.isEmpty else { return nil }
        var g = self
        var defs = subgraphs ?? [:]
        var defID = "sub1"
        var k = 1
        while defs[defID] != nil { k += 1; defID = "sub\(k)" }
        var instance = name.lowercased().filter { $0.isLetter || $0.isNumber }
        if instance.isEmpty { instance = "subgraph" }
        let base = instance
        k = 1
        while self[instance] != nil { k += 1; instance = "\(base)\(k)" }

        var innerNodes = inside
        var inputs: [SubgraphPort] = []
        var instanceInputs: [String: NodeInput] = ["def": .value(.string(defID))]
        var portForSource: [String: String] = [:]  // "src.out" -> port name
        func unique(_ name: String, _ taken: [SubgraphPort]) -> String {
            var n = name
            var i = 2
            while taken.contains(where: { $0.name == n }) || n == "def" { n = "\(name)_\(i)"; i += 1 }
            return n
        }
        for i in innerNodes.indices {
            for (socket, v) in innerNodes[i].inputs.sorted(by: { $0.key < $1.key }) {
                guard case .link(let src, let out) = v, !ids.contains(src) else { continue }
                let key = "\(src).\(out)"
                let target = SubgraphPort.Target(node: innerNodes[i].id, socket: socket)
                if let p = portForSource[key], let j = inputs.firstIndex(where: { $0.name == p }) {
                    inputs[j].targets.append(target)
                } else {
                    let type = self[src].flatMap { typeOf($0, out, false) } ?? typeOf(innerNodes[i], socket, true) ?? .any
                    let port = SubgraphPort(name: unique(socket, inputs), type: type, targets: [target])
                    inputs.append(port)
                    portForSource[key] = port.name
                    instanceInputs[port.name] = v
                }
                innerNodes[i].inputs[socket] = nil
            }
        }

        var outputs: [SubgraphPort] = []
        var portForOutput: [String: String] = [:]
        for i in g.nodes.indices where !ids.contains(g.nodes[i].id) {
            for (socket, v) in g.nodes[i].inputs.sorted(by: { $0.key < $1.key }) {
                guard case .link(let src, let out) = v, ids.contains(src) else { continue }
                let key = "\(src).\(out)"
                var name = portForOutput[key]
                if name == nil {
                    let type = self[src].flatMap { typeOf($0, out, false) } ?? .any
                    let port = SubgraphPort(name: unique(out, outputs), type: type,
                                            targets: [.init(node: src, socket: out)])
                    outputs.append(port)
                    portForOutput[key] = port.name
                    name = port.name
                }
                g.nodes[i].inputs[socket] = .link(node: instance, output: name!)
            }
        }

        // Instance at the selection's top-left; inner positions kept, shifted to start near the origin.
        let xs = inside.compactMap { $0.position?.x }, ys = inside.compactMap { $0.position?.y }
        let (minX, minY) = (xs.min() ?? 0, ys.min() ?? 0)
        innerNodes = innerNodes.map { n in
            var c = n
            c.position = GraphPoint(x: (n.position?.x ?? 0) - minX + 40, y: (n.position?.y ?? 0) - minY + 40)
            return c
        }
        defs[defID] = SubgraphDef(name: name, graph: Graph(nodes: innerNodes), inputs: inputs, outputs: outputs)
        g.subgraphs = defs
        let index = g.nodes.firstIndex { ids.contains($0.id) } ?? g.nodes.count
        g.nodes.removeAll { ids.contains($0.id) }
        g.nodes.insert(GraphNode(id: instance, type: Self.subgraphType, title: name, inputs: instanceInputs,
                                 position: GraphPoint(x: minX, y: minY)), at: min(index, g.nodes.count))
        return (g, instance)
    }

    /// Replaces an instance node with its definition's nodes (renamed if their ids are taken) and rewires the ports.
    /// The definition is dropped when nothing else uses it.
    public func expanding(_ instanceID: String) -> (graph: Graph, nodes: [String])? {
        guard let inst = self[instanceID], let (defID, def) = subgraphDef(for: inst) else { return nil }
        var g = self
        var taken = Set(nodes.map(\.id)).subtracting([instanceID])
        var rename: [String: String] = [:]
        for n in def.graph.nodes {
            var id = n.id
            var k = 2
            while taken.contains(id) { id = "\(n.id)_\(k)"; k += 1 }
            rename[n.id] = id
            taken.insert(id)
        }
        let origin = inst.position ?? GraphPoint(x: 0, y: 0)
        var inner: [GraphNode] = def.graph.nodes.map { n in
            var c = n
            c.id = rename[n.id]!
            c.position = GraphPoint(x: origin.x + (n.position?.x ?? 0) - 40, y: origin.y + (n.position?.y ?? 0) - 40)
            for (k, v) in c.inputs {
                if case .link(let src, let out) = v, let r = rename[src] { c.inputs[k] = .link(node: r, output: out) }
            }
            return c
        }
        for port in def.inputs {
            guard let value = inst.inputs[port.name] else { continue }
            for t in port.targets {
                guard let i = inner.firstIndex(where: { $0.id == rename[t.node] }) else { continue }
                inner[i].inputs[t.socket] = value
            }
        }
        var outMap: [String: NodeInput] = [:]
        for port in def.outputs {
            if let t = port.targets.first, let r = rename[t.node] { outMap[port.name] = .link(node: r, output: t.socket) }
        }
        let index = g.nodes.firstIndex { $0.id == instanceID } ?? g.nodes.count
        g.nodes.remove(at: index)
        for i in g.nodes.indices {
            for (k, v) in g.nodes[i].inputs {
                guard case .link(instanceID, let out) = v else { continue }
                g.nodes[i].inputs[k] = outMap[out]
            }
        }
        g.nodes.insert(contentsOf: inner, at: index)
        if !g.nodes.contains(where: { $0.type == Self.subgraphType && $0.inputs["def"] == .value(.string(defID)) })
            && !(g.subgraphs ?? [:]).values.contains(where: { $0.graph.nodes.contains { $0.inputs["def"] == .value(.string(defID)) } }) {
            g.subgraphs?[defID] = nil
            if g.subgraphs?.isEmpty == true { g.subgraphs = nil }
        }
        return (g, inner.map(\.id))
    }

    /// The type flowing out of `node.output`, looking through reroutes (`nil` when unknown).
    public func resolvedType(_ node: String, _ output: String, schema: (GraphNode) -> NodeSchema?) -> SocketType? {
        var current = (node, output)
        for _ in 0..<64 {
            guard let n = self[current.0] else { return nil }
            if n.type == Self.rerouteType {
                guard case .link(let src, let out)? = n.inputs["in"] else { return nil }
                current = (src, out)
                continue
            }
            return schema(n)?.output(current.1)?.type
        }
        return nil
    }
}
