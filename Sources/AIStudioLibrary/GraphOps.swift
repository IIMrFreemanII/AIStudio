import CoreGraphics
import Foundation
import GraphRuntime

/// Edits a graph from a list of small operations (the graph helper's `edit_graph` tool): add, connect, disconnect,
/// set, remove, rename. Every operation is checked against the node schemas (types, inputs, choices) and the result
/// against cycles; a batch applies completely or not at all. New nodes get ids that don't clash and are placed next to
/// the nodes they connect to. Pure: the app passes schemas and node sizes.
public enum GraphOps {
    public enum Op: Equatable, Sendable {
        /// `id` is a wish (the actual id is made unique and later ops may use the wished one); `inputs` are literals
        /// or links.
        case add(id: String?, type: String, title: String?, inputs: [String: NodeInput])
        /// `output` nil: the source's only output that fits the input.
        case connect(from: String, output: String?, to: String, input: String)
        case disconnect(node: String, input: String)
        case set(node: String, input: String, value: JSONValue)
        case remove(node: String)
        case rename(node: String, title: String)
    }

    public struct Failure: Error, Equatable, CustomStringConvertible, Sendable {
        /// Index of the failing operation (nil: the batch as a whole, e.g. a cycle).
        public var index: Int?
        public var message: String
        public init(index: Int?, _ message: String) {
            self.index = index
            self.message = message
        }
        public var description: String { index.map { "operation \($0 + 1): \(message)" } ?? message }
    }

    public struct Result: Sendable {
        public var graph: Graph
        /// Wished id (or type) → actual id, for added nodes.
        public var added: [String: String]
        /// One line per change, for the card and the model.
        public var changes: [String]
    }

    /// Node schema of `type` (instances of subgraphs are resolved from the graph itself).
    public typealias Schemas = (String) -> NodeSchema?

    public static func apply(_ ops: [Op], to graph: Graph, schemas: Schemas,
                             size: (GraphNode) -> CGSize = { _ in CGSize(width: 250, height: 160) }) throws -> Result {
        var g = graph
        var alias: [String: String] = [:]
        var added: [String] = []
        var changes: [String] = []
        func resolve(_ id: String) -> String { g[id] != nil ? id : (alias[id] ?? id) }
        func schema(_ node: GraphNode) -> NodeSchema? {
            node.type == Graph.subgraphType ? g.subgraphSchema(node) : schemas(node.type)
        }
        func node(_ id: String, _ i: Int) throws -> (GraphNode, NodeSchema) {
            let rid = resolve(id)
            guard let n = g[rid] else {
                throw Failure(index: i, "no node \"\(id)\" (ids: \(g.nodes.map(\.id).joined(separator: ", ")))")
            }
            guard let s = schema(n) else { throw Failure(index: i, "node \"\(rid)\" has unknown type \(n.type)") }
            return (n, s)
        }
        func checkLink(_ src: String, _ out: String?, into spec: InputSpec, of target: String, _ i: Int) throws -> NodeInput {
            let (sn, ss) = try node(src, i)
            let output: OutputSpec
            if let out {
                guard let o = ss.output(out) else {
                    throw Failure(index: i, "\(sn.id) (\(sn.type)) has no output \"\(out)\"; its outputs: "
                                  + ss.outputs.map { "\($0.name): \($0.type)" }.joined(separator: ", "))
                }
                output = o
            } else {
                let fits = ss.outputs.filter { spec.type.accepts($0.type) }
                guard fits.count == 1 else {
                    throw Failure(index: i, fits.isEmpty
                                  ? "\(sn.id) (\(sn.type)) has no \(spec.type) output for \(target).\(spec.name)"
                                  : "say which output of \(sn.id): " + fits.map(\.name).joined(separator: ", "))
                }
                output = fits[0]
            }
            guard spec.type.accepts(output.type) else {
                throw Failure(index: i, "\(target).\(spec.name) takes \(spec.type), but \(sn.id).\(output.name) is \(output.type)")
            }
            return .link(node: sn.id, output: output.name)
        }
        func checkValue(_ v: JSONValue, _ spec: InputSpec, of target: String, _ i: Int) throws -> JSONValue {
            guard spec.type.isScalar else {
                throw Failure(index: i, "\(target).\(spec.name) is a \(spec.type) input: connect it to a node's output")
            }
            let value = coerce(v, to: spec.type)
            guard let value else {
                throw Failure(index: i, "\(target).\(spec.name) takes \(spec.type), not \(v)")
            }
            switch spec.widget {
            case .choice(let options)?:
                if case .string(let s) = value, !options.contains(s) {
                    throw Failure(index: i, "\(target).\(spec.name) must be one of: \(options.joined(separator: ", "))")
                }
            case .number(let lo, let hi, _)?:
                let d: Double? = { if case .int(let n) = value { return Double(n) }; if case .double(let x) = value { return x }; return nil }()
                if let d, let lo, d < lo { throw Failure(index: i, "\(target).\(spec.name) must be at least \(fmt(lo))") }
                if let d, let hi, d > hi { throw Failure(index: i, "\(target).\(spec.name) must be at most \(fmt(hi))") }
            default: break
            }
            return value
        }

        for (i, op) in ops.enumerated() {
            switch op {
            case .add(let wish, let type, let title, let inputs):
                guard type != Graph.subgraphType, let s = schemas(type) else {
                    throw Failure(index: i, "unknown node type \"\(type)\" (list_node_types shows them)")
                }
                let id = uniqueID(wish ?? type, in: g)
                var n = GraphNode(id: id, type: type, title: title)
                for spec in s.inputs where spec.type.isScalar { if let d = spec.defaultValue { n.inputs[spec.name] = .value(d) } }
                g.nodes.append(n)  // before its links, so the error messages can name it
                for (name, input) in inputs.sorted(by: { $0.key < $1.key }) {
                    guard let spec = s.input(name) else {
                        throw Failure(index: i, "\(type) has no input \"\(name)\"; its inputs: \(s.inputs.map(\.name).joined(separator: ", "))")
                    }
                    switch input {
                    case .value(let v): n.inputs[name] = .value(try checkValue(v, spec, of: id, i))
                    case .link(let src, let out): n.inputs[name] = try checkLink(src, out.isEmpty ? nil : out, into: spec, of: id, i)
                    }
                }
                g[id] = n
                if let wish, wish != id { alias[wish] = id }
                alias[type] = alias[type] ?? id
                added.append(id)
                changes.append("added \(id) (\(type))")
            case .connect(let from, let output, let to, let input):
                var (tn, ts) = try node(to, i)
                guard let spec = ts.input(input) else {
                    throw Failure(index: i, "\(tn.id) (\(tn.type)) has no input \"\(input)\"; its inputs: \(ts.inputs.map(\.name).joined(separator: ", "))")
                }
                let link = try checkLink(from, output, into: spec, of: tn.id, i)
                tn.inputs[input] = link
                g[tn.id] = tn
                if case .link(let s, let o) = link { changes.append("connected \(s).\(o) → \(tn.id).\(input)") }
            case .disconnect(let id, let input):
                var (n, _) = try node(id, i)
                guard case .link? = n.inputs[input] else { throw Failure(index: i, "\(n.id).\(input) isn't connected") }
                n.inputs[input] = nil
                g[n.id] = n
                changes.append("disconnected \(n.id).\(input)")
            case .set(let id, let input, let value):
                var (n, s) = try node(id, i)
                guard let spec = s.input(input) else {
                    throw Failure(index: i, "\(n.id) (\(n.type)) has no input \"\(input)\"; its inputs: \(s.inputs.map(\.name).joined(separator: ", "))")
                }
                n.inputs[input] = .value(try checkValue(value, spec, of: n.id, i))
                g[n.id] = n
                changes.append("set \(n.id).\(input) = \(short(value))")
            case .remove(let id):
                let (n, _) = try node(id, i)
                g.nodes.removeAll { $0.id == n.id }
                var cut: [String] = []
                for k in g.nodes.indices {
                    for (name, input) in g.nodes[k].inputs {
                        if case .link(let src, _) = input, src == n.id { g.nodes[k].inputs[name] = nil; cut.append("\(g.nodes[k].id).\(name)") }
                    }
                }
                added.removeAll { $0 == n.id }
                changes.append("removed \(n.id)" + (cut.isEmpty ? "" : " (disconnecting \(cut.sorted().joined(separator: ", ")))"))
            case .rename(let id, let title):
                var (n, _) = try node(id, i)
                n.title = title.isEmpty ? nil : title
                g[n.id] = n
                changes.append("renamed \(n.id) to “\(title)”")
            }
        }
        if let loop = cycle(in: g) { throw Failure(index: nil, "that would make a loop through \(loop.joined(separator: " → "))") }
        place(added, in: &g, size: size)
        var addedMap: [String: String] = [:]
        for (k, v) in alias where added.contains(v) { addedMap[k] = v }
        for id in added { addedMap[id] = id }
        return Result(graph: g, added: addedMap, changes: changes)
    }

    // MARK: values

    /// A model's value as the socket's type (numbers from JSON arrive as doubles, booleans sometimes as strings).
    public static func coerce(_ v: JSONValue, to type: SocketType) -> JSONValue? {
        switch (type, v) {
        case (.int, .int): return v
        case (.int, .double(let d)) where d == d.rounded(): return .int(Int(d))
        case (.int, .string(let s)): return Int(s).map(JSONValue.int)
        case (.float, .double): return v
        case (.float, .int(let n)): return .double(Double(n))
        case (.float, .string(let s)): return Double(s).map(JSONValue.double)
        case (.string, .string): return v
        case (.string, _): return .string(v.description)
        case (.bool, .bool): return v
        case (.bool, .string(let s)) where s == "true" || s == "false": return .bool(s == "true")
        case (.bool, .int(let n)) where n == 0 || n == 1: return .bool(n == 1)
        default: return nil
        }
    }

    static func fmt(_ d: Double) -> String { d == d.rounded() ? String(Int(d)) : String(d) }

    static func short(_ v: JSONValue) -> String {
        if case .string(let s) = v { return "“" + (s.count > 60 ? s.prefix(60) + "…" : s) + "”" }
        return v.description
    }

    // MARK: ids, cycles, placement

    /// `wish` made into an id (lowercase, no spaces) that isn't taken: `upscale`, `upscale2`…
    public static func uniqueID(_ wish: String, in g: Graph) -> String {
        var base = wish.lowercased().map { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "-" ? $0 : "_" }
            .reduce(into: "") { $0.append($1) }
        if base.isEmpty { base = "node" }
        var n = 1
        while g[n == 1 ? base : "\(base)\(n)"] != nil { n += 1 }
        return n == 1 ? base : "\(base)\(n)"
    }

    /// The nodes on a loop, if the links make one.
    static func cycle(in g: Graph) -> [String]? {
        var state: [String: Int] = [:]  // 1 visiting, 2 done
        var stack: [String] = []
        func visit(_ id: String) -> [String]? {
            if state[id] == 2 { return nil }
            if state[id] == 1 { return Array(stack[(stack.firstIndex(of: id) ?? 0)...]) + [id] }
            state[id] = 1
            stack.append(id)
            for case .link(let src, _) in (g[id]?.inputs ?? [:]).values.sorted(by: { "\($0)" < "\($1)" }) where g[src] != nil {
                if let c = visit(src) { return c }
            }
            stack.removeLast()
            state[id] = 2
            return nil
        }
        for n in g.nodes { if let c = visit(n.id) { return c.reversed() } }
        return nil
    }

    /// Puts each new node right of the nodes feeding it (or left of the one it feeds, or below everything), moved
    /// down until it overlaps nothing.
    static func place(_ ids: [String], in g: inout Graph, size: (GraphNode) -> CGSize) {
        let gap: CGFloat = 60
        var frames: [String: CGRect] = [:]
        for n in g.nodes where !ids.contains(n.id) {
            if let p = n.position { frames[n.id] = CGRect(origin: CGPoint(x: p.x, y: p.y), size: size(n)) }
        }
        let all = frames.values.reduce(CGRect.null) { $0.union($1) }
        var pending = ids.filter { g[$0] != nil }
        // Sources first, so a chain of new nodes is laid out left to right.
        var progress = true
        while !pending.isEmpty && progress {
            progress = false
            for id in pending {
                guard let n = g[id] else { continue }
                let sources = n.inputs.values.compactMap { input -> String? in
                    if case .link(let s, _) = input { return s }; return nil
                }
                if sources.contains(where: { pending.contains($0) && $0 != id }) { continue }
                let s = size(n)
                let feeding = sources.compactMap { frames[$0] }
                let fed = g.nodes.filter { m in m.inputs.values.contains { if case .link(let src, _) = $0 { return src == id }; return false } }
                    .compactMap { frames[$0.id] }
                var origin: CGPoint
                if !feeding.isEmpty {
                    origin = CGPoint(x: feeding.map(\.maxX).max()! + gap, y: feeding.map(\.minY).min()!)
                    // Inserted between nodes: what comes after it moves right to make room, so it sits in line.
                    if let firstFed = fed.map(\.minX).min(), case let shift = origin.x + s.width + gap - firstFed, shift > 0 {
                        for d in downstream(of: id, in: g) where frames[d] != nil {
                            frames[d]!.origin.x += shift
                            g[d]?.position?.x += shift
                        }
                    }
                } else if let f = fed.first {
                    origin = CGPoint(x: f.minX - gap - s.width, y: f.minY)
                } else if !all.isNull {
                    origin = CGPoint(x: all.minX, y: all.maxY + gap)
                } else {
                    origin = .zero
                }
                var r = CGRect(origin: origin, size: s)
                while let hit = frames.values.first(where: { $0.insetBy(dx: -8, dy: -8).intersects(r) }) {
                    r.origin.y = hit.maxY + 24
                }
                frames[id] = r
                g[id]?.position = GraphPoint(x: r.minX, y: r.minY)
                pending.removeAll { $0 == id }
                progress = true
            }
        }
    }

    /// Every node fed by `id`, directly or through others.
    static func downstream(of id: String, in g: Graph) -> Set<String> {
        var out = Set<String>()
        var queue = [id]
        while let cur = queue.popLast() {
            for n in g.nodes where !out.contains(n.id) && n.id != id {
                if n.inputs.values.contains(where: { if case .link(let src, _) = $0 { return src == cur }; return false }) {
                    out.insert(n.id)
                    queue.append(n.id)
                }
            }
        }
        return out
    }

    // MARK: describing graphs and types to a model

    /// The graph as compact text: one line per node with its literal inputs and links, and what's missing.
    public static func describe(_ g: Graph, schemas: Schemas) -> String {
        guard !g.nodes.isEmpty else { return "The graph is empty." }
        var lines: [String] = []
        for n in g.nodes where n.type != Graph.noteType {
            let s = n.type == Graph.subgraphType ? g.subgraphSchema(n) : schemas(n.type)
            var head = "\(n.id) [\(n.type)]"
            if let t = n.title, !t.isEmpty { head += " “\(t)”" }
            var parts: [String] = []
            let order = (s?.inputs.map(\.name) ?? []) + n.inputs.keys.sorted().filter { name in s?.input(name) == nil }
            for name in order {
                switch n.inputs[name] {
                case .link(let src, let out)?: parts.append("\(name) ← \(src).\(out)")
                case .value(let v)?:
                    if name == "def", n.type == Graph.subgraphType { continue }
                    parts.append("\(name)=\(short(v))")
                case nil:
                    if let spec = s?.input(name), !spec.optional, spec.defaultValue == nil { parts.append("\(name)=MISSING") }
                }
            }
            if let outs = s?.outputs, !outs.isEmpty { parts.append("outputs: " + outs.map { "\($0.name):\($0.type)" }.joined(separator: ",")) }
            lines.append(head + (parts.isEmpty ? "" : ": " + parts.joined(separator: "; ")))
        }
        return lines.joined(separator: "\n")
    }

    /// Node types by category, one line each: `Sampler — Sampler (in: model, positive…; out: latent)`.
    public static func typeList(_ schemas: [NodeSchema], category: String? = nil) -> String {
        let shown = schemas.filter { category == nil || $0.category.lowercased() == category!.lowercased() }
        let byCat = Dictionary(grouping: shown, by: \.category)
        return byCat.keys.sorted().map { cat in
            "# \(cat)\n" + byCat[cat]!.sorted { $0.type < $1.type }.map { s in
                let ins = s.inputs.filter { !$0.type.isScalar }.map(\.name)
                let outs = s.outputs.map(\.name)
                return "\(s.type)" + (s.displayName != s.type ? " (\(s.displayName))" : "")
                    + " — in: \(ins.isEmpty ? "-" : ins.joined(separator: ", ")); out: \(outs.isEmpty ? "-" : outs.joined(separator: ", "))"
                    + (s.isOutput ? "; output node" : "")
            }.joined(separator: "\n")
        }.joined(separator: "\n\n")
    }

    /// One type in full: every input with its type, default and allowed values, and the outputs.
    public static func describeType(_ s: NodeSchema) -> String {
        var lines = ["\(s.type) (\(s.displayName)), category \(s.category)\(s.isOutput ? ", output node" : "")", "inputs:"]
        for i in s.inputs {
            var l = "- \(i.name): \(i.type)"
            if i.optional { l += ", optional" }
            if let d = i.defaultValue { l += ", default \(short(d))" }
            switch i.widget {
            case .choice(let o)?: l += ", one of: " + o.joined(separator: ", ")
            case .number(let lo, let hi, _)?:
                if let lo { l += ", min \(fmt(lo))" }
                if let hi { l += ", max \(fmt(hi))" }
            case .text(let multi)?: if multi { l += ", text" }
            case .seed?: l += ", seed"
            case .filePath?: l += ", file path"
            case .library(let kind)?: l += ", id of an installed \(kind)"
            default: break
            }
            if !i.type.isScalar { l += " (connect an output)" }
            lines.append(l)
        }
        lines.append("outputs:")
        lines += s.outputs.map { "- \($0.name): \($0.type)" }
        if s.outputs.isEmpty { lines.append("- none") }
        return lines.joined(separator: "\n")
    }
}
