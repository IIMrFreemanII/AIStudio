import Foundation
import GraphRuntime

/// Copied nodes: the nodes themselves (links between them intact), the groups wholly inside the selection, and the
/// subgraph definitions they use. Pasting gives everything fresh ids where needed.
public struct NodeClipboard: Codable, Hashable, Sendable {
    /// The private pasteboard type the editor writes (JSON of this struct).
    public static let pasteboardType = "com.mlxcomfyui.nodes"

    public var nodes: [GraphNode]
    public var groups: [GraphGroup]
    public var subgraphs: [String: SubgraphDef]
    /// Top-left of the copied nodes; pasting moves this corner to the paste point.
    public var origin: GraphPoint

    public init(nodes: [GraphNode], groups: [GraphGroup] = [], subgraphs: [String: SubgraphDef] = [:], origin: GraphPoint) {
        self.nodes = nodes
        self.groups = groups
        self.subgraphs = subgraphs
        self.origin = origin
    }

    /// Copies `ids` (and `groupIDs`) from `level`. `definitions` are the document's subgraph definitions.
    public static func copy(_ ids: Set<String>, groupIDs: Set<String> = [], from level: Graph,
                            definitions: [String: SubgraphDef]) -> NodeClipboard {
        let nodes = level.nodes.filter { ids.contains($0.id) }
        let groups = (level.groups ?? []).filter { groupIDs.contains($0.id) }
        var defs: [String: SubgraphDef] = [:]
        func collect(_ ns: [GraphNode]) {
            for n in ns {
                guard let id = defID(n), defs[id] == nil, let def = definitions[id] else { continue }
                defs[id] = def
                collect(def.graph.nodes)
            }
        }
        collect(nodes)
        let xs = nodes.map { $0.position?.x ?? 0 } + groups.map(\.x)
        let ys = nodes.map { $0.position?.y ?? 0 } + groups.map(\.y)
        return NodeClipboard(nodes: nodes, groups: groups, subgraphs: defs,
                             origin: GraphPoint(x: xs.min() ?? 0, y: ys.min() ?? 0))
    }

    public struct Pasted: Sendable {
        public var nodes: [GraphNode]
        public var groups: [GraphGroup]
        /// Definitions to add to the document (already renamed when an id was taken by a different definition).
        public var subgraphs: [String: SubgraphDef]
        public var nodeIDs: [String] { nodes.map(\.id) }
    }

    /// The nodes as they would be pasted into `level` with their top-left at `point`. Links to nodes outside the
    /// clipboard are dropped, or kept when `keepExternalLinks` and the source exists in `level`.
    public func pasted(into level: Graph, definitions: [String: SubgraphDef], at point: GraphPoint,
                       keepExternalLinks: Bool) -> Pasted {
        // Subgraph definitions: reuse identical ones, rename clashing ones.
        var defRename: [String: String] = [:]
        var takenDefs = Set(definitions.keys)
        for id in subgraphs.keys.sorted() {
            if let existing = definitions[id], existing == subgraphs[id] { continue }
            if definitions[id] == nil { takenDefs.insert(id); continue }
            let new = Self.unique(id, taken: takenDefs)
            takenDefs.insert(new)
            defRename[id] = new
        }
        func renameDefs(_ n: inout GraphNode) {
            if let id = Self.defID(n), let new = defRename[id] { n.inputs["def"] = .value(.string(new)) }
        }
        var newDefs: [String: SubgraphDef] = [:]
        for (id, var def) in subgraphs where definitions[id] != def {
            for i in def.graph.nodes.indices { renameDefs(&def.graph.nodes[i]) }
            newDefs[defRename[id] ?? id] = def
        }

        // Node ids: keep them when free, else `id_2`, `id_3`…
        var taken = Set(level.nodes.map(\.id))
        var mapping: [String: String] = [:]
        for n in nodes {
            let new = taken.contains(n.id) ? Self.unique(n.id, taken: taken) : n.id
            taken.insert(new)
            mapping[n.id] = new
        }
        let dx = point.x - origin.x, dy = point.y - origin.y
        var out: [GraphNode] = []
        for n in nodes {
            var c = n
            c.id = mapping[n.id]!
            c.position = GraphPoint(x: (n.position?.x ?? 0) + dx, y: (n.position?.y ?? 0) + dy)
            for (name, input) in c.inputs {
                guard case .link(let src, let output) = input else { continue }
                if let m = mapping[src] {
                    c.inputs[name] = .link(node: m, output: output)
                } else if !(keepExternalLinks && level[src] != nil) {
                    c.inputs[name] = nil
                }
            }
            renameDefs(&c)
            out.append(c)
        }
        var takenGroups = Set((level.groups ?? []).map(\.id))
        var groupsOut: [GraphGroup] = []
        for g in groups {
            var c = g
            if takenGroups.contains(c.id) { c.id = Self.unique(c.id, taken: takenGroups) }
            takenGroups.insert(c.id)
            c.x += dx
            c.y += dy
            groupsOut.append(c)
        }
        return Pasted(nodes: out, groups: groupsOut, subgraphs: newDefs)
    }

    static func defID(_ n: GraphNode) -> String? {
        guard n.type == Graph.subgraphType, case .value(.string(let id))? = n.inputs["def"] else { return nil }
        return id
    }

    static func unique(_ base: String, taken: Set<String>) -> String {
        var n = 2
        while taken.contains("\(base)_\(n)") { n += 1 }
        return "\(base)_\(n)"
    }
}
