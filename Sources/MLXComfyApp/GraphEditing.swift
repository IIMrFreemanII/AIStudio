import AppKit
import Foundation
import GraphRuntime

/// Editor structure: the level being edited (top-level graph or a subgraph), groups, subgraphs and reroutes.
/// Every edit goes through `mutate`, so it is one undo step on the whole document.
extension AppModel {
    // MARK: editing level

    /// The graph at `path` (a chain of subgraph instance ids), or nil if it no longer resolves.
    func levelGraph(_ path: [String]) -> Graph? {
        var level = graph
        for id in path {
            guard let node = level[id], let (_, def) = graph.subgraphDef(for: node) else { return nil }
            level = def.graph
        }
        return level
    }

    /// The subgraph definition id being edited (nil at the top level).
    var editingDefID: String? {
        guard !editingPath.isEmpty, let parent = levelGraph(Array(editingPath.dropLast())),
              let inst = parent[editingPath.last!] else { return nil }
        return graph.subgraphDef(for: inst)?.id
    }

    /// The graph the editor shows and edits. Setting it writes back into the document (no undo by itself).
    var editorGraph: Graph {
        get { levelGraph(editingPath) ?? graph }
        set {
            if let defID = editingDefID { graph.subgraphs?[defID]?.graph = newValue } else { graph = newValue }
        }
    }

    /// Like `mutate`, applied to the editing level.
    func mutateEditor(_ actionName: String, undoManager: UndoManager?, _ body: (inout Graph) -> Void) {
        guard let defID = editingDefID else { mutate(actionName, undoManager: undoManager, body); return }
        mutate(actionName, undoManager: undoManager) { g in
            guard var inner = g.subgraphs?[defID]?.graph else { return }
            body(&inner)
            g.subgraphs?[defID]?.graph = inner
        }
    }

    /// Prefix of run-state keys for nodes at the editing level (flattened ids are `outer/inner`).
    var nodeStatePrefix: String { editingPath.isEmpty ? "" : editingPath.joined(separator: "/") + "/" }

    /// Run state for a flattened node id, also shown on every subgraph instance that contains it.
    func setState(_ state: NodeRunState, _ id: String, in doc: WorkflowDocument? = nil) {
        let doc = doc ?? active
        doc.nodeStates[id] = state
        var parts = id.split(separator: "/").map(String.init)
        while parts.count > 1 {
            parts.removeLast()
            doc.nodeStates[parts.joined(separator: "/")] = state
        }
    }

    // MARK: literals at the editing level

    func editorLiteral(_ nodeID: String, _ input: String) -> JSONValue? {
        guard let node = editorGraph[nodeID] else { return nil }
        if case .value(let v)? = node.inputs[input] { return v }
        return schema(for: node)?.input(input)?.defaultValue
    }

    func setEditorLiteral(_ value: JSONValue, node: String, input: String, undoManager: UndoManager?) {
        guard editorLiteral(node, input) != value else { return }
        mutateEditor("Change \(input)", undoManager: undoManager) { g in try? g.setValue(value, node: node, input: input) }
    }

    func setTitle(_ title: String, node: String, undoManager: UndoManager?) {
        mutateEditor("Rename", undoManager: undoManager) { g in
            g[node]?.title = title.trimmingCharacters(in: .whitespaces).isEmpty ? nil : title
        }
    }

    // MARK: subgraphs

    func enterSubgraph(_ id: String) {
        guard let n = editorGraph[id], n.type == Graph.subgraphType, graph.subgraphDef(for: n) != nil else { return }
        editingPath.append(id)
        selection = []
        selectedGroup = nil
    }

    /// Leaves the current subgraph (or goes back to `level` entries deep), selecting the instance left.
    func exitSubgraph(to level: Int? = nil) {
        guard !editingPath.isEmpty else { return }
        let keep = level ?? editingPath.count - 1
        let left = editingPath[keep]
        editingPath = Array(editingPath.prefix(keep))
        selection = [left]
        selectedGroup = nil
    }

    /// Breadcrumb titles for the editing path.
    var editingTitles: [String] {
        var titles: [String] = []
        var level = graph
        for id in editingPath {
            guard let n = level[id], let (_, def) = graph.subgraphDef(for: n) else { break }
            titles.append(n.title ?? def.name)
            level = def.graph
        }
        return titles
    }

    /// Port names bound to each inner input of the subgraph being edited: node → input → port.
    var boundInputs: [String: [String: String]] {
        guard let defID = editingDefID, let def = graph.subgraphs?[defID] else { return [:] }
        var out: [String: [String: String]] = [:]
        for p in def.inputs { for t in p.targets { out[t.node, default: [:]][t.socket] = p.name } }
        return out
    }

    /// Inner outputs exposed by the subgraph being edited: node → output → port.
    var exposedOutputs: [String: [String: String]] {
        guard let defID = editingDefID, let def = graph.subgraphs?[defID] else { return [:] }
        var out: [String: [String: String]] = [:]
        for p in def.outputs { if let t = p.targets.first { out[t.node, default: [:]][t.socket] = p.name } }
        return out
    }

    func collapseSelection(undoManager: UndoManager?) {
        let ids = selection
        guard !ids.isEmpty else { return }
        let level = editorGraph
        let defs = graph.subgraphs
        // Collapse works on the level graph, but definitions live at the top level.
        var working = level
        working.subgraphs = defs
        let typeOf: (GraphNode, String, Bool) -> SocketType? = { [self] node, socket, isInput in
            if node.type == Graph.rerouteType, !isInput {
                return working.resolvedType(node.id, socket) { self.schema(for: $0) } ?? .any
            }
            let s = node.type == Graph.subgraphType ? working.subgraphSchema(node) : registry.schema(node.type)
            return isInput ? s?.input(socket)?.type : s?.output(socket)?.type
        }
        let n = (defs?.count ?? 0) + 1
        guard let (collapsed, instance) = working.collapsing(ids, name: "Subgraph \(n)", typeOf: typeOf) else { return }
        let newDefs = collapsed.subgraphs
        var levelResult = collapsed
        levelResult.subgraphs = editingPath.isEmpty ? newDefs : nil
        mutate("Collapse to Subgraph", undoManager: undoManager) { g in
            if editingPath.isEmpty {
                g = levelResult
            } else if let defID = editingDefID {
                g.subgraphs = newDefs
                g.subgraphs?[defID]?.graph = levelResult
            }
        }
        selection = [instance]
    }

    func expandSubgraph(_ id: String, undoManager: UndoManager?) {
        var working = editorGraph
        working.subgraphs = graph.subgraphs
        guard let (expanded, inner) = working.expanding(id) else { return }
        let newDefs = expanded.subgraphs
        var levelResult = expanded
        levelResult.subgraphs = editingPath.isEmpty ? newDefs : nil
        mutate("Expand Subgraph", undoManager: undoManager) { g in
            if editingPath.isEmpty {
                g = levelResult
            } else if let defID = editingDefID {
                // Inside a subgraph `expanding` can't see uses elsewhere, so keep every definition.
                g.subgraphs?[defID]?.graph = levelResult
            }
        }
        selection = Set(inner)
    }

    // MARK: groups

    static let groupColors = ["blue", "green", "yellow", "red", "purple", "gray"]

    /// A group around the selected nodes.
    func groupSelection(undoManager: UndoManager?) {
        let frames = editorGraph.nodes.filter { selection.contains($0.id) }.compactMap { n in
            schema(for: n).map { NodeLayout.frame(n, $0) }
        }
        guard let first = frames.first else { return }
        let box = frames.dropFirst().reduce(first) { $0.union($1) }.insetBy(dx: -24, dy: -24)
        var k = (editorGraph.groups?.count ?? 0) + 1
        while (editorGraph.groups ?? []).contains(where: { $0.id == "group\(k)" }) { k += 1 }
        let group = GraphGroup(id: "group\(k)", title: "Group \(k)", x: box.minX, y: box.minY - 30,
                               width: box.width, height: box.height + 30,
                               color: Self.groupColors[(k - 1) % Self.groupColors.count])
        mutateEditor("Group", undoManager: undoManager) { g in g.groups = (g.groups ?? []) + [group] }
        selectedGroup = group.id
        selection = []
    }

    func updateGroup(_ id: String, actionName: String, undoManager: UndoManager?, _ body: (inout GraphGroup) -> Void) {
        mutateEditor(actionName, undoManager: undoManager) { g in
            guard let i = g.groups?.firstIndex(where: { $0.id == id }) else { return }
            body(&g.groups![i])
        }
    }

    func deleteGroup(_ id: String, withNodes: Bool, undoManager: UndoManager?) {
        let inside = withNodes ? nodes(in: id) : []
        mutateEditor(withNodes ? "Delete Group and Nodes" : "Delete Group", undoManager: undoManager) { g in
            g.groups?.removeAll { $0.id == id }
            if g.groups?.isEmpty == true { g.groups = nil }
            g.nodes.removeAll { inside.contains($0.id) }
            for i in g.nodes.indices {
                for (k, v) in g.nodes[i].inputs {
                    if case .link(let src, _) = v, inside.contains(src) { g.nodes[i].inputs[k] = nil }
                }
            }
        }
        if selectedGroup == id { selectedGroup = nil }
    }

    /// Nodes whose top-left corner lies inside the group (they move with it).
    func nodes(in groupID: String) -> Set<String> {
        guard let grp = editorGraph.groups?.first(where: { $0.id == groupID }) else { return [] }
        return Set(editorGraph.nodes.filter { n in
            grp.contains(x: n.position?.x ?? 0, y: n.position?.y ?? 0)
        }.map(\.id))
    }

    // MARK: reroutes

    /// Splits the link into `node.input` with a Reroute at `point`.
    func insertReroute(into node: String, input: String, at point: GraphPoint, undoManager: UndoManager?) {
        guard let link = editorGraph[node]?.inputs[input], case .link = link else { return }
        var k = 1
        while editorGraph["reroute\(k)"] != nil { k += 1 }
        let id = "reroute\(k)"
        mutateEditor("Add Reroute", undoManager: undoManager) { g in
            g.nodes.append(GraphNode(id: id, type: Graph.rerouteType, inputs: ["in": link],
                                     position: GraphPoint(x: point.x - NodeLayout.rerouteSize.width / 2,
                                                          y: point.y - NodeLayout.rerouteSize.height / 2)))
            g[node]?.inputs[input] = .link(node: id, output: "out")
        }
        selection = [id]
    }

    /// The type a link into `node` carries, looking through reroutes.
    func resolvedType(_ node: String, _ output: String) -> SocketType? {
        var level = editorGraph
        level.subgraphs = graph.subgraphs
        return level.resolvedType(node, output) { self.schema(for: $0) }
    }
}

extension AppModel {
    /// "Subgraph 1 › Sampler" for a flattened id like `subgraph/sampler`.
    func displayName(flatID: String, in doc: WorkflowDocument? = nil) -> String {
        let parts = flatID.split(separator: "/").map(String.init)
        let root = (doc ?? active).graph
        var level = root
        var names: [String] = []
        for (i, id) in parts.enumerated() {
            guard let n = level[id] else { return flatID }
            let s = n.type == Graph.subgraphType ? root.subgraphSchema(n) : registry.schema(n.type)
            names.append(n.title ?? s?.displayName ?? id)
            if i < parts.count - 1 {
                guard let (_, def) = root.subgraphDef(for: n) else { return flatID }
                level = def.graph
            }
        }
        return names.joined(separator: " › ")
    }
}
