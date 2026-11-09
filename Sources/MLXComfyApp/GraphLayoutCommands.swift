import AppKit
import Foundation
import GraphRuntime
import MLXComfyLibrary

/// Arrange, align, collapse and copy/paste for the editing level.
extension AppModel {
    // MARK: geometry of the editing level

    func frame(of node: GraphNode) -> CGRect? {
        schema(for: node).map { NodeLayout.frame(node, $0) }
    }

    /// The links of `level` as layout edges; model links are weak (they're hidden by default and fan out everywhere).
    func layoutEdges(_ level: Graph) -> [GraphArranger.Edge] {
        var out: [GraphArranger.Edge] = []
        for n in level.nodes {
            for (name, input) in n.inputs.sorted(by: { $0.key < $1.key }) {
                guard case .link(let src, let output) = input, level[src] != nil else { continue }
                let type = schema(for: n)?.input(name)?.type
                let weak = type == .model || (type == .any && resolvedType(src, output) == .model)
                out.append(GraphArranger.Edge(src, n.id, weak: weak))
            }
        }
        return out
    }

    // MARK: arrange

    /// Lays the editing level (or just the selection) out in left-to-right columns, as one undo step.
    func arrange(selectionOnly: Bool, undoManager: UndoManager?) {
        let level = editorGraph
        let ids = selectionOnly ? selection : Set(level.nodes.map(\.id))
        var sizes: [String: CGSize] = [:]
        var initial: [String: CGPoint] = [:]
        for n in level.nodes where ids.contains(n.id) {
            guard let f = frame(of: n) else { continue }
            sizes[n.id] = f.size
            initial[n.id] = f.origin
        }
        guard sizes.count > 1 else { return }
        let edges = layoutEdges(level).filter { sizes[$0.source] != nil && sizes[$0.target] != nil }
        var groups: [String: Set<String>] = [:]
        if !selectionOnly { for g in level.groups ?? [] { groups[g.id] = nodes(in: g.id) } }
        var originBox = GraphArranger.bounds(initial, sizes)
        for g in level.groups ?? [] where groups[g.id]?.isEmpty == false {
            originBox = originBox.union(CGRect(x: g.x, y: g.y, width: g.width, height: g.height))
        }
        let r = GraphArranger.arrange(sizes: sizes, edges: edges, groups: groups, initial: initial,
                                      origin: originBox.origin)
        mutateEditor(selectionOnly ? "Arrange Selection" : "Arrange Graph", undoManager: undoManager) { g in
            for (id, p) in r.positions { g[id]?.position = GraphPoint(x: p.x, y: p.y) }
            for (gid, f) in r.groupFrames {
                guard let i = g.groups?.firstIndex(where: { $0.id == gid }) else { continue }
                g.groups![i].x = f.minX
                g.groups![i].y = f.minY
                g.groups![i].width = f.width
                g.groups![i].height = f.height
            }
        }
    }

    /// Places the nodes of a chain that was just spliced into the top-level graph: laid out left to right as a block,
    /// put in free space next to the nodes that feed it, and returned as the frame for its group. Output nodes fed
    /// only by the chain (e.g. Save Image after a hi-res fix) move to the right of the block.
    func placeChain(_ chain: [String], in g: inout Graph) -> CGRect? {
        let ids = Set(chain).filter { g[$0] != nil }
        guard !ids.isEmpty else { return nil }
        let pad: CGFloat = 24, header: CGFloat = 30
        var sizes: [String: CGSize] = [:]
        for id in ids { if let n = g[id], let f = frame(of: n) { sizes[id] = f.size } }
        let edges = layoutEdges(g).filter { ids.contains($0.source) && ids.contains($0.target) }
        let inner = GraphArranger.arrange(sizes: sizes, edges: edges).positions
        let box = GraphArranger.bounds(inner, sizes)
        let block = CGSize(width: box.width + 2 * pad, height: box.height + 2 * pad + header)

        let all = layoutEdges(g)
        let producers = Set(all.filter { !$0.weak && ids.contains($0.target) && !ids.contains($0.source) }.map(\.source))
        let consumers = Set(all.filter { ids.contains($0.source) && !ids.contains($0.target) }.map(\.target))
        // Sinks fed only by the chain follow it.
        let followers = consumers.filter { c in
            !all.contains { $0.source == c } && !all.contains { $0.target == c && !$0.weak && !ids.contains($0.source) }
        }
        func frames(_ s: Set<String>) -> [CGRect] { s.compactMap { g[$0].flatMap(frame(of:)) } }
        let pf = frames(producers), cf = frames(consumers.subtracting(followers))
        let near: CGPoint
        if let maxX = pf.map(\.maxX).max() {
            near = CGPoint(x: maxX + 80, y: pf.map(\.minY).sorted()[pf.count / 2] - header - pad)
        } else if let minX = cf.map(\.minX).min() {
            near = CGPoint(x: minX - block.width - 80, y: cf.map(\.minY).sorted()[cf.count / 2] - header - pad)
        } else {
            near = CGPoint(x: 0, y: (g.nodes.compactMap(frame(of:)).map(\.maxY).max() ?? 0) + 80)
        }
        var avoid = g.nodes.filter { !ids.contains($0.id) && !followers.contains($0.id) }.compactMap(frame(of:))
        avoid += (g.groups ?? []).map { CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height) }
        let spot = GraphArranger.freeSpot(size: block, near: Alignment.snap(near), avoiding: avoid)
        let frameRect = CGRect(origin: spot, size: block)
        for (id, p) in inner {
            g[id]?.position = GraphPoint(x: spot.x + pad + p.x - box.minX, y: spot.y + header + pad + p.y - box.minY)
        }
        avoid.append(frameRect)
        for id in followers.sorted() {
            guard let n = g[id], let f = frame(of: n) else { continue }
            let p = GraphArranger.freeSpot(size: f.size, near: CGPoint(x: frameRect.maxX + 80, y: frameRect.minY + header + pad),
                                           avoiding: avoid)
            g[id]?.position = GraphPoint(x: p.x, y: p.y)
            avoid.append(CGRect(origin: p, size: f.size))
        }
        return frameRect
    }

    // MARK: align

    func alignSelection(_ mode: AlignMode, undoManager: UndoManager?) {
        var frames: [String: CGRect] = [:]
        for n in editorGraph.nodes where selection.contains(n.id) { frames[n.id] = frame(of: n) }
        let moves = Alignment.apply(mode, to: frames)
        guard !moves.isEmpty else { return }
        mutateEditor(mode.title, undoManager: undoManager) { g in
            for (id, p) in moves { g[id]?.position = GraphPoint(x: p.x, y: p.y) }
        }
    }

    // MARK: collapse

    /// Collapses the given nodes, or expands them all when every one is already collapsed.
    func toggleCollapsed(_ ids: Set<String>, undoManager: UndoManager?) {
        let targets = editorGraph.nodes.filter {
            ids.contains($0.id) && !NodeLayout.isReroute($0) && !NodeLayout.isNote($0)
        }
        guard !targets.isEmpty else { return }
        let collapse = targets.contains { !$0.isCollapsed }
        mutateEditor(collapse ? "Collapse Node" : "Expand Node", undoManager: undoManager) { g in
            for n in targets { g[n.id]?.collapsed = collapse ? true : nil }
        }
    }

    var selectionAllCollapsed: Bool {
        let s = editorGraph.nodes.filter { selection.contains($0.id) && !NodeLayout.isReroute($0) && !NodeLayout.isNote($0) }
        return !s.isEmpty && s.allSatisfy(\.isCollapsed)
    }

    // MARK: wires

    /// Removes the link into `node.input`.
    func disconnect(node: String, input: String, undoManager: UndoManager?) {
        guard case .link? = editorGraph[node]?.inputs[input] else { return }
        mutateEditor("Disconnect", undoManager: undoManager) { g in g[node]?.inputs[input] = nil }
    }

    // MARK: clipboard

    /// The selection (and groups wholly inside it) as clipboard contents.
    func clipboardForSelection() -> NodeClipboard? {
        guard !selection.isEmpty else { return nil }
        let level = editorGraph
        let groupIDs = Set((level.groups ?? []).filter { g in
            let inside = nodes(in: g.id)
            return !inside.isEmpty && inside.isSubset(of: selection)
        }.map(\.id))
        return NodeClipboard.copy(selection, groupIDs: groupIDs, from: level, definitions: graph.subgraphs ?? [:])
    }

    func copySelection() {
        guard let clip = clipboardForSelection(), let data = try? JSONEncoder().encode(clip) else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setData(data, forType: NSPasteboard.PasteboardType(NodeClipboard.pasteboardType))
        pb.setString(String(decoding: data, as: UTF8.self), forType: .string)
    }

    func cutSelection(undoManager: UndoManager?) {
        copySelection()
        deleteSelection(undoManager: undoManager)
    }

    static func clipboardOnPasteboard() -> NodeClipboard? {
        let pb = NSPasteboard.general
        let data = pb.data(forType: NSPasteboard.PasteboardType(NodeClipboard.pasteboardType))
            ?? pb.string(forType: .string).map { Data($0.utf8) }
        return data.flatMap { try? JSONDecoder().decode(NodeClipboard.self, from: $0) }
    }

    var canPaste: Bool { Self.clipboardOnPasteboard() != nil }

    /// Pastes the pasteboard's nodes with their top-left at `point` (default: 40 pt below and right of where they were
    /// copied from).
    func paste(at point: GraphPoint? = nil, keepExternalLinks: Bool = false, undoManager: UndoManager?) {
        guard let clip = Self.clipboardOnPasteboard() else { return }
        insert(clip, at: point ?? GraphPoint(x: clip.origin.x + 40, y: clip.origin.y + 40),
               keepExternalLinks: keepExternalLinks, actionName: "Paste", undoManager: undoManager)
    }

    func insert(_ clip: NodeClipboard, at point: GraphPoint, keepExternalLinks: Bool, actionName: String,
                undoManager: UndoManager?) {
        let p = clip.pasted(into: editorGraph, definitions: graph.subgraphs ?? [:], at: point,
                            keepExternalLinks: keepExternalLinks)
        guard !p.nodes.isEmpty else { return }
        let defID = editingDefID
        mutate(actionName, undoManager: undoManager) { g in
            if !p.subgraphs.isEmpty { g.subgraphs = (g.subgraphs ?? [:]).merging(p.subgraphs) { a, _ in a } }
            func add(_ level: inout Graph) {
                level.nodes += p.nodes
                if !p.groups.isEmpty { level.groups = (level.groups ?? []) + p.groups }
            }
            if let defID, var inner = g.subgraphs?[defID]?.graph {
                add(&inner)
                g.subgraphs?[defID]?.graph = inner
            } else {
                add(&g)
            }
        }
        selection = Set(p.nodeIDs)
        selectedGroup = nil
    }

    // MARK: adding a node from a dropped wire

    /// A wire released on empty canvas: the palette offers nodes that can take it (or feed it), and connects the pick.
    struct PendingWire: Equatable {
        /// The node and socket the wire hangs from.
        var node: String
        var socket: String
        var type: SocketType
        /// True when it comes from an output (the new node's input connects), false from an input.
        var fromOutput: Bool
        var point: GraphPoint
    }

    /// The first socket of `schema` a pending wire can connect to.
    static func compatibleSocket(_ schema: NodeSchema, for wire: PendingWire) -> String? {
        if wire.fromOutput {
            return (schema.inputs.first { !$0.type.isScalar && $0.type.accepts(wire.type) && $0.type != .any }
                ?? schema.inputs.first { $0.type.accepts(wire.type) && ($0.type.isScalar ? $0.type == wire.type : true) })?.name
        }
        return schema.outputs.first { wire.type.accepts($0.type) }?.name
    }

    /// Adds `type` where the wire was dropped and connects it, as one undo step.
    func addNode(type: String, connecting wire: PendingWire, undoManager: UndoManager?) {
        guard let schema = registry.schema(type), let socket = Self.compatibleSocket(schema, for: wire) else { return }
        let start = graph
        let w = NodeLayout.width
        // From an output the new node's left edge goes at the drop point; from an input, its right edge does.
        let at = wire.fromOutput ? GraphPoint(x: wire.point.x, y: wire.point.y - NodeLayout.header - NodeLayout.row / 2)
            : GraphPoint(x: wire.point.x - w, y: wire.point.y - NodeLayout.header - NodeLayout.row / 2)
        addNode(type: type, at: at, undoManager: nil)
        guard let id = selection.first, editorGraph[id]?.type == type else { graph = start; return }
        var level = editorGraph
        if wire.fromOutput {
            level[id]?.inputs[socket] = .link(node: wire.node, output: wire.socket)
        } else {
            level[wire.node]?.inputs[wire.socket] = .link(node: id, output: socket)
        }
        editorGraph = level
        commitLiveEdit(from: start, actionName: "Add and Connect \(schema.displayName)", undoManager: undoManager)
        selection = [id]
    }
}
