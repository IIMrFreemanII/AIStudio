import AppKit
import GraphRuntime
import SwiftUI
import UniformTypeIdentifiers

/// A link being dragged from an output socket.
struct WireDrag {
    var fromNode: String
    var fromOutput: String
    var type: SocketType
    var end: CGPoint  // view coordinates
}

/// The node editor for the current editing level (the top-level graph, or a subgraph entered with a double-click).
struct GraphEditorView: View {
    @Environment(AppModel.self) private var model
    /// The tab shown (the view is rebuilt per tab, and keeps the tab's zoom and pan in it).
    let doc: WorkflowDocument
    private var undoManager: UndoManager? { model.activeUndoManager }

    @State private var zoom: CGFloat = 0.9
    @State private var pan = CGSize(width: 40, height: 60)
    @State private var panStart: CGSize?
    @State private var zoomStart: CGFloat?
    @State private var dragStartPositions: [String: GraphPoint]?
    @State private var groupStart: GraphGroup?
    @State private var editStartGraph: Graph?
    @State private var wire: WireDrag?
    @State private var marqueeStart: CGPoint?
    @State private var marquee: CGRect?
    @State private var hoverLocation: CGPoint?
    @State private var viewSize: CGSize = .zero
    @State private var scrollMonitor: Any?
    @State private var didInitialFit = false
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var model = model
        let level = model.editorGraph
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                GridBackground(zoom: zoom, pan: pan)
                    .contentShape(Rectangle())
                    .gesture(backgroundDrag)
                    .gesture(SpatialTapGesture(count: 2, coordinateSpace: .named("canvas")).onEnded { v in
                        model.paletteLocation = toGraph(v.location)
                        model.showPalette = true
                    }.exclusively(before: SpatialTapGesture(coordinateSpace: .named("canvas")).onEnded { v in
                        backgroundTap(at: v.location)
                    }))

                    .contextMenu {
                        Button("Add Node…") { model.paletteLocation = nil; model.showPalette = true }
                        Menu("Add to Workflow") { AddToWorkflowItems(model: model) }
                        Divider()
                        Button("Fit Graph to Window") { fitToContent() }
                    }

                ForEach(level.groups ?? []) { group in groupView(group) }

                Canvas { ctx, _ in drawWires(&ctx, level) }
                    .allowsHitTesting(false)

                ForEach(level.nodes) { node in
                    nodeView(node, level).offset(x: toView(NodeLayout.origin(node)).x, y: toView(NodeLayout.origin(node)).y)
                }

                if let marquee {
                    Rectangle().fill(Color.accentColor.opacity(0.12))
                        .overlay(Rectangle().strokeBorder(Color.accentColor, lineWidth: 1))
                        .frame(width: marquee.width, height: marquee.height)
                        .offset(x: marquee.minX, y: marquee.minY)
                        .allowsHitTesting(false)
                }
            }
            .coordinateSpace(name: "canvas")
            .clipped()
            .onDrop(of: [.fileURL, .plainText], isTargeted: $dropTargeted) { providers, loc in
                handleDrop(providers, at: loc)
            }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                        .padding(6).allowsHitTesting(false)
                }
            }
            .overlay(alignment: .bottomTrailing) { zoomControls.padding(12) }
            .overlay(alignment: .bottom) { RunStatusPill().padding(.bottom, 16) }
            .overlay(alignment: .topLeading) { topBar.padding(12) }
            .onAppear {
                viewSize = geo.size
                if let v = doc.viewport {
                    zoom = v.zoom
                    pan = v.pan
                    didInitialFit = true
                }
                fitOnceIfSized()
                installScrollMonitor()
            }
            .onChange(of: zoom) { _, z in doc.viewport = (z, pan) }
            .onChange(of: pan) { _, p in doc.viewport = (zoom, p) }
            .onChange(of: model.assetAddRequest) { _, url in
                guard let url else { return }
                model.addLoadNode(for: url, at: toGraph(CGPoint(x: viewSize.width / 2 - 120, y: viewSize.height / 2 - 60)))
                model.assetAddRequest = nil
            }
            .onChange(of: model.nodeAddRequest) { _, req in
                guard let req else { return }
                model.addNode(type: req.type, at: toGraph(CGPoint(x: viewSize.width / 2 - 120, y: viewSize.height / 2 - 60)),
                              undoManager: undoManager)
                model.nodeAddRequest = nil
            }
            .onChange(of: geo.size) { _, s in
                viewSize = s
                fitOnceIfSized()
            }
            .onChange(of: model.editingPath) { _, _ in fitToContent() }
            .onDisappear { removeScrollMonitor() }
        }
        .onContinuousHover(coordinateSpace: .named("canvas")) { phase in
            if case .active(let p) = phase { hoverLocation = p } else { hoverLocation = nil }
        }
        .gesture(MagnifyGesture()
            .onChanged { v in
                if zoomStart == nil { zoomStart = zoom }
                setZoom(zoomStart! * v.magnification, around: hoverLocation ?? CGPoint(x: viewSize.width / 2, y: viewSize.height / 2))
            }
            .onEnded { _ in zoomStart = nil })
        .focusable()
        .focusEffectDisabled()
        .onDeleteCommand {
            if model.selection.isEmpty, let g = model.selectedGroup {
                model.deleteGroup(g, withNodes: false, undoManager: undoManager)
            } else {
                model.deleteSelection(undoManager: undoManager)
            }
        }
        .onKeyPress(.escape) {
            if model.selection.isEmpty && model.selectedGroup == nil && !model.editingPath.isEmpty {
                model.exitSubgraph()
            } else {
                model.selection = []
                model.selectedGroup = nil
            }
            return .handled
        }
        .sheet(isPresented: $model.showPalette) {
            NodePalette { type in
                let at = model.paletteLocation ?? toGraph(CGPoint(x: viewSize.width / 2 - 120, y: viewSize.height / 2 - 60))
                model.addNode(type: type, at: at, undoManager: undoManager)
                model.paletteLocation = nil
            }
            .environment(model)
        }
    }

    // MARK: nodes

    @ViewBuilder
    func nodeView(_ node: GraphNode, _ level: Graph) -> some View {
        let selected = model.selection.contains(node.id)
        if NodeLayout.isReroute(node) {
            let type: SocketType = {
                if case .link(let src, let out)? = node.inputs["in"] { return model.resolvedType(src, out) ?? .any }
                return .any
            }()
            RerouteView(node: node, type: type, selected: selected, actions: actions(for: node))
                .scaleEffect(zoom, anchor: .topLeading)
        } else if NodeLayout.isNote(node) {
            NoteView(node: node, selected: selected,
                     text: LiteralBindings(model: model, undoManager: undoManager, editor: true).string(node.id, "text"),
                     actions: actions(for: node))
                .scaleEffect(zoom, anchor: .topLeading)
        } else if let schema = model.schema(for: node) {
            let flat = model.nodeStatePrefix + node.id
            NodeView(node: node, schema: schema, state: doc.nodeStates[flat],
                     selected: selected,
                     bindings: LiteralBindings(model: model, undoManager: undoManager, editor: true),
                     actions: actions(for: node),
                     boundInputs: model.boundInputs[node.id] ?? [:],
                     exposedOutputs: model.exposedOutputs[node.id] ?? [:],
                     result: doc.nodeResults[flat],
                     livePreview: doc.previewNode == flat ? doc.previewImage : nil)
                .scaleEffect(zoom, anchor: .topLeading)
        } else {
            unknownNode(node)
        }
    }

    func groupView(_ group: GraphGroup) -> some View {
        let origin = toView(CGPoint(x: group.x, y: group.y))
        return GroupView(
            group: group, size: CGSize(width: group.width * zoom, height: group.height * zoom), zoom: zoom,
            selected: model.selectedGroup == group.id,
            select: { model.selectedGroup = group.id; model.selection = [] },
            moveChanged: { t in
                if groupStart == nil {
                    groupStart = group
                    editStartGraph = model.graph
                    let inside = model.nodes(in: group.id)
                    dragStartPositions = Dictionary(uniqueKeysWithValues: model.editorGraph.nodes
                        .filter { inside.contains($0.id) }.map { ($0.id, $0.position ?? GraphPoint(x: 0, y: 0)) })
                    model.selectedGroup = group.id
                    model.selection = []
                }
                guard let start = groupStart else { return }
                let (dx, dy) = (Double(t.width / zoom), Double(t.height / zoom))
                var level = model.editorGraph
                if let i = level.groups?.firstIndex(where: { $0.id == group.id }) {
                    level.groups![i].x = start.x + dx
                    level.groups![i].y = start.y + dy
                }
                for (id, p) in dragStartPositions ?? [:] { level[id]?.position = GraphPoint(x: p.x + dx, y: p.y + dy) }
                model.editorGraph = level
            },
            moveEnded: { endLiveEdit("Move Group") },
            resizeChanged: { t in
                if groupStart == nil { groupStart = group; editStartGraph = model.graph }
                guard let start = groupStart else { return }
                var level = model.editorGraph
                if let i = level.groups?.firstIndex(where: { $0.id == group.id }) {
                    level.groups![i].width = max(120, start.width + Double(t.width / zoom))
                    level.groups![i].height = max(80, start.height + Double(t.height / zoom))
                }
                model.editorGraph = level
            },
            resizeEnded: { endLiveEdit("Resize Group") },
            setColor: { c in model.updateGroup(group.id, actionName: "Group Color", undoManager: undoManager) { $0.color = c } },
            delete: { withNodes in model.deleteGroup(group.id, withNodes: withNodes, undoManager: undoManager) })
            .offset(x: origin.x, y: origin.y)
    }

    func endLiveEdit(_ name: String) {
        if let start = editStartGraph { model.commitLiveEdit(from: start, actionName: name, undoManager: undoManager) }
        groupStart = nil
        dragStartPositions = nil
        editStartGraph = nil
    }

    // MARK: drops

    /// Node types from the Node Library, or files: assets and Finder files become Load nodes (or replace the file of the
    /// Load node they land on); workflows and outputs carrying a graph open in a tab (hold ⌥ to load an output as an
    /// input instead).
    func handleDrop(_ providers: [NSItemProvider], at loc: CGPoint) -> Bool {
        let point = toGraph(loc)
        let target = node(at: loc)
        let forceAsInput = NSEvent.modifierFlags.contains(.option)
        var handled = false
        for p in providers {
            if p.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                handled = true
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in model.dropFiles([url], at: point, onto: target, forceAsInput: forceAsInput) }
                }
            } else if p.canLoadObject(ofClass: NSString.self) {
                handled = true
                _ = p.loadObject(ofClass: NSString.self) { text, _ in
                    guard let s = text as? String, s.hasPrefix(NodeAddRequest.dragPrefix) else { return }
                    let type = String(s.dropFirst(NodeAddRequest.dragPrefix.count))
                    Task { @MainActor in model.addNode(type: type, at: point, undoManager: model.activeUndoManager) }
                }
            }
        }
        return handled
    }

    /// The node under a point in view coordinates.
    func node(at loc: CGPoint) -> String? {
        model.editorGraph.nodes.reversed().first { n in
            guard let s = model.schema(for: n) else { return false }
            let f = NodeLayout.frame(n, s)
            return CGRect(origin: toView(f.origin), size: CGSize(width: f.width * zoom, height: f.height * zoom)).contains(loc)
        }?.id
    }

    // MARK: coordinates

    func toView(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * zoom + pan.width, y: p.y * zoom + pan.height) }

    func toGraph(_ p: CGPoint) -> GraphPoint {
        GraphPoint(x: Double((p.x - pan.width) / zoom), y: Double((p.y - pan.height) / zoom))
    }

    func setZoom(_ z: CGFloat, around anchor: CGPoint) {
        let nz = min(max(z, 0.25), 2.5)
        let k = nz / zoom
        pan = CGSize(width: anchor.x - (anchor.x - pan.width) * k, height: anchor.y - (anchor.y - pan.height) * k)
        zoom = nz
    }

    /// Frames the graph the first time the editor has a real size.
    func fitOnceIfSized() {
        guard !didInitialFit, viewSize.width > 300, viewSize.height > 200 else { return }
        didInitialFit = true
        fitToContent()
    }

    func fitToContent() {
        let level = model.editorGraph
        var frames = level.nodes.compactMap { n in model.schema(for: n).map { NodeLayout.frame(n, $0) } }
        frames += (level.groups ?? []).map { CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height) }
        guard let first = frames.first, viewSize.width > 0 else { return }
        let box = frames.dropFirst().reduce(first) { $0.union($1) }.insetBy(dx: -40, dy: -40)
        let z = min(max(min(viewSize.width / box.width, viewSize.height / box.height), 0.25), 1.5)
        zoom = z
        pan = CGSize(width: (viewSize.width - box.width * z) / 2 - box.minX * z,
                     height: (viewSize.height - box.height * z) / 2 - box.minY * z + 16)
    }

    // MARK: gestures

    /// Click on the background clears the selection; ⌥-click on a wire inserts a Reroute there.
    func backgroundTap(at loc: CGPoint) {
        if NSEvent.modifierFlags.contains(.option), let (node, input) = wire(near: loc) {
            model.insertReroute(into: node, input: input, at: toGraph(loc), undoManager: undoManager)
            return
        }
        model.selection = []
        model.selectedGroup = nil
    }

    /// Background drag pans; with ⇧ it draws a selection rectangle.
    var backgroundDrag: some Gesture {
        DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
            .onChanged { v in
                if NSEvent.modifierFlags.contains(.shift) || marqueeStart != nil {
                    if marqueeStart == nil { marqueeStart = v.startLocation }
                    let r = CGRect(x: min(v.startLocation.x, v.location.x), y: min(v.startLocation.y, v.location.y),
                                   width: abs(v.location.x - v.startLocation.x),
                                   height: abs(v.location.y - v.startLocation.y))
                    marquee = r
                    model.selectedGroup = nil
                    model.selection = Set(model.editorGraph.nodes.filter { n in
                        guard let s = model.schema(for: n) else { return false }
                        let f = NodeLayout.frame(n, s)
                        let vf = CGRect(origin: toView(f.origin), size: CGSize(width: f.width * zoom, height: f.height * zoom))
                        return vf.intersects(r)
                    }.map(\.id))
                } else {
                    if panStart == nil { panStart = pan }
                    pan = CGSize(width: panStart!.width + v.translation.width, height: panStart!.height + v.translation.height)
                }
            }
            .onEnded { _ in
                panStart = nil
                marqueeStart = nil
                marquee = nil
            }
    }

    func actions(for node: GraphNode) -> NodeActions {
        let isInstance = node.type == Graph.subgraphType
        let selectionForMenu: () -> Void = {
            if !model.selection.contains(node.id) { model.selection = [node.id] }
        }
        return NodeActions(
            select: { extend in
                model.selectedGroup = nil
                if extend {
                    if model.selection.contains(node.id) { model.selection.remove(node.id) } else { model.selection.insert(node.id) }
                } else {
                    model.selection = [node.id]
                }
            },
            dragChanged: { t in
                if dragStartPositions == nil {
                    if !model.selection.contains(node.id) { model.selection = [node.id] }
                    editStartGraph = model.graph
                    dragStartPositions = Dictionary(uniqueKeysWithValues: model.editorGraph.nodes
                        .filter { model.selection.contains($0.id) }
                        .map { ($0.id, $0.position ?? GraphPoint(x: 0, y: 0)) })
                }
                var level = model.editorGraph
                for (id, start) in dragStartPositions ?? [:] {
                    level[id]?.position = GraphPoint(x: start.x + Double(t.width / zoom), y: start.y + Double(t.height / zoom))
                }
                model.editorGraph = level
            },
            dragEnded: { endLiveEdit("Move") },
            outputDragChanged: { output, loc in
                if wire == nil {
                    guard let type = model.resolvedType(node.id, output) ?? model.schema(for: node)?.output(output)?.type
                    else { return }
                    editStartGraph = model.graph
                    wire = WireDrag(fromNode: node.id, fromOutput: output, type: type, end: loc)
                }
                wire?.end = loc
            },
            outputDragEnded: { _, loc in finishWire(at: loc, actionIfDropped: nil) },
            inputDragChanged: { input, loc in
                if wire == nil {
                    // Dragging a connected input picks the link up from its source.
                    guard case .link(let src, let out)? = model.editorGraph[node.id]?.inputs[input] else { return }
                    let type = model.resolvedType(src, out) ?? model.schema(for: node)?.input(input)?.type ?? .any
                    editStartGraph = model.graph
                    model.editorGraph[node.id]?.inputs[input] = nil
                    wire = WireDrag(fromNode: src, fromOutput: out, type: type, end: loc)
                }
                wire?.end = loc
            },
            inputDragEnded: { _, loc in finishWire(at: loc, actionIfDropped: "Disconnect") },
            open: isInstance ? { model.enterSubgraph(node.id) } : nil,
            openResult: { r in model.show(r, in: doc); model.openViewer(r) },
            menu: NodeMenu(
                open: isInstance ? { model.enterSubgraph(node.id) } : nil,
                expand: isInstance ? { model.expandSubgraph(node.id, undoManager: undoManager) } : nil,
                duplicate: { selectionForMenu(); model.duplicateSelection(undoManager: undoManager) },
                group: { selectionForMenu(); model.groupSelection(undoManager: undoManager) },
                collapse: { selectionForMenu(); model.collapseSelection(undoManager: undoManager) },
                delete: { selectionForMenu(); model.deleteSelection(undoManager: undoManager) }))
    }

    /// Connects the dragged wire to the input under `loc`; otherwise commits `actionIfDropped` (or reverts).
    func finishWire(at loc: CGPoint, actionIfDropped: String?) {
        defer { wire = nil; editStartGraph = nil }
        guard let w = wire, let start = editStartGraph else { return }
        if let (target, input) = inputTarget(at: loc, for: w) {
            var g = model.editorGraph
            g[target]?.inputs[input] = .link(node: w.fromNode, output: w.fromOutput)
            if model.wouldCreateCycle(g) {
                NSSound.beep()
                model.graph = start
                return
            }
            model.editorGraph = g
            model.commitLiveEdit(from: start, actionName: "Connect", undoManager: undoManager)
        } else if let action = actionIfDropped {
            model.commitLiveEdit(from: start, actionName: action, undoManager: undoManager)
        } else {
            model.graph = start
        }
    }

    /// The compatible input socket nearest to `loc` (within reach), or the only compatible input of the node under
    /// it. Dropping a scalar wire (e.g. a STRING) on a node can also target a widget input of that type.
    func inputTarget(at loc: CGPoint, for w: WireDrag) -> (String, String)? {
        let level = model.editorGraph
        var best: (String, String, CGFloat)?
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n) else { continue }
            for spec in s.inputs where spec.type.accepts(w.type) && !spec.type.isScalar {
                guard let p = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let v = toView(p)
                let d = hypot(v.x - loc.x, v.y - loc.y)
                if d < 22 * max(zoom, 0.6), d < (best?.2 ?? .infinity) { best = (n.id, spec.name, d) }
            }
        }
        if let best { return (best.0, best.1) }
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n) else { continue }
            let f = NodeLayout.frame(n, s)
            let vf = CGRect(origin: toView(f.origin), size: CGSize(width: f.width * zoom, height: f.height * zoom))
            guard vf.contains(loc) else { continue }
            let compatible = s.inputs.filter {
                $0.type.isScalar ? $0.type == w.type : $0.type.accepts(w.type)
            }
            if compatible.count == 1 { return (n.id, compatible[0].name) }
        }
        return nil
    }

    /// The link (target node, input) whose curve passes within a few points of `loc`.
    func wire(near loc: CGPoint) -> (String, String)? {
        var best: (String, String, CGFloat)?
        for (node, input, _, a, b, _) in wireSegments(model.editorGraph) {
            let dx = max(abs(b.x - a.x) * 0.5, 50 * zoom)
            for i in 0...32 {
                let t = CGFloat(i) / 32, u = 1 - t
                let c1 = CGPoint(x: a.x + dx, y: a.y), c2 = CGPoint(x: b.x - dx, y: b.y)
                let x = u * u * u * a.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * b.x
                let y = u * u * u * a.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * b.y
                let d = hypot(x - loc.x, y - loc.y)
                if d < 8, d < (best?.2 ?? .infinity) { best = (node, input, d) }
            }
        }
        return best.map { ($0.0, $0.1) }
    }

    // MARK: drawing

    /// Every drawn link in view coordinates: (target node, input, source node, from, to, type).
    func wireSegments(_ level: Graph) -> [(String, String, String, CGPoint, CGPoint, SocketType)] {
        var out: [(String, String, String, CGPoint, CGPoint, SocketType)] = []
        for n in level.nodes {
            guard let s = model.schema(for: n) else { continue }
            for spec in s.inputs {
                guard case .link(let src, let o)? = n.inputs[spec.name], let srcNode = level[src],
                      let srcSchema = model.schema(for: srcNode),
                      let a = NodeLayout.outputPoint(srcNode, srcSchema, o),
                      let b = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let type = spec.type == .any ? (model.resolvedType(src, o) ?? .any) : spec.type
                out.append((n.id, spec.name, src, toView(a), toView(b), type))
            }
        }
        return out
    }

    func drawWires(_ ctx: inout GraphicsContext, _ level: Graph) {
        let width = max(1.5, 2.5 * zoom)
        for (node, _, src, a, b, type) in wireSegments(level) {
            let highlighted = model.selection.contains(node) || model.selection.contains(src)
            stroke(&ctx, from: a, to: b, color: type.color.opacity(highlighted ? 1 : 0.75),
                   width: highlighted ? width * 1.5 : width)
        }
        if let w = wire, let srcNode = level[w.fromNode], let srcSchema = model.schema(for: srcNode),
           let a = NodeLayout.outputPoint(srcNode, srcSchema, w.fromOutput) {
            stroke(&ctx, from: toView(a), to: w.end, color: w.type.color, width: width, dashed: true)
        }
    }

    func stroke(_ ctx: inout GraphicsContext, from a: CGPoint, to b: CGPoint, color: Color, width: CGFloat,
                dashed: Bool = false) {
        var path = Path()
        path.move(to: a)
        let dx = max(abs(b.x - a.x) * 0.5, 50 * zoom)
        path.addCurve(to: b, control1: CGPoint(x: a.x + dx, y: a.y), control2: CGPoint(x: b.x - dx, y: b.y))
        ctx.stroke(path, with: .color(color),
                   style: StrokeStyle(lineWidth: width, lineCap: .round, dash: dashed ? [6, 4] : []))
    }

    func unknownNode(_ node: GraphNode) -> some View {
        Label("Unknown node type “\(node.type)”", systemImage: "questionmark.square.dashed")
            .padding(10)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color.red.opacity(0.2)))
            .scaleEffect(zoom, anchor: .topLeading)
    }

    // MARK: overlays

    /// Breadcrumbs while inside a subgraph; a hint when the level is empty.
    @ViewBuilder var topBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.editingPath.isEmpty {
                HStack(spacing: 6) {
                    Button { model.exitSubgraph() } label: { Image(systemName: "chevron.left") }
                        .help("Back (Esc)")
                    Button("Graph") { model.exitSubgraph(to: 0) }.buttonStyle(.link)
                    ForEach(Array(model.editingTitles.enumerated()), id: \.offset) { i, title in
                        Image(systemName: "chevron.right").font(.caption2).foregroundStyle(.secondary)
                        if i == model.editingTitles.count - 1 {
                            Text(title).fontWeight(.semibold)
                        } else {
                            Button(title) { model.exitSubgraph(to: i + 1) }.buttonStyle(.link)
                        }
                    }
                }
                .font(.callout)
                .buttonStyle(.borderless)
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .background(.regularMaterial, in: Capsule())
            }
            if model.editorGraph.nodes.isEmpty {
                Text("Double-click or press ⌘K to add a node.")
                    .font(.callout).foregroundStyle(.secondary)
                    .padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    var zoomControls: some View {
        HStack(spacing: 2) {
            Button { setZoom(zoom / 1.2, around: CGPoint(x: viewSize.width / 2, y: viewSize.height / 2)) } label: {
                Image(systemName: "minus")
            }
            Text("\(Int((zoom * 100).rounded()))%").font(.caption.monospacedDigit()).frame(width: 44)
            Button { setZoom(zoom * 1.2, around: CGPoint(x: viewSize.width / 2, y: viewSize.height / 2)) } label: {
                Image(systemName: "plus")
            }
            Divider().frame(height: 16)
            Button { fitToContent() } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                .help("Fit graph to window")
            Button {
                model.paletteLocation = nil
                model.showPalette = true
            } label: { Image(systemName: "plus.square.on.square") }
                .help("Add node (⌘K or double-click)")
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(.regularMaterial, in: Capsule())
    }

    // MARK: scroll wheel

    /// Two-finger scroll pans, ⌘-scroll zooms around the pointer (SwiftUI has no scroll-wheel gesture on macOS).
    func installScrollMonitor() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard let loc = hoverLocation, model.viewerResult == nil, !model.showPalette else { return event }
            if event.modifierFlags.contains(.command) {
                setZoom(zoom * exp(event.scrollingDeltaY * (event.hasPreciseScrollingDeltas ? 0.01 : 0.1)), around: loc)
            } else {
                let k: CGFloat = event.hasPreciseScrollingDeltas ? 1 : 10
                pan = CGSize(width: pan.width + event.scrollingDeltaX * k, height: pan.height + event.scrollingDeltaY * k)
            }
            return nil
        }
    }

    func removeScrollMonitor() {
        if let m = scrollMonitor { NSEvent.removeMonitor(m) }
        scrollMonitor = nil
    }
}

struct GridBackground: View {
    let zoom: CGFloat
    let pan: CGSize

    var body: some View {
        Canvas { ctx, size in
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(nsColor: .underPageBackgroundColor)))
            let spacing = 24 * zoom
            guard spacing > 6 else { return }
            let dot = max(1, 1.6 * zoom)
            let color = Color.primary.opacity(0.13)
            var x = pan.width.truncatingRemainder(dividingBy: spacing)
            if x < 0 { x += spacing }
            while x < size.width {
                var y = pan.height.truncatingRemainder(dividingBy: spacing)
                if y < 0 { y += spacing }
                while y < size.height {
                    ctx.fill(Path(ellipseIn: CGRect(x: x - dot / 2, y: y - dot / 2, width: dot, height: dot)),
                             with: .color(color))
                    y += spacing
                }
                x += spacing
            }
        }
    }
}

struct NodePalette: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    let add: (String) -> Void

    var matches: [NodeSchema] {
        let q = query.lowercased()
        return model.schemas.filter {
            q.isEmpty || $0.displayName.lowercased().contains(q) || $0.type.lowercased().contains(q)
                || $0.category.lowercased().contains(q)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            TextField("Search nodes", text: $query)
                .textFieldStyle(.plain)
                .font(.title3)
                .padding(12)
                .onSubmit { if let first = matches.first { choose(first.type) } }
            Divider()
            List(matches, id: \.type) { s in
                Button { choose(s.type) } label: {
                    HStack {
                        Circle().fill(s.tint).frame(width: 8, height: 8)
                        Text(s.displayName)
                        Spacer()
                        Text(s.category).font(.caption).foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            .listStyle(.plain)
        }
        .frame(width: 420, height: 360)
        .onExitCommand { dismiss() }
    }

    func choose(_ type: String) {
        add(type)
        dismiss()
    }
}
