import AppKit
import GraphRuntime
import MLXComfyLibrary
import SwiftUI
import UniformTypeIdentifiers

/// A link being dragged: from an output socket (forward), or backwards from an unlinked input socket.
struct WireDrag {
    var fromNode: String
    /// The output it hangs from (forward), or the input (reversed).
    var fromOutput: String
    var type: SocketType
    var end: CGPoint  // view coordinates
    var reversed = false
}

/// A link, named by the input it feeds.
struct WireRef: Hashable {
    var node: String
    var input: String
}

/// One drawn link, routed in graph coordinates.
struct WireInfo {
    var target: String
    var input: String
    var source: String
    var output: String
    var type: SocketType
    var route: WireRoute
    var ref: WireRef { WireRef(node: target, input: input) }
    var isModel: Bool { type == .model }
}

/// The node editor for the current editing level (the top-level graph, or a subgraph entered with a double-click).
struct GraphEditorView: View {
    @Environment(AppModel.self) private var model
    /// The tab shown (the view is rebuilt per tab, and keeps the tab's zoom and pan in it).
    let doc: WorkflowDocument
    private var undoManager: UndoManager? { model.activeUndoManager }

    @AppStorage("wireStyle") private var wireStyleRaw = WireStyle.rightAngle.rawValue
    @AppStorage("showModelWires") private var showModelWires = false
    @AppStorage("snapToGrid") private var snapToGrid = true
    @AppStorage("showMinimap") private var showMinimap = true

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
    @State private var hoveredNode: String?
    @State private var hoveredWire: WireRef?
    @State private var selectedWire: WireRef?
    @State private var viewSize: CGSize = .zero
    @State private var scrollMonitor: Any?
    @State private var didInitialFit = false
    @State private var dropTargeted = false

    static let grid: CGFloat = 24

    var wireStyle: WireStyle { WireStyle(rawValue: wireStyleRaw) ?? .rightAngle }

    var body: some View {
        @Bindable var model = model
        let level = model.editorGraph
        let wires = wireInfos(level)
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                GridBackground(zoom: zoom, pan: pan)
                    .contentShape(Rectangle())
                    .gesture(backgroundDrag)
                    .gesture(SpatialTapGesture(count: 2, coordinateSpace: .named("canvas")).onEnded { v in
                        model.pendingWire = nil
                        model.paletteLocation = toGraph(v.location)
                        model.showPalette = true
                    }.exclusively(before: SpatialTapGesture(coordinateSpace: .named("canvas")).onEnded { v in
                        backgroundTap(at: v.location, wires: wires)
                    }))

                ForEach(level.groups ?? []) { group in groupView(group) }

                Canvas { ctx, _ in drawWires(&ctx, wires, level) }
                    .allowsHitTesting(false)

                ForEach(level.nodes) { node in
                    nodeView(node, level).offset(x: toView(NodeLayout.origin(node)).x, y: toView(NodeLayout.origin(node)).y)
                }

                if let marquee {
                    Rectangle().fill(MG.accent.opacity(0.12))
                        .overlay(Rectangle().strokeBorder(MG.accent, lineWidth: 1))
                        .frame(width: marquee.width, height: marquee.height)
                        .offset(x: marquee.minX, y: marquee.minY)
                        .allowsHitTesting(false)
                }

                if wire == nil, let ref = hoveredWire, let w = wires.first(where: { $0.ref == ref }), let p = hoverLocation {
                    wireLabel(w, level)
                        .offset(x: p.x + 14, y: p.y + 14)
                        .allowsHitTesting(false)
                }
            }
            .mgContextMenu(at: { loc in contextItems(at: loc, wires: wires) })
            .mgTestAnchor("graph-canvas")
            .coordinateSpace(name: "canvas")
            .clipped()
            .onDrop(of: [.fileURL, .plainText], isTargeted: $dropTargeted) { providers, loc in
                handleDrop(providers, at: loc)
            }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(MG.accent, style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                        .padding(6).allowsHitTesting(false)
                }
            }
            .overlay(alignment: .bottomTrailing) { zoomControls.padding(12) }
            .overlay(alignment: .bottomLeading) { minimap.padding(12) }
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
            .onChange(of: model.selection) { _, s in if !s.isEmpty { selectedWire = nil } }
            .onChange(of: geo.size) { _, s in
                viewSize = s
                fitOnceIfSized()
            }
            .onChange(of: model.editingPath) { _, _ in
                selectedWire = nil
                fitToContent()
            }
            .onDisappear { removeScrollMonitor() }
        }
        .onContinuousHover(coordinateSpace: .named("canvas")) { phase in
            if case .active(let p) = phase {
                hoverLocation = p
                updateHover(at: p, wires: wires)
            } else {
                hoverLocation = nil
                if hoveredNode != nil { hoveredNode = nil }
                if hoveredWire != nil { hoveredWire = nil }
            }
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
            if let w = selectedWire {
                model.disconnect(node: w.node, input: w.input, undoManager: undoManager)
                selectedWire = nil
            } else if model.selection.isEmpty, let g = model.selectedGroup {
                model.deleteGroup(g, withNodes: false, undoManager: undoManager)
            } else {
                model.deleteSelection(undoManager: undoManager)
            }
        }
        .onCopyCommand { copyProviders() }
        .onCutCommand {
            let providers = copyProviders()
            model.deleteSelection(undoManager: undoManager)
            return providers
        }
        .onPasteCommand(of: [.plainText, .utf8PlainText]) { _ in
            model.paste(at: hoverLocation.map(toGraph), undoManager: undoManager)
        }
        .onKeyPress(.escape) {
            if selectedWire != nil {
                selectedWire = nil
            } else if model.selection.isEmpty && model.selectedGroup == nil && !model.editingPath.isEmpty {
                model.exitSubgraph()
            } else {
                model.selection = []
                model.selectedGroup = nil
            }
            return .handled
        }
        .mgSheet("node-palette", isPresented: $model.showPalette, onDismiss: { model.pendingWire = nil }) {
            NodePalette(wire: model.pendingWire) { type in
                if let w = model.pendingWire {
                    model.addNode(type: type, connecting: w, undoManager: undoManager)
                } else {
                    let at = model.paletteLocation ?? toGraph(CGPoint(x: viewSize.width / 2 - 120, y: viewSize.height / 2 - 60))
                    model.addNode(type: type, at: at, undoManager: undoManager)
                }
                model.paletteLocation = nil
                model.pendingWire = nil
            }
            .environment(model)
        }
    }

    // MARK: context menus

    /// What a right-click at `loc` (canvas coordinates) offers: the node's, wire's or group's commands, or the canvas's.
    func contextItems(at loc: CGPoint, wires: [WireInfo]) -> [MGMenuItem] {
        hoverLocation = loc
        if let id = node(at: loc), let n = model.editorGraph[id] { return nodeMenuItems(actions(for: n).menu) }
        if let w = wire(near: loc, in: wires) {
            selectedWire = w.ref
            return [
                .button("Insert Reroute") {
                    model.insertReroute(into: w.target, input: w.input, at: toGraph(loc), undoManager: undoManager)
                },
                .divider,
                .button("Delete Wire", shortcut: KeyboardShortcut(.delete, modifiers: []), destructive: true) {
                    model.disconnect(node: w.target, input: w.input, undoManager: undoManager)
                    selectedWire = nil
                },
            ]
        }
        if let g = group(at: loc) { return groupItems(g) }
        return canvasItems(at: loc)
    }

    func canvasItems(at loc: CGPoint) -> [MGMenuItem] {
        [
            .button("Add Node…", shortcut: KeyboardShortcut("k")) {
                model.pendingWire = nil
                model.paletteLocation = toGraph(loc)
                model.showPalette = true
            },
            .menu("Add to Workflow") { model.addToWorkflowItems() },
            .button("Paste", shortcut: KeyboardShortcut("v"), enabled: model.canPaste) {
                model.paste(at: toGraph(loc), undoManager: undoManager)
            },
            .divider,
            .button("Arrange Graph", shortcut: KeyboardShortcut("l", modifiers: [.command, .shift])) {
                model.arrange(selectionOnly: false, undoManager: undoManager)
            },
            .button("Fit Graph to Window") { fitToContent() },
        ]
    }

    /// The group under `loc`: its frame, or the title floating above it.
    func group(at loc: CGPoint) -> GraphGroup? {
        (model.editorGraph.groups ?? []).reversed().first { g in
            let r = viewRect(CGRect(x: g.x, y: g.y, width: g.width, height: g.height))
            return r.contains(loc) || CGRect(x: r.minX, y: r.minY - 28, width: r.width, height: 28).contains(loc)
        }
    }

    func groupItems(_ group: GraphGroup) -> [MGMenuItem] {
        model.selectedGroup = group.id
        model.selection = []
        return [
            .menu("Color") {
                for c in AppModel.groupColors {
                    MGMenuItem.button(c.capitalized, checked: group.color == c) {
                        model.updateGroup(group.id, actionName: "Group Color", undoManager: undoManager) { $0.color = c }
                    }
                }
            },
            .divider,
            .button("Delete Group") { model.deleteGroup(group.id, withNodes: false, undoManager: undoManager) },
            .button("Delete Group and Its Nodes", destructive: true) {
                model.deleteGroup(group.id, withNodes: true, undoManager: undoManager)
            },
        ]
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
            NodeView(node: node, schema: schema, state: doc.nodeStates[flat], statusText: doc.nodeStatus[flat],
                     selected: selected,
                     bindings: LiteralBindings(model: model, undoManager: undoManager, editor: true),
                     actions: actions(for: node),
                     boundInputs: model.boundInputs[node.id] ?? [:],
                     exposedOutputs: model.exposedOutputs[node.id] ?? [:],
                     result: doc.nodeResults[flat],
                     livePreview: doc.previewNode == flat ? doc.previewImage : nil,
                     zoom: zoom,
                     modelSources: showModelWires ? [:] : modelSources(node, schema, level),
                     linkCount: linkCount(node.id, level))
                .scaleEffect(zoom, anchor: .topLeading)
        } else {
            unknownNode(node)
        }
    }

    /// Model inputs whose wire is hidden: input → the title of the node it comes from.
    func modelSources(_ node: GraphNode, _ schema: NodeSchema, _ level: Graph) -> [String: String] {
        var out: [String: String] = [:]
        for spec in schema.inputs {
            guard case .link(let src, let o)? = node.inputs[spec.name], let s = level[src] else { continue }
            let type = spec.type == .any ? model.resolvedType(src, o) : spec.type
            guard type == .model else { continue }
            out[spec.name] = s.title ?? model.schema(for: s)?.displayName ?? src
        }
        return out
    }

    func linkCount(_ id: String, _ level: Graph) -> Int {
        var n = 0
        for node in level.nodes {
            for input in node.inputs.values {
                guard case .link(let src, _) = input else { continue }
                if src == id || node.id == id { n += 1 }
            }
        }
        return n
    }

    func groupView(_ group: GraphGroup) -> some View {
        let origin = toView(CGPoint(x: group.x, y: group.y))
        return GroupView(
            group: group, size: CGSize(width: group.width * zoom, height: group.height * zoom), zoom: zoom,
            selected: model.selectedGroup == group.id,
            select: { model.selectedGroup = group.id; model.selection = []; selectedWire = nil },
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
                let (dx, dy) = snapped(CGPoint(x: start.x, y: start.y), Double(t.width / zoom), Double(t.height / zoom))
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

    /// A drag offset, adjusted so `start` lands on the grid (unless snapping is off or ⌘ is held).
    func snapped(_ start: CGPoint, _ dx: Double, _ dy: Double) -> (Double, Double) {
        guard snapToGrid, !NSEvent.modifierFlags.contains(.command) else { return (dx, dy) }
        let p = Alignment.snap(CGPoint(x: start.x + dx, y: start.y + dy), grid: Self.grid)
        return (Double(p.x - start.x), Double(p.y - start.y))
    }

    func endLiveEdit(_ name: String) {
        if let start = editStartGraph { model.commitLiveEdit(from: start, actionName: name, undoManager: undoManager) }
        groupStart = nil
        dragStartPositions = nil
        editStartGraph = nil
    }

    // MARK: copy and paste

    func copyProviders() -> [NSItemProvider] {
        guard let clip = model.clipboardForSelection(), let data = try? JSONEncoder().encode(clip) else { return [] }
        return [NSItemProvider(object: String(decoding: data, as: UTF8.self) as NSString)]
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
            guard let f = model.frame(of: n) else { return false }
            return viewRect(f).contains(loc)
        }?.id
    }

    // MARK: coordinates

    func toView(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * zoom + pan.width, y: p.y * zoom + pan.height) }

    func toGraph(_ p: CGPoint) -> GraphPoint {
        GraphPoint(x: Double((p.x - pan.width) / zoom), y: Double((p.y - pan.height) / zoom))
    }

    func viewRect(_ f: CGRect) -> CGRect {
        CGRect(origin: toView(f.origin), size: CGSize(width: f.width * zoom, height: f.height * zoom))
    }

    var viewTransform: CGAffineTransform {
        CGAffineTransform(translationX: pan.width, y: pan.height).scaledBy(x: zoom, y: zoom)
    }

    func setZoom(_ z: CGFloat, around anchor: CGPoint) {
        let nz = min(max(z, 0.2), 2.5)
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

    var contentBounds: CGRect? {
        let level = model.editorGraph
        var frames = level.nodes.compactMap { model.frame(of: $0) }
        frames += (level.groups ?? []).map { CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height) }
        guard let first = frames.first else { return nil }
        return frames.dropFirst().reduce(first) { $0.union($1) }
    }

    func fitToContent() {
        guard let content = contentBounds, viewSize.width > 0 else { return }
        let box = content.insetBy(dx: -40, dy: -40)
        let z = min(max(min(viewSize.width / box.width, viewSize.height / box.height), 0.2), 1.5)
        zoom = z
        pan = CGSize(width: (viewSize.width - box.width * z) / 2 - box.minX * z,
                     height: (viewSize.height - box.height * z) / 2 - box.minY * z + 16)
    }

    /// Scrolls so the node is in the middle of the editor.
    func center(on id: String) {
        guard let n = model.editorGraph[id], let f = model.frame(of: n) else { return }
        withAnimation(.easeInOut(duration: 0.25)) {
            pan = CGSize(width: viewSize.width / 2 - f.midX * zoom, height: viewSize.height / 2 - f.midY * zoom)
        }
    }

    // MARK: gestures

    /// Click on the background: selects a wire under the pointer (⌥-click inserts a Reroute there), else clears the
    /// selection.
    func backgroundTap(at loc: CGPoint, wires: [WireInfo]) {
        if let w = wire(near: loc, in: wires) {
            if NSEvent.modifierFlags.contains(.option) {
                model.insertReroute(into: w.target, input: w.input, at: toGraph(loc), undoManager: undoManager)
            } else {
                model.selection = []
                model.selectedGroup = nil
                selectedWire = w.ref
            }
            return
        }
        selectedWire = nil
        model.selection = []
        model.selectedGroup = nil
    }

    func updateHover(at p: CGPoint, wires: [WireInfo]) {
        guard wire == nil, dragStartPositions == nil, panStart == nil else { return }
        let n = node(at: p)
        if n != hoveredNode { hoveredNode = n }
        let w = n == nil ? wire(near: p, in: wires)?.ref : nil
        if w != hoveredWire { hoveredWire = w }
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
                        guard let f = model.frame(of: n) else { return false }
                        return viewRect(f).intersects(r)
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
        let multi = model.selection.count > 1 && model.selection.contains(node.id)
        let collapsible = !NodeLayout.isReroute(node) && !NodeLayout.isNote(node)
        return NodeActions(
            select: { extend in
                model.selectedGroup = nil
                selectedWire = nil
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
                let anchor = dragStartPositions?[node.id] ?? GraphPoint(x: 0, y: 0)
                let (dx, dy) = snapped(CGPoint(x: anchor.x, y: anchor.y), Double(t.width / zoom), Double(t.height / zoom))
                var level = model.editorGraph
                for (id, start) in dragStartPositions ?? [:] {
                    level[id]?.position = GraphPoint(x: start.x + dx, y: start.y + dy)
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
                    editStartGraph = model.graph
                    if case .link(let src, let out)? = model.editorGraph[node.id]?.inputs[input] {
                        // Dragging a connected input picks the link up from its source.
                        let type = model.resolvedType(src, out) ?? model.schema(for: node)?.input(input)?.type ?? .any
                        model.editorGraph[node.id]?.inputs[input] = nil
                        wire = WireDrag(fromNode: src, fromOutput: out, type: type, end: loc)
                    } else {
                        // An empty input draws a wire backwards, to an output.
                        let type = model.schema(for: node)?.input(input)?.type ?? .any
                        wire = WireDrag(fromNode: node.id, fromOutput: input, type: type, end: loc, reversed: true)
                    }
                }
                wire?.end = loc
            },
            inputDragEnded: { _, loc in
                if wire?.reversed == true { finishReversedWire(at: loc) } else { finishWire(at: loc, actionIfDropped: "Disconnect") }
            },
            open: isInstance ? { model.enterSubgraph(node.id) } : nil,
            openResult: { r in model.show(r, in: doc); model.openViewer(r) },
            toggleCollapse: collapsible ? { model.toggleCollapsed([node.id], undoManager: undoManager) } : nil,
            revealSource: { input in
                guard case .link(let src, _)? = model.editorGraph[node.id]?.inputs[input] else { return }
                model.selection = [src]
                center(on: src)
            },
            menu: NodeMenu(
                open: isInstance ? { model.enterSubgraph(node.id) } : nil,
                expand: isInstance ? { model.expandSubgraph(node.id, undoManager: undoManager) } : nil,
                toggleCollapse: collapsible ? {
                    selectionForMenu()
                    model.toggleCollapsed(model.selection, undoManager: undoManager)
                } : nil,
                isCollapsed: node.isCollapsed,
                copy: { selectionForMenu(); model.copySelection() },
                duplicate: { selectionForMenu(); model.duplicateSelection(undoManager: undoManager) },
                arrange: multi ? { model.arrange(selectionOnly: true, undoManager: undoManager) } : nil,
                align: multi ? { mode in model.alignSelection(mode, undoManager: undoManager) } : nil,
                alignCount: model.selection.count,
                group: { selectionForMenu(); model.groupSelection(undoManager: undoManager) },
                collapse: { selectionForMenu(); model.collapseSelection(undoManager: undoManager) },
                delete: { selectionForMenu(); model.deleteSelection(undoManager: undoManager) }))
    }

    /// Connects the dragged wire to the input under `loc`. Dropped on empty canvas, a new wire opens the node search
    /// for something to connect; otherwise commits `actionIfDropped` (or reverts).
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
            if node(at: loc) == nil {
                openPalette(for: AppModel.PendingWire(node: w.fromNode, socket: w.fromOutput, type: w.type, fromOutput: true,
                                                      point: toGraph(loc)))
            }
        }
    }

    /// Connects a wire drawn backwards from an empty input to the output under `loc`, or offers nodes that can feed it.
    func finishReversedWire(at loc: CGPoint) {
        defer { wire = nil; editStartGraph = nil }
        guard let w = wire, let start = editStartGraph else { return }
        if let (source, output) = outputTarget(at: loc, for: w) {
            var g = model.editorGraph
            g[w.fromNode]?.inputs[w.fromOutput] = .link(node: source, output: output)
            if model.wouldCreateCycle(g) {
                NSSound.beep()
                model.graph = start
                return
            }
            model.editorGraph = g
            model.commitLiveEdit(from: start, actionName: "Connect", undoManager: undoManager)
        } else {
            model.graph = start
            if node(at: loc) == nil {
                openPalette(for: AppModel.PendingWire(node: w.fromNode, socket: w.fromOutput, type: w.type, fromOutput: false,
                                                      point: toGraph(loc)))
            }
        }
    }

    func openPalette(for pending: AppModel.PendingWire) {
        model.pendingWire = pending
        model.paletteLocation = pending.point
        model.showPalette = true
    }

    /// The compatible input socket nearest to `loc` (within reach), or the only compatible input of the node under
    /// it. Dropping a scalar wire (e.g. a STRING) on a node can also target a widget input of that type.
    func inputTarget(at loc: CGPoint, for w: WireDrag) -> (String, String)? {
        let level = model.editorGraph
        var best: (String, String, CGFloat)?
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n), !n.isCollapsed else { continue }
            for spec in s.inputs where spec.type.accepts(w.type) && !spec.type.isScalar {
                guard let p = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let v = toView(p)
                let d = hypot(v.x - loc.x, v.y - loc.y)
                if d < 22 * max(zoom, 0.6), d < (best?.2 ?? .infinity) { best = (n.id, spec.name, d) }
            }
        }
        if let best { return (best.0, best.1) }
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n), let f = model.frame(of: n), viewRect(f).contains(loc) else { continue }
            let compatible = s.inputs.filter {
                $0.type.isScalar ? $0.type == w.type : $0.type.accepts(w.type)
            }
            if compatible.count == 1 { return (n.id, compatible[0].name) }
        }
        return nil
    }

    /// The output socket nearest to `loc` that the dragged input accepts, or the only one of the node under it.
    func outputTarget(at loc: CGPoint, for w: WireDrag) -> (String, String)? {
        let level = model.editorGraph
        var best: (String, String, CGFloat)?
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n) else { continue }
            for spec in s.outputs where w.type.accepts(model.resolvedType(n.id, spec.name) ?? spec.type) {
                guard let p = NodeLayout.outputPoint(n, s, spec.name) else { continue }
                let v = toView(p)
                let d = hypot(v.x - loc.x, v.y - loc.y)
                if d < 22 * max(zoom, 0.6), d < (best?.2 ?? .infinity) { best = (n.id, spec.name, d) }
            }
        }
        if let best { return (best.0, best.1) }
        for n in level.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n), let f = model.frame(of: n), viewRect(f).contains(loc) else { continue }
            let compatible = s.outputs.filter { w.type.accepts(model.resolvedType(n.id, $0.name) ?? $0.type) }
            if compatible.count == 1 { return (n.id, compatible[0].name) }
        }
        return nil
    }

    // MARK: wires

    /// Every link at this level, routed in graph coordinates.
    func wireInfos(_ level: Graph) -> [WireInfo] {
        var out: [WireInfo] = []
        let style = wireStyle
        for n in level.nodes {
            guard let s = model.schema(for: n) else { continue }
            let sockets = NodeLayout.rows(n, s).sockets
            for spec in s.inputs {
                guard case .link(let src, let o)? = n.inputs[spec.name], let srcNode = level[src],
                      let srcSchema = model.schema(for: srcNode),
                      let a = NodeLayout.outputPoint(srcNode, srcSchema, o),
                      let b = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let type = spec.type == .any ? (model.resolvedType(src, o) ?? .any) : spec.type
                let index = sockets.firstIndex { $0.name == spec.name } ?? 0
                let lane = n.isCollapsed ? 0 : index - sockets.count / 2
                let route = WireRouter.route(from: a, to: b, style: style,
                                             sourceFrame: NodeLayout.frame(srcNode, srcSchema),
                                             targetFrame: NodeLayout.frame(n, s), lane: lane)
                out.append(WireInfo(target: n.id, input: spec.name, source: src, output: o, type: type, route: route))
            }
        }
        return out
    }

    /// Whether a wire is drawn: hidden model wires show only while one of their ends is hovered or selected.
    func isVisible(_ w: WireInfo) -> Bool {
        guard w.isModel, !showModelWires else { return true }
        let ends: Set<String> = [w.source, w.target]
        if !ends.isDisjoint(with: model.selection) { return true }
        if let h = hoveredNode, ends.contains(h) { return true }
        if selectedWire == w.ref { return true }
        return wire?.type == .model
    }

    /// The visible wire passing within a few points of `loc`.
    func wire(near loc: CGPoint, in wires: [WireInfo]) -> WireInfo? {
        let p = toGraph(loc)
        let gp = CGPoint(x: p.x, y: p.y)
        let reach = 6 / zoom
        var best: (WireInfo, CGFloat)?
        for w in wires where isVisible(w) {
            let b = w.route.points.reduce(CGRect.null) { $0.union(CGRect(origin: $1, size: .zero)) }.insetBy(dx: -reach, dy: -reach)
            guard b.contains(gp) else { continue }
            let d = w.route.distance(to: gp, step: max(2, 4 / zoom))
            if d < reach, d < (best?.1 ?? .infinity) { best = (w, d) }
        }
        return best?.0
    }

    /// Nodes upstream and downstream of the selection (for highlighting the paths through it).
    func selectionPaths(_ wires: [WireInfo]) -> (up: Set<String>, down: Set<String>) {
        let sel = model.selection
        guard !sel.isEmpty else { return ([], []) }
        var up = Set<String>(), down = Set<String>()
        var stack = Array(sel)
        while let id = stack.popLast() {
            for w in wires where w.target == id && !w.isModel && up.insert(w.source).inserted { stack.append(w.source) }
        }
        stack = Array(sel)
        while let id = stack.popLast() {
            for w in wires where w.source == id && !w.isModel && down.insert(w.target).inserted { stack.append(w.target) }
        }
        return (up, down)
    }

    func drawWires(_ ctx: inout GraphicsContext, _ wires: [WireInfo], _ level: Graph) {
        let t = viewTransform
        let base = max(1.25, 2.2 * zoom)
        let sel = model.selection
        let dimming = !sel.isEmpty || selectedWire != nil
        let (up, down) = selectionPaths(wires)
        func onPath(_ w: WireInfo) -> Bool {
            if selectedWire == w.ref { return true }
            if sel.contains(w.source) || sel.contains(w.target) { return true }
            return (up.contains(w.source) && (up.contains(w.target) || sel.contains(w.target)))
                || (down.contains(w.target) && (down.contains(w.source) || sel.contains(w.source)))
        }
        let visible = wires.filter(isVisible)
        // Dimmed wires first, so the highlighted ones draw on top.
        for w in visible.sorted(by: { !onPath($0) && onPath($1) }) {
            let path = Path(w.route.cgPath()).applying(t)
            let highlighted = onPath(w)
            var opacity: Double = dimming ? (highlighted ? 1 : 0.3) : 0.85
            if w.isModel && !showModelWires { opacity = min(opacity, 0.6) }
            let direct = sel.contains(w.source) || sel.contains(w.target)
            let width = base * (direct ? 1.35 : 1) * (hoveredWire == w.ref ? 1.6 : 1)
            let style = { (lw: CGFloat) in StrokeStyle(lineWidth: lw, lineCap: .round, lineJoin: .round) }
            if selectedWire == w.ref {
                ctx.stroke(path, with: .color(MG.accent.opacity(0.9)), style: style(width + 5))
            }
            ctx.stroke(path, with: .color(MG.shadow.opacity(opacity)), style: style(width + 2))
            ctx.stroke(path, with: .color(w.type.color.opacity(opacity)), style: style(width))
        }
        if let w = wire {
            let end = toGraph(w.end)
            let e = CGPoint(x: end.x, y: end.y)
            var route: WireRoute?
            if w.reversed {
                if let n = level[w.fromNode], let s = model.schema(for: n), let b = NodeLayout.inputPoint(n, s, w.fromOutput) {
                    route = WireRouter.route(from: e, to: b, style: wireStyle)
                }
            } else if let n = level[w.fromNode], let s = model.schema(for: n), let a = NodeLayout.outputPoint(n, s, w.fromOutput) {
                route = WireRouter.route(from: a, to: e, style: wireStyle)
            }
            if let route {
                ctx.stroke(Path(route.cgPath()).applying(t), with: .color(w.type.color),
                           style: StrokeStyle(lineWidth: base, lineCap: .round, lineJoin: .round, dash: [6, 4]))
            }
        }
    }

    func wireLabel(_ w: WireInfo, _ level: Graph) -> some View {
        func title(_ id: String) -> String {
            guard let n = level[id] else { return id }
            return n.title ?? model.schema(for: n)?.displayName ?? id
        }
        return HStack(spacing: 6) {
            Text(w.type.rawValue.uppercased()).font(.mg(size: 10, weight: .bold)).foregroundStyle(w.type.color)
            Text("\(title(w.source)).\(w.output) → \(title(w.target)).\(w.input)").font(.mg(size: 11))
        }
        .padding(.horizontal, 8).padding(.vertical, 4)
        .mgGlass(.floatingPanel, in: Capsule())
        .fixedSize()
    }

    func unknownNode(_ node: GraphNode) -> some View {
        Label("Unknown node type “\(node.type)”", systemImage: "questionmark.square.dashed")
            .padding(10)
            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.destructive.opacity(0.18)))
            .scaleEffect(zoom, anchor: .topLeading)
    }

    // MARK: overlays

    /// Breadcrumbs while inside a subgraph; a hint when the level is empty.
    @ViewBuilder var topBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !model.editingPath.isEmpty {
                HStack(spacing: 6) {
                    Button { model.exitSubgraph() } label: { Image(systemName: "chevron.left") }
                        .mgHelp("Back (Esc)")
                    Button("Graph") { model.exitSubgraph(to: 0) }.buttonStyle(.mg(.borderless))
                    ForEach(Array(model.editingTitles.enumerated()), id: \.offset) { i, title in
                        Image(systemName: "chevron.right").font(.caption2).foregroundStyle(MG.secondaryLabel)
                        if i == model.editingTitles.count - 1 {
                            Text(title).fontWeight(.semibold)
                        } else {
                            Button(title) { model.exitSubgraph(to: i + 1) }.buttonStyle(.mg(.borderless))
                        }
                    }
                }
                .font(.mgCallout)
                .buttonStyle(.mg(.icon))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)
                .mgGlass(.floatingPanel, in: Capsule())
            }
            if model.editorGraph.nodes.isEmpty {
                Text("Double-click or press ⌘K to add a node.")
                    .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                    .padding(8).mgGlass(.floatingPanel, radius: MGRadius.lg)
            }
        }
    }

    var zoomControls: some View {
        HStack(spacing: 2) {
            Button { setZoom(zoom / 1.2, around: CGPoint(x: viewSize.width / 2, y: viewSize.height / 2)) } label: {
                Image(systemName: "minus")
            }
            Text("\(Int((zoom * 100).rounded()))%").font(.mgSubheadline.monospacedDigit()).frame(width: 44)
            Button { setZoom(zoom * 1.2, around: CGPoint(x: viewSize.width / 2, y: viewSize.height / 2)) } label: {
                Image(systemName: "plus")
            }
            Divider().frame(height: 16)
            Button { fitToContent() } label: { Image(systemName: "arrow.up.left.and.arrow.down.right") }
                .mgHelp("Fit graph to window")
            Button { model.arrange(selectionOnly: model.selection.count > 1, undoManager: undoManager) } label: {
                Image(systemName: "rectangle.3.group")
            }
            .mgHelp(model.selection.count > 1 ? "Arrange selection (⇧⌘L)" : "Arrange graph (⇧⌘L)")
            Button { showMinimap.toggle() } label: { Image(systemName: showMinimap ? "map.fill" : "map") }
                .mgHelp(showMinimap ? "Hide minimap" : "Show minimap")
            Button {
                model.pendingWire = nil
                model.paletteLocation = nil
                model.showPalette = true
            } label: { Image(systemName: "plus.square.on.square") }
                .mgHelp("Add node (⌘K or double-click)")
        }
        .buttonStyle(.mg(.icon))
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .mgGlass(.floatingPanel, in: Capsule())
    }

    /// The minimap, when it's on and the graph doesn't already fit in the editor.
    @ViewBuilder var minimap: some View {
        if showMinimap, let content = contentBounds, viewSize.width > 0 {
            let visible = CGRect(x: -pan.width / zoom, y: -pan.height / zoom, width: viewSize.width / zoom,
                                 height: viewSize.height / zoom)
            if !visible.contains(content) {
                let level = model.editorGraph
                MinimapView(
                    nodes: level.nodes.compactMap { n in
                        model.frame(of: n).map { MinimapView.Item(frame: $0, color: model.schema(for: n)?.tint ?? .gray,
                                                                  selected: model.selection.contains(n.id)) }
                    },
                    groups: (level.groups ?? []).map { MinimapView.Item(frame: CGRect(x: $0.x, y: $0.y, width: $0.width, height: $0.height),
                                                                        color: $0.tint, selected: false) },
                    content: content, visible: visible,
                    moveTo: { c in pan = CGSize(width: viewSize.width / 2 - c.x * zoom, height: viewSize.height / 2 - c.y * zoom) })
            }
        }
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
            ctx.fill(Path(CGRect(origin: .zero, size: size)), with: .color(MG.contentBackground))
            let spacing = GraphEditorView.grid * zoom
            guard spacing > 6 else { return }
            let dot = max(1, 1.6 * zoom)
            let color = MG.tertiaryLabel.opacity(0.55)
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

/// Node search. With a pending wire it lists only nodes that can connect to it, naming the socket that would.
struct NodePalette: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    @State private var query = ""
    var wire: AppModel.PendingWire? = nil
    let add: (String) -> Void

    var matches: [(schema: NodeSchema, socket: String?)] {
        let q = query.lowercased()
        return model.schemas.compactMap { s in
            guard q.isEmpty || s.displayName.lowercased().contains(q) || s.type.lowercased().contains(q)
                || s.category.lowercased().contains(q) else { return nil }
            guard let wire else { return (s, nil) }
            guard s.type != Graph.rerouteType, let socket = AppModel.compatibleSocket(s, for: wire) else { return nil }
            return (s, socket)
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                if let wire {
                    Text(wire.type.rawValue.uppercased())
                        .font(.mg(size: 10, weight: .bold)).foregroundStyle(wire.type.color)
                        .padding(.horizontal, 6).padding(.vertical, 3)
                        .background(wire.type.color.opacity(0.15), in: Capsule())
                        .mgHelp(wire.fromOutput ? "Nodes that take this" : "Nodes that can feed this")
                }
                TextField(wire == nil ? "Search nodes" : (wire!.fromOutput ? "Connect to…" : "Feed from…"), text: $query)
                    .textFieldStyle(.plain)
                    .font(.mgTitle3)
                    .onSubmit { if let first = matches.first { choose(first.schema.type) } }
            }
            .padding(12)
            Divider()
            List(matches, id: \.schema.type) { m in
                Button { choose(m.schema.type) } label: {
                    HStack {
                        Circle().fill(m.schema.tint).frame(width: 8, height: 8)
                        Text(m.schema.displayName)
                        if let socket = m.socket {
                            Text(wire?.fromOutput == true ? "→ \(socket)" : "\(socket) →")
                                .font(.mgSubheadline).foregroundStyle(wire?.type.color ?? .secondary)
                        }
                        Spacer()
                        Text(m.schema.category).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
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
