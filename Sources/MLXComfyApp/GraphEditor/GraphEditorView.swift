import AppKit
import GraphRuntime
import SwiftUI

/// A link being dragged from an output socket.
struct WireDrag {
    var fromNode: String
    var fromOutput: String
    var type: SocketType
    var end: CGPoint  // view coordinates
}

struct GraphEditorView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager

    @State private var zoom: CGFloat = 0.9
    @State private var pan = CGSize(width: 40, height: 60)
    @State private var panStart: CGSize?
    @State private var zoomStart: CGFloat?
    @State private var dragStartPositions: [String: GraphPoint]?
    @State private var editStartGraph: Graph?
    @State private var wire: WireDrag?
    @State private var marqueeStart: CGPoint?
    @State private var marquee: CGRect?
    @State private var hoverLocation: CGPoint?
    @State private var viewSize: CGSize = .zero
    @State private var scrollMonitor: Any?
    @State private var didInitialFit = false

    var body: some View {
        @Bindable var model = model
        GeometryReader { geo in
            ZStack(alignment: .topLeading) {
                GridBackground(zoom: zoom, pan: pan)
                    .contentShape(Rectangle())
                    .gesture(backgroundDrag)
                    .gesture(SpatialTapGesture(count: 2, coordinateSpace: .named("canvas")).onEnded { v in
                        model.paletteLocation = toGraph(v.location)
                        model.showPalette = true
                    }.exclusively(before: TapGesture().onEnded { model.selection = [] }))

                Canvas { ctx, _ in drawWires(&ctx) }
                    .allowsHitTesting(false)

                ForEach(model.graph.nodes) { node in
                    if let schema = model.schema(for: node) {
                        let p = toView(NodeLayout.origin(node))
                        NodeView(node: node, schema: schema, state: model.nodeStates[node.id],
                                 selected: model.selection.contains(node.id),
                                 bindings: LiteralBindings(model: model, undoManager: undoManager),
                                 actions: actions(for: node))
                            .scaleEffect(zoom, anchor: .topLeading)
                            .offset(x: p.x, y: p.y)
                    } else {
                        unknownNode(node).offset(x: toView(NodeLayout.origin(node)).x, y: toView(NodeLayout.origin(node)).y)
                    }
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
            .overlay(alignment: .bottomTrailing) { zoomControls.padding(12) }
            .overlay(alignment: .bottomLeading) { MiniPreview().padding(12) }
            .overlay(alignment: .bottom) { RunStatusPill().padding(.bottom, 16) }
            .overlay(alignment: .topLeading) { helpHint.padding(12) }
            .onAppear {
                viewSize = geo.size
                fitOnceIfSized()
                installScrollMonitor()
            }
            .onChange(of: geo.size) { _, s in
                viewSize = s
                fitOnceIfSized()
            }
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
        .onDeleteCommand { model.deleteSelection(undoManager: undoManager) }
        .onKeyPress(.escape) { model.selection = []; return .handled }
        .sheet(isPresented: $model.showPalette) {
            NodePalette { type in
                let at = model.paletteLocation ?? toGraph(CGPoint(x: viewSize.width / 2 - 120, y: viewSize.height / 2 - 60))
                model.addNode(type: type, at: at, undoManager: undoManager)
                model.paletteLocation = nil
            }
            .environment(model)
        }
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
        let frames = model.graph.nodes.compactMap { n in model.schema(for: n).map { NodeLayout.frame(n, $0) } }
        guard let first = frames.first, viewSize.width > 0 else { return }
        let box = frames.dropFirst().reduce(first) { $0.union($1) }.insetBy(dx: -40, dy: -40)
        let z = min(max(min(viewSize.width / box.width, viewSize.height / box.height), 0.25), 1.5)
        zoom = z
        pan = CGSize(width: (viewSize.width - box.width * z) / 2 - box.minX * z,
                     height: (viewSize.height - box.height * z) / 2 - box.minY * z)
    }

    // MARK: gestures

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
                    model.selection = Set(model.graph.nodes.filter { n in
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
        NodeActions(
            select: { extend in
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
                    dragStartPositions = Dictionary(uniqueKeysWithValues: model.graph.nodes
                        .filter { model.selection.contains($0.id) }
                        .map { ($0.id, $0.position ?? GraphPoint(x: 0, y: 0)) })
                }
                for (id, start) in dragStartPositions ?? [:] {
                    guard var n = model.graph[id] else { continue }
                    n.position = GraphPoint(x: start.x + Double(t.width / zoom), y: start.y + Double(t.height / zoom))
                    model.graph[id] = n
                }
            },
            dragEnded: {
                if let start = editStartGraph {
                    model.commitLiveEdit(from: start, actionName: "Move", undoManager: undoManager)
                }
                dragStartPositions = nil
                editStartGraph = nil
            },
            outputDragChanged: { output, loc in
                if wire == nil {
                    guard let type = model.schema(for: node)?.output(output)?.type else { return }
                    editStartGraph = model.graph
                    wire = WireDrag(fromNode: node.id, fromOutput: output, type: type, end: loc)
                }
                wire?.end = loc
            },
            outputDragEnded: { _, loc in finishWire(at: loc, actionIfDropped: nil) },
            inputDragChanged: { input, loc in
                if wire == nil {
                    // Dragging a connected input picks the link up from its source.
                    guard case .link(let src, let out)? = model.graph[node.id]?.inputs[input],
                          let type = model.schema(for: node)?.input(input)?.type else { return }
                    editStartGraph = model.graph
                    var n = model.graph[node.id]!
                    n.inputs[input] = nil
                    model.graph[node.id] = n
                    wire = WireDrag(fromNode: src, fromOutput: out, type: type, end: loc)
                }
                wire?.end = loc
            },
            inputDragEnded: { _, loc in finishWire(at: loc, actionIfDropped: "Disconnect") })
    }

    /// Connects the dragged wire to the input under `loc`; otherwise commits `actionIfDropped` (or reverts).
    func finishWire(at loc: CGPoint, actionIfDropped: String?) {
        defer { wire = nil; editStartGraph = nil }
        guard let w = wire, let start = editStartGraph else { return }
        if let (target, input) = inputTarget(at: loc, for: w) {
            var g = model.graph
            var n = g[target]!
            n.inputs[input] = .link(node: w.fromNode, output: w.fromOutput)
            g[target] = n
            if model.wouldCreateCycle(g) {
                NSSound.beep()
                model.graph = start
                return
            }
            model.graph = g
            model.commitLiveEdit(from: start, actionName: "Connect", undoManager: undoManager)
        } else if let action = actionIfDropped {
            model.commitLiveEdit(from: start, actionName: action, undoManager: undoManager)
        } else {
            model.graph = start
        }
    }

    /// The compatible input socket nearest to `loc` (within reach), or the only compatible input of the node under it.
    func inputTarget(at loc: CGPoint, for w: WireDrag) -> (String, String)? {
        var best: (String, String, CGFloat)?
        for n in model.graph.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n) else { continue }
            for spec in s.inputs where spec.type == w.type && !spec.type.isScalar {
                guard let p = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let v = toView(p)
                let d = hypot(v.x - loc.x, v.y - loc.y)
                if d < 22 * max(zoom, 0.6), d < (best?.2 ?? .infinity) { best = (n.id, spec.name, d) }
            }
        }
        if let best { return (best.0, best.1) }
        for n in model.graph.nodes where n.id != w.fromNode {
            guard let s = model.schema(for: n) else { continue }
            let f = NodeLayout.frame(n, s)
            let vf = CGRect(origin: toView(f.origin), size: CGSize(width: f.width * zoom, height: f.height * zoom))
            guard vf.contains(loc) else { continue }
            let compatible = s.inputs.filter { $0.type == w.type && !$0.type.isScalar }
            if compatible.count == 1 { return (n.id, compatible[0].name) }
        }
        return nil
    }

    // MARK: drawing

    func drawWires(_ ctx: inout GraphicsContext) {
        let width = max(1.5, 2.5 * zoom)
        for n in model.graph.nodes {
            guard let s = model.schema(for: n) else { continue }
            for spec in s.inputs {
                guard case .link(let src, let out)? = n.inputs[spec.name], let srcNode = model.graph[src],
                      let srcSchema = model.schema(for: srcNode),
                      let a = NodeLayout.outputPoint(srcNode, srcSchema, out),
                      let b = NodeLayout.inputPoint(n, s, spec.name) else { continue }
                let highlighted = model.selection.contains(n.id) || model.selection.contains(src)
                stroke(&ctx, from: toView(a), to: toView(b), color: spec.type.color.opacity(highlighted ? 1 : 0.75),
                       width: highlighted ? width * 1.5 : width)
            }
        }
        if let w = wire, let srcNode = model.graph[w.fromNode], let srcSchema = model.schema(for: srcNode),
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

    @ViewBuilder var helpHint: some View {
        if model.graph.nodes.isEmpty {
            Text("Double-click or press ⌘K to add a node.")
                .font(.callout).foregroundStyle(.secondary)
                .padding(8).background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    // MARK: scroll wheel

    /// Two-finger scroll pans, ⌘-scroll zooms around the pointer (SwiftUI has no scroll-wheel gesture on macOS).
    func installScrollMonitor() {
        guard scrollMonitor == nil else { return }
        scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard let loc = hoverLocation, model.workspace == .graph, !model.showPalette else { return event }
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

/// Small live result in the corner of the graph; click to open the image workspace.
struct MiniPreview: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let img = model.previewImage ?? model.displayedImage {
            Image(decorative: img, scale: 1)
                .resizable()
                .interpolation(.medium)
                .aspectRatio(contentMode: .fit)
                .frame(maxWidth: 180, maxHeight: 180)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.white.opacity(0.25)))
                .shadow(radius: 8)
                .onTapGesture { model.workspace = .image }
                .help("Show image (⌘1)")
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
