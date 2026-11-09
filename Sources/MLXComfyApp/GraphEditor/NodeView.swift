import AppKit
import GraphRuntime
import MLXComfyLibrary
import SwiftUI

/// Callbacks a node card raises; locations are in the editor's "canvas" (view) coordinate space.
struct NodeActions {
    var select: (_ extend: Bool) -> Void
    var dragChanged: (_ translation: CGSize) -> Void
    var dragEnded: () -> Void
    var outputDragChanged: (_ output: String, _ location: CGPoint) -> Void
    var outputDragEnded: (_ output: String, _ location: CGPoint) -> Void
    var inputDragChanged: (_ input: String, _ location: CGPoint) -> Void
    var inputDragEnded: (_ input: String, _ location: CGPoint) -> Void
    /// Double-click on the header (subgraph instances: enter it).
    var open: (() -> Void)? = nil
    /// Double-click on a result shown in the node.
    var openResult: ((NodeResult) -> Void)? = nil
    /// The header chevron (nil for nodes that can't collapse).
    var toggleCollapse: (() -> Void)? = nil
    /// Double-click on a hidden model wire's chip: show the node it comes from.
    var revealSource: ((_ input: String) -> Void)? = nil
    var menu: NodeMenu = NodeMenu()
}

/// Context-menu commands for a node (nil = not offered).
struct NodeMenu {
    var open: (() -> Void)? = nil
    var expand: (() -> Void)? = nil
    var toggleCollapse: (() -> Void)? = nil
    var isCollapsed = false
    var copy: (() -> Void)? = nil
    var duplicate: (() -> Void)? = nil
    /// With several nodes selected.
    var arrange: (() -> Void)? = nil
    var align: ((AlignMode) -> Void)? = nil
    var alignCount = 0
    var group: (() -> Void)? = nil
    var collapse: (() -> Void)? = nil
    var delete: (() -> Void)? = nil
}

struct NodeView: View {
    let node: GraphNode
    let schema: NodeSchema
    let state: NodeRunState?
    let selected: Bool
    let bindings: LiteralBindings
    let actions: NodeActions
    /// Inside a subgraph: inputs fed from outside (input → port) and outputs exposed (output → port).
    var boundInputs: [String: String] = [:]
    var exposedOutputs: [String: String] = [:]
    /// The node's latest output (Save / Preview nodes), and the live step preview while it samples.
    var result: NodeResult? = nil
    var livePreview: CGImage? = nil
    /// The editor's zoom: zoomed out, widgets turn into plain values and titles grow so they stay readable.
    var zoom: CGFloat = 1
    /// Model inputs whose wire is hidden: input → the node it comes from.
    var modelSources: [String: String] = [:]
    /// Links in and out (shown on a collapsed node).
    var linkCount = 0

    /// 2: full detail; 1: values as text, no socket labels; 0: just the title (and result).
    var detail: Int { zoom >= 0.6 ? 2 : zoom >= 0.35 ? 1 : 0 }

    var body: some View {
        let rows = NodeLayout.rows(node, schema)
        VStack(spacing: 0) {
            header
            if !node.isCollapsed {
                if detail == 0 {
                    bigTitle(rows)
                } else {
                    ForEach(0..<rows.socketRowCount, id: \.self) { i in
                        HStack(spacing: 0) {
                            if i < rows.sockets.count { inputSocket(rows.sockets[i]) }
                            Spacer(minLength: 4)
                            if i < rows.outputs.count { outputSocket(rows.outputs[i]) }
                        }
                        .frame(height: NodeLayout.row)
                    }
                    ForEach(rows.widgets, id: \.name) { spec in
                        Group {
                            if detail == 2 { widget(spec) } else { plainValue(spec) }
                        }
                        .frame(height: NodeLayout.widgetHeight(spec))
                        .padding(.horizontal, 10)
                    }
                }
                if NodeLayout.showsResult(schema) {
                    NodeResultView(result: result, kind: NodeLayout.resultKind(schema))
                        .frame(height: NodeLayout.resultHeight)
                        .padding(.horizontal, 8)
                        .padding(.top, 4)
                        .onTapGesture(count: 2) { if let result { actions.openResult?(result) } }
                }
                Spacer(minLength: 0)
            }
        }
        .frame(width: NodeLayout.frameWidth(node), height: NodeLayout.height(node, schema), alignment: .top)
        .overlay { if node.isCollapsed { collapsedSockets(rows) } }
        .background(RoundedRectangle(cornerRadius: 9).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(borderColor, lineWidth: borderWidth))
        .shadow(color: .black.opacity(selected ? 0.35 : 0.2), radius: selected ? 10 : 5, y: 2)
        .onTapGesture { actions.select(NSEvent.modifierFlags.contains(.shift) || NSEvent.modifierFlags.contains(.command)) }
        .contextMenu { NodeMenuItems(menu: actions.menu) }
        .overlay(alignment: .top) {
            // The sampler's live preview hangs under the node while it runs, so the node's size doesn't jump.
            if let livePreview, !NodeLayout.showsResult(schema) {
                Image(decorative: livePreview, scale: 1)
                    .resizable().interpolation(.medium).aspectRatio(contentMode: .fit)
                    .frame(maxWidth: NodeLayout.width - 20, maxHeight: 200)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .shadow(radius: 6)
                    .offset(y: NodeLayout.height(node, schema) + 8)
                    .allowsHitTesting(false)
            }
        }
    }

    var borderColor: Color {
        if case .running? = state { return .orange }
        if case .failed? = state { return .red }
        return selected ? .accentColor : Color.primary.opacity(0.12)
    }

    var borderWidth: CGFloat {
        if case .running? = state { return 2.5 }
        return selected ? 2 : 1
    }

    var title: String { node.title ?? schema.displayName }

    var header: some View {
        HStack(spacing: 6) {
            if let toggle = actions.toggleCollapse, detail > 0 {
                Image(systemName: node.isCollapsed ? "chevron.right" : "chevron.down")
                    .font(.system(size: 9, weight: .semibold)).foregroundStyle(.secondary)
                    .frame(width: 12, height: 20)
                    .contentShape(Rectangle())
                    .onTapGesture(perform: toggle)
                    .help(node.isCollapsed ? "Expand (⌥C)" : "Collapse (⌥C)")
            } else {
                Circle().fill(schema.tint).frame(width: 8, height: 8)
            }
            if node.type == Graph.subgraphType {
                Image(systemName: "square.stack.3d.up").font(.system(size: 10)).foregroundStyle(schema.tint)
            }
            Text(title).font(.system(size: detail == 2 ? 12 : min(12 / zoom, 22), weight: .semibold)).lineLimit(1)
                .minimumScaleFactor(0.5)
            Spacer(minLength: 4)
            if node.isCollapsed && linkCount > 0 && !isRunning {
                Text("\(linkCount)").font(.system(size: 10).monospacedDigit()).foregroundStyle(.secondary)
                    .help("\(linkCount) connections")
            }
            if detail > 0 { statusBadge }
        }
        .padding(.horizontal, 10)
        .frame(height: NodeLayout.header)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 9, bottomLeadingRadius: node.isCollapsed ? 9 : 0,
                                   bottomTrailingRadius: node.isCollapsed ? 9 : 0, topTrailingRadius: 9)
                .fill(schema.tint.opacity(detail == 0 ? 0.55 : 0.22)))
        .contentShape(Rectangle())
        .gesture(moveGesture)
        .onTapGesture(count: 2) { actions.open?() }
    }

    var moveGesture: some Gesture {
        DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
            .onChanged { actions.dragChanged($0.translation) }
            .onEnded { _ in actions.dragEnded() }
    }

    var isRunning: Bool {
        if case .running? = state { return true }
        return false
    }

    /// Zoomed far out: the title across the whole card, big enough to read.
    func bigTitle(_ rows: NodeLayout.Rows) -> some View {
        let h = NodeLayout.height(node, schema) - NodeLayout.header - (NodeLayout.showsResult(schema) ? NodeLayout.resultHeight + 4 : 0)
        return Text(title)
            .font(.system(size: min(13 / zoom, 56), weight: .semibold))
            .foregroundStyle(schema.tint)
            .multilineTextAlignment(.center)
            .minimumScaleFactor(0.3)
            .padding(.horizontal, 8)
            .frame(maxWidth: .infinity)
            .frame(height: max(0, h - NodeLayout.bottomPadding))
            .contentShape(Rectangle())
            .gesture(moveGesture)
    }

    /// A collapsed node: one dot at each end for all its links.
    func collapsedSockets(_ rows: NodeLayout.Rows) -> some View {
        HStack {
            if let first = rows.sockets.first {
                socketDot(first.type, filled: rows.sockets.contains { NodeLayout.isLinked(node, $0.name) })
                    .offset(x: -11).allowsHitTesting(false)
            }
            Spacer()
            if let first = rows.outputs.first {
                socketDot(first.type, filled: true).offset(x: 11).allowsHitTesting(false)
            }
        }
    }

    /// Zoomed out: a widget's value as plain text (cheaper than a live field, and readable).
    func plainValue(_ spec: InputSpec) -> some View {
        let value: String = {
            switch bindings.get(node.id, spec.name) {
            case .int(let i)?: return String(i)
            case .double(let d)?: return d.formatted(.number.precision(.fractionLength(0...3)))
            case .bool(let b)?: return b ? "on" : "off"
            case .string(let s)?: return s.isEmpty ? "—" : s
            default: return "—"
            }
        }()
        return HStack(spacing: 6) {
            Text(spec.name).foregroundStyle(.secondary)
            Spacer(minLength: 4)
            Text(verbatim: value).lineLimit(1).truncationMode(.middle)
        }
        .font(.system(size: 11))
    }

    @ViewBuilder var statusBadge: some View {
        switch state {
        case .running(let step, let total)?:
            HStack(spacing: 4) {
                ProgressView().controlSize(.mini)
                if total > 0 { Text("\(step)/\(total)").font(.system(size: 10).monospacedDigit()) }
            }
        case .done(let s, _)?:
            Text(String(format: "%.1fs", s)).font(.system(size: 10).monospacedDigit()).foregroundStyle(.secondary)
        case .cached?:
            Label("cached", systemImage: "checkmark").font(.system(size: 10)).foregroundStyle(.secondary)
                .labelStyle(.titleAndIcon)
        case .failed?:
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red).font(.system(size: 11))
        default:
            EmptyView()
        }
    }

    func socketDot(_ type: SocketType, filled: Bool) -> some View {
        Circle()
            .fill(filled ? type.color : Color(nsColor: .controlBackgroundColor))
            .overlay(Circle().strokeBorder(type.color, lineWidth: 2))
            .frame(width: 12, height: 12)
            .frame(width: 22, height: 22)
            .contentShape(Circle())
    }

    func inputSocket(_ spec: InputSpec) -> some View {
        HStack(spacing: 2) {
            socketDot(spec.type, filled: NodeLayout.isLinked(node, spec.name) || boundInputs[spec.name] != nil)
                .offset(x: -11)
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                        .onChanged { actions.inputDragChanged(spec.name, $0.location) }
                        .onEnded { actions.inputDragEnded(spec.name, $0.location) })
                .help("\(spec.name): \(spec.type.rawValue)")
            if let source = modelSources[spec.name], detail == 2 {
                // The model wire is hidden: name its source instead.
                Text("⟵ \(source)")
                    .font(.system(size: 10, weight: .medium)).foregroundStyle(SocketType.model.color)
                    .lineLimit(1)
                    .padding(.horizontal, 6).padding(.vertical, 2)
                    .background(SocketType.model.color.opacity(0.15), in: Capsule())
                    .offset(x: -11)
                    .onTapGesture(count: 2) { actions.revealSource?(spec.name) }
                    .help("\(spec.name) from \(source) — the wire is hidden (View › Show Model Wires). Double-click to show it.")
            } else if detail == 2 {
                Text(boundInputs[spec.name].map { "\(spec.name) ← \($0)" } ?? spec.name)
                    .font(.system(size: 11)).foregroundStyle(boundInputs[spec.name] != nil ? schema.tint : .secondary)
                    .lineLimit(1).offset(x: -11)
            }
        }
    }

    func outputSocket(_ spec: OutputSpec) -> some View {
        HStack(spacing: 2) {
            if detail == 2 {
                Text(exposedOutputs[spec.name].map { "\($0) ← \(spec.name)" } ?? spec.name)
                    .font(.system(size: 11)).foregroundStyle(exposedOutputs[spec.name] != nil ? schema.tint : .secondary)
                    .lineLimit(1).offset(x: 11)
            }
            socketDot(spec.type, filled: true)
                .offset(x: 11)
                .gesture(
                    DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                        .onChanged { actions.outputDragChanged(spec.name, $0.location) }
                        .onEnded { actions.outputDragEnded(spec.name, $0.location) })
                .help("\(spec.name): \(spec.type.rawValue) — drag to an input")
        }
    }

    @ViewBuilder
    func widget(_ spec: InputSpec) -> some View {
        let font = Font.system(size: 11)
        switch (spec.type, spec.widget) {
        case (.string, .text(multiline: true)?):
            Text(bindings.string(node.id, spec.name).wrappedValue.isEmpty
                 ? "Empty \(spec.name)" : bindings.string(node.id, spec.name).wrappedValue)
                .font(font)
                .foregroundStyle(.primary.opacity(0.85))
                .lineLimit(3)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 5).fill(Color.primary.opacity(0.06)))
                .padding(.vertical, 2)
                .help("Edit in the inspector")
        case (.bool, _):
            Toggle(spec.name, isOn: bindings.bool(node.id, spec.name)).font(font).toggleStyle(.checkbox)
                .frame(maxWidth: .infinity, alignment: .leading)
        default:
            HStack(spacing: 6) {
                Text(spec.name).font(font).foregroundStyle(.secondary).lineLimit(1)
                Spacer(minLength: 4)
                Group {
                    switch spec.type {
                    case .int:
                        TextField("", value: bindings.int(node.id, spec.name), format: .number.grouping(.never))
                    case .float:
                        TextField("", value: bindings.double(node.id, spec.name), format: .number)
                    default:
                        TextField("", text: bindings.string(node.id, spec.name))
                    }
                }
                .font(font)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .frame(maxWidth: spec.type == .string ? 150 : 90)
            }
        }
    }
}

struct NodeMenuItems: View {
    let menu: NodeMenu

    var body: some View {
        if let open = menu.open { Button("Open Subgraph", action: open) }
        if let expand = menu.expand { Button("Expand Subgraph", action: expand) }
        if menu.open != nil || menu.expand != nil { Divider() }
        if let toggle = menu.toggleCollapse { Button(menu.isCollapsed ? "Expand Node" : "Collapse Node", action: toggle) }
        if let copy = menu.copy { Button("Copy", action: copy) }
        if let duplicate = menu.duplicate { Button("Duplicate", action: duplicate) }
        if let arrange = menu.arrange { Button("Arrange Selection", action: arrange) }
        if let align = menu.align {
            Menu("Align") {
                ForEach(AlignMode.allCases) { mode in
                    Button(mode.title) { align(mode) }.disabled(menu.alignCount < mode.minimumCount)
                    if mode == .right || mode == .bottom { Divider() }
                }
            }
        }
        if let group = menu.group { Button("Group Selection", action: group) }
        if let collapse = menu.collapse { Button("Collapse to Subgraph", action: collapse) }
        if let delete = menu.delete {
            Divider()
            Button("Delete", role: .destructive, action: delete)
        }
    }
}

/// The result area of a Save / Preview node: the image, a video poster with a play badge, or a waveform.
struct NodeResultView: View {
    let result: NodeResult?
    let kind: MediaKind

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 6).fill(Color.primary.opacity(0.05))
            if let result {
                switch result.kind {
                case .image:
                    if let img = result.image {
                        Image(decorative: img, scale: 1).resizable().interpolation(.medium).aspectRatio(contentMode: .fit)
                    }
                case .video:
                    if let img = result.image {
                        Image(decorative: img, scale: 1).resizable().interpolation(.medium).aspectRatio(contentMode: .fit)
                    }
                    Image(systemName: "play.fill").font(.system(size: 14)).foregroundStyle(.white)
                        .padding(9).background(.black.opacity(0.55), in: Circle())
                case .audio:
                    if let url = result.url { AudioThumb(url: url).padding(6) }
                }
            } else {
                Image(systemName: kind == .video ? "film" : kind == .audio ? "waveform" : "photo")
                    .font(.system(size: 22, weight: .light)).foregroundStyle(.tertiary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .help(result?.url?.lastPathComponent.appending(" — double-click to open") ?? "The result shows here after a run")
    }
}
