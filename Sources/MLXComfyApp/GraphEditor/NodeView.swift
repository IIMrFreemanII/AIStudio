import AppKit
import GraphRuntime
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
    var menu: NodeMenu = NodeMenu()
}

/// Context-menu commands for a node (nil = not offered).
struct NodeMenu {
    var open: (() -> Void)? = nil
    var expand: (() -> Void)? = nil
    var duplicate: (() -> Void)? = nil
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

    var body: some View {
        let rows = NodeLayout.rows(node, schema)
        VStack(spacing: 0) {
            header
            ForEach(0..<rows.socketRowCount, id: \.self) { i in
                HStack(spacing: 0) {
                    if i < rows.sockets.count { inputSocket(rows.sockets[i]) }
                    Spacer(minLength: 4)
                    if i < rows.outputs.count { outputSocket(rows.outputs[i]) }
                }
                .frame(height: NodeLayout.row)
            }
            ForEach(rows.widgets, id: \.name) { spec in
                widget(spec)
                    .frame(height: NodeLayout.widgetHeight(spec))
                    .padding(.horizontal, 10)
            }
            Spacer(minLength: 0)
        }
        .frame(width: NodeLayout.width, height: NodeLayout.height(node, schema), alignment: .top)
        .background(RoundedRectangle(cornerRadius: 9).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 9).strokeBorder(borderColor, lineWidth: borderWidth))
        .shadow(color: .black.opacity(selected ? 0.35 : 0.2), radius: selected ? 10 : 5, y: 2)
        .onTapGesture { actions.select(NSEvent.modifierFlags.contains(.shift) || NSEvent.modifierFlags.contains(.command)) }
        .contextMenu { NodeMenuItems(menu: actions.menu) }
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

    var header: some View {
        HStack(spacing: 6) {
            Circle().fill(schema.tint).frame(width: 8, height: 8)
            if node.type == Graph.subgraphType {
                Image(systemName: "square.stack.3d.up").font(.system(size: 10)).foregroundStyle(schema.tint)
            }
            Text(node.title ?? schema.displayName).font(.system(size: 12, weight: .semibold)).lineLimit(1)
            Spacer(minLength: 4)
            statusBadge
        }
        .padding(.horizontal, 10)
        .frame(height: NodeLayout.header)
        .background(
            UnevenRoundedRectangle(topLeadingRadius: 9, topTrailingRadius: 9)
                .fill(schema.tint.opacity(0.22)))
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
                .onChanged { actions.dragChanged($0.translation) }
                .onEnded { _ in actions.dragEnded() })
        .onTapGesture(count: 2) { actions.open?() }
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
            Text(boundInputs[spec.name].map { "\(spec.name) ← \($0)" } ?? spec.name)
                .font(.system(size: 11)).foregroundStyle(boundInputs[spec.name] != nil ? schema.tint : .secondary)
                .lineLimit(1).offset(x: -11)
        }
    }

    func outputSocket(_ spec: OutputSpec) -> some View {
        HStack(spacing: 2) {
            Text(exposedOutputs[spec.name].map { "\($0) ← \(spec.name)" } ?? spec.name)
                .font(.system(size: 11)).foregroundStyle(exposedOutputs[spec.name] != nil ? schema.tint : .secondary)
                .lineLimit(1).offset(x: 11)
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
        if let duplicate = menu.duplicate { Button("Duplicate", action: duplicate) }
        if let group = menu.group { Button("Group Selection", action: group) }
        if let collapse = menu.collapse { Button("Collapse to Subgraph", action: collapse) }
        if let delete = menu.delete {
            Divider()
            Button("Delete", role: .destructive, action: delete)
        }
    }
}
