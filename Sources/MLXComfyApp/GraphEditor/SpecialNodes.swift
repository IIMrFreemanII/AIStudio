import AppKit
import GraphRuntime
import SwiftUI

/// A Reroute: a small pill coloured by the type flowing through it. Drag the middle to move it, the left dot to
/// pick up the incoming link, the right dot to draw a new one.
struct RerouteView: View {
    let node: GraphNode
    let type: SocketType
    let selected: Bool
    let actions: NodeActions

    var body: some View {
        let size = NodeLayout.rerouteSize
        HStack(spacing: 0) {
            dot(filled: NodeLayout.isLinked(node, "in"))
                .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                    .onChanged { actions.inputDragChanged("in", $0.location) }
                    .onEnded { actions.inputDragEnded("in", $0.location) })
            Spacer(minLength: 0)
            dot(filled: true)
                .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                    .onChanged { actions.outputDragChanged("out", $0.location) }
                    .onEnded { actions.outputDragEnded("out", $0.location) })
        }
        .frame(width: size.width, height: size.height)
        .background(Capsule().fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(Capsule().strokeBorder(selected ? Color.accentColor : type.color.opacity(0.8), lineWidth: selected ? 2 : 1.5))
        .overlay {
            // Grip in the middle: drag to move.
            Capsule().fill(Color.clear).frame(width: 14, height: size.height).contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
                    .onChanged { actions.dragChanged($0.translation) }
                    .onEnded { _ in actions.dragEnded() })
        }
        .shadow(color: .black.opacity(0.2), radius: 3, y: 1)
        .onTapGesture { actions.select(NSEvent.modifierFlags.contains(.shift) || NSEvent.modifierFlags.contains(.command)) }
        .contextMenu { NodeMenuItems(menu: actions.menu) }
    }

    func dot(filled: Bool) -> some View {
        Circle().fill(filled ? type.color : Color(nsColor: .controlBackgroundColor))
            .overlay(Circle().strokeBorder(type.color, lineWidth: 2))
            .frame(width: 10, height: 10)
            .frame(width: 18, height: NodeLayout.rerouteSize.height)
            .contentShape(Rectangle())
    }
}

/// A sticky note, edited in place.
struct NoteView: View {
    let node: GraphNode
    let selected: Bool
    let text: Binding<String>
    let actions: NodeActions

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "note.text").font(.system(size: 10))
                Text(node.title ?? "Note").font(.system(size: 11, weight: .semibold))
                Spacer()
            }
            .foregroundStyle(Color.black.opacity(0.65))
            .padding(.horizontal, 10)
            .frame(height: 24)
            .contentShape(Rectangle())
            .gesture(DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
                .onChanged { actions.dragChanged($0.translation) }
                .onEnded { _ in actions.dragEnded() })
            TextEditor(text: text)
                .font(.system(size: 12))
                .foregroundStyle(Color.black.opacity(0.85))
                .scrollContentBackground(.hidden)
                .padding(.horizontal, 6)
                .padding(.bottom, 6)
        }
        .frame(width: NodeLayout.width, height: NodeLayout.noteHeight)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color(red: 1, green: 0.93, blue: 0.62)))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(selected ? Color.accentColor : .black.opacity(0.1),
                                                               lineWidth: selected ? 2 : 1))
        .shadow(color: .black.opacity(0.18), radius: 4, y: 2)
        .onTapGesture { actions.select(NSEvent.modifierFlags.contains(.shift) || NSEvent.modifierFlags.contains(.command)) }
        .contextMenu { NodeMenuItems(menu: actions.menu) }
    }
}

extension GraphGroup {
    var tint: Color {
        switch color {
        case "green": return Color(red: 0.35, green: 0.75, blue: 0.45)
        case "yellow": return Color(red: 0.95, green: 0.78, blue: 0.25)
        case "red": return Color(red: 0.92, green: 0.38, blue: 0.35)
        case "purple": return Color(red: 0.62, green: 0.45, blue: 0.95)
        case "gray": return .gray
        default: return Color(red: 0.35, green: 0.58, blue: 0.95)
        }
    }
}

/// A group frame in view coordinates: drag the title bar to move it with its nodes, the corner to resize.
struct GroupView: View {
    let group: GraphGroup
    let size: CGSize  // view points
    let zoom: CGFloat
    let selected: Bool
    var select: () -> Void
    var moveChanged: (CGSize) -> Void
    var moveEnded: () -> Void
    var resizeChanged: (CGSize) -> Void
    var resizeEnded: () -> Void
    var setColor: (String) -> Void
    var delete: (_ withNodes: Bool) -> Void

    var body: some View {
        let bar = max(18, 28 * zoom)
        ZStack(alignment: .topLeading) {
            // The body doesn't take hits, so panning and rubber-band selection work over groups.
            RoundedRectangle(cornerRadius: 8 * zoom).fill(group.tint.opacity(0.10)).allowsHitTesting(false)
            RoundedRectangle(cornerRadius: 8 * zoom)
                .strokeBorder(selected ? Color.accentColor : group.tint.opacity(0.55), lineWidth: selected ? 2 : 1)
                .allowsHitTesting(false)
            HStack {
                Text(group.title).font(.system(size: max(10, 14 * zoom), weight: .semibold)).lineLimit(1)
                Spacer()
            }
            .padding(.horizontal, 10 * zoom)
            .frame(width: size.width, height: bar)
            .background(UnevenRoundedRectangle(topLeadingRadius: 8 * zoom, topTrailingRadius: 8 * zoom)
                .fill(group.tint.opacity(0.30)))
            .contentShape(Rectangle())
            .onTapGesture(perform: select)
            .gesture(DragGesture(minimumDistance: 2, coordinateSpace: .named("canvas"))
                .onChanged { moveChanged($0.translation) }
                .onEnded { _ in moveEnded() })
            .contextMenu {
                Menu("Color") {
                    ForEach(AppModel.groupColors, id: \.self) { c in Button(c.capitalized) { setColor(c) } }
                }
                Divider()
                Button("Delete Group") { delete(false) }
                Button("Delete Group and Its Nodes", role: .destructive) { delete(true) }
            }
            // Resize handle.
            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .font(.system(size: 9)).foregroundStyle(group.tint)
                .frame(width: 18, height: 18)
                .contentShape(Rectangle())
                .offset(x: size.width - 18, y: size.height - 18)
                .gesture(DragGesture(minimumDistance: 1, coordinateSpace: .named("canvas"))
                    .onChanged { resizeChanged($0.translation) }
                    .onEnded { _ in resizeEnded() })
        }
        .frame(width: size.width, height: size.height)
    }
}
