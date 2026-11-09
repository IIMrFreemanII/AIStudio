import GraphRuntime
import SwiftUI

/// Geometry of a node card in graph coordinates. Node views, wires and socket hit-testing all use this, so they
/// always agree on where a socket is.
enum NodeLayout {
    static let width: CGFloat = 250
    static let header: CGFloat = 30
    static let row: CGFloat = 26
    static let bottomPadding: CGFloat = 8
    /// A Reroute is a small pill: input socket on the left, output on the right.
    static let rerouteSize = CGSize(width: 44, height: 22)
    static let noteHeight: CGFloat = 150
    /// A collapsed node is just its title bar, this wide; all its sockets meet at the middle of its ends.
    static let collapsedWidth: CGFloat = 180

    static func isReroute(_ node: GraphNode) -> Bool { node.type == Graph.rerouteType }
    static func isNote(_ node: GraphNode) -> Bool { node.type == Graph.noteType }

    struct Rows {
        var sockets: [InputSpec]  // inputs drawn as sockets on the left edge
        var outputs: [OutputSpec]
        var widgets: [InputSpec]  // literal inputs edited inline
        var socketRowCount: Int { max(sockets.count, outputs.count) }
    }

    static func isLinked(_ node: GraphNode, _ name: String) -> Bool {
        if case .link? = node.inputs[name] { return true }
        return false
    }

    static func rows(_ node: GraphNode, _ schema: NodeSchema) -> Rows {
        Rows(
            sockets: schema.inputs.filter { !$0.type.isScalar || isLinked(node, $0.name) },
            outputs: schema.outputs,
            widgets: schema.inputs.filter { $0.type.isScalar && !isLinked(node, $0.name) })
    }

    static func widgetHeight(_ spec: InputSpec) -> CGFloat {
        if case .text(multiline: true)? = spec.widget { return row * 3 }
        return row
    }

    /// Output nodes that show their result inside the card.
    static let resultTypes: [String: MediaKind] = [
        "SaveImage": .image, "PreviewImage": .image, "SaveVideo": .video, "PreviewVideo": .video,
        "SaveAudio": .audio, "PreviewAudio": .audio,
    ]
    static let resultHeight: CGFloat = 170

    static func showsResult(_ schema: NodeSchema) -> Bool { resultTypes[schema.type] != nil }
    static func resultKind(_ schema: NodeSchema) -> MediaKind { resultTypes[schema.type] ?? .image }

    static func frameWidth(_ node: GraphNode) -> CGFloat {
        if isReroute(node) { return rerouteSize.width }
        return node.isCollapsed && !isNote(node) ? collapsedWidth : width
    }

    static func height(_ node: GraphNode, _ schema: NodeSchema) -> CGFloat {
        if isReroute(node) { return rerouteSize.height }
        if isNote(node) { return noteHeight }
        if node.isCollapsed { return header }
        let r = rows(node, schema)
        return header + CGFloat(r.socketRowCount) * row + r.widgets.map(widgetHeight).reduce(0, +)
            + (showsResult(schema) ? resultHeight + 4 : 0) + bottomPadding
    }

    static func origin(_ node: GraphNode) -> CGPoint {
        CGPoint(x: node.position?.x ?? 0, y: node.position?.y ?? 0)
    }

    static func inputPoint(_ node: GraphNode, _ schema: NodeSchema, _ name: String) -> CGPoint? {
        if isReroute(node) { let o = origin(node); return CGPoint(x: o.x, y: o.y + rerouteSize.height / 2) }
        guard let i = rows(node, schema).sockets.firstIndex(where: { $0.name == name }) else { return nil }
        let o = origin(node)
        if node.isCollapsed { return CGPoint(x: o.x, y: o.y + header / 2) }
        return CGPoint(x: o.x, y: o.y + header + row * (CGFloat(i) + 0.5))
    }

    static func outputPoint(_ node: GraphNode, _ schema: NodeSchema, _ name: String) -> CGPoint? {
        if isReroute(node) {
            let o = origin(node)
            return CGPoint(x: o.x + rerouteSize.width, y: o.y + rerouteSize.height / 2)
        }
        guard let i = schema.outputs.firstIndex(where: { $0.name == name }) else { return nil }
        let o = origin(node)
        if node.isCollapsed { return CGPoint(x: o.x + collapsedWidth, y: o.y + header / 2) }
        return CGPoint(x: o.x + width, y: o.y + header + row * (CGFloat(i) + 0.5))
    }

    static func frame(_ node: GraphNode, _ schema: NodeSchema) -> CGRect {
        CGRect(origin: origin(node), size: CGSize(width: frameWidth(node), height: height(node, schema)))
    }

    /// Row index of an input socket (for spreading parallel wire runs).
    static func socketIndex(_ node: GraphNode, _ schema: NodeSchema, _ name: String) -> Int {
        rows(node, schema).sockets.firstIndex { $0.name == name } ?? 0
    }
}

extension SocketType {
    var color: Color {
        switch self {
        case .model: return Color(red: 0.70, green: 0.58, blue: 0.95)
        case .conditioning: return Color(red: 1.00, green: 0.68, blue: 0.22)
        case .latent: return Color(red: 0.98, green: 0.47, blue: 0.80)
        case .image: return Color(red: 0.33, green: 0.68, blue: 0.98)
        case .mask: return Color(red: 0.40, green: 0.80, blue: 0.45)
        case .string: return Color(red: 0.55, green: 0.80, blue: 0.70)
        case .video: return Color(red: 0.95, green: 0.45, blue: 0.35)
        case .audio: return Color(red: 0.30, green: 0.78, blue: 0.85)
        default: return Color.gray
        }
    }
}

extension NodeSchema {
    var tint: Color {
        switch category {
        case "loaders": return SocketType.model.color
        case "conditioning": return SocketType.conditioning.color
        case "latent": return SocketType.latent.color
        case "sampling": return Color(red: 0.45, green: 0.55, blue: 1.0)
        case "image": return SocketType.image.color
        case "subgraph": return Color(red: 0.62, green: 0.45, blue: 0.95)
        case "utils": return Color(red: 0.95, green: 0.80, blue: 0.30)
        case "video": return SocketType.video.color
        case "audio": return SocketType.audio.color
        default: return .gray
        }
    }
}
