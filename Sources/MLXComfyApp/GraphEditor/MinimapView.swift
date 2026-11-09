import SwiftUI

/// A small overview of the editing level: node and group rectangles, and the part the editor shows. Click or drag to
/// move the view there.
struct MinimapView: View {
    struct Item {
        var frame: CGRect
        var color: Color
        var selected: Bool
    }

    let nodes: [Item]
    let groups: [Item]
    /// Everything in the graph, and the part in view (graph coordinates).
    let content: CGRect
    let visible: CGRect
    /// Centres the editor on a graph point.
    let moveTo: (CGPoint) -> Void

    static let size = CGSize(width: 200, height: 130)

    var body: some View {
        let world = content.union(visible).insetBy(dx: -40, dy: -40)
        let scale = min(Self.size.width / world.width, Self.size.height / world.height)
        let offset = CGPoint(x: (Self.size.width - world.width * scale) / 2 - world.minX * scale,
                             y: (Self.size.height - world.height * scale) / 2 - world.minY * scale)
        func map(_ r: CGRect) -> CGRect {
            CGRect(x: r.minX * scale + offset.x, y: r.minY * scale + offset.y, width: r.width * scale, height: r.height * scale)
        }
        func unmap(_ p: CGPoint) -> CGPoint { CGPoint(x: (p.x - offset.x) / scale, y: (p.y - offset.y) / scale) }
        return Canvas { ctx, _ in
            for g in groups {
                ctx.fill(Path(roundedRect: map(g.frame), cornerRadius: 2), with: .color(g.color.opacity(0.18)))
            }
            for n in nodes {
                let r = map(n.frame)
                ctx.fill(Path(roundedRect: r, cornerRadius: 1.5), with: .color(n.color.opacity(n.selected ? 1 : 0.7)))
                if n.selected { ctx.stroke(Path(roundedRect: r.insetBy(dx: -1, dy: -1), cornerRadius: 2), with: .color(MG.accent)) }
            }
            let v = map(visible)
            ctx.fill(Path(roundedRect: v, cornerRadius: 3), with: .color(MG.selection))
            ctx.stroke(Path(roundedRect: v, cornerRadius: 3), with: .color(MG.accent), lineWidth: 1)
        }
        .frame(width: Self.size.width, height: Self.size.height)
        .mgGlass(.floatingPanel, radius: MGRadius.lg)
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0).onChanged { v in moveTo(unmap(v.location)) })
        .help("Click or drag to move around the graph")
    }
}
