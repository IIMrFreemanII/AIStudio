import CoreGraphics
import Foundation

/// Align and distribute commands for selected nodes.
public enum AlignMode: String, CaseIterable, Identifiable, Sendable {
    case left, centerX, right, top, centerY, bottom, distributeX, distributeY
    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .left: return "Align Left Edges"
        case .centerX: return "Align Horizontal Centers"
        case .right: return "Align Right Edges"
        case .top: return "Align Top Edges"
        case .centerY: return "Align Vertical Centers"
        case .bottom: return "Align Bottom Edges"
        case .distributeX: return "Distribute Horizontally"
        case .distributeY: return "Distribute Vertically"
        }
    }

    /// How many nodes the command needs.
    public var minimumCount: Int { self == .distributeX || self == .distributeY ? 3 : 2 }
}

public enum Alignment {
    /// New top-left corners for `frames` (only ids that move are returned).
    public static func apply(_ mode: AlignMode, to frames: [String: CGRect]) -> [String: CGPoint] {
        guard frames.count >= mode.minimumCount else { return [:] }
        let all = frames.values
        var out: [String: CGPoint] = [:]
        switch mode {
        case .left:
            let x = all.map(\.minX).min()!
            for (id, f) in frames { out[id] = CGPoint(x: x, y: f.minY) }
        case .right:
            let x = all.map(\.maxX).max()!
            for (id, f) in frames { out[id] = CGPoint(x: x - f.width, y: f.minY) }
        case .centerX:
            let x = (all.map(\.minX).min()! + all.map(\.maxX).max()!) / 2
            for (id, f) in frames { out[id] = CGPoint(x: x - f.width / 2, y: f.minY) }
        case .top:
            let y = all.map(\.minY).min()!
            for (id, f) in frames { out[id] = CGPoint(x: f.minX, y: y) }
        case .bottom:
            let y = all.map(\.maxY).max()!
            for (id, f) in frames { out[id] = CGPoint(x: f.minX, y: y - f.height) }
        case .centerY:
            let y = (all.map(\.minY).min()! + all.map(\.maxY).max()!) / 2
            for (id, f) in frames { out[id] = CGPoint(x: f.minX, y: y - f.height / 2) }
        case .distributeX:
            // Equal gaps between neighbours; the first and last stay put.
            let sorted = frames.sorted { ($0.value.minX, $0.key) < ($1.value.minX, $1.key) }
            let span = sorted.last!.value.maxX - sorted.first!.value.minX
            let gap = (span - sorted.map(\.value.width).reduce(0, +)) / CGFloat(sorted.count - 1)
            var x = sorted.first!.value.minX
            for (id, f) in sorted {
                out[id] = CGPoint(x: x, y: f.minY)
                x += f.width + gap
            }
        case .distributeY:
            let sorted = frames.sorted { ($0.value.minY, $0.key) < ($1.value.minY, $1.key) }
            let span = sorted.last!.value.maxY - sorted.first!.value.minY
            let gap = (span - sorted.map(\.value.height).reduce(0, +)) / CGFloat(sorted.count - 1)
            var y = sorted.first!.value.minY
            for (id, f) in sorted {
                out[id] = CGPoint(x: f.minX, y: y)
                y += f.height + gap
            }
        }
        return out.filter { frames[$0.key]?.origin != $0.value }
    }

    /// `p` rounded to the nearest grid point.
    public static func snap(_ p: CGPoint, grid: CGFloat = 24) -> CGPoint {
        CGPoint(x: (p.x / grid).rounded() * grid, y: (p.y / grid).rounded() * grid)
    }
}
