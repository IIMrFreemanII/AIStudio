import CoreGraphics
import Foundation

/// How links between sockets are drawn.
public enum WireStyle: String, CaseIterable, Identifiable, Sendable {
    case rightAngle, curved, straight
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .rightAngle: return "Right-angle"
        case .curved: return "Curved"
        case .straight: return "Straight"
        }
    }
}

/// The shape of one link, in graph coordinates.
public struct WireRoute: Equatable, Sendable {
    public enum Kind: Equatable, Sendable {
        /// Corner points joined by straight runs; corners are rounded with `cornerRadius`.
        case polyline
        /// A chain of cubic Béziers: start, then (control, control, end) triples.
        case bezier
    }

    public var kind: Kind
    public var points: [CGPoint]
    public var cornerRadius: CGFloat

    public init(kind: Kind, points: [CGPoint], cornerRadius: CGFloat = 0) {
        self.kind = kind
        self.points = points
        self.cornerRadius = cornerRadius
    }

    public var start: CGPoint { points.first ?? .zero }
    public var end: CGPoint { points.last ?? .zero }

    /// The drawable path (rounded corners for polylines).
    public func cgPath() -> CGPath {
        let path = CGMutablePath()
        guard let first = points.first else { return path }
        path.move(to: first)
        switch kind {
        case .polyline:
            for i in 1..<points.count {
                let p = points[i]
                if i < points.count - 1, cornerRadius > 0 {
                    let prev = points[i - 1], next = points[i + 1]
                    let r = min(cornerRadius, Self.distance(prev, p) / 2, Self.distance(p, next) / 2)
                    if r > 0.5 { path.addArc(tangent1End: p, tangent2End: next, radius: r) } else { path.addLine(to: p) }
                } else {
                    path.addLine(to: p)
                }
            }
        case .bezier:
            var i = 1
            while i + 2 < points.count {
                path.addCurve(to: points[i + 2], control1: points[i], control2: points[i + 1])
                i += 3
            }
        }
        return path
    }

    /// Points along the route about `step` apart (for hit testing).
    public func sampled(step: CGFloat = 6) -> [CGPoint] {
        guard let first = points.first else { return [] }
        var out = [first]
        switch kind {
        case .polyline:
            for i in 1..<max(points.count, 1) {
                let a = points[i - 1], b = points[i]
                let n = max(1, Int(Self.distance(a, b) / max(step, 1)))
                for k in 1...n {
                    let t = CGFloat(k) / CGFloat(n)
                    out.append(CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t))
                }
            }
        case .bezier:
            var i = 1
            while i + 2 < points.count {
                let p0 = points[i - 1], c1 = points[i], c2 = points[i + 1], p1 = points[i + 2]
                let approx = Self.distance(p0, c1) + Self.distance(c1, c2) + Self.distance(c2, p1)
                let n = max(4, Int(approx / max(step, 1)))
                for k in 1...n {
                    let t = CGFloat(k) / CGFloat(n), u = 1 - t
                    out.append(CGPoint(
                        x: u * u * u * p0.x + 3 * u * u * t * c1.x + 3 * u * t * t * c2.x + t * t * t * p1.x,
                        y: u * u * u * p0.y + 3 * u * u * t * c1.y + 3 * u * t * t * c2.y + t * t * t * p1.y))
                }
                i += 3
            }
        }
        return out
    }

    /// Shortest distance from `p` to the sampled route.
    public func distance(to p: CGPoint, step: CGFloat = 4) -> CGFloat {
        let s = sampled(step: step)
        var best = CGFloat.infinity
        for i in 1..<max(s.count, 1) { best = min(best, Self.segmentDistance(p, s[i - 1], s[i])) }
        return s.count == 1 ? Self.distance(p, s[0]) : best
    }

    static func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(b.x - a.x, b.y - a.y) }

    static func segmentDistance(_ p: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = b.x - a.x, dy = b.y - a.y
        let len2 = dx * dx + dy * dy
        guard len2 > 0 else { return distance(p, a) }
        let t = max(0, min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2))
        return distance(p, CGPoint(x: a.x + t * dx, y: a.y + t * dy))
    }
}

/// Routes links between an output socket `a` (on the right edge of `sourceFrame`) and an input socket `b` (on the left
/// edge of `targetFrame`). Everything is in graph coordinates.
public enum WireRouter {
    /// Every route leaves and enters its socket horizontally for this long.
    public static let stub: CGFloat = 16
    public static let cornerRadius: CGFloat = 8
    /// Spacing between parallel vertical runs (`lane` is a small signed index).
    public static let laneSpacing: CGFloat = 6
    /// Clearance kept between a detour and the frames it goes around.
    public static let clearance: CGFloat = 24

    public static func route(from a: CGPoint, to b: CGPoint, style: WireStyle,
                             sourceFrame: CGRect? = nil, targetFrame: CGRect? = nil, lane: Int = 0) -> WireRoute {
        // Any target to the right is a forward link, however close (two nodes side by side): the stubs and corners
        // shrink to fit the gap. Only a target to the left needs a detour around the nodes.
        let gap = b.x - a.x
        let forward = gap > 0.5
        let s = forward ? min(stub, gap / 3) : stub
        switch style {
        case .straight:
            return WireRoute(kind: .polyline, points: [a, CGPoint(x: a.x + s, y: a.y), CGPoint(x: b.x - s, y: b.y), b])
        case .rightAngle:
            if forward {
                if abs(a.y - b.y) < 0.5 { return WireRoute(kind: .polyline, points: [a, b]) }
                let r = min(cornerRadius, gap / 4)
                let lo = a.x + s + r, hi = b.x - s - r
                let cx = lo < hi ? min(max((a.x + b.x) / 2 + CGFloat(lane) * laneSpacing, lo), hi) : (a.x + b.x) / 2
                return WireRoute(kind: .polyline, points: [a, CGPoint(x: cx, y: a.y), CGPoint(x: cx, y: b.y), b],
                                 cornerRadius: r)
            }
            let y = detourY(a: a, b: b, sourceFrame: sourceFrame, targetFrame: targetFrame, lane: lane)
            let x1 = a.x + stub + CGFloat(max(lane, 0)) * laneSpacing, x2 = b.x - stub - CGFloat(max(-lane, 0)) * laneSpacing
            return WireRoute(kind: .polyline, points: [a, CGPoint(x: x1, y: a.y), CGPoint(x: x1, y: y), CGPoint(x: x2, y: y),
                                                       CGPoint(x: x2, y: b.y), b],
                             cornerRadius: cornerRadius)
        case .curved:
            if forward || (sourceFrame == nil && targetFrame == nil) {
                let dx = max(abs(gap) * 0.5, min(50, max(abs(b.y - a.y) / 2, 8)))
                return WireRoute(kind: .bezier, points: [a, CGPoint(x: a.x + dx, y: a.y), CGPoint(x: b.x - dx, y: b.y), b])
            }
            // A backward link loops under (or between) the two nodes instead of cutting through them.
            let y = detourY(a: a, b: b, sourceFrame: sourceFrame, targetFrame: targetFrame, lane: lane)
            let d = min(60, max(30, abs(y - a.y) / 2))
            let p1 = CGPoint(x: a.x + stub, y: y), p2 = CGPoint(x: b.x - stub, y: y)
            return WireRoute(kind: .bezier, points: [
                a, CGPoint(x: a.x + stub + d, y: a.y), CGPoint(x: a.x + stub + d, y: y), p1,
                CGPoint(x: p1.x + (p2.x - p1.x) / 3, y: y), CGPoint(x: p1.x + 2 * (p2.x - p1.x) / 3, y: y), p2,
                CGPoint(x: b.x - stub - d, y: y), CGPoint(x: b.x - stub - d, y: b.y), b,
            ])
        }
    }

    /// The horizontal run of a backward link: in the gap between the two frames when there is one, else below both.
    static func detourY(a: CGPoint, b: CGPoint, sourceFrame: CGRect?, targetFrame: CGRect?, lane: Int) -> CGFloat {
        let s = sourceFrame ?? CGRect(x: a.x, y: a.y, width: 0, height: 0)
        let t = targetFrame ?? CGRect(x: b.x, y: b.y, width: 0, height: 0)
        let offset = CGFloat(lane) * laneSpacing
        if t.minY - s.maxY >= 2 * clearance { return (s.maxY + t.minY) / 2 + offset }
        if s.minY - t.maxY >= 2 * clearance { return (t.maxY + s.minY) / 2 + offset }
        return max(s.maxY, t.maxY) + clearance + abs(offset)
    }
}
