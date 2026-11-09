import CoreGraphics
import Foundation

/// Where floating surfaces (menus, pop-ups, tooltips, popovers) go on screen. All rectangles are in AppKit screen
/// coordinates: the origin is the bottom-left corner and y grows upwards. Every result fits inside `screen` whenever
/// the surface does; a surface larger than the screen is pinned to the screen's top-left corner.
public enum SurfacePlacement {
    /// A context menu at the mouse: its top-left corner at `point`, opening to the left near the right edge and
    /// upwards near the bottom edge, as macOS does.
    public static func atPoint(_ point: CGPoint, size: CGSize, screen: CGRect) -> CGRect {
        var x = point.x, y = point.y - size.height
        if x + size.width > screen.maxX { x = point.x - size.width }
        if y < screen.minY { y = point.y }
        return clamp(CGRect(x: x, y: y, width: size.width, height: size.height), to: screen)
    }

    /// Below `anchor`, `gap` from it, aligned to its leading edge; above it when there isn't room below.
    public static func below(_ anchor: CGRect, size: CGSize, screen: CGRect, gap: CGFloat = 4) -> CGRect {
        var y = anchor.minY - gap - size.height
        if y < screen.minY, anchor.maxY + gap + size.height <= screen.maxY { y = anchor.maxY + gap }
        return clamp(CGRect(x: anchor.minX, y: y, width: size.width, height: size.height), to: screen)
    }

    /// A submenu beside its parent menu, its first row level with `row`; it flips to the left of the parent when it
    /// would run off the right of the screen. `inset` is the menu's padding, so the rows line up.
    public static func submenu(row: CGRect, parent: CGRect, size: CGSize, screen: CGRect, inset: CGFloat = 5) -> CGRect {
        var x = parent.maxX - 2
        if x + size.width > screen.maxX { x = parent.minX - size.width + 2 }
        let y = row.maxY + inset - size.height
        return clamp(CGRect(x: x, y: y, width: size.width, height: size.height), to: screen)
    }

    /// A tooltip under the mouse pointer (clear of the arrow), or above it near the bottom of the screen.
    public static func tooltip(at point: CGPoint, size: CGSize, screen: CGRect, below: CGFloat = 22, above: CGFloat = 8) -> CGRect {
        var y = point.y - below - size.height
        if y < screen.minY { y = point.y + above }
        return clamp(CGRect(x: point.x - 4, y: y, width: size.width, height: size.height), to: screen)
    }

    /// The tallest a menu may be on `screen` before it scrolls.
    public static func maxMenuHeight(screen: CGRect, margin: CGFloat = 6) -> CGFloat {
        max(60, screen.height - margin * 2)
    }

    /// Moves `rect` inside `screen` (pinned top-left when it can't fit).
    public static func clamp(_ rect: CGRect, to screen: CGRect) -> CGRect {
        var r = rect
        if r.maxX > screen.maxX { r.origin.x = screen.maxX - r.width }
        if r.minX < screen.minX { r.origin.x = screen.minX }
        if r.minY < screen.minY { r.origin.y = screen.minY }
        if r.maxY > screen.maxY { r.origin.y = screen.maxY - r.height }
        return r
    }
}
