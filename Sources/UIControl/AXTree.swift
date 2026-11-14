import ApplicationServices
import Foundation
import Agent

/// Pure formatting helpers for accessibility roles.
public enum AXFormat {
    /// "AXButton" → "button", "AXPopUpButton" → "pop up button", "AXStaticText" → "text".
    public static func roleName(_ role: String, subrole: String?) -> String {
        switch subrole {
        case "AXSecureTextField": return "secure text field"
        // A window's title-bar buttons have no title or label: their subrole says which one they are.
        case "AXCloseButton": return "close button"
        case "AXMinimizeButton": return "minimize button"
        case "AXZoomButton": return "zoom button"
        case "AXFullScreenButton": return "full screen button"
        default: break
        }
        switch role {
        case "AXStaticText": return "text"
        case "AXCheckBox": return "checkbox"
        case "AXMenuBarItem": return "menu bar item"
        default: break
        }
        let body = role.hasPrefix("AX") ? String(role.dropFirst(2)) : role
        var words: [String] = []
        var cur = ""
        let chars = Array(body)
        for (i, c) in chars.enumerated() {
            if c.isUppercase, !cur.isEmpty {
                let prevLower = chars[i - 1].isLowercase
                let nextLower = i + 1 < chars.count && chars[i + 1].isLowercase
                if prevLower || (chars[i - 1].isUppercase && nextLower) {
                    words.append(cur); cur = ""
                }
            }
            cur.append(c)
        }
        if !cur.isEmpty { words.append(cur) }
        return words.map { $0.lowercased() }.joined(separator: " ")
    }
}

/// The query filter and cap of `ui_elements`. Pure.
public enum ElementFilter {
    /// No query: the first `max` nodes, total = all of them. With a query: nodes whose role, title, label or value
    /// contains it (case-insensitive) plus their ancestors (found by walking back through `depth`), in preorder, then
    /// capped; total counts the matches only.
    public static func apply(_ nodes: [UIElement], query: String?, max: Int) -> (elements: [UIElement], total: Int) {
        let cap = Swift.max(0, max)
        let q = query?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        guard !q.isEmpty else { return (Array(nodes.prefix(cap)), nodes.count) }
        func matches(_ n: UIElement) -> Bool {
            [n.role, n.title, n.label, n.value].contains { $0?.lowercased().contains(q) ?? false }
        }
        var keep = Set<Int>()
        var total = 0
        for (i, n) in nodes.enumerated() where matches(n) {
            total += 1
            keep.insert(i)
            var d = n.depth
            var j = i - 1
            while d > 0, j >= 0 {
                if nodes[j].depth < d { keep.insert(j); d = nodes[j].depth }
                j -= 1
            }
        }
        let kept = nodes.indices.filter { keep.contains($0) }.map { nodes[$0] }
        return (Array(kept.prefix(cap)), total)
    }
}

/// Thin accessibility-API readers.
enum AX {
    static func copy(_ el: AXUIElement, _ attr: String) -> CFTypeRef? {
        var v: CFTypeRef?
        return AXUIElementCopyAttributeValue(el, attr as CFString, &v) == .success ? v : nil
    }
    static func string(_ el: AXUIElement, _ attr: String) -> String? { copy(el, attr) as? String }
    static func bool(_ el: AXUIElement, _ attr: String) -> Bool? {
        guard let v = copy(el, attr), CFGetTypeID(v) == CFBooleanGetTypeID() else { return nil }
        return CFBooleanGetValue((v as! CFBoolean))
    }
    static func point(_ el: AXUIElement, _ attr: String) -> CGPoint? {
        guard let v = copy(el, attr), CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var p = CGPoint.zero
        return AXValueGetValue(v as! AXValue, .cgPoint, &p) ? p : nil
    }
    static func size(_ el: AXUIElement, _ attr: String) -> CGSize? {
        guard let v = copy(el, attr), CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var s = CGSize.zero
        return AXValueGetValue(v as! AXValue, .cgSize, &s) ? s : nil
    }
    /// The element's frame in screen points (top-left origin), if it has a non-empty one.
    static func frame(_ el: AXUIElement) -> CGRect? {
        guard let p = point(el, kAXPositionAttribute), let s = size(el, kAXSizeAttribute) else { return nil }
        return CGRect(origin: p, size: s)
    }
    static func children(_ el: AXUIElement) -> [AXUIElement] {
        (copy(el, kAXChildrenAttribute) as? [AXUIElement]) ?? []
    }

    static func error(_ e: AXError, app: String) -> ToolError {
        switch e {
        case .apiDisabled:
            return ToolError("the Accessibility permission is missing: ask the user to allow AIStudio in System Settings > Privacy & Security > Accessibility.")
        case .cannotComplete:
            return ToolError("\(app) isn't answering accessibility requests (it may be busy, or not running any more); try again.")
        case .notImplemented:
            return ToolError("\(app) doesn't support accessibility, so its elements can't be listed; use ui_screenshot and pixel coordinates.")
        default:
            return ToolError("accessibility error \(e.rawValue) reading \(app).")
        }
    }
}

/// The depth-first walk behind `ui_elements`.
struct AXWalker {
    static let maxDepth = 25
    static let maxNodes = 2000
    static let maxSeconds = 10.0

    var nodes: [UIElement] = []
    var map: [String: AXUIElement] = [:]
    private var visited = 0
    private let deadline = Date().addingTimeInterval(maxSeconds)

    /// Walks one window and everything under it.
    mutating func walkWindow(_ window: AXUIElement) {
        AXUIElementSetMessagingTimeout(window, 1.0)
        let origin = AX.point(window, kAXPositionAttribute) ?? .zero
        walk(window, level: 0, depth: 0, origin: origin, isWindow: true)
    }

    /// `level`: real tree depth (capped); `depth`: the depth of the next emitted node (skipped nodes don't count).
    private mutating func walk(_ el: AXUIElement, level: Int, depth: Int, origin: CGPoint, isWindow: Bool) {
        guard visited < Self.maxNodes, Date() < deadline else { return }
        visited += 1

        var childDepth = depth
        let frame = AX.frame(el)
        let hasFrame = (frame?.width ?? 0) > 0 && (frame?.height ?? 0) > 0
        if isWindow || hasFrame {
            let id = "e\(nodes.count + 1)"
            let role = AX.string(el, kAXRoleAttribute) ?? "AXUnknown"
            let subrole = AX.string(el, kAXSubroleAttribute)
            var title = AX.string(el, kAXTitleAttribute)
            if title?.isEmpty == true { title = nil }
            var label = AX.string(el, kAXDescriptionAttribute)
            if label?.isEmpty != false { label = AX.string(el, kAXHelpAttribute) }
            if label?.isEmpty == true { label = nil }
            let rel = frame.map {
                UIRect(x: $0.origin.x - origin.x, y: $0.origin.y - origin.y, width: $0.width, height: $0.height)
            }
            nodes.append(UIElement(
                id: id, role: AXFormat.roleName(role, subrole: subrole), title: title,
                value: subrole == "AXSecureTextField" ? nil : Self.value(el), label: label, frame: rel,
                enabled: AX.bool(el, kAXEnabledAttribute) ?? true, focused: AX.bool(el, kAXFocusedAttribute) ?? false,
                depth: depth))
            map[id] = el
            childDepth = depth + 1
        }
        guard level < Self.maxDepth else { return }
        for child in AX.children(el) {
            walk(child, level: level + 1, depth: childDepth, origin: origin, isWindow: false)
        }
    }

    /// Strings, numbers and booleans (as 0/1), cut to 200 characters.
    static func value(_ el: AXUIElement) -> String? {
        guard let v = AX.copy(el, kAXValueAttribute) else { return nil }
        var s: String?
        if let str = v as? String {
            s = str
        } else if CFGetTypeID(v) == CFBooleanGetTypeID() {
            s = CFBooleanGetValue((v as! CFBoolean)) ? "1" : "0"
        } else if let n = v as? NSNumber {
            s = n.stringValue
        }
        guard let s, !s.isEmpty else { return nil }
        return s.count > 200 ? String(s.prefix(200)) : s
    }
}
