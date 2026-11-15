import Agent
import AppKit
import Foundation
import LLMClient

/// The Studio agent's fallback: clicking and typing in AIStudio's main window like a person, for steps no other tool
/// covers. It works inside the app (its own accessibility tree, a picture of its own window, events posted to itself),
/// so it needs no macOS permission, and it never reaches the agent's panel.
@MainActor
enum StudioFallback {
    /// The elements of the last `app_look`, by id, with their frames on screen.
    static var elements: [String: CGRect] = [:]
    /// The last picture's scale (image pixels per window point) and window, for `x`/`y` targets.
    static var shot: (scale: CGFloat, height: CGFloat)?

    static func tools(_ s: StudioSession) -> [any AgentTool] {
        let target: [String: (String, String)] = [
            "element": ("string", "A control's name from app_look (sidebar-tab:assets)"),
            "x": ("number", "Or a point in app_look's picture: pixels from the left"),
            "y": ("number", "… and from the top"),
        ]
        return [
            StudioTool("app_look", effect: .read,
                       "A picture of AIStudio's main window and its named controls with where they are, filtered by query (part of a name).",
                       GraphHelper.params(["query": ("string", "Only controls whose role, title or value contain this (optional)")]),
                       title: { _ in "Looked at the window" }) { a, _ in
                try await look(s, query: a.string("query"))
            },
            StudioTool("app_click",
                       "Clicks a control in the main window (an element from app_look, or a point in its picture). count 2: double-click; right: a context menu.",
                       GraphHelper.params(target.merging(["count": ("integer", "1 or 2"), "right": ("boolean", "Right-click")]) { a, _ in a }),
                       title: { _ in "Clicked in the window" }) { a, id in
                ToolOutput(try await click(s, a, id))
            },
            StudioTool("app_type",
                       "Types text into the main window: into an element (clicked first to focus it) or wherever the focus is.",
                       GraphHelper.params(target.merging(["text": ("string", "The text")]) { a, _ in a }, required: ["text"]),
                       title: { a in "Typed “\((a.string("text") ?? "").prefix(40))”" }) { a, id in
                ToolOutput(try await type(s, a, id))
            },
            StudioTool("app_key",
                       "Presses a key in the main window: return, tab, escape, delete, space, up, down, left, right, or a letter with modifiers (cmd+a).",
                       GraphHelper.params(["keys": ("string", "return, escape, cmd+a…"), "repeat": ("integer", "How many times (default 1)")], required: ["keys"]),
                       title: { a in "Pressed \(a.string("keys") ?? "a key")" }) { a, id in
                ToolOutput(try await key(s, a, id))
            },
            StudioTool("app_scroll",
                       "Scrolls in the main window at an element or point (dy lines: positive scrolls down).",
                       GraphHelper.params(target.merging(["dy": ("integer", "Lines; positive: down")]) { a, _ in a }, required: ["dy"]),
                       title: { _ in "Scrolled the window" }) { a, id in
                ToolOutput(try await scroll(s, a, id))
            },
        ]
    }

    static func window(_ s: StudioSession) throws -> NSWindow {
        guard let w = s.studio.mainWindow else { throw ToolError("AIStudio's main window isn't open") }
        return w
    }

    // MARK: looking

    static func look(_ s: StudioSession, query: String?) async throws -> ToolOutput {
        let w = try window(s)
        // The window's named controls (the ones the pointer goes to), top to bottom. (Walking the accessibility tree
        // from inside the app stalled on SwiftUI's elements.)
        let q = query?.lowercased()
        var found: [String: CGRect] = [:]
        var rows: [(CGRect, String)] = []
        for name in MGTestAnchors.anchors.keys where q == nil || name.lowercased().contains(q!) {
            guard let f = MGTestAnchors.frame(name), let aw = MGTestAnchors.window(name), aw === w,
                  w.frame.intersects(f) else { continue }
            found[name] = f
            let local = w.convertFromScreen(f)
            rows.append((local, "\(name) (\(Int(local.minX)),\(Int(w.frame.height - local.maxY)) \(Int(f.width))×\(Int(f.height)))"))
        }
        rows.sort { ($0.0.maxY, -$0.0.minX) > ($1.0.maxY, -$1.0.minX) }
        let lines = rows.prefix(300).map(\.1)
        elements = found
        var images: [String] = []
        if let view = w.contentView, let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) {
            view.cacheDisplay(in: view.bounds, to: rep)
            if let img = rep.cgImage, let url = GraphHelper.dataURL(img, maxSide: 1280) {
                images.append(url)
                // Image pixels per window point, after the picture was cut down to 1280 px on its long side.
                let px = CGFloat(max(img.width, img.height))
                let scale = CGFloat(img.width) / max(view.bounds.width, 1) * min(1, 1280 / px)
                shot = (scale: scale, height: view.bounds.height)
            }
        }
        let head = "The main window (\(Int(w.frame.width))×\(Int(w.frame.height)) points\(images.isEmpty ? "" : "; picture attached")). "
            + "Named controls (pass the name as element; x,y top-left in points, size):\n"
        return ToolOutput(head + (lines.isEmpty ? "(none matched)" : lines.joined(separator: "\n")), images: s.vision ? images : [])
    }

    static func num(_ a: ToolArgs, _ key: String) -> Double? {
        (a.raw[key] as? Double) ?? (a.raw[key] as? Int).map(Double.init) ?? (a.raw[key] as? String).flatMap(Double.init)
    }

    /// A target's frame on screen: an element from the last look, or a point of its picture.
    static func target(_ s: StudioSession, _ a: ToolArgs) throws -> CGRect {
        let w = try window(s)
        if let id = a.string("element"), !id.isEmpty {
            guard let f = elements[id] ?? MGTestAnchors.frame(id) else { throw ToolError("no control named \(id) on screen: app_look lists them") }
            return f
        }
        guard let x = num(a, "x"), let y = num(a, "y") else { throw ToolError("say where: element, or x and y") }
        let scale = shot?.scale ?? 1
        let height = shot?.height ?? (w.contentView?.bounds.height ?? w.frame.height)
        let p = CGPoint(x: x / scale, y: height - y / scale)  // window points, bottom-left origin
        let screen = w.convertPoint(toScreen: p)
        return CGRect(x: screen.x - 2, y: screen.y - 2, width: 4, height: 4)
    }

    // MARK: input

    static func post(_ type: NSEvent.EventType, _ screen: CGPoint, in w: NSWindow, clicks: Int = 1) {
        let p = w.convertPoint(fromScreen: screen)
        if let e = NSEvent.mouseEvent(with: type, location: p, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                      windowNumber: w.windowNumber, context: nil, eventNumber: 0, clickCount: clicks, pressure: 1) {
            NSApp.postEvent(e, atStart: false)
        }
    }

    static func keyEvent(_ code: UInt16, _ chars: String, _ flags: NSEvent.ModifierFlags, in w: NSWindow) {
        for type in [NSEvent.EventType.keyDown, .keyUp] {
            if let e = NSEvent.keyEvent(with: type, location: .zero, modifierFlags: flags, timestamp: ProcessInfo.processInfo.systemUptime,
                                        windowNumber: w.windowNumber, context: nil, characters: chars,
                                        charactersIgnoringModifiers: chars, isARepeat: false, keyCode: code) {
                NSApp.postEvent(e, atStart: false)
            }
        }
    }

    static func click(_ s: StudioSession, _ a: ToolArgs, _ id: String) async throws -> String {
        let w = try window(s)
        let rect = try target(s, a)
        let count = max(1, min(a.int("count") ?? 1, 2))
        let right = a.bool("right")
        try await s.studio.act(rect: rect, StudioStep(doing: right ? "Right-clicking" : count == 2 ? "Double-clicking" : "Clicking",
                                                      done: right ? "Right-clicked" : count == 2 ? "Double-clicked" : "Clicked"), callID: id) {
            s.studio.syntheticUntil = Date().addingTimeInterval(0.6)
            w.makeKeyAndOrderFront(nil)
            let p = CGPoint(x: rect.midX, y: rect.midY)
            for k in 1...count {
                post(right ? .rightMouseDown : .leftMouseDown, p, in: w, clicks: k)
                post(right ? .rightMouseUp : .leftMouseUp, p, in: w, clicks: k)
            }
            try? await Task.sleep(for: .milliseconds(300))
        }
        return "Clicked. app_look shows what changed."
    }

    static func type(_ s: StudioSession, _ a: ToolArgs, _ id: String) async throws -> String {
        let w = try window(s)
        let text = try a.required("text")
        let rect = a.string("element") != nil || num(a, "x") != nil ? try target(s, a) : nil
        try await s.studio.act(rect: rect, StudioStep(doing: "Typing “\(text.prefix(40))”", done: "Typed “\(text.prefix(40))”"), callID: id) {
            w.makeKeyAndOrderFront(nil)
            if let rect {
                s.studio.syntheticUntil = Date().addingTimeInterval(0.6)
                post(.leftMouseDown, CGPoint(x: rect.midX, y: rect.midY), in: w)
                post(.leftMouseUp, CGPoint(x: rect.midX, y: rect.midY), in: w)
                try? await Task.sleep(for: .milliseconds(200))
            }
            let delay = s.studio.pace.typingRate.map { 1 / $0 } ?? 0
            for ch in text {
                s.studio.syntheticUntil = Date().addingTimeInterval(0.6)
                let str = String(ch)
                if ch == "\n" { keyEvent(36, "\r", [], in: w) } else { keyEvent(0, str, [], in: w) }
                if delay > 0 { try? await Task.sleep(for: .seconds(delay)) } else { await Task.yield() }
            }
            try? await Task.sleep(for: .milliseconds(150))
        }
        return "Typed \(text.count) characters."
    }

    static let keyCodes: [String: (UInt16, String)] = [
        "return": (36, "\r"), "enter": (36, "\r"), "tab": (48, "\t"), "escape": (53, "\u{1b}"), "esc": (53, "\u{1b}"),
        "delete": (51, "\u{7f}"), "backspace": (51, "\u{7f}"), "space": (49, " "),
        "up": (126, "\u{F700}"), "down": (125, "\u{F701}"), "left": (123, "\u{F702}"), "right": (124, "\u{F703}"),
    ]

    static func key(_ s: StudioSession, _ a: ToolArgs, _ id: String) async throws -> String {
        let w = try window(s)
        let keys = try a.required("keys").lowercased()
        let times = max(1, min(a.int("repeat") ?? 1, 20))
        var parts = keys.split(separator: "+").map(String.init)
        guard let last = parts.popLast() else { throw ToolError("keys is empty") }
        var flags: NSEvent.ModifierFlags = []
        for m in parts {
            switch m {
            case "cmd", "command": flags.insert(.command)
            case "shift": flags.insert(.shift)
            case "option", "alt", "opt": flags.insert(.option)
            case "ctrl", "control": flags.insert(.control)
            default: throw ToolError("unknown modifier \(m)")
            }
        }
        let (code, chars): (UInt16, String)
        if let k = keyCodes[last] { (code, chars) = k } else if last.count == 1 { (code, chars) = (0, last) }
        else { throw ToolError("unknown key \(last)") }
        if flags.contains(.command), ["q", "w"].contains(chars) { throw ToolError("⌘\(chars.uppercased()) would close AIStudio or its window: not allowed") }
        try await s.studio.act(rect: nil, StudioStep(doing: "Pressing \(keys)", done: "Pressed \(keys)"), callID: id) {
            w.makeKeyAndOrderFront(nil)
            for _ in 0..<times {
                s.studio.syntheticUntil = Date().addingTimeInterval(0.6)
                keyEvent(code, chars, flags, in: w)
                try? await Task.sleep(for: .milliseconds(60))
            }
        }
        return "Pressed \(keys)\(times > 1 ? " \(times) times" : "")."
    }

    static func scroll(_ s: StudioSession, _ a: ToolArgs, _ id: String) async throws -> String {
        let w = try window(s)
        let dy = a.int("dy") ?? 3
        let rect = a.string("element") != nil || num(a, "x") != nil ? try target(s, a) : nil
        let p = rect.map { CGPoint(x: $0.midX, y: $0.midY) } ?? CGPoint(x: w.frame.midX, y: w.frame.midY)
        try await s.studio.act(rect: rect, StudioStep(doing: "Scrolling", done: "Scrolled"), callID: id) {
            s.studio.syntheticUntil = Date().addingTimeInterval(0.6)
            let loc = CGPoint(x: p.x, y: (NSScreen.screens.first?.frame.maxY ?? 0) - p.y)  // CG coordinates: top-left
            if let cg = CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: Int32(-dy), wheel2: 0, wheel3: 0) {
                cg.location = loc
                if let e = NSEvent(cgEvent: cg) { NSApp.postEvent(e, atStart: false) }
            }
            try? await Task.sleep(for: .milliseconds(200))
        }
        return "Scrolled \(dy) lines."
    }
}
