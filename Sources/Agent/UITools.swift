import Foundation
import LLMClient

// The `ui_*` tools: the in-chat coding agent looks at and drives other macOS apps to test them. All platform work
// goes through `ctx.ui` (a `UIDriving`); these tools parse arguments, check permissions and shape the answers.

public enum UITools {
    public static let all: [any AgentTool] = [
        UIAppsTool(), UIElementsTool(), UIScreenshotTool(), UIClickTool(), UITypeTool(), UIKeyTool(), UIScrollTool(),
        UIDragTool(), UIRecordTool(),
    ]

    /// Result prefixes the app looks for to show a setup card.
    public static let needsAccessibility = "AIStudio needs the Accessibility permission"
    public static let needsScreenRecording = "AIStudio needs the Screen Recording permission"

    static let unavailable = "The ui_* tools aren't available in this chat."
    static let targetHelp = "give element (an id from ui_elements) or x and y (pixels in the last ui_screenshot)"
    static let appDescription = "A running app's name, bundle id or pid"
    static let shotDescription = "Default true: return a screenshot after the action, for models that see images"

    static func driver(_ ctx: ToolContext) throws -> any UIDriving {
        guard let ui = ctx.ui else { throw ToolError(unavailable) }
        return ui
    }

    /// The driver and the resolved app (preview and run both).
    static func resolve(_ args: ToolArgs, _ ctx: ToolContext) throws -> (ui: any UIDriving, app: UIApp) {
        let ui = try driver(ctx)
        let app = try UIPolicy.resolve(try args.required("app"), driver: ui, ownPID: ctx.ownPID)
        return (ui, app)
    }

    static func preview(_ title: String, _ app: UIApp) -> ToolPreview {
        var p = ToolPreview(kind: .read, title: title)
        p.app = app.ruleKey
        p.appName = app.name
        return p
    }

    static func missing(_ what: String, screen: Bool = false) -> ToolOutput {
        let which = screen ? "Screen Recording" : "Accessibility"
        let head = screen ? needsScreenRecording : needsAccessibility
        return ToolOutput(head + " to do this. Ask the user to turn it on in System Settings › Privacy & Security › \(which) for AIStudio, then try again.",
                          isError: true)
    }

    static func double(_ args: ToolArgs, _ key: String) -> Double? {
        switch args.raw[key] {
        case let v as Int: return Double(v)
        case let v as Double: return v
        case let v as String: return Double(v.trimmingCharacters(in: .whitespaces))
        default: return nil
        }
    }

    static func target(_ args: ToolArgs, prefix: String = "") throws -> UITarget? {
        if let e = args.string("\(prefix)element")?.trimmingCharacters(in: .whitespaces), !e.isEmpty { return .element(e) }
        if let x = double(args, "\(prefix)x"), let y = double(args, "\(prefix)y") { return .point(x: x, y: y) }
        return nil
    }

    static func requiredTarget(_ args: ToolArgs, prefix: String = "") throws -> UITarget {
        guard let t = try target(args, prefix: prefix) else { throw ToolError(targetHelp) }
        return t
    }

    static func describe(_ t: UITarget) -> String {
        switch t {
        case .element(let id): return "element \(id)"
        case .point(let x, let y): return "at \(num(x)),\(num(y))"
        }
    }

    static func num(_ d: Double) -> String { d == d.rounded() ? String(Int(d)) : String(format: "%.1f", d) }

    static func clamp(_ v: Int?, _ dflt: Int, _ range: ClosedRange<Int>) -> Int { min(max(v ?? dflt, range.lowerBound), range.upperBound) }

    static var targetProps: [String: (String, String)] {
        ["element": ("string", "An element id from ui_elements, like \"e12\""),
         "x": ("number", "Pixel x in the last ui_screenshot (when no element)"),
         "y": ("number", "Pixel y in the last ui_screenshot (when no element)")]
    }

    static let screenshotProp: (String, String) = ("boolean", shotDescription)

    /// The common ending of a control action: wait, name the front window, attach a screenshot for vision models.
    static func finish(_ done: String, app: UIApp, args: ToolArgs, ctx: ToolContext, ui: any UIDriving) async throws -> ToolOutput {
        try await Task.sleep(nanoseconds: 300_000_000)
        var text = "\(done) in \(app.name)."
        if let title = await ui.frontWindowTitle(of: app), !title.isEmpty { text += " Front window: “\(title)”." }
        var images: [String] = []
        if ctx.vision, args.raw["screenshot"] as? Bool != false, ui.access().screenRecording,
           let shot = try? await ui.screenshot(of: app, window: nil) {
            images = [shot.dataURL]
            text += " Screenshot attached (\(shot.width)×\(shot.height) px)."
        }
        return ToolOutput(text, images: images)
    }
}

// MARK: ui_apps

public struct UIAppsTool: AgentTool {
    public init() {}
    public let name = "ui_apps"
    public let description = "List the running Mac apps that have windows: name, bundle id, pid and each window's title and size. Use it to find the name to pass as `app` to the other ui_* tools."
    public var parameters: JSONValue { schema([:], required: []) }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        _ = try UITools.driver(ctx)
        return ToolPreview(kind: .read, title: "List apps")
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let ui = try UITools.driver(ctx)
        var lines: [String] = []
        for app in ui.runningApps() {
            let own = app.pid == ctx.ownPID
            if !own, UIPolicy.refusal(app, ownPID: ctx.ownPID) != nil { continue }
            var l = "\(app.name) (\(app.bundleID ?? "no bundle id"), pid \(app.pid))"
            if app.isFrontmost { l += ", frontmost" }
            if own { l += " — this is AIStudio, where this chat runs; can't be driven" }
            lines.append(l)
            for w in app.windows {
                lines.append("  “\(w.title.isEmpty ? "(untitled)" : w.title)” \(Int(w.frame.width.rounded()))×\(Int(w.frame.height.rounded()))")
            }
        }
        return ToolOutput(lines.isEmpty ? "No apps with windows are running." : lines.joined(separator: "\n"))
    }
}

// MARK: ui_elements

public struct UIElementsTool: AgentTool {
    public init() {}
    public let name = "ui_elements"
    public let description = "List the buttons, fields, menus and text in an app's window (its accessibility tree), each with an id like e12 that ui_click and ui_type accept. Cheaper and more exact than a screenshot. Ids last until the next ui_elements call."
    public var parameters: JSONValue {
        schema(["app": ("string", UITools.appDescription),
                "window": ("string", "Only the window whose title contains this"),
                "query": ("string", "Only elements whose role, title, label or value contains this"),
                "max": ("integer", "Most elements to list (default 200, up to 1000)")],
               required: ["app"])
    }
    public let effect = ToolEffect.observe

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        let q = args.string("query").flatMap { $0.isEmpty ? nil : $0 }
        return UITools.preview(q.map { "Find “\($0)” in \(app.name)" } ?? "Read \(app.name)'s window", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        guard ui.access().accessibility else { return UITools.missing("elements") }
        let query = args.string("query").flatMap { $0.isEmpty ? nil : $0 }
        let max = UITools.clamp(args.int("max"), 200, 1...1000)
        let r = try await ui.elements(of: app, window: args.string("window"), query: query, max: max)
        if r.elements.isEmpty {
            return ToolOutput(query.map { "No elements match “\($0)” in \(app.name)." }
                              ?? "\(app.name) has no windows with accessible elements.")
        }
        var head = "\(app.name): \(r.elements.count) elements"
        if r.total > r.elements.count { head += " of \(r.total); narrow it with query or window" }
        let text = head + "\n" + r.elements.map(\.line).joined(separator: "\n")
            + "\nAct on one with ui_click element: \"eN\" (ids last until the next ui_elements call)."
        return ToolOutput(text)
    }
}

// MARK: ui_screenshot

public struct UIScreenshotTool: AgentTool {
    public init() {}
    public let name = "ui_screenshot"
    public let description = "Take a picture of an app's window. Only models that see images get the picture; others should use ui_elements. ui_click x/y are pixels of this picture."
    public var parameters: JSONValue {
        schema(["app": ("string", UITools.appDescription), "window": ("string", "Only the window whose title contains this")],
               required: ["app"])
    }
    public let effect = ToolEffect.observe

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        return UITools.preview("Screenshot \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        guard ui.access().screenRecording else { return UITools.missing("screenshot", screen: true) }
        let shot = try await ui.screenshot(of: app, window: args.string("window"))
        let f = shot.window.frame
        var text = "Screenshot of “\(shot.window.title)” in \(app.name): \(shot.width)×\(shot.height) px. "
            + "The window is at (\(UITools.num(f.x)),\(UITools.num(f.y))) and is \(UITools.num(f.width))×\(UITools.num(f.height)) points. "
            + "ui_click x/y take pixels of this image."
        if !ctx.vision { text += " This model can't see images: use ui_elements to read the window instead." }
        return ToolOutput(text, images: [shot.dataURL])
    }
}

// MARK: ui_click

public struct UIClickTool: AgentTool {
    public init() {}
    public let name = "ui_click"
    public let description = "Click in an app: an element (id from ui_elements) or a pixel (x, y of the last ui_screenshot). Brings the app forward first."
    public var parameters: JSONValue {
        var props = UITools.targetProps
        props["app"] = ("string", UITools.appDescription)
        props["count"] = ("integer", "1 (default) or 2 for a double-click")
        props["screenshot"] = UITools.screenshotProp
        return schema(props, required: ["app"], extra: [
            "button": .object(["type": .string("string"), "enum": .array([.string("left"), .string("right")]),
                               "description": .string("left (default) or right")]),
            "modifiers": .object(["type": .string("array"), "items": .object(["type": .string("string"),
                                  "enum": .array(["cmd", "shift", "option", "ctrl"].map(JSONValue.string))]),
                                  "description": .string("Keys held during the click")]),
        ])
    }
    public let effect = ToolEffect.control

    struct Parsed {
        var target: UITarget
        var button: UIMouseButton
        var count: Int
        var modifiers: [String]
    }

    static func parse(_ args: ToolArgs) throws -> Parsed {
        let button = UIMouseButton(rawValue: (args.string("button") ?? "left").lowercased()) ?? .left
        var mods: [String] = []
        if let a = args.raw["modifiers"] as? [Any] { mods = a.compactMap { $0 as? String } }
        else if let s = args.raw["modifiers"] as? String {
            mods = s.split(whereSeparator: { $0 == "," || $0 == "+" || $0 == " " }).map(String.init)
        }
        return Parsed(target: try UITools.requiredTarget(args), button: button,
                      count: UITools.clamp(args.int("count"), 1, 1...2), modifiers: mods.map { $0.lowercased() })
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        let p = try Self.parse(args)
        let verb = p.count == 2 ? "Double-click" : p.button == .right ? "Right-click" : "Click"
        return UITools.preview("\(verb) \(UITools.describe(p.target)) in \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        let p = try Self.parse(args)
        guard ui.access().accessibility else { return UITools.missing("click") }
        try await ui.click(app, at: p.target, button: p.button, count: p.count, modifiers: p.modifiers)
        return try await UITools.finish("Clicked \(UITools.describe(p.target))", app: app, args: args, ctx: ctx, ui: ui)
    }
}

// MARK: ui_type

public struct UITypeTool: AgentTool {
    public init() {}
    public let name = "ui_type"
    public let description = "Type text into an app. With element (or x, y) it clicks that field first to focus it; otherwise it types where the focus is."
    public var parameters: JSONValue {
        var props = UITools.targetProps
        props["app"] = ("string", UITools.appDescription)
        props["text"] = ("string", "The text to type (up to 5000 characters)")
        props["screenshot"] = UITools.screenshotProp
        return schema(props, required: ["app", "text"])
    }
    public let effect = ToolEffect.control

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        let text = try args.required("text")
        if text.count > 5000 { throw ToolError("text is \(text.count) characters; ui_type takes at most 5000") }
        _ = try UITools.target(args)
        let what = text.count <= 30 ? "“\(text)”" : "\(text.count) characters"
        return UITools.preview("Type \(what) in \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        let text = try args.required("text")
        if text.count > 5000 { throw ToolError("text is \(text.count) characters; ui_type takes at most 5000") }
        let target = try UITools.target(args)
        guard ui.access().accessibility else { return UITools.missing("type") }
        try await ui.type(app, text: text, into: target)
        return try await UITools.finish("Typed \(text.count) characters", app: app, args: args, ctx: ctx, ui: ui)
    }
}

// MARK: ui_key

public struct UIKeyTool: AgentTool {
    public init() {}
    public let name = "ui_key"
    public let description = "Press a key or shortcut in an app, like \"cmd+s\", \"return\", \"shift+tab\", \"esc\" or \"left\"."
    public var parameters: JSONValue {
        schema(["app": ("string", UITools.appDescription),
                "keys": ("string", "A key or shortcut, like \"cmd+s\", \"return\", \"shift+tab\", \"esc\", \"left\""),
                "repeat": ("integer", "How many times (default 1, up to 50)"),
                "screenshot": UITools.screenshotProp],
               required: ["app", "keys"])
    }
    public let effect = ToolEffect.control

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        let keys = try args.required("keys")
        let n = UITools.clamp(args.int("repeat"), 1, 1...50)
        return UITools.preview("Press \(keys) in \(app.name)" + (n > 1 ? " ×\(n)" : ""), app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        let keys = try args.required("keys")
        let n = UITools.clamp(args.int("repeat"), 1, 1...50)
        guard ui.access().accessibility else { return UITools.missing("key") }
        try await ui.key(app, keys: keys, repeat: n)
        return try await UITools.finish("Pressed \(keys)" + (n > 1 ? " ×\(n)" : ""), app: app, args: args, ctx: ctx, ui: ui)
    }
}

// MARK: ui_scroll

public struct UIScrollTool: AgentTool {
    public init() {}
    public let name = "ui_scroll"
    public let description = "Scroll in an app by lines, at an element or pixel (default: the window's centre). Positive dy scrolls down, positive dx right."
    public var parameters: JSONValue {
        var props = UITools.targetProps
        props["app"] = ("string", UITools.appDescription)
        props["dx"] = ("integer", "Lines to scroll right (default 0)")
        props["dy"] = ("integer", "Lines to scroll down (default 5); negative scrolls up")
        props["screenshot"] = UITools.screenshotProp
        return schema(props, required: ["app"])
    }
    public let effect = ToolEffect.control

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        _ = try UITools.target(args)
        return UITools.preview("Scroll \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        let target = try UITools.target(args)
        let dx = args.int("dx") ?? 0
        let dy = args.int("dy") ?? 5
        guard ui.access().accessibility else { return UITools.missing("scroll") }
        try await ui.scroll(app, at: target, dx: dx, dy: dy)
        return try await UITools.finish("Scrolled \(dx),\(dy) lines", app: app, args: args, ctx: ctx, ui: ui)
    }
}

// MARK: ui_drag

public struct UIDragTool: AgentTool {
    public init() {}
    public let name = "ui_drag"
    public let description = "Drag from one place to another in an app. Each end is an element (from_element, to_element) or a pixel (from_x, from_y, to_x, to_y of the last ui_screenshot)."
    public var parameters: JSONValue {
        schema(["app": ("string", UITools.appDescription),
                "from_element": ("string", "Element id to drag from, like \"e3\""),
                "from_x": ("number", "Pixel x to drag from (when no from_element)"),
                "from_y": ("number", "Pixel y to drag from"),
                "to_element": ("string", "Element id to drop on"),
                "to_x": ("number", "Pixel x to drop on (when no to_element)"),
                "to_y": ("number", "Pixel y to drop on"),
                "screenshot": UITools.screenshotProp],
               required: ["app"])
    }
    public let effect = ToolEffect.control

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        _ = try UITools.requiredTarget(args, prefix: "from_")
        _ = try UITools.requiredTarget(args, prefix: "to_")
        return UITools.preview("Drag in \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        let from = try UITools.requiredTarget(args, prefix: "from_")
        let to = try UITools.requiredTarget(args, prefix: "to_")
        guard ui.access().accessibility else { return UITools.missing("drag") }
        try await ui.drag(app, from: from, to: to)
        return try await UITools.finish("Dragged from \(UITools.describe(from)) to \(UITools.describe(to))",
                                        app: app, args: args, ctx: ctx, ui: ui)
    }
}

// MARK: ui_record

public struct UIRecordTool: AgentTool {
    public init() {}
    public let name = "ui_record"
    public let description = "Record an app's window to a video: call with action \"start\", do the steps with the other ui_* tools, then call with action \"stop\" to get the video and a few frames from it."
    public var parameters: JSONValue {
        schema(["app": ("string", UITools.appDescription),
                "window": ("string", "Only the window whose title contains this"),
                "max_seconds": ("integer", "Stop by itself after this long (default 30, up to 120)")],
               required: ["app", "action"],
               extra: ["action": .object(["type": .string("string"), "enum": .array([.string("start"), .string("stop")]),
                                          "description": .string("start or stop")])])
    }
    public let effect = ToolEffect.observe

    static func action(_ args: ToolArgs) throws -> String {
        let a = try args.required("action").lowercased()
        guard a == "start" || a == "stop" else { throw ToolError("action must be \"start\" or \"stop\"") }
        return a
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (_, app) = try UITools.resolve(args, ctx)
        let a = try Self.action(args)
        return UITools.preview(a == "start" ? "Record \(app.name)" : "Stop recording \(app.name)", app)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (ui, app) = try UITools.resolve(args, ctx)
        if try Self.action(args) == "start" {
            guard ui.access().screenRecording else { return UITools.missing("record", screen: true) }
            let secs = UITools.clamp(args.int("max_seconds"), 30, 1...120)
            _ = try await ui.startRecording(app, window: args.string("window"), maxSeconds: secs)
            return ToolOutput("Recording \(app.name) (up to \(secs) s). Do the steps, then call ui_record with action \"stop\".")
        }
        guard let rec = try await ui.stopRecording() else { return ToolOutput("No recording is running.", isError: true) }
        var text = "Recorded \(String(format: "%.1f", rec.seconds)) s of \(app.name): \(rec.file.path)."
        if !rec.keyframes.isEmpty { text += " \(rec.keyframes.count) keyframes attached." }
        var videos: [String] = []
        if ctx.vision, let up = ctx.uploadVideo {
            do {
                videos = [try await up(rec.file)]
                text += " The video is attached too."
            } catch {
                // `String(describing:)`: the client's errors say what went wrong there ("server error 400: …"),
                // their localizedDescription only "error 0".
                text += " (The video couldn't be sent: \(String(describing: error)).)"
            }
        }
        return ToolOutput(text, images: rec.keyframes, videos: videos, attachments: [rec.file])
    }
}
