import Foundation

// The agent's view of the Mac's UI, for testing apps: what the `ui_*` tools need from the platform. The app implements
// it (target UIControl: accessibility, ScreenCaptureKit, CGEvent); tests use a fake. Everything here is Foundation
// only.

/// A rectangle in screen points, top-left origin (as CoreGraphics and the accessibility API report them).
public struct UIRect: Equatable, Sendable, Codable {
    public var x, y, width, height: Double
    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }
    public var midX: Double { x + width / 2 }
    public var midY: Double { y + height / 2 }
}

public struct UIWindowInfo: Equatable, Sendable {
    public var title: String
    public var frame: UIRect
    public init(title: String, frame: UIRect) { self.title = title; self.frame = frame }
}

/// A running app with windows (a GUI app, not a background agent).
public struct UIApp: Equatable, Sendable {
    public var name: String
    public var bundleID: String?
    public var pid: Int32
    public var isFrontmost: Bool
    public var windows: [UIWindowInfo]
    public init(name: String, bundleID: String?, pid: Int32, isFrontmost: Bool = false, windows: [UIWindowInfo] = []) {
        self.name = name; self.bundleID = bundleID; self.pid = pid; self.isFrontmost = isFrontmost; self.windows = windows
    }
    /// What "Always allow" remembers: the bundle id, or the name for an app without one (a bare executable).
    public var ruleKey: String { (bundleID ?? name).lowercased() }
}

/// One accessibility element, as `ui_elements` lists it.
public struct UIElement: Equatable, Sendable {
    /// "e12": valid until the next `ui_elements` call on that app.
    public var id: String
    /// "button", "text field", "checkbox"… (the AX role without its "AX", spaced and lowercased).
    public var role: String
    public var title: String?
    /// The current value (a field's text, a checkbox's 0/1, a slider's number), cut to 200 characters.
    public var value: String?
    /// The accessibility description or help, when there is no title.
    public var label: String?
    /// The frame relative to its window's top-left corner, in points.
    public var frame: UIRect?
    public var enabled: Bool
    public var focused: Bool
    /// Nesting depth below the window (0: the window itself).
    public var depth: Int

    public init(id: String, role: String, title: String? = nil, value: String? = nil, label: String? = nil,
                frame: UIRect? = nil, enabled: Bool = true, focused: Bool = false, depth: Int = 0) {
        self.id = id; self.role = role; self.title = title; self.value = value; self.label = label
        self.frame = frame; self.enabled = enabled; self.focused = focused; self.depth = depth
    }

    /// `  e12 button "Save" (340,210 80×24)` plus `value="…"`, `label="…"`, `disabled`, `focused`.
    public var line: String {
        var s = String(repeating: "  ", count: depth) + id + " " + role
        if let title, !title.isEmpty { s += " " + Self.quoted(title) }
        if let label, !label.isEmpty, label != title { s += " label=" + Self.quoted(label) }
        if let value, !value.isEmpty { s += " value=" + Self.quoted(value) }
        if let f = frame { s += " (\(Int(f.x.rounded())),\(Int(f.y.rounded())) \(Int(f.width.rounded()))×\(Int(f.height.rounded())))" }
        if !enabled { s += " disabled" }
        if focused { s += " focused" }
        return s
    }

    static func quoted(_ s: String) -> String {
        let one = s.replacingOccurrences(of: "\n", with: "⏎")
        return "\"" + (one.count > 80 ? String(one.prefix(79)) + "…" : one) + "\""
    }
}

/// A window's picture, scaled so its long side is at most 1280 pixels.
public struct UIShot: Equatable, Sendable {
    /// `data:image/jpeg;base64,…`
    public var dataURL: String
    /// The image's size in pixels: what `x`/`y` in `ui_click` refer to.
    public var width: Int
    public var height: Int
    public var window: UIWindowInfo
    public init(dataURL: String, width: Int, height: Int, window: UIWindowInfo) {
        self.dataURL = dataURL; self.width = width; self.height = height; self.window = window
    }
}

/// Where an action happens: an element from the last `ui_elements`, or a pixel of the last `ui_screenshot`.
public enum UITarget: Equatable, Sendable {
    case element(String)
    case point(x: Double, y: Double)
}

public enum UIMouseButton: String, Sendable { case left, right }

/// A finished recording.
public struct UIRecording: Equatable, Sendable {
    /// The MP4, in the chat's files folder.
    public var file: URL
    public var seconds: Double
    /// Four JPEG `data:` URLs at even points through it.
    public var keyframes: [String]
    public init(file: URL, seconds: Double, keyframes: [String]) {
        self.file = file; self.seconds = seconds; self.keyframes = keyframes
    }
}

/// Which macOS permissions the app has: Accessibility (elements and input) and Screen Recording (screenshots and
/// recordings).
public struct UIAccess: Equatable, Sendable {
    public var accessibility: Bool
    public var screenRecording: Bool
    public init(accessibility: Bool, screenRecording: Bool) {
        self.accessibility = accessibility; self.screenRecording = screenRecording
    }
}

/// What the `ui_*` tools need from the platform. Calls name the app they act on; input calls bring it forward first
/// and fail if it isn't frontmost when an event would be sent. Errors are `ToolError`s the model can read.
public protocol UIDriving: Sendable {
    /// Running GUI apps (the ones with a Dock icon or windows), frontmost first. Synchronous so a tool's preview can
    /// resolve its app.
    func runningApps() -> [UIApp]
    func access() -> UIAccess

    /// The app's accessibility tree from its windows (or the one whose title contains `window`), filtered by `query`
    /// (role, title, label or value, case-insensitive; ancestors of matches are kept for context), at most `max`
    /// elements. New ids replace the app's old ones. Returns the elements and how many there were before the cap.
    func elements(of app: UIApp, window: String?, query: String?, max: Int) async throws -> (elements: [UIElement], total: Int)
    /// The app's front window (or the one whose title contains `window`). Remembered for `.point` targets.
    func screenshot(of app: UIApp, window: String?) async throws -> UIShot
    func click(_ app: UIApp, at: UITarget, button: UIMouseButton, count: Int, modifiers: [String]) async throws
    /// Types text; with a target, clicks it first to focus it.
    func type(_ app: UIApp, text: String, into: UITarget?) async throws
    /// "cmd+s", "return", "shift+tab", "left"…, `repeat` times.
    func key(_ app: UIApp, keys: String, repeat: Int) async throws
    /// Scrolls by lines (positive dy: down, positive dx: right) at a target, or the window's centre.
    func scroll(_ app: UIApp, at: UITarget?, dx: Int, dy: Int) async throws
    func drag(_ app: UIApp, from: UITarget, to: UITarget) async throws
    /// Starts recording the app's front window (or `window`) to an MP4; one recording at a time. Stops by itself
    /// after `maxSeconds`. Returns the file it writes.
    func startRecording(_ app: UIApp, window: String?, maxSeconds: Int) async throws -> URL
    /// Stops the recording (nil when none is running) and returns it with its keyframes.
    func stopRecording() async throws -> UIRecording?
    /// The title of the app's frontmost window now (for the summary after an action).
    func frontWindowTitle(of app: UIApp) async -> String?
}

/// Which apps the tools may touch, and how a name from the model finds one. Pure, so it's tested without a screen.
public enum UIPolicy {
    /// Apps the agent never sees or touches: they show passwords, unlock the Mac, or change its security.
    public static let refusedBundleIDs: Set<String> = [
        "com.apple.loginwindow", "com.apple.SecurityAgent", "com.apple.systempreferences", "com.apple.Settings",
        "com.apple.keychainaccess", "com.apple.Passwords", "com.apple.coreautha", "com.apple.ScreenSaver.Engine",
        "com.apple.LocalAuthentication.UIAgent", "com.apple.CoreAuthUI",
    ]

    /// Why the agent may not use this app, nil when it may. `ownPID`: this app's process (AIStudio itself is refused:
    /// the agent would type into its own chat; a separately started copy has another pid and is fine).
    public static func refusal(_ app: UIApp, ownPID: Int32) -> String? {
        if app.pid == ownPID {
            return "\(app.name) is AIStudio itself, where this chat runs. To test AIStudio, start a separate copy (its built binary) and use that."
        }
        if let b = app.bundleID, refusedBundleIDs.contains(where: { $0.caseInsensitiveCompare(b) == .orderedSame }) {
            return "\(app.name) is off limits: it handles passwords or the Mac's security. Ask the user to do this part."
        }
        return nil
    }

    /// The app a name, bundle id or pid means: an exact pid, bundle id or name first, then a name that starts with it,
    /// then one that contains it. Several equally good matches: the frontmost, else the first.
    public static func match(_ query: String, in apps: [UIApp]) -> UIApp? {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return nil }
        if let pid = Int32(q) { return apps.first { $0.pid == pid } }
        let lq = q.lowercased()
        let tiers: [(UIApp) -> Bool] = [
            { $0.bundleID?.lowercased() == lq || $0.name.lowercased() == lq },
            { $0.name.lowercased().hasPrefix(lq) },
            { $0.name.lowercased().contains(lq) || ($0.bundleID?.lowercased().contains(lq) ?? false) },
            // A window title ("UI Fixture"), for a copy of an app that shares another's name.
            { $0.windows.contains { $0.title.lowercased().contains(lq) } },
        ]
        for t in tiers {
            let hits = apps.filter(t)
            if let hit = hits.first(where: \.isFrontmost) ?? hits.first { return hit }
        }
        return nil
    }

    /// The app for a tool call, or a `ToolError` saying why not (not running, or refused).
    public static func resolve(_ query: String, driver: any UIDriving, ownPID: Int32) throws -> UIApp {
        let apps = driver.runningApps()
        // A second copy of AIStudio (a dev build, the fixture) has the same name as this one, which is in front while
        // the user chats: look among the other processes first, so the name finds the copy and not this app.
        guard var app = match(query, in: apps.filter { $0.pid != ownPID }) ?? match(query, in: apps) else {
            let names = apps.prefix(15).map(\.name).joined(separator: ", ")
            throw ToolError("no running app matches \"\(query)\". Running: \(names). Start it first (run_command with background: true), then try again.")
        }
        if let why = refusal(app, ownPID: ownPID) { throw ToolError(why) }
        // A copy of this app is named by its window too, so a card doesn't ask to "look at AIStudio" about another
        // window than the one the user is chatting in.
        if apps.contains(where: { $0.pid == ownPID && $0.name == app.name }), let title = app.windows.first?.title,
           !title.isEmpty {
            app.name += " – \(title)"
        }
        return app
    }
}
