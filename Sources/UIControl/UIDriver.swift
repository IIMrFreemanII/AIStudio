import AppKit
import ApplicationServices
import CoreGraphics
import Foundation
import Agent

/// `UIDriving` on the real Mac: accessibility for elements, CGEvent for input, ScreenCaptureKit for pictures.
public final class UIDriver: UIDriving, @unchecked Sendable {
    private let filesFolder: URL
    private let lock = NSLock()
    private var elementMap: [Int32: [String: AXUIElement]] = [:]
    private var lastShot: [Int32: UIShot] = [:]
    private var recorder: Recorder?

    /// `filesFolder`: where recordings are written.
    public init(filesFolder: URL) { self.filesFolder = filesFolder }

    /// Shows the system's permission prompts, for the app's setup card.
    public static func requestAccess(accessibility: Bool, screenRecording: Bool) {
        if accessibility {
            // The value of kAXTrustedCheckOptionPrompt, spelled out (the global isn't concurrency-safe to read).
            _ = AXIsProcessTrustedWithOptions(["AXTrustedCheckOptionPrompt": true] as CFDictionary)
        }
        if screenRecording { _ = CGRequestScreenCaptureAccess() }
    }

    // MARK: UIDriving: reading

    public func runningApps() -> [UIApp] { AppList.running() }

    public func access() -> UIAccess {
        UIAccess(accessibility: AXIsProcessTrusted(), screenRecording: CGPreflightScreenCaptureAccess())
    }

    public func elements(of app: UIApp, window: String?, query: String?, max: Int) async throws -> (elements: [UIElement], total: Int) {
        let root = AXUIElementCreateApplication(app.pid)
        AXUIElementSetMessagingTimeout(root, 1.0)
        var raw: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(root, kAXWindowsAttribute as CFString, &raw)
        if err == .noValue { return ([], 0) }
        guard err == .success else { throw AX.error(err, app: app.name) }
        var windows = (raw as? [AXUIElement]) ?? []
        if let t = window?.lowercased(), !t.isEmpty {
            windows = windows.filter { (AX.string($0, kAXTitleAttribute) ?? "").lowercased().contains(t) }
        }
        guard !windows.isEmpty else { return ([], 0) }
        var walker = AXWalker()
        for w in windows { walker.walkWindow(w) }
        lock.withLock { elementMap[app.pid] = walker.map }
        return ElementFilter.apply(walker.nodes, query: query, max: max)
    }

    public func screenshot(of app: UIApp, window: String?) async throws -> UIShot {
        let shot = try await Capture.screenshot(of: app, window: window)
        lock.withLock { lastShot[app.pid] = shot }
        return shot
    }

    public func frontWindowTitle(of app: UIApp) async -> String? {
        let root = AXUIElementCreateApplication(app.pid)
        AXUIElementSetMessagingTimeout(root, 1.0)
        guard let w = AX.copy(root, kAXFocusedWindowAttribute), CFGetTypeID(w) == AXUIElementGetTypeID() else { return nil }
        return AX.string(w as! AXUIElement, kAXTitleAttribute)
    }

    // MARK: UIDriving: input

    public func click(_ app: UIApp, at: UITarget, button: UIMouseButton, count: Int, modifiers: [String]) async throws {
        let flags = try Input.flags(modifiers)
        try await activate(app)
        let check = frontCheck(app)
        switch try locate(at, in: app) {
        case .point(let p): try await Input.click(at: p, button: button, count: count, flags: flags, check: check)
        case .press(let el): try press(el, app)
        }
    }

    public func type(_ app: UIApp, text: String, into: UITarget?) async throws {
        try await activate(app)
        let check = frontCheck(app)
        if let into {
            switch try locate(into, in: app) {
            case .point(let p): try await Input.click(at: p, button: .left, count: 1, flags: [], check: check)
            case .press(let el): try press(el, app)
            }
            await Input.sleep(100)
        }
        try await Input.type(text, check: check)
    }

    public func key(_ app: UIApp, keys: String, repeat n: Int) async throws {
        let combo = try KeyCombo.parse(keys)
        try await activate(app)
        try await Input.key(combo, repeat: n, check: frontCheck(app))
    }

    public func scroll(_ app: UIApp, at: UITarget?, dx: Int, dy: Int) async throws {
        try await activate(app)
        let p: CGPoint
        if let at {
            p = try point(for: at, in: app)
        } else if let f = AppList.running().first(where: { $0.pid == app.pid })?.windows.first?.frame {
            p = CGPoint(x: f.midX, y: f.midY)   // the list is front to back, so the first window is the front one
        } else {
            throw ToolError("\(app.name) has no window on screen")
        }
        try await Input.scroll(at: p, dx: dx, dy: dy, check: frontCheck(app))
    }

    public func drag(_ app: UIApp, from: UITarget, to: UITarget) async throws {
        try await activate(app)
        let a = try point(for: from, in: app), b = try point(for: to, in: app)
        try await Input.drag(from: a, to: b, check: frontCheck(app))
    }

    // MARK: UIDriving: recording

    public func startRecording(_ app: UIApp, window: String?, maxSeconds: Int) async throws -> URL {
        try lock.withLock {
            if let r = recorder {
                throw ToolError(r.isFinished
                    ? "a recording already finished (its time limit ran out); stop it to collect it before starting another."
                    : "a recording is already running; stop it first.")
            }
        }
        let (win, filter) = try await Capture.window(of: app, titleContains: window)
        try FileManager.default.createDirectory(at: filesFolder, withIntermediateDirectories: true)
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd-HHmmss"
        let file = filesFolder.appendingPathComponent("recording-\(f.string(from: Date())).mp4")
        let r = try await Recorder.start(window: win, filter: filter, file: file, maxSeconds: maxSeconds)
        let clash = lock.withLock { () -> Bool in
            if recorder != nil { return true }
            recorder = r
            return false
        }
        if clash {
            _ = try? await r.finish()
            throw ToolError("a recording is already running; stop it first.")
        }
        return file
    }

    public func stopRecording() async throws -> UIRecording? {
        guard let r = lock.withLock({ recorder }) else { return nil }
        defer { lock.withLock { recorder = nil } }
        return try await r.finish()
    }

    // MARK: Bringing an app forward

    private func frontmostPID() -> Int32? { NSWorkspace.shared.frontmostApplication?.processIdentifier }

    private func gone(_ app: UIApp) -> ToolError { ToolError("\(app.name) isn't running any more.") }

    private func notFront(_ app: UIApp) -> ToolError {
        let front = NSWorkspace.shared.frontmostApplication
        let other = front?.localizedName ?? "another app"
        // macOS shows its own alerts (permission prompts, "… is accessing your screen") from these processes.
        let systemAlert = ["com.apple.UserNotificationCenter", "com.apple.coreservices.uiagent", "com.apple.SecurityAgent"]
        if let id = front?.bundleIdentifier, systemAlert.contains(id) {
            return ToolError("\(app.name) can't come to the front: a macOS alert is showing (\(other)), probably a permission prompt. Ask the user to answer it, then try again.")
        }
        return ToolError("\(app.name) didn't come to the front (\(other) is in front); try again, or ask the user to click its window.")
    }

    private func activate(_ app: UIApp) async throws {
        guard let run = NSRunningApplication(processIdentifier: app.pid), !run.isTerminated else { throw gone(app) }
        if frontmostPID() != app.pid {
            run.activate()
            for _ in 0..<30 {   // 50 ms steps, up to 1.5 s
                if frontmostPID() == app.pid { break }
                try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        guard !run.isTerminated else { throw gone(app) }
        guard frontmostPID() == app.pid else { throw notFront(app) }
    }

    /// Run right before each event batch.
    private func frontCheck(_ app: UIApp) -> Input.Check {
        { [self] in
            guard let run = NSRunningApplication(processIdentifier: app.pid), !run.isTerminated else { throw gone(app) }
            guard frontmostPID() == app.pid else { throw notFront(app) }
        }
    }

    // MARK: Targets

    private enum Located {
        case point(CGPoint)
        case press(AXUIElement)
    }

    private func locate(_ target: UITarget, in app: UIApp) throws -> Located {
        switch target {
        case .element(let id):
            guard let el = lock.withLock({ elementMap[app.pid]?[id] }) else {
                throw ToolError("no element \(id) in \(app.name): call ui_elements again (ids change after each call)")
            }
            if let f = AX.frame(el), f.width > 0, f.height > 0 {
                return .point(CGPoint(x: f.midX, y: f.midY))
            }
            // No frame (not laid out, or off screen): the caller presses it through accessibility instead of clicking.
            return .press(el)
        case .point(let x, let y):
            guard let shot = lock.withLock({ lastShot[app.pid] }) else {
                throw ToolError("take a ui_screenshot of \(app.name) first, or use an element id")
            }
            guard x >= 0, y >= 0, x <= Double(shot.width), y <= Double(shot.height) else {
                throw ToolError("(\(Int(x)),\(Int(y))) is outside the last screenshot of \(app.name) (\(shot.width)×\(shot.height) pixels)")
            }
            return .point(CoordMap.screenPoint(pixel: CGPoint(x: x, y: y),
                                               imageSize: CGSize(width: shot.width, height: shot.height),
                                               windowFrame: shot.window.frame))
        }
    }

    /// A target that must be a position (scroll, drag).
    private func point(for target: UITarget, in app: UIApp) throws -> CGPoint {
        switch try locate(target, in: app) {
        case .point(let p): return p
        case .press: throw ToolError("that element has no position on screen; use another element or a point")
        }
    }

    private func press(_ el: AXUIElement, _ app: UIApp) throws {
        let e = AXUIElementPerformAction(el, kAXPressAction as CFString)
        guard e == .success else {
            throw ToolError("couldn't press the element in \(app.name) (accessibility error \(e.rawValue)); call ui_elements again.")
        }
    }
}
