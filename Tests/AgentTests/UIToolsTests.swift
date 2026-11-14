import Foundation
import Testing
import LLMClient
@testable import Agent

final class FakeUIDriver: UIDriving, @unchecked Sendable {
    private let lock = NSLock()
    private var _calls: [String] = []
    private var _apps: [UIApp]
    private var _access = UIAccess(accessibility: true, screenRecording: true)
    private var _elementsResult: (elements: [UIElement], total: Int) = ([], 0)
    private var _shot: UIShot
    private var _recording = false
    let recordingFile = FileManager.default.temporaryDirectory.appendingPathComponent("fake-\(UUID().uuidString).mp4")

    init(apps: [UIApp]? = nil) {
        let win = UIWindowInfo(title: "Main", frame: UIRect(x: 10, y: 20, width: 800, height: 600))
        _apps = apps ?? [
            UIApp(name: "Notes", bundleID: "com.apple.Notes", pid: 100, isFrontmost: true, windows: [win]),
            UIApp(name: "Calculator", bundleID: "com.apple.calculator", pid: 200, windows: [win]),
        ]
        _shot = UIShot(dataURL: "data:image/jpeg;base64,SHOT", width: 640, height: 480, window: win)
    }

    var apps: [UIApp] { get { lock.withLock { _apps } } set { lock.withLock { _apps = newValue } } }
    var access_: UIAccess { get { lock.withLock { _access } } set { lock.withLock { _access = newValue } } }
    var elementsResult: (elements: [UIElement], total: Int) {
        get { lock.withLock { _elementsResult } } set { lock.withLock { _elementsResult = newValue } }
    }
    var shot: UIShot { get { lock.withLock { _shot } } set { lock.withLock { _shot = newValue } } }
    var calls: [String] { lock.withLock { _calls } }
    private func log(_ s: String) { lock.withLock { _calls.append(s) } }

    func runningApps() -> [UIApp] { apps }
    func access() -> UIAccess { access_ }
    func elements(of app: UIApp, window: String?, query: String?, max: Int) async throws -> (elements: [UIElement], total: Int) {
        log("elements \(app.name) \(query ?? "-") \(max)")
        return elementsResult
    }
    func screenshot(of app: UIApp, window: String?) async throws -> UIShot { log("screenshot \(app.name)"); return shot }
    func click(_ app: UIApp, at: UITarget, button: UIMouseButton, count: Int, modifiers: [String]) async throws {
        log("click \(Self.d(at)) \(button.rawValue) \(count)" + (modifiers.isEmpty ? "" : " " + modifiers.joined(separator: ",")))
    }
    func type(_ app: UIApp, text: String, into: UITarget?) async throws { log("type \(text)" + (into.map { " into \(Self.d($0))" } ?? "")) }
    func key(_ app: UIApp, keys: String, repeat: Int) async throws { log("key \(keys) \(`repeat`)") }
    func scroll(_ app: UIApp, at: UITarget?, dx: Int, dy: Int) async throws { log("scroll \(at.map(Self.d) ?? "-") \(dx) \(dy)") }
    func drag(_ app: UIApp, from: UITarget, to: UITarget) async throws { log("drag \(Self.d(from)) \(Self.d(to))") }
    func startRecording(_ app: UIApp, window: String?, maxSeconds: Int) async throws -> URL {
        log("start \(app.name) \(maxSeconds)")
        lock.withLock { _recording = true }
        return recordingFile
    }
    func stopRecording() async throws -> UIRecording? {
        let was = lock.withLock { () -> Bool in defer { _recording = false }; return _recording }
        guard was else { return nil }
        return UIRecording(file: recordingFile, seconds: 2.34, keyframes: ["data:image/jpeg;base64,K1", "data:image/jpeg;base64,K2"])
    }
    func frontWindowTitle(of app: UIApp) async -> String? { "Main" }

    static func d(_ t: UITarget) -> String {
        switch t {
        case .element(let id): return id
        case .point(let x, let y): return "\(x),\(y)"
        }
    }
}

private func makeCtx(_ fake: FakeUIDriver, vision: Bool = false) throws -> ToolContext {
    let ctx = ToolContext(root: try tempProject())
    ctx.ui = fake
    ctx.ownPID = 1
    ctx.vision = vision
    return ctx
}

private func runLoop(_ model: ScriptedModel, fake: FakeUIDriver, mode: PermissionMode = .ask, vision: Bool = false,
                     ask: @escaping @Sendable (PermissionRequest) -> PermissionAnswer,
                     rules: (@Sendable (AllowRules) -> Void)? = nil) async throws -> AgentOutcome {
    var config = AgentConfig(model: "m", tools: UITools.all, mode: mode)
    config.vision = vision
    return try await AgentLoop.run(client: model, messages: [.user("test it")], config: config, context: try makeCtx(fake),
                                   ask: { ask($0) }, rulesChanged: { rules?($0) }, events: { _ in })
}

struct UIToolsTests {
    @Test func resolvesByNameBundleIdPidAndPrefix() throws {
        let ctx = try makeCtx(FakeUIDriver())
        for q in ["Notes", "com.apple.Notes", "100", "not"] {
            let p = try UIClickTool().preview(ToolArgs(["app": q, "x": 1, "y": 2]), ctx)
            #expect(p.app == "com.apple.notes" && p.appName == "Notes", "\(q)")
        }
    }

    @Test func aSecondCopyOfThisAppIsFoundByNameOrWindowTitle() throws {
        let fixture = UIWindowInfo(title: "UI Fixture", frame: UIRect(x: 0, y: 0, width: 420, height: 300))
        let fake = FakeUIDriver(apps: [
            UIApp(name: "AIStudio", bundleID: "local.aistudio", pid: 1, isFrontmost: true),
            UIApp(name: "AIStudio", bundleID: "local.aistudio", pid: 500, windows: [fixture]),
        ])
        let ctx = try makeCtx(fake)   // ownPID 1
        for q in ["AIStudio", "local.aistudio", "ui fixture", "500"] {
            #expect(try UIPolicy.resolve(q, driver: fake, ownPID: 1).pid == 500, "\(q)")
        }
        #expect(throws: ToolError.self) { try UIScreenshotTool().preview(ToolArgs(["app": "1"]), ctx) }
        // The card names the copy's window, not just "AIStudio"; the rule key stays the bundle id.
        let p = try UIClickTool().preview(ToolArgs(["app": "AIStudio", "x": 1, "y": 2]), ctx)
        #expect(p.appName == "AIStudio – UI Fixture" && p.app == "local.aistudio")
    }

    @Test func unknownAppListsTheRunningOnes() throws {
        let ctx = try makeCtx(FakeUIDriver())
        do {
            _ = try UIScreenshotTool().preview(ToolArgs(["app": "Zork"]), ctx)
            Issue.record("should throw")
        } catch let e as ToolError {
            #expect(e.message.contains("Notes") && e.message.contains("Calculator") && e.message.contains("Zork"))
        }
    }

    @Test func ownAppAndKeychainAreRefused() async throws {
        let keychain = UIApp(name: "Keychain Access", bundleID: "com.apple.keychainaccess", pid: 300)
        let fake = FakeUIDriver(apps: FakeUIDriver().apps + [keychain])
        let ctx = try makeCtx(fake)
        ctx.ownPID = 100
        #expect(throws: ToolError.self) { try UIScreenshotTool().preview(ToolArgs(["app": "Notes"]), ctx) }
        #expect(throws: ToolError.self) { try UIScreenshotTool().preview(ToolArgs(["app": "Keychain Access"]), ctx) }
        let listed = try await UIAppsTool().run(ToolArgs([:]), ctx).text
        #expect(listed.contains("this is AIStudio") && !listed.contains("Keychain") && listed.contains("Calculator"))
    }

    @Test func noDriverMeansNoTools() async throws {
        let ctx = ToolContext(root: try tempProject())
        #expect(throws: ToolError.self) { try UIAppsTool().preview(ToolArgs([:]), ctx) }
        await #expect(throws: ToolError.self) { try await UIClickTool().run(ToolArgs(["app": "x"]), ctx) }
    }

    @Test func controlAsksOncePerAppWithAlways() async throws {
        let fake = FakeUIDriver()
        let model = ScriptedModel([
            [call("ui_click", ["app": "Notes", "x": 5, "y": 6], id: "a")],
            [call("ui_click", ["app": "Notes", "element": "e1"], id: "b")],
            [call("ui_click", ["app": "Calculator", "element": "e2"], id: "c")],
            [.content("done")],
        ])
        let rec = Recorder()
        let saved = SavedRules()
        _ = try await runLoop(model, fake: fake, ask: { rec.ask($0); return .allowAlways }, rules: { saved.set($0) })
        #expect(rec.asks.count == 2)
        #expect(rec.asks.map(\.effect) == [.control, .control])
        #expect(rec.asks[0].preview.app == "com.apple.notes" && rec.asks[1].preview.app == "com.apple.calculator")
        #expect(saved.history.first?.controlApps == ["com.apple.notes"])
        #expect(saved.history.last?.controlApps == ["com.apple.notes", "com.apple.calculator"])
        #expect(fake.calls.filter { $0.hasPrefix("click") }.count == 3)
    }

    @Test func planModeAllowsLookingAfterAskingButNotClicking() async throws {
        let fake = FakeUIDriver()
        let model = ScriptedModel([
            [call("ui_screenshot", ["app": "Notes"], id: "a")],
            [call("ui_click", ["app": "Notes", "x": 1, "y": 1], id: "b")],
            [.content("done")],
        ])
        let rec = Recorder()
        _ = try await runLoop(model, fake: fake, mode: .plan, ask: { rec.ask($0); return .allow })
        #expect(rec.asks.count == 1 && rec.asks[0].effect == .observe)
        #expect(fake.calls == ["screenshot Notes"])
        #expect(model.requests[2].messages.last?.text.contains("plan mode") == true
                || model.requests[2].messages.last?.text.lowercased().contains("plan") == true)
    }

    @Test func visionModelsGetTheScreenshotAfterAClick() async throws {
        let fake = FakeUIDriver()
        let model = ScriptedModel([[call("ui_click", ["app": "Notes", "element": "e3"])], [.content("ok")]])
        let rec = Recorder()
        _ = try await runLoop(model, fake: fake, vision: true, ask: { _ in .allow })
        _ = rec
        let msgs = model.requests[1].messages
        let tool = try #require(msgs.first { $0.role == .tool })
        #expect(tool.text.contains("Clicked element e3 in Notes.") && tool.text.contains("Front window: “Main”")
                && tool.text.contains("Screenshot attached (640×480 px)."))
        #expect(msgs.last?.role == .user)
        #expect(msgs.last?.parts.contains(.image(url: "data:image/jpeg;base64,SHOT")) == true)
    }

    @Test func textModelsAreToldToUseElements() async throws {
        let fake = FakeUIDriver()
        let model = ScriptedModel([[call("ui_screenshot", ["app": "Notes"])], [.content("ok")]])
        _ = try await runLoop(model, fake: fake, vision: false, ask: { _ in .allow })
        let text = try #require(model.requests[1].messages.last).text
        #expect(text.contains("use ui_elements") && text.contains("left out"))
    }

    @Test func missingAccessibilityStopsBeforeTheDriver() async throws {
        let fake = FakeUIDriver()
        fake.access_ = UIAccess(accessibility: false, screenRecording: true)
        let ctx = try makeCtx(fake)
        let out = try await UIClickTool().run(ToolArgs(["app": "Notes", "x": 1, "y": 2]), ctx)
        #expect(out.isError && out.text.hasPrefix(UITools.needsAccessibility) && out.text.contains("Privacy & Security › Accessibility"))
        #expect(fake.calls.isEmpty)
        fake.access_ = UIAccess(accessibility: true, screenRecording: false)
        let shot = try await UIScreenshotTool().run(ToolArgs(["app": "Notes"]), ctx)
        #expect(shot.isError && shot.text.hasPrefix(UITools.needsScreenRecording))
        #expect(fake.calls.isEmpty)
    }

    @Test func targetsParse() async throws {
        let a = ToolArgs(["element": "e7", "x": 1, "y": 2])
        #expect(try UITools.target(a) == .element("e7"))
        #expect(try UITools.target(ToolArgs(["x": "12.5", "y": 3])) == .point(x: 12.5, y: 3))
        #expect(try UITools.target(ToolArgs(["from_x": 1.5, "from_y": "2"]), prefix: "from_") == .point(x: 1.5, y: 2))
        #expect(try UITools.target(ToolArgs([:])) == nil)
        let ctx = try makeCtx(FakeUIDriver())
        await #expect(throws: ToolError.self) { try await UIClickTool().run(ToolArgs(["app": "Notes"]), ctx) }
        await #expect(throws: ToolError.self) {
            try await UIDragTool().run(ToolArgs(["app": "Notes", "from_element": "e1"]), ctx)
        }
        let fake = FakeUIDriver()
        let c2 = try makeCtx(fake)
        _ = try await UIClickTool().run(ToolArgs(["app": "Notes", "element": "e7", "x": 1, "y": 2, "button": "right",
                                                  "count": 9, "modifiers": "cmd+shift"]), c2)
        #expect(fake.calls == ["click e7 right 2 cmd,shift"])
    }

    @Test func otherControlToolsCallTheDriver() async throws {
        let fake = FakeUIDriver()
        let ctx = try makeCtx(fake)
        _ = try await UITypeTool().run(ToolArgs(["app": "Notes", "text": "hello", "element": "e2"]), ctx)
        _ = try await UIKeyTool().run(ToolArgs(["app": "Notes", "keys": "cmd+s"]), ctx)
        _ = try await UIScrollTool().run(ToolArgs(["app": "Notes"]), ctx)
        let d = try await UIDragTool().run(ToolArgs(["app": "Notes", "from_element": "e1", "to_x": 5, "to_y": 6]), ctx)
        #expect(fake.calls == ["type hello into e2", "key cmd+s 1", "scroll - 0 5", "drag e1 5.0,6.0"])
        #expect(d.text.hasPrefix("Dragged from element e1 to at 5,6 in Notes."))
        let p = try UITypeTool().preview(ToolArgs(["app": "Notes", "text": String(repeating: "a", count: 31)]), ctx)
        #expect(p.title == "Type 31 characters in Notes")
        #expect(throws: ToolError.self) {
            try UITypeTool().preview(ToolArgs(["app": "Notes", "text": String(repeating: "a", count: 5001)]), ctx)
        }
    }

    @Test func elementsTextHasHeaderLinesAndFooter() async throws {
        let fake = FakeUIDriver()
        let els = [UIElement(id: "e1", role: "window", title: "Main"),
                   UIElement(id: "e2", role: "button", title: "Save", frame: UIRect(x: 1, y: 2, width: 3, height: 4), depth: 1)]
        fake.elementsResult = (els, 5)
        let ctx = try makeCtx(fake)
        let out = try await UIElementsTool().run(ToolArgs(["app": "Notes", "query": "x", "max": 5000]), ctx)
        let lines = out.text.split(separator: "\n").map(String.init)
        #expect(lines[0] == "Notes: 2 elements of 5; narrow it with query or window")
        #expect(lines[1] == els[0].line && lines[2] == els[1].line)
        #expect(lines[3] == "Act on one with ui_click element: \"eN\" (ids last until the next ui_elements call).")
        #expect(fake.calls == ["elements Notes x 1000"])
        fake.elementsResult = ([], 0)
        let none = try await UIElementsTool().run(ToolArgs(["app": "Notes", "query": "zzz"]), ctx)
        #expect(none.text == "No elements match “zzz” in Notes." && !none.isError)
    }

    @Test func recordingStartThenStop() async throws {
        let fake = FakeUIDriver()
        let ctx = try makeCtx(fake, vision: true)
        ctx.uploadVideo = { _ in "upload:clip" }
        let none = try await UIRecordTool().run(ToolArgs(["app": "Notes", "action": "stop"]), ctx)
        #expect(none.isError && none.text == "No recording is running.")
        let start = try await UIRecordTool().run(ToolArgs(["app": "Notes", "action": "start", "max_seconds": 500]), ctx)
        #expect(start.text.contains("up to 120 s"))
        let stop = try await UIRecordTool().run(ToolArgs(["app": "Notes", "action": "stop"]), ctx)
        #expect(stop.attachments == [fake.recordingFile] && stop.images.count == 2 && stop.videos == ["upload:clip"])
        #expect(stop.text.hasPrefix("Recorded 2.3 s of Notes:") && stop.text.contains("2 keyframes attached."))

        // Without vision or an uploader there's no video.
        let plain = try makeCtx(fake, vision: false)
        plain.uploadVideo = { _ in "upload:clip" }
        _ = try await UIRecordTool().run(ToolArgs(["app": "Notes", "action": "start"]), plain)
        let s2 = try await UIRecordTool().run(ToolArgs(["app": "Notes", "action": "stop"]), plain)
        #expect(s2.videos.isEmpty && s2.attachments.count == 1)
    }

    @Test func elementLineFormatting() {
        let e = UIElement(id: "e12", role: "button", title: "Save", value: "1", label: "Save file",
                          frame: UIRect(x: 340, y: 210, width: 80, height: 24), enabled: false, focused: true, depth: 1)
        #expect(e.line == "  e12 button \"Save\" label=\"Save file\" value=\"1\" (340,210 80×24) disabled focused")
    }
}

final class SavedRules: @unchecked Sendable {
    private let lock = NSLock()
    private var _history: [AllowRules] = []
    var history: [AllowRules] { lock.withLock { _history } }
    func set(_ r: AllowRules) { lock.withLock { _history.append(r) } }
}
