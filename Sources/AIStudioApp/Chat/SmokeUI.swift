import AVFoundation
import Agent
import AppKit
import Foundation
import UIControl

extension SmokeTest {
    /// `--smoke-ui`: starts the fixture (`AIStudioApp --ui-fixture`) as a child process and drives it with the real
    /// `UIDriver`: elements, typing, clicks, keys, a screenshot, a recording and a click by pixel. Needs Accessibility
    /// and Screen Recording for this process (the app that launched it); when they're missing it asks macOS to show its prompts and stops.
    @MainActor static func runUI() async -> Bool {
        func say(_ s: String) { print("[smoke-ui] \(s)"); fflush(stdout) }
        var ok = true
        func expect(_ cond: Bool, _ good: String, _ bad: @autoclosure () -> String) {
            if cond { say("ok: \(good)") } else { say("FAIL: \(bad())"); ok = false }
        }
        func finish() -> Bool { say(ok ? "PASSED" : "FAILED"); return ok }

        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-ui-\(UUID().uuidString.prefix(8))")
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let driver = UIDriver(filesFolder: tmp)

        // 1. Permissions.
        let access = driver.access()
        if !access.accessibility || !access.screenRecording {
            var missing: [String] = []
            if !access.accessibility { missing.append("Accessibility") }
            if !access.screenRecording { missing.append("Screen Recording") }
            // Lists AIStudio in System Settings (and shows macOS's prompts) so the user only has to switch it on.
            UIDriver.requestAccess(accessibility: !access.accessibility, screenRecording: !access.screenRecording)
            say("needs Accessibility and Screen Recording for this process: turn them on for AIStudio (or the app that launched it) in System Settings › Privacy & Security, then run again (missing: \(missing.joined(separator: ", ")))")
            say("FAILED")
            return false
        }

        // 2. The fixture, as a child process.
        let child = Process()
        child.executableURL = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        child.arguments = ["--ui-fixture"]
        do { try child.run() } catch { say("FAIL: couldn't start the fixture: \(error.localizedDescription)"); say("FAILED"); return false }
        defer { if child.isRunning { child.terminate() } }
        let pid = child.processIdentifier

        var found: UIApp?
        for _ in 0..<100 {
            if let a = driver.runningApps().first(where: { $0.pid == pid && !$0.windows.isEmpty }) { found = a; break }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        guard let listed = found else {
            say("FAIL: the fixture (pid \(pid)) didn't show a window within 10 s")
            say("FAILED")
            return false
        }
        say("ok: fixture running, pid \(pid), windows \(listed.windows.map(\.title))")

        // 3. Policy.
        let apps = driver.runningApps()
        guard let app = UIPolicy.match(String(pid), in: apps) else {
            say("FAIL: UIPolicy.match(\"\(pid)\") found nothing"); say("FAILED"); return false
        }
        let refusal = UIPolicy.refusal(app, ownPID: getpid())
        expect(refusal == nil, "the fixture may be driven", "fixture refused: \(refusal ?? "")")
        let own = apps.first { $0.pid == getpid() } ?? UIApp(name: "AIStudio", bundleID: nil, pid: getpid())
        expect(UIPolicy.refusal(own, ownPID: getpid()) != nil, "this process is refused", "own process (pid \(getpid())) wasn't refused")

        func elements() async -> [UIElement] {
            do { return try await driver.elements(of: app, window: nil, query: nil, max: 200).elements } catch {
                say("FAIL: elements threw: \(error.localizedDescription)"); ok = false; return []
            }
        }
        func countButton(_ els: [UIElement]) -> UIElement? { els.first { $0.role == "button" && ($0.title ?? $0.label) == "Count" } }
        func field(_ els: [UIElement]) -> UIElement? { els.first { $0.role.contains("text field") } }
        func countText(_ els: [UIElement]) -> String? {
            els.compactMap { [$0.title, $0.value].compactMap { $0 }.first { $0.hasPrefix("Count: ") } }.first
        }
        func click(_ id: String) async {
            do { try await driver.click(app, at: .element(id), button: .left, count: 1, modifiers: []) } catch {
                say("FAIL: click \(id) threw: \(error.localizedDescription)"); ok = false
            }
        }

        // 4. Elements.
        var els = await elements()
        say("elements:\n" + els.map(\.line).joined(separator: "\n"))
        let fieldEl = field(els), buttonEl = countButton(els)
        expect(fieldEl != nil, "a text field", "no text field in \(els.map { "\($0.role) \($0.title ?? "")" })")
        expect(buttonEl != nil, "a button titled Count", "no Count button in \(els.map { "\($0.role) \($0.title ?? "")" })")
        expect(countText(els) == "Count: 0", "text \"Count: 0\"", "expected \"Count: 0\", got \(countText(els) ?? "none")")
        guard let fieldID = fieldEl?.id, let buttonID = buttonEl?.id else { return finish() }

        // 5. Typing into the field.
        do { try await driver.type(app, text: "hello", into: .element(fieldID)) } catch {
            say("FAIL: type threw: \(error.localizedDescription)"); ok = false
        }
        els = await elements()
        expect(field(els)?.value == "hello", "typed into the field", "field value expected \"hello\", got \(field(els)?.value.map { "\"\($0)\"" } ?? "nil")")

        // 6. Two clicks on Count (ids stay valid until the next elements call).
        await click(buttonID)
        try? await Task.sleep(nanoseconds: 300_000_000)
        await click(buttonID)
        try? await Task.sleep(nanoseconds: 300_000_000)
        els = await elements()
        expect(countText(els) == "Count: 2", "two clicks counted", "expected \"Count: 2\", got \(countText(els) ?? "none")")

        // 7. cmd+a then a letter replaces the field's text; the field has to be focused first.
        if let f = field(els) { await click(f.id) }
        do {
            try await driver.key(app, keys: "cmd+a", repeat: 1)
            try await driver.type(app, text: "x", into: nil)
        } catch { say("FAIL: key/type threw: \(error.localizedDescription)"); ok = false }
        els = await elements()
        expect(field(els)?.value == "x", "cmd+a then \"x\" replaced the text", "field value expected \"x\", got \(field(els)?.value.map { "\"\($0)\"" } ?? "nil")")

        // 8. Screenshot.
        var shot: UIShot?
        do { shot = try await driver.screenshot(of: app, window: nil) } catch {
            say("FAIL: screenshot threw: \(error.localizedDescription)"); ok = false
        }
        if let shot {
            expect(shot.dataURL.hasPrefix("data:image/jpeg;base64,"), "a JPEG data URL", "data URL starts \(shot.dataURL.prefix(30))")
            expect(shot.width > 100 && shot.height > 100, "image \(shot.width)×\(shot.height)", "image is \(shot.width)×\(shot.height), expected both > 100")
            expect(shot.window.title == "UI Fixture", "window title", "window title expected \"UI Fixture\", got \"\(shot.window.title)\"")
        }

        // 9. A 3 s recording with a click during it.
        do {
            let file = try await driver.startRecording(app, window: nil, maxSeconds: 10)
            try? await Task.sleep(nanoseconds: 500_000_000)
            els = await elements()
            if let b = countButton(els) { await click(b.id) }
            // While a window is captured, macOS puts a small "Window" badge over its title bar, owned by the app: it
            // must not be taken for the app's front window.
            let during = try await driver.screenshot(of: app, window: nil)
            expect(during.window.title == "UI Fixture", "a screenshot during the recording is of the window",
                   "a screenshot during the recording got window \"\(during.window.title)\" (\(during.width)×\(during.height))")
            let listed = driver.runningApps().first { $0.pid == pid }?.windows.map(\.title) ?? []
            expect(listed == ["UI Fixture"], "the window list ignores the recording badge", "windows during the recording: \(listed)")
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            if let rec = try await driver.stopRecording() {
                expect(FileManager.default.fileExists(atPath: rec.file.path), "the recording file exists", "no file at \(rec.file.path) (started at \(file.path))")
                expect(rec.seconds > 2.5, "recorded \(String(format: "%.1f", rec.seconds)) s", "seconds expected > 2.5, got \(rec.seconds)")
                expect(rec.keyframes.count == 4, "4 keyframes", "keyframes expected 4, got \(rec.keyframes.count)")
                let dur = (try? await AVURLAsset(url: rec.file).load(.duration).seconds) ?? 0
                expect(dur > 2.5, "the MP4 lasts \(String(format: "%.1f", dur)) s", "AVAsset duration expected > 2.5 s, got \(dur)")
            } else { say("FAIL: stopRecording returned nil"); ok = false }
        } catch { say("FAIL: recording threw: \(error.localizedDescription)"); ok = false }

        // 10. A click at a pixel of a fresh screenshot: the Count button's centre.
        els = await elements()
        let beforeText = countText(els)
        let beforeN = beforeText.flatMap { Int($0.dropFirst("Count: ".count)) }
        if let b = countButton(els), let frame = b.frame, let beforeN {
            do {
                let s = try await driver.screenshot(of: app, window: nil)
                let scaleX = Double(s.width) / s.window.frame.width, scaleY = Double(s.height) / s.window.frame.height
                try await driver.click(app, at: .point(x: frame.midX * scaleX, y: frame.midY * scaleY), button: .left, count: 1, modifiers: [])
                try? await Task.sleep(nanoseconds: 400_000_000)
                let after = countText(await elements())
                expect(after == "Count: \(beforeN + 1)", "a click at a pixel counted", "expected \"Count: \(beforeN + 1)\", got \(after ?? "none")")
            } catch { say("FAIL: point click threw: \(error.localizedDescription)"); ok = false }
        } else { say("FAIL: no Count button or text before the point click (\(beforeText ?? "none"))"); ok = false }

        // 11. Scrolling: a positive dy scrolls down (the list's first visible row moves further into the list), a
        // negative one back up.
        func listState() async -> (area: String, top: Int)? {
            let els = await elements()
            guard let area = els.first(where: { $0.role == "scroll area" }), let box = area.frame else { return nil }
            let visible = els.filter { el in
                el.role == "text" && (el.frame.map { $0.midY > box.y && $0.midY < box.y + box.height } ?? false)
            }
            guard let top = visible.compactMap({ $0.value.flatMap { UIFixture.rows.firstIndex(of: $0) } }).min() else { return nil }
            return (area.id, top)
        }
        func scroll(_ dy: Int, _ at: String) async {
            do { try await driver.scroll(app, at: .element(at), dx: 0, dy: dy) } catch {
                say("FAIL: scroll \(dy) threw: \(error.localizedDescription)"); ok = false
            }
            try? await Task.sleep(nanoseconds: 600_000_000)   // the list animates
        }
        if let start = await listState() {
            await scroll(10, start.area)
            if let down = await listState() {
                expect(down.top > start.top, "dy 10 scrolled down (first row \(UIFixture.rows[start.top]) → \(UIFixture.rows[down.top]))",
                       "dy 10 should scroll down: first visible row \(UIFixture.rows[start.top]) → \(UIFixture.rows[down.top])")
                await scroll(-10, down.area)
                if let up = await listState() {
                    expect(up.top < down.top, "dy -10 scrolled back up (→ \(UIFixture.rows[up.top]))",
                           "dy -10 should scroll up: first visible row \(UIFixture.rows[down.top]) → \(UIFixture.rows[up.top])")
                } else { say("FAIL: no visible list rows after scrolling up"); ok = false }
            } else { say("FAIL: no visible list rows after scrolling down"); ok = false }
        } else { say("FAIL: no scroll area with visible rows in the fixture"); ok = false }

        return finish()
    }
}
