import AppKit
import LLMClient
import QuartzCore
import SwiftUI

extension SmokeTest {
    /// `--perf-chats [pairs]` (no server): two chats with long Markdown replies (headings, lists, code, tables,
    /// thinking), switched back and forth, then scrolled top to bottom. Prints how long each switch and each scroll
    /// step takes to lay out and draw. A measurement, not a pass/fail check: it exits 0 unless the window is missing.
    static func runPerfChats(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[perf] \(s)"); fflush(stdout) }
        func settle(_ s: Double) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }
        let pairs = Int(argument("--perf-chats") ?? "") ?? 30
        // `--perf-repeat N`: N rounds back to back with no pauses, for a profiler to sample.
        let rounds = Int(argument("--perf-repeat") ?? "") ?? 1
        let pause = rounds > 1 ? 0.0 : 1.0
        await settle(1.0)
        guard let window = NSApp.windows.first(where: { !($0 is MGFloatingPanel) && $0.isVisible && $0.frame.width >= 900 }),
              let content = window.contentView else { say("no main window"); return false }

        let a = ChatDocument(transcript: perfTranscript(title: "Perf A", pairs: pairs, seed: 1))
        let b = ChatDocument(transcript: perfTranscript(title: "Perf B", pairs: pairs, seed: 2))
        model.chats += [a, b]
        let chars = a.transcript.turns.reduce(0) { $0 + $1.text.count + ($1.reasoning?.count ?? 0) }
        say("\(pairs) question/answer pairs per chat, \(chars / 1024) KB of text each")

        /// Lays out and draws now; the seconds it took.
        func frame() -> Double {
            let t0 = CACurrentMediaTime()
            content.layoutSubtreeIfNeeded()
            window.displayIfNeeded()
            CATransaction.flush()
            return CACurrentMediaTime() - t0
        }
        func ms(_ s: Double) -> String { String(format: "%.0f ms", s * 1000) }
        func summary(_ xs: [Double]) -> String {
            let s = xs.sorted()
            guard !s.isEmpty else { return "none" }
            let p95 = s[min(s.count - 1, Int(Double(s.count) * 0.95))]
            return "median \(ms(s[s.count / 2])), p95 \(ms(p95)), max \(ms(s.last!)), "
                + "\(s.filter { $0 > 1.0 / 60 }.count)/\(s.count) over 16 ms"
        }

        model.activateChat(a.id)
        _ = frame()
        await settle(0.8)

        // Switching: each switch until the new chat is laid out and drawn.
        var switches: [Double] = []
        for i in 0..<(12 * rounds) {
            let t0 = CACurrentMediaTime()
            model.activateChat(i % 2 == 0 ? b.id : a.id)
            _ = frame()
            switches.append(CACurrentMediaTime() - t0)
            await settle(0.4 * pause)
        }
        say("switch: " + summary(switches.dropFirst(2).map { $0 }))


        // Opening a long chat not used lately (six in turn: none is still kept built from before).
        let more = (3...6).map { ChatDocument(transcript: perfTranscript(title: "Perf \($0)", pairs: pairs, seed: $0)) }
        model.chats += more
        let ring = [a, b] + more
        var cold: [Double] = []
        for i in 0..<(12 * rounds) {
            let t0 = CACurrentMediaTime()
            model.activateChat(ring[i % ring.count].id)
            _ = frame()
            cold.append(CACurrentMediaTime() - t0)
            await settle(0.4 * pause)
        }
        say("open a chat not used lately: " + summary(cold.dropFirst(2).map { $0 }))

        // The same between two empty chats: what a switch costs without any history (the header, composer, …).
        let e1 = ChatDocument(transcript: ChatTranscript(title: "Empty 1", serverID: "perf", modelID: "perf-model"))
        let e2 = ChatDocument(transcript: ChatTranscript(title: "Empty 2", serverID: "perf", modelID: "perf-model"))
        model.chats += [e1, e2]
        var empty: [Double] = []
        for i in 0..<(12 * rounds) {
            let t0 = CACurrentMediaTime()
            model.activateChat(i % 2 == 0 ? e1.id : e2.id)
            _ = frame()
            empty.append(CACurrentMediaTime() - t0)
            await settle(0.4 * pause)
        }
        say("switch between empty chats: " + summary(empty.dropFirst(2).map { $0 }))
        model.openChat(a.id)  // reloads it if it dropped out of the recent chats
        _ = frame()

        if CommandLine.arguments.contains("--perf-hold"), !CommandLine.arguments.contains("--perf-stream") {
            // Left open for a look at the real window (a picture from cacheDisplay may miss scroll view content).
            model.openChat(a.id)
            say("holding for 90 s")
            await settle(90)
        }
        if CommandLine.arguments.contains("--perf-switch-only") { return true }
        if CommandLine.arguments.contains("--perf-stream") {
            // The chat on screen (`openChat` may have loaded a fresh copy of A).
            guard let shown = model.activeChat else { say("no chat in front"); return false }
            return await perfStream(shown, content: content, frame: frame, say: say)
        }

        // Scrolling: the chat's transcript, top to bottom, 120 pt a step.
        guard let scroll = transcriptScrollView(content) else { say("no transcript scroll view"); return true }
        let clip = scroll.contentView
        clip.scroll(to: .zero)
        scroll.reflectScrolledClipView(clip)
        _ = frame()
        await settle(0.5)
        var steps: [Double] = []
        for _ in 0..<rounds {
            var y: CGFloat = 0
            clip.scroll(to: .zero)
            scroll.reflectScrolledClipView(clip)
            _ = frame()
            while y < (scroll.documentView?.frame.height ?? 0) - clip.bounds.height, steps.count < 20000 {
                y += 120
                clip.scroll(to: NSPoint(x: 0, y: y))
                scroll.reflectScrolledClipView(clip)
                steps.append(frame())
            }
        }
        say("scroll (\(steps.count) steps of 120 pt, document \(Int(scroll.documentView?.frame.height ?? 0)) pt): "
            + summary(steps))
        say("total scroll time \(ms(steps.reduce(0, +)))")
        return true
    }

    /// `--perf-stream`: a reply streamed into the long chat at the app's rate (20 updates a second). The view must keep
    /// the end in view; after the user scrolls up mid-reply it must stay where they put it.
    static func perfStream(_ doc: ChatDocument, content: NSView, frame: () -> Double, say: (String) -> Void) async -> Bool {
        func settle(_ s: Double) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }
        guard let scroll = transcriptScrollView(content) else { say("no transcript scroll view"); return false }
        let clip = scroll.contentView
        func distanceFromEnd() -> CGFloat { (scroll.documentView?.frame.height ?? 0) - clip.bounds.maxY }
        var reply = ChatTurn(role: .assistant, text: "")
        reply.model = "perf-model"
        doc.transcript.append(reply)
        doc.liveTurnID = reply.id
        _ = frame()
        await settle(0.5)
        let words = perfAnswer(7).split(separator: " ", omittingEmptySubsequences: false)
        var i = 0
        func push(_ n: Int) {
            let chunk = words[i..<min(i + n, words.count)].joined(separator: " ") + " "
            i = (i + n) % max(words.count - n, 1)
            doc.transcript.update(reply.id) { $0.text += chunk }
        }
        let startHeight = scroll.documentView?.frame.height ?? 0
        var followed = 0
        for _ in 0..<80 {
            push(6)
            await settle(0.05)
            if distanceFromEnd() < 120 { followed += 1 }
        }
        let grew = (scroll.documentView?.frame.height ?? 0) - startHeight
        say("streaming, following: \(followed)/80 updates with the end in view (the transcript grew \(Int(grew)) pt)")
        guard grew > 300 else { say("the transcript didn't grow: not the chat on screen"); return false }
        // The user scrolls up to read; the reply keeps coming.
        let up = max(0, clip.bounds.minY - 1500)
        clip.scroll(to: NSPoint(x: 0, y: up))
        scroll.reflectScrolledClipView(clip)
        _ = frame()
        await settle(0.3)
        say("scrolled up: \(Int(distanceFromEnd())) pt above the end")
        var drift: CGFloat = 0
        for _ in 0..<40 {
            push(6)
            await settle(0.05)
            drift = max(drift, abs(clip.bounds.minY - up))
        }
        say("after scrolling up: the view moved \(Int(drift)) pt in 40 updates (0 = stayed put), "
            + "\(Int(distanceFromEnd())) pt above the end")
        if CommandLine.arguments.contains("--perf-hold") {
            // Still streaming, for scrolling the real window by hand (or through accessibility).
            say("holding for 40 s, the reply still streaming")
            for k in 0..<400 {
                push(3)
                await settle(0.1)
                if k % 20 == 0 { say("  \(Int(distanceFromEnd())) pt above the end") }
            }
        }
        doc.liveTurnID = nil
        return followed >= 70 && drift < 40
    }

    /// The transcript in front: the tallest scroll view that isn't hidden (the chats kept built behind the one in
    /// front are at zero opacity).
    @MainActor static func transcriptScrollView(_ content: NSView) -> NSScrollView? {
        func shown(_ v: NSView) -> Bool {
            var cur: NSView? = v
            while let c = cur {
                if c.isHidden || c.alphaValue == 0 || (c.layer?.opacity ?? 1) == 0 { return false }
                cur = c.superview
            }
            return true
        }
        var best: NSScrollView?
        func walk(_ v: NSView) {
            if let s = v as? NSScrollView, let d = s.documentView, d.frame.height > s.contentSize.height * 2, shown(s),
               d.frame.height > (best?.documentView?.frame.height ?? 0) { best = s }
            v.subviews.forEach(walk)
        }
        walk(content)
        return best
    }

    /// A long chat like a real one: questions and Markdown answers with thinking, each a few KB.
    static func perfTranscript(title: String, pairs: Int, seed: Int) -> ChatTranscript {
        var t = ChatTranscript(title: title, serverID: "perf", modelID: "perf-model")
        for i in 0..<pairs {
            let n = i + seed * 1000
            t.append(ChatTurn(role: .user, text: "Question \(n): how should I structure part \(i) of the plan, and what are the trade-offs?"))
            var r = ChatTurn(role: .assistant, text: perfAnswer(n), reasoning: String(repeating: "Considering option \(n): the user wants a plan with steps and trade-offs. ", count: 12))
            r.thinkingSeconds = 3
            r.model = "perf-model"
            t.append(r)
        }
        return t
    }

    static func perfAnswer(_ n: Int) -> String {
        var s = "## Part \(n): the plan\n\nHere is a **structured** approach with `concrete` steps, trade-offs and a sample. "
            + "Each step builds on the previous one, so work through them in order and *check* the result before moving on.\n\n"
        s += "### Steps\n\n"
        for k in 1...8 {
            s += "\(k). **Step \(k)**: read the `section_\(k)` notes, write a short summary, and link it to [the docs](https://example.com/\(n)/\(k)).\n"
        }
        s += "\n### Things to watch\n\n"
        for k in 1...6 { s += "- Item \(k): keep the *scope* small and measure `metric_\(k)` weekly.\n" }
        s += "  - A nested point that is long enough to wrap onto a second line, so the indent of wrapped list lines shows how it lines up under the text.\n"
        s += "\n```swift\n"
        for k in 0..<22 { s += "let value\(k) = compute(step: \(k), seed: \(n)) // keeps the pipeline \(k) deterministic\n" }
        s += "```\n\n| Option | Cost | Time | Risk |\n|---|---|---|---|\n"
        for k in 1...6 { s += "| Option \(k) | $\(k * 10) | \(k) weeks | \(k % 3 == 0 ? "high" : "low") |\n" }
        s += "\n> Tip: revisit this part after two weeks and adjust the plan.\n\nIn short, start small, measure, and iterate."
        return s
    }
}
