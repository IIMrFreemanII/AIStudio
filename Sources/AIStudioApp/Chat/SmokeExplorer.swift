import AIStudioLibrary
import AppKit
import Agent
import Foundation
import LLMClient
import SwiftUI

extension SmokeTest {
    /// `--smoke-explorer` (no server): a coding chat in a fresh git repository with a reply naming a file, an image
    /// and a plan. Checks the side panel's tabs: the Files tree (git-ignored files left out, folders opening), a
    /// previewed file at a line, a path in the reply as a link that opens it, the Plans tab and the plan's file, PDF
    /// and other attachments, and the preview kinds. `--hold` keeps the window open 60 s for a look.
    static func runExplorer(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke-explorer] \(s)"); fflush(stdout) }
        var ok = true
        func check(_ cond: Bool, _ what: String) {
            if cond { say("ok: \(what)") } else { say("FAIL: \(what)"); ok = false }
        }
        func settle(_ s: Double = 0.4) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }
        func wait(_ seconds: Double, until done: @MainActor () -> Bool) async -> Bool {
            let start = Date()
            while !done() {
                if Date().timeIntervalSince(start) > seconds { return false }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return true
        }

        // A project: sources, a README, an image, an ignored build folder.
        let repo = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-explorer-\(UUID().uuidString.prefix(8))")
            .standardizedFileURL.resolvingSymlinksInPath()
        let fm = FileManager.default
        for d in ["Sources/App", "assets", "build/out"] {
            try? fm.createDirectory(at: repo.appendingPathComponent(d), withIntermediateDirectories: true)
        }
        let main = (1...40).map { "let line\($0) = \($0)" }.joined(separator: "\n") + "\n"
        try? main.write(to: repo.appendingPathComponent("Sources/App/main.swift"), atomically: true, encoding: .utf8)
        try? "# Smoke\n\nA **project**.\n".write(to: repo.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
        try? "build/\n".write(to: repo.appendingPathComponent(".gitignore"), atomically: true, encoding: .utf8)
        try? "ignored".write(to: repo.appendingPathComponent("build/out/x.txt"), atomically: true, encoding: .utf8)
        let png = repo.appendingPathComponent("assets/chart.png")
        if let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 48, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0) {
            try? rep.representation(using: .png, properties: [:])?.write(to: png)
        }
        _ = await ShellRunner().run("git init -q -b main && git config user.email smoke@test && git config user.name Smoke"
                                    + " && git add -A && git commit -qm init", cwd: repo)

        let doc = model.newCodingChat(project: repo)
        doc.transcript.append(ChatTurn(role: .user, text: "Plan the change"))
        let reply = "The entry point is `Sources/App/main.swift:3`; see also [the readme](README.md).\n\n![chart](assets/chart.png)"
        let planArgs = ##"{"plan":"# Add a greeting\n\n1. Edit `Sources/App/main.swift`\n2. Test it"}"##
        doc.transcript.append(ChatTurn(role: .assistant, text: reply, toolCalls: [
            ToolCall(id: "smoke-plan", name: "exit_plan", arguments: planArgs)]))
        var result = ChatTurn(role: .tool, text: "The user approved the plan.", toolCallID: "smoke-plan")
        result.toolStatus = "done"
        result.toolDetail = "Approved — Ask Each Time"
        doc.transcript.append(result)
        model.openChat(doc.id)
        await settle(0.8)

        // The side panel is open by default, on Files; the rail at the right edge switches panes, and its toggle hides
        // the panel and shows it again on its pane.
        check(doc.sidePanel == .files && MGTestAnchors.frame("side-panel") != nil, "a new coding chat shows the side panel on Files")
        check(MGTestAnchors.frame("side-tab:changes") != nil && MGTestAnchors.frame("side-panel-toggle") != nil
              && MGTestAnchors.frame("side-tab:flow") == nil,
              "the rail has the panes and the toggle (no Flow item without a run)")
        if let rail = MGTestAnchors.frame("side-rail"), let panel = MGTestAnchors.frame("side-panel") {
            check(rail.minX >= panel.maxX - 1 && rail.width <= 41, "the rail is a slim column right of the panel (\(rail))")
        }
        model.showSidePanel(doc, .plans)
        model.toggleSidePanel(doc)
        await settle()
        check(doc.sidePanel == nil && doc.transcript.sidePanelClosed == true && !model.sidePanelShown(doc),
              "the toggle hides the panel")
        check(MGTestAnchors.frame("side-panel") == nil && MGTestAnchors.frame("side-rail") != nil
              && MGTestAnchors.frame("side-tab:terminal") != nil, "with the panel hidden, the rail stays")
        model.toggleSidePanel(doc)
        await settle()
        check(doc.sidePanel == .plans && model.sidePanelShown(doc) && MGTestAnchors.frame("side-tab:plans") != nil,
              "showing it again keeps its pane")
        model.toggleSidePanel(doc, .plans)
        check(doc.sidePanel == nil, "clicking the shown pane's item hides the panel")
        model.toggleSidePanel(doc, .changes)
        check(doc.sidePanel == .changes, "clicking another item shows that pane")

        // A width each: resizing Changes leaves Plans' width alone.
        let widths = UserDefaults.standard.string(forKey: "sidePanelWidths")
        var json = ""
        let changesWidth = SidePanelWidths.binding(Binding(get: { json }, set: { json = $0 }), .changes)
        changesWidth.wrappedValue = 610
        check(SidePanelWidths.width(json, .changes) == 610 && SidePanelWidths.width(json, .plans) == SidePanelWidths.defaultWidth(.plans)
              && SidePanelWidths.width(json, .web) == 560, "each pane keeps its own width")
        UserDefaults.standard.set("{\"terminal\":700}", forKey: "sidePanelWidths")
        model.showSidePanel(doc, .terminal)
        await settle()
        let shownWidth = MGTestAnchors.frame("side-panel")?.width ?? 0
        check(abs(shownWidth - 700) < 2, "the panel shows at the pane's width (\(shownWidth))")
        UserDefaults.standard.set(widths, forKey: "sidePanelWidths")

        // A flow run is a pane of its own: its rail item comes with the run, switching away keeps it, Close Run goes
        // back to the pane shown before.
        let run = FlowRun(id: "smokeflw", chatID: doc.id, meta: FlowMeta(name: "smoke-flow", description: "", phases: []),
                          script: "export const meta = {}", maxAgents: 1, concurrency: 1)
        FlowRegistry.controllers["smokeflw"] = FlowController(run: run, chatID: doc.id, readOnly: true)
        model.showSidePanel(doc, .changes)
        model.openFlowPanel("smokeflw", in: doc)
        await settle()
        check(doc.sidePanel == .flow && MGTestAnchors.frame("side-tab:flow") != nil && MGTestAnchors.frame("flow-close") != nil,
              "opening a run shows it in the panel, with a Flow item on the rail")
        model.showSidePanel(doc, .terminal)
        await settle()
        check(doc.sidePanel == .terminal && doc.flowPanelRunID == "smokeflw" && MGTestAnchors.frame("side-tab:flow") != nil,
              "another pane keeps the run on the rail")
        model.toggleSidePanel(doc, .flow)
        model.closeFlowPanel(doc)
        await settle()
        check(doc.flowPanelRunID == nil && doc.sidePanel == .terminal && MGTestAnchors.frame("side-tab:flow") == nil,
              "Close Run takes the item off and goes back to the pane before the run")
        doc.transcript.sidePanel = SidePanel.flow.rawValue
        check(doc.sideTab == .files, "a chat saved on a run's pane opens on Files (the run isn't saved)")
        FlowRegistry.controllers["smokeflw"] = nil
        model.showSidePanel(doc, .plans)

        // Files.
        model.showSidePanel(doc, .files)
        check(await wait(10) { doc.fileTree != nil }, "the project's files are listed")
        let files = doc.fileTree?.files ?? []
        check(files.contains("Sources/App/main.swift") && files.contains("README.md") && files.contains(".gitignore"),
              "the tree has the project's files: \(files.sorted())")
        check(!files.contains("build/out/x.txt"), "git-ignored files are left out")
        await settle()
        check(MGTestAnchors.frame("side-panel") != nil, "the side panel is shown")
        check(MGTestAnchors.frame("file-row:Sources") != nil, "a top-level folder row is shown")
        doc.expandedFolders = ["Sources"]
        await settle()
        check(MGTestAnchors.frame("file-row:Sources/App") != nil, "an opened folder shows its contents")
        let rows = doc.fileTree?.rows(expanded: [], filter: "main") ?? []
        check(rows.map(\.path) == ["Sources", "Sources/App", "Sources/App/main.swift"], "the filter finds a file in its folders")

        // A path in the reply is a link to the file; following it previews the file at its line.
        let text = inlineMarkdown("The entry point is `Sources/App/main.swift:3`.", files: doc.fileIndex)
        let link = text.runs.compactMap(\.link).first
        check(link?.scheme == ChatLinks.scheme, "a code span naming a project file is a link (\(link?.absoluteString ?? "none"))")
        let rel = inlineMarkdown("[the readme](README.md)", files: doc.fileIndex).runs.compactMap(\.link).first
        check(rel?.scheme == ChatLinks.scheme, "a relative link to a project file is a link to it")
        check(inlineMarkdown("`not/a/file.swift`", files: doc.fileIndex).runs.allSatisfy { $0.link == nil },
              "a path that isn't in the project stays text")
        let plain = inlineMarkdown("The bug is in Sources/App/main.swift:3, see README.md. Not e.g. v1.2 or x/y.swift.",
                                   files: doc.fileIndex)
        let plainLinks = plain.runs.filter { $0.link != nil }.map { String(plain[$0.range].characters) }
        check(plainLinks == ["Sources/App/main.swift:3", "README.md"], "bare project paths in text are links (\(plainLinks))")

        // The composer: "@word" suggests project files, "/" skills; the mode menu sits under the field.
        let found = doc.fileIndex?.matches("mai") ?? []
        check(found.first == "Sources/App/main.swift", "@mai suggests main.swift first (\(found))")
        check(doc.fileIndex?.matches("sam").first == "Sources/App/main.swift", "letters in order match a path")
        check(ComposerSuggestions.mentionQuery("fix @gre") == "gre" && ComposerSuggestions.mentionQuery("mail a@b") == nil
              && ComposerSuggestions.mentionQuery("@") == "" && ComposerSuggestions.mentionQuery("see `@x` y") == nil,
              "the @word is found only at the end, after a space")
        check(ComposerSuggestions.accept(mention: "Sources/App/main.swift", in: "look at @mai") == "look at `Sources/App/main.swift` ",
              "choosing a file puts its path in backticks")
        check(ComposerSuggestions.slash("/comp", skills: []).map(\.draft) == ["/compact "]
              && ComposerSuggestions.slash("/compact now", skills: []).isEmpty, "slash suggestions until a space")
        check([PermissionMode.ask, .acceptEdits, .plan, .bypass].map(\.cycled) == [.acceptEdits, .plan, .ask, .ask],
              "⇧⇥ cycles Ask → Auto-accept → Plan")

        // Agent runs: the buttons and stats go under a run's last reply only; mid-run replies with nothing under them
        // get no tail row. Finished look-only tools are plain lines, edits stay cards.
        do {
            var t = ChatTranscript(title: "runs")
            let u1 = ChatTurn(role: .user, text: "do it")
            var a1 = ChatTurn(role: .assistant, text: "Looking first.")
            a1.toolCalls = [ToolCall(id: "r1", name: "read_file", arguments: #"{"path":"README.md"}"#)]
            var r1 = ChatTurn(role: .tool, text: "# Smoke", toolCallID: "r1")
            r1.toolStatus = "done"
            let a2 = ChatTurn(role: .assistant, text: "Thinking it over.")
            let a3 = ChatTurn(role: .assistant, text: "Done.")
            let u2 = ChatTurn(role: .user, text: "thanks")
            let b1 = ChatTurn(role: .assistant, text: "Any time.")
            for x in [u1, a1, r1, a2, a3, u2, b1] { t.append(x) }
            let shown = t.turns.filter { $0.role != .tool }
            let runs = AgentRuns(shown, transcript: t)
            check(Set(runs.steps.keys) == [a3.id, b1.id] && runs.steps[a3.id]?.map(\.id) == [a1.id, a2.id, a3.id],
                  "a run's buttons go under its last reply")
            check(runs.bare == [a2.id] && runs.latest == [b1.id], "a mid-run reply with nothing under it is bare; the newest run is latest")
            let rows = TranscriptRow.rows(shown, live: nil, bare: runs.bare)
            check(!rows.contains { $0.turn.id == a2.id && { if case .tail = $0.part { true } else { false } }($0) },
                  "a bare reply has no tail row")
            let read = ToolCard(doc: doc, call: a1.toolCalls[0], result: r1)
            var w = ChatTurn(role: .tool, text: "", toolCallID: "w1")
            w.toolStatus = "done"
            let edit = ToolCard(doc: doc, call: ToolCall(id: "w1", name: "edit_file", arguments: "{}"), result: w)
            check(read.quiet && !edit.quiet, "a finished read is a plain line, an edit stays a card")
        }
        // Folding: a run with tool calls is one header row; opened, its steps follow on the rail. Questions, the call
        // waiting for permission and the answer stay outside; a live run is a box; a plain chat reply doesn't fold.
        do {
            var t = ChatTranscript(title: "fold")
            let u1 = ChatTurn(role: .user, text: "fix it")
            var a1 = ChatTurn(role: .assistant, text: "Looking first.")
            a1.reasoning = "Read, then edit."
            a1.toolCalls = [ToolCall(id: "f1", name: "read_file", arguments: #"{"path":"README.md"}"#),
                            ToolCall(id: "f2", name: "edit_file", arguments: #"{"path":"README.md"}"#)]
            var r1 = ChatTurn(role: .tool, text: "# Smoke", toolCallID: "f1")
            r1.toolStatus = "done"
            var r2 = ChatTurn(role: .tool, text: "", toolCallID: "f2")
            r2.toolStatus = "done"
            r2.toolDetail = "@@ -1 +1,2 @@\n-a\n+b\n+c"
            var a2 = ChatTurn(role: .assistant, text: "")
            a2.toolCalls = [ToolCall(id: "f3", name: "ask_user", arguments: #"{"question":"ok?"}"#),
                            ToolCall(id: "f4", name: "run_command", arguments: #"{"command":"ls"}"#)]
            var r3 = ChatTurn(role: .tool, text: "yes", toolCallID: "f3")
            r3.toolStatus = "done"
            var r4 = ChatTurn(role: .tool, text: "boom", toolCallID: "f4")
            r4.toolStatus = "failed"
            let a3 = ChatTurn(role: .assistant, text: "All done.")
            let u2 = ChatTurn(role: .user, text: "thanks")
            var b1 = ChatTurn(role: .assistant, text: "Any time.")
            b1.reasoning = "Be nice."
            for x in [u1, a1, r1, r2, a2, r3, r4, a3, u2, b1] { t.append(x) }
            let turns = t.turns
            let shown = turns.filter { $0.role != .tool }
            let results = Dictionary(turns.compactMap { x in x.toolCallID.map { ($0, x) } }, uniquingKeysWith: { _, l in l })
            let runs = AgentRuns(shown, transcript: t)
            let key = RunFolding.key([a1])
            func rows(_ list: [ChatTurn], open: Set<String> = [], streaming: Bool = false, asking: String? = nil) -> [TranscriptRow] {
                TranscriptRow.rows(list, live: nil, bare: runs.bare,
                                   folding: RunFolding(open: open, results: results, streaming: streaming, asking: asking))
            }
            func has(_ r: [TranscriptRow], _ test: (TranscriptRow) -> Bool) -> Int { r.filter(test).count }
            func isHeader(_ r: TranscriptRow) -> Bool { if case .runHeader = r.part { true } else { false } }
            func isStep(_ r: TranscriptRow) -> Bool {
                switch r.part { case .groupHead, .groupProse, .groupCalls: true; default: false }
            }
            func pinned(_ r: TranscriptRow, _ id: String) -> Bool { if case .pinned(let c) = r.part { c.id == id } else { false } }
            let closed = rows(shown)
            check(has(closed, isHeader) == 1 && has(closed, isStep) == 0, "a finished run is one header row")
            check(has(closed) { pinned($0, "f3") } == 1 && has(closed) { pinned($0, "f4") } == 0,
                  "a question stays outside the fold, a command doesn't")
            check(has(closed) { $0.turn.id == a3.id && $0.part.isSegment } == 1
                  && has(closed) { $0.turn.id == a3.id && { if case .foot = $0.part { true } else { false } }($0) } == 1,
                  "the run's answer and buttons stay outside")
            check(has(closed) { $0.turn.id == b1.id && { if case .head = $0.part { true } else { false } }($0) } == 1
                  && has(closed) { $0.turn.id == b1.id && isHeader($0) } == 0, "a plain chat reply doesn't fold")
            let opened = rows(shown, open: [key])
            check(has(opened, isStep) == 4, "opened, the run's thinking, text and calls follow (\(has(opened, isStep)))")
            check(Set(opened.map(\.id)).count == opened.count, "every row has its own ID")
            let asking = rows(shown, open: [key], asking: "f4")
            check(has(asking) { pinned($0, "f4") } == 1 && has(asking, isStep) == 3,
                  "the call waiting for permission comes out of the fold")
            let live = rows(Array(shown.prefix(4)), streaming: true)
            check(has(live) { if case .liveBox = $0.part { true } else { false } } == 1
                  && has(live) { if case .foot = $0.part { true } else { false } } == 0
                  && has(live) { $0.turn.id == a3.id && $0.part.isSegment } == 1,
                  "a live run is a box, its streaming text under it")
            let s = RunSummary.of([a1, a2, a3], results: results, live: false)
            check(s.status == .done && s.actions == 3 && s.kinds == ["read 1", "edited 1", "ran 1"]
                  && s.added == 2 && s.removed == 1 && s.failed == 1,
                  "the header counts what the run did (\(s.kinds), +\(s.added) −\(s.removed), \(s.failed) failed)")
            let paste = "Count the words below. " + String(repeating: "alpha bravo charlie delta ", count: 10_000)
            let p = CollapsedText.message(paste)
            check(p != nil && p!.text.utf16.count <= 901 && CollapsedText.message("short") == nil
                  && CollapsedText.message(String(repeating: "line\n", count: 20))?.hidden ?? 0 > 0,
                  "a long message shows a short preview, a short one shows whole")
        }
        var starting = LLMStatus(state: "starting", phase: nil, progress: nil)
        starting.coldStart = nil
        let chip = EngineChip(status: starting, model: nil, since: Date().addingTimeInterval(-80))
        check(chip.label("starting").hasPrefix("Waiting for a GPU · 1:2"), "the chip counts the wait (\(chip.label("starting")))")
        doc.draft = "look at @mai"
        await settle()
        check(MGTestAnchors.frame("composer-suggestions") != nil, "typing @mai shows the suggestion list")
        check(MGTestAnchors.frame("permission-mode") != nil, "the mode menu is under the composer")
        doc.draft = ""
        await settle()
        check(MGTestAnchors.frame("composer-suggestions") == nil, "the list goes with the @word")
        if let link { _ = model.openChatLink(link, in: doc) }
        await settle()
        check(doc.previewPath == "Sources/App/main.swift" && doc.previewLine == 3, "the link previews the file at line 3")
        check(MGTestAnchors.frame("file-row:Sources/App/main.swift") != nil, "the previewed file's folders open")
        let (chunks, _) = TextPreview.load(repo.appendingPathComponent("Sources/App/main.swift"), highlight: 3)
        check(chunks.first { $0.highlighted }?.id == 3 && chunks.map(\.id) == [1, 3, 4],
              "the text preview marks line 3 (chunks \(chunks.map(\.id)))")

        // Preview kinds.
        let kinds: [(String, PreviewKind)] = [("a.swift", .text), ("README.md", .markdown), ("c.png", .image),
                                              ("d.pdf", .pdf), ("e.mp4", .video), ("f.m4a", .audio), ("g.zip", .other)]
        for (name, kind) in kinds {
            check(PreviewKind.of(URL(fileURLWithPath: "/tmp/\(name)")) == kind, "\(name) previews as \(kind)")
        }

        // An image in the reply.
        check(Markdown.blocks(reply).contains(.media(alt: "chart", src: "assets/chart.png")), "the reply's image is a media block")
        let ctx = ChatFilesContext(index: doc.fileIndex, folder: model.chatStore.filesFolder(doc.id), root: doc.projectURL)
        check(ctx.resolve("assets/chart.png")?.path == png.path, "a relative image resolves in the project")

        // A long plan is shown collapsed in the chat; a short one whole.
        let long = "# Big plan\n\n" + (1...60).map { "\($0). Step \($0): do the thing" }.joined(separator: "\n")
        let cut = PlanCard.collapsed(long)
        check(cut != nil && cut!.text.split(separator: "\n").count <= 14 && cut!.hidden > 40,
              "a long plan is collapsed to its first lines (\(cut?.hidden ?? 0) hidden)")
        check(PlanCard.collapsed("# Small\n\n1. One\n2. Two") == nil, "a short plan is shown whole")
        let para = PlanCard.collapsed("# A\n\n" + String(repeating: "word ", count: 600))?.text ?? ""
        check(para.hasPrefix("# A\n\nword word") && para.hasSuffix("…") && para.count <= 1410,
              "a plan with one huge paragraph shows its start (\(para.count) characters)")

        // Plans: the tab lists the plan and saves it as a file.
        model.showSidePanel(doc, .plans)
        await settle()
        check(MGTestAnchors.frame("plan-row:smoke-plan") != nil, "the Plans tab lists the plan")
        let saved = (try? fm.contentsOfDirectory(atPath: model.plansFolder(doc).path)) ?? []
        check(saved.count == 1 && saved[0].hasSuffix(" Add a greeting.md"), "the plan is saved as a file: \(saved)")
        if let f = saved.first {
            let body = try? String(contentsOf: model.plansFolder(doc).appendingPathComponent(f), encoding: .utf8)
            check(body?.hasPrefix("# Add a greeting") == true, "the plan file holds the plan")
        }
        model.showSidePanel(doc, .changes)
        await settle()
        check(doc.sidePanel == .changes && doc.transcript.sidePanel == "changes", "the Changes tab is remembered with the chat")

        // The Browser tab: a page in the project, then a web link from the chat (to a closed port: the error shows).
        let html = repo.appendingPathComponent("index.html")
        try? "<html><head><title>Smoke page</title></head><body><h1>Hello</h1></body></html>"
            .write(to: html, atomically: true, encoding: .utf8)
        model.openWebPage(doc, html)
        check(await wait(10) { doc.webPage?.title == "Smoke page" }, "an HTML file opens in the Browser tab (title “\(doc.webPage?.title ?? "")”)")
        await settle()
        check(doc.sidePanel == .web && MGTestAnchors.frame("web-pane") != nil && MGTestAnchors.frame("side-tab:web") != nil,
              "the side panel shows the Browser tab")
        check(doc.transcript.browserURL == html.absoluteString, "the chat remembers the page")
        let dead = URL(string: "http://127.0.0.1:9/")!
        check(model.openChatLink(dead, in: doc) && doc.webPage?.url == dead, "a web link in the chat opens in the Browser tab")
        check(await wait(10) { doc.webPage?.error != nil }, "a page that can't load says why (\(doc.webPage?.error ?? "no error"))")
        model.closeWebPage(doc)
        check(doc.webPage == nil && doc.transcript.browserURL == nil, "Close Page empties the tab")

        // The agent's open_page: the tab loads the page; the agent gets its title, text and a screenshot.
        let tool = ToolContext(root: repo)
        tool.openPage = { _, url, shot in await model.agentOpenPage(doc, url, screenshot: shot) }
        let opened = try? await OpenPageTool().run(ToolArgs(["url": "index.html", "screenshot": true]), tool)
        check(opened?.isError == false && opened?.text.contains("Title: Smoke page") == true && opened?.text.hasSuffix("Hello") == true,
              "open_page loads a project page for the agent: \(opened?.text.prefix(120) ?? "nothing")")
        check(opened?.images.first?.hasPrefix("data:image/jpeg;base64,") == true, "open_page can return a screenshot")
        check(doc.sidePanel == .web && doc.webPage?.title == "Smoke page", "the user sees the page in the Browser tab")
        let refused = try? await OpenPageTool().run(ToolArgs(["url": "http://127.0.0.1:9/"]), tool)
        check(refused?.isError == true && refused?.text.hasPrefix("Couldn't load http://127.0.0.1:9/") == true,
              "open_page reports a page that can't load (\(refused?.text ?? ""))")

        // The terminal: a shell in the project folder; what's typed runs; read_terminal sees it; a second tab keeps
        // the first one's shell; closing them ends the shells.
        do {
            model.showSidePanel(doc, .terminal)
            await settle(0.6)
            check(MGTestAnchors.frame("terminal-pane") != nil && doc.terminals.count == 1 && MGTestAnchors.frame("side-tab:terminal") != nil,
                  "the Terminal tab opens a terminal")
            if let t = doc.terminals.first {
                t.send("echo hello-$((40+2)); pwd\r")
                let ran = await wait(10) { t.text(lastLines: 50).contains("hello-42") }
                check(ran && t.running && t.pid > 0, "a command typed in the terminal runs (\(t.text(lastLines: 4)))")
                check(t.text(lastLines: 50).contains(repo.lastPathComponent), "the shell starts in the project folder")
                tool.readTerminal = { which, lines in await MainActor.run { model.terminalReport(doc, which: which, lines: lines) } }
                let read = try? await ReadTerminalTool().run(ToolArgs(["lines": 20]), tool)
                check(read?.text.contains("hello-42") == true && read?.text.hasPrefix("Terminal 1 of 1") == true,
                      "read_terminal shows the agent the terminal (\(read?.text.prefix(80) ?? "nothing"))")
                model.newTerminal(doc)
                await settle(0.4)
                check(doc.terminals.count == 2 && doc.activeTerminal !== t && MGTestAnchors.frame("terminal-tab:2") != nil,
                      "+ opens a second terminal")
                doc.activeTerminalID = t.id
                await settle(0.4)
                check(t.running && t.text(lastLines: 50).contains("hello-42") && t.view.superview != nil,
                      "switching back shows the first terminal, still running")
                model.addTerminalToChat(doc, t)
                check(doc.replyTo?.source == "terminal output" && doc.replyTo?.text.contains("hello-42") == true,
                      "Add to Chat quotes the terminal's last lines")
                doc.replyTo = nil
                let prompt = "~/repo main\n❯"
                let trimmed = TerminalSession.withoutTrailingPrompt("\n\(prompt) make\nerror: boom\n\n\(prompt)\n")
                let kept = TerminalSession.withoutTrailingPrompt("$ make\nbuilding\n\nerror: boom\n$")
                let oneLine = TerminalSession.withoutTrailingPrompt("$ ls\na b\n\n$ ")
                check(trimmed.hasSuffix("error: boom") && kept.hasSuffix("error: boom\n$") && oneLine.hasSuffix("a b"),
                      "Add to Chat leaves out the prompt under the output, only when it repeats an earlier one")
                // A shell that exits with mouse reporting left on (a program that died mid-way): the tab is named
                // after the shell again, and Restart turns mouse reporting off so you can select text.
                t.send("printf '\\033[?1000h\\033[?1006h'; exit 3\r")
                let exited = await wait(10) { !t.running }
                check(exited && t.exitCode == 3 && t.title == (TerminalSession.shell as NSString).lastPathComponent,
                      "an exited shell shows its code, and the tab its name (\(t.title), \(t.exitCode.map(String.init) ?? "-"))")
                let mouseWasOn = t.view.getTerminal().mouseMode != .off
                t.start()
                check(mouseWasOn && t.view.getTerminal().mouseMode == .off && t.running,
                      "Restart turns off what the old shell left on (mouse reporting)")
                let pids = doc.terminals.map(\.pid)
                model.closeTerminals(doc)
                let ended = await wait(5) {
                    pids.allSatisfy { pid in
                        var status: Int32 = 0
                        return kill(pid, 0) != 0 || waitpid(pid, &status, WNOHANG) == pid
                    }
                }
                check(ended && doc.terminals.isEmpty, "closing the terminals ends their shells")
            }
        }

        // Commands the AI suggests: a closed ```bash block has Run (not a Python one, not one still streaming); Run
        // types it into the terminal, or into a new one when a command is running in the shown one.
        do {
            let ask = ChatTurn(role: .user, text: "How do I check it?")
            doc.transcript.append(ask)
            doc.transcript.append(ChatTurn(role: .assistant, text: """
                Run this:

                ```bash
                echo run-me-$((40+2))
                ```

                ```python
                print(1)
                ```

                ```bash
                echo half-writ
                """))
            await settle(0.8)
            let anchors = MGTestAnchors.anchors
            check(MGTestAnchors.frame("code-run:echo run-me-$((40+2))") != nil, "a suggested ```bash command has Run")
            check(!anchors.keys.contains { $0.hasPrefix("code-run:print") }, "a Python block has no Run")
            check(anchors["code-run:echo half-writ"] == nil, "a block still being written has no Run")
            model.runInTerminal(doc, "echo run-me-$((40+2))")
            let first = doc.activeTerminal
            let ran = await wait(10) { first?.text(lastLines: 20).contains("\nrun-me-42") == true }
            check(ran && doc.sidePanel == .terminal && doc.terminals.count == 1,
                  "Run opens the Terminal tab and runs it (\(first?.text(lastLines: 3) ?? "no terminal"))")
            first?.send("sleep 30\r")
            let busy = await wait(5) { first?.isIdle == false }
            model.runInTerminal(doc, "echo two-a\necho two-b")
            let second = doc.activeTerminal
            let both = await wait(10) {
                let t = second?.text(lastLines: 20) ?? ""
                return t.contains("\ntwo-a") && t.contains("\ntwo-b")
            }
            check(busy && second !== first && doc.terminals.count == 2 && both,
                  "with a command running, Run uses a new terminal, and runs every line (\(second?.text(lastLines: 4) ?? "-"))")
            model.closeTerminals(doc)
            doc.transcript.remove(ask.id)  // and the reply after it
            await settle(0.3)
        }

        // Attachments: a PDF (its text for the model) and an archive (only named).
        let pdfURL = repo.appendingPathComponent("notes.pdf")
        let view = NSTextView(frame: NSRect(x: 0, y: 0, width: 400, height: 200))
        view.string = "Hello from the smoke PDF"
        try? view.dataWithPDF(inside: view.bounds).write(to: pdfURL)
        let zipURL = repo.appendingPathComponent("data.zip")
        try? Data([0x50, 0x4B, 0x03, 0x04, 0, 0, 0, 0]).write(to: zipURL)
        let note = model.attach([pdfURL, zipURL], to: doc)
        check(note == nil, "both files are attached (\(note ?? ""))")
        let attached = doc.pending.compactMap { if case .file(let f) = $0 { return f }; return nil }
        let pdf = attached.first { $0.name == "notes.pdf" }
        let zip = attached.first { $0.name == "data.zip" }
        check(pdf?.stored != nil && pdf?.text.contains("Hello from the smoke PDF") == true, "the PDF is kept, with its text")
        check(zip?.stored != nil && zip?.text.isEmpty == true, "the archive is kept as it is")
        if let s = pdf?.stored {
            check(fm.fileExists(atPath: model.chatStore.filesFolder(doc.id).appendingPathComponent(s).path), "the PDF's copy is in the chat's folder")
        }
        let turn = ChatTurn(role: .user, text: "See", files: attached)
        check(turn.textWithFiles.contains("`data.zip` (ZIP file"), "the model is told about the archive by name")
        await settle()
        check(MGTestAnchors.frame("attachment:notes.pdf") != nil, "the composer shows the PDF chip")

        // The sidebar: the rail of panel icons at the left edge, the Sessions panel with the project's group and a
        // one-line row.
        let defaults = UserDefaults.standard
        let keptPanel = defaults.object(forKey: "sidebarPanel"), keptOpen = defaults.object(forKey: "sidebarOpen")
        defaults.set(SidebarPanel.chats.rawValue, forKey: "sidebarPanel")
        defaults.set(true, forKey: "sidebarOpen")
        await settle(0.6)
        let sidebarParts = ["app-sidebar", "sidebar-rail", "sidebar-tab:queue", "new-session"].filter { MGTestAnchors.frame($0) == nil }
        check(sidebarParts.isEmpty, "the sidebar shows the rail of panel icons and New session (missing: \(sidebarParts))")
        if let rail = MGTestAnchors.frame("sidebar-rail"), let panel = MGTestAnchors.frame("app-sidebar") {
            check(rail.maxX <= panel.minX + 1 && rail.width <= 45, "the rail is a slim column left of the panel (\(rail), \(panel))")
        }
        check(MGTestAnchors.frame("session-group:project:\(repo.path)") != nil, "the session's project is a group")
        check(MGTestAnchors.frame("session-row:\(doc.id.uuidString)") != nil, "the session has its row")
        let now = Date()
        check(SessionRow.ago(now.addingTimeInterval(-20), now: now) == "now" && SessionRow.ago(now.addingTimeInterval(-300), now: now) == "5m"
              && SessionRow.ago(now.addingTimeInterval(-7200), now: now) == "2h" && SessionRow.ago(now.addingTimeInterval(-3 * 86_400), now: now) == "3d",
              "times read now, 5m, 2h, 3d")
        if let keptPanel { defaults.set(keptPanel, forKey: "sidebarPanel") } else { defaults.removeObject(forKey: "sidebarPanel") }
        if let keptOpen { defaults.set(keptOpen, forKey: "sidebarOpen") } else { defaults.removeObject(forKey: "sidebarOpen") }

        // A running subagent's card is open and lists its steps as they come.
        do {
            var a = ChatTurn(role: .assistant, text: "")
            a.toolCalls = [ToolCall(id: "smoke-task", name: "task", arguments: #"{"prompt":"Review main.swift"}"#)]
            var r = ChatTurn(role: .tool, text: "", toolCallID: "smoke-task")
            r.toolStatus = "running"
            doc.transcript.append(a)
            doc.transcript.append(r)
            doc.toolOutputs["smoke-task"] = "• Read Sources/App/main.swift\n"
            await settle()
            check(MGTestAnchors.frame("task-steps:smoke-task") != nil, "a running subagent's line shows its newest step")
            check(MGTestAnchors.anchors.keys.contains { $0.hasPrefix("live-box:") && MGTestAnchors.frame($0) != nil },
                  "a working run shows its live box")
            // A long message: its preview and buttons; the run before it is no longer live, so it folds.
            let long = ChatTurn(role: .user, text: String(repeating: "alpha bravo charlie delta\n", count: 400))
            doc.transcript.append(long)
            await settle()
            check(MGTestAnchors.frame("message-more:\(long.id.uuidString)") != nil
                  && MGTestAnchors.frame("message-open:\(long.id.uuidString)") != nil, "a long message shows Show more and Open")
            check(MGTestAnchors.anchors.keys.contains { $0.hasPrefix("run-summary:") && MGTestAnchors.frame($0) != nil },
                  "a finished run folds into its header")
        }
        // A session forked from its parent doesn't show the parent's to-do list, only its own.
        do {
            var t = ChatTranscript(title: "todos")
            var a = ChatTurn(role: .assistant, text: "")
            a.toolCalls = [ToolCall(id: "td", name: "todo_write", arguments: #"{"todos":[{"content":"x","status":"completed"}]}"#)]
            t.append(ChatTurn(role: .user, text: "plan it"))
            t.append(a)
            check(!t.todoList.isEmpty, "the parent has a to-do list")
            t.brief = "Delete the caches"
            t.append(ChatTurn(role: .user, text: "Delete the caches\n\n(You're working in your own git worktree…)"))
            check(t.todoList.isEmpty, "the session starts without the parent's to-do list")
        }

        // The full-window viewer.
        model.chatViewer = ChatViewerItem(urls: [png, pdfURL], index: 0)
        await settle()
        model.chatViewer = nil

        if CommandLine.arguments.contains("--hold") {
            model.showSidePanel(doc, .files)
            model.showProjectFile(doc, path: "Sources/App/main.swift", line: 3)
            if let i = CommandLine.arguments.firstIndex(of: "--hold-url"), i + 1 < CommandLine.arguments.count,
               let u = URL(string: CommandLine.arguments[i + 1]) {
                model.openWebPage(doc, u)
            }
            let keptPanel = UserDefaults.standard.object(forKey: "sidebarPanel")
            UserDefaults.standard.set(SidebarPanel.chats.rawValue, forKey: "sidebarPanel")
            // A run to look at: 25 s live (its box: thinking, reads, an edit, a command running), then finished (folded
            // into its header, the answer under it).
            doc.transcript.append(ChatTurn(role: .user, text: "Fix the failing test"))
            var d1 = ChatTurn(role: .assistant, text: "I'll look at the test and the code first.")
            d1.reasoning = "The test says total_value is wrong. Read both files."
            d1.thinkingSeconds = 3
            d1.toolCalls = [ToolCall(id: "d-read", name: "read_file", arguments: #"{"path":"Sources/App/main.swift"}"#),
                            ToolCall(id: "d-search", name: "search", arguments: #"{"pattern":"line3"}"#)]
            var e1 = ChatTurn(role: .tool, text: "let line1 = 1", toolCallID: "d-read")
            e1.toolStatus = "done"
            e1.toolTitle = "Read Sources/App/main.swift"
            var e2 = ChatTurn(role: .tool, text: "Sources/App/main.swift:3", toolCallID: "d-search")
            e2.toolStatus = "done"
            e2.toolTitle = "Search “line3”"
            var d2 = ChatTurn(role: .assistant, text: "")
            d2.toolCalls = [ToolCall(id: "d-edit", name: "edit_file", arguments: #"{"path":"Sources/App/main.swift"}"#)]
            var e3 = ChatTurn(role: .tool, text: "", toolCallID: "d-edit")
            e3.toolStatus = "done"
            e3.toolTitle = "Edit Sources/App/main.swift"
            e3.toolDetail = "@@ -3 +3,2 @@\n-let line3 = 3\n+let line3 = 30\n+let total = line3 * 2"
            var d3 = ChatTurn(role: .assistant, text: "")
            d3.toolCalls = [ToolCall(id: "d-run", name: "run_command", arguments: #"{"command":"swift test"}"#)]
            var e4 = ChatTurn(role: .tool, text: "", toolCallID: "d-run")
            e4.toolStatus = "running"
            e4.toolTitle = "Run swift test"
            e4.toolDetail = "swift test"
            for x in [d1, e1, e2, d2, e3, d3, e4] { doc.transcript.append(x) }
            doc.toolOutputs["d-run"] = "Building for debugging...\n"
            say("holding for 60 s (live run for 25 s)")
            try? await Task.sleep(nanoseconds: 25_000_000_000)
            doc.transcript.update(e4.id) {
                $0.toolStatus = "done"
                $0.text = "Test Suite 'All tests' passed"
            }
            doc.transcript.append(ChatTurn(role: .assistant, text: """
                The test passes now: `line3` was 3 where the test expects 30. To see what changed, run:

                ```bash
                git status --short && git log --oneline -1
                ```
                """))
            try? await Task.sleep(nanoseconds: 35_000_000_000)
            if let keptPanel { UserDefaults.standard.set(keptPanel, forKey: "sidebarPanel") }
        }
        model.deleteChat(doc.id)
        try? fm.removeItem(at: repo)
        say(ok ? "PASS" : "FAILED")
        return ok
    }
}
