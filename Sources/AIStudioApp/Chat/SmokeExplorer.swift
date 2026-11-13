import AIStudioLibrary
import AppKit
import Agent
import Foundation
import LLMClient

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

        // The side panel is open by default, on Files; the header's toggle hides it and shows it again on its tab.
        check(doc.sidePanel == .files && MGTestAnchors.frame("side-panel") != nil, "a new coding chat shows the side panel on Files")
        check(MGTestAnchors.frame("side-tab:changes") != nil && MGTestAnchors.frame("side-panel-toggle") != nil,
              "the panel has its tabs and the header its toggle")
        model.showSidePanel(doc, .plans)
        model.toggleSidePanel(doc)
        await settle()
        check(doc.sidePanel == nil && doc.transcript.sidePanelClosed == true && !model.sidePanelShown(doc),
              "the toggle hides the panel")
        model.toggleSidePanel(doc)
        await settle()
        check(doc.sidePanel == .plans && model.sidePanelShown(doc) && MGTestAnchors.frame("side-tab:plans") != nil,
              "showing it again keeps its tab")

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

        // The sidebar: the panels' icons along the top, the Sessions panel with the project's group and a one-line row.
        let defaults = UserDefaults.standard
        let keptPanel = defaults.object(forKey: "sidebarPanel"), keptOpen = defaults.object(forKey: "sidebarOpen")
        defaults.set(SidebarPanel.chats.rawValue, forKey: "sidebarPanel")
        defaults.set(true, forKey: "sidebarOpen")
        await settle(0.6)
        check(MGTestAnchors.frame("app-sidebar") != nil && MGTestAnchors.frame("sidebar-tab:queue") != nil
              && MGTestAnchors.frame("new-session") != nil, "the sidebar shows the panels' icons and New session")
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
            check(MGTestAnchors.frame("task-steps:smoke-task") != nil, "a running subagent's card shows its steps")
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
            say("holding for 60 s")
            try? await Task.sleep(nanoseconds: 60_000_000_000)
            if let keptPanel { UserDefaults.standard.set(keptPanel, forKey: "sidebarPanel") }
        }
        model.deleteChat(doc.id)
        try? fm.removeItem(at: repo)
        say(ok ? "PASS" : "FAILED")
        return ok
    }
}
