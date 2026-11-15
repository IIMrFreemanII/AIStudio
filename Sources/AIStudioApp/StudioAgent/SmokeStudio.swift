import Agent
import AppKit
import Foundation
import GraphRuntime
import LLMClient
import MovieCore
import NodesRemote

extension SmokeTest {
    /// `--smoke-studio-agent <gpu server url>`: the Studio agent's panel and tools against the fake GPU server
    /// (`AISTUDIO_FAKE_LLM=1 AISTUDIO_FAKE_MODELS=1`). The tools are called as the model would call them; the test plays
    /// the user (it presses Run on the cost sheets, which the agent can't, and clicks the window to pause it).
    static func runStudioAgent(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-studio] \(s)"); fflush(stdout) }
        var ok = true
        func check(_ cond: Bool, _ what: String) {
            say((cond ? "ok   " : "FAIL ") + what)
            if !cond { ok = false }
        }
        func settle(_ s: Double = 0.4) async { try? await Task.sleep(for: .seconds(s)) }

        // The GPU server and its fake LLM, as the movie smoke test sets them up.
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else { say("server not reachable"); return false }
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/tiny" || $0.id == "fake/tiny" }?.id
        if mid == nil {
            guard let st = try? await client.install(LLMInstallRequest(url: "fake/tiny")) else { say("install failed"); return false }
            mid = st.id
            for _ in 0..<150 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        guard let mid else { say("no model"); return false }
        model.lastChatModel = LLMKey(serverID: conn.id, modelID: mid)
        try? FileManager.default.removeItem(at: model.gpuTimingsURL)
        await settle(1)

        // GPU fallback: a model whose GPU doesn't come moves to the next preset in its list, then goes back.
        let key = LLMKey(serverID: conn.id, modelID: mid)
        if let m0 = llm.model(key) {
            let other = GPUPreset(gpu: m0.gpu == "H100" ? "L40S" : "H100", count: 1)
            llm.setFallbacks([other], for: key)
            let label = await llm.switchGPU(key)
            await llm.refresh(conn.id)
            check(label == other.label && llm.model(key)?.gpu == other.gpu,
                  "a slow start moves the model to its next GPU preset (\(label ?? "none"), now \(llm.model(key)?.gpu ?? "?"))")
            check(llm.gpuNotes[key]?.contains("this start uses \(other.label)") == true, "the chat says so")
            check(await llm.switchGPU(key) == other.label, "a second chat waiting on it doesn't move it again")
            await llm.settleGPU(key)
            await llm.refresh(conn.id)
            check(llm.model(key)?.gpu == m0.gpu && llm.model(key)?.count == m0.count,
                  "once the reply began, its own GPU is back for next time (\(llm.model(key)?.gpu ?? "?"))")
            llm.lastSwitch[key] = nil
            llm.gpuNotes[key] = nil
            llm.setFallbacks([], for: key)
            check(await llm.switchGPU(key) == nil, "with no list it just waits")
        } else {
            check(false, "the fake model is listed")
        }

        // The panel.
        model.openStudioAgent()
        await settle(0.8)
        let studio = model.studio
        check(studio.panel.isVisible && studio.chat?.isStudio == true, "⌥⌘A's panel is up with a Studio agent chat")
        check(studio.chat?.mode == .acceptEdits, "it has no mode menu: deletes ask, the rest is free")
        check(studio.mainWindow != nil, "it finds the main window")
        studio.pace = .fast

        // The tools, called as the model would.
        let session = StudioSession(app: model, vision: true)
        let tools = Dictionary(uniqueKeysWithValues: StudioTools.all(session).map { ($0.name, $0) })
        let ctx = ToolContext(root: model.library.root)
        var n = 0
        func call(_ name: String, _ args: [String: Any] = [:]) async -> ToolOutput {
            n += 1
            ctx.callID = "smoke_\(n)"
            guard let tool = tools[name] else { return ToolOutput("no tool \(name)", isError: true) }
            let json = String(data: (try? JSONSerialization.data(withJSONObject: args)) ?? Data("{}".utf8), encoding: .utf8) ?? "{}"
            do {
                return try await tool.run(try ToolArgs(json: json), ctx)
            } catch {
                return ToolOutput("\(name) failed: \(error.localizedDescription)", isError: true)
            }
        }
        /// Presses Run on the next cost sheet (movie or workflow) once it shows: what the user does.
        func pressRunWhenAsked() -> Task<Bool, Never> {
            Task { @MainActor in
                for _ in 0..<300 {
                    if let doc = model.activeMovie, let g = doc.pendingGate {
                        doc.pendingGate = nil
                        g.run()
                        return true
                    }
                    if let g = model.studioRunGate {
                        model.studioRunGate = nil
                        g.run()
                        return true
                    }
                    try? await Task.sleep(for: .milliseconds(100))
                }
                return false
            }
        }

        let state = await call("studio_state")
        check(state.text.contains("In front:") && state.text.contains("Sidebar:"), "studio_state says where the app is")

        // Navigation: the pointer goes to the rail's tab.
        var out = await call("studio_open", ["what": "movies"])
        check(UserDefaults.standard.string(forKey: "sidebarPanel") == "movies", "studio_open shows the Movies panel (\(out.text))")
        check(studio.pointer != nil && studio.ring != nil, "the pointer and ring went to its tab")
        check(studio.revisits["smoke_\(n)"] != nil, "its step row can go back there")

        // A movie from an idea.
        out = await call("movie_create", ["idea": "A girl keeps a lighthouse lit through a storm.", "seconds": 30])
        guard let doc = model.activeMovie else { say("FAIL no movie: \(out.text)"); return false }
        doc.movie.llm = MovieLLM(serverID: conn.id, modelID: mid)
        check(doc.movie.idea.hasPrefix("A girl keeps") && doc.stage == .idea, "movie_create typed the idea (\(out.text.prefix(60)))")

        // The script: the cost sheet waits for the user's Run.
        var runner = pressRunWhenAsked()
        out = await call("movie_run", ["work": "script"])
        check(await runner.value, "the script's cost sheet came up and the test pressed Run")
        check(out.text.hasPrefix("Finished the script") && doc.movie.scenes.count == 2, "movie_run waited for the script (\(out.text.prefix(80)))")
        check(studio.waitingOnUser == nil, "the hourglass row is gone")

        // Cancelling a sheet: the tool says so and nothing runs.
        doc.movie.approve(.script)
        let canceller = Task { @MainActor in
            for _ in 0..<300 {
                if doc.pendingGate != nil {
                    try? await Task.sleep(for: .milliseconds(400))  // a person reads the sheet first
                    doc.pendingGate = nil
                    return true
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            return false
        }
        out = await call("movie_run", ["work": "shots"])
        check(await canceller.value && out.text.contains("cancelled") && doc.movie.shots.isEmpty,
              "Cancel on the sheet: the tool reports it and makes nothing (\(out.text.prefix(70)))")
        runner = pressRunWhenAsked()
        out = await call("movie_run", ["work": "shots"])
        check(await runner.value && !doc.movie.shots.isEmpty, "the shots came in after Run (\(doc.movie.shots.count))")
        out = await call("movie_approve", ["stage": "shots"])
        check(doc.movie.approved.contains(.shots), "movie_approve approved the shots")

        // Edits: typed on its page, approval taken back, Revert puts it back.
        let before = doc.movie.shots[0].keyframePrompt
        out = await call("movie_approve", ["stage": "cast"])
        let editID = "smoke_\(n + 1)"
        out = await call("movie_edit", ["field": "keyframe_prompt", "shot": 1, "value": "A lighthouse at dawn, one girl at the rail."])
        check(doc.movie.shots[0].keyframePrompt == "A lighthouse at dawn, one girl at the rail." && doc.stage == .keyframes,
              "movie_edit typed shot 1's keyframe prompt on the Keyframes page")
        check(studio.reverts[editID] != nil, "the step can be reverted")
        studio.revert(editID)
        check(doc.movie.shots[0].keyframePrompt == before && studio.reverted.contains(editID), "Revert put the old prompt back")

        // A card below the fold: its page scrolls to it before the pointer goes there.
        if let main = studio.mainWindow {
            let kept = main.frame
            main.setFrame(NSRect(x: kept.minX, y: kept.maxY - 520, width: kept.width, height: 520), display: true)
            await settle(0.5)
            let last = doc.movie.shots.count
            let lastID = doc.movie.shots[last - 1].id.uuidString
            out = await call("movie_edit", ["field": "keyframe_prompt", "shot": last, "value": doc.movie.shots[last - 1].keyframePrompt + " "])
            await settle(0.5)
            check(studio.onScreen("movie-keyframe:\(lastID)"), "shot \(last)'s card was scrolled into view for the pointer")
            if let f = studio.frame("movie-keyframe:\(lastID)"), let ring = studio.ring {
                let local = studio.local(f, in: main)
                check(abs(ring.midY - local.midY) < 8, "the ring is on that card")
            } else {
                check(false, "the ring is on that card")
            }
            main.setFrame(kept, display: true)
        }

        // Sheets, then looking at one.
        runner = pressRunWhenAsked()
        out = await call("movie_run", ["work": "cast"])
        check(await runner.value && doc.movie.characters.allSatisfy { !$0.sheets.isEmpty }, "character sheets made (\(out.text.prefix(60)))")
        let look = await call("look_at", ["kind": "sheet", "character": doc.movie.characters[0].id])
        check(look.images.count == 1 && look.images[0].hasPrefix("data:image/jpeg"), "look_at returns the sheet for the model to see")
        check(model.chatViewer == nil, "the viewer closed after looking")
        out = await call("movie_choose_take", ["kind": "sheet", "character": doc.movie.characters[0].id, "take": 1])
        check(!out.isError, "movie_choose_take chose a sheet")

        // Deleting is the one thing that asks (its effect is destructive).
        check(tools["movie_delete_take"]?.effect == .destructive, "deleting a take asks the user")

        // Your click in the window pauses the agent at its next action.
        if let chat = studio.chat, let main = studio.mainWindow {
            chat.liveTurnID = UUID()  // as if a reply were running
            let p = CGPoint(x: main.frame.midX, y: main.frame.minY + 40)
            if let e = NSEvent.mouseEvent(with: .leftMouseDown, location: main.convertPoint(fromScreen: p), modifierFlags: [],
                                          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: main.windowNumber,
                                          context: nil, eventNumber: 0, clickCount: 1, pressure: 1) {
                NSApp.postEvent(e, atStart: false)
            }
            await settle(0.3)
            check(studio.gate.isPaused, "a click in the main window paused the agent")
            let paused = Task { @MainActor in await call("studio_open", ["what": "queue"]) }
            await settle(0.4)
            check(UserDefaults.standard.string(forKey: "sidebarPanel") != "queue", "its next action waits")
            studio.gate.resume()
            let r = await paused.value
            check(r.text.hasPrefix(StudioGate.rereadNote) && UserDefaults.standard.string(forKey: "sidebarPanel") == "queue",
                  "after Resume it acts and is told to look again")
            chat.liveTurnID = nil
        }

        // Step pace: each action waits for Next.
        studio.pace = .step
        let stepped = Task { @MainActor in await call("studio_open", ["what": "presets"]) }
        await settle(0.4)
        if case .awaitingNext = studio.gate.state { check(true, "Step pace waits for Next") } else { check(false, "Step pace waits for Next") }
        studio.gate.next()
        _ = await stepped.value
        check(UserDefaults.standard.string(forKey: "sidebarPanel") == "presets", "Next let it go")
        studio.pace = .fast

        // A workflow on the cloud GPU: its own cost sheet.
        out = await call("studio_open", ["what": "preset", "name": "cloud-flux"])
        check(model.workflowInFront && out.text.contains("cloud GPU"), "studio_open opened a cloud preset (\(out.text.prefix(70)))")
        out = await call("set_prompt", ["text": "A lighthouse at dawn"])
        check(!out.isError, "set_prompt wrote the prompt (\(out.text.prefix(60)))")

        // Graph nodes have targets of their own: the ring goes to the node, not the whole canvas.
        func ringIs(_ target: String) -> Bool {
            guard let f = MGTestAnchors.frame(target), let main = studio.mainWindow, let ring = studio.ring else { return false }
            let want = studio.local(f, in: main).insetBy(dx: -4, dy: -4)
            return abs(ring.minX - want.minX) < 2 && abs(ring.minY - want.minY) < 2 && abs(ring.width - want.width) < 2
        }
        if let pid = model.active.graph.nodes.first(where: { AppModel.promptFields[$0.type] != nil })?.id {
            let target = "graph-node:\(pid)"
            let canvas = MGTestAnchors.frame("graph-canvas") ?? .zero
            let f = MGTestAnchors.frame(target) ?? .zero
            check(f.width > 0 && f.width < canvas.width && canvas.contains(CGPoint(x: f.midX, y: f.midY)),
                  "the prompt node has its own target on the canvas (\(Int(f.width))×\(Int(f.height)))")
            check(ringIs(target), "set_prompt's ring is on the prompt node")
            // Off the canvas: no target, so the canvas pans to it first.
            let home = model.active.graph[pid]?.position
            model.active.graph[pid]?.position = GraphPoint(x: 9000, y: 9000)
            await settle(0.3)
            check(MGTestAnchors.frame(target) == nil, "a node off the canvas has no target")
            out = await call("set_prompt", ["text": "A lighthouse at dawn"])
            check(MGTestAnchors.frame(target) != nil && ringIs(target), "the canvas panned to it and ringed it")
            model.active.graph[pid]?.position = home
            await settle(0.2)
        } else {
            check(false, "the cloud preset has a prompt node")
        }
        let nodesBefore = Set(model.active.graph.nodes.map(\.id))
        out = await call("edit_graph", ["operations": [["op": "add", "type": "Note", "title": "Smoke note"]]])
        if let added = model.active.graph.nodes.first(where: { !nodesBefore.contains($0.id) }) {
            check(ringIs("graph-node:\(added.id)") && studio.revisits["smoke_\(n)"]?.target == "graph-node:\(added.id)",
                  "edit_graph rings the node it added, and its row goes back there")
            _ = await call("edit_graph", ["operations": [["op": "remove", "node": added.id]]])
        } else {
            check(false, "edit_graph added a note (\(out.text.prefix(80)))")
        }
        runner = pressRunWhenAsked()
        out = await call("run_workflow")
        check(await runner.value, "the workflow's cost sheet came up before the run")
        check(out.text.hasPrefix("Finished"), "run_workflow waited for the run (\(out.text.prefix(80)))")

        // The fallback: the window's controls, and a click on one.
        let lookStart = Date()
        let looked = await call("app_look", ["query": "sidebar-tab"])
        say(String(format: "app_look took %.2f s", Date().timeIntervalSince(lookStart)))
        check(looked.text.contains("sidebar-tab:assets"), "app_look lists the window's named controls")
        check(!looked.images.isEmpty, "with a picture of the window")
        out = await call("app_click", ["element": "sidebar-tab:assets"])
        await settle(0.5)
        check(UserDefaults.standard.string(forKey: "sidebarPanel") == "assets", "app_click clicked the Assets tab (\(out.text))")
        check(!studio.gate.isPaused, "the agent's own clicks don't pause it")

        session.finish()
        studio.replyEnded()
        model.closeStudioAgent()
        await settle(0.3)
        check(!studio.panel.isVisible, "Close hides the panel")
        model.deleteMovie(doc.id)
        say(ok ? "PASS" : "FAILED")
        return ok
    }
}
