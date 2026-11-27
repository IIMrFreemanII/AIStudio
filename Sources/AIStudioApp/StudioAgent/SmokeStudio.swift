import Agent
import AppKit
import Foundation
import SwiftUI
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
        var tools = Dictionary(uniqueKeysWithValues: StudioTools.all(session).map { ($0.name, $0) })
        // The architect's tools, with workers on a scripted model (the test plays it).
        let workers = StudioWorkers(session: session, client: SmokeWorkerModel(), model: mid, effort: nil, inlineThinking: false, vision: true)
        for t in StudioTools.architect(workers) { tools[t.name] = t }
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

        // The story, then the script: each cost sheet waits for the user's Run.
        var runner = pressRunWhenAsked()
        out = await call("movie_run", ["work": "story"])
        check(await runner.value, "the story's cost sheet came up and the test pressed Run")
        check(out.text.hasPrefix("Finished the story") && doc.movie.scenes.count == 2, "movie_run waited for the story (\(out.text.prefix(80)))")
        check(studio.waitingOnUser == nil, "the hourglass row is gone")
        out = await call("movie_get", ["part": "story"])
        check(out.text.contains("wants: keep the light burning") && out.text.contains("beat:"), "movie_get story reads the bible and outline")
        out = await call("movie_edit", ["field": "beat", "scene": "s2", "value": "she wants the top / the wind / she makes it"])
        check(doc.movie.scenes[1].beat.hasPrefix("she wants the top") && doc.stage == .story, "movie_edit typed a beat on the Story page")
        out = await call("movie_approve", ["stage": "story"])
        check(doc.movie.isApproved(.story), "movie_approve approved the story")
        runner = pressRunWhenAsked()
        out = await call("movie_run", ["work": "script"])
        check(await runner.value && out.text.hasPrefix("Finished the script") && doc.movie.scenes.allSatisfy(\.isWritten),
              "movie_run wrote every scene (\(out.text.prefix(80)))")

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
        out = await call("movie_approve", ["stage": "shots", "sequence": "1"])
        check(doc.movie.isApproved(.shots, in: "q1") && !doc.movie.isApproved(.shots, in: "q2"),
              "movie_approve approved sequence 1's shots (\(out.text.prefix(60)))")
        out = await call("movie_approve", ["stage": "shots", "sequence": "all"])
        check(doc.movie.isApproved(.shots), "and with sequence all, every sequence's")

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

        // The architect: a plan the user approves, then workers side by side, each with its own pointer.
        let title = doc.movie.title
        let planTasks: [[String: Any]] = [
            ["id": "k1", "title": "Keyframe for shot 1", "brief": "RUN keyframes shot 1 of \(title)", "paid": ["keyframes"], "estimate": 0.03],
            ["id": "k2", "title": "Keyframe for shot 2", "brief": "RUN keyframes shot 2 of \(title)", "paid": ["keyframes"], "estimate": 0.03],
            ["id": "rev", "title": "Review shot 1", "brief": "LOOK shot 1 of \(title)", "kind": "review", "depends_on": ["k1", "k2"]],
        ]
        func whenPlanShows(_ answer: StudioPlanAnswer) -> Task<Bool, Never> {
            Task { @MainActor in
                for _ in 0..<100 {
                    if studio.pendingPlan != nil {
                        await settle(0.3)
                        let shown = MGTestAnchors.frame("studio-plan-approve") != nil
                        studio.answerPlan(answer)
                        return shown
                    }
                    try? await Task.sleep(for: .milliseconds(50))
                }
                return false
            }
        }
        var answerer = whenPlanShows(.change("Fewer tasks, please"))
        out = await call("studio_plan", ["tasks": planTasks])
        check(await answerer.value, "the plan card waits in the panel with Approve")
        check(out.text.contains("Fewer tasks, please") && studio.plan == nil, "Change… sends the note back to the architect")
        answerer = whenPlanShows(.approved)
        out = await call("studio_plan", ["tasks": planTasks])
        _ = await answerer.value
        check(out.text.contains("approved") && studio.plan?.tasks.count == 3, "Approve: the plan is the panel's checklist (\(out.text.prefix(50)))")
        out = await call("delegate", ["task": "rev"])
        check(out.isError && out.text.contains("waits for"), "a task waiting on others can't start yet")

        // Two workers at once: two pointers, the second waits for the movie, cost sheets one at a time with the worker's name.
        final class Watch { var pointers = 0; var captions: [String] = []; var waited = false; var sheets = 0 }
        let watch = Watch()
        /// With AISTUDIO_SMOKE_SNAPSHOTS=<folder>, a picture of the overlay and the panel (to look at the icons).
        func snapshot(_ w: NSWindow, _ name: String) {
            guard let dir = ProcessInfo.processInfo.environment["AISTUDIO_SMOKE_SNAPSHOTS"], let v = w.contentView,
                  let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { return }
            v.cacheDisplay(in: v.bounds, to: rep)
            try? rep.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: dir).appendingPathComponent(name + ".png"))
        }
        // The faces: every actor in every mood at three sizes, on dark and on light (to look at, with snapshots on).
        if let dir = ProcessInfo.processInfo.environment["AISTUDIO_SMOKE_SNAPSHOTS"] {
            for (name, scheme, bg) in [("dark", ColorScheme.dark, Color(white: 0.06)), ("light", ColorScheme.light, Color(white: 0.97))] {
                let grid = VStack(alignment: .leading, spacing: 18) {
                    ForEach([20.0, 32.0, 44.0], id: \.self) { size in
                        HStack(spacing: size * 0.9) {
                            ForEach(0..<4, id: \.self) { n in
                                ForEach(StudioActorState.allCases, id: \.self) { st in
                                    StudioOrb(actor: StudioActor(n), state: st, size: size, pace: .watchable, still: true)
                                }
                            }
                        }
                    }
                }
                .padding(40)
                .background(bg)
                .environment(\.colorScheme, scheme)
                let r = ImageRenderer(content: grid)
                r.scale = 2
                if let img = r.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                   let png = rep.representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("orbs-\(name).png"))
                }
                // The 3D turn: the architect and Mint at 64 pt looking left, up, ahead, right and down.
                let gazes = [CGVector(dx: -1, dy: 0), CGVector(dx: 0, dy: -1), .zero, CGVector(dx: 1, dy: 0), CGVector(dx: 0.7, dy: 0.7)]
                let turns = VStack(alignment: .leading, spacing: 56) {
                    ForEach([0, 1], id: \.self) { n in
                        HStack(spacing: 64) {
                            ForEach(Array(gazes.enumerated()), id: \.offset) { _, g in
                                StudioOrb(actor: StudioActor(n), size: 64, pace: .watchable, gaze: g, still: true)
                            }
                        }
                    }
                }
                .padding(60)
                .background(bg)
                .environment(\.colorScheme, scheme)
                let gr = ImageRenderer(content: turns)
                gr.scale = 2
                if let img = gr.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                   let png = rep.representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("orbs-turn-\(name).png"))
                }
                // The ghost echoes: the architect and Mint mid-flight, Plum just landed (pulling in), the handoff
                // sparkle on its way; the orbs drawn where they are.
                let flights: [(StudioActor, StudioFlight, Double)] = [
                    (.architect, StudioFlight(from: CGPoint(x: 60, y: 60), to: CGPoint(x: 620, y: 180), pace: .watchable), 0.6),
                    (StudioActor(1), StudioFlight(from: CGPoint(x: 60, y: 300), to: CGPoint(x: 620, y: 240), pace: .watchable), 0.75),
                    (StudioActor(3), StudioFlight(from: CGPoint(x: 80, y: 420), to: CGPoint(x: 520, y: 400), pace: .watchable), 1.0),
                ]
                let groups = flights.map { a, f, at in
                    StudioEchoGroup(actor: a, echoes: StudioTrail.echoes([StudioTrailLeg(flight: f, start: 0)], now: f.duration * at + (at >= 1 ? 0.08 : 0)))
                }
                let spark = StudioFlight(from: CGPoint(x: 420, y: 60), to: CGPoint(x: 680, y: 360), bend: 0.25, duration: 0.5)
                let sparkEchoes = StudioEchoGroup(actor: .architect, echoes: StudioTrail.echoes([StudioTrailLeg(flight: spark, start: 0)], now: 0.32),
                                                  sparkle: true)
                let frame = ZStack(alignment: .topLeading) {
                    StudioEchoesView(groups: groups + [sparkEchoes])
                    ForEach(Array(flights.enumerated()), id: \.offset) { _, item in
                        let (a, f, at) = item
                        let p = f.point(at: f.duration * at)
                        let size: CGFloat = a.isArchitect ? 36 : 32
                        StudioOrb(actor: a, state: .working, size: size, pace: .watchable, still: true)
                            .offset(x: p.x - size / 2, y: p.y - size / 2)
                    }
                    let sp = spark.point(at: 0.32)
                    SparkleShape().fill(MG.accent).frame(width: 12, height: 12).offset(x: sp.x - 6, y: sp.y - 6)
                }
                .frame(width: 740, height: 480, alignment: .topLeading)
                .background(bg)
                .environment(\.colorScheme, scheme)
                let tr = ImageRenderer(content: frame)
                tr.scale = 2
                if let img = tr.nsImage, let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
                   let png = rep.representation(using: .png, properties: [:]) {
                    try? png.write(to: URL(fileURLWithPath: dir).appendingPathComponent("trail-\(name).png"))
                }
            }
        }
        var snapped = false
        let watcher = Task { @MainActor in
            while !Task.isCancelled {
                let on = studio.cursors.filter { !$0.actor.isArchitect && $0.position != nil }.count
                watch.pointers = max(watch.pointers, on)
                if on >= 2, !snapped, studio.cursors.contains(where: { !$0.actor.isArchitect && $0.caption != nil }) {
                    snapped = true
                    try? await Task.sleep(for: .milliseconds(250))
                    snapshot(studio.overlay, "studio-overlay")
                    snapshot(studio.panel, "studio-panel")
                }
                for c in studio.cursors where !c.actor.isArchitect {
                    if let cap = c.caption, cap.contains("Waiting for") { watch.waited = true }
                }
                if let w = studio.waitingOnUser, !watch.captions.contains(w) { watch.captions.append(w) }
                if let g = doc.pendingGate {
                    try? await Task.sleep(for: .milliseconds(300))
                    if let w = studio.waitingOnUser, !watch.captions.contains(w) { watch.captions.append(w) }
                    doc.pendingGate = nil
                    g.run()
                    watch.sheets += 1
                }
                try? await Task.sleep(for: .milliseconds(30))
            }
        }
        let keysBefore = doc.movie.shots.prefix(2).map(\.keyframes.count)
        n += 1
        // One delegate call with both tasks: they run side by side whether or not the model batches its calls.
        let r1 = await call("delegate", ["tasks": ["k1", "k2"]])
        await settle(0.4)
        check(r1.text.contains("Mint on") && r1.text.contains("Ember on") && r1.text.components(separatedBy: "Result: done").count == 3,
              "one delegate call ran both tasks and brought both reports (\(r1.text.prefix(80)))")
        check(studio.plan?.task("k1")?.status == .done && studio.plan?.task("k2")?.status == .done, "their tasks are ticked off")
        check(zip(doc.movie.shots.prefix(2).map(\.keyframes.count), keysBefore).allSatisfy { $0 > $1 }, "each made its shot's keyframe")
        check(watch.pointers >= 2, "two worker pointers were on screen at once (\(watch.pointers))")
        check(watch.waited, "the second worker waited for the movie to be free")
        check(watch.sheets == 2 && watch.captions.contains { $0.hasPrefix("Mint · ") } && watch.captions.contains { $0.hasPrefix("Ember · ") },
              "two cost sheets, one at a time, each labelled with its worker (\(watch.captions))")
        check(r1.text.contains("Steps:\n✓ ") && studio.workerSteps["k1"]?.isEmpty == false, "the worker's steps are in its row and saved with the chat")
        check(studio.panelFrame("studio-task-icon:k1") != nil,
              "the task row is the worker's dock, outside the main window (\(MGTestAnchors.names(prefix: "studio-task-icon")), \(String(describing: MGTestAnchors.frame("studio-task-icon:k1"))), panel \(MGTestAnchors.window("studio-task-icon:k1") === studio.panel))")
        await settle(1.2)
        check(studio.cursor(.worker(1)).position == nil && studio.cursor(.worker(1)).state == .idle, "the finished worker flew home and docked")

        // Off screen: while the architect holds the workflow page, the reviewer looks at shot 1 without navigating.
        let wfID = model.active.id
        await studio.go(.workflow(wfID))
        let front = model.frontItem
        try? await studio.holdScreen(.workflow(wfID))
        var offscreen = false
        let offWatch = Task { @MainActor in
            while !Task.isCancelled {
                if studio.cursors.contains(where: { !$0.actor.isArchitect && $0.offscreen }) { offscreen = true }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        out = await call("delegate", ["task": "rev"])
        offWatch.cancel()
        studio.releaseScreen()
        check(out.text.contains("Result: done") && offscreen && model.frontItem == front && model.chatViewer == nil,
              "a worker on another page acted off screen: the page in front stayed (off \(offscreen), front kept \(model.frontItem == front), viewer \(model.chatViewer == nil ? "shut" : "open"): \(out.text.prefix(160)))")
        _ = await call("delegate", ["task": "rev", "feedback": "Look again"])
        _ = await call("delegate", ["task": "rev", "feedback": "And once more"])
        out = await call("delegate", ["task": "rev", "feedback": "A third time"])
        check(out.isError && out.text.contains("do it yourself"), "a third retry is refused")
        watcher.cancel()

        // A pause stops every actor; Resume lets them all go.
        studio.gate.pauseByUser()
        final class Ran { var count = 0 }
        let ran = Ran()
        let acts = [StudioActor.architect, .worker(1), .worker(2)].map { a in
            Task { @MainActor in
                try? await StudioActorScope.$actor.withValue(a) {
                    try await studio.act(StudioStep(doing: "Waiting", done: "Waited")) { ran.count += 1 }
                }
            }
        }
        await settle(0.3)
        check(ran.count == 0, "a pause holds the architect and every worker")
        studio.gate.resume()
        for t in acts { await t.value }
        check(ran.count == 3, "Resume lets them all go")

        // The overlay: the whole display, above the panel, and it hides when another app is in front.
        check(studio.overlay.frame == studio.displayFrame, "the overlay covers the main window's display")
        check(studio.overlay.level.rawValue > studio.panel.level.rawValue && studio.overlay.ignoresMouseEvents,
              "it sits above the panel and never takes a click")
        check(studio.overlay.hidesOnDeactivate, "it hides while another app is in front")

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

/// The workers' model in the smoke test: a brief "RUN keyframes shot N of T" runs them, "LOOK shot N of T" looks at
/// them; after a tool result it reports.
final class SmokeWorkerModel: ChatStreaming, @unchecked Sendable {
    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        let lastIsTool = request.messages.contains { $0.role == .tool }  // (look_at's images follow as a user message)
        let brief = request.messages.first { $0.role == .user }?.text ?? ""
        return AsyncThrowingStream { c in
            Task {
                try? await Task.sleep(for: .milliseconds(80))
                if lastIsTool {
                    c.yield(.content("Did it.\nResult: done"))
                } else {
                    let parts = brief.components(separatedBy: " of ")
                    let movie = parts.count > 1 ? parts[1] : ""
                    let shot = Int(parts[0].split(separator: " ").last ?? "") ?? 1
                    let json: [String: Any]
                    let name: String
                    if brief.hasPrefix("RUN") {
                        name = "movie_run"
                        json = ["movie": movie, "work": "keyframes", "shots": [shot]]
                    } else {
                        name = "look_at"
                        json = ["movie": movie, "kind": "keyframe", "shot": shot]
                    }
                    let args = String(data: (try? JSONSerialization.data(withJSONObject: json)) ?? Data(), encoding: .utf8) ?? "{}"
                    c.yield(.toolCall(index: 0, id: "w\(UUID().uuidString.prefix(6))", name: name, arguments: args))
                }
                c.finish()
            }
        }
    }
}
