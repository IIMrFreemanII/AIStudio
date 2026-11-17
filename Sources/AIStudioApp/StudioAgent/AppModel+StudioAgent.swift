import Agent
import AppKit
import Foundation
import LLMClient
import MovieCore
import NodesRemote

/// The Studio agent: its chat in the floating panel, its reply (the studio tools and their prompt), and the cost sheet
/// for workflow runs it starts on the cloud GPU.
extension AppModel {
    /// Opens the panel on the Studio agent chat used last (or `id`, a Studio agent session from Sessions), else a new one.
    func openStudioAgent(_ id: UUID? = nil) {
        studio.model = self
        if let id, studio.chat?.id != id, let t = chatStore.load(id), t.studioAgent == true {
            if let old = studio.chat {
                if old.isStreaming { stopReply(old) }
                saveChat(old)
            }
            var t = t
            t.stopRunningTools()
            studio.chat = ChatDocument(transcript: t)
        }
        if studio.chat == nil {
            if let s = UserDefaults.standard.string(forKey: "studioAgentChat"), let uuid = UUID(uuidString: s), !Self.isAutomatedRun,
               var t = chatStore.load(uuid), t.studioAgent == true {
                t.stopRunningTools()
                studio.chat = ChatDocument(transcript: t)
            } else {
                studio.chat = newStudioChat()
            }
        }
        rememberStudioChat()
        studio.installMonitor()
        studio.panel.show(self)
        studio.panelShown = true
    }

    func toggleStudioAgent() {
        if studio.panelShown, studio.panel.isVisible { closeStudioAgent() } else { openStudioAgent() }
    }

    func closeStudioAgent() {
        studio.panel.close()
    }

    /// The panel's New button: this conversation is saved (it stays in Sessions) and a new one starts.
    func newStudioConversation() {
        if let old = studio.chat {
            if old.isStreaming { return }
            saveChat(old)
        }
        studio.reverts = [:]
        studio.revisits = [:]
        studio.reverted = []
        studio.chat = newStudioChat()
        rememberStudioChat()
    }

    private func newStudioChat() -> ChatDocument {
        let k = defaultChatModel
        var t = ChatTranscript(serverID: k?.serverID, modelID: k?.modelID)
        t.studioAgent = true
        t.title = "Studio Agent"
        t.permissionMode = PermissionMode.acceptEdits.rawValue
        return ChatDocument(transcript: t)
    }

    private func rememberStudioChat() {
        guard persistSettings, !Self.isAutomatedRun, let id = studio.chat?.id else { return }
        UserDefaults.standard.set(id.uuidString, forKey: "studioAgentChat")
    }

    // MARK: the reply

    func studioReply(_ doc: ChatDocument, key: LLMKey, client: LLMServerClient, add: @escaping (ChatTurn) -> Void) {
        studio.model = self
        let vision = llm.model(key)?.vision == true
        let session = StudioSession(app: self, vision: vision)
        // Workers: the same model and GPU with less thinking ("low", or "off" for on/off thinking).
        let efforts = llm.model(key)?.efforts ?? []
        let effort = efforts.contains("low") ? "low" : efforts.contains("off") ? "off" : nil
        let workers = StudioWorkers(session: session, client: studioWorkerClient ?? client.chat, model: key.modelID, effort: effort,
                                    inlineThinking: llm.model(key)?.reasoningParser == nil, vision: vision)
        var setup = AgentSetup(tools: StudioTools.all(session) + StudioTools.architect(workers), root: library.root) { [weak self] in
            self.map { StudioTools.systemPrompt($0, vision: vision) } ?? ""
        }
        setup.saveImages = true
        setup.finished = { [weak self, weak doc] in
            session.finish()
            self?.studio.replyEnded()
            if let doc { self?.saveChat(doc) }
        }
        studio.gate.reset()
        runAgent(doc, key: key, client: client, setup: setup, add: add)
    }

    // MARK: workflow runs on the cloud GPU

    /// The cost sheet for a workflow whose Cloud GPU nodes run `tasks`: a rough figure from the server's prices and the
    /// times these tasks took before.
    func workflowRunEstimate(tasks: [String], workflow: String) async -> CostEstimate {
        var lines: [CostEstimate.Line] = []
        var notes: [String] = []
        let server = connections.gpuServers.first
        if let server { await connections.test(server.id) }
        let info: GPUServerInfo? = server.flatMap {
            if case .gpuOnline(let i)? = connections.status[$0.id] { return i }
            return nil
        }
        let timings = GPUTimings.load(gpuTimingsURL)
        let idle = Double(info?.idleSeconds ?? 120)
        var counts: [String: Int] = [:]
        for t in tasks { counts[t, default: 0] += 1 }
        for (task, count) in counts.sorted(by: { $0.key < $1.key }) {
            let t = info?.task(task)
            let price = t?.pricePerSecond ?? t.flatMap { MovieCost.price(gpu: $0.gpu) } ?? 0
            let warm = timings.isWarm(task: task, idleSeconds: idle)
            let seconds = Double(count) * timings.seconds(task: task, megapixels: 1) + (warm ? 0 : timings.loadSeconds(task: task)) + idle
            lines.append(CostEstimate.Line(label: t?.title ?? task, count: count, gpu: t?.gpu ?? "GPU", seconds: seconds, usd: seconds * price))
        }
        notes.append("A rough figure: the job time is what this model took before (or a first guess), plus loading it when "
                     + "it's cold and the \(Int(idle)) s the GPU stays up after the last job. Billing is per GPU second.")
        if info == nil { notes.append("The GPU server didn't answer, so its prices aren't known.") }
        return CostEstimate(title: "Run “\(workflow)” on the cloud GPU", lines: lines, notes: notes)
    }
}

/// The Studio agent's tools and its system prompt.
enum StudioTools {
    static func p(_ props: [String: (String, String)], required: [String] = []) -> LLMClient.JSONValue {
        GraphHelper.params(props, required: required)
    }

    @MainActor
    static func all(_ s: StudioSession) -> [any AgentTool] {
        let movieArg = ("string", "The movie's title (default: the one in front)")
        let shotArg = ("integer", "The shot's number, from 1")
        let takeArg = ("integer", "The take's number, from 1 (default: the chosen one)")
        return [
            TodoWriteTool(),
            StudioTool("studio_state", effect: .read,
                       "Where AIStudio is now: what's in front (a movie and its page, a workflow), the sidebar, open items, running work, a cost sheet waiting for the user, and a summary of the page in front. Call it first, and again after the user used the window.",
                       p([:]), title: { _ in "Looked at the app" }) { _, _ in ToolOutput(try await s.state()) },
            StudioTool("studio_open",
                       "Brings something to the front, moving the pointer there so the user sees it: a sidebar panel (workflows, movies, chats, presets, assets, queue, nodes, servers), a movie (name; stage optional: idea, script, shots, cast, keyframes, voices, clips, sound, edit), a saved or open workflow (name), a preset as a new workflow (name or id), or close_viewer.",
                       p(["what": ("string", "workflows | movies | chats | presets | assets | queue | nodes | servers | movie | workflow | preset | close_viewer"),
                          "name": ("string", "The movie, workflow or preset"), "stage": ("string", "A movie page")], required: ["what"]),
                       title: { a in "Opened \(a.string("name") ?? a.string("what") ?? "")" }) { a, id in
                ToolOutput(try await s.open(a, id))
            },
            StudioTool("studio_list", effect: .read,
                       "Lists movies, workflows (saved and open), presets (ids, kind, model, cloud or this Mac) or servers (the GPU server's models with GPU and price per second).",
                       p(["what": ("string", "movies | workflows | presets | servers")], required: ["what"]),
                       title: { a in "Listed \(a.string("what") ?? "")" }) { a, _ in ToolOutput(try await s.list(a.string("what") ?? "")) },
            StudioTool("movie_create",
                       "Makes a new movie and types its idea (the pitch). Then movie_run with work story writes its bible and outline.",
                       p(["idea": ("string", "The pitch: who, what happens, the feeling"), "title": ("string", "Optional"),
                          "style": ("string", "Visual style (optional)"), "aspect": ("string", "Optional aspect ratio"),
                          "seconds": ("integer", "Target length in seconds (optional)")], required: ["idea"]),
                       title: { _ in "Made a new movie" }) { a, id in ToolOutput(try await s.movieCreate(a, id)) },
            StudioTool("movie_get", effect: .read,
                       "Reads part of a movie: overview (stages and what's approved), story (premise, characters' goals and arcs, places, the outline of sequences and scenes with their beats), script (characters, scenes, dialogue, the editor's notes), shots (every shot's prompts, lines and takes), cast (looks, voices, sheets) or music.",
                       p(["movie": movieArg, "part": ("string", "overview | story | script | shots | cast | music"), "sequence": ("string", "For a film in sequences: its number (1…) or id; per-sequence stages (shots, keyframes, voices, clips, sound) act on it (default: the one shown)")]),
                       title: { a in "Read the \(a.string("part") ?? "overview")" }) { a, _ in ToolOutput(try await s.movieGet(a)) },
            StudioTool("movie_edit",
                       "Types new text into a movie field, on its page: title, idea, style, logline; premise, tone, theme; a character's goal, flaw or arc; a scene's beat (scene); a place's location_look (location); a shot's keyframe_prompt, motion_prompt, sfx or line (shot + line); a character's look or voice; a scene's music_prompt. Changing approved work takes back that stage's approval (and later ones); the result says which.",
                       p(["movie": movieArg, "field": ("string", "title | idea | style | logline | premise | tone | theme | goal | flaw | arc | beat | location_look | keyframe_prompt | motion_prompt | sfx | line | look | voice | music_prompt"),
                          "shot": shotArg, "line": ("integer", "The line's number in the shot, from 1"),
                          "character": ("string", "The character's id or name"), "scene": ("string", "The scene's id"),
                          "location": ("string", "The place's id or name"),
                          "value": ("string", "The new text")], required: ["field", "value"]),
                       title: { a in "Typed the \((a.string("field") ?? "").replacingOccurrences(of: "_", with: " "))" }) { a, id in
                ToolOutput(try await s.movieEdit(a, id))
            },
            StudioTool("movie_run",
                       "Starts a stage's work (a long film's per-sequence stages for one sequence at a time): story (bible and outline), script (each scene, then the critique; scenes: [ids] for only those), shots, cast (character sheets and the places' plates), keyframes, voices (voice samples), lines (spoken dialogue), clips, effects or music. Pass shots / characters / scenes to make (or retake) only those. Paid work opens the app's cost sheet and waits: only the user can press Run there. Then it waits for the work to finish and says what was made.",
                       .object([
                           "type": .string("object"),
                           "properties": .object([
                               "movie": .object(["type": .string("string"), "description": .string("The movie's title (default: the one in front)")]),
                               "work": .object(["type": .string("string"), "description": .string("story | script | shots | cast | keyframes | voices | lines | clips | effects | music")]),
                               "shots": .object(["type": .string("array"), "items": .object(["type": .string("integer")]), "description": .string("Shot numbers to (re)make (optional)")]),
                               "characters": .object(["type": .string("array"), "items": .object(["type": .string("string")]), "description": .string("Character ids (optional)")]),
                               "scenes": .object(["type": .string("array"), "items": .object(["type": .string("string")]), "description": .string("Scene ids (optional)")]),
                               "sequence": .object(["type": .string("string"), "description": .string("For a film in sequences: the sequence's number (1…) or id to make it for (default: the one shown)")]),
                           ]),
                           "required": .array([.string("work")]),
                       ]),
                       title: { a in "Made \(a.string("work") ?? "")" }) { a, id in ToolOutput(try await s.movieRun(a, id)) },
            StudioTool("movie_choose_take",
                       "Chooses which take a movie uses: kind keyframe, clip or effects (shot), line (shot + line), sheet or voice (character), music (scene).",
                       p(["movie": movieArg, "kind": ("string", "keyframe | clip | effects | line | sheet | voice | music"),
                          "shot": shotArg, "line": ("integer", "The line's number"), "character": ("string", "The character"),
                          "scene": ("string", "The scene's id"), "take": ("integer", "The take's number, from 1")], required: ["kind", "take"]),
                       title: { a in "Chose \(a.string("kind") ?? "a") take \(a.int("take") ?? 0)" }) { a, id in ToolOutput(try await s.chooseTake(a, id)) },
            StudioTool("movie_approve",
                       "Approves a movie stage (story, script, shots, cast, keyframes, voices, clips, sound, edit), which unlocks the next page. Approve only after checking the work.",
                       p(["movie": movieArg, "stage": ("string", "The stage"), "sequence": ("string", "For a film in sequences: its number (1…) or id; per-sequence stages (shots, keyframes, voices, clips, sound) act on it (default: the one shown)")], required: ["stage"]),
                       title: { a in "Approved \(a.string("stage") ?? "")" }) { a, id in ToolOutput(try await s.approve(a, id)) },
            StudioTool("movie_delete_take", effect: .destructive,
                       "Deletes a take (its file goes to the Trash): kind keyframe, clip or effects (shot), or sheet (character). The user approves each delete.",
                       p(["movie": movieArg, "kind": ("string", "keyframe | clip | effects | sheet"), "shot": shotArg,
                          "character": ("string", "The character"), "take": ("integer", "The take's number, from 1")], required: ["kind", "take"]),
                       title: { a in "Delete \(a.string("kind") ?? "the") take \(a.int("take") ?? 0)" }) { a, id in ToolOutput(try await s.deleteTake(a, id)) },
            StudioTool("movie_export",
                       "Exports the finished film as an MP4 in ~/Movies under a new name (never over an existing file).",
                       p(["movie": movieArg, "name": ("string", "The file name without .mp4 (default: the title)")]),
                       title: { _ in "Exported the film" }) { a, id in ToolOutput(try await s.export(a, id)) },
            StudioTool("look_at", effect: .read,
                       "Looks at a result, opening it in the viewer while it does: a shot's keyframe or clip (4 frames), a character's sheet, or outputs (the workflow's latest results). Use it to check work before approving or going on.",
                       p(["movie": movieArg, "kind": ("string", "keyframe | clip | sheet | outputs"), "shot": shotArg,
                          "character": ("string", "The character"), "take": takeArg, "count": ("integer", "outputs: how many, 1–4")], required: ["kind"]),
                       title: { a in "Looked at the \(a.string("kind") ?? "result")" }) { a, id in try await s.look(a, id) },
            StudioTool("get_graph", effect: .read,
                       "The workflow in front: every node with its id, type, inputs, links and outputs, and whether it can run.",
                       p([:]), title: { _ in "Read the workflow" }) { _, _ in ToolOutput(try await s.helper().getGraph()) },
            StudioTool("edit_graph",
                       "Changes the workflow in front with operations (all or nothing): {\"op\":\"set\",\"node\":\"sampler\",\"input\":\"steps\",\"value\":8}, add, connect, disconnect, remove, rename — as the graph helper's edit_graph. A reply's edits are one undo step.",
                       .object(["type": .string("object"),
                                "properties": .object(["operations": .object(["type": .string("array"), "items": .object(["type": .string("object")])])]),
                                "required": .array([.string("operations")])]),
                       title: { a in "Changed the workflow (\((a.raw["operations"] as? [Any])?.count ?? 0))" }) { a, id in
                guard let ops = a.raw["operations"] as? [Any], !ops.isEmpty else { throw ToolError("\"operations\" is missing") }
                return ToolOutput(try await s.editGraph(ops, id))
            },
            StudioTool("set_prompt",
                       "Writes the prompt of the workflow in front (node: which prompt node, when there are several).",
                       p(["text": ("string", "The prompt"), "node": ("string", "The node id (optional)")], required: ["text"]),
                       title: { _ in "Wrote the prompt" }) { a, id in
                let text = try a.required("text")
                let node = a.string("node")
                return ToolOutput(try await s.setPrompt(text, node: node?.isEmpty == false ? node : nil, id))
            },
            StudioTool("run_workflow",
                       "Runs the workflow in front once and waits for it. With Cloud GPU nodes it first shows a cost sheet and waits for the user's Run. Then look_at kind outputs shows the result.",
                       p([:]), title: { _ in "Ran the workflow" }) { _, id in ToolOutput(try await s.runWorkflow(id)) },
            StudioTool("get_errors", effect: .read,
                       "Why the workflow in front can't run, and which nodes failed in its last run.",
                       p([:]), title: { _ in "Checked for errors" }) { _, _ in ToolOutput(try await s.helper().errors()) },
            StudioTool("list_models", effect: .read,
                       "The models installed on this Mac and the library's LoRAs (for workflow inputs).",
                       p([:]), title: { _ in "Listed the models" }) { _, _ in ToolOutput(try await s.helper().models()) },
        ] + StudioFallback.tools(s)
    }

    @MainActor
    static func systemPrompt(_ app: AppModel, vision: Bool) -> String {
        let state = (try? StudioSession(app: app, vision: vision).state()) ?? ""
        return """
        You are the Studio agent in AIStudio, a Mac app for making images, video, sound and short films with AI models \
        (many of them on a cloud GPU server billed per second). You use the app for the user while they watch: each \
        of your actions moves a pointer to the control it uses and shows a caption, so act like a careful person at \
        the keyboard, one clear step at a time.

        How to work:
        - Start with studio_state. Open things with studio_open; read with movie_get, get_graph and studio_list.
        - Movies go stage by stage: idea → story → script → shots → cast → keyframes → voices → clips → sound → edit. \
        Each stage's work is made with movie_run, checked, then approved with movie_approve, which unlocks the next. \
        The story is the bible and outline (read it with movie_get story); the script is written from it a scene at \
        a time. A long film is in sequences: shots, keyframes, voices, clips and sound go one sequence at a time \
        (pass sequence).
        - Image, video, sound, music and speech outside movies are workflows: open a preset (studio_list presets) \
        with studio_open, write its prompt with set_prompt, change settings with edit_graph, run it with run_workflow.
        - Paid work (anything on the cloud GPU, and movie stages that use the model) shows the app's cost sheet. You \
        can't press Run: say what it will cost and wait; the tool returns when the user decides. If they cancel, ask \
        how to go on — don't start it again on your own.
        - \(vision ? "Check what you made with look_at before approving or going on: the right number of people, who they are, faces and clothes matching the cast, the setting, and nothing broken (doubled people, extra limbs, garbled text). Retake what's wrong with movie_run and the item, after editing its prompt if the prompt caused it." : "This model can't see images: after making images or clips, ask the user to check them before approving.")
        - Deleting asks the user. Exports never overwrite files.
        - If the user used the window while you worked, a tool result says so: call studio_state before going on.
        - The app_* tools click and type in the window like a person. Use them only when no other tool does the step \
        (look first with app_look).
        - Keep your messages short: say what you're about to do and what you found. Use todo_write for longer jobs.

        You're the architect, with up to \(StudioPlan.maxWorkers) workers (the same model, thinking less, each with its own pointer):
        - A job with two or more parts that can go on side by side (keyframes for different shots, reviewing many \
        takes, voices and music) goes through studio_plan: tasks with complete briefs (a worker sees only its brief: \
        say which movie, shots, what to write or check, and what a good result looks like), their paid stages and a \
        rough cost. Wait for the user's answer; on a change note, plan again.
        - Then delegate every ready task in one call (tasks: [ids]) so they run at the same time. Workers can read, edit \
        prompts, run stages (the user still presses Run on each cost sheet), choose takes and look_at results; they \
        can't navigate on their own, approve, export, delete or ask the user. Workflow tasks need the screen: give \
        them to one worker at a time, or do them yourself.
        - Check each report (look_at, movie_get) before going on. If it's wrong, delegate the same task again with \
        specific feedback, at most twice; then do it yourself, or ask the user when the fix needs another paid run.
        - Small requests (one step or two): just do them yourself, without a plan.

        The app now:
        \(state)
        """
    }
}
