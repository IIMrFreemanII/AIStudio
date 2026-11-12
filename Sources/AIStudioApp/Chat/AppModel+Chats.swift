import Agent
import AppKit
import Foundation
import ImageIO
import LLMClient

/// Chat tabs: opening, closing, sending, and saving them.
extension AppModel {
    var chatStore: ChatStore { ChatStore(root: library.chats) }

    func reloadChats() { chatSummaries = chatStore.list() }

    /// A new chat with the given model (else the last one used, else the first installed one), in front.
    @discardableResult
    func newChat(model key: LLMKey? = nil) -> ChatDocument {
        let k = key ?? defaultChatModel
        let doc = ChatDocument(transcript: ChatTranscript(serverID: k?.serverID, modelID: k?.modelID))
        chats.append(doc)
        activeChatID = doc.id
        return doc
    }

    /// A coding chat in a project folder: the agent's tools work there, with the permission mode chosen in the chat.
    @discardableResult
    func newCodingChat(project: URL, model key: LLMKey? = nil) -> ChatDocument {
        let doc = newChat(model: key)
        doc.transcript.projectPath = project.standardizedFileURL.path
        doc.transcript.title = project.lastPathComponent
        doc.mode = .ask
        return doc
    }

    /// Asks for a project folder, then opens a coding chat in it.
    func chooseProjectAndChat() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.prompt = "Open in Chat"
        panel.message = "The coding agent reads, edits and runs commands in this folder (asking first, by default)."
        let box = NSButton(checkboxWithTitle: "Work in a new git worktree", target: nil, action: nil)
        box.state = (UserDefaults.standard.object(forKey: "newChatWorktree") as? Bool ?? true) ? .on : .off
        let holder = NSView(frame: NSRect(x: 0, y: 0, width: 300, height: 36))
        box.frame = NSRect(x: 10, y: 6, width: 280, height: 24)
        holder.addSubview(box)
        panel.accessoryView = holder
        panel.isAccessoryViewDisclosed = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let wanted = box.state == .on
        if persistSettings { UserDefaults.standard.set(wanted, forKey: "newChatWorktree") }
        guard wanted else { newCodingChat(project: url); return }
        Task { @MainActor in
            guard let repo = await GitWorktree.repoRoot(url) else { newCodingChat(project: url); return }
            let f = DateFormatter()
            f.dateFormat = "yyMMdd-HHmm"
            let base = await GitWorktree.currentBranch(repo)
            do {
                let made = try await GitWorktree.create(repo: repo, branch: worktreeBranchPrefix + "session-" + f.string(from: Date()),
                                                        base: base)
                let doc = newCodingChat(project: made.path)
                doc.transcript.title = url.lastPathComponent
                doc.transcript.worktree = SessionWorktree(repo: repo.path, path: made.path.path, branch: made.branch,
                                                          base: base ?? "HEAD")
                refreshGitSoon(doc)
            } catch {
                let doc = newCodingChat(project: url)
                doc.attachNote = "Couldn't make a worktree: \(error.localizedDescription)"
            }
        }
    }

    /// Opens a saved chat (or brings its tab to the front).
    func openChat(_ id: UUID) {
        if let doc = chats.first(where: { $0.id == id }) {
            doc.tabOpen = true
            activeChatID = id
            return
        }
        // A tab's graph helper opens docked on that tab.
        if let wf = tabs.first(where: { $0.helperChatID == id }) {
            activate(wf.id)
            activeChatID = nil
            showHelper = true
            return
        }
        guard var t = chatStore.load(id) else { reloadChats(); return }
        t.stopRunningTools()
        chats.append(ChatDocument(transcript: t))
        activeChatID = id
    }

    func activateChat(_ id: UUID) {
        guard let doc = chats.first(where: { $0.id == id }) else { return }
        doc.tabOpen = true
        activeChatID = id
    }

    func closeChat(_ id: UUID) {
        guard let i = chats.firstIndex(where: { $0.id == id }) else { return }
        let doc = chats[i]
        let tabs = chats.filter(\.tabOpen)
        let position = tabs.firstIndex { $0 === doc } ?? 0
        if doc.isBusy {
            // A working session goes on in the background: its tab closes, nothing stops.
            doc.tabOpen = false
            saveChat(doc)
        } else {
            doc.task?.cancel()
            doc.jobs.stopAll()
            saveChat(doc)
            chats.remove(at: i)
        }
        if activeChatID == id {
            let rest = chats.filter(\.tabOpen)
            activeChatID = rest.isEmpty ? nil : rest[min(position, rest.count - 1)].id
        }
    }

    /// Saves and unloads a hidden session that has nothing left going on.
    func unloadIfHidden(_ doc: ChatDocument) {
        guard !doc.tabOpen, !doc.isBusy, !doc.compacting, doc.task == nil, activeChatID != doc.id else { return }
        saveChat(doc)
        chats.removeAll { $0 === doc }
    }

    func deleteChat(_ id: UUID) {
        if let doc = chats.first(where: { $0.id == id }) {
            stopReply(doc)
            doc.jobs.stopAll()
            chats.removeAll { $0.id == id }
            if activeChatID == id { activeChatID = chats.last { $0.tabOpen }?.id }
        }
        chatStore.delete(id, useTrash: !Self.isAutomatedRun)
        reloadChats()
    }

    /// Saves a chat that has something in it.
    func saveChat(_ doc: ChatDocument) {
        guard !doc.transcript.turns.isEmpty else { return }
        try? chatStore.save(doc.transcript)
        reloadChats()
    }

    var lastChatModel: LLMKey? {
        get {
            guard let d = UserDefaults.standard.data(forKey: "lastChatModel") else { return nil }
            return try? JSONDecoder().decode(LLMKey.self, from: d)
        }
        set {
            guard persistSettings else { return }
            UserDefaults.standard.set(newValue.flatMap { try? JSONEncoder().encode($0) }, forKey: "lastChatModel")
        }
    }

    func setModel(_ key: LLMKey, for doc: ChatDocument) {
        doc.transcript.serverID = key.serverID
        doc.transcript.modelID = key.modelID
        lastChatModel = key
    }

    // MARK: sending

    /// Sends the draft as a user message and streams the reply.
    func send(_ doc: ChatDocument) {
        let text = doc.draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // A question card is open: what's typed answers it ("Other" for each question), and the reply goes on.
        if let q = doc.question {
            guard !text.isEmpty else { return }
            doc.draft = ""
            doc.answerQuestion(q.questions.map { _ in UserAnswer(other: text) })
            return
        }
        if text == "/compact", !doc.transcript.turns.isEmpty {
            doc.draft = ""
            compact(doc)
            return
        }
        guard !text.isEmpty || !doc.pending.isEmpty, !doc.isStreaming, !doc.compacting, doc.preparingVideos == 0 else { return }
        doc.draft = ""
        if doc.transcript.turns.isEmpty {
            doc.transcript.title = ChatTranscript.title(from: text.isEmpty ? doc.pending.first?.name ?? "" : text)
        }
        let files = doc.pending.compactMap { if case .file(let f) = $0 { return f }; return nil }
        var turn = ChatTurn(role: .user, text: text, images: doc.pending.compactMap {
            if case .image(let name) = $0 { return name }; return nil
        }, files: files.isEmpty ? nil : files)
        turn.quote = doc.replyTo
        turn.context = widgetContext(doc)
        // A coding chat's `/skill args`: the typed text stays, the model also gets the skill's instructions.
        if let root = doc.projectURL, let expansion = SkillStore.expand(text, skills: loadSkills(root: root)) {
            turn.context = [turn.context, expansion].compactMap { $0 }.joined(separator: "\n\n")
        }
        doc.transcript.append(turn)
        doc.pending = []
        doc.replyTo = nil
        doc.autoContinues = 0
        reply(doc)
    }

    /// Sends `text` as the user's message (a follow-up chip, a widget's `sendPrompt`), unless a reply is running.
    func sendText(_ doc: ChatDocument, _ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, !doc.isStreaming, doc.question == nil else { return }
        let draft = doc.draft
        doc.draft = t
        send(doc)
        doc.draft = draft  // what was being written stays
    }

    /// Starts a reply to a passage: the quote bar above the composer.
    func quote(_ doc: ChatDocument, _ text: String, from turn: ChatTurn?) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        doc.replyTo = ChatQuote(turnID: turn?.id, role: turn?.role ?? .assistant, text: t)
    }

    /// A reply to a selection: the source is the newest shown message (or tool result) that contains it.
    func quoteSelection(_ doc: ChatDocument, _ text: String) {
        let needle = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let source = doc.transcript.turns.last { $0.text.contains(needle) || ($0.toolDetail ?? "").contains(needle) }
        quote(doc, needle, from: source)
    }

    /// Shows the turn a quote came from: its branch, scrolled to and flashed.
    func showQuoted(_ doc: ChatDocument, _ quote: ChatQuote) {
        guard let id = quote.turnID, doc.transcript.turn(id) != nil else { return }
        if !doc.transcript.turns.contains(where: { $0.id == id }) { doc.transcript.select(id) }
        // A tool result is shown on the card of the assistant turn that made the call.
        var target = id
        if let t = doc.transcript.turn(id), t.role == .tool, let call = t.toolCallID,
           let owner = doc.transcript.turns.last(where: { $0.toolCalls.contains { $0.id == call } }) {
            target = owner.id
        }
        doc.highlightTurnID = target
        Task { @MainActor [weak doc] in
            try? await Task.sleep(for: .seconds(1.2))
            if doc?.highlightTurnID == target { doc?.highlightTurnID = nil }
        }
    }

    /// Widgets whose state changed since the last message, for the model (hidden in the user's message).
    func widgetContext(_ doc: ChatDocument) -> String? {
        let changed = doc.widgetStates.filter { doc.sentWidgetStates[$0.key] != $0.value }
        guard !changed.isEmpty else { return nil }
        doc.sentWidgetStates.merge(changed) { _, new in new }
        let calls = doc.transcript.turns.flatMap(\.toolCalls)
        let lines = changed.sorted { $0.key < $1.key }.map { id, state -> String in
            let title = calls.last { $0.id == id && $0.name == "show_widget" }
                .flatMap { try? ToolArgs(json: $0.arguments).string("title") } ?? "widget"
            return "[Widget \"\(title)\" state: \(state)]"
        }
        return lines.joined(separator: "\n")
    }

    /// Sends an edited version of a user message: a new branch from the same point, then a reply.
    func resend(_ doc: ChatDocument, editing id: UUID, text: String) {
        guard !doc.isStreaming, let old = doc.transcript.turn(id), old.role == .user else { return }
        var turn = ChatTurn(role: .user, text: text, images: old.images, files: old.files)
        turn.date = Date()
        doc.transcript.branch(from: id, with: turn)
        reply(doc)
    }

    /// Another reply in place of `id` (kept as a sibling), from the same messages before it.
    func regenerate(_ doc: ChatDocument, _ id: UUID) {
        guard !doc.isStreaming, let old = doc.transcript.turn(id), old.role == .assistant else { return }
        // Send only up to the message it answers (that message's newest branch may go on past it).
        doc.transcript.activeLeaf = old.parent
        reply(doc, nextTo: id)
    }

    /// Streams an assistant reply after the shown branch (or next to `sibling`, a retry).
    func reply(_ doc: ChatDocument, nextTo sibling: UUID? = nil) {
        func add(_ t: ChatTurn) {
            if let sibling { doc.transcript.branch(from: sibling, with: t) } else { doc.transcript.append(t) }
        }
        guard let key = doc.modelKey, let client = llm.client(key.serverID) else {
            var t = ChatTurn(role: .assistant, text: "")
            t.error = llm.servers.isEmpty ? "Add a GPU server in Servers first." : "Choose a model for this chat."
            add(t)
            return
        }
        lastChatModel = key
        if let root = doc.projectURL {
            let skills = loadSkills(root: root)
            let inline = llm.model(key)?.reasoningParser == nil
            let task = TaskTool(client: client.chat, model: key.modelID, tools: TaskTool.defaultTools + [WebFetchTool()],
                                reasoningEffort: doc.transcript.reasoningEffort, inlineThinking: inline)
            var tools: [any AgentTool] = CodingTools.all + [WebFetchTool(), ExitPlanTool(), task] + SessionTools.all
            if let k = searchAPIKey { tools.append(WebSearchTool(apiKey: k)) }
            if !skills.isEmpty { tools.append(UseSkillTool(skills: skills)) }
            let coAuthor = gitCoAuthorLine(for: doc)
            var setup = AgentSetup(tools: tools, root: root) {
                let base = await ProjectContext.systemPrompt(root: root, mode: doc.mode, coAuthor: coAuthor)
                let section = SkillStore.promptSection(skills)
                return section.isEmpty ? base : base + "\n\n" + section
            }
            setup.readRoots = skills.map(\.folder)
            setup.coding = true
            runAgent(doc, key: key, client: client, setup: setup, add: add)
            return
        }
        if doc.isGraphHelper {
            guard let workflow = document(id: doc.workflowID) ?? (doc.workflowID == nil ? active : nil) else {
                var t = ChatTurn(role: .assistant, text: "")
                t.error = "Its workflow tab was closed."
                add(t)
                return
            }
            let session = GraphHelperSession(app: self, workflow: workflow)
            let mode = doc.mode
            var setup = AgentSetup(tools: GraphHelper.tools(session), root: library.root) {
                GraphHelper.systemPrompt(session, mode: mode)
            }
            setup.finished = { [weak doc] in
                if let edit = session.commit() { doc?.lastHelperEdit = edit }
            }
            runAgent(doc, key: key, client: client, setup: setup, add: add)
            return
        }
        // A plain chat: the chat tools (questions, widgets) when the model calls tools, in the project-less library.
        let root = library.root
        runAgent(doc, key: key, client: client, setup: AgentSetup(tools: [], root: root) { "" }, add: add)
    }

    // MARK: attachments

    /// Adds files to the message being written: images (resized to at most 1536 px, saved with the chat as JPEG),
    /// videos (made smaller in the background, see `saveVideo`) and text files up to 256 KB. Returns why any were left
    /// out.
    @discardableResult
    func attach(_ urls: [URL], to doc: ChatDocument) -> String? {
        var skipped: [String] = []
        let folder = chatStore.filesFolder(doc.id)
        for url in urls {
            let ext = url.pathExtension.lowercased()
            if Self.videoExtensions.contains(ext) {
                doc.preparingVideos += 1
                Task { @MainActor [weak doc] in
                    let result = await Self.saveVideo(url, into: folder)
                    guard let doc else { return }
                    doc.preparingVideos -= 1
                    switch result {
                    case .success(let name): doc.pending.append(.image(name))
                    case .failure(let e): doc.attachNote = "Couldn't attach \(url.lastPathComponent): \(e.localizedDescription)"
                    }
                }
            } else if ["png", "jpg", "jpeg", "heic", "webp", "gif", "tiff", "bmp"].contains(ext) {
                guard let name = Self.saveImage(url, into: folder) else { skipped.append(url.lastPathComponent); continue }
                doc.pending.append(.image(name))
            } else if let data = try? Data(contentsOf: url), data.count <= 256 * 1024,
                      let text = String(data: data, encoding: .utf8) {
                doc.pending.append(.file(ChatFile(name: url.lastPathComponent, text: text)))
            } else {
                skipped.append(url.lastPathComponent)
            }
        }
        return skipped.isEmpty ? nil
            : "Left out \(skipped.joined(separator: ", ")): only images, videos and text files up to 256 KB can be attached."
    }

    static func saveImage(_ url: URL, into folder: URL) -> String? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 1536,
                  kCGImageSourceCreateThumbnailWithTransform: true] as CFDictionary)
        else { return nil }
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = UUID().uuidString + ".jpg"
        guard let dest = CGImageDestinationCreateWithURL(folder.appendingPathComponent(name) as CFURL,
                                                         "public.jpeg" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        return CGImageDestinationFinalize(dest) ? name : nil
    }

    /// The URL each attachment of the sent branch goes to the model as: images as data URLs, videos uploaded to the GPU
    /// server first (once: uploads are content-addressed) as `upload:<name>`. A model without vision gets neither.
    func mediaURLs(_ transcript: ChatTranscript, doc: ChatDocument, vision: Bool,
                   client: LLMServerClient) async throws -> [String: String] {
        guard vision else { return [:] }
        let folder = chatStore.filesFolder(doc.id)
        var out: [String: String] = [:]
        for name in transcript.attachments where out[name] == nil {
            let file = folder.appendingPathComponent(name)
            if ChatTurn.isVideo(name) {
                guard FileManager.default.fileExists(atPath: file.path) else { continue }
                doc.engine = EngineStatus(state: "starting", phase: "Uploading video")
                out[name] = try await client.uploadMedia(file) { [weak doc] sent, total, _ in
                    let p = total > 0 ? Double(sent) / Double(total) : nil
                    Task { @MainActor in doc?.engine = EngineStatus(state: "starting", phase: "Uploading video", progress: p) }
                }
                doc.engine = nil
            } else {
                out[name] = Self.imageDataURL(file)
            }
        }
        return out
    }

    static func imageDataURL(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return "data:image/jpeg;base64," + data.base64EncodedString()
    }

    func stopReply(_ doc: ChatDocument) {
        doc.task?.cancel()
        if doc.permission != nil { doc.answer(.deny) }
        if doc.question != nil { doc.answerQuestion(nil) }
        if doc.plan != nil { doc.answerPlan(.keepPlanning(nil)) }
    }

    // MARK: the coding agent

    /// Runs the agent for a coding chat or a graph helper: each model message becomes an assistant turn, each tool call
    /// a tool turn (its card), questions go to the chat's permission card.
    func runAgent(_ doc: ChatDocument, key: LLMKey, client: LLMServerClient, setup: AgentSetup,
                  add: @escaping (ChatTurn) -> Void) {
        let vision = llm.model(key)?.vision == true
        let sent = doc.transcript  // the branch as it is now, before the reply's turns
        // Models without a tool parser can't take tools (vLLM refuses them): no chat tools, nothing said about them.
        let callsTools = llm.model(key)?.toolParser != nil
        let plain = setup.tools.isEmpty
        var config = AgentConfig(model: key.modelID, tools: setup.tools + (callsTools ? ChatTools.all : []),
                                 mode: doc.mode, rules: doc.rules)
        config.reasoningEffort = doc.transcript.reasoningEffort
        config.inlineThinking = llm.model(key)?.reasoningParser == nil
        config.vision = vision
        let limit = contextLimit(for: key)
        config.contextLimit = limit
        let ctx = ToolContext(root: setup.root)
        ctx.readRoots = setup.readRoots
        if setup.coding {
            ctx.checkpoints = checkpoints(for: doc)
            ctx.turnID = sent.turns.last { $0.role == .user && $0.summary != true }?.id
            ctx.jobs = doc.jobs
            ctx.sessions = AppSessionHost(model: self, parent: doc.id, depth: doc.transcript.depth ?? 0)
            if !doc.jobsObserved {
                doc.jobsObserved = true
                doc.jobs.onChange = { [weak self, weak doc] in
                    Task { @MainActor in
                        guard let doc else { return }
                        doc.runningJobs = doc.jobs.running.map(\.id)
                        if doc.runningJobs.isEmpty, !doc.isStreaming { self?.unloadIfHidden(doc) }
                    }
                }
            }
        }
        doc.onPendingChange = { [weak self, weak doc] in
            if let doc { self?.sessionChanged(doc) }
        }
        ctx.approvePlan = { [weak self, weak doc] callID, plan in
            await withCheckedContinuation { cont in
                Task { @MainActor in
                    guard let doc else { cont.resume(returning: .keepPlanning(nil)); return }
                    doc.plan = PendingPlan(callID: callID, plan: plan)
                    doc.planAnswer = cont
                    self?.sessionChanged(doc)
                }
            }
        }
        ctx.onOutput = { [weak doc] id, chunk in
            Task { @MainActor in doc?.toolOutputs[id, default: ""] += chunk }
        }
        ctx.askUser = { [weak self, weak doc] callID, questions in
            await withCheckedContinuation { cont in
                Task { @MainActor in
                    guard let doc else { cont.resume(returning: nil); return }
                    doc.question = PendingQuestion(callID: callID, questions: questions)
                    doc.questionAnswer = cont
                    self?.sessionChanged(doc)
                }
            }
        }
        ctx.seedTodos(doc.todos)
        doc.todoStepStarted = nil
        doc.requestStarted = Date()
        doc.engine = nil
        llm.lastActivity = Date()
        // The first assistant message takes the reply's place (a retry's sibling); the rest follow it.
        let start = ChatTurn(role: .assistant, text: "", model: key.modelID)
        add(start)
        doc.liveTurnID = start.id
        let st = AgentRunState(turn: start.id, inline: config.inlineThinking)
        doc.task = Task { @MainActor [weak self, weak doc] in
            guard let doc else { return }
            @MainActor func publish() { st.publish(into: doc) }
            // One system message: the chat's own prompt, the agent's, and the chat features.
            let own = await setup.system()
            let system = plain ? ChatPrompt.system(user: sent.systemPrompt, tools: callsTools)
                : [sent.systemPrompt ?? "", own, ChatPrompt.section(tools: callsTools)]
                    .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.joined(separator: "\n\n")
            do {
                let media = try await self?.mediaURLs(sent, doc: doc, vision: vision, client: client) ?? [:]
                var history = sent.messages(system: false) { media[$0] }
                // A history that already fills most of the context: summarize all but the last message first.
                if let limit, history.count >= 2, Double(Compactor.estimate(history)) > 0.75 * Double(limit),
                   let lastUser = sent.turns.last(where: { $0.role == .user && $0.summary != true }),
                   let summary = try? await Compactor.summarize(client: client.chat, model: key.modelID,
                                                                messages: [.system(system)] + history.dropLast(), limit: limit) {
                    var t = ChatTurn(role: .user, text: summary + "\n\n## The latest message\n" + lastUser.textForModel)
                    t.summary = true
                    // The summary goes between the last message and the reply being written.
                    if let parent = doc.transcript.turn(start.id)?.parent {
                        doc.transcript.activeLeaf = parent
                        doc.transcript.append(t)
                        doc.transcript.update(start.id) { $0.parent = t.id }
                        doc.transcript.select(start.id)
                    }
                    history = [.user(ChatTurn.summaryPrefix + t.text)]
                }
                let outcome = try await AgentLoop.run(
                    client: client.chat, messages: [.system(system)] + history, config: config, context: ctx,
                    ask: { [weak self, weak doc] request in
                        await withCheckedContinuation { cont in
                            Task { @MainActor in
                                guard let doc else { cont.resume(returning: .deny); return }
                                doc.permissionAnswer = cont
                                doc.permission = request
                                self?.sessionChanged(doc)
                            }
                        }
                    },
                    rulesChanged: { [weak doc] rules in
                        await MainActor.run {
                            doc?.transcript.allowEdits = rules.edits
                            doc?.transcript.allowCommands = rules.commandPrefixes
                            doc?.transcript.allowRuns = rules.runs ? true : nil
                            doc?.transcript.allowDomains = rules.domains.isEmpty ? nil : rules.domains
                        }
                    },
                    modeChanged: { [weak doc] m in await MainActor.run { doc?.mode = m } },
                    events: { event in
                        await MainActor.run {
                            switch event {
                            case .assistantStarted:
                                if !st.first {
                                    let t = ChatTurn(role: .assistant, text: "", model: key.modelID)
                                    doc.transcript.append(t)
                                    st.turn = t.id
                                    doc.liveTurnID = t.id
                                }
                                st.first = false
                                st.reset()
                            case .delta(let e):
                                st.acc.add(e)
                                switch e {
                                case .status(let s): doc.engine = s
                                case .reasoning: if st.thinkStart == nil { st.thinkStart = Date() }; doc.engine = nil
                                case .content, .toolCall:
                                    if st.thinkStart != nil, st.thinkEnd == nil { st.thinkEnd = Date() }
                                    doc.engine = nil
                                default: break
                                }
                                if Date().timeIntervalSince(st.lastPublish) > 0.05 {
                                    st.lastPublish = Date()
                                    publish()
                                }
                            case .assistantFinished:
                                st.acc.finish()
                                publish()
                                if let p = st.acc.promptTokens {
                                    doc.contextUsed = p + (st.acc.completionTokens ?? 0)
                                }
                            case .toolStarted(let call, let preview):
                                var t = ChatTurn(role: .tool, text: "", toolCallID: call.id)
                                t.toolTitle = preview.title
                                t.toolDetail = preview.detail
                                t.toolStatus = "running"
                                doc.transcript.append(t)
                            case .toolOutput:
                                break  // ctx.onOutput streams it
                            case .toolFinished(let id, let output, let status):
                                if doc.isCoding { self?.refreshGitSoon(doc) }
                                if let t = doc.transcript.turns.last(where: { $0.toolCallID == id }) {
                                    doc.transcript.update(t.id) {
                                        $0.text = output.text
                                        $0.toolStatus = status.rawValue
                                        if let detail = output.detail { $0.toolDetail = detail }
                                    }
                                }
                                if doc.transcript.turns.contains(where: { $0.toolCalls.contains { $0.id == id && $0.name == "todo_write" } }) {
                                    let list = ctx.todos
                                    doc.liveTodos = list
                                    if let cur = list.first(where: { $0.status == "in_progress" }) {
                                        if doc.todoStepStarted?.content != cur.content { doc.todoStepStarted = (cur.content, Date()) }
                                    } else {
                                        doc.todoStepStarted = nil
                                    }
                                }
                            case .compacted(let summary):
                                // The next assistant message follows the summary.
                                var t = ChatTurn(role: .user, text: summary)
                                t.summary = true
                                doc.transcript.append(t)
                            }
                        }
                    })
                if outcome == .stepLimit {
                    doc.transcript.update(st.turn) { $0.error = "Stopped after \(config.maxSteps) steps: say how to continue." }
                }
            } catch {
                st.acc.finish()
                publish()
                let message = (error is CancellationError || Task.isCancelled) ? "Stopped" : error.localizedDescription
                doc.transcript.update(st.turn) { $0.error = message }
                // Tool cards still running: stopped.
                for t in doc.transcript.turns where t.role == .tool && t.toolStatus == "running" {
                    doc.transcript.update(t.id) { $0.toolStatus = "failed"; $0.text = "Stopped." }
                }
            }
            if doc.permission != nil { doc.answer(.deny) }
            if doc.question != nil { doc.answerQuestion(nil) }
            if doc.plan != nil { doc.answerPlan(.keepPlanning(nil)) }
            setup.finished()
            doc.liveTurnID = nil
            doc.liveTodos = nil
            doc.todoStepStarted = nil
            doc.engine = nil
            doc.task = nil
            doc.transcript.updated = Date()
            self?.sessionChanged(doc, finished: true)
            self?.saveChat(doc)
            self?.sessionRunEnded(doc)
            self?.unloadIfHidden(doc)
            await self?.llm.refreshStatus(key)
        }
    }

    // MARK: session

    /// The open chat tabs and the one in front, for the next launch (transcripts are saved after each reply).
    func saveOpenChats() {
        guard persistSettings, !Self.isAutomatedRun else { return }
        let ids = chats.filter { $0.tabOpen && !$0.transcript.turns.isEmpty }.map(\.id.uuidString)
        UserDefaults.standard.set(ids, forKey: "openChats")
        UserDefaults.standard.set(activeChatID?.uuidString, forKey: "activeChat")
    }

    func restoreOpenChats() {
        reloadChats()
        guard persistSettings, !Self.isAutomatedRun else { return }
        for s in UserDefaults.standard.stringArray(forKey: "openChats") ?? [] {
            if let id = UUID(uuidString: s), var t = chatStore.load(id) {
                t.stopRunningTools()
                chats.append(ChatDocument(transcript: t))
            }
        }
        if let s = UserDefaults.standard.string(forKey: "activeChat"), let id = UUID(uuidString: s),
           chats.contains(where: { $0.id == id }) {
            activeChatID = id
        }
    }
}

/// What an agent chat runs with: its tools, the folder they work in, its system prompt, and what to do after a reply.
struct AgentSetup {
    var tools: [any AgentTool]
    var root: URL
    var system: @MainActor () async -> String
    var finished: @MainActor () -> Void = {}
    /// Folders (skills') the tools may read besides the project.
    var readRoots: [URL] = []
    /// A coding chat: its file changes are checkpointed and its background commands tracked.
    var coding = false

    init(tools: [any AgentTool], root: URL, system: @escaping @MainActor () async -> String) {
        self.tools = tools
        self.root = root
        self.system = system
    }
}

/// The agent run's bookkeeping, shared by its event handler (a reference, so concurrent closures can update it; they
/// all run on the main actor).
final class AgentRunState: @unchecked Sendable {
    var turn: UUID
    var first = true
    var acc: ReplyAccumulator
    var thinkStart: Date?
    var thinkEnd: Date?
    var lastPublish = Date.distantPast
    let inline: Bool

    init(turn: UUID, inline: Bool) {
        self.turn = turn
        self.inline = inline
        acc = ReplyAccumulator(inlineThinking: inline)
    }

    func reset() {
        acc = ReplyAccumulator(inlineThinking: inline)
        thinkStart = nil
        thinkEnd = nil
    }

    @MainActor func publish(into doc: ChatDocument) {
        let acc = acc, start = thinkStart, end = thinkEnd
        let (text, followUps) = FollowUps.split(acc.content, streaming: !acc.finished)
        doc.transcript.update(turn) { t in
            t.text = text
            t.followUps = followUps.isEmpty ? nil : followUps
            t.reasoning = acc.reasoning.isEmpty ? nil : acc.reasoning
            t.toolCalls = acc.tools.calls
            t.stats = acc.stats
            if let start { t.thinkingSeconds = (end ?? Date()).timeIntervalSince(start) }
        }
    }
}
