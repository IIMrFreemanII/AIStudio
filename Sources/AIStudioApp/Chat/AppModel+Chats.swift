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
        let k = key ?? lastChatModel.flatMap { llm.model($0) != nil ? $0 : nil }
            ?? llm.models.first.map { LLMKey(serverID: $0.server.id, modelID: $0.model.id) }
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
        if panel.runModal() == .OK, let url = panel.url { newCodingChat(project: url) }
    }

    /// Opens a saved chat (or brings its tab to the front).
    func openChat(_ id: UUID) {
        if chats.contains(where: { $0.id == id }) { activeChatID = id; return }
        guard let t = chatStore.load(id) else { reloadChats(); return }
        chats.append(ChatDocument(transcript: t))
        activeChatID = id
    }

    func activateChat(_ id: UUID) {
        guard chats.contains(where: { $0.id == id }) else { return }
        activeChatID = id
    }

    func closeChat(_ id: UUID) {
        guard let i = chats.firstIndex(where: { $0.id == id }) else { return }
        let doc = chats[i]
        doc.task?.cancel()
        saveChat(doc)
        chats.remove(at: i)
        if activeChatID == id { activeChatID = chats.isEmpty ? nil : chats[min(i, chats.count - 1)].id }
    }

    func deleteChat(_ id: UUID) {
        if let doc = chats.first(where: { $0.id == id }) {
            doc.task?.cancel()
            chats.removeAll { $0.id == id }
            if activeChatID == id { activeChatID = chats.last?.id }
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
        guard !text.isEmpty || !doc.pending.isEmpty, !doc.isStreaming else { return }
        doc.draft = ""
        if doc.transcript.turns.isEmpty {
            doc.transcript.title = ChatTranscript.title(from: text.isEmpty ? doc.pending.first?.name ?? "" : text)
        }
        let files = doc.pending.compactMap { if case .file(let f) = $0 { return f }; return nil }
        doc.transcript.append(ChatTurn(role: .user, text: text, images: doc.pending.compactMap {
            if case .image(let name) = $0 { return name }; return nil
        }, files: files.isEmpty ? nil : files))
        doc.pending = []
        reply(doc)
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
            runAgent(doc, key: key, client: client, root: root, add: add)
            return
        }
        let vision = llm.model(key)?.vision == true
        let folder = chatStore.filesFolder(doc.id)
        var request = ChatRequest(model: key.modelID, messages: doc.transcript.messages { name in
            vision ? Self.imageDataURL(folder.appendingPathComponent(name)) : nil
        })
        request.reasoningEffort = doc.transcript.reasoningEffort
        let turn = ChatTurn(role: .assistant, text: "", model: key.modelID)
        add(turn)
        doc.liveTurnID = turn.id
        doc.requestStarted = Date()
        doc.engine = nil
        llm.lastActivity = Date()
        let inline = llm.model(key)?.reasoningParser == nil
        doc.task = Task { @MainActor [weak self, weak doc] in
            guard let doc else { return }
            var acc = ReplyAccumulator(inlineThinking: inline)
            var lastPublish = Date.distantPast
            var thinkStart: Date?
            var thinkEnd: Date?
            @MainActor func publish() {
                doc.transcript.update(turn.id) { t in
                    t.text = acc.content
                    t.reasoning = acc.reasoning.isEmpty ? nil : acc.reasoning
                    t.toolCalls = acc.tools.calls
                    t.stats = acc.stats
                    if let s = thinkStart { t.thinkingSeconds = (thinkEnd ?? Date()).timeIntervalSince(s) }
                }
            }
            do {
                for try await e in client.chat.stream(request) {
                    acc.add(e)
                    switch e {
                    case .status(let s): doc.engine = s
                    case .reasoning: if thinkStart == nil { thinkStart = Date() }; doc.engine = nil
                    case .content, .toolCall:
                        if thinkStart != nil, thinkEnd == nil { thinkEnd = Date() }
                        doc.engine = nil
                    default: break
                    }
                    if Date().timeIntervalSince(lastPublish) > 0.05 {
                        lastPublish = Date()
                        publish()
                    }
                }
                acc.finish()
                publish()
            } catch {
                acc.finish()
                publish()
                let message = (error is CancellationError || Task.isCancelled) ? "Stopped" : error.localizedDescription
                doc.transcript.update(turn.id) { $0.error = message }
            }
            doc.liveTurnID = nil
            doc.engine = nil
            doc.task = nil
            doc.transcript.updated = Date()
            self?.saveChat(doc)
            await self?.llm.refreshStatus(key)
        }
    }

    // MARK: attachments

    /// Adds files to the message being written: images (resized to at most 1536 px, saved with the chat as JPEG)
    /// and text files up to 256 KB. Returns why any were left out.
    @discardableResult
    func attach(_ urls: [URL], to doc: ChatDocument) -> String? {
        var skipped: [String] = []
        let folder = chatStore.filesFolder(doc.id)
        for url in urls {
            let ext = url.pathExtension.lowercased()
            if ["png", "jpg", "jpeg", "heic", "webp", "gif", "tiff", "bmp"].contains(ext) {
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
            : "Left out \(skipped.joined(separator: ", ")): only images and text files up to 256 KB can be attached."
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

    static func imageDataURL(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return "data:image/jpeg;base64," + data.base64EncodedString()
    }

    func stopReply(_ doc: ChatDocument) {
        doc.task?.cancel()
        if doc.permission != nil { doc.answer(.deny) }
    }

    // MARK: the coding agent

    /// Runs the agent for a coding chat: each model message becomes an assistant turn, each tool call a tool turn
    /// (its card), questions go to the chat's permission card.
    func runAgent(_ doc: ChatDocument, key: LLMKey, client: LLMServerClient, root: URL, add: @escaping (ChatTurn) -> Void) {
        let vision = llm.model(key)?.vision == true
        let folder = chatStore.filesFolder(doc.id)
        let history = doc.transcript.messages { name in vision ? Self.imageDataURL(folder.appendingPathComponent(name)) : nil }
        var config = AgentConfig(model: key.modelID, mode: doc.mode, rules: doc.rules)
        config.reasoningEffort = doc.transcript.reasoningEffort
        config.inlineThinking = llm.model(key)?.reasoningParser == nil
        let ctx = ToolContext(root: root)
        ctx.onOutput = { [weak doc] id, chunk in
            Task { @MainActor in doc?.toolOutputs[id, default: ""] += chunk }
        }
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
            let system = await ProjectContext.systemPrompt(root: root, mode: config.mode)
            do {
                let outcome = try await AgentLoop.run(
                    client: client.chat, messages: [.system(system)] + history, config: config, context: ctx,
                    ask: { [weak doc] request in
                        await withCheckedContinuation { cont in
                            Task { @MainActor in
                                guard let doc else { cont.resume(returning: .deny); return }
                                doc.permissionAnswer = cont
                                doc.permission = request
                            }
                        }
                    },
                    rulesChanged: { [weak doc] rules in
                        await MainActor.run {
                            doc?.transcript.allowEdits = rules.edits
                            doc?.transcript.allowCommands = rules.commandPrefixes
                        }
                    },
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
                            case .toolStarted(let call, let preview):
                                var t = ChatTurn(role: .tool, text: "", toolCallID: call.id)
                                t.toolTitle = preview.title
                                t.toolDetail = preview.detail
                                t.toolStatus = "running"
                                doc.transcript.append(t)
                            case .toolOutput:
                                break  // ctx.onOutput streams it
                            case .toolFinished(let id, let output, let status):
                                if let t = doc.transcript.turns.last(where: { $0.toolCallID == id }) {
                                    doc.transcript.update(t.id) {
                                        $0.text = output.text
                                        $0.toolStatus = status.rawValue
                                    }
                                }
                                doc.todos = ctx.todos
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
            doc.liveTurnID = nil
            doc.engine = nil
            doc.task = nil
            doc.transcript.updated = Date()
            self?.saveChat(doc)
            await self?.llm.refreshStatus(key)
        }
    }

    // MARK: session

    /// The open chat tabs and the one in front, for the next launch (transcripts are saved after each reply).
    func saveOpenChats() {
        guard persistSettings, !Self.isAutomatedRun else { return }
        let ids = chats.filter { !$0.transcript.turns.isEmpty }.map(\.id.uuidString)
        UserDefaults.standard.set(ids, forKey: "openChats")
        UserDefaults.standard.set(activeChatID?.uuidString, forKey: "activeChat")
    }

    func restoreOpenChats() {
        reloadChats()
        guard persistSettings, !Self.isAutomatedRun else { return }
        for s in UserDefaults.standard.stringArray(forKey: "openChats") ?? [] {
            if let id = UUID(uuidString: s), let t = chatStore.load(id) { chats.append(ChatDocument(transcript: t)) }
        }
        if let s = UserDefaults.standard.string(forKey: "activeChat"), let id = UUID(uuidString: s),
           chats.contains(where: { $0.id == id }) {
            activeChatID = id
        }
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
        doc.transcript.update(turn) { t in
            t.text = acc.content
            t.reasoning = acc.reasoning.isEmpty ? nil : acc.reasoning
            t.toolCalls = acc.tools.calls
            t.stats = acc.stats
            if let start { t.thinkingSeconds = (end ?? Date()).timeIntervalSince(start) }
        }
    }
}
