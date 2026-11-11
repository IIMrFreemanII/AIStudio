import Foundation
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
        guard !text.isEmpty, !doc.isStreaming else { return }
        doc.draft = ""
        if doc.transcript.turns.isEmpty { doc.transcript.title = ChatTranscript.title(from: text) }
        doc.transcript.turns.append(ChatTurn(role: .user, text: text))
        reply(doc)
    }

    /// Streams an assistant reply to the transcript as it stands.
    func reply(_ doc: ChatDocument) {
        guard let key = doc.modelKey, let client = llm.client(key.serverID) else {
            var t = ChatTurn(role: .assistant, text: "")
            t.error = llm.servers.isEmpty ? "Add a GPU server in Servers first." : "Choose a model for this chat."
            doc.transcript.turns.append(t)
            return
        }
        lastChatModel = key
        var request = ChatRequest(model: key.modelID, messages: doc.transcript.messages())
        request.reasoningEffort = doc.transcript.reasoningEffort
        let turn = ChatTurn(role: .assistant, text: "", model: key.modelID)
        doc.transcript.turns.append(turn)
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
                guard let i = doc.transcript.turns.firstIndex(where: { $0.id == turn.id }) else { return }
                var t = doc.transcript.turns[i]
                t.text = acc.content
                t.reasoning = acc.reasoning.isEmpty ? nil : acc.reasoning
                t.toolCalls = acc.tools.calls
                t.stats = acc.stats
                if let s = thinkStart { t.thinkingSeconds = (thinkEnd ?? Date()).timeIntervalSince(s) }
                doc.transcript.turns[i] = t
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
                if let i = doc.transcript.turns.firstIndex(where: { $0.id == turn.id }) {
                    doc.transcript.turns[i].error = (error is CancellationError || Task.isCancelled)
                        ? "Stopped" : error.localizedDescription
                }
            }
            doc.liveTurnID = nil
            doc.engine = nil
            doc.task = nil
            doc.transcript.updated = Date()
            self?.saveChat(doc)
            await self?.llm.refreshStatus(key)
        }
    }

    func stopReply(_ doc: ChatDocument) {
        doc.task?.cancel()
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
