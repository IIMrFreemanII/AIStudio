import Agent
import CharacterCore
import Foundation
import LLMClient

/// Character chats: the saved characters, and what makes a chat with them reply as one.
extension AppModel {
    /// `Character` would collide with Swift's own.
    typealias Companion = CharacterCore.Character

    var characterStore: CharacterStore {
        CharacterStore(root: library.root.appendingPathComponent("characters", isDirectory: true))
    }

    func reloadCharacters() { characters = characterStore.list() }

    func character(_ id: UUID) -> Companion? {
        characters.first { $0.id == id } ?? characterStore.load(id)
    }

    /// Saves (`updated` = now); returns the age-guard problems (nothing is saved) or [] when saved.
    @discardableResult
    func saveCharacter(_ c: Companion) -> [String] {
        let problems = AgeGuard.problems(c)
        guard problems.isEmpty else { return problems }
        var saved = c
        saved.updated = Date()
        do {
            try characterStore.save(saved)
        } catch {
            return [error.localizedDescription]
        }
        reloadCharacters()
        return []
    }

    func deleteCharacter(_ id: UUID) {
        characterStore.delete(id, useTrash: !Self.isAutomatedRun)
        reloadCharacters()
    }

    // MARK: chats

    /// A new chat with these characters, in front: titled with their names, in the default scene, and, with one
    /// character that has a greeting, started by it.
    @discardableResult
    func newCharacterChat(_ ids: [UUID], model key: LLMKey? = nil) -> ChatDocument {
        let doc = newChat(model: key)
        let people = ids.compactMap { character($0) }
        doc.transcript.characterIDs = ids
        doc.transcript.sceneID = Scenario.defaultID
        doc.transcript.title = people.map(\.name).joined(separator: " & ")
        if ids.count == 1, let c = people.first, !c.greeting.isEmpty {
            var t = ChatTurn(role: .assistant, text: c.greeting, model: doc.transcript.modelID)
            t.speaker = c.id
            doc.transcript.append(t)
        }
        return doc
    }

    /// Opens the newest chat (not archived, saved or loaded) with just this character, else starts one.
    func chat(withCharacter id: UUID) {
        var best: (id: UUID, updated: Date)?
        func consider(_ chat: UUID, _ updated: Date) {
            if best == nil || updated > best!.updated { best = (chat, updated) }
        }
        for doc in chats where doc.transcript.characterIDs == [id] && doc.transcript.archived != true {
            consider(doc.id, doc.transcript.updated)
        }
        for s in chatSummaries where s.character && !s.archived && !chats.contains(where: { $0.id == s.id }) {
            if chatStore.load(s.id)?.characterIDs == [id] { consider(s.id, s.updated) }
        }
        if let best { openChat(best.id) } else { newCharacterChat([id]) }
    }

    /// The chat's characters in `characterIDs` order (deleted ones are skipped).
    func characters(of doc: ChatDocument) -> [Companion] {
        (doc.transcript.characterIDs ?? []).compactMap { character($0) }
    }

    func scene(of doc: ChatDocument) -> Scenario? {
        guard let id = doc.transcript.sceneID else { return nil }
        return characterStore.scenarios().first { $0.id == id }
    }

    func setScene(_ id: String?, for doc: ChatDocument) {
        doc.transcript.sceneID = id
        saveChat(doc)
    }

    // MARK: replying

    /// Streams the character's reply: the agent loop with no tools and the character's prompt as the whole system
    /// message, then the memory pass when it is due.
    func characterReply(_ doc: ChatDocument, key: LLMKey, client: LLMServerClient, add: @escaping (ChatTurn) -> Void) {
        let people = characters(of: doc)
        guard let c = people.first else {
            var t = ChatTurn(role: .assistant, text: "")
            t.error = "This character was deleted."
            add(t)
            return
        }
        let others = Array(people.dropFirst())
        let store = characterStore
        let scene = scene(of: doc)
        let kinds = mediaKinds(c, server: photoServer(for: doc))
        var setup = AgentSetup(tools: [], root: library.root) {
            let memory = store.memory(c.id)
            return CharacterPrompt.system(c, memory: memory, scene: scene, others: others, media: kinds,
                                          initiativeLeft: CharacterCost.initiativeLeft(c, memory: memory))
        }
        setup.character = true
        setup.speaker = c.id
        setup.temperature = c.temperature
        setup.topP = c.topP
        setup.summarySystem = MemoryPass.summarySystem
        setup.finished = { [weak self, weak doc] in
            guard let self, let doc else { return }
            if let turn = doc.transcript.turns.last(where: { $0.role == .assistant && $0.speaker == c.id }) {
                self.startCharacterMedia(doc, turnID: turn.id, character: c)
            }
            self.memoryPassIfDue(doc, character: c, key: key, client: client)
        }
        runAgent(doc, key: key, client: client, setup: setup, add: add)
    }

    /// Characters whose memory pass is running (main actor only).
    private static var memoryPassesRunning = Set<UUID>()

    /// Counts a finished reply; every `MemoryPass.every` replies a model updates what the character remembers.
    func memoryPassIfDue(_ doc: ChatDocument, character c: Companion, key: LLMKey, client: LLMServerClient) {
        // A failed or stopped reply isn't a turn of the conversation.
        guard doc.transcript.turns.last?.error == nil else { return }
        let store = characterStore
        var memory = store.memory(c.id)
        memory.turnsSinceUpdate += 1
        try? store.saveMemory(memory, for: c.id)
        guard memory.turnsSinceUpdate >= MemoryPass.every, !Self.memoryPassesRunning.contains(c.id) else { return }
        Self.memoryPassesRunning.insert(c.id)
        let recent = doc.transcript.turns
            .filter { ($0.role == .user || $0.role == .assistant) && $0.summary != true && $0.error == nil }
            .suffix(12)
            .map { (speaker: $0.role == .user ? "User" : c.name, text: $0.text) }
        let chatID = doc.id
        let snapshot = memory
        Task { @MainActor in
            defer { Self.memoryPassesRunning.remove(c.id) }
            do {
                let result = try await MemoryPass.run(client: client.chat, model: key.modelID, character: c,
                                                      memory: snapshot, recent: Array(recent))
                var fresh = store.memory(c.id)
                fresh.apply(result.update, chatID: chatID)
                fresh.spentUSD += result.cost
                try? store.saveMemory(fresh, for: c.id)
            } catch {
                print("[characters] memory pass failed: \(error.localizedDescription)")
            }
        }
    }
}
