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
        doc.transcript.title = ListFormatter.localizedString(byJoining: people.map(\.name))
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

    /// Streams the characters' replies to the user's latest message: in a group, whoever it names (else everyone in
    /// turn, `GroupTurns`), one after another. Each is the agent loop with no tools and that character's prompt as the
    /// whole system message, then their media and memory pass. `speakers` overrides who answers (a retry, a check-in);
    /// `nudge` goes to the model after the history (`CheckIn`).
    func characterReply(_ doc: ChatDocument, key: LLMKey, client: LLMServerClient, add: @escaping (ChatTurn) -> Void,
                        speakers: [UUID]? = nil, nudge: String? = nil) {
        let people = characters(of: doc)
        let queue = (speakers ?? GroupTurns.speakers(for: lastUserText(doc), cast: people, lastFirst: lastFirstSpeaker(doc)))
            .filter { id in people.contains { $0.id == id } }
        guard let first = queue.first, let c = people.first(where: { $0.id == first }) else {
            var t = ChatTurn(role: .assistant, text: "")
            t.error = "This character was deleted."
            add(t)
            return
        }
        let rest = Array(queue.dropFirst())
        let others = people.filter { $0.id != c.id }
        let names = Dictionary(people.map { ($0.id, $0.name) }, uniquingKeysWith: { a, _ in a })
        let store = characterStore
        let scene = scene(of: doc)
        let kinds = mediaKinds(c, server: photoServer(for: doc))
        let adult = Self.adultOptions
        var setup = AgentSetup(tools: [], root: library.root) {
            let memory = store.memory(c.id)
            return CharacterPrompt.system(c, memory: memory, scene: scene, others: others, media: kinds,
                                          initiativeLeft: CharacterCost.initiativeLeft(c, memory: memory), adult: adult)
        }
        setup.character = true
        setup.speaker = c.id
        setup.temperature = c.temperature
        setup.topP = c.topP
        setup.summarySystem = MemoryPass.summarySystem
        setup.nudge = nudge
        if people.count > 1 {
            setup.group = true
            setup.others = { t in GroupTurns.line(t.speaker.flatMap { names[$0] } ?? "Someone", t.text) }
        }
        let group = people.count > 1
        setup.finished = { [weak self, weak doc] in
            guard let self, let doc else { return }
            if let turn = doc.transcript.turns.last(where: { $0.role == .assistant && $0.speaker == c.id }) {
                // In a group a model sometimes signs its message with its name: the chat shows who wrote it.
                if group {
                    let clean = GroupTurns.stripName(turn.text, name: c.name)
                    if clean != turn.text { doc.transcript.update(turn.id) { $0.text = clean } }
                }
                self.startCharacterMedia(doc, turnID: turn.id, character: c)
            }
            self.memoryPassIfDue(doc, character: c, key: key, client: client)
            // The next one answers once this reply has wrapped up, unless it failed or was stopped, or the user
            // sent (or queued) something meanwhile.
            guard !rest.isEmpty, doc.transcript.turns.last?.error == nil else { return }
            Task { @MainActor [weak self, weak doc] in
                guard let self, let doc, !doc.isStreaming, doc.queued == nil else { return }
                self.characterReply(doc, key: key, client: client, add: { doc.transcript.append($0) }, speakers: rest)
            }
        }
        runAgent(doc, key: key, client: client, setup: setup, add: add)
    }

    /// The text of the user's latest message on the shown branch.
    func lastUserText(_ doc: ChatDocument) -> String {
        doc.transcript.turns.last { $0.role == .user && $0.summary != true }?.text ?? ""
    }

    /// Who answered the user's message before the latest one first (the next round starts after them).
    func lastFirstSpeaker(_ doc: ChatDocument) -> UUID? {
        let turns = doc.transcript.turns
        let users = turns.indices.filter { turns[$0].role == .user && turns[$0].summary != true }
        guard users.count >= 2 else { return nil }
        return turns[(users[users.count - 2] + 1)...].first { $0.role == .assistant && $0.speaker != nil }?.speaker
    }

    /// Album › Use This Look Again: a scene with the photo's place, time and mood and what they wore in it, set on
    /// your chat with them (the one in front when it's theirs, else your newest one with them, else a new one).
    func reuseAlbumLook(_ item: AlbumItem, characterID: UUID) {
        let store = characterStore
        let all = store.scenarios()
        let base = all.first { $0.title == item.scene }
        let scene = Scenario.like(item, base: base, date: item.date.formatted(date: .abbreviated, time: .omitted))
        var custom = all.filter { s in !Scenario.builtIns.contains { $0.id == s.id } && s.id != scene.id }
        custom.append(scene)
        try? store.saveCustomScenarios(custom)
        if activeChat?.transcript.characterIDs?.contains(characterID) != true { chat(withCharacter: characterID) }
        guard let doc = activeChat, doc.transcript.characterIDs?.contains(characterID) == true else { return }
        setScene(scene.id, for: doc)
        memoryCharacter = nil
    }

    // MARK: writing first

    /// A character chat came into view (opened, back in front, or the app came back): it may text first. Automated
    /// runs call `characterCheckInIfDue` themselves.
    func characterChatShown(_ doc: ChatDocument) {
        guard doc.isCharacter, !Self.isAutomatedRun else { return }
        characterCheckInIfDue(doc)
    }

    /// Opening a one-to-one character chat after time away (`checkInHours`, the shown chat or the app coming back):
    /// the character texts first. Not while a reply runs, nor after a message of yours that has no answer.
    func characterCheckInIfDue(_ doc: ChatDocument, now: Date = Date()) {
        let people = characters(of: doc)
        guard doc.isCharacter, people.count == 1, let c = people.first, !doc.isStreaming, doc.question == nil,
              !doc.compacting, let key = doc.modelKey, let client = llm.client(key.serverID),
              let last = doc.transcript.turns.last(where: { $0.role == .user || $0.role == .assistant }) else { return }
        let when = last.ended ?? last.date
        guard CheckIn.due(hours: c.checkInHours, last: when, lastIsUser: last.role == .user, now: now) else { return }
        lastChatModel = key
        characterReply(doc, key: key, client: client, add: { doc.transcript.append($0) }, speakers: [c.id],
                       nudge: CheckIn.nudge(name: c.name, away: now.timeIntervalSince(when)))
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
            .map { t in (speaker: t.role == .user ? "User" : t.speaker.flatMap { self.character($0)?.name } ?? c.name, text: t.text) }
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

extension AppModel {
    /// The chat model writes `c`'s details (story, personality, likes, texting style, first message, a look detail and,
    /// when `newName`, a name) from its picks, on the default chat model. Returns them and the cost.
    func writeCharacterDetails(_ c: Companion, newName: Bool, model chosen: LLMKey? = nil,
                               progress: @escaping @MainActor (String) -> Void = { _ in }) async throws
        -> (details: CharacterDetails, cost: Double)
    {
        guard let key = chosen ?? defaultChatModel, let client = llm.client(key.serverID) else {
            throw CharacterDetailsError.noModel
        }
        return try await CharacterWriter.run(client: client.chat, model: key.modelID, character: c, newName: newName,
                                             onProgress: { p in
            let text = p.phase ?? (p.characters > 0 ? "Writing…" : "Waiting for the model…")
            Task { @MainActor in progress(text) }
        })
    }
}

enum CharacterDetailsError: Error, CustomStringConvertible {
    case noModel
    var description: String { "Install a chat model on your GPU server first (Models › Chat)." }
}
