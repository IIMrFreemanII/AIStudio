import Agent
import AppKit
import Foundation
import LLMClient

/// Coding-agent extras: skills and the web search key, checkpoints and rewinding, compacting the conversation.
extension AppModel {
    // MARK: settings

    /// Skills the user turned off (by name).
    var disabledSkills: Set<String> {
        get { Set(UserDefaults.standard.stringArray(forKey: "disabledSkills") ?? []) }
        set {
            guard persistSettings, !Self.isAutomatedRun else { return }
            UserDefaults.standard.set(newValue.sorted(), forKey: "disabledSkills")
        }
    }

    /// The Brave Search API key (saved in `TokenFile`), which `web_search` needs.
    var searchAPIKey: String? {
        guard let k = ConnectionToken.read("brave-search"), !k.isEmpty else { return nil }
        return k
    }

    func setSearchAPIKey(_ key: String) {
        ConnectionToken.save(key.trimmingCharacters(in: .whitespacesAndNewlines), for: "brave-search",
                             persist: !Self.isAutomatedRun)
    }

    /// The skills a coding chat in `root` can use: personal, the project's, minus the turned-off ones.
    func loadSkills(root: URL?) -> [Skill] {
        SkillStore.load(root: root, disabled: disabledSkills)
    }

    /// The model's context length in tokens, when the server says.
    func contextLimit(for key: LLMKey) -> Int? {
        guard let c = llm.model(key)?.context, c > 0 else { return nil }
        return c
    }

    // MARK: checkpoints and rewind

    /// Where the chat's file checkpoints are kept (made on first use).
    func checkpoints(for doc: ChatDocument) -> CheckpointStore {
        if let s = doc.checkpointStore { return s }
        let s = CheckpointStore(folder: chatStore.filesFolder(doc.id).appendingPathComponent("checkpoints"))
        doc.checkpointStore = s
        return s
    }

    /// The user messages on the shown branch from `id` on (the ones whose file changes a rewind undoes).
    func userTurns(from id: UUID, in doc: ChatDocument) -> Set<UUID> {
        let shown = doc.transcript.turns
        guard let i = shown.firstIndex(where: { $0.id == id }) else { return [] }
        return Set(shown[i...].filter { $0.role == .user && $0.summary != true }.map(\.id))
    }

    /// Opens the rewind card for a message, with the files that would be restored.
    func prepareRewind(_ doc: ChatDocument, to id: UUID) {
        guard !doc.isStreaming else { return }
        doc.rewindTarget = id
        doc.rewindFiles = nil
        let turns = userTurns(from: id, in: doc)
        let store = checkpoints(for: doc)
        Task { @MainActor [weak doc] in
            let files = await store.changedPaths(in: turns)
            guard let doc, doc.rewindTarget == id else { return }
            doc.rewindFiles = files
        }
    }

    /// Goes back to before the message `rewindTarget`: its files (and later ones) put back, and/or the chat shown up to
    /// it (the message goes back in the field; the old messages stay as a branch).
    func rewind(_ doc: ChatDocument, files: Bool, chat: Bool) {
        guard let id = doc.rewindTarget, !doc.isStreaming, let target = doc.transcript.turn(id) else {
            doc.rewindTarget = nil
            doc.rewindFiles = nil
            return
        }
        let turns = userTurns(from: id, in: doc)
        let store = checkpoints(for: doc)
        doc.rewindTarget = nil
        doc.rewindFiles = nil
        Task { @MainActor [weak self, weak doc] in
            guard let self, let doc else { return }
            if files {
                do {
                    let result = try await store.restore(turns: turns)
                    if !result.skipped.isEmpty {
                        doc.attachNote = "Not restored (too large): " + result.skipped.map(\.lastPathComponent).joined(separator: ", ")
                    }
                } catch {
                    doc.attachNote = "Couldn't restore the files: \(error.localizedDescription)"
                }
                // The Changes pane and branch chip show the restored files now, not at the next refresh.
                self.refreshGitSoon(doc)
            }
            if chat {
                doc.draft = target.text
                doc.transcript.activeLeaf = target.parent
                doc.contextUsed = nil
                doc.liveTodos = nil
                doc.editingTurnID = nil
            }
            doc.transcript.updated = Date()
            try? self.chatStore.save(doc.transcript)
            self.reloadChats()
        }
    }

    // MARK: compacting

    /// Replaces the conversation so far, for the model, with a summary (the Compact button, `/compact`).
    func compact(_ doc: ChatDocument) {
        guard !doc.isStreaming, !doc.compacting, !doc.transcript.turns.isEmpty,
              let key = doc.modelKey, let client = llm.client(key.serverID) else { return }
        doc.compacting = true
        let history = doc.transcript.messages(system: false)
        let limit = contextLimit(for: key)
        llm.lastActivity = Date()
        doc.task = Task { @MainActor [weak self, weak doc] in
            guard let doc else { return }
            do {
                let summary = try await Compactor.summarize(client: client.chat, model: key.modelID, messages: history,
                                                            limit: limit)
                try Task.checkCancellation()
                var t = ChatTurn(role: .user, text: summary)
                t.summary = true
                doc.transcript.append(t)
                doc.contextUsed = nil
            } catch is CancellationError {
                // Stopped: nothing changes.
            } catch {
                var t = ChatTurn(role: .assistant, text: "")
                t.error = "Couldn't compact: \(error.localizedDescription)"
                doc.transcript.append(t)
            }
            doc.compacting = false
            doc.task = nil
            doc.transcript.updated = Date()
            self?.saveChat(doc)
        }
    }

    // MARK: skills

    /// A new personal skill from a template, revealed in Finder.
    func newPersonalSkill() {
        let fm = FileManager.default
        let base = SkillStore.personalFolder
        var folder = base.appendingPathComponent("new-skill", isDirectory: true)
        var n = 2
        while fm.fileExists(atPath: folder.path) {
            folder = base.appendingPathComponent("new-skill-\(n)", isDirectory: true)
            n += 1
        }
        let file = folder.appendingPathComponent("SKILL.md")
        let template = """
        ---
        name: \(folder.lastPathComponent)
        description: One sentence: what this skill does and when the agent should use it.
        ---

        Write the instructions here, in Markdown. The agent reads them when it loads the skill (or when you type
        /\(folder.lastPathComponent) in a coding chat). A skill is a folder with this SKILL.md; put scripts or
        reference files beside it and mention them by relative path.

        """
        do {
            try fm.createDirectory(at: folder, withIntermediateDirectories: true)
            try template.write(to: file, atomically: true, encoding: .utf8)
            NSWorkspace.shared.activateFileViewerSelecting([file])
        } catch {
            NSSound.beep()
        }
    }

    /// The personal skills folder, made if missing, in Finder.
    func showPersonalSkillsFolder() {
        let folder = SkillStore.personalFolder
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        NSWorkspace.shared.open(folder)
    }
}
