import Agent
import AppKit
import Foundation
import LLMClient

/// A `propose_session` card: what the agent suggests starting.
struct SessionProposal: Equatable {
    var callID: String
    var spec: SessionSpec
}

extension ChatTranscript {
    /// The last assistant message's text on the shown branch, if any.
    var lastAssistantText: String? {
        turns.last { $0.role == .assistant && !$0.text.isEmpty }?.text
    }

    /// Marks tool turns left "running" (the app quit mid-run) as stopped.
    mutating func stopRunningTools() {
        for t in nodes where t.role == .tool && t.toolStatus == "running" {
            update(t.id) { $0.toolStatus = "failed"; $0.text = "Stopped." }
        }
    }
}

/// Sessions: coding chats that run in the background, start each other, report back, and work in git worktrees.
extension AppModel {
    var worktreeBranchPrefix: String {
        get { UserDefaults.standard.string(forKey: "worktreeBranchPrefix") ?? "aistudio/" }
        set {
            guard persistSettings else { return }
            UserDefaults.standard.set(newValue, forKey: "worktreeBranchPrefix")
        }
    }

    private static func clip(_ s: String, _ n: Int = 600) -> String { String(s.prefix(n)) }

    // MARK: starting

    /// Starts a session from `parent`: its own chat (a fresh conversation or a fork of the parent's), in its own git
    /// worktree when asked and possible, running at once. Hidden unless `show`.
    @discardableResult
    func startChild(_ spec: SessionSpec, parent: ChatDocument, show: Bool = false) async throws -> SessionInfo {
        guard let parentProject = parent.projectURL else { throw ToolError("Sessions start from coding chats only.") }
        let pt = parent.transcript
        var t: ChatTranscript
        if spec.context == .fork,
           let last = pt.turns.last(where: { $0.role == .user && $0.sessionEvent == nil && $0.summary != true }),
           let forked = pt.fork(upTo: last.id) {
            t = forked
        } else {
            t = ChatTranscript(serverID: pt.serverID, modelID: pt.modelID)
            t.projectPath = pt.projectPath
            t.permissionMode = pt.permissionMode
            t.allowEdits = pt.allowEdits
            t.allowCommands = pt.allowCommands
            t.allowDomains = pt.allowDomains
            t.reasoningEffort = pt.reasoningEffort
        }
        t.title = spec.title
        t.brief = spec.prompt
        t.parentID = parent.id
        t.depth = (pt.depth ?? 0) + 1

        var note = "\n\n(This folder isn't a git repository: you share it with the session that started you.)"
        if spec.worktree {
            let repo: URL?
            if let w = pt.worktree { repo = URL(fileURLWithPath: w.repo) } else { repo = await GitWorktree.repoRoot(parentProject) }
            if let repo {
                var base = pt.worktree?.branch
                if base == nil { base = await GitWorktree.currentBranch(repo) }
                let made = try await GitWorktree.create(repo: repo, branch: worktreeBranchPrefix + GitWorktree.slug(spec.title),
                                                        base: base)
                t.projectPath = made.path.path
                t.worktree = SessionWorktree(repo: repo.path, path: made.path.path, branch: made.branch, base: base ?? "HEAD")
                note = "\n\n(You're working in your own git worktree on branch \(made.branch), cut from \(base ?? "HEAD"). "
                    + "Changes the parent hasn't committed aren't here. Commit your work on this branch when done.)"
            }
        }
        t.updated = Date()
        let child = ChatDocument(transcript: t)
        child.tabOpen = show
        chats.append(child)
        child.transcript.append(ChatTurn(role: .user, text: spec.prompt + note))
        saveChat(child)
        if show { activeChatID = child.id }
        refreshWorktreeStatus(child)
        reply(child)
        // No reply could start (no model): nothing will end it, so report now.
        if !child.isStreaming { sessionRunEnded(child) }
        return SessionInfo(id: child.id, title: spec.title, status: "working", branch: child.transcript.worktree?.branch)
    }

    private func statusText(_ s: SessionStatus) -> String {
        switch s {
        case .working: return "working"
        case .waiting: return "waiting"
        case .failed: return "failed"
        case .unread, .idle: return "done"
        }
    }

    /// The sessions `id` started, oldest first.
    func children(of id: UUID) -> [SessionInfo] {
        var found: [(Date, SessionInfo)] = []
        var seen = Set<UUID>()
        for d in chats where d.transcript.parentID == id && (d.transcript.depth ?? 0) > 0 {
            seen.insert(d.id)
            found.append((d.transcript.created, SessionInfo(
                id: d.id, title: d.title, status: statusText(status(of: d)), branch: d.transcript.worktree?.branch,
                lastReport: d.transcript.lastAssistantText.map { Self.clip($0) })))
        }
        for s in chatSummaries where s.parentID == id && !seen.contains(s.id) {
            guard let t = chatStore.load(s.id), (t.depth ?? 0) > 0 else { continue }
            found.append((t.created, SessionInfo(
                id: t.id, title: t.title, status: statusText(t.savedStatus), branch: t.worktree?.branch,
                lastReport: t.lastAssistantText.map { Self.clip($0) })))
        }
        return found.sorted { $0.0 < $1.0 }.map(\.1)
    }

    /// A session that is loaded, else loaded now (hidden, with interrupted tool calls marked stopped).
    func loadSession(_ id: UUID) -> ChatDocument? {
        if let d = chats.first(where: { $0.id == id }) { return d }
        guard var t = chatStore.load(id) else { return nil }
        t.stopRunningTools()
        let d = ChatDocument(transcript: t)
        d.tabOpen = false
        chats.append(d)
        return d
    }

    func sendToChild(_ text: String, id: UUID, parent: UUID) throws {
        let saved = chats.first(where: { $0.id == id })?.transcript ?? chatStore.load(id)
        guard let saved, saved.parentID == parent else { throw ToolError("That isn't one of your sessions.") }
        guard let child = loadSession(id) else { throw ToolError("That session is gone.") }
        if child.isStreaming || child.compacting { throw ToolError("It's still working: wait for its report.") }
        child.transcript.append(ChatTurn(role: .user, text: text))
        child.autoContinues = 0
        reply(child)
        if !child.isStreaming { sessionRunEnded(child) }
    }

    // MARK: reporting back

    /// Called at the end of every agent run: delivers the reports queued for this session, and reports this session's
    /// result to the one that started it.
    func sessionRunEnded(_ doc: ChatDocument) {
        refreshWorktreeStatus(doc)
        if !doc.pendingEvents.isEmpty {
            let events = doc.pendingEvents
            doc.pendingEvents = []
            deliver(events, to: doc)
        }
        guard (doc.transcript.depth ?? 0) > 0, let parentID = doc.transcript.parentID else { return }
        let last = doc.transcript.turns.last
        let status: String
        if let e = last?.error { status = e == "Stopped" ? "stopped" : "failed" } else { status = "done" }
        let report = doc.transcript.lastAssistantText.map { Self.clip($0) } ?? last?.error ?? ""
        let event = SessionEvent(childID: doc.id, title: doc.title, status: status,
                                 branch: doc.transcript.worktree?.branch, report: report)
        guard let parent = loadSession(parentID) else { return }
        if parent.isStreaming || parent.compacting {
            parent.pendingEvents.append(event)
        } else {
            deliver([event], to: parent)
        }
    }

    /// Adds the reports to `parent` (one turn each) and, when the agent spoke last, lets it carry on (at most 5 times
    /// in a row without the user).
    private func deliver(_ events: [SessionEvent], to parent: ChatDocument) {
        let lastTurn = parent.transcript.turns.last
        let agentSpokeLast = lastTurn.map { ($0.role == .assistant || $0.role == .tool) && $0.error == nil } ?? false
        for e in events {
            var t = ChatTurn(role: .user, text: "")
            t.sessionEvent = e
            parent.transcript.append(t)
        }
        parent.transcript.updated = Date()
        saveChat(parent)
        if agentSpokeLast, parent.autoContinues < 5 {
            parent.autoContinues += 1
            reply(parent)
        } else {
            sessionChanged(parent)
            unloadIfHidden(parent)
        }
    }

    // MARK: forking and moving

    /// A new chat from the shown branch up to `turnID`, in front. In a worktree it gets a new worktree from that
    /// branch; else it works in the same folder.
    func fork(_ doc: ChatDocument, at turnID: UUID) {
        guard var t = doc.transcript.fork(upTo: turnID) else { return }
        t.depth = nil
        t.worktree = nil
        let source = doc.transcript.worktree
        let sourceTitle = doc.title
        let oldFolder = chatStore.filesFolder(doc.id)
        if FileManager.default.fileExists(atPath: oldFolder.path) {
            try? FileManager.default.copyItem(at: oldFolder, to: chatStore.filesFolder(t.id))
        }
        let copy = ChatDocument(transcript: t)
        func open() {
            chats.append(copy)
            activeChatID = copy.id
            saveChat(copy)
            refreshWorktreeStatus(copy)
        }
        guard let w = source else {
            if t.projectPath != nil { copy.attachNote = "This fork works in the same folder as “\(sourceTitle)”." }
            open()
            return
        }
        Task { @MainActor in
            do {
                let made = try await GitWorktree.create(repo: URL(fileURLWithPath: w.repo),
                                                        branch: worktreeBranchPrefix + GitWorktree.slug(copy.title), base: w.branch)
                copy.transcript.projectPath = made.path.path
                copy.transcript.worktree = SessionWorktree(repo: w.repo, path: made.path.path, branch: made.branch, base: w.branch)
            } catch {
                copy.attachNote = "Couldn't make a worktree (\(error.localizedDescription)). This fork works in the same folder as “\(sourceTitle)”."
            }
            open()
        }
    }

    /// Stops the reply and carries on with the task in a new session (a fork of this one).
    func moveTaskToNewSession(_ doc: ChatDocument) {
        stopReply(doc)
        guard let text = doc.transcript.turns.last(where: { $0.role == .user && $0.sessionEvent == nil && $0.summary != true })?.text
        else { return }
        let spec = SessionSpec(title: doc.title + " (continued)", prompt: "Continue this task: " + text, context: .fork,
                               worktree: doc.transcript.worktree != nil)
        Task { @MainActor in
            do {
                let info = try await startChild(spec, parent: doc)
                doc.attachNote = "Continuing in the new session “\(info.title)”."
            } catch {
                doc.attachNote = error.localizedDescription
            }
        }
    }

    // MARK: pin, archive, rename, stop (the sidebar)

    private func changeSession(_ id: UUID, _ change: (inout ChatTranscript) -> Void) {
        if let d = chats.first(where: { $0.id == id }) {
            change(&d.transcript)
            saveChat(d)
        } else if var t = chatStore.load(id) {
            change(&t)
            try? chatStore.save(t)
        }
        reloadChats()
    }

    private func savedTranscript(_ id: UUID) -> ChatTranscript? {
        chats.first(where: { $0.id == id })?.transcript ?? chatStore.load(id)
    }

    func setPinned(_ id: UUID, _ on: Bool) {
        changeSession(id) { $0.pinned = on ? true : nil }
    }

    func setArchived(_ id: UUID, _ on: Bool) {
        guard on, let w = savedTranscript(id)?.worktree else {
            changeSession(id) { $0.archived = on ? true : nil }
            return
        }
        confirmRemoveWorktree(w, title: "Remove its worktree too?", removeLabel: "Remove Worktree", keepLabel: "Keep It",
                              keep: { [self] in changeSession(id) { $0.archived = true } },
                              remove: { [self] in
            Task { @MainActor in
                if await removeWorktreeNow(id) { changeSession(id) { $0.archived = true } }
            }
        })
    }

    func rename(_ id: UUID, to title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        changeSession(id) { $0.title = t }
    }

    func stopSession(_ id: UUID) {
        if let d = chats.first(where: { $0.id == id }) { stopReply(d) }
    }

    // MARK: worktrees

    /// Asks whether to remove a worktree (naming unmerged commits), then runs `remove` or `keep`.
    private func confirmRemoveWorktree(_ w: SessionWorktree, title: String, removeLabel: String, keepLabel: String?,
                                       keep: @escaping @MainActor () -> Void, remove: @escaping @MainActor () -> Void) {
        Task { @MainActor in
            let n = await GitWorktree.unmergedCommits(branch: w.branch, base: w.base, repo: URL(fileURLWithPath: w.repo))
            var message = "Uncommitted changes in it are lost."
            if n > 0 {
                message = "Its branch \(w.branch) has \(n) commit\(n == 1 ? "" : "s") not merged into \(w.base). The branch is kept. " + message
            }
            var actions: [MGAlertAction] = [.destructive(removeLabel, action: remove)]
            if let keepLabel { actions.append(MGAlertAction(label: keepLabel, action: keep)) }
            actions.append(.cancel())
            presenter.alert(title, message: message, actions: actions)
        }
    }

    /// Removes the session's worktree (the branch too when everything is merged); the session keeps working in the
    /// repo's folder. False when it failed (an alert says why).
    @discardableResult
    private func removeWorktreeNow(_ id: UUID) async -> Bool {
        guard let w = savedTranscript(id)?.worktree else { return true }
        if let d = chats.first(where: { $0.id == id }) {
            stopReply(d)
            d.jobs.stopAll()
        }
        let repo = URL(fileURLWithPath: w.repo)
        let unmerged = await GitWorktree.unmergedCommits(branch: w.branch, base: w.base, repo: repo)
        do {
            try await GitWorktree.remove(URL(fileURLWithPath: w.path), repo: repo, branch: w.branch, force: true,
                                         deleteBranch: unmerged == 0)
        } catch {
            presenter.alert("Couldn't remove the worktree", message: error.localizedDescription, actions: [.cancel("OK")])
            return false
        }
        changeSession(id) { $0.worktree = nil; $0.projectPath = w.repo }
        chats.first(where: { $0.id == id })?.worktreeStatus = nil
        return true
    }

    func removeWorktree(_ doc: ChatDocument) {
        guard let w = doc.transcript.worktree else { return }
        confirmRemoveWorktree(w, title: "Remove the worktree of “\(doc.title)”?", removeLabel: "Remove Worktree", keepLabel: nil,
                              keep: {}, remove: { [self] in Task { @MainActor in await removeWorktreeNow(doc.id) } })
    }

    func mergeWorktree(_ doc: ChatDocument) {
        guard let w = doc.transcript.worktree else { return }
        presenter.alert("Merge \(w.branch) into \(w.base)?",
                        message: "It merges into the branch checked out in the main folder.", actions: [
            MGAlertAction(label: "Merge") { [self] in
                Task { @MainActor in
                    do {
                        let out = try await GitWorktree.merge(branch: w.branch, into: URL(fileURLWithPath: w.repo))
                        presenter.alert("Merged \(w.branch)", message: String(out.prefix(1000)), actions: [.cancel("OK")])
                    } catch {
                        presenter.alert("Couldn't merge", message: error.localizedDescription, actions: [.cancel("OK")])
                    }
                    refreshWorktreeStatus(doc)
                }
            },
            .cancel(),
        ])
    }

    func createPullRequest(_ doc: ChatDocument) {
        guard let w = doc.transcript.worktree, doc.canCreatePR else { return }
        presenter.alert("Push \(w.branch) and open a pull request?", message: "This sends the branch to the remote.", actions: [
            MGAlertAction(label: "Push and Open") { [self] in
                Task { @MainActor in
                    let dir = URL(fileURLWithPath: w.path)
                    let branch = shellQuoted(w.branch)
                    let push = await ShellRunner().run("git push -u origin \(branch)", cwd: dir, timeout: 180)
                    guard push.status == 0 else {
                        presenter.alert("Couldn't push", message: String(push.output.suffix(1000)), actions: [.cancel("OK")])
                        return
                    }
                    let pr = await ShellRunner().run("gh pr create --fill --head \(branch)", cwd: dir, timeout: 120)
                    presenter.alert(pr.status == 0 ? "Pull request opened" : "Couldn't open the pull request",
                                    message: String(pr.output.suffix(1000)), actions: [.cancel("OK")])
                    refreshWorktreeStatus(doc)
                }
            },
            .cancel(),
        ])
    }

    private func shellQuoted(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    /// Updates the doc's worktree status (ahead / dirty) and whether a pull request can be made.
    func refreshWorktreeStatus(_ doc: ChatDocument) {
        guard let w = doc.transcript.worktree else {
            doc.worktreeStatus = nil
            return
        }
        Task { @MainActor [weak doc] in
            let status = try? await GitWorktree.status(URL(fileURLWithPath: w.path), base: w.base)
            let pr = await GitWorktree.canCreatePR(URL(fileURLWithPath: w.repo))
            doc?.worktreeStatus = status
            doc?.canCreatePR = pr
        }
    }
}

/// What a coding chat's session tools talk to: the app, for the chat that owns them.
final class AppSessionHost: SessionHost, @unchecked Sendable {
    private weak var model: AppModel?
    private let parentID: UUID
    let depth: Int

    init(model: AppModel, parent: UUID, depth: Int) {
        self.model = model
        parentID = parent
        self.depth = depth
    }

    @MainActor private func parent() throws -> (AppModel, ChatDocument) {
        guard let model, let doc = model.chats.first(where: { $0.id == parentID }) else {
            throw ToolError("The session that asked is gone.")
        }
        return (model, doc)
    }

    func propose(_ spec: SessionSpec, callID: String) async -> String {
        await MainActor.run {
            (try? parent())?.1.proposals[callID] = SessionProposal(callID: callID, spec: spec)
            return "Proposed “\(spec.title)”: the user decides whether to start it."
        }
    }

    func start(_ spec: SessionSpec, callID: String) async throws -> SessionInfo {
        let (model, doc) = try await MainActor.run { try parent() }
        return try await model.startChild(spec, parent: doc)
    }

    func children() async -> [SessionInfo] {
        await MainActor.run { model?.children(of: parentID) ?? [] }
    }

    func send(_ text: String, to id: UUID) async throws {
        try await MainActor.run {
            let (model, _) = try parent()
            try model.sendToChild(text, id: id, parent: parentID)
        }
    }

    func handOff(_ spec: SessionSpec, callID: String) async throws -> SessionInfo {
        let (model, doc) = try await MainActor.run { try parent() }
        return try await model.startChild(spec, parent: doc)
    }
}
