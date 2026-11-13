import Agent
import AppKit
import Foundation
import LLMClient

/// A coding chat's repository at a glance.
struct GitState: Equatable {
    var root: URL                // the repo's top folder
    var status: GitStatus
    var base: String             // the worktree's base branch, else Git.defaultBranch
    var baseChanges: [GitChange] // Git.changes(root, base:): everything that differs from base
    var aheadOfBase: Int         // GitWorktree.unmergedCommits(branch: "HEAD", base: base, repo: root)
    var ghReady: Bool
    var pr: PRInfo?
    var onBase: Bool { status.branch == base }
}

enum ChangesScope: Hashable { case uncommitted, base }

/// The pull request being written in the Create Pull Request sheet.
struct PRDraft: Identifiable {
    let id = UUID()
    var title = ""
    var body = ""
    var base: String
    var draft = false
    var drafting = true
}

@MainActor enum GitActions {
    /// How a reverted new file is removed (the Trash; the smoke test deletes instead).
    static var trash: (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }
    /// Whether `gh` is signed in, by repo path, with when it was asked (`gh auth status` is slow).
    fileprivate static var ghCache: [String: (at: Date, ready: Bool)] = [:]
}

extension AppModel {
    // MARK: setting

    /// The co-author trailer the agent and the Commit button add (nil: the setting is off).
    func gitCoAuthorLine(for doc: ChatDocument) -> String? {
        guard UserDefaults.standard.object(forKey: "gitCoAuthor") as? Bool ?? true else { return nil }
        return Git.coAuthorLine(model: doc.transcript.modelID ?? "model")
    }

    // MARK: refreshing

    /// Asks for a refresh without waiting: runs one now, or after the one in progress.
    func refreshGitSoon(_ doc: ChatDocument) {
        guard doc.isCoding else { return }
        if doc.gitRefreshing {
            doc.gitRefreshAgain = true
            return
        }
        Task { @MainActor [weak self, weak doc] in
            guard let self, let doc else { return }
            await refreshGit(doc)
        }
    }

    /// Reads the repository's state into `doc.git`. Waits for a refresh in progress, then runs its own, so the state is
    /// fresh when it returns.
    func refreshGit(_ doc: ChatDocument) async {
        guard let project = doc.projectURL else {
            doc.git = nil
            return
        }
        // A folder that isn't on disk (a snapshot's fake project): leave the state alone.
        guard FileManager.default.fileExists(atPath: project.path) else { return }
        while doc.gitRefreshing {
            if Task.isCancelled { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        doc.gitRefreshing = true
        defer { doc.gitRefreshing = false }
        repeat {
            doc.gitRefreshAgain = false
            await loadGit(doc, project: project)
        } while doc.gitRefreshAgain
    }

    private func loadGit(_ doc: ChatDocument, project: URL) async {
        guard let root = await GitWorktree.repoRoot(project), let status = await Git.status(root) else {
            doc.git = nil
            return
        }
        let base: String
        if let b = doc.transcript.worktree?.base { base = b } else { base = await Git.defaultBranch(root) }
        let baseChanges = await Git.changes(root, base: base)
        let ahead = await GitWorktree.unmergedCommits(branch: "HEAD", base: base, repo: root)
        var ready: Bool
        if let c = GitActions.ghCache[root.path], Date().timeIntervalSince(c.at) < 300 {
            ready = c.ready
        } else {
            ready = await Git.ghReady(root)
            GitActions.ghCache[root.path] = (Date(), ready)
        }
        var pr: PRInfo?
        if ready, let branch = status.branch, branch != base { pr = await Git.pullRequest(root, branch: branch) }
        let state = GitState(root: root, status: status, base: base, baseChanges: baseChanges, aheadOfBase: ahead,
                             ghReady: ready, pr: pr)
        if doc.git != state { doc.git = state }
        let paths = Set(state.status.files.map(\.path))
        if !doc.commitExcluded.isSubset(of: paths) { doc.commitExcluded.formIntersection(paths) }
        doc.diffCache = [:]
        let list = doc.changesScope == .uncommitted ? state.status.files : state.baseChanges
        if let sel = doc.selectedChange, !list.contains(where: { $0.path == sel }) { doc.selectedChange = nil }
    }

    // MARK: drafting

    private func diffText(_ root: URL, _ changes: [GitChange], base: String? = nil) async -> String {
        var text = ""
        for c in changes { text += await Git.diff(root, change: c, base: base) + "\n" }
        return text
    }

    /// Has the chat's model write the commit message (into `doc.commitDraft`).
    func draftCommitMessage(_ doc: ChatDocument) async {
        doc.gitError = nil
        guard let git = doc.git, let key = doc.modelKey, let client = llm.client(key.serverID) else {
            doc.gitError = "Couldn't draft: choose a model for this chat."
            return
        }
        doc.drafting = true
        defer { doc.drafting = false }
        llm.lastActivity = Date()
        let diff = await diffText(git.root, git.status.files.filter { !doc.commitExcluded.contains($0.path) })
        let log = await Git.recentLog(git.root, count: 10)
        do {
            doc.commitDraft = try await GitDrafts.commitMessage(client: client.chat, model: key.modelID, diff: diff, log: log)
        } catch {
            doc.gitError = "Couldn't draft: \(error.localizedDescription)"
        }
    }

    // MARK: changing

    /// Commits the ticked files, everything when none is unticked (with the co-author line when it's on), and pushes
    /// when asked.
    func commit(_ doc: ChatDocument, push: Bool) async -> Bool {
        doc.gitError = nil
        guard let git = doc.git else { return false }
        let draft = doc.commitDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !draft.isEmpty else {
            doc.gitError = "Write a commit message first."
            return false
        }
        let files = git.status.files
        let included = files.filter { !doc.commitExcluded.contains($0.path) }
        guard !included.isEmpty else {
            doc.gitError = "Tick at least one file to commit."
            return false
        }
        // A renamed file's old path goes too, so the commit has the rename and not only the new copy.
        let paths = included.count == files.count ? nil : included.flatMap { [$0.path] + ($0.oldPath.map { [$0] } ?? []) }
        var ok = false
        do {
            _ = try await Git.commit(git.root, message: Git.withTrailer(draft, trailer: gitCoAuthorLine(for: doc)), paths: paths)
            if push, let branch = git.status.branch { try await Git.push(git.root, branch: branch) }
            doc.commitDraft = ""
            ok = true
        } catch {
            doc.gitError = error.localizedDescription
        }
        await refreshGit(doc)
        return ok
    }

    func revertFile(_ doc: ChatDocument, _ change: GitChange) async -> Bool {
        doc.gitError = nil
        guard let git = doc.git else { return false }
        var ok = false
        do {
            try await Git.revert(git.root, change: change, trash: GitActions.trash)
            ok = true
        } catch {
            doc.gitError = error.localizedDescription
        }
        await refreshGit(doc)
        return ok
    }

    func pushBranch(_ doc: ChatDocument) async -> Bool {
        doc.gitError = nil
        guard let git = doc.git, let branch = git.status.branch else { return false }
        var ok = false
        do {
            try await Git.push(git.root, branch: branch)
            ok = true
        } catch {
            doc.gitError = error.localizedDescription
        }
        await refreshGit(doc)
        return ok
    }

    // MARK: pull requests

    /// Opens the Create Pull Request sheet and has the model write the title and body.
    func openPRSheet(_ doc: ChatDocument) {
        Task { @MainActor [weak self, weak doc] in
            guard let self, let doc else { return }
            if doc.git == nil { await refreshGit(doc) }
            guard let git = doc.git else { return }
            doc.gitError = nil
            let sheet = PRDraft(base: git.base)
            let id = sheet.id
            doc.prSheet = sheet
            guard let branch = git.status.branch, let key = doc.modelKey, let client = llm.client(key.serverID) else {
                doc.prSheet?.drafting = false
                return
            }
            llm.lastActivity = Date()
            let diff = await diffText(git.root, git.baseChanges, base: git.base)
            let log = await Git.recentLog(git.root, count: 10)
            do {
                let r = try await GitDrafts.pullRequest(client: client.chat, model: key.modelID, diff: diff, log: log,
                                                        branch: branch, base: git.base)
                guard doc.prSheet?.id == id else { return }
                if doc.prSheet?.title.isEmpty == true { doc.prSheet?.title = r.title }
                if doc.prSheet?.body.isEmpty == true { doc.prSheet?.body = r.body }
                doc.prSheet?.drafting = false
            } catch {
                guard doc.prSheet?.id == id else { return }
                doc.prSheet?.drafting = false
                doc.gitError = "Couldn't draft: \(error.localizedDescription)"
            }
        }
    }

    /// Pushes the branch if needed and opens the pull request written in `doc.prSheet`.
    func createPullRequest(_ doc: ChatDocument) async -> Bool {
        doc.gitError = nil
        guard let git = doc.git, let sheet = doc.prSheet, let branch = git.status.branch else { return false }
        do {
            if git.status.upstream == nil || git.status.ahead > 0 { try await Git.push(git.root, branch: branch) }
            _ = try await Git.createPullRequest(git.root, title: sheet.title.trimmingCharacters(in: .whitespacesAndNewlines),
                                                body: sheet.body, base: sheet.base, head: branch, draft: sheet.draft)
        } catch {
            doc.gitError = error.localizedDescription
            return false
        }
        doc.prSheet = nil
        await refreshGit(doc)
        return true
    }

    func mergePullRequest(_ doc: ChatDocument, method: Git.MergeMethod) async -> Bool {
        doc.gitError = nil
        guard let git = doc.git, let pr = git.pr else { return false }
        var ok = false
        do {
            try await Git.mergePullRequest(git.root, number: pr.number, method: method)
            ok = true
        } catch {
            doc.gitError = error.localizedDescription
        }
        await refreshGit(doc)
        return ok
    }

    func openPullRequest(_ doc: ChatDocument) {
        if let s = doc.git?.pr?.url, let url = URL(string: s) { NSWorkspace.shared.open(url) }
    }
}

extension AppModel {
    /// Shows or hides the Changes pane (refreshing when it opens) and remembers it with the chat.
    func toggleChanges(_ doc: ChatDocument) { toggleSidePanel(doc, .changes) }

    /// Shows a tab of the side panel, or closes the panel when it already shows that tab; remembered with the chat.
    func toggleSidePanel(_ doc: ChatDocument, _ panel: SidePanel) {
        showSidePanel(doc, doc.sidePanel == panel && doc.flowPanelRunID == nil ? nil : panel)
    }

    /// Shows or hides the side panel (on the tab it last showed).
    func toggleSidePanel(_ doc: ChatDocument) {
        showSidePanel(doc, doc.sidePanel != nil && doc.flowPanelRunID == nil ? nil : doc.sideTab)
    }

    /// The side panel is on screen (a flow's run, when shown, takes its place).
    func sidePanelShown(_ doc: ChatDocument) -> Bool { doc.isCoding && doc.sidePanel != nil && doc.flowPanelRunID == nil }

    func showSidePanel(_ doc: ChatDocument, _ panel: SidePanel?) {
        guard doc.isCoding else { return }
        doc.sidePanel = panel
        if panel != nil { doc.flowPanelRunID = nil }
        switch panel {
        case .changes: refreshGitSoon(doc)
        case .files: refreshProjectFiles(doc)
        case .plans: savePlanFiles(doc)
        case .web: _ = webPage(doc)
        case nil: break
        }
        saveChat(doc)
    }
}
