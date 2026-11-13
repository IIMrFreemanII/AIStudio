import Agent
import AppKit
import SwiftUI

// MARK: shared

enum GitUI {
    /// Asks before pushing, then runs `then`.
    @MainActor static func confirmPush(_ model: AppModel, branch: String, then: @escaping @MainActor () -> Void) {
        model.presenter.alert("Push \(branch) to origin?", message: "This sends its commits to the remote.", actions: [
            MGAlertAction(label: "Push") { then() },
            .cancel(),
        ])
    }

    static func letter(_ kind: GitChangeKind) -> (String, Color) {
        switch kind {
        case .modified: return ("M", MG.warning)
        case .added: return ("A", MG.success)
        case .untracked: return ("U", MG.success)
        case .deleted: return ("D", MG.destructive)
        case .renamed: return ("R", MG.accent)
        case .conflicted: return ("!", MG.destructive)
        }
    }
}

// MARK: header: branch menu and pull request pill

/// The branch chip in a coding chat's header: branch, what's ahead / behind / uncommitted, and the git actions.
struct GitBranchMenu: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    /// `.plain` inside the chat header's project group (the group draws the capsule).
    var kind: MGButtonKind = .bordered

    var body: some View {
        if let git = doc.git {
            let s = git.status
            let ahead = s.upstream != nil ? s.ahead : git.aheadOfBase
            MGMenuButton(kind: kind) {
                MGMenuItem.button(doc.changesOpen ? "Hide Changes" : "Show Changes", icon: "doc.badge.ellipsis") {
                    model.toggleChanges(doc)
                }
                MGMenuItem.button("Commit…", icon: "checkmark.circle", enabled: !s.files.isEmpty) {
                    doc.changesScope = .uncommitted
                    if !doc.changesOpen { model.toggleChanges(doc) }
                }
                MGMenuItem.button("Push", icon: "arrow.up", enabled: s.branch != nil && (s.upstream == nil || s.ahead > 0)) {
                    guard let branch = s.branch else { return }
                    GitUI.confirmPush(model, branch: branch) { Task { _ = await model.pushBranch(doc) } }
                }
                MGMenuItem.divider
                if let pr = git.pr {
                    MGMenuItem.button("Open Pull Request #\(pr.number)", icon: "arrow.up.right.square") { model.openPullRequest(doc) }
                    if pr.state == "OPEN" {
                        MGMenuItem.button("Merge Pull Request…", icon: "arrow.triangle.merge") {
                            model.presenter.alert("Merge pull request #\(pr.number)?", message: pr.title, actions: [
                                MGAlertAction(label: "Squash and Merge") { Task { _ = await model.mergePullRequest(doc, method: .squash) } },
                                MGAlertAction(label: "Rebase and Merge") { Task { _ = await model.mergePullRequest(doc, method: .rebase) } },
                                MGAlertAction(label: "Create a Merge Commit") { Task { _ = await model.mergePullRequest(doc, method: .merge) } },
                                .cancel(),
                            ])
                        }
                    }
                } else if git.ghReady && !git.onBase && git.aheadOfBase > 0 {
                    MGMenuItem.button("Create Pull Request…", icon: "arrow.up.right.square") { model.openPRSheet(doc) }
                } else if !git.ghReady && !git.onBase {
                    MGMenuItem.header("Pull requests need gh signed in and an origin remote")
                }
                if let w = doc.transcript.worktree {
                    MGMenuItem.divider
                    MGMenuItem.button("Merge into \(w.base)…", icon: "arrow.triangle.merge") { model.mergeWorktree(doc) }
                    MGMenuItem.button("Show Worktree in Finder", icon: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: w.path)])
                    }
                    MGMenuItem.button("Remove Worktree…", icon: "trash") { model.removeWorktree(doc) }
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.triangle.branch").foregroundStyle(MG.secondaryLabel)
                    Text(s.branch ?? s.head).lineLimit(1).truncationMode(.middle)
                    if ahead > 0 { Text("↑\(ahead)").foregroundStyle(MG.success) }
                    if s.behind > 0 { Text("↓\(s.behind)").foregroundStyle(MG.warning) }
                    if !s.files.isEmpty { Text("•\(s.files.count)").foregroundStyle(MG.warning) }
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp("\(s.branch ?? s.head) · base \(git.base)" + (doc.transcript.worktree != nil ? " · worktree" : ""))
        }
    }
}

/// A pull request's number and state (checks, merged, closed…); a click opens it.
struct PRPill: View {
    let pr: PRInfo
    /// `.plain` inside the chat header's project group.
    var kind: MGButtonKind = .bordered

    var help: String {
        var s = pr.title
        if pr.review == "APPROVED" { s += " · approved" }
        if pr.review == "CHANGES_REQUESTED" { s += " · changes requested" }
        return s
    }

    var body: some View {
        Button {
            if let url = URL(string: pr.url) { NSWorkspace.shared.open(url) }
        } label: {
            HStack(spacing: 4) {
                Text("#\(pr.number)").fontWeight(.medium)
                state
            }
            .font(.mgFootnote)
        }
        .buttonStyle(.mg(kind))
        .mgHelp(help)
    }

    @ViewBuilder var state: some View {
        if pr.state == "MERGED" {
            Text("Merged").foregroundStyle(Color.purple)
        } else if pr.state == "CLOSED" {
            Text("Closed").foregroundStyle(MG.secondaryLabel)
        } else if pr.draft {
            Text("Draft").foregroundStyle(MG.secondaryLabel)
        } else if pr.failed > 0 {
            Text("\(pr.failed) failed").foregroundStyle(MG.destructive)
        } else if pr.pending > 0 {
            HStack(spacing: 4) {
                MGSpinner().controlSize(.mini)
                Text("Checks running").foregroundStyle(MG.secondaryLabel)
            }
        } else if pr.passed > 0 {
            Text("Checks passed").foregroundStyle(MG.success)
        } else {
            Text("Open").foregroundStyle(MG.secondaryLabel)
        }
    }
}

// MARK: the pane

/// Changed files, the selected file's diff, the commit box and pull request actions of a coding chat.
struct GitChangesPane: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument

    private var scope: ChangesScope {
        guard let git = doc.git, doc.changesScope == .base, !(git.onBase && git.aheadOfBase == 0) else { return .uncommitted }
        return .base
    }

    private var files: [GitChange] {
        guard let git = doc.git else { return [] }
        return scope == .base ? git.baseChanges : git.status.files
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            MGDivider()
            if let e = doc.gitError {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(4).textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button { doc.gitError = nil } label: { Image(systemName: "xmark") }.accessibilityLabel("Dismiss Error")
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                }
                .padding(.horizontal, 12).padding(.vertical, 6)
                .background(MG.destructive.opacity(0.10))
                MGDivider()
            }
            content
            if let git = doc.git {
                if scope == .uncommitted && !git.status.files.isEmpty {
                    MGDivider()
                    commitBox(git)
                }
                if let row = footer(git) {
                    MGDivider()
                    row
                }
            }
        }
        .background(MG.windowBackground)
    }

    // MARK: header

    private var header: some View {
        HStack(spacing: 8) {
            if let git = doc.git {
                let showBase = !(git.onBase && git.aheadOfBase == 0)
                MGSegmentedPicker(selection: Binding(get: { scope }, set: { doc.changesScope = $0 }),
                                  options: [(ChangesScope.uncommitted, "Uncommitted (\(git.status.files.count))")]
                                  + (showBase ? [(ChangesScope.base, "vs \(git.base) (\(git.baseChanges.count))")] : []))
            }
            Spacer(minLength: 0)
            Button { model.refreshGitSoon(doc) } label: {
                if doc.gitRefreshing { MGSpinner().controlSize(.mini) } else { Image(systemName: "arrow.clockwise") }
            }
            .buttonStyle(.mg(.icon))
            .mgHelp("Refresh")
            .accessibilityLabel("Refresh")
        }
        .padding(.horizontal, 12)
        .frame(height: 38)
    }

    // MARK: files and diff

    @ViewBuilder private var content: some View {
        if let git = doc.git {
            let list = files
            if list.isEmpty {
                MGEmptyState(symbol: "checkmark.circle", title: "No changes",
                             message: scope == .base ? "Nothing differs from \(git.base)." : nil)
            } else {
                let selected = list.first { $0.path == doc.selectedChange } ?? list[0]
                VSplitView {
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(list) { change in
                                ChangeRow(doc: doc, git: git, change: change, selected: change.path == selected.path, scope: scope)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .mgScrollIndicators()
                    .frame(minHeight: 70, idealHeight: 160)
                    ChangeDiff(doc: doc, git: git, change: selected, scope: scope)
                        .frame(minHeight: 120, idealHeight: 300)
                }
            }
        } else if doc.gitRefreshing {
            VStack { MGSpinner().controlSize(.regular) }.frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            MGEmptyState(symbol: "arrow.triangle.branch", title: "Not a git repository",
                         message: "Run git init in the project folder to see changes here.")
        }
    }

    // MARK: commit

    private func commitBox(_ git: GitState) -> some View {
        let empty = doc.commitDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        let modelName = doc.modelKey.flatMap { model.llm.model($0)?.name } ?? doc.transcript.modelID ?? "the model"
        return VStack(alignment: .leading, spacing: 8) {
            TextField("Commit message", text: $doc.commitDraft, axis: .vertical).lineLimit(2...6)
            HStack(spacing: 8) {
                Button { Task { await model.draftCommitMessage(doc) } } label: {
                    if doc.drafting {
                        HStack(spacing: 5) { MGSpinner().controlSize(.mini); Text("Draft") }
                    } else {
                        Label("Draft", systemImage: "sparkles")
                    }
                }
                .buttonStyle(.mg(.bordered))
                .disabled(doc.modelKey == nil || doc.drafting)
                .mgHelp("Write the message with \(modelName)")
                Spacer(minLength: 0)
                let total = git.status.files.count
                let picked = git.status.files.filter { !doc.commitExcluded.contains($0.path) }.count
                Button(picked == total ? "Commit" : "Commit \(picked) of \(total)") { Task { _ = await model.commit(doc, push: false) } }
                    .buttonStyle(.mg(.prominent))
                    .disabled(empty || picked == 0)
                    .mgHelp(picked == total ? "Commit all the changed files" : "Commit the ticked files; the others stay uncommitted")
                    .mgTestAnchor("commit-button")
                MGMenuButton(kind: .icon) {
                    MGMenuItem.button("Commit and Push", icon: "arrow.up", enabled: !empty && git.status.branch != nil) {
                        guard let branch = git.status.branch else { return }
                        GitUI.confirmPush(model, branch: branch) { Task { _ = await model.commit(doc, push: true) } }
                    }
                } label: {
                    Image(systemName: "chevron.down")
                }
            }
            if let line = model.gitCoAuthorLine(for: doc) {
                Text("Adds: \(line)").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2).truncationMode(.middle)
            }
        }
        .padding(12)
    }

    // MARK: pull request

    private func footer(_ git: GitState) -> AnyView? {
        if let pr = git.pr {
            return AnyView(HStack(spacing: 8) {
                PRPill(pr: pr)
                Text(pr.title).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                Spacer(minLength: 0)
                Button("Open") { model.openPullRequest(doc) }.buttonStyle(.mg(.bordered))
            }
            .padding(.horizontal, 12).padding(.vertical, 8))
        }
        if git.ghReady && !git.onBase && git.aheadOfBase > 0 {
            return AnyView(Button { model.openPRSheet(doc) } label: {
                Text("Create Pull Request…").frame(maxWidth: .infinity)
            }
            .buttonStyle(.mg(.bordered))
            .padding(12))
        }
        if !git.ghReady && !git.onBase && git.aheadOfBase > 0 {
            return AnyView(Text("Pull requests need the GitHub CLI (gh) signed in and an origin remote.")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12))
        }
        return nil
    }
}

/// One changed file: status letter, name and folder, and the line counts.
private struct ChangeRow: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let git: GitState
    let change: GitChange
    let selected: Bool
    let scope: ChangesScope

    var body: some View {
        let (letter, tint) = GitUI.letter(change.kind)
        let name = (change.path as NSString).lastPathComponent
        let folder = (change.path as NSString).deletingLastPathComponent
        HStack(spacing: 6) {
            if scope == .uncommitted {
                Toggle("", isOn: Binding(get: { !doc.commitExcluded.contains(change.path) },
                                         set: { on in if on { doc.commitExcluded.remove(change.path) } else { doc.commitExcluded.insert(change.path) } }))
                    .toggleStyle(MGCheckboxStyle())
                    .labelsHidden()
                    .accessibilityLabel("Include \(name) in the commit")
                    .mgHelp("Include it in the next commit")
            }
            Text(letter).font(.mgMono.weight(.semibold)).foregroundStyle(tint).frame(width: 16)
            Text(name).font(.mgCallout.weight(.medium)).lineLimit(1)
            if !folder.isEmpty {
                Text(folder)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
            }
            Spacer(minLength: 4)
            if let a = change.added { Text("+\(a)").foregroundStyle(MG.success) }
            if let r = change.removed { Text("−\(r)").foregroundStyle(MG.destructive) }
        }
        .font(.mgFootnote.monospacedDigit())
        .padding(.horizontal, 12)
        .frame(height: 26)
        .background(selected ? MG.selection : .clear)
        .contentShape(Rectangle())
        .onTapGesture { doc.selectedChange = change.path }
        .mgContextMenu {
            MGMenuItem.button("Open in Default Editor") { NSWorkspace.shared.open(git.root.appendingPathComponent(change.path)) }
            MGMenuItem.button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([git.root.appendingPathComponent(change.path)])
            }
            MGMenuItem.button("Copy Path") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(change.path, forType: .string)
            }
            if scope == .uncommitted {
                MGMenuItem.divider
                MGMenuItem.button("Commit Only This File") {
                    doc.commitExcluded = Set(git.status.files.map(\.path)).subtracting([change.path])
                }
                MGMenuItem.button("Include All Files", enabled: !doc.commitExcluded.isEmpty) { doc.commitExcluded = [] }
                MGMenuItem.divider
                MGMenuItem.button("Revert File…", destructive: true) {
                    let isNew = change.kind == .untracked || change.kind == .added
                    model.presenter.alert("Revert \(change.path)?",
                                          message: isNew ? "The new file is moved to the Trash." : "Its uncommitted changes are lost.",
                                          actions: [.destructive("Revert") { Task { _ = await model.revertFile(doc, change) } }, .cancel()])
                }
            }
        }
    }
}

/// The selected file's diff (cached in the chat until the next refresh).
private struct ChangeDiff: View {
    let doc: ChatDocument
    let git: GitState
    let change: GitChange
    let scope: ChangesScope

    private var key: String { "\(scope)|\(change.path)" }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Text(change.path).font(.mgFootnote.weight(.medium)).lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 12)
            .frame(height: 26)
            .background(MG.groupedBackground)
            MGDivider()
            if let text = doc.diffCache[key] {
                if text.isEmpty {
                    Text("No textual diff (binary file)").font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    DiffView(diff: text, maxHeight: nil)
                }
            } else {
                MGSpinner().controlSize(.small).frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: "\(key)|\(git.status.head)|\(doc.diffCache.count)") {
            guard doc.diffCache[key] == nil else { return }
            let text = await Git.diff(git.root, change: change, base: scope == .base ? git.base : nil)
            doc.diffCache[key] = text
        }
    }
}

// MARK: create pull request

struct CreatePRSheet: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument

    private var drafting: Bool { doc.prSheet?.drafting ?? false }

    var body: some View {
        let modelName = doc.modelKey.flatMap { model.llm.model($0)?.name } ?? doc.transcript.modelID ?? "the model"
        let title = Binding(get: { doc.prSheet?.title ?? "" }, set: { doc.prSheet?.title = $0 })
        let text = Binding(get: { doc.prSheet?.body ?? "" }, set: { doc.prSheet?.body = $0 })
        let base = Binding(get: { doc.prSheet?.base ?? "" }, set: { doc.prSheet?.base = $0 })
        let asDraft = Binding(get: { doc.prSheet?.draft ?? false }, set: { doc.prSheet?.draft = $0 })
        VStack(alignment: .leading, spacing: 12) {
            Text("Create Pull Request").font(.mgHeadline)
            if drafting {
                HStack(spacing: 6) { MGSpinner().controlSize(.small); Text("Drafting with \(modelName)…").foregroundStyle(MG.secondaryLabel) }
                    .font(.mgCallout)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Title").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                TextField("Title", text: title).disabled(drafting)
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Description").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                TextEditor(text: text)
                    .font(.mgMono)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .frame(height: 220)
                    .disabled(drafting)
                    .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
                    .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            }
            VStack(alignment: .leading, spacing: 4) {
                Text("Base branch").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                TextField("Base", text: base).disabled(drafting)
            }
            Toggle("Create as a draft", isOn: asDraft).disabled(drafting)
            if let e = doc.gitError {
                Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(5).textSelection(.enabled)
            }
            HStack {
                Spacer()
                Button("Cancel") { doc.prSheet = nil }.keyboardShortcut(.cancelAction)
                Button("Push and Create") { Task { _ = await model.createPullRequest(doc) } }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                    .disabled((doc.prSheet?.title.trimmingCharacters(in: .whitespaces).isEmpty ?? true) || drafting)
            }
        }
        .padding(20)
        .frame(width: 560)
    }
}
