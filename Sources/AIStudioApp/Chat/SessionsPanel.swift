import AppKit
import LLMClient
import SwiftUI

/// One line of the sessions list: a saved chat, or a loaded one that isn't saved yet, with its live state.
struct SessionItem: Identifiable, Equatable {
    var id: UUID
    var title: String
    var modelID: String
    var date: Date?
    var status: SessionStatus
    var pinned: Bool
    var archived: Bool
    var parentID: UUID?
    var branch: String?
    var projectName: String?
    var coding: Bool
    var helper: Bool
    var open: Bool
    var snippet: String?
    var depth = 0
    var todoDone: Int?
    var todoTotal: Int?
}

/// Rail panel: sessions with live status, grouped by project, with pinned and archived sections and transcript search.
struct SessionsPanel: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    @State private var showArchived = false
    @State private var renamingID: UUID?
    @State private var renameText = ""
    @State private var found: [UUID: String] = [:]

    private var trimmedQuery: String { query.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Sessions") {
                Button { model.chooseProjectAndChat() } label: { Image(systemName: "chevron.left.forwardslash.chevron.right") }
                    .mgHelp("New coding session in a project folder (⇧⌥⌘N)").accessibilityLabel("New coding session")
                Button { model.newChat() } label: { Image(systemName: "square.and.pencil") }
                    .mgHelp("New chat (⌥⌘N)").accessibilityLabel("New chat")
            }
            MGSearchField(prompt: "Search sessions", text: $query)
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            MGDivider()
            let items = allItems()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    if items.isEmpty {
                        Text("Sessions with language models on your GPU server. Add models in Servers › Language Models, from any Hugging Face link.")
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(12)
                    } else if !trimmedQuery.isEmpty {
                        searchResults(items)
                    } else {
                        sections(items)
                    }
                }
                .padding(.vertical, 6)
            }
            .mgScrollIndicators()
        }
        .task { model.reloadChats() }
        .task(id: query) {
            let q = trimmedQuery
            guard !q.isEmpty else { found = [:]; return }
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            let store = model.chatStore
            let result = await Task.detached(priority: .userInitiated) { store.search(q) }.value
            guard !Task.isCancelled else { return }
            found = result
        }
    }

    // MARK: the list

    /// Every saved session plus loaded ones that aren't saved yet; loaded ones show their live title and status.
    private func allItems() -> [SessionItem] {
        let docs = model.chats + model.helperChats
        var items: [SessionItem] = model.chatSummaries.map { s in
            if let d = docs.first(where: { $0.id == s.id }) {
                return item(of: d, date: s.updated)
            }
            return SessionItem(id: s.id, title: s.title, modelID: s.modelID ?? "", date: s.updated, status: s.status,
                               pinned: s.pinned, archived: s.archived, parentID: s.parentID, branch: s.branch,
                               projectName: s.projectName, coding: s.coding, helper: s.helper, open: false,
                               todoDone: s.todoDone, todoTotal: s.todoTotal)
        }
        let saved = Set(model.chatSummaries.map(\.id))
        for d in model.chats where !saved.contains(d.id) { items.append(item(of: d, date: nil)) }
        return items.sorted { ($0.date ?? .distantFuture) > ($1.date ?? .distantFuture) }
    }

    private func item(of d: ChatDocument, date: Date?) -> SessionItem {
        let t = d.transcript
        let progress = d.isStreaming && d.liveTodos != nil
            ? d.liveTodos.map { l in (l.filter { $0.status == "completed" }.count, l.count) } : t.todoProgress
        return SessionItem(id: d.id, title: d.title, modelID: t.modelID ?? "", date: date, status: model.status(of: d),
                           pinned: t.pinned == true, archived: t.archived == true, parentID: t.parentID,
                           branch: t.worktree?.branch,
                           projectName: (t.worktree?.repo ?? t.projectPath).map { ($0 as NSString).lastPathComponent },
                           coding: d.isCoding, helper: d.isGraphHelper, open: d.tabOpen,
                           todoDone: progress?.0, todoTotal: progress?.1)
    }

    /// The items of a section with each child right under its parent (when the parent is in the same section).
    private func nested(_ items: [SessionItem]) -> [SessionItem] {
        let ids = Set(items.map(\.id))
        var out: [SessionItem] = []
        var seen = Set<UUID>()
        func add(_ i: SessionItem, depth: Int) {
            guard seen.insert(i.id).inserted else { return }
            var x = i
            x.depth = depth
            out.append(x)
            for c in items where c.parentID == i.id { add(c, depth: depth + 1) }
        }
        for i in items where i.parentID == nil || !ids.contains(i.parentID!) { add(i, depth: 0) }
        for i in items { add(i, depth: 0) }   // a cycle left over
        return out
    }

    @ViewBuilder private func sections(_ items: [SessionItem]) -> some View {
        let live = items.filter { !$0.archived }
        let pinned = live.filter(\.pinned)
        let rest = live.filter { !$0.pinned }
        let coding = rest.filter { $0.coding && !$0.helper }
        let projects = Dictionary(grouping: coding) { $0.projectName ?? "Projects" }
            .sorted { ($0.value.first?.date ?? .distantFuture) > ($1.value.first?.date ?? .distantFuture) }
        let chats = rest.filter { !$0.coding && !$0.helper }
        let helpers = rest.filter(\.helper)
        let archived = items.filter(\.archived)
        section("Pinned", pinned)
        ForEach(projects, id: \.key) { section($0.key, $0.value) }
        section("Chats", chats)
        section("Workflow helpers", helpers)
        if !archived.isEmpty {
            Button { showArchived.toggle() } label: {
                HStack(spacing: 6) {
                    Image(systemName: showArchived ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .semibold))
                        .frame(width: 10)
                    Text("Archived (\(archived.count))")
                    Spacer(minLength: 0)
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .padding(.horizontal, 16).padding(.top, 10).padding(.bottom, 2)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if showArchived { ForEach(nested(archived)) { row($0) } }
        }
    }

    @ViewBuilder private func section(_ title: String, _ items: [SessionItem]) -> some View {
        if !items.isEmpty {
            Text(title.uppercased()).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                .padding(.horizontal, 16).padding(.top, 10).padding(.bottom, 2)
            ForEach(nested(items)) { row($0) }
        }
    }

    @ViewBuilder private func searchResults(_ items: [SessionItem]) -> some View {
        let q = trimmedQuery
        let byTitle = items.filter { $0.title.range(of: q, options: .caseInsensitive) != nil }
        let titleIDs = Set(byTitle.map(\.id))
        let byText = items.filter { !titleIDs.contains($0.id) && found[$0.id] != nil }
        if byTitle.isEmpty && byText.isEmpty {
            Text("No sessions match “\(q)”.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).padding(12)
        }
        let shown = (byTitle + byText).map { i -> SessionItem in
            var x = i
            x.snippet = found[i.id]
            return x
        }
        ForEach(shown) { row($0, query: q) }
    }

    private func row(_ i: SessionItem, query: String? = nil) -> some View {
        SessionRow(item: i, active: i.id == model.activeChatID, query: query,
                   renaming: renamingID == i.id, renameText: $renameText,
                   commitRename: { commitRename(i.id) }, cancelRename: { renamingID = nil })
            .onTapGesture(count: 2) { renameText = i.title; renamingID = i.id }
            .onTapGesture { openIfNeeded(i) }
            .mgContextMenu { menu(for: i) }
    }

    private func openIfNeeded(_ i: SessionItem) {
        guard renamingID != i.id else { return }
        model.openChat(i.id)
    }

    private func commitRename(_ id: UUID) {
        model.rename(id, to: renameText)
        renamingID = nil
    }

    // MARK: actions

    @MainActor private func menu(for i: SessionItem) -> [MGMenuItem] {
        var items: [MGMenuItem] = [
            .button("Open") { model.openChat(i.id) },
            .button("Rename…") { renameText = i.title; renamingID = i.id },
            .button(i.pinned ? "Unpin" : "Pin") { model.setPinned(i.id, !i.pinned) },
            .button(i.archived ? "Unarchive" : "Archive") { model.setArchived(i.id, !i.archived) },
            .button("Fork") { fork(i.id) },
        ]
        if i.status == .working || i.status == .waiting {
            items.append(.button("Stop") { model.stopSession(i.id) })
        }
        if i.branch != nil {
            items.append(.divider)
            items.append(.button("Show Worktree in Finder") {
                if let w = loaded(i.id)?.transcript.worktree ?? model.chatStore.load(i.id)?.worktree {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: w.path)])
                }
            })
            items.append(.button("Merge Branch…") { withDoc(i.id) { model.mergeWorktree($0) } })
            items.append(.button("Create Pull Request…") { withDoc(i.id) { model.createPullRequest($0) } })
            items.append(.button("Remove Worktree…") { withDoc(i.id) { model.removeWorktree($0) } })
        }
        items.append(.divider)
        items.append(.button("Delete…", destructive: true) { confirmDelete(i) })
        return items
    }

    private func loaded(_ id: UUID) -> ChatDocument? { model.chats.first { $0.id == id } }

    /// Opens the session if needed, then runs `body` with its document.
    @MainActor private func withDoc(_ id: UUID, _ body: (ChatDocument) -> Void) {
        model.openChat(id)
        if let d = loaded(id) { body(d) }
    }

    /// A new session from the last turn of this one.
    @MainActor private func fork(_ id: UUID) {
        let wasLoaded = loaded(id) != nil
        guard let doc = model.loadSession(id), let last = doc.transcript.turns.last else { return }
        model.fork(doc, at: last.id)
        if !wasLoaded { model.unloadIfHidden(doc) }
    }

    private func confirmDelete(_ i: SessionItem) {
        model.presenter.alert("Delete “\(i.title)”?", message: "The session moves to the Trash.", actions: [
            .destructive("Delete") { model.deleteChat(i.id) },
            .cancel(),
        ])
    }
}

struct SessionRow: View {
    let item: SessionItem
    let active: Bool
    var query: String?
    var renaming = false
    @Binding var renameText: String
    var commitRename: () -> Void = {}
    var cancelRename: () -> Void = {}
    @FocusState private var focused: Bool

    private var kindIcon: String {
        item.helper ? "wand.and.sparkles" : item.coding ? "chevron.left.forwardslash.chevron.right" : "bubble.left"
    }

    var body: some View {
        HStack(spacing: 8) {
            glyph.frame(width: 14, height: 14)
            VStack(alignment: .leading, spacing: 1) {
                if renaming {
                    TextField("Title", text: $renameText)
                        .textFieldStyle(.plain).font(.mgCallout)
                        .focused($focused)
                        .onSubmit(commitRename)
                        .onExitCommand(perform: cancelRename)
                        .onAppear { focused = true }
                } else {
                    Text(item.title).font(.mgCallout).lineLimit(1)
                }
                HStack(spacing: 6) {
                    if !item.modelID.isEmpty { Text(item.modelID).lineLimit(1) }
                    if let b = item.branch {
                        HStack(spacing: 2) {
                            Image(systemName: "arrow.triangle.branch")
                            Text(b).font(.system(.caption, design: .monospaced)).lineLimit(1)
                        }
                    }
                    if let n = item.todoDone, let total = item.todoTotal {
                        HStack(spacing: 2) {
                            Image(systemName: "checklist")
                            Text("\(n)/\(total)").monospacedDigit()
                        }
                        .mgHelp("\(n) of \(total) tasks done")
                    }
                    if let d = item.date { Text(d, format: .relative(presentation: .named)).lineLimit(1) }
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                if let s = item.snippet {
                    Text(highlighted(s)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2)
                }
            }
            Spacer(minLength: 0)
            if item.open { Circle().fill(active ? MG.accent : MG.tertiaryLabel).frame(width: 6, height: 6) }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
            .fill(active ? MG.selection : .clear))
        .padding(.leading, 6 + CGFloat(min(item.depth, 2)) * 14)
        .padding(.trailing, 6)
        .contentShape(Rectangle())
    }

    @ViewBuilder private var glyph: some View {
        switch item.status {
        case .working:
            ProgressView().controlSize(.mini)
        case .waiting:
            Circle().fill(MG.warning).frame(width: 8, height: 8)
        case .unread:
            Circle().fill(MG.accent).frame(width: 8, height: 8)
        case .failed:
            Image(systemName: "exclamationmark.circle.fill").font(.system(size: 12)).foregroundStyle(MG.destructive)
        case .idle:
            Image(systemName: kindIcon).font(.system(size: 11)).foregroundStyle(active ? MG.accent : MG.secondaryLabel)
        }
    }

    /// The snippet with the first match of the query in bold.
    private func highlighted(_ s: String) -> AttributedString {
        var a = AttributedString(s)
        if let q = query, !q.isEmpty, let r = a.range(of: q, options: .caseInsensitive) {
            a[r].font = .mgFootnote.weight(.bold)
        }
        return a
    }
}
