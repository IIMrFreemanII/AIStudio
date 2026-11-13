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
    var projectPath: String?
    var coding: Bool
    var helper: Bool
    var open: Bool
    var snippet: String?
    var depth = 0
    var todoDone: Int?
    var todoTotal: Int?
}

/// The sidebar's Sessions panel, as in Claude Code: a New session button, search, then one-line sessions with live
/// status, grouped by project (groups fold away), with pinned and archived sections and transcript search.
struct SessionsPanel: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    /// Folded groups ("project:/path", "pinned", "chats", "helpers", "archived"), kept between launches.
    @AppStorage("foldedSessionGroups") private var foldedRaw = "archived"
    @State private var renamingID: UUID?
    @State private var renameText = ""
    @State private var found: [UUID: String] = [:]

    private var trimmedQuery: String { query.trimmingCharacters(in: .whitespacesAndNewlines) }

    var body: some View {
        let items = allItems()
        VStack(spacing: 0) {
            newSessionButton(items)
                .padding(.horizontal, 10)
                .padding(.top, 10)
                .padding(.bottom, 8)
            MGSearchField(prompt: "Search sessions", text: $query)
                .padding(.horizontal, 10)
                .padding(.bottom, 8)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 1) {
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
                .padding(.bottom, 8)
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
                               projectName: s.projectName, projectPath: s.projectPath, coding: s.coding, helper: s.helper, open: false,
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
                           projectPath: t.worktree?.repo ?? t.projectPath,
                           coding: d.isCoding, helper: d.isGraphHelper, open: d.isRecent,
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

    // MARK: new session

    /// Where "New session" starts one: the project of the chat in front, else of the latest coding session.
    private func defaultProject(_ items: [SessionItem]) -> URL? {
        if let p = model.activeChat?.transcript.worktree?.repo ?? model.activeChat?.transcript.projectPath {
            return URL(fileURLWithPath: p)
        }
        return items.first { $0.coding && !$0.helper && $0.projectPath != nil }.map { URL(fileURLWithPath: $0.projectPath!) }
    }

    @ViewBuilder private func newSessionButton(_ items: [SessionItem]) -> some View {
        let project = defaultProject(items)
        let recent = Array(NSOrderedSet(array: items.compactMap { $0.coding && !$0.helper ? $0.projectPath : nil })
            .compactMap { $0 as? String }.prefix(6))
        HStack(spacing: 6) {
            Button {
                if let project { model.startCodingSession(in: project) } else { model.chooseProjectAndChat() }
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: "plus").font(.system(size: 12, weight: .semibold))
                    Text("New session").font(.mgCallout.weight(.medium))
                    Spacer(minLength: 4)
                    if let project {
                        Text(project.lastPathComponent).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                    }
                }
                .padding(.horizontal, 10)
                .frame(height: 30)
                .contentShape(Rectangle())
            }
            .buttonStyle(.mg(.bordered))
            .mgHelp(project.map { "New coding session in \($0.lastPathComponent) (⇧⌥⌘N picks a folder)" }
                    ?? "New coding session in a project folder (⇧⌥⌘N)")
            .mgTestAnchor("new-session")
            MGMenuButton(kind: .bordered) {
                MGMenuItem.header("New coding session in")
                for path in recent {
                    MGMenuItem.button((path as NSString).lastPathComponent, icon: "folder") {
                        model.startCodingSession(in: URL(fileURLWithPath: path))
                    }
                }
                MGMenuItem.button("Choose a Folder…", icon: "folder.badge.plus") { model.chooseProjectAndChat() }
                MGMenuItem.divider
                MGMenuItem.button("New Chat", icon: "bubble.left") { _ = model.newChat() }
            } label: {
                Image(systemName: "chevron.down").font(.system(size: 10, weight: .semibold)).frame(height: 30)
            }
            .mgHelp("More ways to start: another project, or a chat without one (⌥⌘N)")
        }
    }

    // MARK: groups

    private var folded: Set<String> { Set(foldedRaw.split(separator: "\n").map(String.init)) }

    private func toggleFold(_ key: String) {
        var f = folded
        if f.contains(key) { f.remove(key) } else { f.insert(key) }
        withAnimation(.easeOut(duration: 0.15)) { foldedRaw = f.sorted().joined(separator: "\n") }
    }

    @ViewBuilder private func sections(_ items: [SessionItem]) -> some View {
        let live = items.filter { !$0.archived }
        let pinned = live.filter(\.pinned)
        let rest = live.filter { !$0.pinned }
        let coding = rest.filter { $0.coding && !$0.helper }
        let projects = Dictionary(grouping: coding) { $0.projectPath ?? $0.projectName ?? "" }
            .sorted { ($0.value.first?.date ?? .distantFuture) > ($1.value.first?.date ?? .distantFuture) }
        let chats = rest.filter { !$0.coding && !$0.helper }
        let helpers = rest.filter(\.helper)
        let archived = items.filter(\.archived)
        group("pinned", "Pinned", icon: "pin", pinned)
        ForEach(projects, id: \.key) { p in
            group("project:" + p.key, p.value.first?.projectName ?? "Project", icon: "folder", p.value,
                  newIn: p.value.first?.projectPath.map { URL(fileURLWithPath: $0) })
        }
        group("chats", "Chats", icon: "bubble.left", chats)
        group("helpers", "Workflow helpers", icon: "wand.and.sparkles", helpers)
        group("archived", "Archived", icon: "archivebox", archived)
    }

    @ViewBuilder private func group(_ key: String, _ title: String, icon: String, _ items: [SessionItem],
                                    newIn project: URL? = nil) -> some View {
        if !items.isEmpty {
            let open = !folded.contains(key)
            SessionGroupHeader(title: title, icon: icon, count: items.count, open: open,
                               working: items.contains { $0.status == .working || $0.status == .waiting },
                               onToggle: { toggleFold(key) },
                               onNew: project.map { p in { model.startCodingSession(in: p) } })
                .mgTestAnchor("session-group:\(key)")
            if open { ForEach(nested(items)) { row($0) } }
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
                   commitRename: { commitRename(i.id) }, cancelRename: { renamingID = nil },
                   onPin: { model.setPinned(i.id, !i.pinned) }, onArchive: { model.setArchived(i.id, !i.archived) })
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
            items.append(.button("Create Pull Request…") { withDoc(i.id) { model.openPRSheet($0) } })
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

/// A group's header in the sessions list: a chevron that folds it, its name, how many sessions (folded), and for a
/// project a + that starts a session there.
struct SessionGroupHeader: View {
    let title: String
    let icon: String
    let count: Int
    let open: Bool
    var working = false
    let onToggle: () -> Void
    var onNew: (() -> Void)?
    @State private var hover = false

    var body: some View {
        HStack(spacing: 6) {
            Image(systemName: "chevron.right").font(.system(size: 8.5, weight: .bold))
                .foregroundStyle(MG.tertiaryLabel)
                .rotationEffect(.degrees(open ? 90 : 0))
                .frame(width: 10)
            Image(systemName: icon).font(.system(size: 11)).foregroundStyle(MG.secondaryLabel).frame(width: 14)
            Text(title).font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.secondaryLabel).lineLimit(1)
            if !open {
                Text("\(count)").font(.mgFootnote).monospacedDigit().foregroundStyle(MG.tertiaryLabel)
                if working { MGSpinner().controlSize(.mini) }
            }
            Spacer(minLength: 4)
            if let onNew, hover {
                Button(action: onNew) { Image(systemName: "plus").font(.system(size: 11, weight: .semibold)) }.accessibilityLabel("New Session in Project")
                    .buttonStyle(.mg(.icon))
                    .controlSize(.small)
                    .mgHelp("New session in \(title)")
            }
        }
        .padding(.leading, 10)
        .padding(.trailing, 6)
        .frame(height: 26)
        .contentShape(Rectangle())
        .onTapGesture(perform: onToggle)
        .onHover { hover = $0 }
        .padding(.horizontal, 6)
        .padding(.top, 6)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isButton)
        .accessibilityLabel("\(title), \(count) sessions, \(open ? "expanded" : "collapsed")")
    }
}

/// One session: its status (or kind), its title, and when it was last active; pin and archive buttons on hover.
struct SessionRow: View {
    let item: SessionItem
    let active: Bool
    var query: String?
    var renaming = false
    @Binding var renameText: String
    var commitRename: () -> Void = {}
    var cancelRename: () -> Void = {}
    var onPin: () -> Void = {}
    var onArchive: () -> Void = {}
    @FocusState private var focused: Bool
    @State private var hover = false

    private var kindIcon: String {
        item.helper ? "wand.and.sparkles" : item.coding ? "chevron.left.forwardslash.chevron.right" : "bubble.left"
    }

    var body: some View {
        HStack(spacing: 8) {
            glyph.frame(width: 14, height: 14)
            VStack(alignment: .leading, spacing: 2) {
                if renaming {
                    TextField("Title", text: $renameText)
                        .textFieldStyle(.plain).font(.mgCallout)
                        .focused($focused)
                        .onSubmit(commitRename)
                        .onExitCommand(perform: cancelRename)
                        .onAppear { focused = true }
                } else {
                    Text(item.title)
                        .font(.mgCallout.weight(item.status == .unread ? .semibold : .regular))
                        .foregroundStyle(MG.label)
                        .lineLimit(1)
                }
                if let s = item.snippet {
                    Text(highlighted(s)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2)
                }
            }
            Spacer(minLength: 4)
            trailing
        }
        .padding(.horizontal, 8)
        .frame(minHeight: 28)
        .padding(.vertical, item.snippet == nil ? 0 : 4)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
            .fill(active ? MG.selection : hover ? MG.hover : .clear))
        .padding(.leading, 6 + CGFloat(min(item.depth, 2)) * 14)
        .padding(.trailing, 6)
        .contentShape(Rectangle())
        .onHover { hover = $0 }
        .mgHelp(details)
        .mgTestAnchor("session-row:\(item.id.uuidString)")
    }

    /// Pin and archive while the pointer is on the row; else the to-dos (while working) and how long ago.
    @ViewBuilder private var trailing: some View {
        if hover && !renaming {
            HStack(spacing: 0) {
                Button(action: onPin) { Image(systemName: item.pinned ? "pin.slash" : "pin") }.accessibilityLabel(item.pinned ? "Unpin" : "Pin")
                    .mgHelp(item.pinned ? "Unpin" : "Pin")
                Button(action: onArchive) { Image(systemName: item.archived ? "tray.and.arrow.up" : "archivebox") }.accessibilityLabel(item.archived ? "Unarchive" : "Archive")
                    .mgHelp(item.archived ? "Unarchive" : "Archive")
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.small)
            .font(.system(size: 11))
        } else {
            HStack(spacing: 6) {
                if item.status == .working, let n = item.todoDone, let total = item.todoTotal, total > 0 {
                    Text("\(n)/\(total)").monospacedDigit()
                }
                if item.branch != nil { Image(systemName: "arrow.triangle.branch").font(.system(size: 9.5)) }
                if let d = item.date { Text(Self.ago(d)).monospacedDigit() }
            }
            .font(.mgFootnote)
            .foregroundStyle(MG.tertiaryLabel)
            .lineLimit(1)
        }
    }

    /// The tooltip: the model, the branch, the to-dos and the date.
    private var details: String {
        var parts: [String] = [item.title]
        if !item.modelID.isEmpty { parts.append(item.modelID) }
        if let b = item.branch { parts.append("Branch \(b)") }
        if let n = item.todoDone, let t = item.todoTotal, t > 0 { parts.append("\(n) of \(t) tasks done") }
        if let d = item.date { parts.append(d.formatted(date: .abbreviated, time: .shortened)) }
        return parts.joined(separator: " · ")
    }

    /// "now", "5m", "3h", "4d", then the date ("Nov 3").
    static func ago(_ date: Date, now: Date = Date()) -> String {
        let s = max(0, now.timeIntervalSince(date))
        if s < 60 { return "now" }
        if s < 3600 { return "\(Int(s / 60))m" }
        if s < 86_400 { return "\(Int(s / 3600))h" }
        if s < 7 * 86_400 { return "\(Int(s / 86_400))d" }
        return date.formatted(.dateTime.month(.abbreviated).day())
    }

    @ViewBuilder private var glyph: some View {
        switch item.status {
        case .working:
            MGSpinner().controlSize(.mini)
        case .waiting:
            Circle().fill(MG.warning).frame(width: 7, height: 7)
        case .unread:
            Circle().fill(MG.accent).frame(width: 7, height: 7)
        case .failed:
            Image(systemName: "exclamationmark.circle.fill").font(.system(size: 11)).foregroundStyle(MG.destructive)
        case .idle:
            Image(systemName: kindIcon).font(.system(size: 10.5)).foregroundStyle(active ? MG.accent : MG.tertiaryLabel)
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
