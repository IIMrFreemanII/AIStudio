import AIStudioLibrary
import Agent
import AppKit
import LLMClient
import SwiftUI

/// A coding chat's right-hand panel: the pane the rail picked (files, changes, plans, browser, terminal, or a flow's
/// run). Each pane's own top row is its header; the rail (`SidePanelRail`) switches panes.
struct CodingSidePanel: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument

    var body: some View {
        VStack(spacing: 0) {  // not a Group: its modifiers would go on each pane, and the anchor leave with the old one
            switch doc.sidePanel {
            case .files: FilesPane(doc: doc)
            case .changes: GitChangesPane(doc: doc)
            case .plans: PlansPane(doc: doc)
            case .terminal: TerminalPane(doc: doc)
            case .web:
                if let page = doc.webPage { WebPane(doc: doc, page: page) } else {
                    Color.clear.onAppear { _ = model.webPage(doc) }
                }
            case .flow:
                if let id = doc.flowPanelRunID, let run = model.flowController(id) { FlowPanel(doc: doc, run: run) } else {
                    Color.clear
                }
            case nil: Color.clear
            }
        }
        .frame(maxHeight: .infinity)
        .background(MG.windowBackground)
        .mgTestAnchor("side-panel")
    }
}

/// Each side pane's width, kept as JSON in one setting (`[pane: points]`). Until resized, a pane gets its default
/// (Changes and Flow start from the widths the single panel and the flow panel had).
enum SidePanelWidths {
    static let range: ClosedRange<Double> = 300...1000

    static func defaultWidth(_ pane: SidePanel) -> Double {
        let d = UserDefaults.standard
        switch pane {
        case .files: return 340
        case .changes: return d.object(forKey: "changesWidth") as? Double ?? 440
        case .plans: return 420
        case .web: return 560
        case .terminal: return 520
        case .flow: return d.object(forKey: "flowPanelWidth") as? Double ?? 460
        }
    }

    static func decode(_ json: String) -> [String: Double] {
        (try? JSONDecoder().decode([String: Double].self, from: Data(json.utf8))) ?? [:]
    }

    static func width(_ json: String, _ pane: SidePanel) -> Double {
        min(max(decode(json)[pane.rawValue] ?? defaultWidth(pane), range.lowerBound), range.upperBound)
    }

    /// The pane's width, read from and written to the setting.
    static func binding(_ json: Binding<String>, _ pane: SidePanel) -> Binding<Double> {
        Binding(get: { width(json.wrappedValue, pane) }, set: { w in
            var all = decode(json.wrappedValue)
            all[pane.rawValue] = w.rounded()
            json.wrappedValue = (try? JSONEncoder().encode(all)).flatMap { String(data: $0, encoding: .utf8) } ?? ""
        })
    }
}

/// A pane's header row when the pane has no toolbar of its own to be it: a title (and count), actions on the right.
struct SidePaneHeader<Actions: View>: View {
    let title: String
    var count = 0
    @ViewBuilder var actions: Actions

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Text(title).font(.mgHeadline)
                if count > 0 {
                    Text("\(count)").font(.mgFootnote).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                }
                Spacer(minLength: 4)
                actions
            }
            .buttonStyle(.mg(.icon))
            .padding(.leading, 12).padding(.trailing, 8)
            .frame(height: 38)
            MGDivider()
        }
    }
}

extension SidePaneHeader where Actions == EmptyView {
    init(title: String, count: Int = 0) {
        self.init(title: title, count: count) { EmptyView() }
    }
}

/// A coding chat's right edge: an icon per pane (with its count or a dot), a flow run's item while one is open, and
/// show/hide at the bottom. On screen with the panel hidden too, so every pane is one click away; clicking the pane
/// that's shown hides the panel.
struct SidePanelRail: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    static let width: CGFloat = 40

    var body: some View {
        let shown = model.sidePanelShown(doc)
        VStack(spacing: 2) {
            ForEach(SidePanel.allCases.filter { $0 != .flow }) { tab in
                item(tab, shown: shown)
            }
            if let id = doc.flowPanelRunID {
                MGDivider().frame(width: 20).padding(.vertical, 4)
                item(.flow, shown: shown, run: model.flowController(id)?.run)
            }
            Spacer(minLength: 8)
            Button { model.toggleSidePanel(doc) } label: {
                Image(systemName: "sidebar.right").foregroundStyle(shown ? MG.accent : MG.secondaryLabel)
            }
            .buttonStyle(.mg(.icon))
            .mgHelp(shown ? "Hide the side panel (⌥⌘S)" : "Show the side panel (⌥⌘S)")
            .accessibilityLabel(shown ? "Hide the Side Panel" : "Show the Side Panel")
            .mgTestAnchor("side-panel-toggle")
        }
        .padding(.vertical, 8)
        .frame(width: Self.width)
        .frame(maxHeight: .infinity)
        .background(MG.windowBackground)
        .mgTestAnchor("side-rail")
    }

    private func item(_ tab: SidePanel, shown: Bool, run: FlowRun? = nil) -> some View {
        SidePanelRailItem(tab: tab, badge: badge(tab), flowStatus: run?.status, runName: run?.meta.name,
                          selected: shown && doc.sidePanel == tab) { model.toggleSidePanel(doc, tab) }
    }

    private func badge(_ tab: SidePanel) -> SidePanelRailItem.Badge {
        switch tab {
        case .changes: .count(doc.git?.status.files.count ?? 0)
        case .plans: .count(doc.transcript.plans.count)
        case .terminal: doc.terminals.count >= 2 ? .count(doc.terminals.count) : .none
        case .web: doc.webPage?.loading == true ? .dot : .none
        case .files, .flow: .none
        }
    }
}

private struct SidePanelRailItem: View {
    enum Badge { case none, dot, count(Int) }

    let tab: SidePanel
    let badge: Badge
    var flowStatus: FlowStatus?
    var runName: String?
    let selected: Bool
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        // A button, not a tap gesture: it takes the click that brings the window forward, and the keyboard's press.
        Button(action: action) { face }
            .buttonStyle(.plain)
            .onHover { hover = $0 }
            .mgHelp(help)
            .mgTestAnchor("side-tab:\(tab.rawValue)")
            .accessibilityAddTraits(selected ? .isSelected : [])
            .accessibilityLabel(label)
    }

    private var face: some View {
        ZStack(alignment: .topTrailing) {
            Group {
                if let flowStatus {
                    FlowStatusGlyph(status: flowStatus).font(.system(size: 13))
                } else {
                    Image(systemName: tab.symbol)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(selected ? MG.accent : MG.secondaryLabel)
                }
            }
            .frame(width: 30, height: 30)
            .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(selected ? MG.accent.opacity(0.15) : hover ? MG.hover : .clear))
            badgeView.offset(x: 5, y: -2)
        }
        .frame(width: SidePanelRail.width, height: 34)
        .overlay(alignment: .leading) {
            if selected { Capsule().fill(MG.accent).frame(width: 2, height: 18) }  // the edge facing the panel
        }
        .contentShape(Rectangle())
    }

    @ViewBuilder private var badgeView: some View {
        switch badge {
        case .none: EmptyView()
        case .dot: Circle().fill(MG.accent).frame(width: 6, height: 6).offset(x: -4, y: 4)
        case .count(let n):
            if n > 0 {
                Text(n > 99 ? "99+" : "\(n)")
                    .font(.system(size: 9, weight: .semibold)).monospacedDigit()
                    .foregroundStyle(selected ? MG.accent : MG.secondaryLabel)
                    .padding(.horizontal, 4)
                    .frame(minWidth: 15, minHeight: 14)
                    .background(Capsule().fill(selected ? MG.accent.opacity(0.18) : MG.fill))
                    .background(Capsule().fill(MG.windowBackground))
                    .fixedSize()
            }
        }
    }

    private var help: String {
        if tab == .flow { return "Flow" + (runName.map { ": \($0)" } ?? "") }
        return tab.shortcut.map { "\(tab.title) (\($0))" } ?? tab.title
    }

    private var label: String {
        if case .count(let n) = badge, n > 0 { return "\(tab.title), \(n)" }
        return tab.title
    }
}

// MARK: files

/// The project's files as a tree (what the agent's `list_files` sees: without git-ignored files), a filter, and the
/// selected file's preview below.
struct FilesPane: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument
    /// The row the keyboard moves (a file or a folder).
    @State private var cursor: String?
    @FocusState private var listFocused: Bool
    /// The tree's height while a file is previewed under it (the preview gets the rest).
    @AppStorage("filesTreeHeight") private var treeHeight = 200.0

    var body: some View {
        VStack(spacing: 0) {
            SidePaneHeader(title: "Files") {
                Button { model.refreshProjectFiles(doc) } label: {
                    if doc.filesRefreshing { MGSpinner().controlSize(.mini) } else { Image(systemName: "arrow.clockwise") }
                }
                .mgHelp("List the files again")
                .accessibilityLabel("Refresh")
                Button { doc.expandedFolders = [] } label: { Image(systemName: "rectangle.compress.vertical") }.accessibilityLabel("Collapse All Folders")
                    .mgHelp("Collapse all folders")
            }
            panes
        }
        .task(id: doc.id) { if doc.fileTree == nil { model.refreshProjectFiles(doc) } }
    }

    private var panes: some View {
        // Not a VSplitView: one whose second pane comes and goes keeps the first pane's height, so a preview opened
        // under a tall tree got a sliver.
        GeometryReader { geo in
            let maxTree = max(100, geo.size.height - 200)
            VStack(spacing: 0) {
                tree.frame(height: previewURL == nil ? nil : min(max(treeHeight, 100), maxTree))
                    .frame(maxHeight: previewURL == nil ? .infinity : nil)
                if let url = previewURL, let rel = doc.previewPath {
                    MGRowResizeHandle(height: $treeHeight, range: 100...maxTree)
                    FilePreview(url: url, line: doc.previewLine, revision: doc.filesRevision,
                                onMention: { model.mentionInDraft(doc, rel) },
                                onClose: { doc.previewPath = nil; doc.previewLine = nil })
                        .frame(maxHeight: .infinity)
                }
            }
        }
    }

    private var previewURL: URL? {
        guard let rel = doc.previewPath, let root = doc.projectURL else { return nil }
        return root.appendingPathComponent(rel)
    }

    @ViewBuilder var tree: some View {
        VStack(spacing: 0) {
            MGSearchField(prompt: "Filter files", text: $doc.fileFilter)
                .padding(.horizontal, 10).padding(.vertical, 6)
            list
        }
    }

    @ViewBuilder var list: some View {
        if let tree = doc.fileTree {
            let rows = tree.rows(expanded: doc.expandedFolders, filter: doc.fileFilter)
            let status = GitMarks(doc.git)
            if rows.isEmpty {
                MGEmptyState(symbol: "doc.text.magnifyingglass", title: doc.fileFilter.isEmpty ? "No files" : "No matches")
            } else {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 0) {
                            ForEach(rows) { row in
                                FileTreeRow(doc: doc, row: row, mark: status.mark(row),
                                            selected: row.path == (cursor ?? doc.previewPath)) { click(row) }
                                    .id(row.path)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                    .mgScrollIndicators()
                    .focusable()
                    .focusEffectDisabled()
                    .focused($listFocused)
                    .onKeyPress(.downArrow) { move(rows, 1, proxy); return .handled }
                    .onKeyPress(.upArrow) { move(rows, -1, proxy); return .handled }
                    .onKeyPress(.rightArrow) { fold(rows, open: true); return .handled }
                    .onKeyPress(.leftArrow) { fold(rows, open: false); return .handled }
                    .onKeyPress(.return) {
                        if let r = rows.first(where: { $0.path == cursor }) { click(r) }
                        return .handled
                    }
                    .onAppear {
                        if let p = doc.previewPath { proxy.scrollTo(p, anchor: .center) }
                    }
                    .onChange(of: doc.previewPath) { _, p in
                        cursor = nil
                        if let p { withAnimation { proxy.scrollTo(p, anchor: .center) } }
                    }
                }
            }
        } else if doc.filesRefreshing {
            VStack { MGSpinner() }.frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            MGEmptyState(symbol: "folder", title: "No project folder")
        }
    }

    func click(_ row: FileTree.Row) {
        listFocused = true
        if row.isFolder {
            cursor = row.path
            toggle(row.path)
        } else {
            doc.previewPath = row.path
            doc.previewLine = nil
            cursor = nil
        }
    }

    func toggle(_ path: String) {
        if doc.expandedFolders.contains(path) { doc.expandedFolders.remove(path) } else { doc.expandedFolders.insert(path) }
    }

    func move(_ rows: [FileTree.Row], _ d: Int, _ proxy: ScrollViewProxy) {
        let cur = cursor ?? doc.previewPath
        let i = rows.firstIndex { $0.path == cur }.map { $0 + d } ?? (d > 0 ? 0 : rows.count - 1)
        guard rows.indices.contains(i) else { return }
        let r = rows[i]
        if r.isFolder { cursor = r.path } else {
            doc.previewPath = r.path
            doc.previewLine = nil
            cursor = nil
        }
        proxy.scrollTo(r.path)
    }

    func fold(_ rows: [FileTree.Row], open: Bool) {
        guard let cur = cursor ?? doc.previewPath, let r = rows.first(where: { $0.path == cur }) else { return }
        if r.isFolder {
            if open != r.open { toggle(r.path) } else if !open, let parent = FileTree.parents(of: r.path).last {
                cursor = parent
            }
        } else if !open, let parent = FileTree.parents(of: r.path).last {
            cursor = parent
        }
    }
}

/// Git's marks for the tree: a file's status letter, and which folders hold changes.
struct GitMarks {
    var files: [String: GitChange] = [:]
    var folders: Set<String> = []

    init(_ git: GitState?) {
        for c in git?.status.files ?? [] {
            files[c.path] = c
            folders.formUnion(FileTree.parents(of: c.path))
        }
    }

    func mark(_ row: FileTree.Row) -> (letter: String, tint: Color)? {
        if row.isFolder { return folders.contains(row.path) ? ("•", MG.warning) : nil }
        return files[row.path].map { GitUI.letter($0.kind) }
    }
}

struct FileTreeRow: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let row: FileTree.Row
    let mark: (letter: String, tint: Color)?
    let selected: Bool
    let action: () -> Void

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                .foregroundStyle(MG.tertiaryLabel)
                .rotationEffect(.degrees(row.open ? 90 : 0))
                .frame(width: 10)
                .opacity(row.isFolder ? 1 : 0)
            Image(nsImage: FileIcons.icon(for: row.path, folder: row.isFolder)).resizable().frame(width: 16, height: 16)
            Text(row.name).font(.mgCallout).lineLimit(1).truncationMode(.middle)
            Spacer(minLength: 4)
            if let mark {
                Text(mark.letter).font(.mgMono.weight(.semibold)).foregroundStyle(mark.tint)
            }
        }
        .mgRow(selected: selected, height: 24, indent: CGFloat(row.depth) * 14)
        .onTapGesture(perform: action)
        .mgTestAnchor("file-row:\(row.path)")
        .mgContextMenu {
            if let root = doc.projectURL {
                let url = root.appendingPathComponent(row.path)
                if !row.isFolder {
                    MGMenuItem.button("Mention in Message", icon: "text.badge.plus") { model.mentionInDraft(doc, row.path) }
                    if ["html", "htm"].contains(url.pathExtension.lowercased()) {
                        MGMenuItem.button("Open in Browser Tab", icon: "globe") { model.openWebPage(doc, url) }
                    }
                    MGMenuItem.button("Open in Default App", icon: "arrow.up.forward.app") { NSWorkspace.shared.open(url) }
                }
                MGMenuItem.button("Reveal in Finder", icon: "folder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                MGMenuItem.divider
                MGMenuItem.button("Copy Relative Path") { copy(row.path) }
                MGMenuItem.button("Copy Full Path") { copy(url.path) }
            }
        }
    }

    func copy(_ s: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
    }
}
