import AIStudioLibrary
import Agent
import AppKit
import LLMClient
import SwiftUI

/// A coding chat's right-hand panel: the project's files, the git changes, the agent's plans; a tab each.
struct CodingSidePanel: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument

    var body: some View {
        let counts: [SidePanel: Int] = [.changes: doc.git?.status.files.count ?? 0, .plans: doc.transcript.plans.count]
        VStack(spacing: 0) {
            HStack(spacing: 0) {
                SidePanelTabs(selection: doc.sidePanel ?? doc.sideTab, counts: counts) { model.showSidePanel(doc, $0) }
                Spacer(minLength: 4)
                Button { model.showSidePanel(doc, nil) } label: { Image(systemName: "sidebar.right") }.accessibilityLabel("Hide the Side Panel")
                    .buttonStyle(.mg(.icon))
                    .mgHelp("Hide the side panel (⌥⌘S)")
                    .padding(.bottom, 4)
            }
            .padding(.horizontal, 8)
            .frame(height: 38, alignment: .bottom)
            MGDivider()
            switch doc.sidePanel {
            case .files: FilesPane(doc: doc)
            case .changes: GitChangesPane(doc: doc)
            case .plans: PlansPane(doc: doc)
            case .web:
                if let page = doc.webPage { WebPane(doc: doc, page: page) } else {
                    Color.clear.onAppear { _ = model.webPage(doc) }
                }
            case nil: Color.clear
            }
        }
        .background(MG.windowBackground)
        .mgTestAnchor("side-panel")
    }
}

/// The side panel's tabs: an icon, a name and a count each; the shown one in full colour over an accent underline
/// that slides between tabs.
struct SidePanelTabs: View {
    let selection: SidePanel
    let counts: [SidePanel: Int]
    let select: (SidePanel) -> Void
    @Namespace private var underline

    var body: some View {
        ViewThatFits(in: .horizontal) {
            tabs(titles: true)
            tabs(titles: false)
        }
        .animation(.spring(response: 0.28, dampingFraction: 0.86), value: selection)
    }

    func tabs(titles: Bool) -> some View {
        HStack(alignment: .bottom, spacing: 2) {
            ForEach(SidePanel.allCases) { tab in
                SidePanelTab(tab: tab, count: counts[tab] ?? 0, selected: tab == selection, titles: titles,
                             underline: underline) { select(tab) }
            }
        }
    }
}

private struct SidePanelTab: View {
    let tab: SidePanel
    let count: Int
    let selected: Bool
    let titles: Bool
    let underline: Namespace.ID
    let action: () -> Void
    @State private var hover = false

    var shortcut: String {
        switch tab {
        case .files: "⇧⌘E"
        case .changes: "⇧⌘D"
        case .plans: "⇧⌘P"
        case .web: "⇧⌘B"
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: tab.symbol)
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(selected ? MG.accent : MG.secondaryLabel)
                if titles {
                    Text(tab.title)
                        .font(.mgCallout.weight(selected ? .semibold : .medium))
                        .foregroundStyle(selected ? MG.label : MG.secondaryLabel)
                }
                if count > 0 {
                    Text("\(count)")
                        .font(.system(size: 10, weight: .semibold)).monospacedDigit()
                        .foregroundStyle(selected ? MG.accent : MG.secondaryLabel)
                        .padding(.horizontal, 5)
                        .frame(minWidth: 17, minHeight: 16)
                        .background(Capsule().fill(selected ? MG.accent.opacity(0.15) : MG.fill))
                }
            }
            .lineLimit(1)
            .fixedSize()
            .padding(.horizontal, 9)
            .frame(height: 26)
            .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(hover && !selected ? MG.hover : .clear))
            .padding(.bottom, 4)
            ZStack {
                Rectangle().fill(.clear)
                if selected {
                    Capsule().fill(MG.accent).matchedGeometryEffect(id: "underline", in: underline)
                }
            }
            .frame(height: 2)
            .padding(.horizontal, 6)
        }
        .contentShape(Rectangle())
        .onTapGesture(perform: action)
        .onHover { hover = $0 }
        .mgHelp("\(tab.title) (\(shortcut))")
        .mgTestAnchor("side-tab:\(tab.rawValue)")
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
        .accessibilityLabel(count > 0 ? "\(tab.title), \(count)" : tab.title)
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
        .task(id: doc.id) { if doc.fileTree == nil { model.refreshProjectFiles(doc) } }
    }

    private var previewURL: URL? {
        guard let rel = doc.previewPath, let root = doc.projectURL else { return nil }
        return root.appendingPathComponent(rel)
    }

    @ViewBuilder var tree: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                MGSearchField(prompt: "Filter files", text: $doc.fileFilter)
                Button { model.refreshProjectFiles(doc) } label: {
                    if doc.filesRefreshing { MGSpinner().controlSize(.mini) } else { Image(systemName: "arrow.clockwise") }
                }
                .buttonStyle(.mg(.icon))
                .mgHelp("List the files again")
                Button { doc.expandedFolders = [] } label: { Image(systemName: "rectangle.compress.vertical") }.accessibilityLabel("Collapse All Folders")
                    .buttonStyle(.mg(.icon))
                    .mgHelp("Collapse all folders")
            }
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
