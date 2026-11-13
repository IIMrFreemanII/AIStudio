import AppKit
import GraphRuntime
import ImageIO
import AIStudioLibrary
import MLXDiffusionCore
import SwiftUI
import UniformTypeIdentifiers

/// The panels the icon rail switches between.
enum SidebarPanel: String, CaseIterable, Identifiable {
    case workflows, chats, presets, assets, queue, nodes, servers
    var id: String { rawValue }

    var title: String {
        switch self {
        case .workflows: return "Workflows"
        case .chats: return "Sessions"
        case .presets: return "Presets"
        case .assets: return "Assets"
        case .queue: return "Queue"
        case .nodes: return "Node Library"
        case .servers: return "Servers"
        }
    }

    /// The rail's caption under the icon (short enough for its width).
    var caption: String { self == .nodes ? "Nodes" : title }

    var symbol: String {
        switch self {
        case .workflows: return "folder"
        case .chats: return "bubble.left.and.bubble.right"
        case .presets: return "square.grid.2x2"
        case .assets: return "photo.stack"
        case .queue: return "list.bullet.rectangle"
        case .nodes: return "point.3.connected.trianglepath.dotted"
        case .servers: return "server.rack"
        }
    }
}

/// The collapsed sidebar: a narrow column of panel icons; clicking one opens the sidebar on that panel.
struct SidebarRail: View {
    @Environment(AppModel.self) private var model
    @Binding var selection: SidebarPanel
    @Binding var open: Bool

    var body: some View {
        VStack(spacing: 4) {
            ForEach(SidebarPanel.allCases) { p in
                Button {
                    if open && selection == p { open = false } else { selection = p; open = true }
                } label: {
                    // Icon with a small caption, so the rail reads without hovering.
                    VStack(spacing: 2) {
                        Image(systemName: p.symbol)
                            .font(.system(size: 15))
                            .frame(height: 20)
                        Text(p.caption).font(.mg(size: 9)).lineLimit(1).fixedSize()
                    }
                        .frame(width: 54, height: 42)
                        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                            .fill(open && selection == p ? MG.selection : .clear))
                        .foregroundStyle(open && selection == p ? MG.accent : MG.secondaryLabel)
                        .contentShape(Rectangle())
                        .overlay(alignment: .topTrailing) {
                            if p == .queue, model.pendingCount + (model.isRunning ? 1 : 0) > 0 {
                                Text("\(model.pendingCount + (model.isRunning ? 1 : 0))")
                                    .font(.mg(size: 9, weight: .bold)).foregroundStyle(MG.accentForeground)
                                    .padding(.horizontal, 4).background(Capsule().fill(MG.warning))
                                    .offset(x: -8, y: 1)
                            }
                        }
                }
                .buttonStyle(.plain)
                .mgHelp(p.title)
            }
            Spacer()
        }
        .padding(.vertical, 10)
        .frame(width: 60)
        .mgChrome(MG.sidebarTint)
        .overlay(alignment: .trailing) { MGDivider(vertical: true) }
    }
}

/// The sidebar, as in Claude: the panels' icons along the top (the shown one highlighted) and the panel under them.
/// The title bar's sidebar button (⌃⌘S) collapses it to the icon rail.
struct AppSidebar: View {
    @Environment(AppModel.self) private var model
    @Binding var selection: SidebarPanel
    @Binding var open: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 2) {
                ForEach(SidebarPanel.allCases) { p in
                    SidebarIconTab(panel: p, selected: p == selection,
                                   badge: p == .queue ? model.pendingCount + (model.isRunning ? 1 : 0) : 0) {
                        selection = p
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .frame(height: 40)
            MGDivider()
            SidebarPanelView(panel: selection)
        }
        .mgTestAnchor("app-sidebar")
    }
}

/// One panel's icon in the sidebar's top row.
private struct SidebarIconTab: View {
    let panel: SidebarPanel
    let selected: Bool
    let badge: Int
    let action: () -> Void
    @State private var hover = false

    var body: some View {
        Button(action: action) {
            Image(systemName: panel.symbol)
                .font(.system(size: 13.5, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? MG.accent : hover ? MG.label : MG.secondaryLabel)
                .frame(width: 30, height: 28)
                .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                    .fill(selected ? MG.selection : hover ? MG.hover : .clear))
                .overlay(alignment: .topTrailing) {
                    if badge > 0 {
                        Text("\(badge)")
                            .font(.mg(size: 8.5, weight: .bold)).foregroundStyle(MG.accentForeground)
                            .padding(.horizontal, 3.5).frame(minHeight: 12)
                            .background(Capsule().fill(MG.warning))
                            .offset(x: 3, y: -2)
                    }
                }
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { hover = $0 }
        .mgHelp(panel.title)
        .accessibilityLabel(panel.title)
        .accessibilityAddTraits(selected ? [.isButton, .isSelected] : .isButton)
        .mgTestAnchor("sidebar-tab:\(panel.rawValue)")
    }
}

struct SidebarPanelView: View {
    let panel: SidebarPanel

    var body: some View {
        switch panel {
        case .workflows: WorkflowsPanel()
        case .chats: SessionsPanel()
        case .presets: PresetsPanel()
        case .assets: AssetsPanel()
        case .queue: QueuePanel()
        case .nodes: NodeLibraryPanel()
        case .servers: ServersPanel()
        }
    }
}

struct PanelHeader<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 8) {
            Text(title).font(.mgHeadline)
            Spacer()
            trailing
        }
        .buttonStyle(.mg(.icon))
        .padding(.leading, 16)
        .padding(.trailing, 10)
        .frame(height: 40)
    }
}

// MARK: - workflows

struct WorkflowsPanel: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    @State private var expanded = Set<URL>()

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Workflows") {
                Button { model.newWorkflowFolder(in: nil) } label: { Image(systemName: "folder.badge.plus") }
                    .mgHelp("New folder")
                Button { NSWorkspace.shared.open(model.library.workflows) } label: { Image(systemName: "arrow.up.forward.app") }
                    .mgHelp("Open the workflows folder in Finder")
                Button { model.reloadWorkflows() } label: { Image(systemName: "arrow.clockwise") }
                    .mgHelp("Refresh")
            }
            MGSearchField(prompt: "Search workflows", text: $query)
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    OpenWorkflowsSection()
                    let q = query.trimmingCharacters(in: .whitespaces).lowercased()
                    if q.isEmpty {
                        folderContents(model.workflowTree, depth: 0)
                    } else {
                        ForEach(model.workflowTree.allWorkflows.filter { $0.name.lowercased().contains(q) }) { w in
                            row(w, depth: 0)
                        }
                    }
                    if model.workflowTree.allWorkflows.isEmpty {
                        Text("Saved workflows appear here. Press ⌘S to save the current workflow.")
                            .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                            .padding(.horizontal, 16).padding(.vertical, 6)
                    }
                }
                .padding(.bottom, 8)
            }
            .mgScrollIndicators()
            .dropDestination(for: URL.self) { urls, _ in
                for u in urls where u.pathExtension == "json" { model.moveWorkflow(u, to: model.library.workflows) }
                return true
            }
        }
        .onAppear { model.reloadWorkflows() }
    }

    func rename(_ url: URL, name: String) {
        model.presenter.present(id: "rename") {
            MGPromptSheet(title: "Rename “\(name)”", actionTitle: "Rename", text: name) { model.renameWorkflow(url, to: $0) }
        }
    }

    func folderContents(_ folder: WorkflowFolder, depth: Int) -> AnyView {
        AnyView(Group {
            ForEach(folder.folders) { sub in
                let isOpen = expanded.contains(sub.url)
                HStack(spacing: 4) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9, weight: .semibold)).foregroundStyle(MG.secondaryLabel)
                        .rotationEffect(.degrees(isOpen ? 90 : 0))
                        .frame(width: 12)
                    Image(systemName: isOpen ? "folder.fill" : "folder").foregroundStyle(MG.hueFolder)
                    Text(sub.name).lineLimit(1)
                    Spacer()
                }
                .mgRow(indent: CGFloat(depth) * 14)
                .onTapGesture {
                    withAnimation(.easeOut(duration: 0.18)) {
                        if isOpen { expanded.remove(sub.url) } else { expanded.insert(sub.url) }
                    }
                }
                .mgContextMenu { folderMenu(sub) }
                .dropDestination(for: URL.self) { urls, _ in
                    for u in urls { model.moveWorkflow(u, to: sub.url) }
                    return true
                }
                if isOpen { folderContents(sub, depth: depth + 1) }
            }
            ForEach(folder.workflows) { w in row(w, depth: depth) }
        })
    }

    func row(_ w: WorkflowEntry, depth: Int) -> some View {
        let open = model.tabs.first { $0.fileURL == w.url }
        let active = open?.id == model.activeTabID
        return HStack(spacing: 6) {
            Image(systemName: "doc.text").foregroundStyle(active ? MG.accent : MG.secondaryLabel)
            Text(w.name).lineLimit(1)
            Spacer()
            if let open {
                Circle().fill(open.id == model.activeTabID ? MG.accent : MG.tertiaryLabel)
                    .frame(width: 6, height: 6).mgHelp("Open")
            }
        }
        .mgRow(selected: active, indent: CGFloat(depth) * 14 + 16)
        .mgTestAnchor("workflow-row:\(w.name)")
        .onTapGesture { model.open(w.url) }
        .draggable(w.url)
        .mgContextMenu {
            MGMenuItem.button("Open") { model.open(w.url) }
            MGMenuItem.button("Rename…") { rename(w.url, name: w.name) }
            MGMenuItem.button("Duplicate") { model.duplicateWorkflow(w.url) }
            MGMenuItem.menu("Move to", icon: nil) {
                MGMenuItem.button("Workflows", icon: "folder") { model.moveWorkflow(w.url, to: model.library.workflows) }
                for f in allFolders(model.workflowTree) {
                    MGMenuItem.button(f.path.replacingOccurrences(of: model.library.workflows.path + "/", with: ""),
                                      icon: "folder") { model.moveWorkflow(w.url, to: f) }
                }
            }
            MGMenuItem.divider
            MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([w.url]) }
            MGMenuItem.button("Move to Trash", destructive: true) { model.trashWorkflow(w.url) }
        }
    }

    func folderMenu(_ f: WorkflowFolder) -> [MGMenuItem] {
        [
            .button("New Folder Inside") { model.newWorkflowFolder(in: f.url) },
            .button("Rename…") { rename(f.url, name: f.name) },
            .button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([f.url]) },
            .divider,
            .button("Move to Trash", destructive: true) { model.trashWorkflow(f.url) },
        ]
    }

    func allFolders(_ f: WorkflowFolder) -> [URL] {
        f.folders.flatMap { [$0.url] + allFolders($0) }
    }
}

// MARK: - presets

struct PresetsPanel: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        let installed = model.installedModelIDs
        VStack(spacing: 0) {
            PanelHeader(title: "Presets") {
                Button("Gallery…") { model.showPresetGallery = true }
                    .buttonStyle(.mg(.borderless)).controlSize(.small)
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                ForEach(Preset.Category.allCases) { cat in
                    let items = model.presets.filter { $0.category == cat }
                    if !items.isEmpty {
                        MGListHeader(title: cat.title)
                            ForEach(items) { p in
                                let ok = model.presetAvailable(p, installed: installed)
                                HStack(spacing: 8) {
                                    MGKindBadge(color: ok ? cat.hue : MG.hueGray, symbol: p.symbol, size: 20)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text(p.name).lineLimit(1)
                                        Text(p.modelName).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                                    }
                                    Spacer()
                                    if !ok {
                                        if p.needsServer == "gpu", model.connections.gpuServers.isEmpty {
                                            Button("Add") { model.showServersPanel = true }
                                                .controlSize(.mini).mgHelp("Add a GPU server (Servers panel)")
                                        } else {
                                            Button("Get") { openWindow(id: "models") }
                                                .controlSize(.mini).mgHelp("Get \(p.modelName) in the Models window")
                                        }
                                    }
                                }
                                .mgRow(height: 38)
                                .opacity(ok ? 1 : 0.55)
                                .onTapGesture { if ok { model.openPreset(p) } else { NSSound.beep() } }
                                .mgHelp(p.summary)
                                .mgContextMenu {
                                    MGMenuItem.button("Open", enabled: ok) { model.openPreset(p) }
                                    if !p.isBuiltIn {
                                        MGMenuItem.divider
                                        MGMenuItem.button("Delete Preset", destructive: true) { model.deletePreset(p) }
                                    }
                                }
                            }
                    }
                }
                }
                .padding(.bottom, 8)
            }
            .mgScrollIndicators()
            MGDivider()
            Button("Save Current Workflow as Preset…") { model.showSavePresetSheet = true }
                .buttonStyle(.mg(.borderless)).padding(10)
        }
    }
}

// MARK: - node library

struct NodeLibraryPanel: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""

    var body: some View {
        let q = query.lowercased()
        let matches = model.schemas.filter {
            q.isEmpty || $0.displayName.lowercased().contains(q) || $0.type.lowercased().contains(q)
                || $0.category.lowercased().contains(q)
        }
        let categories = Array(Set(matches.map(\.category))).sorted()
        VStack(spacing: 0) {
            PanelHeader(title: "Node Library") { EmptyView() }
            MGSearchField(prompt: "Search nodes", text: $query)
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(categories, id: \.self) { cat in
                        MGListHeader(title: cat.capitalized)
                        ForEach(matches.filter { $0.category == cat }, id: \.type) { s in
                            HStack(spacing: 8) {
                                MGKindBadge(color: s.tint, letter: String(s.displayName.prefix(1)), size: 16)
                                Text(s.displayName).lineLimit(1)
                                Spacer()
                            }
                            .mgRow()
                            .onTapGesture { model.nodeAddRequest = NodeAddRequest(type: s.type) }
                            .draggable(NodeAddRequest.dragPrefix + s.type)
                            .mgHelp("Click to add at the centre of the graph, or drag onto it")
                        }
                    }
                }
                .padding(.bottom, 8)
            }
            .mgScrollIndicators()
        }
    }
}

/// A node the library asked the editor to add (at the view centre, or where it was dropped).
struct NodeAddRequest: Equatable {
    static let dragPrefix = "aistudio-node:"
    let id = UUID()
    let type: String
}

// MARK: - assets

/// Assets: Input (files to use in workflows, copied into the library's input folder) and Output (everything
/// generated).
struct AssetsPanel: View {
    @Environment(AppModel.self) private var model
    @AppStorage("assetsTab") private var tab = "input"

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Assets") {
                if tab == "input" {
                    Button { importPanel() } label: { Image(systemName: "square.and.arrow.down") }
                        .mgHelp("Import files into the input folder")
                    Button { NSWorkspace.shared.open(model.library.input) } label: { Image(systemName: "arrow.up.forward.app") }
                        .mgHelp("Open the input folder in Finder")
                    Button { model.reloadAssets() } label: { Image(systemName: "arrow.clockwise") }.mgHelp("Refresh")
                } else {
                    Button { NSWorkspace.shared.open(URL(fileURLWithPath: (model.outputDirectory as NSString).expandingTildeInPath)) } label: {
                        Image(systemName: "arrow.up.forward.app")
                    }
                    .mgHelp("Open the output folder in Finder")
                    Button { model.reloadHistory() } label: { Image(systemName: "arrow.clockwise") }.mgHelp("Refresh")
                }
            }
            MGSegmentedPicker(selection: $tab, options: [("input", "Input"), ("output", "Output")])
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            if tab == "input" { InputAssetsView() } else { OutputList() }
        }
        .onAppear { model.reloadAssets() }
    }

    func importPanel() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.image, .movie, .audio]
        panel.message = "Choose images, videos or sounds to copy into the input folder."
        if panel.runModal() == .OK { model.importAssets(panel.urls) }
    }
}

struct InputAssetsView: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    @State private var filter: AssetKind? = nil
    @State private var dropTargeted = false

    var body: some View {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        let items = model.assets.filter { (filter == nil || $0.kind == filter) && (q.isEmpty || $0.relativePath.lowercased().contains(q)) }
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                MGSearchField(prompt: "Search files", text: $query)
                MGPicker(selection: $filter, options: [(AssetKind?.none, "All"), (AssetKind?.some(.image), "Images"),
                                                      (AssetKind?.some(.mask), "Masks"), (AssetKind?.some(.video), "Videos"),
                                                      (AssetKind?.some(.audio), "Sounds")])
                    .mgTestAnchor("assets-filter")
            }
            .padding(.horizontal, 12)
            ScrollView {
                if model.assets.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "tray.and.arrow.down").font(.largeTitle).foregroundStyle(.tertiary)
                        Text("Drop images, videos or sounds here, or click Import. They are copied into \(model.library.input.path), and Load nodes refer to them by name.")
                            .font(.mgCallout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(24)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 82, maximum: 130), spacing: 8)], spacing: 8) {
                    ForEach(items) { item in
                        VStack(spacing: 3) {
                            AssetThumb(url: item.url, kind: item.kind)
                                .aspectRatio(1, contentMode: .fit)
                                .overlay(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).strokeBorder(
                                    model.displayedURL == item.url ? MG.accent : MG.separator,
                                    lineWidth: model.displayedURL == item.url ? 2 : 0.5))
                            Text(item.relativePath).font(.mgFootnote).lineLimit(1).truncationMode(.middle)
                                .foregroundStyle(MG.secondaryLabel)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) { model.showFile(item.url, kind: item.kind); model.openViewer() }
                        .onTapGesture { model.showFile(item.url, kind: item.kind) }
                        .draggable(item.url)
                        .mgHelp("\(item.relativePath) — drag onto the graph for a Load node, or onto a Load node to use it there")
                        .mgContextMenu { menu(item) }
                    }
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 10)
            }
            .mgScrollIndicators()
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 10).strokeBorder(MG.accent, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
                        .padding(6).allowsHitTesting(false)
                }
            }
            .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
                for p in providers {
                    _ = p.loadObject(ofClass: URL.self) { url, _ in
                        guard let url else { return }
                        Task { @MainActor in model.importAssets([url]) }
                    }
                }
                return !providers.isEmpty
            }
        }
    }

    func menu(_ item: AssetItem) -> [MGMenuItem] {
        var items: [MGMenuItem] = [
            .button("Add to Workflow as \(item.kind.loadNodeType.replacingOccurrences(of: "Load", with: "Load "))") {
                model.assetAddRequest = item.url
            },
        ]
        if item.kind == .image && model.canAdd(.reference) {
            items.append(.button("Use as Reference Image") {
                model.setReferenceImage(item.url, undoManager: model.activeUndoManager)
            })
        }
        if item.kind == .image && model.canAdd(.firstFrame) {
            items.append(.button("Use as First Frame") { model.setFirstFrame(item.url, undoManager: model.activeUndoManager) })
        }
        let name = item.url.deletingPathExtension().lastPathComponent
        items += [
            .divider,
            .button("Rename…") {
                model.presenter.present(id: "rename") {
                    MGPromptSheet(title: "Rename “\(name)”", actionTitle: "Rename", text: name) { model.renameAsset(item.url, to: $0) }
                }
            },
            .button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) },
            .button("Move to Trash", destructive: true) { model.trashAsset(item.url) },
        ]
        return items
    }
}

/// An input asset's thumbnail: the image, a video's first frame, a waveform, or a mask on dark.
struct AssetThumb: View {
    let url: URL
    let kind: AssetKind
    @State private var image: CGImage?

    var body: some View {
        Rectangle().fill(kind == .mask ? AnyShapeStyle(Color.black) : AnyShapeStyle(MG.fill))
            .overlay {
                if let image {
                    Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: kind == .mask ? .fit : .fill)
                } else if kind == .audio {
                    AudioThumb(url: url)
                }
            }
            .overlay(alignment: .bottomLeading) {
                if kind == .video {
                    Image(systemName: "play.fill").font(.system(size: 8)).foregroundStyle(.white)
                        .padding(3).background(.black.opacity(0.55), in: Circle()).padding(4)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            .task(id: url) {
                switch kind {
                case .video: image = await MediaIO.posterFrame(url, maxSize: 240)
                case .audio: break
                case .image, .mask:
                    let u = url
                    image = await Task.detached(priority: .utility) { () -> CGImage? in
                        guard let src = CGImageSourceCreateWithURL(u as CFURL, nil) else { return nil }
                        return CGImageSourceCreateThumbnailAtIndex(src, 0, [
                            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 240,
                        ] as CFDictionary)
                    }.value
                }
            }
    }
}
