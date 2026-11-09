import AppKit
import GraphRuntime
import ImageIO
import MLXComfyLibrary
import MLXDiffusionCore
import SwiftUI
import UniformTypeIdentifiers

/// The panels the icon rail switches between.
enum SidebarPanel: String, CaseIterable, Identifiable {
    case workflows, presets, assets, queue, nodes
    var id: String { rawValue }

    var title: String {
        switch self {
        case .workflows: return "Workflows"
        case .presets: return "Presets"
        case .assets: return "Assets"
        case .queue: return "Queue"
        case .nodes: return "Node Library"
        }
    }

    var symbol: String {
        switch self {
        case .workflows: return "folder"
        case .presets: return "square.grid.2x2"
        case .assets: return "photo.stack"
        case .queue: return "list.bullet.rectangle"
        case .nodes: return "point.3.connected.trianglepath.dotted"
        }
    }
}

/// The narrow column of panel icons; clicking the open panel's icon hides the panel.
struct SidebarRail: View {
    @Environment(AppModel.self) private var model
    @Binding var selection: SidebarPanel
    @Binding var open: Bool

    var body: some View {
        VStack(spacing: 6) {
            ForEach(SidebarPanel.allCases) { p in
                Button {
                    if open && selection == p { open = false } else { selection = p; open = true }
                } label: {
                    Image(systemName: p.symbol)
                        .font(.system(size: 16))
                        .frame(width: 34, height: 34)
                        .background(RoundedRectangle(cornerRadius: 7)
                            .fill(open && selection == p ? Color.accentColor.opacity(0.18) : .clear))
                        .foregroundStyle(open && selection == p ? Color.accentColor : .secondary)
                        .overlay(alignment: .topTrailing) {
                            if p == .queue, model.pendingCount + (model.isRunning ? 1 : 0) > 0 {
                                Text("\(model.pendingCount + (model.isRunning ? 1 : 0))")
                                    .font(.system(size: 9, weight: .bold)).foregroundStyle(.white)
                                    .padding(.horizontal, 4).background(Capsule().fill(Color.accentColor))
                            }
                        }
                }
                .buttonStyle(.plain)
                .help(p.title)
            }
            Spacer()
        }
        .padding(.vertical, 10)
        .frame(width: 48)
        .background(.bar)
    }
}

struct SidebarPanelView: View {
    let panel: SidebarPanel

    var body: some View {
        switch panel {
        case .workflows: WorkflowsPanel()
        case .presets: PresetsPanel()
        case .assets: AssetsPanel()
        case .queue: QueuePanel()
        case .nodes: NodeLibraryPanel()
        }
    }
}

struct PanelHeader<Trailing: View>: View {
    let title: String
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(spacing: 8) {
            Text(title).font(.headline)
            Spacer()
            trailing
        }
        .buttonStyle(.borderless)
        .padding(.horizontal, 12)
        .frame(height: 38)
    }
}

// MARK: - workflows

struct WorkflowsPanel: View {
    @Environment(AppModel.self) private var model
    @State private var query = ""
    @State private var renaming: URL?
    @State private var renameText = ""

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Workflows") {
                Button { model.newWorkflowFolder(in: nil) } label: { Image(systemName: "folder.badge.plus") }
                    .help("New folder")
                Button { NSWorkspace.shared.open(model.library.workflows) } label: { Image(systemName: "arrow.up.forward.app") }
                    .help("Open the workflows folder in Finder")
                Button { model.reloadWorkflows() } label: { Image(systemName: "arrow.clockwise") }
                    .help("Refresh")
            }
            TextField("Search workflows", text: $query)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal, 10)
                .padding(.bottom, 6)
            List {
                let q = query.trimmingCharacters(in: .whitespaces).lowercased()
                if q.isEmpty {
                    folderContents(model.workflowTree)
                } else {
                    ForEach(model.workflowTree.allWorkflows.filter { $0.name.lowercased().contains(q) }) { w in
                        row(w)
                    }
                }
                if model.workflowTree.allWorkflows.isEmpty {
                    Text("Saved workflows appear here. Press ⌘S to save the current tab.")
                        .font(.callout).foregroundStyle(.secondary)
                }
            }
            .listStyle(.sidebar)
            .dropDestination(for: URL.self) { urls, _ in
                for u in urls where u.pathExtension == "json" { model.moveWorkflow(u, to: model.library.workflows) }
                return true
            }
        }
        .onAppear { model.reloadWorkflows() }
        .alert("Rename", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $renameText)
            Button("Rename") {
                if let url = renaming { model.renameWorkflow(url, to: renameText) }
                renaming = nil
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
    }

    func folderContents(_ folder: WorkflowFolder) -> AnyView {
        AnyView(Group {
            ForEach(folder.folders) { sub in
                DisclosureGroup {
                    folderContents(sub)
                } label: {
                    Label(sub.name, systemImage: "folder")
                        .contextMenu { folderMenu(sub) }
                        .dropDestination(for: URL.self) { urls, _ in
                            for u in urls { model.moveWorkflow(u, to: sub.url) }
                            return true
                        }
                }
            }
            ForEach(folder.workflows) { w in row(w) }
        })
    }

    func row(_ w: WorkflowEntry) -> some View {
        let open = model.tabs.first { $0.fileURL == w.url }
        return HStack(spacing: 6) {
            Image(systemName: "doc.text").foregroundStyle(.secondary)
            Text(w.name).lineLimit(1)
            Spacer()
            if let open {
                Circle().fill(open.id == model.activeTabID ? Color.accentColor : Color.secondary.opacity(0.6))
                    .frame(width: 6, height: 6).help("Open in a tab")
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { model.open(w.url) }
        .draggable(w.url)
        .contextMenu {
            Button("Open") { model.open(w.url) }
            Button("Rename…") { renameText = w.name; renaming = w.url }
            Button("Duplicate") { model.duplicateWorkflow(w.url) }
            Menu("Move to") {
                Button("Workflows") { model.moveWorkflow(w.url, to: model.library.workflows) }
                ForEach(allFolders(model.workflowTree), id: \.self) { f in
                    Button(f.path.replacingOccurrences(of: model.library.workflows.path + "/", with: "")) {
                        model.moveWorkflow(w.url, to: f)
                    }
                }
            }
            Divider()
            Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([w.url]) }
            Button("Move to Trash", role: .destructive) { model.trashWorkflow(w.url) }
        }
    }

    @ViewBuilder func folderMenu(_ f: WorkflowFolder) -> some View {
        Button("New Folder Inside") { model.newWorkflowFolder(in: f.url) }
        Button("Rename…") { renameText = f.name; renaming = f.url }
        Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([f.url]) }
        Divider()
        Button("Move to Trash", role: .destructive) { model.trashWorkflow(f.url) }
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
                Button("Gallery…") { model.showPresetGallery = true }.controlSize(.small)
            }
            List {
                ForEach(Preset.Category.allCases) { cat in
                    let items = model.presets.filter { $0.category == cat }
                    if !items.isEmpty {
                        Section(cat.title) {
                            ForEach(items) { p in
                                let ok = p.isAvailable(installed: installed)
                                HStack(spacing: 8) {
                                    Image(systemName: p.symbol).frame(width: 18).foregroundStyle(ok ? Color.accentColor : .secondary)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text(p.name).lineLimit(1)
                                        Text(p.modelName).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                                    }
                                    Spacer()
                                    if !ok {
                                        Button("Get") { openWindow(id: "models") }
                                            .controlSize(.mini).help("Get \(p.modelName) in the Models window")
                                    }
                                }
                                .opacity(ok ? 1 : 0.55)
                                .contentShape(Rectangle())
                                .onTapGesture { if ok { model.openPreset(p) } else { NSSound.beep() } }
                                .help(p.summary)
                                .contextMenu {
                                    Button("Open in New Tab") { model.openPreset(p) }.disabled(!ok)
                                    if !p.isBuiltIn {
                                        Divider()
                                        Button("Delete Preset", role: .destructive) { model.deletePreset(p) }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            Divider()
            Button("Save Current Workflow as Preset…") { model.showSavePresetSheet = true }
                .buttonStyle(.borderless).padding(10)
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
            TextField("Search nodes", text: $query)
                .textFieldStyle(.roundedBorder)
                .padding(.horizontal, 10)
                .padding(.bottom, 6)
            List {
                ForEach(categories, id: \.self) { cat in
                    Section(cat.capitalized) {
                        ForEach(matches.filter { $0.category == cat }, id: \.type) { s in
                            HStack {
                                Circle().fill(s.tint).frame(width: 8, height: 8)
                                Text(s.displayName)
                                Spacer()
                            }
                            .contentShape(Rectangle())
                            .onTapGesture { model.nodeAddRequest = NodeAddRequest(type: s.type) }
                            .draggable(NodeAddRequest.dragPrefix + s.type)
                            .help("Click to add at the centre of the graph, or drag onto it")
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }
}

/// A node the library asked the editor to add (at the view centre, or where it was dropped).
struct NodeAddRequest: Equatable {
    static let dragPrefix = "mlxcomfy-node:"
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
                        .help("Import files into the input folder")
                    Button { NSWorkspace.shared.open(model.library.input) } label: { Image(systemName: "arrow.up.forward.app") }
                        .help("Open the input folder in Finder")
                    Button { model.reloadAssets() } label: { Image(systemName: "arrow.clockwise") }.help("Refresh")
                } else {
                    Button { NSWorkspace.shared.open(URL(fileURLWithPath: (model.outputDirectory as NSString).expandingTildeInPath)) } label: {
                        Image(systemName: "arrow.up.forward.app")
                    }
                    .help("Open the output folder in Finder")
                    Button { model.reloadHistory() } label: { Image(systemName: "arrow.clockwise") }.help("Refresh")
                }
            }
            Picker("", selection: $tab) {
                Text("Input").tag("input")
                Text("Output").tag("output")
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 10)
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
    @State private var renaming: URL?
    @State private var renameText = ""

    var body: some View {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        let items = model.assets.filter { (filter == nil || $0.kind == filter) && (q.isEmpty || $0.relativePath.lowercased().contains(q)) }
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                TextField("Search files", text: $query).textFieldStyle(.roundedBorder)
                Picker("", selection: $filter) {
                    Text("All").tag(AssetKind?.none)
                    Text("Images").tag(AssetKind?.some(.image))
                    Text("Masks").tag(AssetKind?.some(.mask))
                    Text("Videos").tag(AssetKind?.some(.video))
                    Text("Sounds").tag(AssetKind?.some(.audio))
                }
                .labelsHidden()
                .frame(width: 92)
            }
            .padding(.horizontal, 10)
            ScrollView {
                if model.assets.isEmpty {
                    VStack(spacing: 8) {
                        Image(systemName: "tray.and.arrow.down").font(.largeTitle).foregroundStyle(.tertiary)
                        Text("Drop images, videos or sounds here, or click Import. They are copied into \(model.library.input.path), and Load nodes refer to them by name.")
                            .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                    }
                    .padding(24)
                }
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 82, maximum: 130), spacing: 8)], spacing: 8) {
                    ForEach(items) { item in
                        VStack(spacing: 3) {
                            AssetThumb(url: item.url, kind: item.kind)
                                .aspectRatio(1, contentMode: .fit)
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(
                                    model.displayedURL == item.url ? Color.accentColor : .clear, lineWidth: 2))
                            Text(item.relativePath).font(.caption2).lineLimit(1).truncationMode(.middle)
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) { model.showFile(item.url, kind: item.kind); model.openViewer() }
                        .onTapGesture { model.showFile(item.url, kind: item.kind) }
                        .draggable(item.url)
                        .help("\(item.relativePath) — drag onto the graph for a Load node, or onto a Load node to use it there")
                        .contextMenu { menu(item) }
                    }
                }
                .padding(.horizontal, 10)
                .padding(.bottom, 10)
            }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 10).strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 2, dash: [6, 4]))
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
        .alert("Rename", isPresented: Binding(get: { renaming != nil }, set: { if !$0 { renaming = nil } })) {
            TextField("Name", text: $renameText)
            Button("Rename") {
                if let url = renaming { model.renameAsset(url, to: renameText) }
                renaming = nil
            }
            Button("Cancel", role: .cancel) { renaming = nil }
        }
    }

    @ViewBuilder func menu(_ item: AssetItem) -> some View {
        Button("Add to Workflow as \(item.kind.loadNodeType.replacingOccurrences(of: "Load", with: "Load "))") {
            model.assetAddRequest = item.url
        }
        if item.kind == .image && model.canAdd(.reference) {
            Button("Use as Reference Image") { model.setReferenceImage(item.url, undoManager: model.activeUndoManager) }
        }
        if item.kind == .image && model.canAdd(.firstFrame) {
            Button("Use as First Frame") { model.setFirstFrame(item.url, undoManager: model.activeUndoManager) }
        }
        Divider()
        Button("Rename…") { renameText = item.url.deletingPathExtension().lastPathComponent; renaming = item.url }
        Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
        Button("Move to Trash", role: .destructive) { model.trashAsset(item.url) }
    }
}

/// An input asset's thumbnail: the image, a video's first frame, a waveform, or a mask on dark.
struct AssetThumb: View {
    let url: URL
    let kind: AssetKind
    @State private var image: CGImage?

    var body: some View {
        Rectangle().fill(kind == .mask ? AnyShapeStyle(Color.black) : AnyShapeStyle(.quaternary))
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
            .clipShape(RoundedRectangle(cornerRadius: 6))
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
