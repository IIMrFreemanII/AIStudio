import AppKit
import Foundation
import GraphRuntime
import AIStudioLibrary
import MLXDiffusionCore
import NodesStandard

/// Tabs, the workflow library, presets and the restored session.
extension AppModel {
    // MARK: launch

    /// Smoke tests, snapshots and preset export run against a throwaway library and never restore or save tabs.
    static var isAutomatedRun: Bool {
        CommandLine.arguments.contains { $0.hasPrefix("--smoke") || $0 == "--snapshot" || $0 == "--export-presets" }
    }

    /// `--library-root <dir>`, a temporary library for automated runs, else the user's.
    static func launchLibrary() -> LibraryPaths {
        if let root = SmokeTest.argument("--library-root") {
            return LibraryPaths(root: URL(fileURLWithPath: (root as NSString).expandingTildeInPath))
        }
        if isAutomatedRun {
            return LibraryPaths(root: FileManager.default.temporaryDirectory
                .appendingPathComponent("AIStudio-library-\(ProcessInfo.processInfo.processIdentifier)"))
        }
        if let custom = UserDefaults.standard.string(forKey: "libraryRoot") {
            return LibraryPaths(root: URL(fileURLWithPath: (custom as NSString).expandingTildeInPath),
                                sessionDirectory: LibraryPaths.defaultSessionDirectory)
        }
        return LibraryPaths()
    }

    /// Automated runs delete outright instead of filling the user's Trash.
    var workflowStore: WorkflowStore { WorkflowStore(root: library.workflows, useTrash: !Self.isAutomatedRun) }

    func reloadWorkflows() { workflowTree = workflowStore.tree() }

    // MARK: tabs

    func activate(_ id: UUID) {
        guard tabs.contains(where: { $0.id == id }) else { return }
        activeChatID = nil
        activeTabID = id
    }

    /// A tab name not used by another open tab: "Untitled", "Untitled 2"…
    func uniqueTabName(_ base: String) -> String {
        let taken = Set(tabs.map(\.name))
        var name = base
        var n = 2
        while taken.contains(name) { name = "\(base) \(n)"; n += 1 }
        return name
    }

    @discardableResult
    func openTab(_ graph: Graph, name: String, fileURL: URL? = nil, presetID: String? = nil) -> WorkflowDocument {
        activeChatID = nil
        let doc = WorkflowDocument(name: fileURL == nil ? uniqueTabName(name) : name, graph: graph, fileURL: fileURL,
                                   presetID: presetID)
        if let i = tabs.firstIndex(where: { $0.id == activeTabID }) { tabs.insert(doc, at: i + 1) } else { tabs.append(doc) }
        activeTabID = doc.id
        return doc
    }

    /// ⌘T: a new tab from the default preset (Qwen-Image text to image, the graph the app always started with).
    @discardableResult
    func newTab() -> WorkflowDocument {
        openTab(GraphTemplates.textToImage(modelPath: modelPath, outputDirectory: outputDirectory), name: "Untitled")
    }

    /// Closes a tab without asking (the caller confirms unsaved changes). The last tab is replaced by a new one.
    func closeTab(_ id: UUID) {
        guard let i = tabs.firstIndex(where: { $0.id == id }) else { return }
        let wasActive = id == activeTabID
        tabs.remove(at: i)
        if tabs.isEmpty { newTab(); return }
        if wasActive { activeTabID = tabs[min(i, tabs.count - 1)].id }
    }

    func moveTab(_ id: UUID, before target: UUID?) {
        guard let from = tabs.firstIndex(where: { $0.id == id }) else { return }
        let doc = tabs.remove(at: from)
        if let target, let to = tabs.firstIndex(where: { $0.id == target }) { tabs.insert(doc, at: to) } else { tabs.append(doc) }
    }

    func selectTab(offset: Int) {
        guard let i = tabs.firstIndex(where: { $0.id == activeTabID }), tabs.count > 1 else { return }
        activeTabID = tabs[(i + offset + tabs.count) % tabs.count].id
    }

    // MARK: open & save

    /// Opens a workflow JSON, or the graph embedded in a PNG / video / audio file made by the app, in a new tab
    /// (or switches to the tab that already has that file).
    @discardableResult
    func open(_ url: URL, undoManager: UndoManager? = nil) -> Bool {
        let ext = url.pathExtension.lowercased()
        let name = url.deletingPathExtension().lastPathComponent
        if MediaIO.videoExtensions.contains(ext) || MediaIO.audioExtensions.contains(ext) {
            // Media metadata loads asynchronously; the tab opens when it arrives.
            Task {
                guard let json = StandardNodes.embeddedGraph(await MediaIO.readMetadata(url)),
                      let g = try? JSONDecoder().decode(Graph.self, from: Data(json.utf8)) else {
                    lastError = "\(url.lastPathComponent) has no embedded graph"
                    NSSound.beep()
                    return
                }
                openTab(Self.restorable(g), name: name)
            }
            return true
        }
        if ext == "png" {
            guard let g = Self.embeddedGraph(in: url) else {
                lastError = "\(url.lastPathComponent) has no embedded graph"
                NSSound.beep()
                return false
            }
            openTab(Self.restorable(g), name: name)
            return true
        }
        let std = url.standardizedFileURL
        if let open = tabs.first(where: { $0.fileURL?.standardizedFileURL == std }) {
            activeTabID = open.id
            return true
        }
        do {
            let g = try Graph.load(url)
            openTab(g, name: name, fileURL: std)
            return true
        } catch {
            lastError = "Could not open \(url.lastPathComponent): \(error)"
            NSSound.beep()
            return false
        }
    }

    /// A graph restored from an output: the metadata written at queue time (expanded templates, workflow name) is
    /// dropped and the prompt templates put back, so it edits like the graph that made it.
    static func restorable(_ g: Graph) -> Graph {
        var g = g
        guard var meta = g.metadata else { return g }
        for (key, raw) in meta where key.hasPrefix("prompt_template.") {
            let node = String(key.dropFirst("prompt_template.".count))
            let field = g[node].flatMap { promptFields[$0.type] } ?? "prompt"
            try? g.setValue(.string(raw), node: node, input: field)
            meta[key] = nil
        }
        for key in ["workflow", "prompt_style"] { meta[key] = nil }
        g.metadata = meta.isEmpty ? nil : meta
        return g
    }

    /// Writes the active tab to its file. Returns false when it has none yet (the caller asks for a name).
    @discardableResult
    func saveActive() -> Bool {
        guard let url = active.fileURL else { return false }
        return save(active, to: url)
    }

    @discardableResult
    func save(_ doc: WorkflowDocument, to url: URL) -> Bool {
        do {
            try workflowStore.write(doc.graph, to: url)
            doc.markSaved(to: url.standardizedFileURL)
            reloadWorkflows()
            return true
        } catch {
            lastError = "Could not save: \(error)"
            return false
        }
    }

    /// Saves under `name` in the workflows folder (made unique), e.g. the first ⌘S of an untitled tab.
    @discardableResult
    func saveAs(_ doc: WorkflowDocument, name: String, in folder: URL? = nil) -> Bool {
        let url = workflowStore.uniqueURL(name: name, in: folder ?? library.workflows,
                                          excluding: doc.fileURL)
        return save(doc, to: url)
    }

    // MARK: library actions

    func renameWorkflow(_ url: URL, to name: String) {
        do {
            let new = try workflowStore.rename(url, to: name)
            retarget(from: url, to: new)
            reloadWorkflows()
        } catch { lastError = "Could not rename: \(error)" }
    }

    func duplicateWorkflow(_ url: URL) {
        do { try workflowStore.duplicate(url); reloadWorkflows() } catch { lastError = "Could not duplicate: \(error)" }
    }

    func moveWorkflow(_ url: URL, to folder: URL) {
        do {
            let new = try workflowStore.move(url, to: folder)
            retarget(from: url, to: new)
            reloadWorkflows()
        } catch { lastError = "Could not move: \(error)" }
    }

    func newWorkflowFolder(in parent: URL?) {
        do { try workflowStore.createFolder(named: "New Folder", in: parent); reloadWorkflows() }
        catch { lastError = "Could not create the folder: \(error)" }
    }

    func trashWorkflow(_ url: URL) {
        do {
            try workflowStore.delete(url)
            // Open tabs keep their graph but are no longer saved anywhere.
            for doc in tabs where doc.fileURL.map({ $0 == url.standardizedFileURL || $0.path.hasPrefix(url.standardizedFileURL.path + "/") }) == true {
                doc.fileURL = nil
                doc.cleanGraph = Graph()
            }
            reloadWorkflows()
        } catch { lastError = "Could not move to the Trash: \(error)" }
    }

    /// Open tabs follow a renamed or moved workflow (or folder).
    private func retarget(from old: URL, to new: URL) {
        let o = old.standardizedFileURL.path, n = new.standardizedFileURL.path
        for doc in tabs {
            guard let p = doc.fileURL?.standardizedFileURL.path else { continue }
            if p == o { doc.fileURL = new.standardizedFileURL; doc.name = new.deletingPathExtension().lastPathComponent }
            else if p.hasPrefix(o + "/") { doc.fileURL = URL(fileURLWithPath: n + p.dropFirst(o.count)) }
        }
    }

    // MARK: presets

    func reloadPresets() {
        presets = PresetCatalog.builtIn() + PresetCatalog.user(in: library.presets)
    }

    /// Catalog ids of the models installed now.
    var installedModelIDs: Set<String> {
        guard let store else { return [] }
        var ids = Set(ModelCatalog.entries.filter { store.installedURL($0.id) != nil }.map(\.id))
        // The Qwen folder set in Settings counts even when it isn't the catalog download.
        if FileManager.default.fileExists(atPath: modelPath + "/model_index.json") { ids.insert("qwen-image-2.1-turbo") }
        return ids
    }

    /// Whether a preset can run: its models are on this Mac, and the server it needs (a GPU server) is set up.
    func presetAvailable(_ p: Preset, installed: Set<String>) -> Bool {
        p.isAvailable(installed: installed) && (p.needsServer != "gpu" || !connections.gpuServers.isEmpty)
    }

    /// What a preset that can't run yet is missing, for its button.
    func presetMissing(_ p: Preset) -> (label: String, action: () -> Void) {
        if p.needsServer == "gpu", connections.gpuServers.isEmpty {
            return ("Add GPU Server…", { [weak self] in self?.showServersPanel = true })
        }
        return ("Get Model…", {})
    }

    /// A model folder name in a preset → the installed folder on this Mac (nil: leave the value).
    func resolveModelFolder(_ name: String) -> String? {
        guard let entry = ModelCatalog.entries.first(where: { $0.folder == name || $0.id == name }) else { return nil }
        if entry.id == "qwen-image-2.1-turbo" { return store?.installedURL(entry.id)?.path ?? modelPath }
        if let u = store?.installedURL(entry.id) { return u.path }
        return (store?.modelsDirectory ?? ModelPaths.directory).appendingPathComponent(entry.folder).path
    }

    /// The reverse, for saving a workflow as a preset.
    func modelFolderName(_ path: String) -> String? {
        let last = URL(fileURLWithPath: path).lastPathComponent
        return ModelCatalog.entries.first { $0.folder == last }?.folder
    }

    /// Opens a preset in a new tab.
    @discardableResult
    func openPreset(_ preset: Preset) -> WorkflowDocument {
        let g = preset.instantiate(resolveModel: resolveModelFolder, outputDirectory: outputDirectory)
        return openTab(g, name: preset.name, presetID: preset.id)
    }

    func preset(id: String) -> Preset? { presets.first { $0.id == id } }

    /// Saves the active workflow as one of your presets (with the latest result as its thumbnail).
    func saveAsPreset(name: String, category: Preset.Category, summary: String) {
        let doc = active
        let g = Preset.portable(doc.graph, folderName: modelFolderName, outputDirectory: outputDirectory)
        let models = doc.graph.nodes.filter { $0.type == "LoadModel" }.compactMap { n -> String? in
            guard case .value(.string(let p))? = n.inputs["path"], let f = modelFolderName(p) else { return nil }
            return ModelCatalog.entries.first { $0.folder == f }?.id
        }
        let modelName = models.first.flatMap { id in ModelCatalog.entries.first { $0.id == id }?.name } ?? "Custom"
        let p = Preset(id: "", name: name, category: category, models: models, modelName: modelName, summary: summary,
                       symbol: category.symbol, graph: g)
        var png: Data?
        if let img = doc.outputs.last(where: { $0.kind != .audio })?.image, let thumb = Self.thumbnail(img, maxSize: 360) {
            png = NSBitmapImageRep(cgImage: thumb).representation(using: .png, properties: [:])
        }
        do {
            try PresetCatalog.save(p, thumbnailPNG: png, in: library.presets)
            reloadPresets()
        } catch { lastError = "Could not save the preset: \(error)" }
    }

    func deletePreset(_ p: Preset) {
        do { try PresetCatalog.delete(p); reloadPresets() } catch { lastError = "Could not delete the preset: \(error)" }
    }

    static func thumbnail(_ image: CGImage, maxSize: Int) -> CGImage? {
        let scale = min(1, Double(maxSize) / Double(max(image.width, image.height)))
        let w = max(1, Int(Double(image.width) * scale)), h = max(1, Int(Double(image.height) * scale))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return ctx.makeImage()
    }

    // MARK: output folders

    /// The output folder for a workflow's results: `<output folder>/<workflow name>`.
    func outputFolder(forWorkflow name: String) -> String {
        (outputDirectory as NSString).appendingPathComponent(WorkflowStore.sanitize(name))
    }

    // MARK: session

    var sessionStore: SessionStore { SessionStore(directory: library.sessionDirectory) }

    func sessionManifest() -> SessionManifest {
        SessionManifest(tabs: tabs.map { .init(id: $0.id, name: $0.name, fileURL: $0.fileURL, presetID: $0.presetID,
                                               isDirty: $0.isDirty) },
                        activeTab: activeTabID)
    }

    /// Reopens the tabs of the last session from their drafts (or files).
    func restoreTabs() {
        let (manifest, drafts) = sessionStore.load()
        var restored: [WorkflowDocument] = []
        for t in manifest.tabs {
            let fileGraph = t.fileURL.flatMap { try? Graph.load($0) }
            guard let graph = (t.isDirty ? drafts[t.id] : nil) ?? fileGraph ?? drafts[t.id] else { continue }
            // Dirty: compare against the file as saved; untitled and dirty: against an empty graph, so the dot stays.
            let clean = t.isDirty ? (fileGraph ?? Graph()) : graph
            let doc = WorkflowDocument(id: t.id, name: t.name, graph: graph,
                                       fileURL: fileGraph == nil ? nil : t.fileURL, presetID: t.presetID,
                                       cleanGraph: clean)
            doc.lastDraft = drafts[t.id]
            restored.append(doc)
        }
        guard !restored.isEmpty else { return }
        tabs = restored
        activeTabID = restored.first { $0.id == manifest.activeTab }?.id ?? restored[0].id
        lastManifest = manifest
    }

    /// Writes changed drafts and the tab list; called every couple of seconds and on quit.
    func saveSession() {
        guard persistSettings else { return }
        saveOpenChats()
        let store = sessionStore
        for doc in tabs where doc.lastDraft != doc.graph {
            do { try store.writeDraft(doc.graph, id: doc.id); doc.lastDraft = doc.graph } catch {}
        }
        let manifest = sessionManifest()
        if manifest != lastManifest {
            try? store.save(manifest)
            store.prune(keeping: Set(tabs.map(\.id)))
            lastManifest = manifest
        }
    }

    func startSessionAutosave() {
        sessionTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.saveSession() }
        }
    }
}

/// Which image the mask editor paints over.
struct MaskEditorTarget: Identifiable {
    let id = UUID()
    let imageNode: String
    let imagePath: String
}

// MARK: - results & viewer

extension AppModel {
    /// What the results pane shows now, as a result.
    var displayedResult: NodeResult? {
        let doc = active
        if let url = doc.displayedMediaURL, doc.displayedMediaKind != .image {
            return NodeResult(node: "", kind: doc.displayedMediaKind, image: doc.displayedImage, url: url)
        }
        guard doc.displayedImage != nil || doc.displayedURL != nil else { return nil }
        return doc.outputs.last { $0.url != nil && $0.url == doc.displayedURL }
            ?? NodeResult(node: "", kind: .image, image: doc.displayedImage, url: doc.displayedURL)
    }

    func openViewer(_ r: NodeResult? = nil) {
        viewerResult = r ?? displayedResult
    }

    /// The results the viewer steps through: the tab's own when it shows one of them, else the history.
    var viewerItems: [NodeResult] {
        let doc = active
        if let v = viewerResult, doc.outputs.contains(where: { $0.id == v.id || ($0.url != nil && $0.url == v.url) }) {
            return doc.outputs
        }
        return history.map { NodeResult(node: "", kind: $0.kind, image: nil, url: $0.url) }
    }

    /// Shows one of a tab's results in the results pane.
    func show(_ r: NodeResult, in doc: WorkflowDocument) {
        doc.displayedImage = r.image
        doc.displayedURL = r.url
        doc.displayedMediaURL = r.kind == .image ? nil : r.url
        doc.displayedMediaKind = r.kind
        doc.compareImage = nil
        doc.compareURL = nil
        showResultsPane = true
    }
}
