import AppKit
import Foundation
import GraphRuntime
import ImageIO
import MLXDiffusionCore
import MLXComfyLibrary
import NodesStandard
import Observation

enum NodeRunState: Equatable {
    case pending
    case running(step: Int, total: Int)
    case cached
    case done(seconds: Double, peakMemory: Int?)
    case failed(String)
}

struct QueueItem: Identifiable, Equatable {
    enum Status: Equatable {
        case pending
        case running(step: Int, total: Int)
        case done(seconds: Double)
        case failed(String)
        case cancelled
    }

    let id = UUID()
    let graph: Graph
    let title: String
    let seed: Int?
    var status: Status = .pending
    /// Set for the cells of an XY plot: which plot, and the cell index (row-major).
    var groupID: UUID? = nil
    var cell: Int? = nil
    /// The tab that queued it (its results and node states go there), and that tab's name at the time.
    var docID: UUID? = nil
    var workflow: String = ""

    static func == (a: QueueItem, b: QueueItem) -> Bool { a.id == b.id && a.status == b.status }

    var isFinished: Bool {
        switch status {
        case .done, .failed, .cancelled: return true
        default: return false
        }
    }
}

struct HistoryItem: Identifiable, Hashable {
    let url: URL
    let date: Date
    var kind: MediaKind = .image
    /// The first Text Encode prompt from the embedded graph (filled in the background).
    var prompt: String?
    var isFavorite = false
    /// The workflow subfolder it was saved in (nil: the output folder itself).
    var workflow: String? = nil
    var id: URL { url }
}

@MainActor @Observable
final class AppModel {
    // MARK: tabs

    /// The open workflows, in tab order; there is always at least one.
    var tabs: [WorkflowDocument]
    var activeTabID: UUID
    var active: WorkflowDocument { tabs.first { $0.id == activeTabID } ?? tabs[0] }
    var activeUndoManager: UndoManager { active.undoManager }

    // The active tab's document state, so editing code reads like there is one graph.
    var graph: Graph {
        get { active.graph }
        set { active.graph = newValue }
    }
    var selection: Set<String> {
        get { active.selection }
        set { active.selection = newValue }
    }
    var editingPath: [String] {
        get { active.editingPath }
        set { active.editingPath = newValue }
    }
    var selectedGroup: String? {
        get { active.selectedGroup }
        set { active.selectedGroup = newValue }
    }
    var documentURL: URL? { active.fileURL }
    /// Bumped on every undoable change, so the Undo / Redo menu items refresh.
    var undoRevision = 0

    // MARK: run state

    var queue: [QueueItem] = []
    /// The tab whose job is running.
    var runningDocID: UUID?
    var nodeStates: [String: NodeRunState] {
        get { active.nodeStates }
        set { active.nodeStates = newValue }
    }
    var previewImage: CGImage? {
        get { active.previewImage }
        set { active.previewImage = newValue }
    }
    var displayedImage: CGImage? {
        get { active.displayedImage }
        set { active.displayedImage = newValue }
    }
    var displayedURL: URL? {
        get { active.displayedURL }
        set { active.displayedURL = newValue }
    }
    /// A video or audio result shown in the results pane (nil for images).
    var displayedMedia: (url: URL, kind: MediaKind)? {
        get { active.displayedMediaURL.map { ($0, active.displayedMediaKind) } }
        set { active.displayedMediaURL = newValue?.url; active.displayedMediaKind = newValue?.kind ?? .image }
    }
    var displayedMediaURL: URL? { active.displayedMediaURL }
    /// Models held in memory (refreshed with the memory gauge).
    var loadedModels: [LoadedModel] = []
    /// Share of RAM models may occupy before the least recently used is unloaded.
    var memoryBudget: Double = 0.8 {
        didSet {
            if persistSettings { UserDefaults.standard.set(memoryBudget, forKey: "memoryBudget") }
            let f = memoryBudget
            Task { await engine.setBudget(fraction: f) }
        }
    }
    var lastError: String?
    var history: [HistoryItem] = []
    var randomizeSeed = true
    var batchCount = 1
    var showPalette = false
    var paletteLocation: GraphPoint?

    // MARK: engine state

    var activeMemory: Int = 0
    var modelStatus: String = "Model not loaded"
    var modelLoading = false

    // MARK: graph editor requests

    /// A wire dropped on empty canvas, waiting for the palette to pick a node to connect.
    var pendingWire: PendingWire?

    // MARK: prompt assistant

    /// The assistant task in progress (Enhance / Describe), if any.
    var assistantMode: PromptAssistant.Mode?
    /// The Text Encode node it writes into.
    var assistantNode: String?
    @ObservationIgnored var assistantTask: Task<Void, Never>?
    @ObservationIgnored var assistantRun: UUID?

    // MARK: batch & compare

    /// Prompt styles (built-in ones until edited); each Text Encode picks one (stored in the graph's metadata).
    var styles: [PromptStyle] = PromptStyle.builtIn {
        didSet { if persistSettings, let d = try? JSONEncoder().encode(styles) { UserDefaults.standard.set(d, forKey: "promptStyles") } }
    }
    /// XY plots in progress, by group ID.
    @ObservationIgnored var plots: [UUID: PlotJob] = [:]
    /// A/B compare: the image shown against the displayed one, and how.
    var compareURL: URL? {
        get { active.compareURL }
        set { active.compareURL = newValue }
    }
    var compareImage: CGImage? {
        get { active.compareImage }
        set { active.compareImage = newValue }
    }
    var compareMode: CompareMode = .split
    /// Prompts read from history PNGs, so reloads don't re-read files.
    @ObservationIgnored var promptCache: [URL: String] = [:]

    @ObservationIgnored let engine = Engine()
    @ObservationIgnored let registry = StandardNodes.makeRegistry()
    @ObservationIgnored private var runTask: Task<Void, Never>?
    @ObservationIgnored private var memoryTimer: Timer?
    @ObservationIgnored var sessionTimer: Timer?
    @ObservationIgnored var lastManifest: SessionManifest?
    /// The model downloads/installs, for resolving presets' model folders (set by the window).
    @ObservationIgnored weak var store: ModelStore?

    // MARK: library

    /// Where workflows, presets and assets live.
    let library: LibraryPaths
    var workflowTree: WorkflowFolder
    var presets: [Preset] = []
    /// The files in the input folder (Assets › Input).
    var assets: [AssetItem] = []
    /// The preset gallery sheet.
    var showPresetGallery = false
    /// The results pane under the graph, and the full-window viewer.
    var showResultsPane = true
    var viewerResult: NodeResult?
    /// The Node Library asked the editor to add a node; Assets asked for a Load node for a file.
    var nodeAddRequest: NodeAddRequest?
    var assetAddRequest: URL?
    var showSavePresetSheet = false
    /// The Load Image node whose mask is being painted.
    var maskEditorTarget: MaskEditorTarget?
    /// Live previews received during the current/last run (for the smoke test).
    @ObservationIgnored var previewCount = 0

    // MARK: settings

    var modelPath: String {
        didSet { if persistSettings { UserDefaults.standard.set(modelPath, forKey: "modelPath") } }
    }
    var outputDirectory: String {
        didSet {
            if persistSettings { UserDefaults.standard.set(outputDirectory, forKey: "outputDirectory") }
            reloadHistory()
        }
    }
    /// False for offscreen snapshots: no settings writes, no model loading.
    @ObservationIgnored let persistSettings: Bool
    @ObservationIgnored let autoloadModel: Bool

    /// `library` nil: the user's library (or a throwaway one when the app runs a smoke test or snapshot);
    /// `restoreSession` false starts with one new tab.
    init(persistSettings: Bool = true, autoloadModel: Bool = true, library: LibraryPaths? = nil,
         restoreSession: Bool? = nil) {
        self.persistSettings = persistSettings
        self.autoloadModel = autoloadModel
        let defaults = UserDefaults.standard
        let path = defaults.string(forKey: "modelPath") ?? AppModel.guessModelPath()
        let out = defaults.string(forKey: "outputDirectory")
            ?? (NSHomeDirectory() as NSString).appendingPathComponent("Pictures/MLXComfyUI")
        modelPath = path
        outputDirectory = out
        let lib = library ?? Self.launchLibrary()
        self.library = lib
        lib.ensure()
        InputFolder.url = lib.input
        workflowTree = WorkflowStore(root: lib.workflows).tree()
        let first = WorkflowDocument(name: "Untitled", graph: GraphTemplates.textToImage(modelPath: path, outputDirectory: out))
        tabs = [first]
        activeTabID = first.id
        if persistSettings {
            if let d = defaults.data(forKey: "promptStyles"), let s = try? JSONDecoder().decode([PromptStyle].self, from: d) {
                styles = s
            }
            if defaults.object(forKey: "memoryBudget") != nil { memoryBudget = defaults.double(forKey: "memoryBudget") }
        }
        reloadPresets()
        reloadAssets()
        if restoreSession ?? (persistSettings && !Self.isAutomatedRun) { restoreTabs() }
        if persistSettings { startSessionAutosave() }
        reloadHistory()
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in
                guard let self else { return }
                self.activeMemory = Engine.activeMemory()
                let models = self.engine.loadedModelsSnapshot()
                if models != self.loadedModels { self.loadedModels = models }
            }
        }
    }

    /// The Qwen-Image folder: `qwen-turbo-mlx` in the models folder, else next to the working directory or in the
    /// project checkout (older setups).
    static func guessModelPath() -> String {
        let candidates = [
            (UserDefaults.standard.string(forKey: "modelsDirectory").map { URL(fileURLWithPath: $0) } ?? ModelPaths.directory)
                .appendingPathComponent("qwen-turbo-mlx").path,
            FileManager.default.currentDirectoryPath + "/qwen-turbo-mlx",
            (NSHomeDirectory() as NSString).appendingPathComponent("ai/MLXComfyUI/qwen-turbo-mlx"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0 + "/model_index.json") } ?? candidates[0]
    }

    // MARK: schemas

    func schema(for node: GraphNode) -> NodeSchema? {
        if node.type == Graph.subgraphType { return graph.subgraphSchema(node) }
        return registry.schema(node.type)
    }
    var schemas: [NodeSchema] { registry.schemas }

    // MARK: undoable edits

    /// Replaces a tab's graph (the active one by default), registering the inverse with `undoManager`.
    func setGraph(_ new: Graph, in doc: WorkflowDocument? = nil, actionName: String, undoManager: UndoManager?) {
        let doc = doc ?? active
        let old = doc.graph
        guard old != new else { return }
        doc.graph = new
        undoRevision &+= 1
        if doc === active {
            if !editingPath.isEmpty, levelGraph(editingPath) == nil { editingPath = []; selection = [] }
            selection = selection.filter { id in editorGraph.nodes.contains { $0.id == id } }
            if let g = selectedGroup, !(editorGraph.groups ?? []).contains(where: { $0.id == g }) { selectedGroup = nil }
        } else {
            doc.editingPath = []
            doc.selection = []
            doc.selectedGroup = nil
        }
        undoManager?.registerUndo(withTarget: doc) { [weak self] target in
            MainActor.assumeIsolated {
                self?.setGraph(old, in: target, actionName: actionName, undoManager: undoManager)
            }
        }
        undoManager?.setActionName(actionName)
    }

    func mutate(_ actionName: String, undoManager: UndoManager?, _ body: (inout Graph) -> Void) {
        var g = graph
        body(&g)
        setGraph(g, actionName: actionName, undoManager: undoManager)
    }

    /// Records an edit whose live updates already happened (e.g. a drag) as one undo step.
    func commitLiveEdit(from start: Graph, in doc: WorkflowDocument? = nil, actionName: String, undoManager: UndoManager?) {
        let doc = doc ?? active
        let end = doc.graph
        doc.graph = start
        setGraph(end, in: doc, actionName: actionName, undoManager: undoManager)
    }

    // MARK: node helpers

    func firstNode(ofType type: String) -> GraphNode? { graph.nodes.first { $0.type == type } }

    func literal(_ nodeID: String, _ input: String) -> JSONValue? {
        guard let node = graph[nodeID] else { return nil }
        if case .value(let v)? = node.inputs[input] { return v }
        return schema(for: node)?.input(input)?.defaultValue
    }

    func setLiteral(_ value: JSONValue, node: String, input: String, undoManager: UndoManager?) {
        guard literal(node, input) != value else { return }
        mutate("Change \(input)", undoManager: undoManager) { g in try? g.setValue(value, node: node, input: input) }
    }

    func addNode(type: String, at point: GraphPoint, undoManager: UndoManager?) {
        guard let schema = registry.schema(type) else { return }
        var n = 1
        let base = type.lowercased()
        while editorGraph[n == 1 ? base : "\(base)\(n)"] != nil { n += 1 }
        let id = n == 1 ? base : "\(base)\(n)"
        var inputs: [String: NodeInput] = [:]
        for spec in schema.inputs where spec.type.isScalar {
            if let d = spec.defaultValue { inputs[spec.name] = .value(d) }
        }
        if type == "LoadModel" { inputs["path"] = .value(.string(modelPath)) }
        if type == "SaveImage" { inputs["directory"] = .value(.string(outputDirectory)) }
        // Auto-wire a MODEL input to the first model loader, the common case.
        if schema.inputs.contains(where: { $0.type == .model }),
           let m = editorGraph.nodes.first(where: { $0.type == "LoadModel" }) {
            inputs["model"] = .link(node: m.id, output: "model")
        }
        mutateEditor("Add \(schema.displayName)", undoManager: undoManager) { g in
            g.nodes.append(GraphNode(id: id, type: type, inputs: inputs, position: point))
        }
        selection = [id]
    }

    func deleteSelection(undoManager: UndoManager?) {
        let ids = selection
        guard !ids.isEmpty else { return }
        mutateEditor(ids.count == 1 ? "Delete Node" : "Delete Nodes", undoManager: undoManager) { g in
            g.nodes.removeAll { ids.contains($0.id) }
            for i in g.nodes.indices {
                for (name, input) in g.nodes[i].inputs {
                    if case .link(let src, _) = input, ids.contains(src) { g.nodes[i].inputs[name] = nil }
                }
            }
        }
        selection = []
    }

    func duplicateSelection(undoManager: UndoManager?) {
        guard let clip = clipboardForSelection() else { return }
        // Duplicates keep their inputs from nodes outside the selection.
        insert(NodeClipboard(nodes: clip.nodes, subgraphs: clip.subgraphs, origin: clip.origin),
               at: GraphPoint(x: clip.origin.x + 40, y: clip.origin.y + 40), keepExternalLinks: true,
               actionName: "Duplicate", undoManager: undoManager)
    }

    /// Validation error for the current graph, or nil when it can run.
    var validationError: String? {
        do { _ = try GraphExecutor(registry: registry).validate(graph); return nil }
        catch { return String(describing: error) }
    }

    func wouldCreateCycle(_ g: Graph) -> Bool {
        (try? GraphExecutor(registry: registry).topologicalOrder(g)) == nil
    }

    // MARK: queue

    var isRunning: Bool { queue.contains { if case .running = $0.status { return true } else { return false } } }
    var pendingCount: Int { queue.filter { $0.status == .pending }.count }

    func enqueue(undoManager: UndoManager?) {
        if let err = validationError { lastError = err; NSSound.beep(); return }
        lastError = nil
        let samplers = graph.nodes.filter { ["Sampler", "VideoSampler", "TextToAudio", "TextToMusic", "TextToSpeech"].contains($0.type) }.map(\.id)
        for i in 0..<max(1, batchCount) {
            var g = graph
            var seed: Int?
            if let s = samplers.first {
                let current = { if case .int(let v)? = literal(s, "seed") { return v } else { return 0 } }()
                let newSeed = randomizeSeed ? Int.random(in: 0...Int(Int32.max)) : current + i
                for id in samplers { try? g.setValue(.int(newSeed), node: id, input: "seed") }
                seed = newSeed
                if randomizeSeed || i > 0 {
                    // Reflect the seed actually used so the panel shows what produced the last image.
                    for id in samplers { try? graph.setValue(.int(newSeed), node: id, input: "seed") }
                }
            }
            g = prepared(g, seed: seed ?? i, workflow: active.name)
            queue.append(QueueItem(graph: g, title: title(for: g), seed: seed, docID: active.id, workflow: active.name))
        }
        processQueue()
    }

    /// Speaks a short sample in `voice` through the queue; the clip is shown and played, not saved to history.
    func previewVoice(_ voice: String, speed: Double, modelPath: String) {
        let g = GraphTemplates.voicePreview(modelPath: modelPath, voice: voice, speed: speed)
        let name = SpeechVoices.all.first { $0.id == voice }?.label ?? voice
        queue.append(QueueItem(graph: g, title: "Voice preview: \(name)", seed: nil, docID: active.id, workflow: active.name))
        processQueue()
    }

    func title(for g: Graph) -> String {
        if let p = g.nodes.first(where: { $0.type == "TextEncode" || $0.type == "TextToAudio" }),
           case .value(.string(let s))? = p.inputs["prompt"],
           !s.isEmpty { return s }
        if let p = g.nodes.first(where: { $0.type == "TextToMusic" }),
           case .value(.string(let s))? = p.inputs["tags"],
           !s.isEmpty { return s }
        if let p = g.nodes.first(where: { $0.type == "TextToSpeech" }),
           case .value(.string(let s))? = p.inputs["text"],
           !s.isEmpty { return s }
        return "Untitled graph"
    }

    /// The tab a queue item belongs to (the active one if that tab was closed).
    func document(for item: QueueItem) -> WorkflowDocument {
        tabs.first { $0.id == item.docID } ?? active
    }

    func document(id: UUID?) -> WorkflowDocument? { tabs.first { $0.id == id } }

    func processQueue() {
        guard runTask == nil, let idx = queue.firstIndex(where: { $0.status == .pending }) else { return }
        let item = queue[idx]
        queue[idx].status = .running(step: 0, total: 0)
        let doc = document(for: item)
        runningDocID = doc.id
        doc.nodeStates = Dictionary(uniqueKeysWithValues: item.graph.nodes.map { ($0.id, NodeRunState.pending) })
        doc.previewImage = nil
        doc.previewNode = nil
        runTask = Task { [weak self] in
            guard let self else { return }
            do {
                let start = Date()
                let outputs = try await engine.run(item.graph) { event in
                    Task { @MainActor in self.handle(event, item: item.id) }
                }
                self.finish(item.id, status: .done(seconds: Date().timeIntervalSince(start)), outputs: outputs)
            } catch is CancellationError {
                self.finish(item.id, status: .cancelled, outputs: [])
            } catch {
                self.lastError = String(describing: error)
                self.finish(item.id, status: .failed(String(describing: error)), outputs: [])
            }
        }
    }

    private func handle(_ event: EngineEvent, item: UUID) {
        guard let qi = queue.first(where: { $0.id == item }) else { return }
        let doc = document(for: qi)
        switch event {
        case .started: break
        case .nodeStarted(let n): setState(.running(step: 0, total: 0), n, in: doc)
        case .nodeCached(let n): setState(.cached, n, in: doc)
        case .progress(let n, let step, let total, let preview):
            setState(.running(step: step, total: total), n, in: doc)
            if let preview { doc.previewImage = preview; doc.previewNode = n; previewCount += 1 }
            if let i = queue.firstIndex(where: { $0.id == item }) { queue[i].status = .running(step: step, total: total) }
        case .nodeFinished(let n, let s, let peak): setState(.done(seconds: s, peakMemory: peak), n, in: doc)
        case .nodeFailed(let n, let msg): setState(.failed(msg), n, in: doc)
        case .finished, .cancelled: break
        }
    }

    private func finish(_ id: UUID, status: QueueItem.Status, outputs: [EngineOutput]) {
        // Let queued progress events land before marking the item finished.
        Task { @MainActor in
            if let i = queue.firstIndex(where: { $0.id == id }) { queue[i].status = status }
            let item = queue.first { $0.id == id }
            let doc = item.map(document(for:)) ?? active
            doc.record(outputs)
            // After the line above, so a finished plot's grid replaces the last cell's image.
            if let gid = item?.groupID, let cell = item?.cell {
                recordPlotCell(gid, cell: cell, image: outputs.last?.image)
            }
            doc.previewImage = nil
            doc.previewNode = nil
            runningDocID = nil
            runTask = nil
            if !outputs.isEmpty { reloadHistory() }
            processQueue()
        }
    }

    func cancelCurrent() { runTask?.cancel() }

    func cancelAll() {
        for i in queue.indices where queue[i].status == .pending { queue[i].status = .cancelled }
        runTask?.cancel()
        completePlotsIfDone()
    }

    func removeFromQueue(_ id: UUID) {
        guard let i = queue.firstIndex(where: { $0.id == id }) else { return }
        if case .running = queue[i].status { return }
        queue.remove(at: i)
        completePlotsIfDone()
    }

    func clearFinished() { queue.removeAll { $0.isFinished } }

    func movePending(from source: IndexSet, to destination: Int) { queue.move(fromOffsets: source, toOffset: destination) }

    // MARK: model

    func preloadModel() {
        guard !modelLoading else { return }
        modelLoading = true
        modelStatus = "Loading model…"
        let path = modelPath
        Task {
            do {
                let name = try await engine.preload(modelPath: path)
                modelStatus = name
            } catch {
                modelStatus = "Model failed to load"
                lastError = String(describing: error)
            }
            modelLoading = false
        }
    }

    func unloadModel(_ key: String) {
        Task {
            await engine.unload(key)
            loadedModels = engine.loadedModelsSnapshot()
            if loadedModels.isEmpty { modelStatus = "Model not loaded" }
        }
    }

    func unloadModels() {
        Task {
            await engine.unloadModels()
            modelStatus = "Model not loaded"
        }
    }

    /// Changes the Qwen-Image folder setting; Load Model nodes of the open tabs that used the old folder follow.
    func applyModelPath(_ path: String, undoManager: UndoManager?) {
        let old = modelPath
        modelPath = path
        for doc in tabs {
            var g = doc.graph
            for n in g.nodes where n.type == "LoadModel" {
                if case .value(.string(let p))? = n.inputs["path"], p == old {
                    try? g.setValue(.string(path), node: n.id, input: "path")
                }
            }
            setGraph(Self.syncingImg2Img(g), in: doc, actionName: "Change Model",
                     undoManager: doc === active ? undoManager : doc.undoManager)
        }
    }

    // MARK: history & files

    /// The output folder and its per-workflow subfolders (one level; not `masks`).
    func reloadHistory() {
        let dir = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath)
        let fm = FileManager.default
        let top = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
                                               options: [.skipsHiddenFiles])) ?? []
        var urls: [URL] = []
        for url in top {
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                guard url.lastPathComponent != "masks" else { continue }
                urls += (try? fm.contentsOfDirectory(at: url, includingPropertiesForKeys: [.contentModificationDateKey],
                                                     options: [.skipsHiddenFiles])) ?? []
            } else {
                urls.append(url)
            }
        }
        history = urls.compactMap { url -> HistoryItem? in
            let ext = url.pathExtension.lowercased()
            let kind: MediaKind
            if ext == "png" { kind = .image }
            else if MediaIO.videoExtensions.contains(ext) { kind = .video }
            else if MediaIO.audioExtensions.contains(ext) { kind = .audio }
            else { return nil }
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            let folder = url.deletingLastPathComponent().standardizedFileURL == dir.standardizedFileURL
                ? nil : url.deletingLastPathComponent().lastPathComponent
            return HistoryItem(url: url, date: date, kind: kind, prompt: promptCache[url],
                               isFavorite: Favorites.isFavorite(url), workflow: folder)
        }.sorted { $0.date > $1.date }
        loadHistoryPrompts()
    }

    func show(_ item: HistoryItem) {
        if item.kind != .image {
            displayedMedia = (item.url, item.kind)
            displayedURL = item.url
            displayedImage = nil
            if item.kind == .video {
                let url = item.url
                Task { if displayedMediaURL == url { displayedImage = await MediaIO.posterFrame(url) } }
            }
            return
        }
        displayedMedia = nil
        if let src = CGImageSourceCreateWithURL(item.url as CFURL, nil),
           let img = CGImageSourceCreateImageAtIndex(src, 0, nil) {
            displayedImage = img
            displayedURL = item.url
        }
    }

    /// Graph embedded in a PNG saved by SaveImage.
    nonisolated static func embeddedGraph(in url: URL) -> Graph? {
        guard let json = ImageArray.readPNGText(url)[StandardNodes.graphMetadataKey] else { return nil }
        return try? JSONDecoder().decode(Graph.self, from: Data(json.utf8))
    }

    func trash(_ item: HistoryItem) {
        try? FileManager.default.trashItem(at: item.url, resultingItemURL: nil)
        if displayedURL == item.url { displayedImage = nil; displayedURL = nil; displayedMedia = nil }
        reloadHistory()
    }
}
