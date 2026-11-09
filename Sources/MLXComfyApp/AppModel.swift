import AppKit
import Foundation
import GraphRuntime
import ImageIO
import MLXDiffusionCore
import NodesStandard
import Observation

enum Workspace: String, CaseIterable, Identifiable {
    case image = "Image"
    case graph = "Graph"
    var id: String { rawValue }
}

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
    var id: URL { url }
}

@MainActor @Observable
final class AppModel {
    // MARK: document

    var graph: Graph
    var selection: Set<String> = []
    /// Subgraph instances entered in the editor, outermost first (empty = the top-level graph).
    var editingPath: [String] = []
    /// The group selected in the editor, if any (at the current editing level).
    var selectedGroup: String?
    var workspace: Workspace = .image
    var documentURL: URL?

    // MARK: run state

    var queue: [QueueItem] = []
    var nodeStates: [String: NodeRunState] = [:]
    var previewImage: CGImage?
    var displayedImage: CGImage?
    var displayedURL: URL?
    /// A video or audio result shown in the canvas (nil for images).
    var displayedMedia: (url: URL, kind: MediaKind)? {
        get { displayedMediaURL.map { ($0, displayedMediaKind) } }
        set { displayedMediaURL = newValue?.url; displayedMediaKind = newValue?.kind ?? .image }
    }
    var displayedMediaURL: URL?
    var displayedMediaKind: MediaKind = .image
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
    /// The Simple panel's mode.
    /// A Simple-panel mode that has no graph of its own yet (music, speech, sound until their models land).
    var pendingMode: SimpleMode?
    /// Image or video follows the graph (a Video Sampler means video), so undo and opening files switch it too.
    var simpleMode: SimpleMode {
        if let pendingMode { return pendingMode }
        return Self.mode(of: graph)
    }

    /// Which Simple-panel mode a graph belongs to, from the nodes it uses.
    static func mode(of g: Graph) -> SimpleMode {
        if g.nodes.contains(where: { $0.type == "VideoSampler" }) { return .video }
        if g.nodes.contains(where: { $0.type == "TextToAudio" }) { return .sound }
        if g.nodes.contains(where: { $0.type == "TextToMusic" }) { return .music }
        if g.nodes.contains(where: { $0.type == "TextToSpeech" }) { return .speech }
        return .image
    }
    /// The graph each mode had when the user switched away, restored on switching back.
    @ObservationIgnored var modeGraphs: [SimpleMode: Graph] = [:]
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

    // MARK: prompt assistant

    /// The assistant task in progress (Enhance / Describe), if any.
    var assistantMode: PromptAssistant.Mode?
    @ObservationIgnored var assistantTask: Task<Void, Never>?
    @ObservationIgnored var assistantRun: UUID?

    // MARK: batch & compare

    /// Prompt styles (built-in ones until edited) and the one applied to the Simple prompt at queue time.
    var styles: [PromptStyle] = PromptStyle.builtIn {
        didSet { if persistSettings, let d = try? JSONEncoder().encode(styles) { UserDefaults.standard.set(d, forKey: "promptStyles") } }
    }
    var selectedStyleName: String? {
        didSet { if persistSettings { UserDefaults.standard.set(selectedStyleName, forKey: "promptStyle") } }
    }
    /// XY plots in progress, by group ID.
    @ObservationIgnored var plots: [UUID: PlotJob] = [:]
    /// A/B compare: the image shown against the displayed one, and how.
    var compareURL: URL?
    var compareImage: CGImage?
    var compareMode: CompareMode = .split
    /// Prompts read from history PNGs, so reloads don't re-read files.
    @ObservationIgnored var promptCache: [URL: String] = [:]

    @ObservationIgnored let engine = Engine()
    @ObservationIgnored let registry = StandardNodes.makeRegistry()
    @ObservationIgnored private var runTask: Task<Void, Never>?
    @ObservationIgnored private var memoryTimer: Timer?
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

    init(persistSettings: Bool = true, autoloadModel: Bool = true) {
        self.persistSettings = persistSettings
        self.autoloadModel = autoloadModel
        let defaults = UserDefaults.standard
        let path = defaults.string(forKey: "modelPath") ?? AppModel.guessModelPath()
        let out = defaults.string(forKey: "outputDirectory")
            ?? (NSHomeDirectory() as NSString).appendingPathComponent("Pictures/MLXComfyUI")
        modelPath = path
        outputDirectory = out
        graph = GraphTemplates.textToImage(modelPath: path, outputDirectory: out)
        if persistSettings {
            if let d = defaults.data(forKey: "promptStyles"), let s = try? JSONDecoder().decode([PromptStyle].self, from: d) {
                styles = s
            }
            selectedStyleName = defaults.string(forKey: "promptStyle")
            if defaults.object(forKey: "memoryBudget") != nil { memoryBudget = defaults.double(forKey: "memoryBudget") }
        }
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

    /// Replaces the document graph, registering the inverse with `undoManager`.
    func setGraph(_ new: Graph, actionName: String, undoManager: UndoManager?) {
        let old = graph
        guard old != new else { return }
        graph = new
        if !editingPath.isEmpty, levelGraph(editingPath) == nil { editingPath = []; selection = [] }
        selection = selection.filter { id in editorGraph.nodes.contains { $0.id == id } }
        if let g = selectedGroup, !(editorGraph.groups ?? []).contains(where: { $0.id == g }) { selectedGroup = nil }
        undoManager?.registerUndo(withTarget: self) { target in
            MainActor.assumeIsolated {
                target.setGraph(old, actionName: actionName, undoManager: undoManager)
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
    func commitLiveEdit(from start: Graph, actionName: String, undoManager: UndoManager?) {
        let end = graph
        graph = start
        setGraph(end, actionName: actionName, undoManager: undoManager)
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
        let ids = selection
        guard !ids.isEmpty else { return }
        var mapping: [String: String] = [:]
        var copies: [GraphNode] = []
        let level = editorGraph
        for node in level.nodes where ids.contains(node.id) {
            var n = 2
            while level[node.id + "_\(n)"] != nil || mapping.values.contains(node.id + "_\(n)") { n += 1 }
            mapping[node.id] = node.id + "_\(n)"
        }
        for node in level.nodes where ids.contains(node.id) {
            var c = node
            c.id = mapping[node.id]!
            c.position = GraphPoint(x: (node.position?.x ?? 0) + 40, y: (node.position?.y ?? 0) + 40)
            for (name, input) in c.inputs {
                if case .link(let src, let out) = input, let m = mapping[src] { c.inputs[name] = .link(node: m, output: out) }
            }
            copies.append(c)
        }
        mutateEditor("Duplicate", undoManager: undoManager) { g in g.nodes += copies }
        selection = Set(copies.map(\.id))
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
            g = prepared(g, seed: seed ?? i)
            queue.append(QueueItem(graph: g, title: title(for: g), seed: seed))
        }
        processQueue()
    }

    /// Speaks a short sample in `voice` through the queue; the clip is shown and played, not saved to history.
    func previewVoice(_ voice: String, speed: Double, modelPath: String) {
        let g = GraphTemplates.voicePreview(modelPath: modelPath, voice: voice, speed: speed)
        let name = SpeechVoices.all.first { $0.id == voice }?.label ?? voice
        queue.append(QueueItem(graph: g, title: "Voice preview: \(name)", seed: nil))
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

    func processQueue() {
        guard runTask == nil, let idx = queue.firstIndex(where: { $0.status == .pending }) else { return }
        let item = queue[idx]
        queue[idx].status = .running(step: 0, total: 0)
        nodeStates = Dictionary(uniqueKeysWithValues: item.graph.nodes.map { ($0.id, NodeRunState.pending) })
        previewImage = nil
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
        switch event {
        case .started: break
        case .nodeStarted(let n): setState(.running(step: 0, total: 0), n)
        case .nodeCached(let n): setState(.cached, n)
        case .progress(let n, let step, let total, let preview):
            setState(.running(step: step, total: total), n)
            if let preview { previewImage = preview; previewCount += 1 }
            if let i = queue.firstIndex(where: { $0.id == item }) { queue[i].status = .running(step: step, total: total) }
        case .nodeFinished(let n, let s, let peak): setState(.done(seconds: s, peakMemory: peak), n)
        case .nodeFailed(let n, let msg): setState(.failed(msg), n)
        case .finished, .cancelled: break
        }
    }

    private func finish(_ id: UUID, status: QueueItem.Status, outputs: [EngineOutput]) {
        // Let queued progress events land before marking the item finished.
        Task { @MainActor in
            if let i = queue.firstIndex(where: { $0.id == id }) { queue[i].status = status }
            if let out = outputs.last {
                displayedImage = out.image
                displayedURL = out.url
                displayedMedia = out.kind == .image ? nil : out.url.map { ($0, out.kind) }
            }
            // After the line above, so a finished plot's grid replaces the last cell's image.
            let item = queue.first { $0.id == id }
            if let gid = item?.groupID, let cell = item?.cell {
                recordPlotCell(gid, cell: cell, image: outputs.last?.image)
            }
            previewImage = nil
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

    /// Points every LoadModel node at `path` (used when the model folder setting changes).
    func applyModelPath(_ path: String, undoManager: UndoManager?) {
        modelPath = path
        mutate("Change Model", undoManager: undoManager) { g in
            for n in g.nodes where n.type == "LoadModel" { try? g.setValue(.string(path), node: n.id, input: "path") }
            g = Self.syncingImg2Img(g)
        }
    }

    // MARK: history & files

    func reloadHistory() {
        let dir = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath)
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        history = urls.compactMap { url -> HistoryItem? in
            let ext = url.pathExtension.lowercased()
            let kind: MediaKind
            if ext == "png" { kind = .image }
            else if MediaIO.videoExtensions.contains(ext) { kind = .video }
            else if MediaIO.audioExtensions.contains(ext) { kind = .audio }
            else { return nil }
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return HistoryItem(url: url, date: date, kind: kind, prompt: promptCache[url],
                               isFavorite: Favorites.isFavorite(url))
        }.sorted { $0.date > $1.date }
        loadHistoryPrompts()
    }

    func show(_ item: HistoryItem) {
        if item.kind != .image {
            displayedMedia = (item.url, item.kind)
            displayedURL = item.url
            displayedImage = nil
            workspace = .image
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
            workspace = .image
        }
    }

    /// Graph embedded in a PNG saved by SaveImage.
    nonisolated static func embeddedGraph(in url: URL) -> Graph? {
        guard let json = ImageArray.readPNGText(url)[StandardNodes.graphMetadataKey] else { return nil }
        return try? JSONDecoder().decode(Graph.self, from: Data(json.utf8))
    }

    @discardableResult
    func open(_ url: URL, undoManager: UndoManager?) -> Bool {
        let ext = url.pathExtension.lowercased()
        if MediaIO.videoExtensions.contains(ext) || MediaIO.audioExtensions.contains(ext) {
            // Media metadata loads asynchronously; the graph is restored when it arrives.
            Task {
                guard let json = await MediaIO.readMetadata(url)[StandardNodes.graphMetadataKey],
                      let g = try? JSONDecoder().decode(Graph.self, from: Data(json.utf8)) else {
                    lastError = "\(url.lastPathComponent) has no embedded graph"
                    NSSound.beep()
                    return
                }
                setGraph(g, actionName: "Restore Graph", undoManager: undoManager)
                selection = []
                workspace = .graph
            }
            return true
        }
        let g: Graph?
        if ext == "png" {
            g = Self.embeddedGraph(in: url)
            if g == nil { lastError = "\(url.lastPathComponent) has no embedded graph" }
        } else {
            do { g = try Graph.load(url); documentURL = url } catch {
                lastError = "Could not open \(url.lastPathComponent): \(error)"
                g = nil
            }
        }
        guard let g else { NSSound.beep(); return false }
        setGraph(g, actionName: "Restore Graph", undoManager: undoManager)
        selection = []
        return true
    }

    func save(to url: URL) {
        do {
            try graph.jsonData().write(to: url, options: .atomic)
            documentURL = url
        } catch { lastError = "Could not save: \(error)" }
    }

    /// Switches the Simple panel's mode; image and video swap the graph for that mode's (kept per mode).
    func setSimpleMode(_ mode: SimpleMode, videoModelPath: String?, soundModelPath: String? = nil,
                       musicModelPath: String? = nil, speechModelPath: String? = nil, undoManager: UndoManager?) {
        switch mode {
        case .image, .video, .sound, .music, .speech:
            pendingMode = nil
            let current = Self.mode(of: graph)
            guard current != mode else { return }
            modeGraphs[current] = graph
            let next = modeGraphs[mode] ?? {
                switch mode {
                case .video: return GraphTemplates.textToVideo(modelPath: videoModelPath ?? "", outputDirectory: outputDirectory)
                case .sound: return GraphTemplates.textToSound(modelPath: soundModelPath ?? "", outputDirectory: outputDirectory)
                case .music: return GraphTemplates.textToMusic(modelPath: musicModelPath ?? "", outputDirectory: outputDirectory)
                case .speech: return GraphTemplates.textToSpeech(modelPath: speechModelPath ?? "", outputDirectory: outputDirectory)
                default: return GraphTemplates.textToImage(modelPath: modelPath, outputDirectory: outputDirectory)
                }
            }()
            setGraph(next, actionName: "Switch to \(mode.rawValue)", undoManager: undoManager)
            selection = []
        default:
            pendingMode = mode
        }
    }

    func newGraph(undoManager: UndoManager?) {
        setGraph(GraphTemplates.textToImage(modelPath: modelPath, outputDirectory: outputDirectory),
                 actionName: "New Graph", undoManager: undoManager)
        documentURL = nil
        selection = []
    }

    func trash(_ item: HistoryItem) {
        try? FileManager.default.trashItem(at: item.url, resultingItemURL: nil)
        if displayedURL == item.url { displayedImage = nil; displayedURL = nil; displayedMedia = nil }
        reloadHistory()
    }
}
