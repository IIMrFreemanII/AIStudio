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
    var id: URL { url }
}

@MainActor @Observable
final class AppModel {
    // MARK: document

    var graph: Graph
    var selection: Set<String> = []
    var workspace: Workspace = .image
    var documentURL: URL?

    // MARK: run state

    var queue: [QueueItem] = []
    var nodeStates: [String: NodeRunState] = [:]
    var previewImage: CGImage?
    var displayedImage: CGImage?
    var displayedURL: URL?
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
        reloadHistory()
        memoryTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.activeMemory = Engine.activeMemory() }
        }
    }

    /// Prefers a `qwen-turbo-mlx` folder next to the working directory or the executable's project.
    static func guessModelPath() -> String {
        let candidates = [
            FileManager.default.currentDirectoryPath + "/qwen-turbo-mlx",
            (NSHomeDirectory() as NSString).appendingPathComponent("ai/MLXComfyUI/qwen-turbo-mlx"),
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0 + "/model_index.json") } ?? candidates[0]
    }

    // MARK: schemas

    func schema(for node: GraphNode) -> NodeSchema? { registry.schema(node.type) }
    var schemas: [NodeSchema] { registry.schemas }

    // MARK: undoable edits

    /// Replaces the document graph, registering the inverse with `undoManager`.
    func setGraph(_ new: Graph, actionName: String, undoManager: UndoManager?) {
        let old = graph
        guard old != new else { return }
        graph = new
        selection = selection.filter { id in new.nodes.contains { $0.id == id } }
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
        while graph[n == 1 ? base : "\(base)\(n)"] != nil { n += 1 }
        let id = n == 1 ? base : "\(base)\(n)"
        var inputs: [String: NodeInput] = [:]
        for spec in schema.inputs where spec.type.isScalar {
            if let d = spec.defaultValue { inputs[spec.name] = .value(d) }
        }
        if type == "LoadModel" { inputs["path"] = .value(.string(modelPath)) }
        if type == "SaveImage" { inputs["directory"] = .value(.string(outputDirectory)) }
        // Auto-wire a MODEL input to the first model loader, the common case.
        if schema.inputs.contains(where: { $0.type == .model }), let m = firstNode(ofType: "LoadModel") {
            inputs["model"] = .link(node: m.id, output: "model")
        }
        mutate("Add \(schema.displayName)", undoManager: undoManager) { g in
            g.nodes.append(GraphNode(id: id, type: type, inputs: inputs, position: point))
        }
        selection = [id]
    }

    func deleteSelection(undoManager: UndoManager?) {
        let ids = selection
        guard !ids.isEmpty else { return }
        mutate(ids.count == 1 ? "Delete Node" : "Delete Nodes", undoManager: undoManager) { g in
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
        for node in graph.nodes where ids.contains(node.id) {
            var n = 2
            while graph[node.id + "_\(n)"] != nil || mapping.values.contains(node.id + "_\(n)") { n += 1 }
            mapping[node.id] = node.id + "_\(n)"
        }
        for node in graph.nodes where ids.contains(node.id) {
            var c = node
            c.id = mapping[node.id]!
            c.position = GraphPoint(x: (node.position?.x ?? 0) + 40, y: (node.position?.y ?? 0) + 40)
            for (name, input) in c.inputs {
                if case .link(let src, let out) = input, let m = mapping[src] { c.inputs[name] = .link(node: m, output: out) }
            }
            copies.append(c)
        }
        mutate("Duplicate", undoManager: undoManager) { g in g.nodes += copies }
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
        let samplers = graph.nodes.filter { $0.type == "Sampler" }.map(\.id)
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
            queue.append(QueueItem(graph: g, title: title(for: g), seed: seed))
        }
        processQueue()
    }

    func title(for g: Graph) -> String {
        if let p = g.nodes.first(where: { $0.type == "TextEncode" }), case .value(.string(let s))? = p.inputs["prompt"],
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
        case .nodeStarted(let n): nodeStates[n] = .running(step: 0, total: 0)
        case .nodeCached(let n): nodeStates[n] = .cached
        case .progress(let n, let step, let total, let preview):
            nodeStates[n] = .running(step: step, total: total)
            if let preview { previewImage = preview; previewCount += 1 }
            if let i = queue.firstIndex(where: { $0.id == item }) { queue[i].status = .running(step: step, total: total) }
        case .nodeFinished(let n, let s, let peak): nodeStates[n] = .done(seconds: s, peakMemory: peak)
        case .nodeFailed(let n, let msg): nodeStates[n] = .failed(msg)
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
    }

    func removeFromQueue(_ id: UUID) {
        guard let i = queue.firstIndex(where: { $0.id == id }) else { return }
        if case .running = queue[i].status { return }
        queue.remove(at: i)
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
        }
    }

    // MARK: history & files

    func reloadHistory() {
        let dir = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath)
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        history = urls.filter { $0.pathExtension.lowercased() == "png" }.map { url in
            let date = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate)
                ?? .distantPast
            return HistoryItem(url: url, date: date)
        }.sorted { $0.date > $1.date }
    }

    func show(_ item: HistoryItem) {
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
        let g: Graph?
        if url.pathExtension.lowercased() == "png" {
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

    func newGraph(undoManager: UndoManager?) {
        setGraph(GraphTemplates.textToImage(modelPath: modelPath, outputDirectory: outputDirectory),
                 actionName: "New Graph", undoManager: undoManager)
        documentURL = nil
        selection = []
    }

    func trash(_ item: HistoryItem) {
        try? FileManager.default.trashItem(at: item.url, resultingItemURL: nil)
        if displayedURL == item.url { displayedImage = nil; displayedURL = nil }
        reloadHistory()
    }
}
