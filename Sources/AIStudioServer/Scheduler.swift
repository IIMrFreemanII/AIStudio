import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO
import MLX
import AIStudioProtocol
import MLXDiffusionCore
import NodesStandard
import UniformTypeIdentifiers

/// The server's queue: graph runs and prompt-assistant requests, executed one at a time (one GPU, one MLX stream).
actor Scheduler {
    struct Run {
        var summary: RunSummary
        var graph: Graph
        var metadataGraph: Graph?
        var files: [String]
        var models: [String]
        var folder: URL
        var session: String
        var nodeIndex: [String: UInt16]
    }

    struct TextJob {
        var request: TextRequest
        var session: String
    }

    let config: ServerConfigBox
    let hub: Hub
    let files: FileStore
    let log = ServerLog.shared
    /// Not thread-safe: only the worker touches it, one job at a time.
    nonisolated(unsafe) let executor: GraphExecutor

    private var runs: [UInt32: Run] = [:]
    private var waiting: [UInt32] = []
    private var history: [UInt32] = []
    private var textJobs: [TextJob] = []
    private var current: UInt32?
    private var currentTask: Task<Void, Never>?
    private var currentText: (id: String, task: Task<Void, Never>)?
    private var cancelledTexts: Set<String> = []
    private var nextRun: UInt32
    private let wake: AsyncStream<Void>.Continuation
    private let wakes: AsyncStream<Void>
    private var started = false
    /// Bumped on every change the dashboard shows.
    private(set) var revision = 0

    init(config: ServerConfigBox, hub: Hub, files: FileStore) {
        self.config = config
        self.hub = hub
        self.files = files
        let ex = GraphExecutor(registry: StandardNodes.makeRegistry(), hooks: StandardNodes.mlxHooks)
        executor = ex
        ex.resources.budgetBytes = Int(Double(SystemInfo.physicalMemory) * min(max(config.value.budgetFraction, 0.3), 0.95))
        ex.resources.onEvict = { [weak ex] keys in
            ex?.clearCache()
            Memory.clearCache()
            ServerLog.shared.info("unloaded \(keys.joined(separator: ", ")) to make room")
        }
        (wakes, wake) = AsyncStream.makeStream(of: Void.self, bufferingPolicy: .bufferingNewest(1))
        let runsDir = config.value.runsDirectory
        try? FileManager.default.createDirectory(at: runsDir, withIntermediateDirectories: true)
        let existing = ((try? FileManager.default.contentsOfDirectory(atPath: runsDir.path)) ?? []).compactMap(UInt32.init)
        nextRun = (existing.max() ?? 0) &+ 1
    }

    func start() {
        guard !started else { return }
        started = true
        Task.detached { [self] in await self.workLoop() }
    }

    var nodeTypes: [String] { executor.registry.schemas.map(\.type) }

    // MARK: submitting

    enum SubmitError: Error, CustomStringConvertible {
        case invalid(String)
        var description: String { if case .invalid(let m) = self { return m }; return "" }
    }

    /// Validates and queues a run; returns its id and node table.
    func submit(_ req: SubmitRequest, session: String, client: String) throws -> (UInt32, [String]) {
        let g = req.graph
        // Same checks the executor makes, before anything is queued: unknown nodes, bad links, missing inputs.
        let order: [GraphNode]
        do { order = try executor.validate(g) } catch { throw SubmitError.invalid(String(describing: error)) }
        guard order.count < Int(WireEvent.noNode) else { throw SubmitError.invalid("graph has too many nodes") }
        for name in req.files where !GraphTransport.isServerName(name) {
            throw SubmitError.invalid("bad input file name \(name)")
        }
        // Models must be installed here; say which are missing instead of failing mid-run.
        let refs = GraphTransport.references(in: try g.flattened(), schema: executor.registry.schema)
        let models = refs.filter { $0.role == .model }.map(\.value)
        let modelsDir = config.value.modelsDirectory
        let missing = models.filter { !FileManager.default.fileExists(atPath: modelsDir.appendingPathComponent($0).path) }
        if !missing.isEmpty {
            throw SubmitError.invalid("models not installed on this server: \(missing.joined(separator: ", "))")
        }
        let id = nextRun
        nextRun &+= 1
        let folder = config.value.runsDirectory.appendingPathComponent(String(id))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let nodes = g.nodes.map(\.id)
        let index = Dictionary(uniqueKeysWithValues: nodes.enumerated().map { ($1, UInt16($0)) })
        let summary = RunSummary(run: id, workflow: req.workflow, client: client, status: .waiting,
                                 created: Date().timeIntervalSince1970, nodes: nodes, nodeTypes: g.nodes.map(\.type))
        runs[id] = Run(summary: summary, graph: g, metadataGraph: req.metadataGraph, files: req.files, models: models,
                       folder: folder, session: session, nodeIndex: index)
        waiting.append(id)
        hub.subscribe(session, run: id, after: .max)
        revision += 1
        log.info("run \(id) queued from \(client): \(req.workflow.isEmpty ? "untitled" : req.workflow), \(nodes.count) nodes")
        wake.yield()
        return (id, nodes)
    }

    func submitText(_ req: TextRequest, session: String) {
        textJobs.append(TextJob(request: req, session: session))
        wake.yield()
    }

    func cancelText(_ id: String) {
        if let i = textJobs.firstIndex(where: { $0.request.requestID == id }) {
            let job = textJobs.remove(at: i)
            hub.send(.textDone(requestID: id, text: nil, error: "cancelled"), to: job.session)
        } else if currentText?.id == id {
            cancelledTexts.insert(id)
            currentText?.task.cancel()
        }
    }

    func cancel(_ id: UInt32) {
        if let i = waiting.firstIndex(of: id) {
            waiting.remove(at: i)
            finish(id, status: .cancelled, error: nil, seconds: nil)
            hub.publish(WireEvent(kind: .cancelled, run: id), urgent: true)
        } else if current == id {
            currentTask?.cancel()
            files.wakeAll()
        }
    }

    func clearWaiting() {
        for id in waiting { cancel(id) }
    }

    func folder(of run: UInt32) -> URL? { runs[run]?.folder }

    /// Input files that waiting or running jobs still need (kept when the cache is purged).
    var neededFiles: Set<String> {
        Set(waiting.compactMap { runs[$0]?.files }.joined()).union(current.flatMap { runs[$0]?.files } ?? [])
    }

    var queueLength: Int { waiting.count + (current == nil ? 0 : 1) }

    func snapshot() -> (queue: [RunSummary], history: [RunSummary]) {
        let q = (current.map { [$0] } ?? []) + waiting
        return (q.compactMap { runs[$0]?.summary }, history.compactMap { runs[$0]?.summary })
    }

    func setBudget(fraction: Double) {
        executor.resources.budgetBytes = Int(Double(SystemInfo.physicalMemory) * min(max(fraction, 0.3), 0.95))
        revision += 1
    }

    // MARK: models

    func loadedModels() -> [LoadedModelInfo] {
        executor.resources.entries.filter { $0.key.hasPrefix("model:") }.map {
            LoadedModelInfo(key: $0.key, name: URL(fileURLWithPath: String($0.key.dropFirst(6))).lastPathComponent,
                            bytes: $0.bytes, pinned: $0.pinned, lastUsed: $0.lastUsed.timeIntervalSince1970)
        }
    }

    func unload(_ key: String) {
        executor.resources.remove(key)
        Memory.clearCache()
        revision += 1
    }

    /// Loads a model now (between jobs), so the first run doesn't wait for it.
    func preload(_ name: String) async throws {
        let url = try PathSandbox.contained(name, in: config.value.modelsDirectory)
        _ = try await loadModel(url)
        revision += 1
    }

    private func loadModel(_ url: URL) async throws -> GenerativeModel {
        try await executor.resources.get("model:\(url.path)", estimatedBytes: ModelRegistry.weightBytes(at: url)) {
            try await ModelRegistry.shared.load(url)
        }
    }

    // MARK: worker

    private func workLoop() async {
        for await _ in wakes {
            while let job = nextJob() {
                switch job {
                case .text(let t): await runText(t)
                case .graph(let id): await runGraph(id)
                }
            }
        }
    }

    private enum Job { case text(TextJob), graph(UInt32) }

    private func nextJob() -> Job? {
        // Prompt-assistant requests are interactive: they go before queued runs (never during one).
        if !textJobs.isEmpty { return .text(textJobs.removeFirst()) }
        if !waiting.isEmpty { return .graph(waiting.removeFirst()) }
        return nil
    }

    private func runGraph(_ id: UInt32) async {
        guard var run = runs[id] else { return }
        current = id
        run.summary.status = .running
        run.summary.started = Date().timeIntervalSince1970
        runs[id] = run
        revision += 1
        let tracker = RunTracker(run: id, folder: run.folder, nodeIndex: run.nodeIndex, hub: hub, log: log)
        let graph = run.graph, meta = run.metadataGraph, needed = run.files, models = run.models
        let modelsDir = config.value.modelsDirectory
        let executor = self.executor
        let files = self.files
        let task = Task.detached {
            do {
                if !needed.allSatisfy(files.exists) {
                    // While the inputs upload, load the run's models: the two overlap instead of adding up.
                    for m in models { _ = try? await self.loadModel(modelsDir.appendingPathComponent(m)) }
                    while !needed.allSatisfy(files.exists) {
                        try Task.checkCancellation()
                        await files.wait(for: needed)
                    }
                    try Task.checkCancellation()
                }
                let result = try await PathSandbox.$runFolder.withValue(run.folder) {
                    try await executor.run(graph, metadataGraph: meta) { event in tracker.handle(event) }
                }
                await self.finish(id, status: .done, error: nil, seconds: result.seconds, tracker: tracker)
            } catch is CancellationError {
                tracker.publish(WireEvent(kind: .cancelled, run: id), urgent: true)
                await self.finish(id, status: .cancelled, error: nil, seconds: nil, tracker: tracker)
            } catch {
                let message = String(describing: error)
                tracker.publish(WireEvent(kind: .runFailed, run: id, payload: Data(message.utf8)), urgent: true)
                await self.finish(id, status: .failed, error: message, seconds: nil, tracker: tracker)
            }
        }
        currentTask = task
        await task.value
        currentTask = nil
        current = nil
        pruneRunFolders()
    }

    private func finish(_ id: UInt32, status: RunStatus, error: String?, seconds: Double?, tracker: RunTracker? = nil) {
        guard var run = runs[id] else { return }
        run.summary.status = status
        run.summary.error = error
        run.summary.seconds = seconds
        run.summary.finished = Date().timeIntervalSince1970
        if let tracker {
            run.summary.outputs = tracker.outputs
            run.summary.peakMemory = tracker.peakMemory
        }
        runs[id] = run
        history.insert(id, at: 0)
        if history.count > 200 {
            for old in history.dropFirst(200) { runs.removeValue(forKey: old) }
            history = Array(history.prefix(200))
        }
        revision += 1
        let secs = seconds.map { String(format: " in %.1fs", $0) } ?? ""
        switch status {
        case .done: log.info("run \(id) done\(secs)")
        case .failed: log.error("run \(id) failed: \(error ?? "")")
        case .cancelled: log.info("run \(id) cancelled")
        default: break
        }
    }

    private func pruneRunFolders() {
        let dir = config.value.runsDirectory
        let ids = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).compactMap(UInt32.init).sorted()
        let active = Set(waiting + (current.map { [$0] } ?? []))
        for id in ids.dropLast(config.value.keepRuns) where !active.contains(id) {
            try? FileManager.default.removeItem(at: dir.appendingPathComponent(String(id)))
        }
    }

    private func runText(_ job: TextJob) async {
        let req = job.request
        let session = job.session
        let hub = self.hub
        let inputs = config.value.inputsDirectory
        let modelsDir = config.value.modelsDirectory
        let task = Task.detached {
            do {
                let url = try PathSandbox.contained(req.model, in: modelsDir)
                let model = try await self.loadModel(url)
                guard let writer = model as? TextGenerating else {
                    throw DiffusionError.unknownModel(url, "\(model.displayName) can't generate text")
                }
                let images = try req.images.map { name -> MLXArray in
                    let u = try PathSandbox.contained(name, in: inputs)
                    return ImageArray.array(from: try ImageArray.load(u))
                }
                var last = Date.distantPast
                let text = try writer.generateText(
                    system: req.system, user: req.user, images: images,
                    options: TextGenerationOptions(maxTokens: req.maxTokens, temperature: Float(req.temperature))
                ) { partial in
                    // Throttle deltas to ~20/s; the final text always follows.
                    if Date().timeIntervalSince(last) > 0.05 {
                        last = Date()
                        hub.send(.textDelta(requestID: req.requestID, text: partial), to: session)
                    }
                    return !Task.isCancelled
                }
                hub.send(.textDone(requestID: req.requestID, text: text, error: nil), to: session)
            } catch {
                hub.send(.textDone(requestID: req.requestID, text: nil, error: String(describing: error)), to: session)
            }
        }
        currentText = (req.requestID, task)
        await task.value
        currentText = nil
        if cancelledTexts.remove(req.requestID) != nil {
            hub.send(.textDone(requestID: req.requestID, text: nil, error: "cancelled"), to: session)
        }
    }
}

/// Settings that can change while the server runs (the dashboard's Settings panel).
final class ServerConfigBox: @unchecked Sendable {
    private var _value: ServerConfig
    private let lock = NSLock()
    init(_ v: ServerConfig) { _value = v }
    var value: ServerConfig { lock.withLock { _value } }
    func update(_ body: (inout ServerConfig) -> Void) { lock.withLock { body(&_value) } }
}

/// Translates one run's executor events into wire records. Called synchronously from the executor (on the
/// sampler's thread), so it does only cheap work — except preview frames, which it renders inline but no more
/// often than the fastest watching client asks for.
final class RunTracker: @unchecked Sendable {
    let run: UInt32
    let folder: URL
    let nodeIndex: [String: UInt16]
    let hub: Hub
    let log: ServerLog
    private let lock = NSLock()
    private var _outputs: [OutputDescriptor] = []
    private var _peak: Int?
    private var lastPreview: UInt64 = 0

    init(run: UInt32, folder: URL, nodeIndex: [String: UInt16], hub: Hub, log: ServerLog) {
        self.run = run
        self.folder = folder
        self.nodeIndex = nodeIndex
        self.hub = hub
        self.log = log
    }

    var outputs: [OutputDescriptor] { lock.withLock { _outputs } }
    var peakMemory: Int? { lock.withLock { _peak } }

    func publish(_ e: WireEvent, urgent: Bool = false) { hub.publish(e, urgent: urgent) }

    func handle(_ event: ExecutionEvent) {
        switch event {
        case .started:
            publish(WireEvent(kind: .started, run: run), urgent: true)
        case .nodeStarted(let n):
            publish(WireEvent(kind: .nodeStarted, node: index(n), run: run), urgent: true)
        case .nodeCached(let n):
            publish(WireEvent(kind: .nodeCached, node: index(n), run: run), urgent: true)
        case .nodeProgress(let n, let step, let total, let preview):
            let node = index(n)
            publish(WireEvent(kind: .progress, node: node, run: run, step: UInt16(clamping: step),
                              total: UInt16(clamping: total)))
            if let preview { renderPreview(preview, node: node, step: step, total: total) }
        case .nodeStatus(let n, let text):
            publish(WireEvent(kind: .nodeStatus, node: index(n), run: run, payload: Data((text ?? "").utf8)))
        case .nodeFinished(let n, let s, let peak):
            lock.withLock { if let peak { _peak = max(_peak ?? 0, peak) } }
            publish(WireEvent(kind: .nodeFinished, node: index(n), run: run,
                              payload: WireEvent.nodeFinishedPayload(seconds: s, peakMemory: peak)), urgent: true)
        case .nodeFailed(let n, let e):
            publish(WireEvent(kind: .nodeFailed, node: index(n), run: run, payload: Data(String(describing: e).utf8)),
                    urgent: true)
        case .nodeOutput(let n, let values):
            for d in describe(values, node: n) {
                lock.withLock { _outputs.append(d) }
                if let json = try? JSONEncoder().encode(d) {
                    publish(WireEvent(kind: .outputReady, node: index(n), run: run, payload: json), urgent: true)
                }
            }
        case .finished(let s):
            publish(WireEvent(kind: .finished, run: run, payload: WireEvent.float64(s)), urgent: true)
        case .cancelled:
            break
        }
    }

    private func index(_ n: String) -> UInt16 { nodeIndex[n] ?? WireEvent.noNode }

    /// Files an output node produced, as names inside the run folder. Images shown by Preview nodes have no file
    /// yet: they are written here (PNG) so clients can fetch them.
    private func describe(_ values: [String: Value], node: String) -> [OutputDescriptor] {
        let isPreview: Bool = { if case .bool(true)? = values["preview"] { return true }; return false }()
        let idx = Int(index(node))
        var urls: [(URL, OutputDescriptor.Kind)] = []
        if case .object(let o)? = values["images"], let us = o as? [URL] { urls += us.map { ($0, .image) } }
        if case .object(let o)? = values["video_url"], let u = o as? URL { urls.append((u, .video)) }
        if case .object(let o)? = values["audio_url"], let u = o as? URL { urls.append((u, .audio)) }
        var previewImage = false
        if urls.isEmpty, case .object(let o)? = values["image"], let array = o as? MLXArray {
            let url = folder.appendingPathComponent("preview_\(node)_\(UUID().uuidString.prefix(8)).png")
            do {
                try ImageArray.savePNG(array, to: url)
                urls.append((url, .image))
                previewImage = true
            } catch {
                log.warning("run \(run): could not write the preview of \(node): \(error)")
            }
        }
        return urls.compactMap { url, kind in
            // Only files inside this run's folder can be served.
            let path = url.standardizedFileURL.path
            guard path.hasPrefix(folder.standardizedFileURL.path + "/") else { return nil }
            let name = String(path.dropFirst(folder.standardizedFileURL.path.count + 1))
            let bytes = ((try? FileManager.default.attributesOfItem(atPath: path))?[.size] as? Int) ?? 0
            return OutputDescriptor(name: name, kind: kind, bytes: bytes, isPreview: isPreview || previewImage, node: idx)
        }
    }

    private func renderPreview(_ preview: Any, node: UInt16, step: Int, total: Int) {
        let demand = hub.previewDemand(run: run)
        guard demand.hz > 0 else { return }
        let now = DispatchTime.now().uptimeNanoseconds
        // Always send the last step, so the final preview is the finished latent.
        guard step >= total || now &- lastPreview >= UInt64(1e9 / demand.hz) else { return }
        lastPreview = now
        var rgb: MLXArray?
        if let p = preview as? LatentPreview {
            rgb = p.model.latentPreview(p.latents, width: p.width, height: p.height)
        } else if let a = preview as? MLXArray {
            rgb = a
        }
        guard let rgb, let image = try? ImageArray.cgImage(from: rgb),
              let jpeg = PreviewEncoder.shared.jpeg(image, maxSide: demand.maxSide) else { return }
        hub.publishPreview(run: run, node: node, step: step, total: total, jpeg: jpeg)
    }
}

/// Downscales and JPEG-encodes preview frames, reusing one bitmap context per size.
final class PreviewEncoder: @unchecked Sendable {
    static let shared = PreviewEncoder()
    private var context: CGContext?
    private let lock = NSLock()

    func jpeg(_ image: CGImage, maxSide: Int, quality: Double = 0.7) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        let scale = min(1, Double(max(64, maxSide)) / Double(max(image.width, image.height)))
        let w = max(1, Int(Double(image.width) * scale)), h = max(1, Int(Double(image.height) * scale))
        var source = image
        if w != image.width || h != image.height {
            if context?.width != w || context?.height != h {
                context = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                context?.interpolationQuality = .medium
            }
            guard let ctx = context else { return nil }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            guard let scaled = ctx.makeImage() else { return nil }
            source = scaled
        }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, source, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
