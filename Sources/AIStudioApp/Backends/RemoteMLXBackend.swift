import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO
import AIStudioProtocol
import MLXDiffusionCore
import NodesRemote
import NodesStandard

/// A Mac running `aistudio-cli serve`: the whole graph runs there; inputs are uploaded once (by content hash) and
/// results are downloaded into the same folders a local run would write.
final class RemoteMLXBackend: ExecutionBackend, @unchecked Sendable {
    let id: String
    let displayName: String
    let isRemote = true
    let client: RemoteClient
    let registry: NodeRegistry
    let hasher: FileHasher

    struct RemoteError: Error, CustomStringConvertible {
        let description: String
    }

    init(id: String, name: String, endpoint: RemoteClient.Endpoint, registry: NodeRegistry, hasher: FileHasher) {
        self.id = id
        displayName = name
        self.registry = registry
        self.hasher = hasher
        let host = Host.current().localizedName ?? "Mac"
        client = RemoteClient(endpoint: endpoint,
                              hello: ClientHello(role: .app, name: "AIStudio on \(host)", previewHz: 4, previewMaxSide: 512))
    }

    deinit { client.close() }

    func run(_ graph: Graph, workflow: String, onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws
        -> [EngineOutput]
    {
        let start = Date()
        // Cloud GPU nodes call GPU servers with tokens kept on this Mac; a Mac server can't run them.
        let remote = Set(graph.nodes.map(\.type)).intersection(RemoteNodes.types)
        guard remote.isEmpty else {
            throw RemoteError(description: "This graph has Cloud GPU nodes, which run from this Mac: set the tab's "
                              + "target to This Mac (they still compute on the GPU server)")
        }
        let prepared = try GraphTransport.prepare(graph, schema: registry.schema,
                                                  resolveInput: StandardNodes.resolveInputPath,
                                                  hash: hasher.sha256(of:))
        hasher.save()
        let state = RunState(prepared: prepared, client: client, onEvent: onEvent)
        // Uploads and the submit go out together: the server loads the run's models while the files arrive.
        let uploads = Task {
            try await withThrowingTaskGroup(of: Void.self) { group in
                for (i, u) in prepared.uploads.enumerated() {
                    if i >= 4 { try await group.next() }
                    group.addTask { try await self.client.upload(u) }
                }
                try await group.waitForAll()
            }
        }
        let request = SubmitRequest(requestID: UUID().uuidString, graph: prepared.graph, metadataGraph: prepared.original,
                                    workflow: workflow, files: prepared.uploads.map(\.name))
        let run: UInt32
        do {
            (run, _) = try await client.submit(request) { e in state.handle(e) }
        } catch {
            uploads.cancel()
            throw error
        }
        state.run = run
        return try await withTaskCancellationHandler {
            do { try await uploads.value } catch {
                await client.cancel(run: run)
                throw RemoteError(description: "upload failed: \(error)")
            }
            let outputs = try await state.result()
            onEvent(.finished(seconds: Date().timeIntervalSince(start)))
            return outputs
        } onCancel: {
            uploads.cancel()
            Task { await self.client.cancel(run: run) }
        }
    }

    func generateText(modelPath: String, system: String, user: String, imagePaths: [String],
                      options: TextGenerationOptions, onText: @escaping @Sendable (String) -> Void) async throws -> String
    {
        var names: [String] = []
        for path in imagePaths {
            let url = URL(fileURLWithPath: path)
            let upload = GraphTransport.Upload(
                name: GraphTransport.serverName(sha256: try hasher.sha256(of: url), for: url), url: url,
                bytes: ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int) ?? 0)
            try await client.upload(upload)
            names.append(upload.name)
        }
        let model = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).lastPathComponent
        return try await client.generateText(TextRequest(
            requestID: UUID().uuidString, model: model, system: system, user: user, images: names,
            maxTokens: options.maxTokens, temperature: Double(options.temperature)), onText: onText)
    }

    /// One run's records → engine events, plus the downloads its outputs need.
    final class RunState: @unchecked Sendable {
        let prepared: GraphTransport.Prepared
        let client: RemoteClient
        let onEvent: @Sendable (EngineEvent) -> Void
        var run: UInt32 = 0
        private var nodes: [String]
        private let lock = NSLock()
        private var downloads: [Task<EngineOutput?, Error>] = []
        private var terminal: CheckedContinuation<Void, Error>?
        private var ended: Result<Void, Error>?

        init(prepared: GraphTransport.Prepared, client: RemoteClient, onEvent: @escaping @Sendable (EngineEvent) -> Void) {
            self.prepared = prepared
            self.client = client
            self.onEvent = onEvent
            nodes = prepared.graph.nodes.map(\.id)
        }

        func node(_ i: UInt16) -> String { Int(i) < nodes.count ? nodes[Int(i)] : "?" }

        func handle(_ e: WireEvent) {
            switch e.kind {
            case .started: onEvent(.started(nodes: nodes))
            case .nodeStarted: onEvent(.nodeStarted(node(e.node)))
            case .nodeCached: onEvent(.nodeCached(node(e.node)))
            case .progress: onEvent(.progress(node: node(e.node), step: Int(e.step), total: Int(e.total), preview: nil))
            case .preview:
                // Decoding happens here, off the main actor; the coalescer keeps only the newest frame.
                let image = CGImageSourceCreateWithData(e.payload as CFData, nil)
                    .flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
                onEvent(.progress(node: node(e.node), step: Int(e.step), total: Int(e.total), preview: image))
            case .nodeFinished:
                onEvent(.nodeFinished(node: node(e.node), seconds: e.seconds ?? 0, peakMemory: e.peakMemory))
            case .nodeFailed:
                onEvent(.nodeFailed(node: node(e.node), message: e.message))
            case .nodeStatus:
                onEvent(.status(node: node(e.node), text: e.payload.isEmpty ? nil : e.message))
            case .outputReady:
                guard let d = try? JSONDecoder().decode(OutputDescriptor.self, from: e.payload) else { return }
                startDownload(d)
            case .finished: end(.success(()))
            case .cancelled: end(.failure(CancellationError()))
            case .runFailed: end(.failure(RemoteError(description: e.message)))
            case .metrics, .metricsHistory: break
            }
        }

        private func end(_ r: Result<Void, Error>) {
            let c: CheckedContinuation<Void, Error>? = lock.withLock {
                guard ended == nil else { return nil }
                ended = r
                defer { terminal = nil }
                return terminal
            }
            c?.resume(with: r)
        }

        /// Waits for the run to end, then for its downloads.
        func result() async throws -> [EngineOutput] {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
                let r: Result<Void, Error>? = lock.withLock {
                    if let ended { return ended }
                    terminal = c
                    return nil
                }
                if let r { c.resume(with: r) }
            }
            var outputs: [EngineOutput] = []
            for t in lock.withLock({ downloads }) {
                if let o = try await t.value { outputs.append(o) }
            }
            return outputs
        }

        /// Downloads an output as soon as the server announces it, while the rest of the run continues.
        private func startDownload(_ d: OutputDescriptor) {
            let nodeID = node(UInt16(clamping: d.node))
            let run = self.run
            let client = self.client
            let dir: URL = {
                guard !d.isPreview, let local = prepared.outputDirectories[nodeID] else { return previewDirectory }
                let expanded = (local as NSString).expandingTildeInPath
                return expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded)
                    : URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(expanded)
            }()
            let task = Task<EngineOutput?, Error> {
                let dest = RemoteMLXBackend.localName(for: d.name, in: dir)
                try await client.download(run: run, name: d.name, to: dest)
                switch d.kind {
                case .image:
                    let image = CGImageSourceCreateWithURL(dest as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
                    return EngineOutput(node: nodeID, kind: .image, image: image, url: d.isPreview ? nil : dest,
                                        isPreview: d.isPreview)
                case .video:
                    return EngineOutput(node: nodeID, kind: .video, image: await MediaIO.posterFrame(dest), url: dest,
                                        isPreview: d.isPreview)
                case .audio:
                    return EngineOutput(node: nodeID, kind: .audio, image: nil, url: dest, isPreview: d.isPreview)
                }
            }
            lock.withLock { downloads.append(task) }
        }
    }

    /// A free local file name for a server output: `prefix_00003.png` keeps its prefix and takes the next number
    /// in `dir`, the way a local Save node numbers its files.
    static func localName(for serverName: String, in dir: URL) -> URL {
        // Concurrent downloads must not pick the same number: allocate under a lock and reserve the name on disk.
        nameLock.withLock {
            let url = freeName(for: serverName, in: dir)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            return url
        }
    }

    private static let nameLock = NSLock()

    private static func freeName(for serverName: String, in dir: URL) -> URL {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = URL(fileURLWithPath: serverName)
        let ext = url.pathExtension
        let stem = url.deletingPathExtension().lastPathComponent
        guard let underscore = stem.lastIndex(of: "_"), Int(stem[stem.index(after: underscore)...]) != nil else {
            let candidate = dir.appendingPathComponent(serverName)
            if !FileManager.default.fileExists(atPath: candidate.path) { return candidate }
            return dir.appendingPathComponent("\(stem)-\(UUID().uuidString.prefix(6)).\(ext)")
        }
        let prefix = String(stem[..<underscore])
        let existing = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let used = existing.compactMap { name -> Int? in
            guard name.hasPrefix(prefix + "_"), name.hasSuffix("." + ext) else { return nil }
            return Int(name.dropFirst(prefix.count + 1).dropLast(ext.count + 1))
        }
        return dir.appendingPathComponent(String(format: "%@_%05d.%@", prefix, (used.max() ?? 0) + 1, ext))
    }
}
