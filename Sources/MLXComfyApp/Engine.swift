import CoreGraphics
import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore
import NodesStandard

/// Events forwarded to the UI, with MLX payloads already converted to images.
enum EngineEvent {
    case started(nodes: [String])
    case nodeStarted(String)
    case nodeCached(String)
    case progress(node: String, step: Int, total: Int, preview: CGImage?)
    case nodeFinished(node: String, seconds: Double, peakMemory: Int?)
    case nodeFailed(node: String, message: String)
    case finished(seconds: Double)
    case cancelled
}

enum MediaKind: String, Sendable { case image, video, audio }

/// One result of a run: an image (with its file, if saved) or a video/audio file.
struct EngineOutput {
    let node: String
    let kind: MediaKind
    let image: CGImage?
    let url: URL?
    /// Written to the temporary preview folder rather than the output folder.
    var isPreview = false
}

/// A model held in memory, for the toolbar's model list.
struct LoadedModel: Identifiable, Hashable {
    let key: String
    let bytes: Int
    let pinned: Bool
    var id: String { key }
    var name: String { URL(fileURLWithPath: String(key.dropFirst("model:".count))).lastPathComponent }
}

/// Owns the graph executor and the loaded models; every MLX computation runs on this actor.
actor Engine {
    private let executor: GraphExecutor
    /// The executor's resource cache, readable without waiting for the actor (it is lock-protected).
    nonisolated(unsafe) private let cache: ResourceCache

    init(budgetFraction: Double = 0.8) {
        let ex = GraphExecutor(registry: StandardNodes.makeRegistry(), hooks: StandardNodes.mlxHooks)
        executor = ex
        cache = ex.resources
        ex.resources.budgetBytes = Self.budget(fraction: budgetFraction)
        // Cached node outputs (conditionings, latents) hold on to the model that made them: drop them with it.
        ex.resources.onEvict = { [weak ex] _ in
            ex?.clearCache()
            Memory.clearCache()
        }
    }

    static func budget(fraction: Double) -> Int {
        Int(Double(ProcessInfo.processInfo.physicalMemory) * min(max(fraction, 0.3), 0.95))
    }

    func setBudget(fraction: Double) { executor.resources.budgetBytes = Self.budget(fraction: fraction) }

    func loadedModels() -> [LoadedModel] {
        executor.resources.entries.filter { $0.key.hasPrefix("model:") }
            .map { LoadedModel(key: $0.key, bytes: $0.bytes, pinned: $0.pinned) }
    }

    func unload(_ key: String) { executor.resources.remove(key) }

    /// For the media smoke test: occupies `bytes` of budget under `key` (an empty object), evicting as needed.
    func debugReserve(_ key: String, bytes: Int) async throws {
        let _: NSObject = try await executor.resources.get(key, estimatedBytes: bytes) { NSObject() }
    }

    /// Loaded models without waiting for a run to finish.
    nonisolated func loadedModelsSnapshot() -> [LoadedModel] {
        cache.entries.filter { $0.key.hasPrefix("model:") }.map { LoadedModel(key: $0.key, bytes: $0.bytes, pinned: $0.pinned) }
    }

    /// Runs a graph, converting progress previews and outputs to `CGImage` before they leave the actor.
    func run(_ graph: Graph, onEvent: @escaping (EngineEvent) -> Void) async throws -> [EngineOutput] {
        let result = try await executor.run(graph) { event in
            switch event {
            case .started(let nodes): onEvent(.started(nodes: nodes))
            case .nodeStarted(let n): onEvent(.nodeStarted(n))
            case .nodeCached(let n): onEvent(.nodeCached(n))
            case .nodeProgress(let n, let step, let total, let preview):
                var image: CGImage?
                if let p = preview as? LatentPreview,
                   let rgb = p.model.latentPreview(p.latents, width: p.width, height: p.height) {
                    image = try? ImageArray.cgImage(from: rgb)
                }
                onEvent(.progress(node: n, step: step, total: total, preview: image))
            case .nodeFinished(let n, let s, let peak): onEvent(.nodeFinished(node: n, seconds: s, peakMemory: peak))
            case .nodeFailed(let n, let e): onEvent(.nodeFailed(node: n, message: String(describing: e)))
            case .finished(let s): onEvent(.finished(seconds: s))
            case .cancelled: onEvent(.cancelled)
            }
        }
        var outputs: [EngineOutput] = []
        for (node, values) in result.outputs.sorted(by: { $0.key < $1.key }) {
            let preview: Bool = { if case .bool(true)? = values["preview"] { return true }; return false }()
            if case .object(let u)? = values["video_url"], let url = u as? URL {
                outputs.append(EngineOutput(node: node, kind: .video, image: await MediaIO.posterFrame(url), url: url,
                                            isPreview: preview))
                continue
            }
            if case .object(let u)? = values["audio_url"], let url = u as? URL {
                outputs.append(EngineOutput(node: node, kind: .audio, image: nil, url: url, isPreview: preview))
                continue
            }
            guard case .object(let o)? = values["image"], let array = o as? MLXArray,
                  let cg = try? ImageArray.cgImage(from: array) else { continue }
            var url: URL?
            if case .object(let u)? = values["images"] { url = (u as? [URL])?.first }
            outputs.append(EngineOutput(node: node, kind: .image, image: cg, url: url))
        }
        return outputs
    }

    /// Loads a model into the shared resource cache so the first run doesn't pay for it.
    func preload(modelPath: String) async throws -> String {
        let url = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).standardizedFileURL
        let model: GenerativeModel = try await executor.resources.get(
            "model:\(url.path)", estimatedBytes: ModelRegistry.weightBytes(at: url)) {
            try await ModelRegistry.shared.load(url)
        }
        return model.displayName
    }

    /// Runs the prompt assistant on the (loaded, or loaded now) model. `onText` gets the full text after each token;
    /// cancelling the calling task stops generation.
    func generateText(modelPath: String, system: String, user: String, imagePaths: [String],
                      options: TextGenerationOptions, onText: @escaping (String) -> Void) async throws -> String {
        let url = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).standardizedFileURL
        let model: GenerativeModel = try await executor.resources.get(
            "model:\(url.path)", estimatedBytes: ModelRegistry.weightBytes(at: url)) {
            try await ModelRegistry.shared.load(url)
        }
        guard let writer = model as? TextGenerating else {
            throw DiffusionError.unknownModel(url, "\(model.displayName) can't generate text")
        }
        let images = try imagePaths.map { ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: $0))) }
        return try writer.generateText(system: system, user: user, images: images, options: options) { partial in
            onText(partial)
            return !Task.isCancelled
        }
    }

    func isLoaded(modelPath: String) -> Bool {
        let url = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).standardizedFileURL
        return executor.resources.contains("model:\(url.path)")
    }

    func unloadModels() {
        executor.resources.removeAll()
        executor.clearCache()
        Memory.clearCache()
    }

    nonisolated static func activeMemory() -> Int { Memory.activeMemory }
}
