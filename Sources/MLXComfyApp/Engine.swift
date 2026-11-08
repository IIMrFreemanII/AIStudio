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

struct EngineOutput {
    let node: String
    let image: CGImage
    let url: URL?
}

/// Owns the graph executor and the loaded models; every MLX computation runs on this actor.
actor Engine {
    private let executor: GraphExecutor

    init() {
        executor = GraphExecutor(registry: StandardNodes.makeRegistry(), hooks: StandardNodes.mlxHooks)
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
            guard case .object(let o)? = values["image"], let array = o as? MLXArray,
                  let cg = try? ImageArray.cgImage(from: array) else { continue }
            var url: URL?
            if case .object(let u)? = values["images"] { url = (u as? [URL])?.first }
            outputs.append(EngineOutput(node: node, image: cg, url: url))
        }
        return outputs
    }

    /// Loads a model into the shared resource cache so the first run doesn't pay for it.
    func preload(modelPath: String) async throws -> String {
        let url = URL(fileURLWithPath: (modelPath as NSString).expandingTildeInPath).standardizedFileURL
        let model: DiffusionModel = try await executor.resources.get("model:\(url.path)") {
            try await ModelRegistry.shared.load(url)
        }
        return model.displayName
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
