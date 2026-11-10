import CoreGraphics
import Foundation
import GraphRuntime
import MLXComfyProtocol
import MLXDiffusionCore

/// Where a graph runs: this Mac's engine, or a server. The queue keeps one worker per backend, so a local and a
/// remote run can go at the same time.
protocol ExecutionBackend: AnyObject, Sendable {
    /// "local", or the connection id.
    var id: String { get }
    var displayName: String { get }
    var isRemote: Bool { get }

    /// Runs a graph (as the queue prepared it); `onEvent` may be called from any thread.
    func run(_ graph: Graph, workflow: String, onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws
        -> [EngineOutput]

    /// The prompt assistant on this backend's copy of the model at `modelPath` (remote: its folder name).
    func generateText(modelPath: String, system: String, user: String, imagePaths: [String],
                      options: TextGenerationOptions, onText: @escaping @Sendable (String) -> Void) async throws -> String
}

enum BackendID {
    static let local = "local"
}

/// This Mac: the in-process MLX engine.
final class LocalBackend: ExecutionBackend, @unchecked Sendable {
    let engine: Engine
    let id = BackendID.local
    let displayName = "This Mac"
    let isRemote = false

    init(engine: Engine) { self.engine = engine }

    func run(_ graph: Graph, workflow: String, onEvent: @escaping @Sendable (EngineEvent) -> Void) async throws
        -> [EngineOutput]
    {
        try await engine.run(graph, onEvent: onEvent)
    }

    func generateText(modelPath: String, system: String, user: String, imagePaths: [String],
                      options: TextGenerationOptions, onText: @escaping @Sendable (String) -> Void) async throws -> String
    {
        try await engine.generateText(modelPath: modelPath, system: system, user: user, imagePaths: imagePaths,
                                      options: options, onText: onText)
    }
}

/// Engine events of queue items, handed to the main actor at most once per display frame.
typealias EventCoalescer = Coalescer<(UUID, EngineEvent)>

extension Coalescer where Event == (UUID, EngineEvent) {
    /// Progress events of the same node replace each other while waiting (only the latest step matters), keeping a
    /// preview if either has one; everything else stays in order.
    static func engineEvents(_ handler: @escaping @MainActor ([(UUID, EngineEvent)]) -> Void) -> EventCoalescer {
        EventCoalescer(merge: { last, new in
            guard last.0 == new.0, case .progress(let n1, _, _, let p1) = last.1,
                  case .progress(let n2, let step, let total, let p2) = new.1, n1 == n2 else { return nil }
            return (new.0, .progress(node: n2, step: step, total: total, preview: p2 ?? p1))
        }, deliver: { batch in MainActor.assumeIsolated { handler(batch) } })
    }
}
