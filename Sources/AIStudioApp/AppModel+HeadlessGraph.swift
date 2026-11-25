import Foundation
import GraphRuntime
import MovieCore
import NodesRemote

/// Finished Cloud GPU jobs heard while one render runs.
final class JobRecords: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [RemoteJobRecord] = []
    func append(_ r: RemoteJobRecord) { lock.withLock { records.append(r) } }
    var all: [RemoteJobRecord] { lock.withLock { records } }
}

extension AppModel {
    /// Binds and runs a template on this Mac's graph runtime (its Cloud GPU nodes go to the server), calling
    /// `progress` with "step 3/30" or a node's status; returns the outputs and the GPU jobs it ran.
    func runHeadlessGraph(template: MovieTemplate, render: MovieArt.Render, title: String,
                          progress: @escaping @MainActor (String) -> Void = { _ in })
        async throws -> (outputs: [EngineOutput], jobs: [RemoteJobRecord])
    {
        let graph = try template.bind(render.values, optional: Set(render.values.keys), remove: render.unusedPictures)
        // Only this graph's GPU jobs (others may run at the same time).
        let jobs = JobRecords()
        let scope = UUID()
        let token = RemoteJobLog.observe(scope: scope) { jobs.append($0) }
        defer { RemoteJobLog.remove(token) }
        let outputs = try await RemoteJobLog.$scope.withValue(scope) {
            try await movieRenderer.run(graph, on: localBackend, title: title) { event in
                if case .progress(_, let step, let total, _) = event, total > 0 {
                    progress("step \(step)/\(total)")
                } else if case .status(_, let text?) = event, !text.isEmpty, !text.hasPrefix("GPU ") {
                    progress(text)
                }
            }
        }
        return (outputs, jobs.all)
    }

    /// Adds the jobs' times to the learned GPU timings; returns their total cost (nil when none reported one).
    /// `work`: as recordJobs' (a job's size when it isn't its picture's megapixels).
    func recordGPUJobs(_ records: [RemoteJobRecord], work: Double? = nil) -> Double? {
        var cost: Double?
        var timings = GPUTimings.load(gpuTimingsURL)
        for r in records {
            if let c = r.costUSD { cost = (cost ?? 0) + c }
            let mp = work ?? Double((r.width ?? 0) * (r.height ?? 0)) / 1e6
            timings.record(task: r.task, megapixels: mp > 0 ? mp : 1, gpuSeconds: r.gpuSeconds, loadSeconds: r.loadSeconds)
        }
        try? timings.save(gpuTimingsURL)
        return cost
    }
}
