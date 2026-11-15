import Foundation
import GraphRuntime
import MovieCore

/// Runs a movie's graphs (a template with a shot's values bound in) on this Mac's engine, whose cloud GPU nodes reach
/// the server. It works outside the workflow queue, which would put the results into whichever tab is in front, and
/// runs one graph at a time so jobs for one GPU family don't race for its single container.
@MainActor
final class MovieRenderer {
    private var last: Task<Void, Never>?

    func run(_ graph: Graph, on backend: ExecutionBackend, title: String,
             onEvent: @escaping @MainActor (EngineEvent) -> Void = { _ in }) async throws -> [EngineOutput] {
        let previous = last
        let job = Task<[EngineOutput], Error> { @MainActor in
            await previous?.value
            try Task.checkCancellation()
            return try await backend.run(graph, workflow: title) { event in
                Task { @MainActor in onEvent(event) }
            }
        }
        last = Task { _ = try? await job.value }
        return try await withTaskCancellationHandler {
            try await job.value
        } onCancel: {
            job.cancel()
        }
    }
}

extension AppModel {
    /// The template a movie renders a stage with: the user's edited copy in the movie's `templates/` folder
    /// (`<name>.json`, the graph opened as a workflow, plus `<name>.slots.json`) when it's still usable, else the
    /// built-in `fallback`.
    func movieTemplate(_ name: String, for doc: MovieDocument, fallback: () -> MovieTemplate) -> MovieTemplate {
        let dir = movieStore.folder(doc.id, .templates)
        if let graph = try? Graph.load(dir.appendingPathComponent(name + ".json")),
           let data = try? Data(contentsOf: dir.appendingPathComponent(name + ".slots.json")),
           let slots = try? JSONDecoder().decode([String: MovieTemplate.Slot].self, from: data) {
            let t = MovieTemplate(name: name, graph: graph, slots: slots)
            if t.brokenSlots.isEmpty { return t }
        }
        return fallback()
    }

    /// Opens a stage's template as a workflow to edit. The movie keeps its own copy, so editing it (and saving with
    /// ⌘S) changes only this movie's renders.
    func editMovieTemplate(_ name: String, for doc: MovieDocument, fallback: () -> MovieTemplate) {
        let template = movieTemplate(name, for: doc, fallback: fallback)
        let dir = movieStore.folder(doc.id, .templates)
        let graphURL = dir.appendingPathComponent(name + ".json")
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let e = JSONEncoder()
            e.outputFormatting = [.prettyPrinted, .sortedKeys]
            if !FileManager.default.fileExists(atPath: graphURL.path) {
                try e.encode(template.graph).write(to: graphURL, options: .atomic)
            }
            try e.encode(template.slots).write(to: dir.appendingPathComponent(name + ".slots.json"), options: .atomic)
            open(graphURL)
        } catch {
            doc.error = "Couldn't open the template: \(error.localizedDescription)"
        }
    }
}
