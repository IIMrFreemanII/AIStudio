import Foundation
import GraphRuntime
import MovieCore
import NodesRemote

/// Runs a movie's graphs (a template with a shot's values bound in) on this Mac's engine, whose cloud GPU nodes reach
/// the server. It works outside the workflow queue, which would put the results into whichever tab is in front.
///
/// Several graphs can run at once (the movie sends as many jobs together as the server runs in parallel), but they
/// take turns with this Mac's side of the work, which isn't safe to run side by side: a graph holds `gate` while it
/// runs, and its GPU nodes let go of it while they wait on the server, so only the GPU jobs overlap.
@MainActor
final class MovieRenderer {
    /// The most renders the movie runs at once, whatever the server allows.
    static let maxParallel = 4

    private let gate = AsyncGate()

    func run(_ graph: Graph, on backend: ExecutionBackend, title: String,
             onEvent: @escaping @MainActor (EngineEvent) -> Void = { _ in }) async throws -> [EngineOutput] {
        await gate.acquire()
        do {
            try Task.checkCancellation()
            let outputs = try await RemoteWait.$gate.withValue(gate) {
                try await backend.run(graph, workflow: title) { event in
                    Task { @MainActor in onEvent(event) }
                }
            }
            await gate.release()
            return outputs
        } catch {
            await gate.release()
            throw error
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
