import AppKit
import Foundation
import GraphRuntime
import ImageIO
import MLXDiffusionCore
import NodesStandard

/// An XY plot being generated: the axes, the graph it varies, and the cell images received so far.
struct PlotJob {
    let id: UUID
    let spec: PlotSpec
    let base: Graph
    var images: [Int: CGImage] = [:]
}

enum CompareMode: String, CaseIterable, Identifiable {
    case split = "Split"
    case sideBySide = "Side by Side"
    case difference = "Difference"
    var id: String { rawValue }
}

/// Favorites live in an extended attribute on the PNG, so they survive moving the file or the output folder.
enum Favorites {
    static let attribute = "com.mlxcomfy.favorite"

    static func isFavorite(_ url: URL) -> Bool {
        getxattr(url.path, attribute, nil, 0, 0, 0) >= 0
    }

    static func set(_ on: Bool, _ url: URL) {
        if on {
            _ = "1".withCString { setxattr(url.path, attribute, $0, 1, 0, 0) }
        } else {
            removexattr(url.path, attribute, 0)
        }
    }
}

extension AppModel {
    // MARK: styles & wildcards

    static var supportDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLXComfyUI")
    }

    /// `<name>.txt` files here are the `__name__` wildcard lists.
    static var wildcardDirectory: URL { supportDirectory.appendingPathComponent("wildcards") }

    var selectedStyle: PromptStyle? { styles.first { $0.name == selectedStyleName } }

    /// The graph as it will run: the selected style applied to the first Text Encode prompt, then choices and
    /// wildcards expanded in every literal prompt (seeded by the job's seed). The templates are kept in
    /// `metadata` so the saved PNG records how its prompt was made.
    func prepared(_ graph: Graph, seed: Int) -> Graph {
        var g = graph
        let wildcards = PromptTemplate.wildcardFiles(in: Self.wildcardDirectory)
        let first = g.nodes.first { $0.type == "TextEncode" }?.id
        for n in g.nodes where n.type == "TextEncode" || n.type == "TextToAudio" || n.type == "TextToMusic" {
            let field = n.type == "TextToMusic" ? "tags" : "prompt"
            guard case .value(.string(let raw))? = n.inputs[field] else { continue }
            var text = raw
            if n.id == first, let style = selectedStyle { text = style.apply(to: text) }
            let expanded = PromptTemplate.expand(text, seed: UInt64(truncatingIfNeeded: seed), wildcards: wildcards)
            guard expanded != raw else { continue }
            try? g.setValue(.string(expanded), node: n.id, input: field)
            var meta = g.metadata ?? [:]
            meta["prompt_template.\(n.id)"] = raw
            if n.id == first, let style = selectedStyle { meta["prompt_style"] = style.name }
            g.metadata = meta
        }
        return g
    }

    // MARK: XY plot

    /// Literal inputs of the current graph an axis can vary: (node, input, type, label).
    var plotCandidates: [(node: String, input: String, type: SocketType, label: String)] {
        graph.nodes.flatMap { n -> [(String, String, SocketType, String)] in
            guard let schema = schema(for: n), !schema.isOutput else { return [] }
            return schema.inputs.compactMap { spec in
                guard spec.type.isScalar, spec.widget != .filePath else { return nil }
                if case .link? = n.inputs[spec.name] { return nil }
                return (n.id, spec.name, spec.type, "\(n.title ?? schema.displayName) › \(spec.name)")
            }
        }
    }

    /// Queues one job per cell. Cells preview instead of saving; the finished grid is saved as one PNG.
    func enqueuePlot(_ spec: PlotSpec) {
        if let err = validationError { lastError = err; NSSound.beep(); return }
        lastError = nil
        let id = UUID()
        let base = graph
        plots[id] = PlotJob(id: id, spec: spec, base: base)
        for row in 0..<spec.rows {
            for col in 0..<spec.columns {
                var g = spec.graph(for: base, column: col, row: row)
                for n in g.nodes where n.type == "SaveImage" {
                    g[n.id] = GraphNode(id: n.id, type: "PreviewImage", title: n.title,
                                        inputs: ["image": n.inputs["image"] ?? .value(.string(""))],
                                        position: n.position)
                }
                let sampler = g.nodes.first { $0.type == "Sampler" }
                let seed: Int? = { if case .value(.int(let s))? = sampler?.inputs["seed"] { return s }; return nil }()
                g = prepared(g, seed: seed ?? 0)
                let index = row * spec.columns + col
                queue.append(QueueItem(graph: g, title: spec.label(column: col, row: row), seed: seed,
                                       groupID: id, cell: index))
            }
        }
        processQueue()
    }

    func recordPlotCell(_ id: UUID, cell: Int, image: CGImage?) {
        if let image { plots[id]?.images[cell] = image }
        completePlotsIfDone()
    }

    /// Composes and saves every plot whose cells have all finished (or were removed from the queue).
    func completePlotsIfDone() {
        for (id, job) in plots {
            let open = queue.contains { $0.groupID == id && !$0.isFinished }
            guard !open else { continue }
            plots[id] = nil
            guard !job.images.isEmpty else { continue }
            let grid = GridComposer.compose(job.spec, images: job.images)
            var g = job.base
            var meta = g.metadata ?? [:]
            if let d = try? JSONEncoder().encode(job.spec), let s = String(data: d, encoding: .utf8) {
                meta["xy_plot"] = s
            }
            g.metadata = meta
            let dir = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath)
            let url = Self.nextFilename(in: dir, prefix: "plot")
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try ImageArray.savePNG(grid, to: url, text: [
                    StandardNodes.graphMetadataKey: g.jsonString(), "mlxcomfy.plot": meta["xy_plot"] ?? "",
                ])
                displayedImage = grid
                displayedURL = url
                reloadHistory()
            } catch {
                lastError = "Could not save the plot: \(error)"
                displayedImage = grid
                displayedURL = nil
            }
        }
    }

    /// Progress of a plot for its collapsed queue row.
    func plotProgress(_ id: UUID) -> (done: Int, total: Int, running: Bool) {
        let cells = queue.filter { $0.groupID == id }
        let running = cells.contains { if case .running = $0.status { return true }; return false }
        return (cells.filter(\.isFinished).count, cells.count, running)
    }

    func cancelPlot(_ id: UUID) {
        for i in queue.indices where queue[i].groupID == id && queue[i].status == .pending {
            queue[i].status = .cancelled
        }
        if queue.contains(where: { $0.groupID == id && { if case .running = $0.status { return true }; return false }($0) }) {
            cancelCurrent()
        }
        completePlotsIfDone()
    }

    static func nextFilename(in dir: URL, prefix: String) -> URL {
        let existing = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let used = existing.compactMap { name -> Int? in
            guard name.hasPrefix(prefix + "_"), name.hasSuffix(".png") else { return nil }
            return Int(name.dropFirst(prefix.count + 1).dropLast(4))
        }
        return dir.appendingPathComponent(String(format: "%@_%05d.png", prefix, (used.max() ?? 0) + 1))
    }

    // MARK: history

    /// Reads the prompts of history items not yet cached, off the main thread, then updates the list.
    func loadHistoryPrompts() {
        let missing = history.filter { promptCache[$0.url] == nil }.map(\.url)
        guard !missing.isEmpty else { return }
        Task {
            let found = await Task.detached(priority: .utility) { () -> [URL: String] in
                var out: [URL: String] = [:]
                for url in missing {
                    var g: Graph?
                    if url.pathExtension.lowercased() == "png" {
                        g = AppModel.embeddedGraph(in: url)
                    } else if let json = await MediaIO.readMetadata(url)[StandardNodes.graphMetadataKey] {
                        g = try? JSONDecoder().decode(Graph.self, from: Data(json.utf8))
                    }
                    if let n = g?.nodes.first(where: { ["TextEncode", "TextToAudio", "TextToMusic"].contains($0.type) }),
                       case .value(.string(let s))? = n.inputs[n.type == "TextToMusic" ? "tags" : "prompt"] {
                        out[url] = s
                    } else {
                        out[url] = ""
                    }
                }
                return out
            }.value
            promptCache.merge(found) { _, new in new }
            for i in history.indices where history[i].prompt == nil {
                history[i].prompt = promptCache[history[i].url]
            }
        }
    }

    func toggleFavorite(_ item: HistoryItem) {
        Favorites.set(!item.isFavorite, item.url)
        if let i = history.firstIndex(where: { $0.url == item.url }) { history[i].isFavorite.toggle() }
    }

    // MARK: compare

    /// Shows `url` against the displayed image.
    func compare(with url: URL) {
        guard displayedImage != nil, url != displayedURL,
              let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { NSSound.beep(); return }
        compareURL = url
        compareImage = img
        workspace = .image
    }

    func endCompare() {
        compareURL = nil
        compareImage = nil
    }
}
