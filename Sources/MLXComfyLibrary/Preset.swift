import Foundation
import GraphRuntime

/// A ready-made workflow to start from. Built-in presets ship with the app (`Presets/*.json` in this module);
/// your own live in the library's `presets/` folder.
///
/// Graphs are stored portable: a Load Model `path` is the model's folder name in the models folder (e.g.
/// `flux-1-dev`) and an output `directory` is ``outputPlaceholder``. ``instantiate(resolveModel:outputDirectory:)``
/// fills both in for this Mac.
public struct Preset: Codable, Identifiable, Equatable, Sendable {
    public enum Category: String, Codable, CaseIterable, Sendable, Identifiable {
        case image, video, sound, music, speech
        public var id: String { rawValue }

        public var title: String {
            switch self {
            case .image: return "Image"
            case .video: return "Video"
            case .sound: return "Sound"
            case .music: return "Music"
            case .speech: return "Speech"
            }
        }

        public var symbol: String {
            switch self {
            case .image: return "photo"
            case .video: return "film"
            case .sound: return "speaker.wave.2"
            case .music: return "music.note"
            case .speech: return "waveform"
            }
        }
    }

    public static let outputPlaceholder = "{output}"

    public var id: String
    public var name: String
    public var category: Category
    /// Model catalog ids the graph needs (all of them must be installed).
    public var models: [String]
    /// The model's display name on the card, e.g. "FLUX.1 [dev]".
    public var modelName: String
    public var summary: String
    /// SF Symbol for the card when there is no thumbnail.
    public var symbol: String
    /// Sort order among built-ins (lower first).
    public var order: Int
    public var graph: Graph
    /// File name of a thumbnail next to a user preset's JSON.
    public var thumbnail: String?
    /// Set when loaded, not stored.
    public var isBuiltIn: Bool = false
    public var fileURL: URL? = nil

    enum CodingKeys: String, CodingKey {
        case id, name, category, models, modelName, summary, symbol, order, graph, thumbnail
    }

    public init(id: String, name: String, category: Category, models: [String], modelName: String, summary: String,
                symbol: String, order: Int = 1000, graph: Graph, thumbnail: String? = nil) {
        self.id = id
        self.name = name
        self.category = category
        self.models = models
        self.modelName = modelName
        self.summary = summary
        self.symbol = symbol
        self.order = order
        self.graph = graph
        self.thumbnail = thumbnail
    }

    /// The preset's graph for this Mac: model folder names resolved to paths (`resolveModel` returns nil to keep the
    /// value) and the output placeholder replaced.
    public func instantiate(resolveModel: (String) -> String?, outputDirectory: String) -> Graph {
        var g = graph
        for i in g.nodes.indices {
            for (name, input) in g.nodes[i].inputs {
                guard case .value(.string(let s)) = input else { continue }
                if g.nodes[i].type == "LoadModel", name == "path", let p = resolveModel(s) {
                    g.nodes[i].inputs[name] = .value(.string(p))
                } else if name == "directory", s == Self.outputPlaceholder {
                    g.nodes[i].inputs[name] = .value(.string(outputDirectory))
                }
            }
        }
        return g
    }

    /// The reverse of ``instantiate(resolveModel:outputDirectory:)`` for saving a workflow as a preset: model paths
    /// become folder names where `folderName` knows them, and directories inside the output folder the placeholder.
    public static func portable(_ graph: Graph, folderName: (String) -> String?, outputDirectory: String) -> Graph {
        var g = graph
        let out = URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath).standardizedFileURL.path
        for i in g.nodes.indices {
            for (name, input) in g.nodes[i].inputs {
                guard case .value(.string(let s)) = input else { continue }
                if g.nodes[i].type == "LoadModel", name == "path", let f = folderName(s) {
                    g.nodes[i].inputs[name] = .value(.string(f))
                } else if name == "directory" {
                    let p = URL(fileURLWithPath: (s as NSString).expandingTildeInPath).standardizedFileURL.path
                    if p == out || p.hasPrefix(out + "/") { g.nodes[i].inputs[name] = .value(.string(outputPlaceholder)) }
                }
            }
        }
        return g
    }

    /// True when every model the preset needs is in `installed` (catalog ids).
    public func isAvailable(installed: Set<String>) -> Bool { Set(models).isSubset(of: installed) }
}

public enum PresetCatalog {
    /// The presets bundled with the app, in display order.
    public static func builtIn() -> [Preset] {
        guard let dir = Bundle.module.url(forResource: "Presets", withExtension: nil) else { return [] }
        return load(from: dir, builtIn: true).sorted { ($0.order, $0.name) < ($1.order, $1.name) }
    }

    /// Your presets in `dir`, by file name.
    public static func user(in dir: URL) -> [Preset] {
        load(from: dir, builtIn: false).sorted {
            let a = $0.fileURL?.deletingPathExtension().lastPathComponent ?? $0.name
            let b = $1.fileURL?.deletingPathExtension().lastPathComponent ?? $1.name
            return a.localizedStandardCompare(b) == .orderedAscending
        }
    }

    static func load(from dir: URL, builtIn: Bool) -> [Preset] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.compactMap { url in
            guard let data = try? Data(contentsOf: url), var p = try? JSONDecoder().decode(Preset.self, from: data)
            else { return nil }
            p.isBuiltIn = builtIn
            p.fileURL = url
            return p
        }
    }

    /// The thumbnail file of a user preset, if it has one.
    public static func thumbnailURL(_ preset: Preset) -> URL? {
        guard let t = preset.thumbnail, let file = preset.fileURL else { return nil }
        return file.deletingLastPathComponent().appendingPathComponent(t)
    }

    /// Writes a user preset (and its PNG thumbnail) into `dir`; the id is derived from the name and made unique.
    @discardableResult
    public static func save(_ preset: Preset, thumbnailPNG: Data?, in dir: URL) throws -> URL {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var p = preset
        let store = WorkflowStore(root: dir)
        let url = store.uniqueURL(name: preset.name, in: dir)
        p.id = "user." + url.deletingPathExtension().lastPathComponent
        if let thumbnailPNG {
            let thumb = url.deletingPathExtension().lastPathComponent + ".png"
            try thumbnailPNG.write(to: dir.appendingPathComponent(thumb), options: .atomic)
            p.thumbnail = thumb
        }
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try e.encode(p).write(to: url, options: .atomic)
        return url
    }

    /// Removes a user preset and its thumbnail.
    public static func delete(_ preset: Preset) throws {
        guard !preset.isBuiltIn, let url = preset.fileURL else { return }
        if let t = thumbnailURL(preset) { try? FileManager.default.removeItem(at: t) }
        try FileManager.default.removeItem(at: url)
    }
}
