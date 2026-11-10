import Foundation
import MLXDiffusionCore

/// Models and LoRAs added from links (Civitai, Hugging Face) or files, kept in `<models>/library.json`. The app's
/// Models window manages it; the LoRA nodes (and the CLI, and a Mac server) read it to find a LoRA by its id.
public struct LibraryEntry: Codable, Identifiable, Sendable, Equatable {
    public var id: String
    public var kind: LoRARemote.Kind
    /// The app's family (`flux.1`, `qwen-image`, `qwen-image-2.1`, `wan-2.2-t2v`, `wan-2.2-i2v`); nil when unknown.
    public var family: String?
    public var title: String
    /// The link it was added from (nil for a file added from disk).
    public var link: String?
    /// The resolved file behind the link: download URL, SHA-256, size, base model, trigger words, sample image.
    public var source: LoRARemote?
    public var fileName: String
    public var sha256: String?
    public var bytes: Int64?
    public var baseModel: String?
    public var triggerWords: [String]
    /// The sample image, cached under `<models>/library/thumbs` (relative to the models folder).
    public var thumbnail: String?
    /// Where it's installed on this Mac, relative to the models folder ("loras/…"); nil when it isn't.
    public var localFile: String?
    public var added: Date

    public init(id: String, kind: LoRARemote.Kind, family: String?, title: String, link: String?, source: LoRARemote?,
                fileName: String, sha256: String?, bytes: Int64?, baseModel: String?, triggerWords: [String],
                thumbnail: String? = nil, localFile: String? = nil, added: Date = Date()) {
        self.id = id
        self.kind = kind
        self.family = family
        self.title = title
        self.link = link
        self.source = source
        self.fileName = fileName
        self.sha256 = sha256
        self.bytes = bytes
        self.baseModel = baseModel
        self.triggerWords = triggerWords
        self.thumbnail = thumbnail
        self.localFile = localFile
        self.added = added
    }

    /// The name a GPU server stores it under (uploads and server downloads are named by SHA-256).
    public var serverName: String? { sha256.map { $0 + ".lora" } }
}

public enum Library {
    public static let changed = Notification.Name("MLXComfyLibraryChanged")
    private static let lock = NSLock()

    public static func file(models: URL = ModelPaths.directory) -> URL { models.appendingPathComponent("library.json") }

    public static func load(models: URL = ModelPaths.directory) -> [LibraryEntry] {
        lock.withLock {
            guard let data = try? Data(contentsOf: file(models: models)) else { return [] }
            let d = JSONDecoder()
            d.dateDecodingStrategy = .iso8601
            return (try? d.decode([LibraryEntry].self, from: data)) ?? []
        }
    }

    public static func save(_ items: [LibraryEntry], models: URL = ModelPaths.directory) throws {
        try lock.withLock {
            try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
            let e = JSONEncoder()
            e.outputFormatting = [.prettyPrinted, .sortedKeys]
            e.dateEncodingStrategy = .iso8601
            try e.encode(items).write(to: file(models: models), options: .atomic)
        }
        NotificationCenter.default.post(name: changed, object: nil)
    }

    /// Changes one item (read, modify, write), so the app and a running node don't overwrite each other.
    public static func update(_ id: String, models: URL = ModelPaths.directory, _ change: (inout LibraryEntry) -> Void) throws {
        var items = load(models: models)
        guard let i = items.firstIndex(where: { $0.id == id }) else { return }
        change(&items[i])
        try save(items, models: models)
    }

    public static func item(_ id: String, models: URL = ModelPaths.directory) -> LibraryEntry? {
        load(models: models).first { $0.id == id }
    }

    /// The installed file on this Mac, when it's there.
    public static func localURL(_ item: LibraryEntry, models: URL = ModelPaths.directory) -> URL? {
        guard let rel = item.localFile else { return nil }
        let url = models.appendingPathComponent(rel)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The other Wan 2.2 expert's file of a pair, next to the installed one.
    public static func companionURL(_ item: LibraryEntry, models: URL = ModelPaths.directory) -> URL? {
        guard let c = item.source?.companion, let first = localURL(item, models: models) else { return nil }
        let url = first.deletingLastPathComponent().appendingPathComponent(c.cacheName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// A new id: the title as a slug, unique among `existing`.
    public static func newID(for title: String, existing: [LibraryEntry]) -> String {
        let base = String(title.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" })
            .split(separator: "-").joined(separator: "-").prefix(48)
        var id = base.isEmpty ? "item" : String(base)
        var n = 2
        while existing.contains(where: { $0.id == id }) { id = "\(base)-\(n)"; n += 1 }
        return id
    }

    /// Installs a LoRA on this Mac: downloads it into `<models>/loras` (or finds it there) and records it.
    @discardableResult
    public static func installLocally(_ id: String, models: URL = ModelPaths.directory,
                                      status: @escaping @Sendable (String) -> Void) async throws -> URL {
        guard let item = item(id, models: models) else { throw LibraryError.missing(id) }
        if let url = localURL(item, models: models), item.source?.companion == nil || companionURL(item, models: models) != nil {
            return url
        }
        guard let source = item.source else { throw LibraryError.notOnThisMac(item.title) }
        let url = try await LoRADownloads.fetch(source, models: models, status: status)
        try update(id, models: models) { $0.localFile = "loras/" + url.lastPathComponent }
        return url
    }
}

public enum LibraryError: Error, CustomStringConvertible, Equatable {
    case missing(String)
    case notOnThisMac(String)
    case noneChosen
    case wanExpertUnknown(String)

    public var description: String {
        switch self {
        case .missing(let id): return "\(id) isn't in the library any more: pick another LoRA (Models window)"
        case .notOnThisMac(let t): return "\(t) isn't on this Mac and has no link to download it from"
        case .noneChosen: return "Pick a LoRA in the node's menu (or add one: Models window › Add from Link)"
        case .wanExpertUnknown(let f):
            return "\(f): can't tell which Wan 2.2 expert this is (high or low noise); the file name should say"
        }
    }
}
