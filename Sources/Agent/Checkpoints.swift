import Foundation

/// Copies of files from before the agent changed them, filed by the user message whose reply changed them, so the
/// user can rewind. Saved in `folder` (`index.json` and `blobs/`).
public actor CheckpointStore {
    /// One saved file: its path, the user message (turn) it was saved for, and its blob (nil: it didn't exist).
    public struct Entry: Codable, Equatable, Sendable {
        public var path: String
        public var turn: UUID
        /// The blob's file name in `blobs/`; nil when the file didn't exist before the change.
        public var blob: String?
        /// Larger than `maxBytes`: not saved, can't be restored.
        public var tooLarge: Bool
    }

    public static let maxBytes = 5 * 1024 * 1024
    public let folder: URL
    private var list: [Entry] = []

    public init(folder: URL) {
        self.folder = folder
        if let data = try? Data(contentsOf: folder.appendingPathComponent("index.json")),
           let loaded = try? JSONDecoder().decode([Entry].self, from: data) {
            list = loaded
        }
    }

    private var indexURL: URL { folder.appendingPathComponent("index.json") }
    private var blobsFolder: URL { folder.appendingPathComponent("blobs") }

    private func writeIndex() throws {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try JSONEncoder().encode(list).write(to: indexURL, options: .atomic)
    }

    /// Before the first change to `url` for `turn`: saves its bytes (or that it didn't exist). Later calls for the same
    /// path and turn do nothing.
    public func save(_ url: URL, turn: UUID) throws {
        let path = url.standardizedFileURL.path
        if list.contains(where: { $0.path == path && $0.turn == turn }) { return }
        let fm = FileManager.default
        var entry = Entry(path: path, turn: turn, blob: nil, tooLarge: false)
        if fm.fileExists(atPath: path) {
            let size = (try fm.attributesOfItem(atPath: path)[.size] as? NSNumber)?.intValue ?? 0
            if size > Self.maxBytes {
                entry.tooLarge = true
            } else {
                try fm.createDirectory(at: blobsFolder, withIntermediateDirectories: true)
                let name = UUID().uuidString + ".bin"
                try Data(contentsOf: URL(fileURLWithPath: path)).write(to: blobsFolder.appendingPathComponent(name), options: .atomic)
                entry.blob = name
            }
        }
        list.append(entry)
        try writeIndex()
    }

    /// Every entry, oldest first.
    public func entries() -> [Entry] { list }

    /// The files changed for any of `turns`, each once, sorted by path.
    public func changedPaths(in turns: Set<UUID>) -> [URL] {
        Set(list.filter { turns.contains($0.turn) }.map(\.path)).sorted().map { URL(fileURLWithPath: $0) }
    }

    /// Puts every file changed for any of `turns` back as it was before the oldest of those changes. Returns what was
    /// restored and what couldn't be (too large).
    @discardableResult
    public func restore(turns: Set<UUID>) throws -> (restored: [URL], skipped: [URL]) {
        let fm = FileManager.default
        let chosen = list.filter { turns.contains($0.turn) }
        var seen = Set<String>()
        var restored: [String] = []
        var skipped: [String] = []
        for e in chosen where seen.insert(e.path).inserted {
            let url = URL(fileURLWithPath: e.path)
            if e.tooLarge {
                skipped.append(e.path)
            } else if let blob = e.blob {
                let data = try Data(contentsOf: blobsFolder.appendingPathComponent(blob))
                try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                restored.append(e.path)
            } else {
                if fm.fileExists(atPath: e.path) { try fm.removeItem(at: url) }
                restored.append(e.path)
            }
        }
        list.removeAll { turns.contains($0.turn) }
        for e in chosen { if let b = e.blob { try? fm.removeItem(at: blobsFolder.appendingPathComponent(b)) } }
        try writeIndex()
        return (restored.sorted().map { URL(fileURLWithPath: $0) }, skipped.sorted().map { URL(fileURLWithPath: $0) })
    }
}
