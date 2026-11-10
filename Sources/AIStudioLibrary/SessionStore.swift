import Foundation
import GraphRuntime

/// The open tabs, restored on the next launch.
public struct SessionManifest: Codable, Equatable, Sendable {
    public struct Tab: Codable, Equatable, Sendable {
        public var id: UUID
        public var name: String
        /// The workflow file the tab was opened from or saved to (nil: never saved).
        public var fileURL: URL?
        /// The preset it was started from, if any.
        public var presetID: String?
        /// Has changes not written to `fileURL` (untitled tabs: changes since they were created).
        public var isDirty: Bool

        public init(id: UUID, name: String, fileURL: URL? = nil, presetID: String? = nil, isDirty: Bool = false) {
            self.id = id
            self.name = name
            self.fileURL = fileURL
            self.presetID = presetID
            self.isDirty = isDirty
        }
    }

    public var tabs: [Tab]
    public var activeTab: UUID?

    public init(tabs: [Tab] = [], activeTab: UUID? = nil) {
        self.tabs = tabs
        self.activeTab = activeTab
    }
}

/// `session.json` (the tab list) and `drafts/<tab id>.json` (each tab's graph as it was last seen), written
/// atomically so a crash or a forced quit loses at most the last couple of seconds.
public struct SessionStore: Sendable {
    public let directory: URL

    public init(directory: URL) { self.directory = directory }

    var manifestURL: URL { directory.appendingPathComponent("session.json") }
    var draftsDirectory: URL { directory.appendingPathComponent("drafts", isDirectory: true) }
    func draftURL(_ id: UUID) -> URL { draftsDirectory.appendingPathComponent("\(id.uuidString).json") }

    /// The saved tabs and the drafts that could be read. Tabs whose draft is missing are kept only if their file still
    /// opens; the caller decides.
    public func load() -> (manifest: SessionManifest, drafts: [UUID: Graph]) {
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(SessionManifest.self, from: data) else {
            return (SessionManifest(), [:])
        }
        var drafts: [UUID: Graph] = [:]
        for tab in manifest.tabs {
            if let g = try? Graph.load(draftURL(tab.id)) { drafts[tab.id] = g }
        }
        return (manifest, drafts)
    }

    public func save(_ manifest: SessionManifest) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try e.encode(manifest).write(to: manifestURL, options: .atomic)
    }

    public func writeDraft(_ graph: Graph, id: UUID) throws {
        try FileManager.default.createDirectory(at: draftsDirectory, withIntermediateDirectories: true)
        try graph.jsonData(pretty: false).write(to: draftURL(id), options: .atomic)
    }

    public func removeDraft(id: UUID) {
        try? FileManager.default.removeItem(at: draftURL(id))
    }

    /// Deletes drafts of tabs that are no longer open.
    public func prune(keeping ids: Set<UUID>) {
        let files = (try? FileManager.default.contentsOfDirectory(at: draftsDirectory, includingPropertiesForKeys: nil)) ?? []
        for f in files where f.pathExtension == "json" {
            guard let id = UUID(uuidString: f.deletingPathExtension().lastPathComponent), !ids.contains(id) else { continue }
            try? FileManager.default.removeItem(at: f)
        }
    }
}
