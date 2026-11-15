import Foundation

/// A movie in the list, without loading all of it.
public struct MovieSummary: Identifiable, Sendable, Equatable {
    public var id: UUID
    public var title: String
    public var stage: MovieStage
    public var shots: Int
    public var modified: Date
}

/// Movies on disk: `<library>/movies/<id>/movie.json`, with the generated files in folders beside it.
public struct MovieStore: Sendable {
    public let root: URL

    /// The folders a movie keeps its files in.
    public enum Folder: String, CaseIterable, Sendable {
        case cast, keyframes, shots, audio, templates, renders
    }

    public init(root: URL) { self.root = root }

    public func folder(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }

    public func folder(_ id: UUID, _ sub: Folder) -> URL {
        folder(id).appendingPathComponent(sub.rawValue, isDirectory: true)
    }

    public func file(_ id: UUID) -> URL { folder(id).appendingPathComponent("movie.json") }

    /// A take's file, from its path relative to the movie's folder.
    public func url(_ id: UUID, relative: String) -> URL { folder(id).appendingPathComponent(relative) }

    private static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .deferredToDate  // exact (reference-date seconds): a reload compares equal
        return e
    }()

    private static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .deferredToDate
        return d
    }()

    public func save(_ movie: MovieProject) throws {
        try FileManager.default.createDirectory(at: folder(movie.id), withIntermediateDirectories: true)
        try Self.encoder.encode(movie).write(to: file(movie.id), options: .atomic)
    }

    public func load(_ id: UUID) throws -> MovieProject {
        try Self.decoder.decode(MovieProject.self, from: Data(contentsOf: file(id)))
    }

    /// Every movie, newest change first. Folders without a readable `movie.json` are skipped.
    public func list() -> [MovieSummary] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir -> MovieSummary? in
            guard let id = UUID(uuidString: dir.lastPathComponent), let m = try? load(id) else { return nil }
            return MovieSummary(id: m.id, title: m.title, stage: m.currentStage, shots: m.shots.count, modified: m.modified)
        }
        .sorted { $0.modified > $1.modified }
    }

    /// Moves the movie's folder to the Trash (or deletes it when there's no Trash, as in tests).
    public func delete(_ id: UUID, useTrash: Bool = true) throws {
        let dir = folder(id)
        guard FileManager.default.fileExists(atPath: dir.path) else { return }
        if useTrash { try FileManager.default.trashItem(at: dir, resultingItemURL: nil) }
        else { try FileManager.default.removeItem(at: dir) }
    }
}
