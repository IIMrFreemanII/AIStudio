import Foundation

/// Why a character could not be saved.
public enum CharacterError: Error, Equatable, LocalizedError {
    case invalid([String])
    public var errorDescription: String? {
        switch self {
        case .invalid(let problems): problems.joined(separator: " ")
        }
    }
}

/// Characters on disk: `<root>/<uuid>/character.json`, `memory.json`, `portraits/` and `album/`.
public struct CharacterStore: Sendable {
    public let root: URL
    public init(root: URL) { self.root = root }

    public static var encoder: JSONEncoder {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }

    public static var decoder: JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    public func folder(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }
    public func portraitsFolder(_ id: UUID) -> URL { folder(id).appendingPathComponent("portraits", isDirectory: true) }
    public func albumFolder(_ id: UUID) -> URL { folder(id).appendingPathComponent("album", isDirectory: true) }

    /// Every character, newest `updated` first; unreadable ones are skipped.
    public func list() -> [Character] {
        let dirs = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return dirs.compactMap { dir in
            guard let id = UUID(uuidString: dir.lastPathComponent) else { return nil }
            return load(id)
        }.sorted { $0.updated > $1.updated }
    }

    public func load(_ id: UUID) -> Character? {
        guard let data = try? Data(contentsOf: folder(id).appendingPathComponent("character.json")) else { return nil }
        return try? Self.decoder.decode(Character.self, from: data)
    }

    /// Throws `CharacterError.invalid` when `AgeGuard` finds problems; else writes character.json atomically.
    public func save(_ c: Character) throws {
        let problems = AgeGuard.problems(c)
        if !problems.isEmpty { throw CharacterError.invalid(problems) }
        try FileManager.default.createDirectory(at: folder(c.id), withIntermediateDirectories: true)
        try Self.encoder.encode(c).write(to: folder(c.id).appendingPathComponent("character.json"), options: .atomic)
    }

    /// Removes the whole folder.
    public func delete(_ id: UUID, useTrash: Bool) {
        let url = folder(id)
        if useTrash {
            try? FileManager.default.trashItem(at: url, resultingItemURL: nil)
        } else {
            try? FileManager.default.removeItem(at: url)
        }
    }

    public func memory(_ id: UUID) -> CharacterMemory {
        guard let data = try? Data(contentsOf: folder(id).appendingPathComponent("memory.json")),
              let m = try? Self.decoder.decode(CharacterMemory.self, from: data) else { return CharacterMemory() }
        return m
    }

    public func saveMemory(_ m: CharacterMemory, for id: UUID) throws {
        try FileManager.default.createDirectory(at: folder(id), withIntermediateDirectories: true)
        try Self.encoder.encode(m).write(to: folder(id).appendingPathComponent("memory.json"), options: .atomic)
    }

    /// The built-ins followed by the custom ones from `<root>/scenarios.json`.
    public func scenarios() -> [Scenario] {
        let custom = (try? Data(contentsOf: root.appendingPathComponent("scenarios.json")))
            .flatMap { try? Self.decoder.decode([Scenario].self, from: $0) } ?? []
        return Scenario.builtIns + custom
    }

    public func saveCustomScenarios(_ s: [Scenario]) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Self.encoder.encode(s).write(to: root.appendingPathComponent("scenarios.json"), options: .atomic)
    }
}
