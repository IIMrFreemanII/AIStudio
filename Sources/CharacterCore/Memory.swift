import Foundation

/// What a character remembers about the user and how they get along.
public struct CharacterMemory: Codable, Equatable, Sendable {
    public struct Fact: Codable, Identifiable, Equatable, Sendable, Hashable {
        public var id: UUID
        public var text: String
        public var date: Date
        public var chatID: UUID?
        public init(id: UUID = UUID(), text: String, date: Date = Date(), chatID: UUID? = nil) {
            self.id = id
            self.text = text
            self.date = date
            self.chatID = chatID
        }
    }

    public enum Stage: String, Codable, Sendable, CaseIterable {
        case strangers, flirting, dating, partners
        public var label: String {
            switch self {
            case .strangers: "Strangers"
            case .flirting: "Flirting"
            case .dating: "Dating"
            case .partners: "Partners"
            }
        }
    }

    public struct Relationship: Codable, Equatable, Sendable {
        /// 0...100.
        public var affection = 20
        public var stage: Stage = .strangers
        public var mood = "curious"
        public var lastSeen: Date?
        public init() {}

        enum CodingKeys: String, CodingKey { case affection, stage, mood, lastSeen }
        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            affection = try c.decodeIfPresent(Int.self, forKey: .affection) ?? 20
            stage = try c.decodeIfPresent(Stage.self, forKey: .stage) ?? .strangers
            mood = try c.decodeIfPresent(String.self, forKey: .mood) ?? "curious"
            lastSeen = try c.decodeIfPresent(Date.self, forKey: .lastSeen)
        }
    }

    public var facts: [Fact] = []
    public var relationship = Relationship()
    public var summary = ""
    public var album: [AlbumItem] = []
    public var spentUSD: Double = 0
    /// User turns since the last memory pass.
    public var turnsSinceUpdate = 0
    /// When the character sent media on its own initiative (not asked).
    public var ownMedia: [Date] = []
    public init() {}

    public static let maxFacts = 60

    enum CodingKeys: String, CodingKey { case facts, relationship, summary, album, spentUSD, turnsSinceUpdate, ownMedia }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        facts = try c.decodeIfPresent([Fact].self, forKey: .facts) ?? []
        relationship = try c.decodeIfPresent(Relationship.self, forKey: .relationship) ?? Relationship()
        summary = try c.decodeIfPresent(String.self, forKey: .summary) ?? ""
        album = try c.decodeIfPresent([AlbumItem].self, forKey: .album) ?? []
        spentUSD = try c.decodeIfPresent(Double.self, forKey: .spentUSD) ?? 0
        turnsSinceUpdate = try c.decodeIfPresent(Int.self, forKey: .turnsSinceUpdate) ?? 0
        ownMedia = try c.decodeIfPresent([Date].self, forKey: .ownMedia) ?? []
    }

    /// How many unasked media were sent on the same day as `now`.
    public func ownMediaToday(now: Date = Date(), calendar: Calendar = .current) -> Int {
        ownMedia.filter { calendar.isDate($0, inSameDayAs: now) }.count
    }

    /// Records an unasked media and forgets entries older than a week.
    public mutating func noteOwnMedia(at date: Date = Date()) {
        ownMedia.append(date)
        ownMedia.removeAll { date.timeIntervalSince($0) > 7 * 86400 }
    }

    /// Merges a memory pass's result: adds new facts (trimmed, skipping empty ones and case-insensitive duplicates),
    /// removes facts named in `forget`, keeps at most `maxFacts` (dropping the oldest), moves affection by the
    /// clamped delta, and sets stage, mood and summary when given.
    public mutating func apply(_ update: MemoryUpdate, chatID: UUID?, date: Date = Date()) {
        let forgotten = Set(update.forget.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })
        if !forgotten.isEmpty {
            facts.removeAll { forgotten.contains($0.text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()) }
        }
        var seen = Set(facts.map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() })
        for raw in update.newFacts {
            let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, seen.insert(text.lowercased()).inserted else { continue }
            facts.append(Fact(text: text, date: date, chatID: chatID))
        }
        if facts.count > Self.maxFacts {
            // Stable: oldest by date first, ties keep insertion order.
            let drop = facts.count - Self.maxFacts
            let order = facts.enumerated().sorted { ($0.element.date, $0.offset) < ($1.element.date, $1.offset) }
            let dropped = Set(order.prefix(drop).map(\.offset))
            facts = facts.enumerated().filter { !dropped.contains($0.offset) }.map(\.element)
        }
        let delta = max(-10, min(10, update.affectionDelta))
        relationship.affection = max(0, min(100, relationship.affection + delta))
        if let s = update.stage?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           let stage = Stage(rawValue: s) {
            relationship.stage = stage
        }
        if let m = update.mood?.trimmingCharacters(in: .whitespacesAndNewlines), !m.isEmpty { relationship.mood = m }
        if let s = update.summary?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty { summary = s }
        relationship.lastSeen = date
        turnsSinceUpdate = 0
    }
}

/// What kind of generated media a message carries.
public enum MediaKind: String, Codable, Sendable, CaseIterable { case photo, video, voice, talk }

/// A photo, video or voice note kept in the character's album.
public struct AlbumItem: Codable, Identifiable, Equatable, Sendable {
    public var id: UUID
    public var file: String
    public var kind: MediaKind
    public var prompt: String
    public var scene: String?
    public var chatID: UUID?
    public var date: Date
    public init(id: UUID = UUID(), file: String, kind: MediaKind, prompt: String, scene: String? = nil,
                chatID: UUID? = nil, date: Date = Date()) {
        self.id = id
        self.file = file
        self.kind = kind
        self.prompt = prompt
        self.scene = scene
        self.chatID = chatID
        self.date = date
    }
}

/// The result of a memory pass.
public struct MemoryUpdate: Codable, Equatable, Sendable {
    public var newFacts: [String] = []
    public var forget: [String] = []
    public var affectionDelta = 0
    public var stage: String?
    public var mood: String?
    public var summary: String?

    public init(newFacts: [String] = [], forget: [String] = [], affectionDelta: Int = 0, stage: String? = nil,
                mood: String? = nil, summary: String? = nil) {
        self.newFacts = newFacts
        self.forget = forget
        self.affectionDelta = affectionDelta
        self.stage = stage
        self.mood = mood
        self.summary = summary
    }

    enum CodingKeys: String, CodingKey {
        case newFacts = "new_facts", forget, affectionDelta = "affection_delta", stage, mood, summary
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        newFacts = (try? c.decodeIfPresent([String].self, forKey: .newFacts)) ?? []
        forget = (try? c.decodeIfPresent([String].self, forKey: .forget)) ?? []
        if let i = try? c.decodeIfPresent(Int.self, forKey: .affectionDelta) {
            affectionDelta = i
        } else if let d = try? c.decodeIfPresent(Double.self, forKey: .affectionDelta) {
            affectionDelta = Int(d.rounded())
        }
        stage = try? c.decodeIfPresent(String.self, forKey: .stage)
        mood = try? c.decodeIfPresent(String.self, forKey: .mood)
        summary = try? c.decodeIfPresent(String.self, forKey: .summary)
    }
}
