import Foundation
import MovieCore

/// A virtual companion the user chats with. Every character is a fictional adult (see `AgeGuard`).
public struct Character: Codable, Identifiable, Equatable, Sendable, Hashable {
    public var id: UUID
    public var name: String
    /// Must be 18 or older.
    public var age: Int
    /// Free text: "woman", "man", "non-binary".
    public var gender: String
    /// Appearance, used in image prompts.
    public var look: String
    public var personality: String
    public var backstory: String
    public var speakingStyle: String
    public var likes: String
    /// The first message of a new chat ("" = none).
    public var greeting: String
    /// A voice description (used for speech).
    public var voice: String
    /// Picked catalog options: category id → option ids (see `CharacterOptions`).
    public var traits: [String: [String]]
    public var artModel: ArtModel
    public var loras: [CharacterLoRA]
    /// File names in the character's portraits/ folder.
    public var portraits: [String]
    public var selectedPortrait: String?
    /// The speech model that says the character's voice messages.
    public var voiceEngine: VoiceEngine
    /// File names in the character's voice/ folder.
    public var voiceSamples: [String]
    public var selectedVoiceSample: String?
    /// The kinds of media the character may send.
    public var media: [MediaKind]
    /// Own-initiative media per day.
    public var initiative: Int
    /// Hours away after which they text you first when you open the chat; 0: never.
    public var checkInHours: Int
    public var temperature: Double
    public var topP: Double
    /// The built-in template it was made from.
    public var templateID: String?
    public var created: Date
    public var updated: Date

    public init(id: UUID = UUID(), name: String, age: Int, gender: String, look: String = "", personality: String = "",
                backstory: String = "", speakingStyle: String = "", likes: String = "", greeting: String = "",
                voice: String = "", date: Date = Date()) {
        self.id = id
        self.name = name
        self.age = age
        self.gender = gender
        self.look = look
        self.personality = personality
        self.backstory = backstory
        self.speakingStyle = speakingStyle
        self.likes = likes
        self.greeting = greeting
        self.voice = voice
        self.traits = [:]
        self.artModel = ArtModel()
        self.loras = []
        self.portraits = []
        self.selectedPortrait = nil
        self.voiceEngine = .qwen3
        self.voiceSamples = []
        self.selectedVoiceSample = nil
        self.media = MediaKind.allCases
        self.initiative = 3
        self.checkInHours = 8
        self.temperature = 0.9
        self.topP = 0.95
        self.templateID = nil
        self.created = date
        self.updated = date
    }

    enum CodingKeys: String, CodingKey {
        case id, name, age, gender, look, personality, backstory, speakingStyle, likes, greeting, voice, traits, artModel
        case loras, portraits, selectedPortrait, voiceEngine, voiceSamples, selectedVoiceSample, media, initiative, checkInHours, temperature, topP, templateID, created, updated
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        age = try c.decodeIfPresent(Int.self, forKey: .age) ?? 18
        gender = try c.decodeIfPresent(String.self, forKey: .gender) ?? ""
        look = try c.decodeIfPresent(String.self, forKey: .look) ?? ""
        personality = try c.decodeIfPresent(String.self, forKey: .personality) ?? ""
        backstory = try c.decodeIfPresent(String.self, forKey: .backstory) ?? ""
        speakingStyle = try c.decodeIfPresent(String.self, forKey: .speakingStyle) ?? ""
        likes = try c.decodeIfPresent(String.self, forKey: .likes) ?? ""
        greeting = try c.decodeIfPresent(String.self, forKey: .greeting) ?? ""
        voice = try c.decodeIfPresent(String.self, forKey: .voice) ?? ""
        traits = try c.decodeIfPresent([String: [String]].self, forKey: .traits) ?? [:]
        artModel = try c.decodeIfPresent(ArtModel.self, forKey: .artModel) ?? ArtModel()
        loras = try c.decodeIfPresent([CharacterLoRA].self, forKey: .loras) ?? []
        portraits = try c.decodeIfPresent([String].self, forKey: .portraits) ?? []
        selectedPortrait = try c.decodeIfPresent(String.self, forKey: .selectedPortrait)
        voiceEngine = try c.decodeIfPresent(VoiceEngine.self, forKey: .voiceEngine) ?? .qwen3
        voiceSamples = try c.decodeIfPresent([String].self, forKey: .voiceSamples) ?? []
        selectedVoiceSample = try c.decodeIfPresent(String.self, forKey: .selectedVoiceSample)
        media = try c.decodeIfPresent([MediaKind].self, forKey: .media) ?? MediaKind.allCases
        initiative = try c.decodeIfPresent(Int.self, forKey: .initiative) ?? 3
        checkInHours = try c.decodeIfPresent(Int.self, forKey: .checkInHours) ?? 8
        temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.9
        topP = try c.decodeIfPresent(Double.self, forKey: .topP) ?? 0.95
        templateID = try c.decodeIfPresent(String.self, forKey: .templateID)
        created = try c.decodeIfPresent(Date.self, forKey: .created) ?? Date(timeIntervalSince1970: 0)
        updated = try c.decodeIfPresent(Date.self, forKey: .updated) ?? Date(timeIntervalSince1970: 0)
    }
}

/// Which image model draws the character.
public struct ArtModel: Codable, Equatable, Sendable, Hashable {
    public enum Family: String, Codable, Sendable, CaseIterable {
        case sdxl, chroma, minimax

        /// The name in pickers.
        public var title: String {
            switch self {
            case .sdxl: "SDXL"
            case .chroma: "Chroma"
            case .minimax: "MiniMax"
            }
        }

        /// Fine-tunes from the Models library (MiniMax H3 Image takes none).
        public var takesLibraryModels: Bool { self != .minimax }

        /// LoRAs from the Models library (every family; on MiniMax, H3 Image LoRAs and the video model's).
        public var takesLoRAs: Bool { true }

        /// The Models library family of its fine-tunes and LoRAs.
        public var libraryFamily: String {
            switch self {
            case .sdxl: "sdxl"
            case .chroma: "chroma"
            case .minimax: "minimax-h3-image"
            }
        }
    }
    public enum Style: String, Codable, Sendable, CaseIterable { case photoreal, anime }
    public var family: Family = .sdxl
    public var style: Style = .photoreal
    /// A fine-tune id on the GPU server (nil: the base model).
    public var variant: String?
    public init(family: Family = .sdxl, style: Style = .photoreal, variant: String? = nil) {
        self.family = family
        self.style = style
        self.variant = variant
    }
}

/// A LoRA applied when drawing the character.
public struct CharacterLoRA: Codable, Equatable, Sendable, Hashable {
    /// The library item id of the LoRA (the app's Models library; the `RemoteLoRA` node's `lora` input takes it).
    public var lora: String
    public var strength: Double = 0.8
    public init(lora: String, strength: Double = 0.8) {
        self.lora = lora
        self.strength = strength
    }

    enum CodingKeys: String, CodingKey { case lora, link, strength }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Older files called it `link`.
        lora = try c.decodeIfPresent(String.self, forKey: .lora) ?? c.decode(String.self, forKey: .link)
        strength = try c.decodeIfPresent(Double.self, forKey: .strength) ?? 0.8
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(lora, forKey: .lora)
        try c.encode(strength, forKey: .strength)
    }
}
