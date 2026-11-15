import Foundation

/// The stages a movie goes through, in order. Each ends at a review gate: the user approves it before the next one
/// spends anything.
public enum MovieStage: String, Codable, CaseIterable, Sendable, Identifiable {
    case idea, script, shots, cast, keyframes, voices, video, sound, edit
    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .idea: "Idea"
        case .script: "Script"
        case .shots: "Shot List"
        case .cast: "Cast"
        case .keyframes: "Keyframes"
        case .voices: "Voices"
        case .video: "Shots"
        case .sound: "Sound"
        case .edit: "Edit"
        }
    }

    public var symbol: String {
        switch self {
        case .idea: "lightbulb"
        case .script: "doc.text"
        case .shots: "rectangle.split.3x3"
        case .cast: "person.2"
        case .keyframes: "photo.on.rectangle"
        case .voices: "waveform"
        case .video: "film"
        case .sound: "music.note"
        case .edit: "timeline.selection"
        }
    }

    /// Stages built so far (the rest show "coming in a later phase", and are skipped: the film goes on without
    /// them, silent until Voices and Sound arrive).
    public var isAvailable: Bool { [.idea, .script, .shots, .cast, .keyframes, .video, .edit].contains(self) }

    /// The available stage before this one: the one that must be approved first.
    public var previous: MovieStage? {
        let all = Self.allCases.filter(\.isAvailable)
        return all.firstIndex(of: self).flatMap { $0 > 0 ? all[$0 - 1] : nil }
    }
}

public enum MovieAspect: String, Codable, CaseIterable, Sendable, Identifiable {
    case wide, vertical, square
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .wide: "16:9"
        case .vertical: "9:16"
        case .square: "1:1"
        }
    }
    /// Wan 2.2's 480p sizes (multiples of 16).
    public var videoSize: (width: Int, height: Int) {
        switch self {
        case .wide: (832, 480)
        case .vertical: (480, 832)
        case .square: (640, 640)
        }
    }
    /// Keyframe stills: about a megapixel at the same shape (SDXL / Qwen sizes, multiples of 64).
    public var keyframeSize: (width: Int, height: Int) {
        switch self {
        case .wide: (1344, 768)
        case .vertical: (768, 1344)
        case .square: (1024, 1024)
        }
    }
}

/// The LLM that writes the script and the shot list: a model on a GPU server.
public struct MovieLLM: Codable, Hashable, Sendable {
    public var serverID: String
    public var modelID: String
    public init(serverID: String, modelID: String) {
        self.serverID = serverID
        self.modelID = modelID
    }
}

/// One generated file of a shot, character or cue (a keyframe still, a clip, a voice line…): the user keeps several
/// and picks one.
public struct Take: Codable, Identifiable, Sendable, Equatable {
    public var id: UUID
    /// Relative to the movie's folder ("keyframes/…png").
    public var file: String
    /// A still that stands for it in lists, for a clip (relative too); nil for a picture, which is its own.
    public var poster: String?
    public var seed: Int?
    public var costUSD: Double?
    public var date: Date

    public init(id: UUID = UUID(), file: String, poster: String? = nil, seed: Int? = nil, costUSD: Double? = nil,
                date: Date = Date()) {
        self.id = id
        self.file = file
        self.poster = poster
        self.seed = seed
        self.costUSD = costUSD
        self.date = date
    }
}

public struct DialogueLine: Codable, Identifiable, Sendable, Equatable {
    public var id: UUID
    /// A character's id.
    public var character: String
    public var text: String
    /// How it's said ("quietly, scared"); goes into the voice prompt.
    public var delivery: String

    public init(id: UUID = UUID(), character: String, text: String, delivery: String = "") {
        self.id = id
        self.character = character
        self.text = text
        self.delivery = delivery
    }
}

public struct MovieCharacter: Codable, Identifiable, Sendable, Equatable {
    /// A slug ("mara"), what the script and shots refer to.
    public var id: String
    public var name: String
    /// What they look like (face, hair, build, clothes): the reference sheet's prompt and every keyframe's.
    public var look: String
    /// What they sound like, for the voice designer ("warm low female voice, slight rasp, 30s").
    public var voice: String
    public var sheets: [Take]
    public var selectedSheet: UUID?
    /// A character LoRA in the library, when one was trained.
    public var loraID: String?

    public init(id: String, name: String, look: String, voice: String, sheets: [Take] = [], selectedSheet: UUID? = nil,
                loraID: String? = nil) {
        self.id = id
        self.name = name
        self.look = look
        self.voice = voice
        self.sheets = sheets
        self.selectedSheet = selectedSheet
        self.loraID = loraID
    }
}

public struct MovieScene: Codable, Identifiable, Sendable, Equatable {
    /// "s1", "s2"…
    public var id: String
    /// "EXT. PINE FOREST – DAWN"
    public var heading: String
    public var summary: String
    /// For the score ("tense, sparse strings").
    public var mood: String
    /// What happens, in screenplay action lines.
    public var action: String
    public var dialogue: [DialogueLine]

    public init(id: String, heading: String, summary: String = "", mood: String = "", action: String = "",
                dialogue: [DialogueLine] = []) {
        self.id = id
        self.heading = heading
        self.summary = summary
        self.mood = mood
        self.action = action
        self.dialogue = dialogue
    }
}

public enum ShotTransition: String, Codable, CaseIterable, Sendable {
    case cut, crossfade, fadeFromBlack, fadeToBlack
    public var title: String {
        switch self {
        case .cut: "Cut"
        case .crossfade: "Crossfade"
        case .fadeFromBlack: "Fade In"
        case .fadeToBlack: "Fade Out"
        }
    }
}

public struct Shot: Codable, Identifiable, Sendable, Equatable {
    public var id: UUID
    public var scene: String
    /// Seconds on screen (Wan makes up to about 7.5 s per clip).
    public var duration: Double
    /// "wide", "medium", "close-up", "over the shoulder"…
    public var framing: String
    /// "static", "slow push in", "pan left"…
    public var camera: String
    /// The still the shot starts from: who, where, light, composition (no motion).
    public var keyframePrompt: String
    /// What moves during the shot.
    public var motionPrompt: String
    public var characters: [String]
    public var dialogue: [DialogueLine]
    public var sfx: String
    /// Someone speaks on screen: rendered with lip-sync from the voice audio.
    public var talking: Bool
    /// Starts from the previous shot's last frame (one continuous moment) instead of its own keyframe.
    public var continuesPrevious: Bool
    /// How this shot starts.
    public var transition: ShotTransition
    public var keyframes: [Take]
    public var selectedKeyframe: UUID?
    public var videos: [Take]
    public var selectedVideo: UUID?
    /// Seconds cut from the start and the end of the chosen clip in the edit.
    public var trimStart: Double
    public var trimEnd: Double

    public init(id: UUID = UUID(), scene: String, duration: Double = 5, framing: String = "", camera: String = "",
                keyframePrompt: String = "", motionPrompt: String = "", characters: [String] = [],
                dialogue: [DialogueLine] = [], sfx: String = "", talking: Bool = false, continuesPrevious: Bool = false,
                transition: ShotTransition = .cut, keyframes: [Take] = [], selectedKeyframe: UUID? = nil,
                videos: [Take] = [], selectedVideo: UUID? = nil, trimStart: Double = 0, trimEnd: Double = 0) {
        self.id = id
        self.scene = scene
        self.duration = duration
        self.framing = framing
        self.camera = camera
        self.keyframePrompt = keyframePrompt
        self.motionPrompt = motionPrompt
        self.characters = characters
        self.dialogue = dialogue
        self.sfx = sfx
        self.talking = talking
        self.continuesPrevious = continuesPrevious
        self.transition = transition
        self.keyframes = keyframes
        self.selectedKeyframe = selectedKeyframe
        self.videos = videos
        self.selectedVideo = selectedVideo
        self.trimStart = trimStart
        self.trimEnd = trimEnd
    }

    /// Wan clips are 1–7.5 s.
    public static let durationRange: ClosedRange<Double> = 1...7.5

    private enum CodingKeys: String, CodingKey {
        case id, scene, duration, framing, camera, keyframePrompt, motionPrompt, characters, dialogue, sfx, talking,
             continuesPrevious, transition, keyframes, selectedKeyframe, videos, selectedVideo, trimStart, trimEnd
    }

    /// Movies saved before the edit stage have no trims.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        scene = try c.decode(String.self, forKey: .scene)
        duration = try c.decode(Double.self, forKey: .duration)
        framing = try c.decode(String.self, forKey: .framing)
        camera = try c.decode(String.self, forKey: .camera)
        keyframePrompt = try c.decode(String.self, forKey: .keyframePrompt)
        motionPrompt = try c.decode(String.self, forKey: .motionPrompt)
        characters = try c.decode([String].self, forKey: .characters)
        dialogue = try c.decode([DialogueLine].self, forKey: .dialogue)
        sfx = try c.decode(String.self, forKey: .sfx)
        talking = try c.decode(Bool.self, forKey: .talking)
        continuesPrevious = try c.decode(Bool.self, forKey: .continuesPrevious)
        transition = try c.decode(ShotTransition.self, forKey: .transition)
        keyframes = try c.decode([Take].self, forKey: .keyframes)
        selectedKeyframe = try c.decodeIfPresent(UUID.self, forKey: .selectedKeyframe)
        videos = try c.decode([Take].self, forKey: .videos)
        selectedVideo = try c.decodeIfPresent(UUID.self, forKey: .selectedVideo)
        trimStart = try c.decodeIfPresent(Double.self, forKey: .trimStart) ?? 0
        trimEnd = try c.decodeIfPresent(Double.self, forKey: .trimEnd) ?? 0
    }
}

/// A movie: the idea, script, cast and shots, what was generated for them, and which stages the user approved.
/// Saved as `movie.json` in its own folder, with the generated files beside it.
public struct MovieProject: Codable, Identifiable, Sendable, Equatable {
    public static let formatVersion = 1

    public var version = MovieProject.formatVersion
    public var id: UUID
    public var title: String
    /// The user's idea or logline, what the script is written from.
    public var idea: String
    /// A `MovieStyle` id.
    public var style: String
    public var aspect: MovieAspect
    /// How long the film should be, roughly (seconds).
    public var targetSeconds: Int
    public var llm: MovieLLM?
    public var logline: String
    public var characters: [MovieCharacter]
    public var scenes: [MovieScene]
    public var shots: [Shot]
    public var approved: Set<MovieStage>
    /// What the GPU work for this movie has cost so far (USD, from the servers' figures).
    public var costSpent: Double
    public var created: Date
    public var modified: Date

    public init(id: UUID = UUID(), title: String = "Untitled Movie", idea: String = "",
                style: String = MovieStyle.cinematic.id, aspect: MovieAspect = .wide, targetSeconds: Int = 90,
                llm: MovieLLM? = nil, logline: String = "", characters: [MovieCharacter] = [], scenes: [MovieScene] = [],
                shots: [Shot] = [], approved: Set<MovieStage> = [], costSpent: Double = 0, created: Date = Date(),
                modified: Date = Date()) {
        self.id = id
        self.title = title
        self.idea = idea
        self.style = style
        self.aspect = aspect
        self.targetSeconds = targetSeconds
        self.llm = llm
        self.logline = logline
        self.characters = characters
        self.scenes = scenes
        self.shots = shots
        self.approved = approved
        self.costSpent = costSpent
        self.created = created
        self.modified = modified
    }

    public var runtime: Double { shots.reduce(0) { $0 + $1.duration } }

    public func character(_ id: String) -> MovieCharacter? { characters.first { $0.id == id } }

    public func shots(in scene: String) -> [Shot] { shots.filter { $0.scene == scene } }

    /// The stage to show: the first available one not approved yet.
    public var currentStage: MovieStage {
        MovieStage.allCases.first { $0.isAvailable && !approved.contains($0) } ?? .edit
    }

    /// A stage can be worked on once the one before it is approved.
    public func isUnlocked(_ stage: MovieStage) -> Bool {
        guard let prev = stage.previous else { return true }
        return approved.contains(prev)
    }

    /// Approving a stage; changing an approved stage's content takes back the approval of everything after it
    /// (the later work was made from the old version).
    public mutating func approve(_ stage: MovieStage) { approved.insert(stage) }

    public mutating func reopen(_ stage: MovieStage) {
        let all = MovieStage.allCases
        guard let i = all.firstIndex(of: stage) else { return }
        for s in all[i...] { approved.remove(s) }
    }
}
