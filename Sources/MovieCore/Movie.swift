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

    /// Stages built so far (any others would show "coming in a later phase" and be skipped). All of them now.
    public var isAvailable: Bool { true }

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
    /// A sound's length in seconds (voice lines, effects, music), measured when it was made.
    public var seconds: Double?
    /// A sound's loudness correction in dB, measured when it was made (`MovieLoudness`), so every line and cue
    /// starts from the same level before the mix.
    public var gainDB: Double?

    public init(id: UUID = UUID(), file: String, poster: String? = nil, seed: Int? = nil, costUSD: Double? = nil,
                date: Date = Date(), seconds: Double? = nil, gainDB: Double? = nil) {
        self.id = id
        self.file = file
        self.poster = poster
        self.seed = seed
        self.costUSD = costUSD
        self.date = date
        self.seconds = seconds
        self.gainDB = gainDB
    }
}

public struct DialogueLine: Codable, Identifiable, Sendable, Equatable {
    public var id: UUID
    /// A character's id.
    public var character: String
    public var text: String
    /// How it's said ("quietly, scared"); goes into the voice prompt.
    public var delivery: String
    /// The line spoken in the character's voice (a shot's lines; a scene's are only the script).
    public var takes: [Take]
    public var selectedTake: UUID?

    public init(id: UUID = UUID(), character: String, text: String, delivery: String = "", takes: [Take] = [],
                selectedTake: UUID? = nil) {
        self.id = id
        self.character = character
        self.text = text
        self.delivery = delivery
        self.takes = takes
        self.selectedTake = selectedTake
    }

    private enum CodingKeys: String, CodingKey { case id, character, text, delivery, takes, selectedTake }

    /// Lines saved before the Voices stage have no takes.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        character = try c.decode(String.self, forKey: .character)
        text = try c.decode(String.self, forKey: .text)
        delivery = try c.decodeIfPresent(String.self, forKey: .delivery) ?? ""
        takes = try c.decodeIfPresent([Take].self, forKey: .takes) ?? []
        selectedTake = try c.decodeIfPresent(UUID.self, forKey: .selectedTake)
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
    /// Their voice, designed from `voice` saying a sample sentence: every line of theirs is spoken in it (cloned),
    /// so the voice stays the same from line to line.
    public var voiceSamples: [Take]
    public var selectedVoiceSample: UUID?

    public init(id: String, name: String, look: String, voice: String, sheets: [Take] = [], selectedSheet: UUID? = nil,
                loraID: String? = nil, voiceSamples: [Take] = [], selectedVoiceSample: UUID? = nil) {
        self.id = id
        self.name = name
        self.look = look
        self.voice = voice
        self.sheets = sheets
        self.selectedSheet = selectedSheet
        self.loraID = loraID
        self.voiceSamples = voiceSamples
        self.selectedVoiceSample = selectedVoiceSample
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, look, voice, sheets, selectedSheet, loraID, voiceSamples, selectedVoiceSample
    }

    /// Characters saved before the Voices stage have no voice samples.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        look = try c.decode(String.self, forKey: .look)
        voice = try c.decode(String.self, forKey: .voice)
        sheets = try c.decodeIfPresent([Take].self, forKey: .sheets) ?? []
        selectedSheet = try c.decodeIfPresent(UUID.self, forKey: .selectedSheet)
        loraID = try c.decodeIfPresent(String.self, forKey: .loraID)
        voiceSamples = try c.decodeIfPresent([Take].self, forKey: .voiceSamples) ?? []
        selectedVoiceSample = try c.decodeIfPresent(UUID.self, forKey: .selectedVoiceSample)
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
    /// The shot's sound effects (from `sfx`), as long as the shot.
    public var sounds: [Take]
    public var selectedSound: UUID?

    public init(id: UUID = UUID(), scene: String, duration: Double = 5, framing: String = "", camera: String = "",
                keyframePrompt: String = "", motionPrompt: String = "", characters: [String] = [],
                dialogue: [DialogueLine] = [], sfx: String = "", talking: Bool = false, continuesPrevious: Bool = false,
                transition: ShotTransition = .cut, keyframes: [Take] = [], selectedKeyframe: UUID? = nil,
                videos: [Take] = [], selectedVideo: UUID? = nil, trimStart: Double = 0, trimEnd: Double = 0,
                sounds: [Take] = [], selectedSound: UUID? = nil) {
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
        self.sounds = sounds
        self.selectedSound = selectedSound
    }

    /// Wan clips are 1–7.5 s.
    public static let durationRange: ClosedRange<Double> = 1...7.5

    private enum CodingKeys: String, CodingKey {
        case id, scene, duration, framing, camera, keyframePrompt, motionPrompt, characters, dialogue, sfx, talking,
             continuesPrevious, transition, keyframes, selectedKeyframe, videos, selectedVideo, trimStart, trimEnd,
             sounds, selectedSound
    }

    /// Movies saved before the edit stage have no trims, and before the sound stage no sounds.
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
        sounds = try c.decodeIfPresent([Take].self, forKey: .sounds) ?? []
        selectedSound = try c.decodeIfPresent(UUID.self, forKey: .selectedSound)
    }
}

/// A scene's score: one piece of music under all of its shots.
public struct MusicCue: Codable, Identifiable, Sendable, Equatable {
    /// The scene's id.
    public var id: String
    /// The style of the music (ACE-Step tags: genre, instruments, mood), from the scene's mood and the movie's style.
    public var prompt: String
    public var takes: [Take]
    public var selected: UUID?

    public init(id: String, prompt: String, takes: [Take] = [], selected: UUID? = nil) {
        self.id = id
        self.prompt = prompt
        self.takes = takes
        self.selected = selected
    }

    public var selectedTake: Take? { takes.first { $0.id == selected } }
}

/// How loud each kind of sound plays in the film (linear gains on top of each take's loudness correction).
public struct MovieMix: Codable, Sendable, Equatable {
    public var dialogue: Double
    public var effects: Double
    public var music: Double
    /// The music's gain while someone speaks (it dips under the line and comes back after).
    public var duck: Double
    /// Burn the subtitles into the exported picture (an SRT is written beside the MP4 either way).
    public var burnSubtitles: Bool

    public init(dialogue: Double = 1, effects: Double = 0.5, music: Double = 0.35, duck: Double = 0.3,
                burnSubtitles: Bool = false) {
        self.dialogue = dialogue
        self.effects = effects
        self.music = music
        self.duck = duck
        self.burnSubtitles = burnSubtitles
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
    /// The scenes' music.
    public var music: [MusicCue]
    public var mix: MovieMix
    /// Talking shots are animated by Wan 2.2 S2V from their lines' audio (lips in sync); off: like any other shot.
    public var lipSync: Bool

    public init(id: UUID = UUID(), title: String = "Untitled Movie", idea: String = "",
                style: String = MovieStyle.cinematic.id, aspect: MovieAspect = .wide, targetSeconds: Int = 90,
                llm: MovieLLM? = nil, logline: String = "", characters: [MovieCharacter] = [], scenes: [MovieScene] = [],
                shots: [Shot] = [], approved: Set<MovieStage> = [], costSpent: Double = 0, created: Date = Date(),
                modified: Date = Date(), music: [MusicCue] = [], mix: MovieMix = MovieMix(), lipSync: Bool = true) {
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
        self.music = music
        self.mix = mix
        self.lipSync = lipSync
    }

    private enum CodingKeys: String, CodingKey {
        case version, id, title, idea, style, aspect, targetSeconds, llm, logline, characters, scenes, shots, approved,
             costSpent, created, modified, music, mix, lipSync
    }

    /// Movies saved before the sound stages have no music, mix or lip-sync setting.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        version = try c.decodeIfPresent(Int.self, forKey: .version) ?? Self.formatVersion
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        idea = try c.decode(String.self, forKey: .idea)
        style = try c.decode(String.self, forKey: .style)
        aspect = try c.decode(MovieAspect.self, forKey: .aspect)
        targetSeconds = try c.decode(Int.self, forKey: .targetSeconds)
        llm = try c.decodeIfPresent(MovieLLM.self, forKey: .llm)
        logline = try c.decode(String.self, forKey: .logline)
        characters = try c.decode([MovieCharacter].self, forKey: .characters)
        scenes = try c.decode([MovieScene].self, forKey: .scenes)
        shots = try c.decode([Shot].self, forKey: .shots)
        approved = try c.decode(Set<MovieStage>.self, forKey: .approved)
        costSpent = try c.decode(Double.self, forKey: .costSpent)
        created = try c.decode(Date.self, forKey: .created)
        modified = try c.decode(Date.self, forKey: .modified)
        music = try c.decodeIfPresent([MusicCue].self, forKey: .music) ?? []
        mix = try c.decodeIfPresent(MovieMix.self, forKey: .mix) ?? MovieMix()
        lipSync = try c.decodeIfPresent(Bool.self, forKey: .lipSync) ?? true
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
