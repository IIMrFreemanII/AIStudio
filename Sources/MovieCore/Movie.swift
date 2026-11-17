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

    /// Stages made and approved once per sequence (a long film goes through them a sequence at a time); the others
    /// are the whole film's.
    public var isPerSequence: Bool {
        switch self {
        case .shots, .keyframes, .voices, .video, .sound: true
        default: false
        }
    }

    /// The stages that must be approved before this one can be worked on (a per-sequence one in the same sequence).
    public var requires: [MovieStage] {
        switch self {
        case .idea: []
        case .script: [.idea]
        case .shots, .cast: [.script]
        case .keyframes: [.shots, .cast]
        case .voices: [.keyframes]
        case .video: [.voices]
        case .sound: [.video]
        case .edit: [.video]
        }
    }

    /// The available stage before this one in the rail.
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
    /// MiniMax H3's own canvas for the shape (its 768-pixel short edge, multiples of 32).
    public var h3Size: (width: Int, height: Int) {
        switch self {
        case .wide: (1344, 768)
        case .vertical: (768, 1344)
        case .square: (768, 768)
        }
    }
    /// The shape as H3's `aspect` parameter.
    public var h3Aspect: String {
        switch self {
        case .wide: "16:9"
        case .vertical: "9:16"
        case .square: "1:1"
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
    /// The video engine that made a clip (`VideoEngine` raw value); nil for clips made before engines were recorded,
    /// which are Wan 2.2's.
    public var engine: String?
    /// Where someone speaks in a clip with its own sound (MiniMax H3), found when it was made (`MovieSpeech`); nil
    /// when not looked for yet, empty when no voice was found.
    public var voice: [VoiceSpan]?

    public init(id: UUID = UUID(), file: String, poster: String? = nil, seed: Int? = nil, costUSD: Double? = nil,
                date: Date = Date(), seconds: Double? = nil, gainDB: Double? = nil, engine: String? = nil) {
        self.id = id
        self.file = file
        self.poster = poster
        self.seed = seed
        self.costUSD = costUSD
        self.date = date
        self.seconds = seconds
        self.gainDB = gainDB
        self.engine = engine
    }
}

/// The cloud model that animates a talking shot from its lines.
public enum LipSyncEngine: String, Codable, Sendable, CaseIterable, Identifiable {
    /// Wan 2.2 S2V: from the keyframe, 40 steps (several minutes of GPU time a clip).
    case wanS2V = "wan-s2v"
    /// LongCat-Video-Avatar 1.6: from the keyframe, distilled to 8 steps; one or two speakers.
    case longcat
    /// InfiniteTalk: makes the shot's ordinary Wan clip, then re-syncs the speaker to the lines (keeps its motion).
    case infiniteTalk = "infinitetalk"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .wanS2V: "Wan 2.2 S2V"
        case .longcat: "LongCat Avatar"
        case .infiniteTalk: "InfiniteTalk"
        }
    }

    /// The server task that does the lip-sync.
    public var task: String {
        switch self {
        case .wanS2V: "wan22-s2v"
        case .longcat: "longcat-avatar"
        case .infiniteTalk: "infinitetalk"
        }
    }
}

/// The model that makes the shots' clips.
public enum VideoEngine: String, Codable, Sendable, CaseIterable, Identifiable {
    /// Wan 2.2: silent clips from the keyframes (lines, effects and lip-sync are added by the later stages).
    case wan22 = "wan-2.2"
    /// MiniMax H3: each clip with its own sound (its lines in the characters' voices, effects, ambience), so the
    /// Lines and Effects steps are skipped. Its licence excludes the US, EU, UK and Korea without MiniMax's own.
    case minimaxH3 = "minimax-h3"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .wan22: "Wan 2.2"
        case .minimaxH3: "MiniMax H3 (with sound)"
        }
    }

    /// Whether its clips carry the shot's own sound.
    public var makesSound: Bool { self == .minimaxH3 }
}

/// The model that speaks the characters' voice samples and lines.
public enum VoiceEngine: String, Codable, Sendable, CaseIterable, Identifiable {
    /// Qwen3-TTS: 10 languages; a line clones the sample's voice but can't take its delivery.
    case qwen3 = "qwen3-tts"
    /// Breeze TTS 2: English and Chinese, research and non-commercial use only; a line clones the sample's voice and
    /// is said as its delivery directs ("quietly, scared").
    case breeze = "breeze-tts"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .qwen3: "Qwen3-TTS"
        case .breeze: "Breeze TTS 2 (non-commercial)"
        }
    }

    /// The server task that speaks.
    public var task: String { rawValue }
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

/// A run of scenes made and approved together (an act's chapter of a long film): shot list, keyframes, voices,
/// clips and sound go a sequence at a time.
public struct MovieSequence: Codable, Identifiable, Sendable, Equatable, Hashable {
    /// "q1", "q2"…
    public var id: String
    public var title: String
    /// The act it belongs to (1…).
    public var act: Int
    public var summary: String

    public init(id: String, title: String, act: Int = 1, summary: String = "") {
        self.id = id
        self.title = title
        self.act = act
        self.summary = summary
    }

    /// What movies made before sequences have: one holding every scene.
    public static let wholeFilm = MovieSequence(id: "q1", title: "Whole film")
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
    /// The sequence it's in (a `MovieSequence` id).
    public var sequence: String

    public init(id: String, heading: String, summary: String = "", mood: String = "", action: String = "",
                dialogue: [DialogueLine] = [], sequence: String = MovieSequence.wholeFilm.id) {
        self.id = id
        self.heading = heading
        self.summary = summary
        self.mood = mood
        self.action = action
        self.dialogue = dialogue
        self.sequence = sequence
    }

    private enum CodingKeys: String, CodingKey { case id, heading, summary, mood, action, dialogue, sequence }

    /// Scenes saved before sequences are in the one sequence such movies have.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        heading = try c.decode(String.self, forKey: .heading)
        summary = try c.decodeIfPresent(String.self, forKey: .summary) ?? ""
        mood = try c.decodeIfPresent(String.self, forKey: .mood) ?? ""
        action = try c.decodeIfPresent(String.self, forKey: .action) ?? ""
        dialogue = try c.decodeIfPresent([DialogueLine].self, forKey: .dialogue) ?? []
        sequence = try c.decodeIfPresent(String.self, forKey: .sequence) ?? MovieSequence.wholeFilm.id
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
    /// The film's sequences, in order (at least one once there's a script).
    public var sequences: [MovieSequence]
    /// The whole-film stages approved (per-sequence ones are in `sequenceApproved`).
    public var approved: Set<MovieStage>
    /// Per-sequence stages approved, by sequence id.
    public var sequenceApproved: [String: Set<MovieStage>]
    /// What the user means to spend on this movie at most (USD); nil: no budget.
    public var budget: Double?
    /// What the GPU work for this movie has cost so far (USD, from the servers' figures).
    public var costSpent: Double
    public var created: Date
    public var modified: Date
    /// The scenes' music.
    public var music: [MusicCue]
    public var mix: MovieMix
    /// Talking shots are animated from their lines' audio (lips in sync, by `lipSyncEngine`); off: like any other shot.
    public var lipSync: Bool
    /// The model that makes lip-synced talking shots.
    public var lipSyncEngine: LipSyncEngine
    /// The model that speaks the voice samples and lines.
    public var voiceEngine: VoiceEngine
    /// The model that makes the clips.
    public var videoEngine: VideoEngine

    public init(id: UUID = UUID(), title: String = "Untitled Movie", idea: String = "",
                style: String = MovieStyle.cinematic.id, aspect: MovieAspect = .wide, targetSeconds: Int = 90,
                llm: MovieLLM? = nil, logline: String = "", characters: [MovieCharacter] = [], scenes: [MovieScene] = [],
                shots: [Shot] = [], sequences: [MovieSequence] = [.wholeFilm], approved: Set<MovieStage> = [],
                sequenceApproved: [String: Set<MovieStage>] = [:], budget: Double? = nil, costSpent: Double = 0,
                created: Date = Date(),
                modified: Date = Date(), music: [MusicCue] = [], mix: MovieMix = MovieMix(), lipSync: Bool = true,
                lipSyncEngine: LipSyncEngine = .wanS2V, voiceEngine: VoiceEngine = .qwen3,
                videoEngine: VideoEngine = .wan22) {
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
        self.sequences = sequences
        self.approved = approved.filter { !$0.isPerSequence }
        self.sequenceApproved = sequenceApproved
        self.budget = budget
        self.costSpent = costSpent
        self.created = created
        self.modified = modified
        self.music = music
        self.mix = mix
        self.lipSync = lipSync
        self.lipSyncEngine = lipSyncEngine
        self.voiceEngine = voiceEngine
        self.videoEngine = videoEngine
        // Per-sequence stages given in `approved` count for every sequence.
        for stage in approved where stage.isPerSequence { approve(stage) }
    }

    private enum CodingKeys: String, CodingKey {
        case version, id, title, idea, style, aspect, targetSeconds, llm, logline, characters, scenes, shots, sequences,
             approved, sequenceApproved, budget, costSpent, created, modified, music, mix, lipSync, lipSyncEngine,
             voiceEngine, videoEngine
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
        sequences = try c.decodeIfPresent([MovieSequence].self, forKey: .sequences) ?? [.wholeFilm]
        if sequences.isEmpty { sequences = [.wholeFilm] }
        let approvedAll = try c.decode(Set<MovieStage>.self, forKey: .approved)
        approved = approvedAll.filter { !$0.isPerSequence }
        if let per = try c.decodeIfPresent([String: Set<MovieStage>].self, forKey: .sequenceApproved) {
            sequenceApproved = per
        } else {
            // Movies from before sequences: their approvals were the whole film's, now its one sequence's.
            let per = approvedAll.filter(\.isPerSequence)
            sequenceApproved = per.isEmpty ? [:] : Dictionary(uniqueKeysWithValues: sequences.map { ($0.id, per) })
        }
        budget = try c.decodeIfPresent(Double.self, forKey: .budget)
        costSpent = try c.decode(Double.self, forKey: .costSpent)
        created = try c.decode(Date.self, forKey: .created)
        modified = try c.decode(Date.self, forKey: .modified)
        music = try c.decodeIfPresent([MusicCue].self, forKey: .music) ?? []
        mix = try c.decodeIfPresent(MovieMix.self, forKey: .mix) ?? MovieMix()
        lipSync = try c.decodeIfPresent(Bool.self, forKey: .lipSync) ?? true
        // Movies from before the choice used Wan S2V; an engine this version doesn't know falls back to it.
        lipSyncEngine = (try? c.decodeIfPresent(LipSyncEngine.self, forKey: .lipSyncEngine)) ?? .wanS2V
        voiceEngine = (try? c.decodeIfPresent(VoiceEngine.self, forKey: .voiceEngine)) ?? .qwen3
        videoEngine = (try? c.decodeIfPresent(VideoEngine.self, forKey: .videoEngine)) ?? .wan22
    }

    public var runtime: Double { shots.reduce(0) { $0 + $1.duration } }

    /// The clips' size: the video engine's canvas for the movie's shape (also the export's).
    public var videoSize: (width: Int, height: Int) {
        videoEngine == .minimaxH3 ? aspect.h3Size : aspect.videoSize
    }

    public func character(_ id: String) -> MovieCharacter? { characters.first { $0.id == id } }

    public func shots(in scene: String) -> [Shot] { shots.filter { $0.scene == scene } }

    /// The sequence a scene is in (the first sequence when the scene names none that exists).
    public func sequence(ofScene scene: String) -> String {
        let id = scenes.first { $0.id == scene }?.sequence
        return id.flatMap { s in sequences.contains { $0.id == s } ? s : nil } ?? sequences.first?.id ?? MovieSequence.wholeFilm.id
    }

    /// The sequence a shot is in (its scene's).
    public func sequence(of shot: Shot) -> String { sequence(ofScene: shot.scene) }

    /// Scene id → sequence id, for lookups over many shots.
    public var sequenceByScene: [String: String] {
        var out: [String: String] = [:]
        let known = Set(sequences.map(\.id))
        let first = sequences.first?.id ?? MovieSequence.wholeFilm.id
        for s in scenes { out[s.id] = known.contains(s.sequence) ? s.sequence : first }
        return out
    }

    /// The shots of one sequence (nil: the whole film), in film order.
    public func shots(inSequence sequence: String?) -> [Shot] {
        guard let sequence else { return shots }
        let map = sequenceByScene
        let first = sequences.first?.id
        return shots.filter { (map[$0.scene] ?? first) == sequence }
    }

    /// The scenes of one sequence (nil: all).
    public func scenes(inSequence sequence: String?) -> [MovieScene] {
        guard let sequence else { return scenes }
        let map = sequenceByScene
        return scenes.filter { map[$0.id] == sequence }
    }

    /// Shot id → its index in `shots`, for lookups over many shots.
    public var shotIndex: [UUID: Int] {
        Dictionary(shots.enumerated().map { ($1.id, $0) }, uniquingKeysWith: { a, _ in a })
    }

    /// Whether `stage` is approved: a whole-film stage for the film; a per-sequence one in `sequence`, or (nil) in
    /// every sequence.
    public func isApproved(_ stage: MovieStage, in sequence: String? = nil) -> Bool {
        guard stage.isPerSequence else { return approved.contains(stage) }
        if let sequence { return sequenceApproved[sequence]?.contains(stage) ?? false }
        return !sequences.isEmpty && sequences.allSatisfy { sequenceApproved[$0.id]?.contains(stage) ?? false }
    }

    /// Whether any sequence has `stage` approved.
    public func isApprovedAnywhere(_ stage: MovieStage) -> Bool {
        guard stage.isPerSequence else { return approved.contains(stage) }
        return sequences.contains { sequenceApproved[$0.id]?.contains(stage) ?? false }
    }

    /// The stage to show for a sequence (nil: the whole film): the first not approved yet.
    public func currentStage(in sequence: String? = nil) -> MovieStage {
        MovieStage.allCases.first { $0.isAvailable && !isApproved($0, in: sequence) } ?? .edit
    }

    /// The stage to show for the whole film.
    public var currentStage: MovieStage { currentStage() }

    /// A stage can be worked on once the ones it needs are approved: in `sequence` for per-sequence ones (nil: in
    /// any sequence, for the whole-film view). The edit opens once any sequence has its clips approved.
    public func isUnlocked(_ stage: MovieStage, in sequence: String? = nil) -> Bool {
        if stage == .edit { return isApprovedAnywhere(.video) }
        return stage.requires.allSatisfy { need in
            if need.isPerSequence && stage.isPerSequence {
                if let sequence { return isApproved(need, in: sequence) }
                return isApprovedAnywhere(need)
            }
            return need.isPerSequence ? isApprovedAnywhere(need) : approved.contains(need)
        }
    }

    /// Whether the edit can be approved: every sequence's sound is.
    public var editIsApprovable: Bool { isApproved(.sound) }

    /// Approving a stage (a per-sequence one in `sequence`, or nil: in every sequence).
    public mutating func approve(_ stage: MovieStage, in sequence: String? = nil) {
        guard stage.isPerSequence else { approved.insert(stage); return }
        for s in sequence.map({ [$0] }) ?? sequences.map(\.id) { sequenceApproved[s, default: []].insert(stage) }
    }

    /// Takes back a stage's approval and everything after it (the later work was made from the old version). A
    /// per-sequence stage in `sequence` only (nil: every sequence); the film's edit goes too. A whole-film stage takes
    /// back the later whole-film stages and the later per-sequence ones everywhere.
    public mutating func reopen(_ stage: MovieStage, in sequence: String? = nil) {
        let all = MovieStage.allCases
        guard let i = all.firstIndex(of: stage) else { return }
        let later = all[i...]
        let targets = stage.isPerSequence ? (sequence.map { [$0] } ?? Array(sequenceApproved.keys)) : Array(sequenceApproved.keys)
        for s in later {
            if s.isPerSequence {
                for q in targets { sequenceApproved[q]?.remove(s) }
            } else if !stage.isPerSequence || s == .edit {
                approved.remove(s)
            }
        }
        sequenceApproved = sequenceApproved.filter { !$0.value.isEmpty }
    }

    /// Takes back one approval only (the studio agent's undo).
    public mutating func unapprove(_ stage: MovieStage, in sequence: String? = nil) {
        guard stage.isPerSequence else { approved.remove(stage); return }
        for s in sequence.map({ [$0] }) ?? Array(sequenceApproved.keys) { sequenceApproved[s]?.remove(stage) }
    }

    /// Every approval as "stage" or "stage@sequence" (to compare before and after a change).
    public var approvalKeys: Set<String> {
        var out = Set(approved.map(\.rawValue))
        for (q, set) in sequenceApproved { for s in set { out.insert("\(s.rawValue)@\(q)") } }
        return out
    }

    /// The film's longest target (20 minutes).
    public static let maxTargetSeconds = 1200
}
