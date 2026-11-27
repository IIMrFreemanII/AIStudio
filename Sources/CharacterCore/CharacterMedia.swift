import Foundation
import GraphRuntime
import MovieCore

/// Whether the user's message asks for a kind of media (so it doesn't count against the character's initiative).
public enum MediaAsk {
    private static let photoWords = ["photo", "photos", "pic", "pics", "picture", "pictures", "selfie", "selfies", "snap",
                                     "image", "show me", "see you", "what you look like", "what are you wearing"]
    private static let videoWords = ["video", "videos", "clip", "film yourself", "record yourself", "on camera"]
    private static let voiceWords = ["voice", "hear you", "hear your", "audio", "say it", "say something", "voice note",
                                     "voice message"]

    private static func has(_ text: String, _ words: [String]) -> Bool {
        words.contains { has(text, pattern: "\\b" + NSRegularExpression.escapedPattern(for: $0) + "\\b") }
    }

    private static func has(_ text: String, pattern: String) -> Bool {
        text.range(of: pattern, options: [.regularExpression, .caseInsensitive]) != nil
    }

    public static func asked(_ kind: MediaKind, in text: String) -> Bool {
        let t = text.lowercased()
        switch kind {
        case .photo: return has(t, photoWords)
        case .video: return has(t, videoWords)
        case .voice: return has(t, voiceWords)
        case .talk:
            if has(t, ["video message", "on camera"]) { return true }
            return has(t, videoWords) && (has(t, voiceWords) || has(t, pattern: "\\bsay\\b|\\bsaying\\b|\\btalk\\b|\\btalking\\b"))
        case .motion:
            return has(t, videoWords) || has(t, pattern: "\\bdo (this|that|it)\\b|\\blike (this|that)\\b|\\bcopy\\b|\\bdance\\b")
        }
    }
}

/// What happens to one media request of a reply.
public enum MediaDecision: Equatable, Sendable { case run, ask, skip(String) }

public enum CharacterCost {
    /// Videos and video messages always wait for the user's OK; photos and voice only when they cost more than `autoLimit`.
    public static func alwaysAsks(_ kind: MediaKind) -> Bool { kind == .video || kind == .talk || kind == .motion }

    public static func decide(_ kind: MediaKind, estimate: Double?, autoLimit: Double, asked: Bool,
                              initiativeLeft: Int) -> MediaDecision {
        if !asked && initiativeLeft <= 0 { return .skip("No more unasked media today") }
        if alwaysAsks(kind) { return .ask }
        if let e = estimate, e > autoLimit { return .ask }
        return .run
    }

    public static func initiativeLeft(_ c: Character, memory: CharacterMemory, now: Date = Date(),
                                      calendar: Calendar = .current) -> Int {
        max(0, c.initiative - memory.ownMediaToday(now: now, calendar: calendar))
    }
}

/// Voice messages, videos and video messages: the graphs (from Movie Studio's templates) and their values.
public enum CharacterMedia {
    /// Wan 2.2 clips and LongCat clips are portrait: 480 × 832.
    public static let videoSize = (width: 480, height: 832)
    public static let videoSeconds = 5.0
    /// LongCat clips: the voice must fit.
    public static let maxTalkSeconds = MovieVoices.maxTalking
    /// Motion clips copy at most this much of the user's clip (from its start).
    public static let maxMotionSeconds = 10.0
    /// Wan-Animate-2's target area for chat (portrait; the shape is the still's) and its frame rate.
    public static let motionArea = (width: 640, height: 800)
    public static let motionFPS = 24.0

    /// Wan-Animate-2's work for `seconds` of clip: megapixels × frames, as GPUTimings counts it.
    public static func motionWork(seconds: Double) -> Double {
        Double(motionArea.width * motionArea.height) / 1e6 * (min(seconds, maxMotionSeconds) * motionFPS).rounded()
    }

    /// One GPU job of a request: its server task and its work for GPUTimings.
    public struct Job: Equatable, Sendable {
        public var task: String
        public var work: Double
        public init(task: String, work: Double) {
            self.task = task
            self.work = work
        }
    }

    /// Seconds a line takes to say: words / 2.6, at least 1.
    public static func speechSeconds(_ text: String) -> Double {
        let words = text.split(whereSeparator: \.isWhitespace).count
        return max(1, Double(words) / 2.6)
    }

    /// `clipSeconds`: a motion clip's length (the user's clip, capped at `maxMotionSeconds`).
    public static func jobs(_ kind: MediaKind, character c: Character, text: String, hasVoiceSample: Bool,
                            clipSeconds: Double = maxMotionSeconds) -> [Job] {
        let photo = Job(task: CharacterArt.task(c.artModel.family),
                        work: Double(CharacterArt.photoSize.width * CharacterArt.photoSize.height) / 1e6)
        let sample = Job(task: c.voiceEngine.rawValue, work: speechSeconds(MovieVoices.sampleText))
        let line = Job(task: c.voiceEngine.rawValue, work: speechSeconds(text))
        let wan = Job(task: "wan22-i2v", work: MovieShots.work(width: videoSize.width, height: videoSize.height,
                                                               frames: MovieShots.frames(for: videoSeconds)))
        let longcat = Job(task: "longcat-avatar", work: MovieShots.talkingWork(width: videoSize.width, height: videoSize.height,
                                                                                seconds: speechSeconds(text)))
        switch kind {
        case .photo: return [photo]
        case .voice: return (hasVoiceSample ? [] : [sample]) + [line]
        case .video: return [photo, wan]
        case .talk: return [photo] + (hasVoiceSample ? [] : [sample]) + [line, longcat]
        case .motion: return [photo, Job(task: "wan-animate2", work: motionWork(seconds: clipSeconds))]
        }
    }

    public static func estimate(_ kind: MediaKind, character c: Character, text: String, hasVoiceSample: Bool,
                                clipSeconds: Double = maxMotionSeconds, timings: GPUTimings,
                                price: (String) -> Double?, warm: (String) -> Bool) -> Double? {
        var total = 0.0
        var loaded = Set<String>()
        for job in jobs(kind, character: c, text: text, hasVoiceSample: hasVoiceSample, clipSeconds: clipSeconds) {
            guard let p = price(job.task) else { return nil }
            total += timings.seconds(task: job.task, megapixels: job.work) * p
            if !warm(job.task), loaded.insert(job.task).inserted { total += timings.loadSeconds(task: job.task) * p }
        }
        return total
    }

    // MARK: Voice

    public static func voiceSampleTemplate(_ c: Character) -> MovieTemplate {
        MovieVoices.sampleTemplate(engine: c.voiceEngine)
    }

    public static func voiceSampleRender(_ c: Character, seed: Int, server: String) -> MovieArt.Render {
        MovieVoices.sampleRender(MovieCharacter(id: "character", name: c.name, look: c.fullLook, voice: voiceDescription(c)),
                                 engine: c.voiceEngine, seed: seed, server: server)
    }

    /// What the voice is designed from: who speaks (age and gender: the voice options alone, "low-pitched, husky",
    /// gave a woman a man's voice), then the options and description.
    public static func voiceDescription(_ c: Character) -> String {
        let gender = c.gender.trimmingCharacters(in: .whitespacesAndNewlines)
        let who = "A \(c.age)-year-old " + (gender.isEmpty ? "adult" : gender) + "'s voice"
        let voice = c.fullVoice
        return voice.isEmpty ? who : "\(who): \(voice)"
    }

    public static func voiceLineTemplate(_ c: Character) -> MovieTemplate {
        MovieVoices.lineTemplate(engine: c.voiceEngine)
    }

    public static func voiceLineRender(_ c: Character, text: String, sample: URL, seed: Int, server: String) -> MovieArt.Render {
        MovieVoices.lineRender(DialogueLine(character: "character", text: text, delivery: ""), engine: c.voiceEngine,
                               sample: sample, seed: seed, server: server)
    }

    // MARK: Video

    private static func clean(_ s: String) -> String {
        s.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            .trimmingCharacters(in: CharacterSet(charactersIn: " ,.;"))
    }

    public static func videoTemplate() -> MovieTemplate { MovieShots.template(name: "character-video") }

    public static func videoStillPrompt(_ c: Character, request: String, scene: Scenario?,
                                        context: CharacterArt.Context = .init()) -> String {
        CharacterArt.photoPrompt(c, request: request, scene: scene, context: context)
    }

    public static func motionPrompt(_ c: Character, request: String) -> String {
        let gender = clean(c.gender), look = clean(c.fullLook), req = CharacterArt.caption(clean(request))
        var out = "A \(c.age)-year-old adult" + (gender.isEmpty ? "" : " \(gender)")
        out += look.isEmpty ? "." : ", \(look)."
        if !req.isEmpty { out += " \(req)." }
        return out + " Natural, smooth motion; the camera holds steady."
    }

    public static func videoRender(_ c: Character, prompt: String, start: URL, seed: Int, server: String) -> MovieArt.Render {
        let values: [String: JSONValue] = [
            "prompt": .string(prompt), "negative": .string(CharacterArt.negative(c)), "seed": .int(seed),
            "width": .int(videoSize.width), "height": .int(videoSize.height),
            "length": .int(MovieShots.frames(for: videoSeconds)), "server": .string(server), "start": .string(start.path),
        ]
        return MovieArt.Render(values: values, unusedPictures: ["end"])
    }

    // MARK: Motion

    /// The motion clip's graph: Load Image (the still) and Load Video (the user's clip) into Wan Animate (Cloud GPU)
    /// into a Preview Video.
    public static func motionTemplate(name: String = "character-motion") -> MovieTemplate {
        let still = GraphNode(id: "still", type: "LoadImage", inputs: ["path": .value(.string(""))],
                              position: GraphPoint(x: 40, y: 60))
        let clip = GraphNode(id: "clip", type: "LoadVideo", inputs: ["path": .value(.string(""))],
                             position: GraphPoint(x: 40, y: 300))
        let animate = GraphNode(id: "animate", type: "RemoteWanAnimate", inputs: [
            "image": .link(node: "still", output: "image"), "video": .link(node: "clip", output: "video"),
            "prompt": .value(.string("")), "scene": .value(.string("")), "seed": .value(.int(0)),
            "seconds": .value(.double(5)), "server": .value(.string("")),
        ], position: GraphPoint(x: 340, y: 60))
        let out = GraphNode(id: "out", type: "PreviewVideo", inputs: ["video": .link(node: "animate", output: "video")],
                            position: GraphPoint(x: 720, y: 80))
        var slots = Dictionary(uniqueKeysWithValues: ["prompt", "scene", "seed", "seconds", "server"].map {
            ($0, MovieTemplate.Slot(node: "animate", input: $0))
        })
        slots["still"] = .init(node: "still", input: "path")
        slots["clip"] = .init(node: "clip", input: "path")
        return MovieTemplate(name: name, graph: Graph(nodes: [still, clip, animate, out]), slots: slots)
    }

    /// The still the motion starts from: the whole body, standing, empty hands, so a dance's arms and legs have
    /// somewhere to go (a seated, phone-holding selfie made the first real one dance sitting down); a plain photo, not
    /// a smartphone shot (which puts a phone in the hand). Its background is the reply's or the scene's place
    /// (Wan-Animate-2 keeps the picture's).
    public static func motionStillPrompt(_ c: Character, request: String, scene: Scenario?,
                                         context: CharacterArt.Context = .init()) -> String {
        var s = CharacterArt.spec(c, request: "", scene: scene, context: context)
        if c.artModel.style == .photoreal { s.medium = "photo" }
        s.shot = "full-body shot, head to feet in frame"
        s.action = "standing upright, arms relaxed at the sides, hands empty, facing the camera"
        let req = clean(request)
        if !req.isEmpty { s.place = CharacterArt.caption(req) }
        return PromptWriter.write(s, for: CharacterArt.promptModel(c, context: context)).prompt
    }

    /// The face reference's strength for the still: the training pictures' (weaker than photos'), so the portrait's
    /// framing and pose don't carry over.
    public static let motionFaceStrength = CharacterTraining.faceStrength

    /// The reply's words as the motion clip's scene (a caption without "you"/"I").
    public static func motionScene(_ request: String) -> String { CharacterArt.caption(clean(request)) }

    /// The look for Wan-Animate-2's caption: who they are and what they wear in the still.
    public static func motionLook(_ c: Character) -> String {
        let gender = clean(c.gender), look = clean(c.fullLook)
        return "A \(c.age)-year-old adult" + (gender.isEmpty ? "" : " \(gender)") + (look.isEmpty ? "" : ", \(look)")
    }

    public static func motionRender(_ c: Character, still: URL, clip: URL, seconds: Double, scene: String, seed: Int,
                                    server: String) -> MovieArt.Render {
        let values: [String: JSONValue] = [
            "prompt": .string(motionLook(c)), "scene": .string(scene), "seed": .int(seed),
            "seconds": .double(min(max(seconds, 1), maxMotionSeconds)), "server": .string(server),
            "still": .string(still.path), "clip": .string(clip.path),
        ]
        return MovieArt.Render(values: values, unusedPictures: [])
    }

    // MARK: Talk

    public static func talkTemplate() -> MovieTemplate { MovieShots.longCatTemplate(name: "character-talk") }

    public static func talkStillPrompt(_ c: Character, scene: Scenario?, context: CharacterArt.Context = .init()) -> String {
        CharacterArt.photoPrompt(c, request: "looking into the camera, head and shoulders, about to speak", scene: scene,
                                 context: context)
    }

    public static func talkPrompt(_ c: Character) -> String {
        let gender = clean(c.gender)
        return "A \(c.age)-year-old adult" + (gender.isEmpty ? "" : " \(gender)")
            + " talks to the camera, lips moving with the words, natural expression."
    }

    public static func talkRender(_ c: Character, prompt: String, start: URL, voice: URL, seed: Int, server: String) -> MovieArt.Render {
        let values: [String: JSONValue] = [
            "prompt": .string(prompt), "seed": .int(seed), "server": .string(server),
            "start": .string(start.path), "voice": .string(voice.path),
        ]
        return MovieArt.Render(values: values, unusedPictures: ["voice2"])
    }
}
