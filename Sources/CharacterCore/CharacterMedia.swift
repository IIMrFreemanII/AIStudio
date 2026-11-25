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
        }
    }
}

/// What happens to one media request of a reply.
public enum MediaDecision: Equatable, Sendable { case run, ask, skip(String) }

public enum CharacterCost {
    /// Videos and video messages always wait for the user's OK; photos and voice only when they cost more than `autoLimit`.
    public static func alwaysAsks(_ kind: MediaKind) -> Bool { kind == .video || kind == .talk }

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

    public static func jobs(_ kind: MediaKind, character c: Character, text: String, hasVoiceSample: Bool) -> [Job] {
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
        }
    }

    public static func estimate(_ kind: MediaKind, character c: Character, text: String, hasVoiceSample: Bool,
                                timings: GPUTimings, price: (String) -> Double?, warm: (String) -> Bool) -> Double? {
        var total = 0.0
        var loaded = Set<String>()
        for job in jobs(kind, character: c, text: text, hasVoiceSample: hasVoiceSample) {
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
        MovieVoices.sampleRender(MovieCharacter(id: "character", name: c.name, look: c.look, voice: c.voice),
                                 engine: c.voiceEngine, seed: seed, server: server)
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

    public static func videoStillPrompt(_ c: Character, request: String, scene: Scenario?) -> String {
        CharacterArt.photoPrompt(c, request: request, scene: scene)
    }

    public static func motionPrompt(_ c: Character, request: String) -> String {
        let gender = clean(c.gender), look = clean(c.look), req = CharacterArt.caption(clean(request))
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

    // MARK: Talk

    public static func talkTemplate() -> MovieTemplate { MovieShots.longCatTemplate(name: "character-talk") }

    public static func talkStillPrompt(_ c: Character, scene: Scenario?) -> String {
        CharacterArt.photoPrompt(c, request: "looking into the camera, head and shoulders, about to speak", scene: scene)
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
