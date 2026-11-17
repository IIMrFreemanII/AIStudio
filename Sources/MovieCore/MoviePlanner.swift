import Agent
import Foundation
import LLMClient

public enum MoviePlannerError: Error, CustomStringConvertible, Equatable {
    case empty
    case noJSON
    case invalid(String)

    public var description: String {
        switch self {
        case .empty: "The model returned nothing."
        case .noJSON: "The model's reply had no JSON in it."
        case .invalid(let why): "The model's reply didn't fit: \(why)"
        }
    }
}

/// A script as the LLM wrote it, before it goes into the movie.
public struct ScriptDraft: Sendable, Equatable {
    public var title: String
    public var logline: String
    public var characters: [MovieCharacter]
    public var scenes: [MovieScene]
}

/// Writing the script and the shot list with an LLM: the prompts, reading the replies (tolerant of what models do:
/// fences, names for ids, numbers as strings), and asking once more when a reply doesn't parse.
public enum MoviePlanner {
    // MARK: prompts

    public static func scriptPrompt(for movie: MovieProject) -> (system: String, user: String) {
        let style = MovieStyle.named(movie.style)
        let shots = max(3, Int((Double(movie.targetSeconds) / 5).rounded()))
        let scenes = max(1, min(40, Int((Double(shots) / 5).rounded())))
        let system = """
        You are a screenwriter for short AI-generated films. Every shot will be generated as a separate 3–7 second \
        video clip from a still image, so write a story told mostly through clear, filmable visuals: few locations, \
        few characters (at most 4 with lines), simple physical actions, short spoken lines. Avoid things video models \
        get wrong: crowds, text on screen, fast fights, complex hand work, characters changing clothes.

        Reply with only JSON, in this shape:
        {"title": "...", "logline": "one sentence",
         "characters": [{"id": "lowercase-slug", "name": "...", \
        "look": "fixed visual description: age, face, hair, build, clothing, colours", \
        "voice": "how they sound: gender, age, pitch, texture, accent, pace"}],
         "scenes": [{"id": "s1", "heading": "EXT. PLACE – TIME", "summary": "one sentence", \
        "mood": "for the music: genre, tempo, instruments", "action": "screenplay action lines", \
        "dialogue": [{"character": "<character id>", "text": "...", "delivery": "how it's said"}]}]}
        """
        let user = """
        Idea: \(movie.idea.isEmpty ? movie.title : movie.idea)

        Style: \(style.name) – \(style.brief).
        Length: about \(movie.targetSeconds) seconds on screen, so about \(shots) shots in \(scenes) \
        scene\(scenes == 1 ? "" : "s"). Keep dialogue to what fits: about 2.5 words per second of screen time, \
        and leave room for shots without lines.
        """
        return (system, user)
    }

    /// How long each scene should run: the film's target split by how much happens in it (its action and dialogue
    /// words), at least 4 s each. Without a budget, models write a shot or two per scene and the film comes out short.
    public static func sceneSeconds(for movie: MovieProject) -> [String: Double] {
        func words(_ s: String) -> Int { s.split(whereSeparator: { $0.isWhitespace }).count }
        let weights = movie.scenes.map { s in
            Double(max(8, words(s.action) + s.dialogue.reduce(0) { $0 + words($1.text) } * 2))
        }
        let total = weights.reduce(0, +)
        guard total > 0 else { return [:] }
        var out: [String: Double] = [:]
        for (s, w) in zip(movie.scenes, weights) {
            out[s.id] = max(4, (Double(movie.targetSeconds) * w / total).rounded())
        }
        return out
    }

    public static func shotPrompt(for movie: MovieProject, scene: MovieScene) -> (system: String, user: String) {
        let style = MovieStyle.named(movie.style)
        let system = """
        You are a director breaking a scene into shots for AI video generation. Each shot becomes one clip: first a \
        still keyframe image is generated, then it is animated for the shot's duration (1–7 s, usually 3–6). \
        One clear action per shot. Vary framing (wide, medium, close-up, over the shoulder) and keep screen direction \
        consistent. Put each spoken line on a medium shot or close-up of the speaker, so it can be lip-synced; use wide \
        shots for places and action.

        For each shot write:
        - keyframe: a self-contained image prompt for the first frame: who is on screen, described by their look \
        (not only their name), where, pose, expression, lighting, composition. No motion words.
        - motion: what moves during the shot (people, camera, environment), one or two sentences.
        - dialogue: only the lines spoken during this shot (copied from the scene), each by a character id.
        - talking: true when the speaker's face is on screen while they talk, in a medium shot or closer (it gets \
        lip-sync); false for wide shots, voice-over or an off-screen speaker.
        - continues: true only when the shot continues the previous shot's exact moment and place without a cut in \
        time (it will start from the previous clip's last frame).
        - sfx: the sound effects and ambience heard (no music).

        Reply with only JSON:
        {"shots": [{"duration": 5, "framing": "wide", "camera": "slow push in", "keyframe": "...", "motion": "...", \
        "characters": ["<character id>"], "dialogue": [{"character": "<id>", "text": "...", "delivery": "..."}], \
        "talking": false, "continues": false, "transition": "cut | crossfade | fadeFromBlack | fadeToBlack", \
        "sfx": "..."}]}
        """
        let cast = movie.characters.map { "- \($0.id): \($0.name) – \($0.look)" }.joined(separator: "\n")
        let lines = scene.dialogue.map { "\($0.character): \($0.text)\($0.delivery.isEmpty ? "" : " (\($0.delivery))")" }
        let others = movie.scenes.map { "\($0.id). \($0.heading) – \($0.summary)" }.joined(separator: "\n")
        let seconds = sceneSeconds(for: movie)[scene.id] ?? 15
        let shots = max(1, Int((seconds / 4.5).rounded()))
        let user = """
        Film: \(movie.title) – \(movie.logline)
        Style: \(style.name) – \(style.brief).

        Characters:
        \(cast.isEmpty ? "(none)" : cast)

        All scenes, for context:
        \(others)

        Break down this scene into about \(shots) shot\(shots == 1 ? "" : "s") that together run about \
        \(Int(seconds)) seconds:
        \(scene.id). \(scene.heading)
        \(scene.action)
        \(lines.isEmpty ? "(no dialogue)" : "Dialogue:\n" + lines.joined(separator: "\n"))
        """
        return (system, user)
    }

    // MARK: reading replies

    private struct ScriptReply: Decodable {
        var title: String?
        var logline: String?
        var characters: [CharacterReply]?
        var scenes: [SceneReply]?
    }

    private struct CharacterReply: Decodable {
        var id: String?
        var name: String?
        var look: String?
        var voice: String?
    }

    private struct SceneReply: Decodable {
        var id: String?
        var heading: String?
        var summary: String?
        var mood: String?
        var action: String?
        var dialogue: [LineReply]?
    }

    private struct LineReply: Decodable {
        var character: String?
        var text: String?
        var delivery: String?
    }

    private struct ShotsReply: Decodable {
        var shots: [ShotReply]?
    }

    private struct ShotReply: Decodable {
        var duration: Flexible?
        var framing: String?
        var camera: String?
        var keyframe: String?
        var motion: String?
        var characters: [String]?
        var dialogue: [LineReply]?
        var talking: Flexible?
        var continues: Flexible?
        var transition: String?
        var sfx: String?
    }

    /// A number or bool that may come as a string ("5", "5s", "true").
    private struct Flexible: Decodable {
        var number: Double?
        var bool: Bool?

        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let d = try? c.decode(Double.self) { number = d; bool = d != 0 }
            else if let b = try? c.decode(Bool.self) { bool = b; number = b ? 1 : 0 }
            else if let s = try? c.decode(String.self) {
                let t = s.trimmingCharacters(in: .whitespaces).lowercased()
                bool = ["true", "yes", "1"].contains(t)
                number = Double(t.filter { $0.isNumber || $0 == "." })
            }
        }
    }

    private static func decode<T: Decodable>(_ type: T.Type, from reply: String) throws -> T {
        let text = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw MoviePlannerError.empty }
        guard let value = FlowJSON.extract(text) else { throw MoviePlannerError.noJSON }
        do {
            return try JSONDecoder().decode(T.self, from: JSONEncoder().encode(value))
        } catch {
            throw MoviePlannerError.invalid("not the expected shape")
        }
    }

    static func slug(_ s: String) -> String {
        let parts = s.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" }
        return String(parts).split(separator: "-").joined(separator: "-")
    }

    private static func clean(_ s: String?) -> String { (s ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }

    /// A character reference as the model wrote it (an id, a name, "MARA") → a known id; nil when it's nobody known.
    static func resolve(_ ref: String?, in cast: [MovieCharacter]) -> String? {
        let r = clean(ref)
        guard !r.isEmpty else { return nil }
        if let c = cast.first(where: { $0.id == r }) { return c.id }
        let s = slug(r)
        if let c = cast.first(where: { $0.id == s || slug($0.name) == s }) { return c.id }
        // "Tomas" for "Old Tomas", "Mara Voss" for "mara": the one character sharing a word with the reference.
        let words = Set(s.split(separator: "-"))
        let matches = cast.filter { !Set(slug($0.name).split(separator: "-")).isDisjoint(with: words) }
        return matches.count == 1 ? matches[0].id : nil
    }

    private static func lines(_ replies: [LineReply]?, cast: [MovieCharacter]) -> [DialogueLine] {
        (replies ?? []).compactMap { l in
            let text = clean(l.text)
            guard !text.isEmpty, let who = resolve(l.character, in: cast) else { return nil }
            return DialogueLine(character: who, text: text, delivery: clean(l.delivery))
        }
    }

    public static func parseScript(_ reply: String) throws -> ScriptDraft {
        let r = try decode(ScriptReply.self, from: reply)
        var cast: [MovieCharacter] = []
        for c in r.characters ?? [] {
            let name = clean(c.name).isEmpty ? clean(c.id) : clean(c.name)
            guard !name.isEmpty else { continue }
            var id = slug(clean(c.id).isEmpty ? name : clean(c.id))
            if id.isEmpty { id = "character" }
            let base = id
            var n = 2
            while cast.contains(where: { $0.id == id }) { id = "\(base)-\(n)"; n += 1 }
            cast.append(MovieCharacter(id: id, name: name, look: clean(c.look), voice: clean(c.voice)))
        }
        var scenes: [MovieScene] = []
        for (i, s) in (r.scenes ?? []).enumerated() {
            let heading = clean(s.heading)
            let action = clean(s.action)
            guard !heading.isEmpty || !action.isEmpty else { continue }
            var id = clean(s.id).isEmpty ? "s\(i + 1)" : slug(clean(s.id))
            if scenes.contains(where: { $0.id == id }) { id = "s\(i + 1)-\(scenes.count)" }
            scenes.append(MovieScene(id: id, heading: heading.isEmpty ? "SCENE \(i + 1)" : heading,
                                     summary: clean(s.summary), mood: clean(s.mood), action: action,
                                     dialogue: lines(s.dialogue, cast: cast)))
        }
        guard !scenes.isEmpty else { throw MoviePlannerError.invalid("no scenes") }
        return ScriptDraft(title: clean(r.title), logline: clean(r.logline), characters: cast, scenes: scenes)
    }

    public static func parseShots(_ reply: String, scene: String, cast: [MovieCharacter]) throws -> [Shot] {
        let r = try decode(ShotsReply.self, from: reply)
        let shots: [Shot] = (r.shots ?? []).compactMap { s in
            let keyframe = clean(s.keyframe)
            guard !keyframe.isEmpty else { return nil }
            let dialogue = lines(s.dialogue, cast: cast)
            var who = (s.characters ?? []).compactMap { resolve($0, in: cast) }
            for l in dialogue where (s.talking?.bool ?? false) && !who.contains(l.character) { who.append(l.character) }
            let seconds = s.duration?.number ?? 5
            return Shot(scene: scene,
                        duration: min(max(seconds.isFinite ? seconds : 5, Shot.durationRange.lowerBound),
                                      Shot.durationRange.upperBound),
                        framing: clean(s.framing), camera: clean(s.camera), keyframePrompt: keyframe,
                        motionPrompt: clean(s.motion), characters: who, dialogue: dialogue, sfx: clean(s.sfx),
                        talking: (s.talking?.bool ?? false) && !dialogue.isEmpty && !isWide(clean(s.framing)),
                        continuesPrevious: s.continues?.bool ?? false,
                        transition: ShotTransition(rawValue: clean(s.transition)) ?? .cut)
        }
        guard !shots.isEmpty else { throw MoviePlannerError.invalid("no shots") }
        var out = shots
        out[0].continuesPrevious = false  // a scene starts fresh
        return out
    }

    /// Framings too far away for lip-sync to show.
    static func isWide(_ framing: String) -> Bool {
        let f = framing.lowercased()
        return ["wide", "establishing", "aerial", "long shot", "extreme long"].contains { f.contains($0) }
            && !f.contains("medium")
    }

    /// Puts a new script into the movie: cast and scenes replaced, shots dropped (they were made from the old one).
    public static func apply(_ draft: ScriptDraft, to movie: inout MovieProject) {
        if !draft.title.isEmpty, movie.title == "Untitled Movie" || movie.title.isEmpty { movie.title = draft.title }
        movie.logline = draft.logline
        movie.characters = draft.characters
        movie.scenes = draft.scenes
        movie.shots = []
        groupSequences(&movie)
        movie.reopen(.shots)
        movie.reopen(.cast)  // a new cast
    }

    /// About how long a sequence runs: a long film is made a few minutes at a time.
    public static let sequenceSeconds = 150.0

    /// Splits the scenes into sequences of about `sequenceSeconds` by their time budgets (one, "Whole film", for a
    /// film that short), in order; per-sequence approvals of sequences that no longer exist are dropped.
    public static func groupSequences(_ movie: inout MovieProject) {
        let budgets = sceneSeconds(for: movie)
        let total = movie.scenes.reduce(0) { $0 + (budgets[$1.id] ?? 0) }
        let count = max(1, Int((total / sequenceSeconds).rounded()))
        guard count > 1 else {
            movie.sequences = [.wholeFilm]
            for i in movie.scenes.indices { movie.scenes[i].sequence = MovieSequence.wholeFilm.id }
            movie.sequenceApproved = movie.sequenceApproved.filter { $0.key == MovieSequence.wholeFilm.id }
            return
        }
        let per = total / Double(count)
        var sequences: [MovieSequence] = []
        var run = 0.0
        for i in movie.scenes.indices {
            let n = sequences.count
            if n == 0 || (run >= per * Double(n) && n < count) {
                sequences.append(MovieSequence(id: "q\(n + 1)", title: movie.scenes[i].heading, act: 1))
            }
            movie.scenes[i].sequence = sequences[sequences.count - 1].id
            run += budgets[movie.scenes[i].id] ?? 0
        }
        // Acts: the first quarter, the middle half, the last quarter.
        for i in sequences.indices {
            let f = Double(i) / Double(sequences.count)
            sequences[i].act = f < 0.25 ? 1 : f < 0.75 ? 2 : 3
        }
        movie.sequences = sequences
        let ids = Set(sequences.map(\.id))
        movie.sequenceApproved = movie.sequenceApproved.filter { ids.contains($0.key) }
    }

    /// Replaces one scene's shots, keeping the scenes' order.
    public static func replaceShots(_ shots: [Shot], scene: String, in movie: inout MovieProject) {
        var out: [Shot] = []
        var placed = false
        for sceneID in movie.scenes.map(\.id) {
            if sceneID == scene { out += shots; placed = true } else { out += movie.shots(in: sceneID) }
        }
        if !placed { out += shots }
        movie.shots = out
    }

    // MARK: asking

    /// How a request is going: the reply's length so far (thinking included), or what the model's engine is doing
    /// before the first token (a cold start: "Starting GPU", "Loading weights"…).
    public struct Progress: Sendable, Equatable {
        public var characters: Int
        public var phase: String?
        public var fraction: Double?
    }

    /// Asks the model, and once more (with what was wrong) when the reply doesn't parse. `onCost` gets each request's
    /// cost as the server bills it (GPU seconds from the request's start, a cold start included).
    public static func ask<T>(client: any ChatStreaming, model: String, system: String, user: String,
                              onProgress: @escaping @Sendable (Progress) -> Void = { _ in },
                              onCost: @escaping @Sendable (Double) -> Void = { _ in },
                              parse: (String) throws -> T) async throws -> T {
        var messages: [ChatMessage] = [.system(system), .user(user)]
        var lastError: Error = MoviePlannerError.empty
        for attempt in 0..<2 {
            var request = ChatRequest(model: model, messages: messages)
            request.temperature = attempt == 0 ? 0.8 : 0.4
            var acc = ReplyAccumulator(inlineThinking: true)
            for try await e in client.stream(request) {
                try Task.checkCancellation()
                acc.add(e)
                let n = acc.content.count + acc.reasoning.count
                onProgress(Progress(characters: n, phase: n == 0 ? acc.status?.phase ?? acc.status?.state : nil,
                                    fraction: n == 0 ? acc.status?.progress : nil))
            }
            acc.finish()
            if let cost = acc.stats?.costUSD, cost > 0 { onCost(cost) }
            do {
                return try parse(acc.content)
            } catch {
                lastError = error
                messages += [.assistant(acc.content),
                             .user("That reply couldn't be used (\(error)). Reply again with only the JSON, in the shape asked for.")]
            }
        }
        throw lastError
    }
}
