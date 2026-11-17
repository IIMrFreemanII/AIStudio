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

/// Writing the script and the shot list with an LLM: the prompts, reading the replies (tolerant of what models do:
/// fences, names for ids, numbers as strings), and asking once more when a reply doesn't parse.
public enum MoviePlanner {
    // MARK: prompts

    /// How long each scene should run: the outline's seconds where it gave them; the rest of the film's target split
    /// by how much happens in each (its action and dialogue words), at least 4 s each. Without a budget, models write
    /// a shot or two per scene and the film comes out short.
    public static func sceneSeconds(for movie: MovieProject) -> [String: Double] {
        func words(_ s: String) -> Int { s.split(whereSeparator: { $0.isWhitespace }).count }
        var out: [String: Double] = [:]
        for s in movie.scenes { if let secs = s.seconds { out[s.id] = secs } }
        let rest = movie.scenes.filter { $0.seconds == nil }
        guard !rest.isEmpty else { return out }
        let left = max(Double(rest.count) * 4, Double(movie.targetSeconds) - out.values.reduce(0, +))
        let weights = rest.map { s in
            Double(max(8, words(s.action) + s.dialogue.reduce(0) { $0 + words($1.text) } * 2))
        }
        let total = weights.reduce(0, +)
        for (s, w) in zip(rest, weights) { out[s.id] = max(4, (left * w / total).rounded()) }
        return out
    }

    /// The longest a lip-synced talking shot can be with the movie's engines (a silent one: 7.5 s).
    public static func talkingCap(for movie: MovieProject) -> Double {
        if movie.videoEngine == .minimaxH3 { return MovieH3.maxSeconds }
        guard movie.lipSync else { return Shot.durationRange.upperBound }
        return movie.lipSyncEngine == .infiniteTalk ? Shot.durationRange.upperBound : MovieVoices.maxTalking
    }

    public static func shotPrompt(for movie: MovieProject, scene: MovieScene) -> (system: String, user: String) {
        let style = MovieStyle.named(movie.style)
        let silent = Shot.durationRange.upperBound
        let cap = talkingCap(for: movie)
        let system = """
        You are a director breaking a scene into shots for AI video generation. Each shot becomes one clip: first a \
        still keyframe image is generated, then it is animated for the shot's duration: 1–\(fmt(silent)) s (usually \
        3–6) for a shot without lip-sync, up to \(fmt(cap)) s for a talking shot. One clear action per shot.

        Cover conversations the way films do: open on an establishing shot or a two-shot master, then singles or \
        over-the-shoulder shots on each speaker in turn, with reaction shots of the listener; inserts for objects \
        that matter. Keep screen direction: each character stays on their side of the frame for the whole scene \
        (say which in "blocking", and put it in every keyframe: "Mara on the left"). Put each spoken line on a \
        medium shot or close-up of its speaker, so it can be lip-synced; a narrator is never on screen. Use wide \
        shots for places and action.

        For each shot write:
        - keyframe: a self-contained image prompt for the first frame: who is on screen, described by their look \
        (not only their name), where they stand in the frame, pose, expression, the place, lighting, composition. \
        No motion words.
        - motion: what moves during the shot (people, camera, environment), one or two sentences.
        - coverage: establishing, master, single, ots, reaction or insert.
        - dialogue: only the lines spoken during this shot (copied from the scene, every line in exactly one shot), \
        each by a character id.
        - talking: true when the speaker's face is on screen while they talk, in a medium shot or closer (it gets \
        lip-sync); false for wide shots, voice-over or an off-screen speaker.
        - continues: true only when the shot continues the previous shot's exact moment and place without a cut in \
        time (it will start from the previous clip's last frame).
        - sfx: the sound effects and ambience heard (no music).

        Reply with only JSON:
        {"blocking": {"<character id>": "left | right"}, \
        "shots": [{"duration": 5, "framing": "wide", "camera": "slow push in", "coverage": "master", "keyframe": "...", \
        "motion": "...", "characters": ["<character id>"], "dialogue": [{"character": "<id>", "text": "...", \
        "delivery": "..."}], "talking": false, "continues": false, \
        "transition": "cut | crossfade | fadeFromBlack | fadeToBlack", "sfx": "..."}]}
        """
        let cast = movie.characters.map {
            $0.isNarrator ? "- \($0.id): \($0.name) – narrator, heard only" : "- \($0.id): \($0.name) – \($0.look)"
        }.joined(separator: "\n")
        let place = movie.locations.first { $0.id == scene.location }
        let sides = scene.blocking.sorted { $0.key < $1.key }.map { "\($0.key) on the \($0.value.rawValue)" }
        let lines = scene.dialogue.map { "\($0.character): \($0.text)\($0.delivery.isEmpty ? "" : " (\($0.delivery))")" }
        let others = movie.scenes.map { "\($0.id). \($0.heading) – \($0.summary)" }.joined(separator: "\n")
        let seconds = sceneSeconds(for: movie)[scene.id] ?? 15
        let shots = max(1, Int((seconds / (scene.dialogue.count > 2 ? 5 : 4.5)).rounded()))
        let user = """
        Film: \(movie.title) – \(movie.logline)
        Style: \(style.name) – \(style.brief).

        Characters:
        \(cast.isEmpty ? "(none)" : cast)

        All scenes, for context:
        \(others)

        Break down this scene into about \(shots) shot\(shots == 1 ? "" : "s") that together run about \
        \(Int(seconds)) seconds:
        \(scene.id). \(scene.heading)\(place.map { "\nPlace: \($0.name) – \($0.look)" } ?? "")\(sides.isEmpty ? "" : "\nBlocking: " + sides.joined(separator: ", "))
        \(scene.beat.isEmpty ? "" : "Beat: \(scene.beat)\n")\(scene.action)
        \(lines.isEmpty ? "(no dialogue)" : "Dialogue:\n" + lines.joined(separator: "\n"))
        """
        return (system, user)
    }

    // MARK: reading replies

    struct LineReply: Decodable {
        var character: String?
        var text: String?
        var delivery: String?
    }

    private struct ShotsReply: Decodable {
        var shots: [ShotReply]?
        var blocking: [String: String]?
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
        var coverage: String?
    }

    /// A number or bool that may come as a string ("5", "5s", "true").
    struct Flexible: Decodable {
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

    static func decode<T: Decodable>(_ type: T.Type, from reply: String) throws -> T {
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

    static func lines(_ replies: [LineReply]?, cast: [MovieCharacter]) -> [DialogueLine] {
        (replies ?? []).compactMap { l in
            let text = clean(l.text)
            guard !text.isEmpty, let who = resolve(l.character, in: cast) else { return nil }
            return DialogueLine(character: who, text: text, delivery: clean(l.delivery))
        }
    }

    public static func parseShots(_ reply: String, scene: String, cast: [MovieCharacter],
                                  talkingCap: Double = Shot.durationRange.upperBound) throws -> [Shot] {
        try parseShotList(reply, scene: scene, cast: cast, talkingCap: talkingCap).shots
    }

    /// A scene's shots and who keeps which side of the frame. Durations are clamped to what the clips can be: a
    /// talking shot up to `talkingCap`, others up to 7.5 s. A narrator is never on screen.
    public static func parseShotList(_ reply: String, scene: String, cast: [MovieCharacter],
                                     talkingCap: Double = Shot.durationRange.upperBound)
        throws -> (shots: [Shot], blocking: [String: ScreenSide])
    {
        let r = try decode(ShotsReply.self, from: reply)
        let narrators = Set(cast.filter(\.isNarrator).map(\.id))
        let shots: [Shot] = (r.shots ?? []).compactMap { s in
            let keyframe = clean(s.keyframe)
            guard !keyframe.isEmpty else { return nil }
            let dialogue = lines(s.dialogue, cast: cast)
            let onScreen = dialogue.filter { !narrators.contains($0.character) }
            var who = (s.characters ?? []).compactMap { resolve($0, in: cast) }.filter { !narrators.contains($0) }
            for l in onScreen where (s.talking?.bool ?? false) && !who.contains(l.character) { who.append(l.character) }
            let talking = (s.talking?.bool ?? false) && !onScreen.isEmpty && !isWide(clean(s.framing))
            let seconds = s.duration?.number ?? 5
            let top = talking ? max(talkingCap, Shot.durationRange.upperBound) : Shot.durationRange.upperBound
            let coverage = clean(s.coverage).lowercased()
            return Shot(scene: scene,
                        duration: min(max(seconds.isFinite ? seconds : 5, Shot.durationRange.lowerBound), top),
                        framing: clean(s.framing), camera: clean(s.camera), keyframePrompt: keyframe,
                        motionPrompt: clean(s.motion), characters: who, dialogue: dialogue, sfx: clean(s.sfx),
                        talking: talking,
                        continuesPrevious: s.continues?.bool ?? false,
                        transition: ShotTransition(rawValue: clean(s.transition)) ?? .cut,
                        coverage: ShotCoverage(rawValue: coverage)
                            ?? (coverage.contains("over") ? .ots : coverage.contains("two") ? .master : nil))
        }
        guard !shots.isEmpty else { throw MoviePlannerError.invalid("no shots") }
        var out = shots
        out[0].continuesPrevious = false  // a scene starts fresh
        var blocking: [String: ScreenSide] = [:]
        for (ref, side) in r.blocking ?? [:] {
            if let id = resolve(ref, in: cast), let s = ScreenSide(rawValue: clean(side).lowercased()) { blocking[id] = s }
        }
        return (out, blocking)
    }

    static func fmt(_ s: Double) -> String { s == s.rounded() ? "\(Int(s))" : String(format: "%.1f", s) }

    /// Framings too far away for lip-sync to show.
    static func isWide(_ framing: String) -> Bool {
        let f = framing.lowercased()
        return ["wide", "establishing", "aerial", "long shot", "extreme long"].contains { f.contains($0) }
            && !f.contains("medium")
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

    /// Asks the model, and once more (with what was wrong) when the reply doesn't parse or `check` finds problems
    /// in it; a reply that parsed is kept when the second try is no better, and its problems go to `onProblems`.
    /// `onCost` gets each request's cost as the server bills it (GPU seconds from the request's start, a cold start
    /// included).
    public static func ask<T>(client: any ChatStreaming, model: String, system: String, user: String,
                              maxTokens: Int? = nil,
                              onProgress: @escaping @Sendable (Progress) -> Void = { _ in },
                              onCost: @escaping @Sendable (Double) -> Void = { _ in },
                              check: (T) -> [String] = { _ in [] },
                              onProblems: ([String]) -> Void = { _ in },
                              parse: (String) throws -> T) async throws -> T {
        var messages: [ChatMessage] = [.system(system), .user(user)]
        var lastError: Error = MoviePlannerError.empty
        var kept: (value: T, problems: [String])?
        for attempt in 0..<2 {
            var request = ChatRequest(model: model, messages: messages)
            request.temperature = attempt == 0 ? 0.8 : 0.4
            request.maxTokens = maxTokens
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
                let value = try parse(acc.content)
                let problems = check(value)
                if problems.isEmpty { return value }
                if kept == nil || problems.count <= kept!.problems.count { kept = (value, problems) }
                messages += [.assistant(acc.content),
                             .user("Almost: \(problems.joined(separator: "; ")). Reply again with the whole JSON, fixed.")]
            } catch {
                lastError = error
                messages += [.assistant(acc.content),
                             .user("That reply couldn't be used (\(error)). Reply again with only the JSON, in the shape asked for.")]
            }
        }
        if let kept {
            onProblems(kept.problems)
            return kept.value
        }
        throw lastError
    }
}
