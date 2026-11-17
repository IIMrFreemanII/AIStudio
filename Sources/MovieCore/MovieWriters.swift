import Foundation

/// The writers' room: a long film's story is written in steps, each one LLM request that sees what came before.
/// 1. The **bible**: title, logline, premise, tone, the score's theme, the characters (goal, flaw, arc,
///    relationships, look, voice, motif) and the places.
/// 2. The **outline**: acts → sequences → scenes, each scene with its place, cast, beat (goal / conflict / turn) and
///    its seconds on screen (rescaled to the film's length).
/// 3. Each **scene**, in order: its action and dialogue, written with the bible, the outline and the scenes just
///    before it in full.
/// 4. A **critique** per sequence: an editor's notes, and the scenes it revised.
///
/// Every user message starts with a `Task:` line naming its step (the fake server answers by it).
public enum MovieWriters {
    /// About how long a sequence runs.
    public static let sequenceMinutes = 2.5
    /// About how many scenes a minute of film has.
    public static let scenesPerMinute = 1.6
    /// Words of dialogue per second of a scene, by how many speak in it (a conversation leaves room for looks and
    /// action; people speak about 2.5 words a second).
    public static func dialogueWordsPerSecond(speakers: Int) -> Double {
        speakers >= 2 ? 1.6 : speakers == 1 ? 0.7 : 0
    }

    /// How many sequences and scenes a film of this length gets.
    public static func counts(targetSeconds: Int) -> (sequences: Int, scenes: Int) {
        let minutes = Double(targetSeconds) / 60
        let sequences = max(1, Int((minutes / sequenceMinutes).rounded()))
        let scenes = max(sequences, Int((minutes * scenesPerMinute).rounded()), 1)
        return (sequences, scenes)
    }

    // MARK: prompts

    static let filmable = """
    Every shot will be generated as a separate AI video clip of up to 7–15 seconds from a still image, so the story \
    is told through clear, filmable moments: few locations that recur, a small cast (at most 6 with lines), simple \
    physical action, faces and conversations. Avoid what video models get wrong: crowds, text on screen, fast fights, \
    complex hand work, characters changing clothes mid-scene.
    """

    public static func biblePrompt(for movie: MovieProject) -> (system: String, user: String) {
        let style = MovieStyle.named(movie.style)
        let (sequences, scenes) = counts(targetSeconds: movie.targetSeconds)
        let system = """
        You are a screenwriter developing an original short film for AI-generated video. \(filmable)

        Build characters with real wants and inner obstacles, relationships with tension, and an arc that pays off. \
        A narrator (heard, never seen) is allowed when the story is better told with one.

        Reply with only JSON, in this shape:
        {"title": "...", "logline": "one sentence",
         "premise": "the whole story in one or two paragraphs: setup, escalation, crisis, climax, resolution",
         "tone": "how it feels", "theme": "the score's main theme: instruments, melody, how it develops",
         "characters": [{"id": "lowercase-slug", "name": "...", "role": "protagonist | antagonist | mentor | ...", \
        "goal": "what they want", "flaw": "what stands in their way from inside", "arc": "how they change", \
        "relationships": "who they are to the others", \
        "look": "fixed visual description: age, face, hair, build, clothing, colours", \
        "voice": "how they sound: gender, age, pitch, texture, accent, pace, way of speaking", \
        "motif": "their musical motif", "narrator": false}],
         "locations": [{"id": "lowercase-slug", "name": "...", "look": "fixed visual description: architecture, \
        materials, light, colours, weather"}]}
        """
        let user = """
        Task: bible
        Idea: \(movie.idea.isEmpty ? movie.title : movie.idea)

        Style: \(style.name) – \(style.brief).
        Length: about \(minutes(movie.targetSeconds)) on screen, which will be about \(scenes) scene\(scenes == 1 ? "" : "s") \
        in \(sequences) sequence\(sequences == 1 ? "" : "s").
        """
        return (system, user)
    }

    public static func outlinePrompt(for movie: MovieProject) -> (system: String, user: String) {
        let (sequences, scenes) = counts(targetSeconds: movie.targetSeconds)
        let system = """
        You are a screenwriter outlining a film for AI-generated video. \(filmable)

        Structure it in three acts. Group the scenes into sequences (each a few minutes, with its own small arc). \
        Give every scene a beat: what the main character wants in it, what opposes them, and how the scene turns \
        (something must change by its end). Vary the rhythm: conversations, quiet visual moments, action. \
        Give each scene its seconds on screen; together they must add up to the film's length.

        Reply with only JSON:
        {"sequences": [{"title": "...", "act": 1, "summary": "one sentence", "scenes": [{"id": "s1", \
        "heading": "INT. PLACE – TIME", "location": "<location id>", "characters": ["<character id>"], \
        "summary": "one sentence", "beat": "goal / conflict / turn", "mood": "for the music: genre, tempo, \
        instruments", "seconds": 40}]}]}
        """
        let user = """
        Task: outline
        \(bible(movie))

        Length: \(movie.targetSeconds) seconds in all. Write \(sequences) sequence\(sequences == 1 ? "" : "s") with \
        about \(scenes) scene\(scenes == 1 ? "" : "s") in all.
        """
        return (system, user)
    }

    public static func scenePrompt(for movie: MovieProject, scene: MovieScene) -> (system: String, user: String) {
        let seconds = MoviePlanner.sceneSeconds(for: movie)[scene.id] ?? scene.seconds ?? 30
        let speakers = scene.characters.filter { id in movie.character(id).map { !$0.isNarrator } ?? false }.count
        let words = Int((seconds * dialogueWordsPerSecond(speakers: speakers)).rounded())
        let system = """
        You are a screenwriter writing one scene of a film for AI-generated video. \(filmable)

        Write dialogue people would actually say: subtext over exposition, each character in their own voice, short \
        lines that answer each other, interruptions and silences in the action. Every line has a delivery (how it's \
        said: "quietly, holding back tears"). The scene must play its beat and end changed. Keep continuity with the \
        scenes before it.

        Reply with only JSON:
        {"action": "screenplay action lines: what we see and hear, in order", \
        "dialogue": [{"character": "<character id>", "text": "...", "delivery": "how it's said"}], \
        "mood": "for the music: genre, tempo, instruments"}
        """
        let index = movie.scenes.firstIndex { $0.id == scene.id } ?? 0
        let before = movie.scenes[..<index].suffix(2).filter(\.isWritten)
        let place = movie.locations.first { $0.id == scene.location }
        let user = """
        Task: scene
        \(bible(movie))

        Outline:
        \(outline(movie, marking: scene.id))

        \(before.isEmpty ? "" : "The scenes just before, in full:\n" + before.map { sceneText($0, in: movie) }.joined(separator: "\n\n") + "\n\n")\
        Write scene \(scene.id): \(scene.heading)
        Where: \(place.map { "\($0.name) – \($0.look)" } ?? scene.heading)
        Who: \(scene.characters.compactMap { movie.character($0)?.name }.joined(separator: ", "))
        Beat: \(scene.beat.isEmpty ? scene.summary : scene.beat)
        Length: about \(Int(seconds)) seconds on screen\(words > 0 ? ", with about \(words) words of dialogue in all" : ", told without dialogue").
        """
        return (system, user)
    }

    public static func critiquePrompt(for movie: MovieProject, sequence: String) -> (system: String, user: String) {
        let scenes = movie.scenes(inSequence: sequence)
        let q = movie.sequences.first { $0.id == sequence }
        let system = """
        You are a script editor. Read this sequence of a film for AI-generated video and make it better: sharper, \
        more natural dialogue with subtext, distinct voices, scenes that play their beats and turn, continuity \
        (names, objects, time of day), setups that pay off, no line longer than it needs to be. Keep each scene's \
        length about the same, and keep what is filmable. \(filmable)

        Reply with only JSON: {"notes": "a short list of what you changed and why", \
        "scenes": [{"id": "<scene id>", "action": "...", "dialogue": [{"character": "<character id>", "text": "...", \
        "delivery": "..."}]}]}. Include only the scenes you changed, each in full.
        """
        let user = """
        Task: critique
        \(bible(movie))

        Outline:
        \(outline(movie, marking: nil))

        The sequence to edit: \(q?.title ?? sequence)\(q.map { $0.summary.isEmpty ? "" : " – \($0.summary)" } ?? "")

        \(scenes.map { sceneText($0, in: movie) }.joined(separator: "\n\n"))
        """
        return (system, user)
    }

    // MARK: prompt parts

    static func minutes(_ seconds: Int) -> String {
        seconds < 90 ? "\(seconds) seconds" : String(format: "%.1f minutes", Double(seconds) / 60)
            .replacingOccurrences(of: ".0 ", with: " ")
    }

    /// The bible as the later steps read it.
    static func bible(_ movie: MovieProject) -> String {
        let style = MovieStyle.named(movie.style)
        let cast = movie.characters.map { c in
            var parts = ["- \(c.id): \(c.name)" + (c.isNarrator ? " (narrator: heard only, never on screen)" : "")]
            if !c.role.isEmpty { parts.append("role: \(c.role)") }
            if !c.goal.isEmpty { parts.append("wants: \(c.goal)") }
            if !c.flaw.isEmpty { parts.append("flaw: \(c.flaw)") }
            if !c.arc.isEmpty { parts.append("arc: \(c.arc)") }
            if !c.relationships.isEmpty { parts.append("relationships: \(c.relationships)") }
            if !c.isNarrator, !c.look.isEmpty { parts.append("looks: \(c.look)") }
            if !c.voice.isEmpty { parts.append("speaks: \(c.voice)") }
            return parts.joined(separator: "; ")
        }
        let places = movie.locations.map { "- \($0.id): \($0.name) – \($0.look)" }
        return """
        Film: \(movie.title)\(movie.logline.isEmpty ? "" : " – \(movie.logline)")
        Style: \(style.name) – \(style.brief).\(movie.tone.isEmpty ? "" : "\nTone: \(movie.tone)")
        \(movie.premise.isEmpty ? "Idea: \(movie.idea)" : "Premise: \(movie.premise)")

        Characters:
        \(cast.isEmpty ? "(none yet)" : cast.joined(separator: "\n"))

        Locations:
        \(places.isEmpty ? "(none yet)" : places.joined(separator: "\n"))
        """
    }

    /// Every scene in a line, by sequence and act (`marking` gets an arrow).
    static func outline(_ movie: MovieProject, marking: String?) -> String {
        let seconds = MoviePlanner.sceneSeconds(for: movie)
        return movie.sequences.enumerated().map { i, q in
            let head = "Act \(q.act), sequence \(i + 1): \(q.title)"
            let rows = movie.scenes(inSequence: q.id).map { s in
                "\(s.id == marking ? "→ " : "  ")\(s.id). \(s.heading) (\(Int(seconds[s.id] ?? 0)) s) – "
                    + (s.beat.isEmpty ? s.summary : s.beat)
            }
            return ([head] + rows).joined(separator: "\n")
        }.joined(separator: "\n")
    }

    /// A written scene as screenplay text.
    public static func sceneText(_ scene: MovieScene, in movie: MovieProject) -> String {
        var out = ["\(scene.id). \(scene.heading)"]
        if !scene.action.isEmpty { out.append(scene.action) }
        for l in scene.dialogue {
            let name = (movie.character(l.character)?.name ?? l.character).uppercased()
            out.append("\(name)\(l.delivery.isEmpty ? "" : " (\(l.delivery))"): \(l.text)")
        }
        return out.joined(separator: "\n")
    }

    // MARK: replies

    public struct Bible: Sendable, Equatable {
        public var title: String
        public var logline: String
        public var premise: String
        public var tone: String
        public var theme: String
        public var characters: [MovieCharacter]
        public var locations: [MovieLocation]
    }

    public struct Outline: Sendable, Equatable {
        public var sequences: [MovieSequence]
        public var scenes: [MovieScene]
    }

    public struct SceneDraft: Sendable, Equatable {
        public var action: String
        public var dialogue: [DialogueLine]
        public var mood: String
    }

    public struct Critique: Sendable, Equatable {
        public var notes: String
        public var scenes: [String: SceneDraft]
    }

    private struct BibleReply: Decodable {
        var title: String?
        var logline: String?
        var premise: String?
        var tone: String?
        var theme: String?
        var characters: [CharacterReply]?
        var locations: [LocationReply]?
    }

    private struct CharacterReply: Decodable {
        var id: String?
        var name: String?
        var role: String?
        var goal: String?
        var flaw: String?
        var arc: String?
        var relationships: String?
        var look: String?
        var voice: String?
        var motif: String?
        var narrator: MoviePlanner.Flexible?
    }

    private struct LocationReply: Decodable {
        var id: String?
        var name: String?
        var look: String?
    }

    private struct OutlineReply: Decodable {
        var sequences: [SequenceReply]?
    }

    private struct SequenceReply: Decodable {
        var title: String?
        var act: MoviePlanner.Flexible?
        var summary: String?
        var scenes: [OutlineSceneReply]?
    }

    private struct OutlineSceneReply: Decodable {
        var id: String?
        var heading: String?
        var location: String?
        var characters: [String]?
        var summary: String?
        var beat: String?
        var mood: String?
        var seconds: MoviePlanner.Flexible?
    }

    private struct SceneReply: Decodable {
        var id: String?
        var action: String?
        var dialogue: [MoviePlanner.LineReply]?
        var mood: String?
    }

    private struct CritiqueReply: Decodable {
        var notes: Notes?
        var scenes: [SceneReply]?
    }

    /// Notes as text, or as a list of strings.
    private struct Notes: Decodable {
        var text: String
        init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let s = try? c.decode(String.self) { text = s }
            else if let a = try? c.decode([String].self) { text = a.map { "- " + $0 }.joined(separator: "\n") }
            else { text = "" }
        }
    }

    private static func clean(_ s: String?) -> String { (s ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }

    /// A unique slug for `raw` among `taken`.
    private static func uniqueSlug(_ raw: String, fallback: String, taken: [String]) -> String {
        var id = MoviePlanner.slug(raw)
        if id.isEmpty { id = fallback }
        let base = id
        var n = 2
        while taken.contains(id) { id = "\(base)-\(n)"; n += 1 }
        return id
    }

    public static func parseBible(_ reply: String) throws -> Bible {
        let r = try MoviePlanner.decode(BibleReply.self, from: reply)
        var cast: [MovieCharacter] = []
        for c in r.characters ?? [] {
            let name = clean(c.name).isEmpty ? clean(c.id) : clean(c.name)
            guard !name.isEmpty else { continue }
            let id = uniqueSlug(clean(c.id).isEmpty ? name : clean(c.id), fallback: "character", taken: cast.map(\.id))
            cast.append(MovieCharacter(id: id, name: name, look: clean(c.look), voice: clean(c.voice), role: clean(c.role),
                                       goal: clean(c.goal), flaw: clean(c.flaw), arc: clean(c.arc),
                                       relationships: clean(c.relationships), motif: clean(c.motif),
                                       isNarrator: c.narrator?.bool ?? false))
        }
        guard !cast.isEmpty else { throw MoviePlannerError.invalid("no characters") }
        var places: [MovieLocation] = []
        for l in r.locations ?? [] {
            let name = clean(l.name).isEmpty ? clean(l.id) : clean(l.name)
            guard !name.isEmpty else { continue }
            places.append(MovieLocation(id: uniqueSlug(clean(l.id).isEmpty ? name : clean(l.id), fallback: "place",
                                                       taken: places.map(\.id)),
                                        name: name, look: clean(l.look)))
        }
        return Bible(title: clean(r.title), logline: clean(r.logline), premise: clean(r.premise), tone: clean(r.tone),
                     theme: clean(r.theme), characters: cast, locations: places)
    }

    /// A place as the model named it (an id or a name) → a known id; empty when it's none of them.
    static func resolvePlace(_ ref: String?, in places: [MovieLocation]) -> String {
        let r = MoviePlanner.slug(clean(ref))
        guard !r.isEmpty else { return "" }
        return places.first { $0.id == r || MoviePlanner.slug($0.name) == r }?.id
            ?? places.first { r.contains($0.id) || $0.id.contains(r) }?.id ?? ""
    }

    /// The outline, its scenes' seconds scaled to the film's length when they're off by more than a tenth.
    public static func parseOutline(_ reply: String, cast: [MovieCharacter], locations: [MovieLocation],
                                    targetSeconds: Int) throws -> Outline {
        let r = try MoviePlanner.decode(OutlineReply.self, from: reply)
        var sequences: [MovieSequence] = []
        var scenes: [MovieScene] = []
        for q in r.sequences ?? [] {
            let rows = q.scenes ?? []
            guard !rows.isEmpty else { continue }
            let qid = "q\(sequences.count + 1)"
            sequences.append(MovieSequence(id: qid, title: clean(q.title).isEmpty ? "Sequence \(sequences.count + 1)" : clean(q.title),
                                           act: max(1, min(3, Int(q.act?.number ?? 1))), summary: clean(q.summary)))
            for s in rows {
                let heading = clean(s.heading)
                guard !heading.isEmpty || !clean(s.summary).isEmpty else { continue }
                var id = clean(s.id).isEmpty ? "s\(scenes.count + 1)" : MoviePlanner.slug(clean(s.id))
                if id.isEmpty || scenes.contains(where: { $0.id == id }) { id = "s\(scenes.count + 1)" }
                while scenes.contains(where: { $0.id == id }) { id += "b" }
                var who: [String] = []
                for c in s.characters ?? [] {
                    if let id = MoviePlanner.resolve(c, in: cast), !who.contains(id) { who.append(id) }
                }
                let secs = s.seconds?.number.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
                scenes.append(MovieScene(id: id, heading: heading.isEmpty ? "SCENE \(scenes.count + 1)" : heading,
                                         summary: clean(s.summary), mood: clean(s.mood), sequence: qid,
                                         location: resolvePlace(s.location, in: locations), characters: who,
                                         beat: clean(s.beat), seconds: secs))
            }
            if !scenes.contains(where: { $0.sequence == qid }) { sequences.removeLast() }
        }
        guard !scenes.isEmpty else { throw MoviePlannerError.invalid("no scenes") }
        // Seconds: missing ones get an even share of what's left, then all are scaled to the length when off.
        let target = Double(targetSeconds)
        let given = scenes.compactMap(\.seconds).reduce(0, +)
        let missing = scenes.filter { $0.seconds == nil }.count
        let share = missing > 0 ? max(8, (target - given) / Double(missing)) : 0
        for i in scenes.indices where scenes[i].seconds == nil { scenes[i].seconds = share }
        let total = scenes.compactMap(\.seconds).reduce(0, +)
        if total > 0, abs(total - target) / target > 0.1 {
            for i in scenes.indices {
                scenes[i].seconds = max(4, ((scenes[i].seconds ?? 0) * target / total * 2).rounded() / 2)
            }
        }
        return Outline(sequences: sequences, scenes: scenes)
    }

    public static func parseScene(_ reply: String, cast: [MovieCharacter]) throws -> SceneDraft {
        let r = try MoviePlanner.decode(SceneReply.self, from: reply)
        let draft = SceneDraft(action: clean(r.action), dialogue: MoviePlanner.lines(r.dialogue, cast: cast),
                               mood: clean(r.mood))
        guard !draft.action.isEmpty || !draft.dialogue.isEmpty else { throw MoviePlannerError.invalid("an empty scene") }
        return draft
    }

    public static func parseCritique(_ reply: String, cast: [MovieCharacter]) throws -> Critique {
        let r = try MoviePlanner.decode(CritiqueReply.self, from: reply)
        var scenes: [String: SceneDraft] = [:]
        for s in r.scenes ?? [] {
            let id = clean(s.id)
            let draft = SceneDraft(action: clean(s.action), dialogue: MoviePlanner.lines(s.dialogue, cast: cast),
                                   mood: clean(s.mood))
            guard !id.isEmpty, !draft.action.isEmpty || !draft.dialogue.isEmpty else { continue }
            scenes[id] = draft
        }
        return Critique(notes: r.notes?.text.trimmingCharacters(in: .whitespacesAndNewlines) ?? "", scenes: scenes)
    }

    // MARK: applying

    /// A new bible: title (when the movie has none), logline, premise, tone, theme, cast and places. The outline and
    /// script that follow are made from it, so they're dropped.
    public static func apply(_ b: Bible, to movie: inout MovieProject) {
        if !b.title.isEmpty, movie.title == "Untitled Movie" || movie.title.isEmpty { movie.title = b.title }
        movie.logline = b.logline
        movie.premise = b.premise
        movie.tone = b.tone
        movie.theme = b.theme
        movie.characters = b.characters
        movie.locations = b.locations
        movie.scenes = []
        movie.shots = []
        movie.sequences = [.wholeFilm]
        movie.critiqueNotes = [:]
        movie.reopen(.story)
    }

    /// A new outline: the sequences and the (unwritten) scenes; shots dropped.
    public static func apply(_ o: Outline, to movie: inout MovieProject) {
        movie.sequences = o.sequences
        movie.scenes = o.scenes
        movie.shots = []
        movie.critiqueNotes = [:]
        movie.reopen(.story)
    }

    /// A scene's writing: its action, lines and (when given) mood.
    public static func apply(_ d: SceneDraft, scene id: String, to movie: inout MovieProject) {
        guard let i = movie.scenes.firstIndex(where: { $0.id == id }) else { return }
        movie.scenes[i].action = d.action
        movie.scenes[i].dialogue = d.dialogue
        if !d.mood.isEmpty { movie.scenes[i].mood = d.mood }
    }

    /// The editor's revisions of a sequence's scenes (others are left alone) and its notes.
    public static func apply(_ c: Critique, sequence: String, to movie: inout MovieProject) {
        let ids = Set(movie.scenes(inSequence: sequence).map(\.id))
        for (id, d) in c.scenes where ids.contains(id) { apply(d, scene: id, to: &movie) }
        movie.critiqueNotes[sequence] = c.notes
    }

    // MARK: checking a shot list

    /// What's wrong with a scene's shot list: lines of the scene no shot speaks, and a runtime far from the scene's
    /// seconds (more than a quarter off). Empty when it's fine.
    public static func shotProblems(_ shots: [Shot], scene: MovieScene, seconds: Double) -> [String] {
        func norm(_ s: String) -> String { String(s.lowercased().filter { $0.isLetter || $0.isNumber }) }
        let spoken = Set(shots.flatMap { $0.dialogue.map { norm($0.text) } })
        var out: [String] = []
        let missing = scene.dialogue.filter { !spoken.contains(norm($0.text)) && !norm($0.text).isEmpty }
        if !missing.isEmpty {
            out.append("these lines are in no shot: " + missing.map { "\($0.character): \"\($0.text)\"" }.joined(separator: "; "))
        }
        let runtime = shots.reduce(0) { $0 + $1.duration }
        if seconds > 0, abs(runtime - seconds) / seconds > 0.25 {
            out.append("the shots run \(Int(runtime.rounded())) s but the scene should run about \(Int(seconds.rounded())) s")
        }
        return out
    }
}
