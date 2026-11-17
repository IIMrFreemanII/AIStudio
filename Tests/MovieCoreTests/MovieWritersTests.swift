import Foundation
@testable import MovieCore
import Testing

// MARK: the writers' room

private let cast = [
    MovieCharacter(id: "mara", name: "Mara", look: "girl of 12, red raincoat", voice: "bright", goal: "keep the light on",
                   flaw: "afraid of the dark", arc: "scared to steady"),
    MovieCharacter(id: "tomas", name: "Tomas", look: "man of 70", voice: "gravelly"),
    MovieCharacter(id: "narrator", name: "Narrator", look: "", voice: "warm", isNarrator: true),
]
private let places = [MovieLocation(id: "lighthouse", name: "Lamp room", look: "brass and salt glass"),
                      MovieLocation(id: "cliff", name: "Cliff path", look: "wet black rock")]

private let outlineReply = """
{"sequences": [
  {"title": "The Lamp Fails", "act": "1", "summary": "dark", "scenes": [
    {"id": "s1", "heading": "INT. LIGHTHOUSE – NIGHT", "location": "Lamp room", "characters": ["Mara", "TOMAS", "nobody"],
     "beat": "wants light / lamp dies / goes alone", "seconds": "120"},
    {"id": "s1", "heading": "INT. STAIRS – NIGHT", "location": "stairs", "characters": ["mara"], "seconds": 60}]},
  {"title": "", "act": 7, "scenes": []},
  {"title": "The Run", "act": 3, "scenes": [
    {"heading": "EXT. CLIFF – NIGHT", "location": "cliff", "characters": ["mara"], "beat": "climb / wind / top"}]}]}
"""

private func outlined() throws -> MovieProject {
    var m = MovieProject(title: "The Last Lantern", idea: "a storm", targetSeconds: 300)
    m.characters = cast
    m.locations = places
    MovieWriters.apply(try MovieWriters.parseOutline(outlineReply, cast: cast, locations: places, targetSeconds: 300), to: &m)
    return m
}

@Test func anOutlineBecomesSequencesAndScenesThatAddUpToTheLength() throws {
    let o = try MovieWriters.parseOutline(outlineReply, cast: cast, locations: places, targetSeconds: 300)
    // An empty sequence is dropped; ids run in order; acts are kept in 1…3.
    #expect(o.sequences.map(\.id) == ["q1", "q2"] && o.sequences.map(\.act) == [1, 3])
    #expect(o.sequences[1].title == "The Run")
    // Duplicate and missing scene ids get their own; places and people resolve by name; unknowns are dropped.
    #expect(o.scenes.map(\.id) == ["s1", "s2", "s3"])
    #expect(o.scenes.map(\.sequence) == ["q1", "q1", "q2"])
    #expect(o.scenes[0].location == "lighthouse" && o.scenes[1].location == "" && o.scenes[2].location == "cliff")
    #expect(o.scenes[0].characters == ["mara", "tomas"])
    // 120 + 60 given, the missing one gets the rest: 300 in all.
    #expect(o.scenes.compactMap(\.seconds) == [120, 60, 120])

    // Seconds far from the length are scaled to it.
    let short = try MovieWriters.parseOutline(outlineReply, cast: cast, locations: places, targetSeconds: 120)
    let total = short.scenes.compactMap(\.seconds).reduce(0, +)
    #expect(abs(total - 120) <= 2, "scaled to the film's length: \(total)")
}

@Test func theOutlinesSecondsAreTheScenesBudgets() throws {
    let m = try outlined()
    let s = MoviePlanner.sceneSeconds(for: m)
    #expect(s == ["s1": 120, "s2": 60, "s3": 120])
    #expect(m.sequences.count == 2 && m.approved.isEmpty)
}

@Test func filmLengthsGetSequencesAndScenes() {
    #expect(MovieWriters.counts(targetSeconds: 30) == (1, 1))
    #expect(MovieWriters.counts(targetSeconds: 300) == (2, 8))
    #expect(MovieWriters.counts(targetSeconds: 1200) == (8, 32))
}

@Test func eachStepsPromptCarriesWhatCameBefore() throws {
    var m = try outlined()
    m.style = "noir"
    m.premise = "The lamp fails in a storm."
    let bible = MovieWriters.biblePrompt(for: m)
    #expect(bible.user.hasPrefix("Task: bible") && bible.user.contains("about 5 minutes") && bible.user.contains("Film Noir"))
    #expect(bible.system.contains("\"goal\"") && bible.system.contains("\"locations\""))

    let outline = MovieWriters.outlinePrompt(for: m)
    #expect(outline.user.hasPrefix("Task: outline") && outline.user.contains("wants: keep the light on")
            && outline.user.contains("Lamp room – brass") && outline.user.contains("300 seconds"))
    #expect(outline.user.contains("narrator: heard only"))

    // A scene sees the outline (itself marked), the scenes just before it in full, and its dialogue budget.
    m.scenes[0].action = "Wind howls."
    m.scenes[0].dialogue = [DialogueLine(character: "mara", text: "Grandpa, the light!", delivery: "scared")]
    let scene = MovieWriters.scenePrompt(for: m, scene: m.scenes[1])
    #expect(scene.user.hasPrefix("Task: scene"))
    #expect(scene.user.contains("→ s2.") && scene.user.contains("  s1."))
    #expect(scene.user.contains("MARA (scared): Grandpa, the light!"))
    // Mara alone for 60 s: 0.7 words a second.
    #expect(scene.user.contains("about 42 words of dialogue"))
    let two = MovieWriters.scenePrompt(for: m, scene: m.scenes[0]).user
    #expect(two.contains("about 192 words of dialogue") && two.contains("Where: Lamp room – brass"))

    let critique = MovieWriters.critiquePrompt(for: m, sequence: "q1")
    #expect(critique.user.hasPrefix("Task: critique") && critique.user.contains("The Lamp Fails")
            && critique.user.contains("Wind howls.") && !critique.user.contains("EXT. CLIFF – NIGHT\nclimb"))
}

@Test func scenesAndCritiquesAreWrittenIn() throws {
    var m = try outlined()
    let scene = try MovieWriters.parseScene("""
    {"action": "Wind howls.", "dialogue": [{"character": "Mara", "text": "Grandpa!", "delivery": "scared"},
     {"character": "ghost", "text": "dropped"}], "mood": "tense"}
    """, cast: cast)
    #expect(scene.dialogue.map(\.character) == ["mara"])
    MovieWriters.apply(scene, scene: "s1", to: &m)
    #expect(m.scenes[0].isWritten && m.scenes[0].mood == "tense" && !m.scenes[1].isWritten)
    #expect(throws: MoviePlannerError.invalid("an empty scene")) { try MovieWriters.parseScene("{}", cast: cast) }

    // The editor revises s1 and tries s3 (another sequence): only s1 changes, and the notes are kept.
    let c = try MovieWriters.parseCritique("""
    {"notes": ["Slower lamp", "Fewer words"], "scenes": [
      {"id": "s1", "action": "The lamp gutters and dies.", "dialogue": [{"character": "tomas", "text": "Go."}]},
      {"id": "s3", "action": "changed elsewhere"}]}
    """, cast: cast)
    MovieWriters.apply(c, sequence: "q1", to: &m)
    #expect(m.scenes[0].action == "The lamp gutters and dies." && m.scenes[0].dialogue.map(\.text) == ["Go."])
    #expect(m.scenes[2].action.isEmpty)
    #expect(m.critiqueNotes["q1"] == "- Slower lamp\n- Fewer words")
}

@Test func shotListsAreCheckedForMissingLinesAndRuntime() {
    let scene = MovieScene(id: "s1", heading: "H", dialogue: [
        DialogueLine(character: "mara", text: "Grandpa, the light!"), DialogueLine(character: "tomas", text: "Go.")])
    let ok = [Shot(scene: "s1", duration: 5, dialogue: [DialogueLine(character: "mara", text: "grandpa the light")]),
              Shot(scene: "s1", duration: 5, dialogue: [DialogueLine(character: "tomas", text: "Go!")])]
    #expect(MovieWriters.shotProblems(ok, scene: scene, seconds: 11).isEmpty)
    let bad = MovieWriters.shotProblems([ok[0]], scene: scene, seconds: 20)
    #expect(bad.count == 2 && bad[0].contains("\"Go.\"") && bad[1].contains("run 5 s"))
}

@Test func askAsksAgainWhenTheCheckFailsAndKeepsTheBetterReply() async throws {
    let reply = { (n: Int) in "{\"shots\": [" + (0..<n).map { _ in "{\"duration\": 5, \"keyframe\": \"k\"}" }.joined(separator: ",") + "]}" }
    let scene = MovieScene(id: "s1", heading: "H")
    func check(_ s: [Shot]) -> [String] { MovieWriters.shotProblems(s, scene: scene, seconds: 20) }

    // Fixed on the second try.
    var chat = ScriptedChat([reply(1), reply(4)])
    var problems: [String] = []
    var shots = try await MoviePlanner.ask(client: chat, model: "m", system: "s", user: "u", check: check,
                                           onProblems: { problems = $0 }) { try MoviePlanner.parseShots($0, scene: "s1", cast: []) }
    #expect(shots.count == 4 && problems.isEmpty && chat.requests.count == 2)
    #expect(chat.requests[1].messages.last?.text.contains("Almost: the shots run 5 s") == true)

    // Not fixed, and the second reply doesn't even parse: the first is kept with its problems.
    chat = ScriptedChat([reply(1), "sorry"])
    shots = try await MoviePlanner.ask(client: chat, model: "m", system: "s", user: "u", check: check,
                                       onProblems: { problems = $0 }) { try MoviePlanner.parseShots($0, scene: "s1", cast: []) }
    #expect(shots.count == 1 && problems.count == 1)
}

@Test func shotListsKeepScreenSidesCoverageAndLongerTalkingShots() throws {
    let reply = """
    {"blocking": {"Mara": "left", "tomas": "RIGHT", "ghost": "left"},
     "shots": [
      {"duration": 4, "framing": "wide", "coverage": "two-shot", "keyframe": "both at the lamp"},
      {"duration": 11, "framing": "close-up", "coverage": "single", "keyframe": "Mara on the left", "talking": true,
       "dialogue": [{"character": "mara", "text": "Grandpa, the light!"}]},
      {"duration": 11, "framing": "close-up", "coverage": "over the shoulder", "keyframe": "Tomas", "talking": true,
       "characters": ["narrator"], "dialogue": [{"character": "narrator", "text": "That night, she was alone."}]}]}
    """
    let (shots, blocking) = try MoviePlanner.parseShotList(reply, scene: "s1", cast: cast, talkingCap: 12)
    #expect(blocking == ["mara": .left, "tomas": .right])
    #expect(shots.map(\.coverage) == [.master, .single, .ots])
    // A lip-synced shot may run to the engine's cap; a narrator is never on screen and never lip-synced.
    #expect(shots[1].talking && shots[1].duration == 11)
    #expect(!shots[2].talking && shots[2].duration == 7.5 && shots[2].characters.isEmpty)

    var m = MovieProject()
    #expect(MoviePlanner.talkingCap(for: m) == 12)
    m.lipSyncEngine = .infiniteTalk
    #expect(MoviePlanner.talkingCap(for: m) == 7.5)
    m.videoEngine = .minimaxH3
    #expect(MoviePlanner.talkingCap(for: m) == 15)
}

@Test func theShotPromptUsesThePlaceBeatBlockingAndCaps() throws {
    var m = try outlined()
    m.scenes[0].blocking = ["mara": .left]
    let p = MoviePlanner.shotPrompt(for: m, scene: m.scenes[0])
    #expect(p.user.contains("Place: Lamp room – brass and salt glass") && p.user.contains("Blocking: mara on the left"))
    #expect(p.user.contains("Beat: wants light") && p.user.contains("Narrator – narrator, heard only"))
    #expect(p.system.contains("up to 12 s for a talking shot") && p.system.contains("\"blocking\""))
    #expect(p.user.contains("about 120 seconds"))
}

@Test func storyFieldsRoundTripAndOldCastsDecode() throws {
    var m = try outlined()
    m.premise = "p"
    m.theme = "t"
    m.critiqueNotes = ["q1": "n"]
    m.shots = [Shot(scene: "s1", coverage: .reaction)]
    let e = JSONEncoder()
    let back = try JSONDecoder().decode(MovieProject.self, from: e.encode(m))
    #expect(back == m)
    // An outlined movie with the idea approved keeps its story unapproved after a save (it isn't an old movie).
    m.approve(.idea)
    #expect(try JSONDecoder().decode(MovieProject.self, from: e.encode(m)).approved == [.idea])
    // A character saved before the bible decodes with empty story fields.
    let old = try JSONDecoder().decode(MovieCharacter.self, from: Data("""
    {"id": "a", "name": "A", "look": "l", "voice": "v"}
    """.utf8))
    #expect(old.goal.isEmpty && !old.isNarrator)
}
