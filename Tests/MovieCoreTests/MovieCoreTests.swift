import Agent
import Foundation
import GraphRuntime
import LLMClient
@testable import MovieCore
import Testing

/// Streams canned replies, one per request, and records the requests.
final class ScriptedChat: ChatStreaming, @unchecked Sendable {
    var replies: [String]
    var requests: [ChatRequest] = []
    init(_ replies: [String]) { self.replies = replies }

    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        requests.append(request)
        let reply = replies.isEmpty ? "" : replies.removeFirst()
        return AsyncThrowingStream { c in
            for chunk in stride(from: 0, to: reply.count, by: 40).map({ i -> String in
                let s = reply.index(reply.startIndex, offsetBy: i)
                let e = reply.index(s, offsetBy: min(40, reply.count - i))
                return String(reply[s..<e])
            }) { c.yield(.content(chunk)) }
            c.yield(.stats(RequestStats(seconds: 10, usdPerSecond: 0.001, costUSD: 0.01)))
            c.finish()
        }
    }
}

final class Spent: @unchecked Sendable {
    private let lock = NSLock()
    private var sum = 0.0
    func add(_ x: Double) { lock.withLock { sum += x } }
    var total: Double { lock.withLock { sum } }
}

let scriptReply = """
Here's the script.
```json
{"title": "The Last Lantern", "logline": "A lighthouse keeper's daughter keeps the light on through a storm.",
 "characters": [
  {"id": "Mara", "name": "Mara", "look": "girl of 12, freckles, red raincoat", "voice": "young girl, bright, brave"},
  {"name": "Old Tomas", "look": "weathered man of 70, grey beard, wool sweater", "voice": "deep, gravelly, slow"}],
 "scenes": [
  {"id": "s1", "heading": "INT. LIGHTHOUSE – NIGHT", "summary": "The lamp fails.", "mood": "tense strings",
   "action": "Wind howls. The lamp flickers out.",
   "dialogue": [{"character": "MARA", "text": "Grandpa, the light!", "delivery": "scared"},
                {"character": "Tomas", "text": "Fetch the oil, quick."},
                {"character": "Nobody", "text": "dropped"}]},
  {"heading": "EXT. CLIFF – NIGHT", "action": "Mara runs through the rain."}]}
```
"""

@Test func parsesAScriptTolerantly() throws {
    let d = try MoviePlanner.parseScript(scriptReply)
    #expect(d.title == "The Last Lantern")
    #expect(d.characters.map(\.id) == ["mara", "old-tomas"])
    #expect(d.scenes.map(\.id) == ["s1", "s2"])
    // Names and capitals resolve to ids; an unknown speaker's line is dropped.
    #expect(d.scenes[0].dialogue.map(\.character) == ["mara", "old-tomas"])
    #expect(d.scenes[0].dialogue[0].delivery == "scared")
}

@Test func rejectsRepliesWithoutScenes() {
    #expect(throws: MoviePlannerError.noJSON) { try MoviePlanner.parseScript("Sorry, I can't.") }
    #expect(throws: MoviePlannerError.invalid("no scenes")) { try MoviePlanner.parseScript("{\"scenes\": []}") }
}

@Test func parsesShotsAndClampsThem() throws {
    let cast = [MovieCharacter(id: "mara", name: "Mara", look: "", voice: ""),
                MovieCharacter(id: "old-tomas", name: "Old Tomas", look: "", voice: "")]
    let reply = """
    {"shots": [
      {"duration": "12s", "framing": "wide", "keyframe": "lighthouse lamp room at night", "motion": "lamp flickers",
       "continues": true, "transition": "fadeFromBlack", "sfx": "wind"},
      {"duration": 3, "keyframe": "close-up of a girl", "characters": ["Mara"], "talking": "true",
       "dialogue": [{"character": "mara", "text": "Grandpa, the light!"}], "transition": "wipe"},
      {"keyframe": "", "motion": "skipped: no keyframe"},
      {"duration": 0.2, "keyframe": "old man turns", "talking": true}]}
    """
    let shots = try MoviePlanner.parseShots(reply, scene: "s1", cast: cast)
    #expect(shots.count == 3)
    #expect(shots[0].duration == 7.5 && shots[0].transition == .fadeFromBlack)
    #expect(shots[0].continuesPrevious == false, "a scene's first shot never continues the one before")
    #expect(shots[1].talking && shots[1].characters == ["mara"] && shots[1].transition == .cut)
    #expect(shots[2].duration == 1 && shots[2].talking == false, "talking needs a line")
}

@Test func scenesGetATimeBudgetAndShotCount() throws {
    var movie = MovieProject(targetSeconds: 60)
    movie.scenes = [MovieScene(id: "a", heading: "A", action: String(repeating: "word ", count: 60)),
                    MovieScene(id: "b", heading: "B", action: "short",
                               dialogue: [DialogueLine(character: "x", text: "one two three four five six seven eight nine ten")])]
    let s = MoviePlanner.sceneSeconds(for: movie)
    #expect(s["a"]! > s["b"]! && s["b"]! >= 4)
    #expect(abs(s["a"]! + s["b"]! - 60) <= 2)
    let p = MoviePlanner.shotPrompt(for: movie, scene: movie.scenes[0]).user
    #expect(p.contains("into about \(Int((s["a"]! / 4.5).rounded())) shots") && p.contains("about \(Int(s["a"]!)) seconds"))
}

@Test func noLipSyncOnWideShots() throws {
    let cast = [MovieCharacter(id: "c", name: "Captain", look: "", voice: "")]
    let reply = """
    {"shots": [{"framing": "wide", "keyframe": "a boat", "talking": true, "dialogue": [{"character": "c", "text": "See it."}]},
               {"framing": "medium wide", "keyframe": "the captain", "talking": true, "dialogue": [{"character": "c", "text": "Now."}]}]}
    """
    let shots = try MoviePlanner.parseShots(reply, scene: "s", cast: cast)
    #expect(!shots[0].talking && shots[1].talking)
}

@Test func askRetriesOnceWithTheProblem() async throws {
    let chat = ScriptedChat(["no json here", scriptReply])
    let spent = Spent()
    let draft = try await MoviePlanner.ask(client: chat, model: "m", system: "sys", user: "u",
                                           onCost: { spent.add($0) }, parse: MoviePlanner.parseScript)
    #expect(draft.scenes.count == 2)
    #expect(abs(spent.total - 0.02) < 1e-9, "both requests' costs are reported")
    #expect(chat.requests.count == 2)
    let retry = chat.requests[1].messages
    #expect(retry.count == 4 && retry[2].role == .assistant && retry[3].text.contains("no JSON"))
}

@Test func askGivesUpAfterTwoBadReplies() async {
    let chat = ScriptedChat(["nope", "still nope"])
    await #expect(throws: MoviePlannerError.noJSON) {
        try await MoviePlanner.ask(client: chat, model: "m", system: "s", user: "u", parse: MoviePlanner.parseScript)
    }
}

@Test func promptsCarryLengthStyleAndCast() throws {
    var movie = MovieProject(idea: "a storm at a lighthouse", style: "noir", targetSeconds: 60)
    let (system, user) = MoviePlanner.scriptPrompt(for: movie)
    #expect(system.contains("\"scenes\""))
    #expect(user.contains("about 60 seconds") && user.contains("about 12 shots") && user.contains("Film Noir"))
    MoviePlanner.apply(try MoviePlanner.parseScript(scriptReply), to: &movie)
    let shot = MoviePlanner.shotPrompt(for: movie, scene: movie.scenes[0])
    #expect(shot.user.contains("mara: Mara – girl of 12") && shot.user.contains("Grandpa, the light!"))
}

@Test func applyingAScriptDropsShotsAndLaterApprovals() throws {
    var movie = MovieProject(shots: [Shot(scene: "old")], approved: [.idea, .script, .shots, .cast])
    MoviePlanner.apply(try MoviePlanner.parseScript(scriptReply), to: &movie)
    #expect(movie.title == "The Last Lantern")
    #expect(movie.shots.isEmpty)
    #expect(movie.approved == [.idea, .script])
}

@Test func replacingASceneKeepsSceneOrder() {
    var movie = MovieProject(scenes: [MovieScene(id: "a", heading: "A"), MovieScene(id: "b", heading: "B")],
                             shots: [Shot(scene: "b", keyframePrompt: "b1"), Shot(scene: "a", keyframePrompt: "a1")])
    MoviePlanner.replaceShots([Shot(scene: "a", keyframePrompt: "a2"), Shot(scene: "a", keyframePrompt: "a3")],
                              scene: "a", in: &movie)
    #expect(movie.shots.map(\.keyframePrompt) == ["a2", "a3", "b1"])
}

@Test func stagesUnlockInOrder() {
    var m = MovieProject()
    #expect(m.currentStage == .idea && m.isUnlocked(.idea) && !m.isUnlocked(.script))
    m.approve(.idea)
    m.approve(.script)
    #expect(m.currentStage == .shots && m.isUnlocked(.shots) && !m.isUnlocked(.cast))
    m.reopen(.script)
    #expect(m.approved == [.idea])
}

@Test func storeRoundTripsAndLists() throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("movies-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: root) }
    let store = MovieStore(root: root)
    var a = MovieProject(title: "A", modified: Date(timeIntervalSince1970: 100))
    a.shots = [Shot(scene: "s1", dialogue: [DialogueLine(character: "x", text: "hi")],
                    keyframes: [Take(file: "keyframes/1.png", seed: 7, costUSD: 0.01)])]
    a.approved = [.idea]
    let b = MovieProject(title: "B", modified: Date(timeIntervalSince1970: 200))
    try store.save(a)
    try store.save(b)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("not-a-movie"), withIntermediateDirectories: true)
    #expect(try store.load(a.id) == a)
    #expect(store.list().map(\.title) == ["B", "A"])
    #expect(store.list().last?.stage == .script && store.list().last?.shots == 1)
    try store.delete(b.id, useTrash: false)
    #expect(store.list().map(\.title) == ["A"])
}

@Test func templateBindsSlots() throws {
    let graph = Graph(nodes: [
        GraphNode(id: "p", type: "TextEncode", inputs: ["prompt": .value(.string("old"))], position: GraphPoint(x: 0, y: 0)),
        GraphNode(id: "s", type: "Sampler", inputs: ["seed": .value(.int(1))], position: GraphPoint(x: 0, y: 0)),
    ])
    let t = MovieTemplate(name: "t", graph: graph, slots: ["prompt": .init(node: "p", input: "prompt"),
                                                          "seed": .init(node: "s", input: "seed"),
                                                          "gone": .init(node: "zz", input: "x")])
    let g = try t.bind(["prompt": .string("new"), "seed": .int(42), "negative": .string("n")], optional: ["negative"])
    #expect(g["p"]?.inputs["prompt"] == .value(.string("new")))
    #expect(g["s"]?.inputs["seed"] == .value(.int(42)))
    #expect(t.brokenSlots == ["gone"])
    #expect(throws: MovieTemplate.BindError.unknownSlot("width")) { try t.bind(["width": .int(1)]) }
    #expect(throws: MovieTemplate.BindError.missingNode(slot: "gone", node: "zz")) { try t.bind(["gone": .int(1)]) }

    let url = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-\(UUID().uuidString)/t.json")
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
    try t.save(url)
    #expect(try MovieTemplate.load(url) == t)
}

@Test func llmCostCountsColdStartAndIdle() {
    let cold = MovieCost.llm(label: "Script", requests: 1, gpu: "H100", usdPerSecond: 0.001, warm: false,
                             coldStartSeconds: 100, secondsPerRequest: 50, idleSeconds: 300)
    #expect(cold.seconds == 450 && abs(cold.usd - 0.45) < 1e-9)
    let warm = MovieCost.llm(label: "Shots", requests: 4, gpu: "H100", usdPerSecond: 0.001, warm: true,
                             coldStartSeconds: 100, secondsPerRequest: 50, idleSeconds: 300)
    #expect(warm.seconds == 500)
    #expect(CostEstimate.format(0.004) == "<$0.01" && CostEstimate.format(1.2) == "$1.20")
}
