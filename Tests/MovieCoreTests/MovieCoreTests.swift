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

let bibleReply = """
Here's the bible.
```json
{"title": "The Last Lantern", "logline": "A lighthouse keeper's daughter keeps the light on through a storm.",
 "premise": "The lamp fails; Mara carries oil up the cliff.", "tone": "tense", "theme": "music box",
 "characters": [
  {"id": "Mara", "name": "Mara", "role": "protagonist", "goal": "keep the light on", "flaw": "afraid of the dark",
   "look": "girl of 12, freckles, red raincoat", "voice": "young girl, bright, brave"},
  {"name": "Old Tomas", "look": "weathered man of 70, grey beard, wool sweater", "voice": "deep, gravelly, slow"},
  {"name": "Narrator", "voice": "warm, older woman", "narrator": "true"},
  {"name": ""}],
 "locations": [{"id": "Lighthouse", "name": "Lamp room", "look": "brass, salt glass"}, {"name": "Cliff path"}]}
```
"""

@Test func parsesABibleTolerantly() throws {
    let b = try MovieWriters.parseBible(bibleReply)
    #expect(b.title == "The Last Lantern" && b.premise.contains("oil") && b.theme == "music box")
    #expect(b.characters.map(\.id) == ["mara", "old-tomas", "narrator"])
    #expect(b.characters[0].goal == "keep the light on" && b.characters[0].flaw == "afraid of the dark")
    #expect(b.characters[2].isNarrator && !b.characters[0].isNarrator)
    #expect(b.locations.map(\.id) == ["lighthouse", "cliff-path"])
}

@Test func rejectsRepliesWithoutScenes() {
    #expect(throws: MoviePlannerError.noJSON) { try MovieWriters.parseBible("Sorry, I can't.") }
    #expect(throws: MoviePlannerError.invalid("no scenes")) {
        try MovieWriters.parseOutline("{\"sequences\": []}", cast: [], locations: [], targetSeconds: 60)
    }
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
    let chat = ScriptedChat(["no json here", bibleReply])
    let spent = Spent()
    let bible = try await MoviePlanner.ask(client: chat, model: "m", system: "sys", user: "u", maxTokens: 900,
                                           onCost: { spent.add($0) }, parse: MovieWriters.parseBible)
    #expect(bible.characters.count == 3)
    #expect(abs(spent.total - 0.02) < 1e-9, "both requests' costs are reported")
    #expect(chat.requests.count == 2 && chat.requests.allSatisfy { $0.maxTokens == 900 })
    let retry = chat.requests[1].messages
    #expect(retry.count == 4 && retry[2].role == .assistant && retry[3].text.contains("no JSON"))
}

@Test func askGivesUpAfterTwoBadReplies() async {
    let chat = ScriptedChat(["nope", "still nope"])
    await #expect(throws: MoviePlannerError.noJSON) {
        try await MoviePlanner.ask(client: chat, model: "m", system: "s", user: "u", parse: MovieWriters.parseBible)
    }
}

@Test func applyingABibleDropsTheOutlineShotsAndLaterApprovals() throws {
    var movie = MovieProject(scenes: [MovieScene(id: "s1", heading: "H")], shots: [Shot(scene: "old")],
                             approved: [.idea, .story, .script, .shots, .cast])
    MovieWriters.apply(try MovieWriters.parseBible(bibleReply), to: &movie)
    #expect(movie.title == "The Last Lantern" && movie.premise.contains("oil") && movie.locations.count == 2)
    #expect(movie.shots.isEmpty && movie.scenes.isEmpty && movie.sequences == [.wholeFilm])
    #expect(movie.approved == [.idea] && movie.sequenceApproved.isEmpty)
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
    #expect(m.currentStage == .story && m.isUnlocked(.story) && !m.isUnlocked(.script))
    m.approve(.story)
    m.approve(.script)
    // The cast is the whole film's: it opens with the script, beside the shot list.
    #expect(m.currentStage == .shots && m.isUnlocked(.shots) && m.isUnlocked(.cast) && !m.isUnlocked(.keyframes))
    m.reopen(.story)
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
    #expect(store.list().last?.stage == .story && store.list().last?.shots == 1)
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

// MARK: cast and keyframes

private func castMovie() -> MovieProject {
    var m = MovieProject(style: "anime")
    let sheet = Take(file: "cast/mara-1.png")
    m.characters = [
        MovieCharacter(id: "mara", name: "Mara", look: "girl of 12, red raincoat", voice: "", sheets: [sheet],
                       selectedSheet: sheet.id),
        MovieCharacter(id: "tomas", name: "Tomas", look: "old man", voice: ""),
    ]
    let key = Take(file: "keyframes/1.png")
    m.scenes = [MovieScene(id: "s1", heading: "INT. LIGHTHOUSE – NIGHT"), MovieScene(id: "s2", heading: "EXT. CLIFF")]
    m.shots = [
        Shot(scene: "s1", framing: "wide", keyframePrompt: "the lamp room", characters: ["mara"], keyframes: [key],
             selectedKeyframe: key.id),
        Shot(scene: "s1", framing: "close-up", keyframePrompt: "Mara shouts", characters: ["mara"],
             continuesPrevious: true),
        Shot(scene: "s2", keyframePrompt: "", characters: [], continuesPrevious: true),
        Shot(scene: "s2", keyframePrompt: "Tomas waits", characters: ["tomas"]),
    ]
    return m
}

@Test func keyframesAreMadeFromSheetsAndThePreviousShot() {
    let m = castMovie()
    let close = MovieArt.references(for: m.shots[1], in: m)
    #expect(close.map(\.label) == ["Mara", "the previous shot"] && close.map(\.file) == ["cast/mara-1.png", "keyframes/1.png"])
    // A shot that "continues" across a scene change doesn't take the other scene's keyframe.
    #expect(MovieArt.references(for: m.shots[2], in: m).isEmpty)
    #expect(MovieArt.missingSheets(for: m.shots[3], in: m) == ["Tomas"])
    let prompt = MovieArt.keyframePrompt(for: m.shots[1], in: m, references: close)
    #expect(prompt.hasPrefix("Picture 1 is Mara. Picture 2 is the previous shot"))
    #expect(prompt.contains("Mara shouts.") && prompt.contains("Close-up shot.") && prompt.contains("anime key visual"))
    // No pictures: a plain prompt, the scene's heading when the shot has none.
    let plain = MovieArt.keyframePrompt(for: m.shots[2], in: m, references: [])
    #expect(plain.hasPrefix("EXT. CLIFF.") && !plain.contains("Picture"))
    #expect(!m.castIsReady && !m.keyframesAreReady)
}

@Test func keyframeRenderBindsPicturesAndDropsTheUnused() throws {
    let m = castMovie()
    let folder = URL(fileURLWithPath: "/movies/x")
    let (render, refs) = MovieArt.keyframeRender(m.shots[0], movie: m, folder: folder, seed: 5, server: "gpu1")
    #expect(refs.count == 1 && render.unusedPictures == ["image2", "image3"])
    #expect(render.values["image1"] == .string("/movies/x/cast/mara-1.png") && render.values["width"] == .int(1344))
    let graph = try MovieArt.editTemplate().bind(render.values, optional: Set(render.values.keys),
                                                 remove: render.unusedPictures)
    #expect(graph.nodes.map(\.id).sorted() == ["edit", "out", "picture1"])
    let edit = try #require(graph["edit"])
    #expect(edit.inputs["image1"] == .link(node: "picture1", output: "image") && edit.inputs["image2"] == nil)
    #expect(edit.inputs["seed"] == .value(.int(5)) && edit.inputs["server"] == .value(.string("gpu1")))
    #expect(graph["picture1"]?.inputs["path"] == .value(.string("/movies/x/cast/mara-1.png")))
    // Sheets: Qwen-Image at its wide size.
    let sheet = MovieArt.sheetRender(m.characters[0], movie: m, seed: 1, server: "")
    let g = try MovieArt.textToImageTemplate(name: "character-sheet").bind(sheet.values)
    #expect(g["image"]?.type == "RemoteQwenImage" && g["image"]?.inputs["width"] == .value(.int(1664)))
    #expect(MovieArt.sheetPrompt(m.characters[0], style: .anime).contains("Mara, girl of 12, red raincoat"))
}

@Test func planPartsIgnoreTakes() {
    var m = castMovie()
    let before = m.shots.map(\.planPart)
    m.shots[1].keyframes.append(Take(file: "k.png"))
    #expect(m.shots.map(\.planPart) == before)
    m.shots[1].framing = "medium"
    #expect(m.shots.map(\.planPart) != before)
    let cast = m.characters.map(\.scriptPart)
    m.characters[1].sheets.append(Take(file: "t.png"))
    #expect(m.characters.map(\.scriptPart) == cast)
}

@Test func gpuTimingsLearnAndEstimate() throws {
    var t = GPUTimings()
    // Defaults until a job ran, scaled by pixels.
    #expect(t.seconds(task: "qwen-image-edit", megapixels: 2) == 16 && t.loadSeconds(task: "qwen-image-edit") == 60)
    #expect(t.seconds(task: "unknown", megapixels: 1) == 60)
    let now = Date()
    t.record(task: "qwen-image-edit", megapixels: 1.03, gpuSeconds: 8, loadSeconds: 70, at: now)
    t.record(task: "qwen-image-edit", megapixels: 1.03, gpuSeconds: 10, loadSeconds: 0, at: now)
    #expect(t.seconds(task: "qwen-image-edit", megapixels: 1.0) == 9)  // the average, same bucket
    #expect(abs(t.seconds(task: "qwen-image-edit", megapixels: 2.06) - 18) < 1e-9)  // scaled from the learned size
    #expect(t.loadSeconds(task: "qwen-image-edit") == 70)  // only cold starts count
    #expect(t.isWarm(task: "qwen-image-edit", idleSeconds: 120, now: now.addingTimeInterval(60)))
    #expect(!t.isWarm(task: "qwen-image-edit", idleSeconds: 120, now: now.addingTimeInterval(110)))
    let line = MovieCost.images(label: "Keyframes", task: "qwen-image-edit", jobs: 4, megapixels: 1, gpu: "H100",
                                usdPerSecond: 0.001, timings: t, warm: false, idleSeconds: 120)
    #expect(line.seconds == 70 + 4 * 9 + 120 && abs(line.usd - 0.226) < 1e-9)
    // Over 3 GPUs at once: each extra one loads the model and idles too (a warm one skips its load).
    let wide = MovieCost.images(label: "Keyframes", task: "qwen-image-edit", jobs: 4, megapixels: 1, gpu: "H100",
                                usdPerSecond: 0.001, timings: t, warm: true, idleSeconds: 120, containers: 3)
    #expect(wide.seconds == 2 * 70 + 4 * 9 + 3 * 120 && wide.gpu == "3× H100")
    // Busy GPUs (other work) aren't ours to start: only the free ones are, and with none free the jobs wait.
    #expect(MovieCost.gpus(jobs: 10, parallel: 4, busy: 0, limit: 4) == (4, false))
    #expect(MovieCost.gpus(jobs: 10, parallel: 4, busy: 3, limit: 4) == (1, false))
    #expect(MovieCost.gpus(jobs: 2, parallel: 4, busy: 1, limit: 4) == (2, false))
    #expect(MovieCost.gpus(jobs: 5, parallel: 4, busy: 4, limit: 4) == (1, true))
    #expect(MovieCost.gpus(jobs: 5, parallel: 1, busy: 0, limit: 4) == (1, false))
    // Wall time: a cold GPU loads first, then the jobs in rounds of the GPUs; an extra GPU is always cold.
    #expect(MovieCost.wallSeconds(jobs: 10, containers: 1, perJob: 4, load: 40, warm: false) == 80)
    #expect(MovieCost.wallSeconds(jobs: 10, containers: 4, perJob: 4, load: 40, warm: true) == 52)
    #expect(MovieCost.wallSeconds(jobs: 3, containers: 1, perJob: 10, load: 40, warm: true) == 30)
    // Several GPUs only by default when they save five minutes or more: not for quick sound effects, yes for clips.
    let effects1 = CostEstimate.Plan(parallel: 1, lines: [], wallSeconds: 80)
    let effects4 = CostEstimate.Plan(parallel: 4, lines: [], wallSeconds: 52)
    #expect(!MovieCost.prefersParallel(single: effects1, parallel: effects4))
    let clips1 = CostEstimate.Plan(parallel: 1, lines: [], wallSeconds: 90 + 4 * 450)
    let clips4 = CostEstimate.Plan(parallel: 4, lines: [], wallSeconds: 90 + 450)
    #expect(MovieCost.prefersParallel(single: clips1, parallel: clips4))
    // An estimate with plans shows the chosen one's lines.
    let a = CostEstimate.Line(label: "x", count: 1, gpu: "H100", seconds: 10, usd: 0.1)
    let b = CostEstimate.Line(label: "x", count: 1, gpu: "4× H100", seconds: 40, usd: 0.4)
    let e = CostEstimate(title: "t", plans: [.init(parallel: 1, lines: [a], wallSeconds: 1),
                                             .init(parallel: 4, lines: [b], wallSeconds: 1)], chosen: 4)
    #expect(e.lines == [b] && e.plans.map(\.usd) == [0.1, 0.4])
    // Never more GPUs than jobs.
    #expect(MovieCost.images(label: "", task: "qwen-image-edit", jobs: 1, megapixels: 1, gpu: "H100", usdPerSecond: 0.001,
                             timings: t, warm: false, idleSeconds: 120, containers: 4).seconds == 70 + 9 + 120)

    let url = FileManager.default.temporaryDirectory.appendingPathComponent("timings-\(UUID().uuidString).json")
    defer { try? FileManager.default.removeItem(at: url) }
    try t.save(url)
    #expect(GPUTimings.load(url) == t)
}
