import Foundation
import GraphRuntime
@testable import MovieCore
import Testing

// MARK: the movie's Fast switch for sheets, plates and keyframes

@Test func fastModeSetsEveryQwenNodeAndOnlyThose() throws {
    for template in [MovieArt.textToImageTemplate(name: MovieArt.TemplateName.sheet), MovieArt.editTemplate()] {
        // The built-in templates run in full.
        let qwen = template.graph.nodes.filter { MovieArt.qwenNodeTypes.contains($0.type) }
        #expect(qwen.count == 1 && qwen[0].inputs["fast"] == .value(.bool(false)))

        let fast = MovieArt.fastMode(true, template)
        for node in fast.graph.nodes {
            if MovieArt.qwenNodeTypes.contains(node.type) {
                #expect(node.inputs["fast"] == .value(.bool(true)))
            } else {
                #expect(node.inputs["fast"] == nil, "\(node.type) has no fast mode")
            }
        }
        #expect(fast.slots == template.slots && fast.name == template.name)
        // Off again: back to the full model.
        #expect(MovieArt.fastMode(false, fast) == template)
    }
}

@Test func theSwitchDecidesForAnEditedCopyToo() throws {
    // A copy saved before the full mode (fast on), or edited by hand: the movie's switch wins.
    var saved = MovieArt.editTemplate()
    let i = try #require(saved.graph.nodes.firstIndex { $0.id == "edit" })
    saved.graph.nodes[i].inputs["fast"] = .value(.bool(true))
    saved.graph.nodes[i].inputs["steps"] = .value(.int(12))
    let full = MovieArt.fastMode(false, saved)
    #expect(full.graph["edit"]?.inputs["fast"] == .value(.bool(false)))
    #expect(full.graph["edit"]?.inputs["steps"] == .value(.int(12)), "the user's other settings stay")
    // A copy without a `fast` input gets one.
    var bare = MovieArt.textToImageTemplate(name: MovieArt.TemplateName.plate)
    bare.graph.nodes[0].inputs["fast"] = nil
    #expect(MovieArt.fastMode(true, bare).graph["image"]?.inputs["fast"] == .value(.bool(true)))
}

@Test func fastPicturesIsOffForOldMoviesAndSaved() throws {
    let old = """
    {"version":1,"id":"\(UUID().uuidString)","title":"Old","idea":"x","style":"cinematic","aspect":"wide",
     "targetSeconds":60,"logline":"","characters":[],"scenes":[],"shots":[],"approved":[],"costSpent":0,
     "created":"2026-11-01T00:00:00Z","modified":"2026-11-01T00:00:00Z"}
    """
    let d = JSONDecoder()
    d.dateDecodingStrategy = .iso8601
    #expect(try d.decode(MovieProject.self, from: Data(old.utf8)).fastPictures == false)
    #expect(MovieProject().fastPictures == false)
    var m = MovieProject(title: "Quick")
    m.fastPictures = true
    let e = JSONEncoder()
    e.dateEncodingStrategy = .iso8601
    #expect(try d.decode(MovieProject.self, from: e.encode(m)).fastPictures == true)
}

@Test func fastAndFullJobsAreTimedApart() {
    #expect(GPUTimings.timingTask("qwen-image", fast: true) == "qwen-image:fast")
    #expect(GPUTimings.timingTask("qwen-image-edit", fast: true) == "qwen-image-edit:fast")
    #expect(GPUTimings.timingTask("qwen-image", fast: false) == "qwen-image")
    // Other tasks keep one timing whatever their mode (Wan's defaults are its fast mode's).
    #expect(GPUTimings.timingTask("wan22-i2v", fast: true) == "wan22-i2v")

    var t = GPUTimings()
    // Defaults: the fast modes as measured, far below the full modes' guesses.
    #expect(t.seconds(task: "qwen-image:fast", megapixels: 1.5) == 5 && t.seconds(task: "qwen-image", megapixels: 1.5) == 30)
    #expect(t.seconds(task: "qwen-image-edit:fast", megapixels: 1) == 8)
    #expect(t.loadSeconds(task: "qwen-image-edit:fast") == 60)

    let now = Date()
    t.record(task: "qwen-image-edit:fast", megapixels: 1, gpuSeconds: 7, loadSeconds: 50, at: now)
    t.record(task: "qwen-image-edit", megapixels: 1, gpuSeconds: 70, loadSeconds: 0, at: now)
    #expect(t.seconds(task: "qwen-image-edit:fast", megapixels: 1) == 7)
    #expect(t.seconds(task: "qwen-image-edit", megapixels: 1) == 70, "a fast job doesn't pull the full mode's time down")
    // One worker loads the model for both modes: its load and warmth are the task's.
    #expect(t.loads.keys.sorted() == ["qwen-image-edit"] && t.loadSeconds(task: "qwen-image-edit") == 50)
    #expect(t.loadSeconds(task: "qwen-image-edit:fast") == 50)
    #expect(t.lastJob.keys.sorted() == ["qwen-image-edit"])
    #expect(t.isWarm(task: "qwen-image-edit", idleSeconds: 120, now: now.addingTimeInterval(30)))

    let fast = MovieCost.images(label: "Keyframes", task: "qwen-image-edit:fast", jobs: 4, megapixels: 1, gpu: "H100",
                                usdPerSecond: 0.001, timings: t, warm: true, idleSeconds: 0)
    #expect(abs(fast.seconds - 28) < 1e-9)
}

@Test func theFilmEstimateFollowsTheFastSwitch() {
    let timings = GPUTimings()
    var m = MovieProject(targetSeconds: 60)
    m.characters = [MovieCharacter(id: "mara", name: "Mara", look: "red raincoat", voice: "")]
    let full = FilmEstimate(movie: m, timings: timings)
    m.fastPictures = true
    let fast = FilmEstimate(movie: m, timings: timings)
    func usd(_ e: FilmEstimate, _ label: String) -> Double { e.parts.first { $0.label == label }?.usd ?? 0 }
    for label in ["Character sheets", "Keyframes"] {
        #expect(usd(fast, label) > 0 && usd(fast, label) < usd(full, label), "\(label)")
    }
    #expect(usd(fast, "Clips (Wan 2.2)") == usd(full, "Clips (Wan 2.2)"))
}
