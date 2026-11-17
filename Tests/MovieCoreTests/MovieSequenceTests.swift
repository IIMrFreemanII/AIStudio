import Foundation
@testable import MovieCore
import Testing

// MARK: sequences

/// A film of `n` sequences with two scenes each and two shots per scene.
private func film(sequences n: Int) -> MovieProject {
    var m = MovieProject(targetSeconds: 600)
    m.sequences = (1...n).map { MovieSequence(id: "q\($0)", title: "Part \($0)") }
    m.scenes = (1...n).flatMap { q in
        [MovieScene(id: "s\(q)a", heading: "A\(q)", sequence: "q\(q)"), MovieScene(id: "s\(q)b", heading: "B\(q)", sequence: "q\(q)")]
    }
    m.shots = m.scenes.flatMap { s in [Shot(scene: s.id, duration: 4), Shot(scene: s.id, duration: 5)] }
    return m
}

@Test func oldMoviesDecodeAsOneSequenceWithTheirApprovals() throws {
    // A movie saved before sequences: no sequences, no sequence on its scenes, per-sequence stages in `approved`.
    let old = """
    {"version":1,"id":"\(UUID().uuidString)","title":"Old","idea":"x","style":"cinematic","aspect":"wide",
     "targetSeconds":60,"logline":"","characters":[],
     "scenes":[{"id":"s1","heading":"H","summary":"","mood":"","action":"","dialogue":[]}],
     "shots":[],"approved":["idea","script","shots","cast","keyframes"],"costSpent":0,
     "created":"2026-11-01T00:00:00Z","modified":"2026-11-01T00:00:00Z"}
    """
    let d = JSONDecoder()
    d.dateDecodingStrategy = .iso8601
    let m = try d.decode(MovieProject.self, from: Data(old.utf8))
    #expect(m.sequences == [.wholeFilm] && m.scenes[0].sequence == "q1")
    #expect(m.approved == [.idea, .story, .script, .cast], "a movie with a script gets its story approved")
    #expect(m.sequenceApproved == ["q1": [.shots, .keyframes]])
    #expect(m.isApproved(.keyframes) && m.isApproved(.keyframes, in: "q1"))
    #expect(m.currentStage == .voices && m.isUnlocked(.voices) && !m.isUnlocked(.video))

    // And it round-trips with its new fields.
    let e = JSONEncoder()
    e.dateEncodingStrategy = .iso8601
    #expect(try d.decode(MovieProject.self, from: e.encode(m)) == m)
}

@Test func perSequenceStagesGateEachSequenceOnItsOwn() {
    var m = film(sequences: 3)
    m.approve(.idea)
    m.approve(.story)
    m.approve(.script)
    #expect(m.isUnlocked(.shots, in: "q2") && m.isUnlocked(.cast) && !m.isUnlocked(.keyframes, in: "q1"))

    // Sequence 1 goes ahead: its shot list and the film's cast.
    m.approve(.shots, in: "q1")
    m.approve(.cast)
    #expect(m.isUnlocked(.keyframes, in: "q1") && !m.isUnlocked(.keyframes, in: "q2"))
    #expect(!m.isApproved(.shots) && m.isApprovedAnywhere(.shots))
    #expect(m.currentStage(in: "q1") == .keyframes && m.currentStage(in: "q2") == .shots)

    for s in [MovieStage.keyframes, .voices, .video] { m.approve(s, in: "q1") }
    // The edit opens once one sequence has clips, and is approvable when every sequence's sound is.
    #expect(m.isUnlocked(.edit) && !m.editIsApprovable)
    #expect(m.isUnlocked(.sound, in: "q1") && !m.isUnlocked(.sound, in: "q3"))
    for q in ["q1", "q2", "q3"] {
        for s in [MovieStage.shots, .keyframes, .voices, .video, .sound] { m.approve(s, in: q) }
    }
    #expect(m.editIsApprovable && m.isApproved(.sound))
    m.approve(.edit)
    #expect(m.currentStage == .edit && m.isApproved(.edit))
}

@Test func reopeningASequenceLeavesTheOthersAndTheFilmsStages() {
    var m = film(sequences: 2)
    m.approve(.idea)
    m.approve(.story)
    m.approve(.script)
    m.approve(.cast)
    for q in ["q1", "q2"] {
        for s in [MovieStage.shots, .keyframes, .voices, .video, .sound] { m.approve(s, in: q) }
    }
    m.approve(.edit)

    // A keyframe changed in sequence 2: its keyframes and later stages, and the film's edit.
    m.reopen(.keyframes, in: "q2")
    #expect(m.approved == [.idea, .story, .script, .cast], "a movie with a script gets its story approved")
    #expect(m.sequenceApproved["q1"] == [.shots, .keyframes, .voices, .video, .sound])
    #expect(m.sequenceApproved["q2"] == [.shots])

    // A new sheet (the film's cast): every sequence's keyframes and on.
    m.reopen(.cast)
    #expect(m.approved == [.idea, .story, .script])
    #expect(m.sequenceApproved == ["q1": [.shots], "q2": [.shots]])

    // Nil: every sequence.
    m.reopen(.shots)
    #expect(m.sequenceApproved.isEmpty && m.approved == [.idea, .story, .script])

    m.approve(.voices, in: "q1")
    m.unapprove(.voices, in: "q1")
    #expect(m.sequenceApproved["q1"]?.isEmpty ?? true)
}

@Test func sequencesScopeShotsAndScenes() {
    let m = film(sequences: 3)
    #expect(m.shots(inSequence: "q2").count == 4 && m.shots(inSequence: nil).count == 12)
    #expect(m.scenes(inSequence: "q3").map(\.id) == ["s3a", "s3b"])
    #expect(m.sequence(of: m.shots[5]) == "q2")
    #expect(m.shotIndex[m.shots[7].id] == 7)
    var lost = m
    lost.scenes[0].sequence = "gone"  // a sequence that no longer exists counts as the first
    #expect(lost.sequence(ofScene: "s1a") == "q1" && lost.shots(inSequence: "q1").count == 4)
}

@Test func aLongScriptIsGroupedIntoSequencesOfAboutTwoAndAHalfMinutes() {
    var m = MovieProject(targetSeconds: 900)
    m.scenes = (1...20).map { MovieScene(id: "s\($0)", heading: "SCENE \($0)", action: "Something happens here.") }
    MoviePlanner.groupSequences(&m)
    #expect(m.sequences.count == 6)
    #expect(m.sequences.map(\.id) == (1...6).map { "q\($0)" })
    #expect(m.scenes.first?.sequence == "q1" && m.scenes.last?.sequence == "q6")
    // In order, every sequence with a scene, and acts 1, 2, 3.
    let order = m.scenes.map { Int($0.sequence.dropFirst())! }
    #expect(order == order.sorted() && Set(order).count == 6)
    #expect(m.sequences.map(\.act) == [1, 1, 2, 2, 2, 3])

    var short = MovieProject(targetSeconds: 120)
    short.scenes = (1...3).map { MovieScene(id: "s\($0)", heading: "S\($0)") }
    MoviePlanner.groupSequences(&short)
    #expect(short.sequences == [.wholeFilm] && short.scenes.allSatisfy { $0.sequence == "q1" })
}

// MARK: budget

@Test func theFilmEstimateCountsWhatIsLeftAndTheBudgetWarns() {
    let timings = GPUTimings()
    var m = film(sequences: 2)
    m.characters = [MovieCharacter(id: "a", name: "A", look: "", voice: "")]
    m.shots[0].dialogue = [DialogueLine(character: "a", text: "one two three four five")]
    m.shots[0].talking = true
    let all = FilmEstimate(movie: m, timings: timings)
    #expect(all.usd > 0)
    #expect(Set(all.parts.map(\.label)).isSuperset(of: ["Character sheets", "Keyframes", "Clips (Wan 2.2)", "Music"]))

    // Keyframes made: that part goes.
    var made = m
    for i in made.shots.indices {
        made.shots[i].keyframes = [Take(file: "k\(i).png")]
        made.shots[i].selectedKeyframe = made.shots[i].keyframes[0].id
    }
    let less = FilmEstimate(movie: made, timings: timings)
    #expect(!less.parts.contains { $0.label == "Keyframes" } && less.usd < all.usd)

    // Before the shot list: from the target length, and longer films cost more.
    let short = FilmEstimate(movie: MovieProject(targetSeconds: 120), timings: timings)
    let long = FilmEstimate(movie: MovieProject(targetSeconds: 1200), timings: timings)
    #expect(long.usd > short.usd * 5)

    // H3 makes the sound in the clip: no lines or effects.
    var h3 = m
    h3.videoEngine = .minimaxH3
    let h3e = FilmEstimate(movie: h3, timings: timings)
    #expect(h3e.parts.contains { $0.label == "Clips (MiniMax H3)" } && !h3e.parts.contains { $0.label == "Lines" })

    m.costSpent = 4
    m.budget = 5
    #expect(!m.wouldExceedBudget(0.9) && m.wouldExceedBudget(1.5))
    #expect(m.budgetWarning(for: 1.5)?.contains("$5.00 budget") == true && m.budgetWarning(for: 0.5) == nil)
    m.budget = nil
    #expect(!m.wouldExceedBudget(100))
}

// MARK: scale

@Test func aTwentyMinuteTimelineBuildsQuickly() {
    var m = MovieProject(targetSeconds: 1200)
    m.scenes = (1...40).map { MovieScene(id: "s\($0)", heading: "S\($0)") }
    var durations: [String: Double] = [:]
    for i in 0..<260 {
        var shot = Shot(scene: "s\(i / 7 + 1)", duration: 4.6, transition: i % 5 == 0 ? .crossfade : .cut)
        let clip = Take(file: "shots/\(i).mp4")
        shot.videos = [clip]
        shot.selectedVideo = clip.id
        let line = Take(file: "voices/\(i).wav", seconds: 2)
        shot.dialogue = [DialogueLine(character: "a", text: "a line", takes: [line], selectedTake: line.id)]
        m.shots.append(shot)
        durations[clip.file] = 4.6
    }
    m.music = m.scenes.map { s in
        let t = Take(file: "music/\(s.id).wav", seconds: 60)
        return MusicCue(id: s.id, prompt: "", takes: [t], selected: t.id)
    }
    let start = Date()
    let t = MovieTimeline(movie: m, durations: durations)
    let segments = t.segments
    let volumes = t.sounds.filter { $0.kind == .music }.map { t.volume(for: $0) }
    let elapsed = Date().timeIntervalSince(start)
    #expect(t.clips.count == 260 && t.duration > 1100)
    #expect(t.subtitles.count == 260 && volumes.count == 38)
    #expect(elapsed < 2, "built in \(elapsed) s")
    // The sweep finds the same layers as checking every clip.
    for seg in segments.prefix(400) {
        let mid = (seg.start + seg.end) / 2
        let brute = t.clips.indices.filter { t.clips[$0].start <= mid && mid < t.clips[$0].end }
        #expect(Set(seg.layers.map(\.clip)) == Set(brute))
    }
    // Ducking: the music dips under every line it plays under.
    for (sound, points) in zip(t.sounds.filter { $0.kind == .music }, volumes).prefix(5) {
        let inLine = t.subtitles.first { $0.start > sound.start + 1 && $0.end < sound.end - 2 }!
        let p = points.first { $0.time >= inLine.start && $0.time <= inLine.end }!
        #expect(abs(p.gain - sound.gain * t.duck) < 1e-6)
    }
}
