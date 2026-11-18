import AVFoundation
import Foundation
@testable import MovieCore
import Testing

/// Three shots in two sequences: s1 (shots 0, 1) in "The Lamp", s2 (shot 2) in act 2's "The Storm".
private func twoSequences() -> MovieProject {
    var m = editMovie([.cut, .cut, .cut])
    m.title = "The Last Light"
    m.sequences = [MovieSequence(id: "q1", title: "The Lamp"), MovieSequence(id: "q2", title: "The Storm", act: 2)]
    m.scenes = [MovieScene(id: "s1", heading: "INT. LAMP ROOM", sequence: "q1"),
                MovieScene(id: "s2", heading: "EXT. CLIFF", sequence: "q2")]
    m.shots[2].scene = "s2"
    return m
}

private let fourSeconds = Dictionary(uniqueKeysWithValues: (0..<3).map { ("shots/\($0).mp4", 4.0) })

@Test func aFilmInSequencesHasAChapterForEach() {
    var m = twoSequences()
    let t = MovieTimeline(movie: m, durations: fourSeconds)
    #expect(t.chapters == [MovieTimeline.Chapter(start: 0, title: "Act 1 · The Lamp"),
                           MovieTimeline.Chapter(start: 8, title: "Act 2 · The Storm")])
    // The first chapter starts the film even when its first shot has no clip yet.
    m.shots[0].selectedVideo = nil
    #expect(MovieTimeline(movie: m, durations: fourSeconds).chapters.map(\.start) == [0, 4])
    // An untitled sequence is named by its number; a sequence with no clips has no chapter.
    m.sequences[1].title = " "
    #expect(MovieTimeline(movie: m, durations: fourSeconds).chapters.last?.title == "Act 2 · Sequence 2")
    m.shots[2].selectedVideo = nil
    #expect(MovieTimeline(movie: m, durations: fourSeconds).chapters.isEmpty, "one chapter is no chapters")
    // A film of one sequence has none.
    #expect(MovieTimeline(movie: editMovie([.cut, .cut]), durations: fourSeconds).chapters.isEmpty)
}

@Test func oneSequenceIsExportedOnItsOwn() {
    let m = twoSequences()
    let storm = m.excerpt(sequence: "q2")
    #expect(storm.shots.map(\.id) == [m.shots[2].id] && storm.scenes == m.scenes)
    let t = MovieTimeline(movie: storm, durations: fourSeconds)
    #expect(t.duration == 4 && t.clips.first?.start == 0 && t.chapters.isEmpty)
    #expect(m.excerpt(sequence: "q1").shots.count == 2)
}

@Test func theSrtNamesWhoSpeaks() {
    let subs = [MovieTimeline.Subtitle(start: 0, end: 1.5, speaker: "Mara", text: "The lamp is out."),
                MovieTimeline.Subtitle(start: 2, end: 3, speaker: "Narrator", text: "It never went out.", mode: .voiceOver),
                MovieTimeline.Subtitle(start: 3, end: 4, speaker: "Tomas", text: "Mara?", mode: .offScreen),
                MovieTimeline.Subtitle(start: 4, end: 5, speaker: "", text: "Hm.")]
    let srt = MovieSubtitles.srt(subs)
    #expect(srt.hasPrefix("1\n00:00:00,000 --> 00:00:01,500\nMARA: The lamp is out.\n\n"))
    #expect(srt.contains("\nNARRATOR (V.O.): It never went out.\n") && srt.contains("\nTOMAS (O.S.): Mara?\n"))
    #expect(srt.contains("\nHm.\n"), "no speaker, no name")
    let plain = MovieSubtitles.srt(subs, speakers: false)
    #expect(plain.contains("\nThe lamp is out.\n") && plain.contains("\n(V.O.) It never went out.\n") && !plain.contains("MARA"))
    // Mixes saved before the setting name the speakers.
    let old = try? JSONDecoder().decode(MovieMix.self, from: Data(#"{"dialogue": 0.9, "effects": 0.5, "music": 0.3, "duck": 0.3, "burnSubtitles": true}"#.utf8))
    #expect(old?.speakerNames == true && old?.burnSubtitles == true && old?.dialogue == 0.9)
}

@Test func longLinesAreCutIntoCardsOfTwoRows() {
    let short = MovieSubtitles.cards("The lamp is out.", start: 1, end: 2)
    #expect(short == [MovieSubtitles.Card(start: 1, end: 2, rows: ["The lamp is out."])])
    // Two even rows rather than a full one and a scrap.
    let two = MovieSubtitles.cards("MARA: I kept it burning every night for eleven years, and you never came.", start: 0, end: 4)
    #expect(two.count == 1 && two[0].rows.count == 2 && two[0].rows.allSatisfy { $0.count <= 42 })
    #expect(abs(two[0].rows[0].count - two[0].rows[1].count) < 12, "\(two[0].rows)")

    let speech = """
        I kept it burning every night for eleven years, through every storm the sea could throw at this rock, \
        and you never once came back to see it, not even when Mother died, not even when I wrote to you.
        """
    let cards = MovieSubtitles.cards(speech, start: 10, end: 22)
    #expect(cards.count >= 3 && cards.allSatisfy { (1...2).contains($0.rows.count) && $0.rows.allSatisfy { $0.count <= 42 } })
    #expect(cards.flatMap(\.rows).joined(separator: " ") == speech, "every word, in order")
    #expect(cards.first?.start == 10 && cards.last?.end == 22 && zip(cards, cards.dropFirst()).allSatisfy { $0.end == $1.start })
    #expect(cards.last!.text.count >= 21, "no scrap of a word or two at the end: \(cards.last!.rows)")
    // The .srt numbers each card.
    let srt = MovieSubtitles.srt([MovieTimeline.Subtitle(start: 10, end: 22, speaker: "Mara", text: speech)])
    #expect(srt.contains("\n\n\(cards.count)\n") || srt.contains("\n\n\(cards.count + 1)\n"))
    #expect(srt.hasPrefix("1\n00:00:10,000 --> ") && srt.contains("MARA: I kept"))
}

@Test func theExportedFilmHasItsChaptersAndASequenceExportsAlone() async throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("movie-export-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: folder.appendingPathComponent("shots"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for (i, c) in [(230, 20, 20), (20, 230, 20), (20, 20, 230)].enumerated() {
        try await writeClip(folder.appendingPathComponent("shots/\(i).mp4"), frames: 32, rgb: (UInt8(c.0), UInt8(c.1), UInt8(c.2)))  // 2 s
    }
    let m = twoSequences()
    let durations = await MovieComposer.durations(of: (0..<3).map { "shots/\($0).mp4" }, in: folder)
    let film = MovieTimeline(movie: m, durations: durations)
    let out = folder.appendingPathComponent("renders/film.mp4")
    try await MovieComposer.export(try await MovieComposer.build(film, folder: folder, size: (128, 96)), to: out)
    let chapters = try await MovieChapters.read(out)
    #expect(chapters.map(\.title) == ["Act 1 · The Lamp", "Act 2 · The Storm"])
    #expect(chapters.count == 2 && abs(chapters[0].start) < 0.01 && abs(chapters[1].start - 4) < 0.01, "\(chapters)")
    // Still the film: one video track (the chapters' text track isn't shown), as long as the timeline.
    let asset = AVURLAsset(url: out)
    let video = try await asset.loadTracks(withMediaType: .video)
    let range = try #require(try await video.first?.load(.timeRange))
    #expect(video.count == 1 && abs(range.duration.seconds - 6) < 0.1)
    let text = try await asset.loadTracks(withMediaType: .text).first
    let enabled = try await text?.load(.isEnabled)
    #expect(enabled == false)

    let storm = MovieTimeline(movie: m.excerpt(sequence: "q2"), durations: durations)
    let one = folder.appendingPathComponent("renders/storm.mp4")
    try await MovieComposer.export(try await MovieComposer.build(storm, folder: folder, size: (128, 96)), to: one)
    let d = try await AVURLAsset(url: one).load(.duration).seconds
    let none = try await MovieChapters.read(one)
    #expect(abs(d - 2) < 0.1 && none.isEmpty)
}
