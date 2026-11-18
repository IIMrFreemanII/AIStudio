import AVFoundation
import Foundation
@testable import MovieCore
import Testing

// MARK: dialogue across cuts

/// A voiced line `seconds` long (its take's file need not exist unless a track is written).
private func voiced(_ text: String, _ seconds: Double, by who: String = "mara", file: String? = nil,
                    mode: LineMode = .onScreen, lead: Double = 0, carry: Bool = false) -> DialogueLine {
    let t = Take(file: file ?? "audio/\(UUID().uuidString).wav", seconds: seconds)
    return DialogueLine(character: who, text: text, takes: [t], selectedTake: t.id, mode: mode, lead: lead, carry: carry)
}

private func sine(_ url: URL, seconds: Double) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let file = try AVAudioFile(forWriting: url, settings: [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: 16_000.0, AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
    ], commonFormat: .pcmFormatFloat32, interleaved: false)
    let n = Int(seconds * 16_000)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(n))!
    buffer.frameLength = AVAudioFrameCount(n)
    for i in 0..<n { buffer.floatChannelData![0][i] = 0.5 * Float(sin(2 * .pi * 440 * Double(i) / 16_000)) }
    try file.write(from: buffer)
}

/// Two shots of one scene and one of the next, each with a clip of its planned length.
private func conversation(_ a: [DialogueLine], _ b: [DialogueLine], _ c: [DialogueLine] = [],
                          durations: [Double] = [4, 4, 4]) -> MovieProject {
    var m = MovieProject()
    m.scenes = [MovieScene(id: "s1", heading: "A"), MovieScene(id: "s2", heading: "B")]
    m.shots = [Shot(scene: "s1", duration: durations[0], dialogue: a, talking: true),
               Shot(scene: "s1", duration: durations[1], dialogue: b, talking: true),
               Shot(scene: "s2", duration: durations[2], dialogue: c, talking: true)]
    for i in m.shots.indices {
        let clip = Take(file: "shots/\(i).mp4")
        m.shots[i].videos = [clip]
        m.shots[i].selectedVideo = clip.id
    }
    return m
}

@Test func aJCutStartsTheLineBeforeItsShot() {
    let shot = Shot(scene: "s", dialogue: [voiced("Wait.", 1, lead: 1.5), voiced("Listen.", 1, lead: 2)])
    let p = MovieVoices.layout(shot)
    // Only the first line leads; the next follows it.
    #expect(p.map(\.start) == [-1.5, -1.5 + 1 + MovieVoices.gap])
    #expect(MovieVoices.layout(Shot(scene: "s", dialogue: [voiced("x", 1, lead: 9)])).first?.start == -DialogueLine.maxLead)

    let m = conversation([voiced("Hi.", 1)], [voiced("Wait.", 1, lead: 1)])
    let t = MovieTimeline(movie: m, durations: [:])
    let line = t.sounds.first { $0.label == "mara" && $0.file == m.shots[1].dialogue[0].takes[0].file }!
    #expect(abs(line.start - (t.clips[1].start - 1)) < 1e-6 && abs(line.duration - 1) < 1e-6)
    #expect(t.subtitles.contains { abs($0.start - (t.clips[1].start - 1)) < 1e-6 })
}

@Test func anLCutRunsTheLineOnOverTheNextShotWhichWaitsForIt() {
    // A 3 s line carried out of a 2 s shot: 1.3 s of it plays over shot 2, whose own line waits.
    var m = conversation([voiced("You knew all along.", 3, carry: true)], [voiced("I did.", 1)], durations: [2, 4, 4])
    #expect(MovieVoices.incoming(m) == [m.shots[1].id: 0.3 + 3 - 2])
    let next = MovieVoices.layout(m.shots[1], incoming: 1.3)
    #expect(abs(next[0].start - (1.3 + MovieVoices.gap)) < 1e-9)
    let t = MovieTimeline(movie: m, durations: [:])
    let carried = t.sounds.first { $0.file == m.shots[0].dialogue[0].takes[0].file }!
    let answer = t.sounds.first { $0.file == m.shots[1].dialogue[0].takes[0].file }!
    #expect(abs(carried.end - (0.3 + 3)) < 1e-6 && carried.end > t.clips[0].end, "it isn't cut at the cut")
    #expect(abs(answer.start - (t.clips[1].start + 1.3 + MovieVoices.gap)) < 1e-6)

    // A shot only needs to hold the start of its carried line: the 2 s shot isn't stretched.
    let before = m.shots.map(\.duration)
    #expect(MovieVoices.fitDurations(&m).isEmpty && m.shots.map(\.duration) == before)
    #expect(MovieVoices.neededDuration(m.shots[0], canCarry: true) == 0.3 + MovieVoices.carryInShot)

    // No next shot in the scene (the last of a scene): it can't carry, so its shot stretches to the whole line.
    var last = conversation([], [voiced("Goodbye.", 3, carry: true)], durations: [4, 2, 4])
    #expect(!MovieVoices.canCarry(1, in: last) && MovieVoices.incoming(last).isEmpty)
    MovieVoices.fitDurations(&last)
    #expect(abs(last.shots[1].duration - 3.7) < 1e-9)
    let tl = MovieTimeline(movie: last, durations: [:])
    #expect(tl.sounds.allSatisfy { $0.end <= tl.clips[1].end + 1e-6 || $0.start >= tl.clips[2].start })
}

@Test func aLongCarryMakesTheNextShotLongerWhenItsOwnLinesWontFit() {
    // 6 s carried into shot 2 leaves too little room for its line: shot 2 stretches.
    var m = conversation([voiced("A long speech that goes on.", 6.3, carry: true)], [voiced("And?", 1)],
                         durations: [2, 4, 4])
    MovieVoices.fitDurations(&m)
    #expect(m.shots[0].duration == 2)
    #expect(abs(m.shots[1].duration - ((0.3 + 6.3 - 2) + MovieVoices.gap + 1 + MovieVoices.tail)) < 0.11)
}

@Test func offScreenLinesAndNarrationAreHeardButNotLipSynced() throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent("dialogue-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: folder) }
    try sine(folder.appendingPathComponent("audio/a.wav"), seconds: 1)
    try sine(folder.appendingPathComponent("audio/b.wav"), seconds: 1)
    let shot = Shot(scene: "s", dialogue: [voiced("Who's there?", 1, by: "tomas", file: "audio/a.wav", mode: .offScreen),
                                           voiced("Only me.", 1, by: "mara", file: "audio/b.wav")], talking: true)
    var m = MovieProject()
    m.shots = [shot]
    #expect(MovieVoices.speakers(shot) == ["mara"] && MovieShots.usesLipSync(shot, in: m))
    // The lips follow only Mara's line, at its place after Tomas's (0.3 + 1 + gap).
    let out = folder.appendingPathComponent("audio/t.wav")
    try MovieVoices.voiceTrack(shot, folder: folder, duration: 3, to: out)
    let file = try AVAudioFile(forReading: out)
    let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 48_000)!
    try file.read(into: buf)
    let d = buf.floatChannelData![0]
    func rms(_ a: Int, _ b: Int) -> Double { sqrt((a..<b).reduce(0.0) { $0 + Double(d[$1] * d[$1]) } / Double(b - a)) }
    #expect(rms(6_000, 19_000) < 0.001, "Tomas off screen isn't on the track")
    #expect(rms(26_000, 38_000) > 0.3, "Mara is, at 1.55 s")

    // Only narration: no lip-sync at all, and its subtitle says so.
    var vo = conversation([voiced("That night, she was alone.", 2, by: "narrator", mode: .voiceOver)], [])
    #expect(!MovieShots.usesLipSync(vo.shots[0], in: vo))
    vo.characters = [MovieCharacter(id: "narrator", name: "Narrator", look: "", voice: "", isNarrator: true)]
    let t = MovieTimeline(movie: vo, durations: [:])
    #expect(t.subtitles.first?.shown == "(V.O.) That night, she was alone.")
    #expect(MovieSubtitles.srt(t.subtitles).contains("NARRATOR (V.O.): That\nnight, she was alone."), "two even rows")
    #expect(MovieSubtitles.srt(t.subtitles, speakers: false).contains("(V.O.) That night"))
}

@Test func h3SpeaksOnScreenLinesInTheClipAndTheRestOverIt() {
    var m = conversation([voiced("Who's there?", 1, by: "tomas", mode: .offScreen), voiced("Only me.", 1)], [])
    m.videoEngine = .minimaxH3
    let shot = m.shots[0]
    #expect(MovieH3.speakers(shot) == ["mara"] && MovieH3.lineTimes(shot).count == 1)
    let t = MovieTimeline(movie: m, durations: [:])
    // The clip's own sound, plus the off-screen line's spoken take over it.
    #expect(t.sounds.filter { $0.kind == .clip && $0.id == "clip-\(shot.id.uuidString)" }.count == 1)
    let over = t.sounds.filter { $0.kind == .dialogue }
    #expect(over.count == 1 && over[0].file == shot.dialogue[0].takes[0].file)
    #expect(t.subtitles.contains { $0.mode == .offScreen } && t.subtitles.contains { $0.mode == .onScreen })
}

@Test func linesComeFromThePlannerWithTheirModesAndCuts() throws {
    let cast = [MovieCharacter(id: "mara", name: "Mara", look: "", voice: ""),
                MovieCharacter(id: "tomas", name: "Tomas", look: "", voice: ""),
                MovieCharacter(id: "narrator", name: "Narrator", look: "", voice: "", isNarrator: true)]
    let reply = """
    {"shots": [{"duration": 5, "framing": "close-up", "keyframe": "Mara", "talking": true, "dialogue": [
      {"character": "mara", "text": "Grandpa?", "mode": "on", "lead": "1.5"},
      {"character": "tomas", "text": "Up here!", "mode": "O.S."},
      {"character": "narrator", "text": "She climbed.", "mode": "on"},
      {"character": "mara", "text": "Coming.", "mode": "voice-over", "lead": 9, "carry": "true"}]}]}
    """
    let lines = try MoviePlanner.parseShots(reply, scene: "s1", cast: cast)[0].dialogue
    #expect(lines.map(\.mode) == [.onScreen, .offScreen, .voiceOver, .voiceOver], "a narrator is always heard over")
    #expect(lines[0].lead == 1.5 && lines[3].lead == DialogueLine.maxLead && lines[3].carry)
    #expect(MoviePlanner.mode("Off-screen") == .offScreen && MoviePlanner.mode("narration") == .voiceOver
            && MoviePlanner.mode(nil) == .onScreen)
    let prompt = MoviePlanner.shotPrompt(for: MovieProject(), scene: MovieScene(id: "s1", heading: "H",
        dialogue: [DialogueLine(character: "tomas", text: "Up here!", mode: .offScreen)])).user
    #expect(prompt.contains("tomas [off]: Up here!"))

    // Old lines decode on screen, with no cut.
    let old = try JSONDecoder().decode(DialogueLine.self, from: Data("""
    {"id": "\(UUID().uuidString)", "character": "a", "text": "hi"}
    """.utf8))
    #expect(old.mode == .onScreen && old.lead == 0 && !old.carry)
}

@Test func longCatPutsTheLeftSpeakerFirstAndBreezeSuitsEnglishAndChinese() {
    let scene = MovieScene(id: "s", heading: "H", blocking: ["mara": .right, "tomas": .left])
    #expect(MovieShots.leftToRight(["mara", "tomas"], scene: scene) == ["tomas", "mara"])
    #expect(MovieShots.leftToRight(["mara", "tomas"], scene: MovieScene(id: "s", heading: "H")) == ["mara", "tomas"])
    #expect(MovieVoices.suitsBreeze("A girl keeps the lighthouse lit through a storm, and her grandfather is hurt."))
    #expect(MovieVoices.suitsBreeze("灯塔的灯在暴风雨中熄灭了，玛拉必须把油送上悬崖。"))
    #expect(!MovieVoices.suitsBreeze("Une fille garde le phare allumé pendant la tempête, et son grand-père est blessé."))
    #expect(!MovieVoices.suitsBreeze("Ein Mädchen hält den Leuchtturm im Sturm am Leuchten, ihr Großvater ist verletzt."))
}
