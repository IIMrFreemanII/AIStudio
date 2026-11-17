import AVFoundation
import Foundation
import GraphRuntime
@testable import MovieCore
import Testing

// MARK: helpers

private func take(_ file: String, seconds: Double? = nil, gainDB: Double? = nil) -> Take {
    Take(file: file, seconds: seconds, gainDB: gainDB)
}

private func line(_ text: String, _ seconds: Double?, character: String = "ann", file: String? = nil) -> DialogueLine {
    var l = DialogueLine(character: character, text: text)
    if let seconds {
        let t = take(file ?? "audio/\(UUID().uuidString).wav", seconds: seconds)
        l.takes = [t]
        l.selectedTake = t.id
    }
    return l
}

/// Writes a mono 16-bit WAV: `seconds` of a sine, optionally with the second half silent.
private func writeWAV(_ url: URL, seconds: Double, amplitude: Float = 0.1, rate: Double = 44_100, hz: Double = 440,
                      silentHalf: Bool = false) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    let file = try AVAudioFile(forWriting: url, settings: [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: rate, AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
    ], commonFormat: .pcmFormatFloat32, interleaved: false)
    let n = Int(seconds * rate)
    let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(n))!
    buffer.frameLength = AVAudioFrameCount(n)
    for i in 0..<n {
        let silent = silentHalf && i >= n / 2
        buffer.floatChannelData![0][i] = silent ? 0 : amplitude * Float(sin(2 * .pi * hz * Double(i) / rate))
    }
    try file.write(from: buffer)
}

private func tempFolder() throws -> URL {
    let f = FileManager.default.temporaryDirectory.appendingPathComponent("movie-sound-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: f, withIntermediateDirectories: true)
    return f
}

// MARK: model

@Test func oldMovieJSONDecodesWithoutTheSoundKeys() throws {
    var m = MovieProject()
    m.characters = [MovieCharacter(id: "ann", name: "Ann", look: "x", voice: "v", voiceSamples: [take("a.wav")])]
    m.shots = [Shot(scene: "s1", dialogue: [line("Hi", 1)], sounds: [take("fx.wav")])]
    m.music = [MusicCue(id: "s1", prompt: "p")]
    m.lipSync = false
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m)) as! [String: Any]
    for k in ["music", "mix", "lipSync", "lipSyncEngine"] { json[k] = nil }
    var chars = json["characters"] as! [[String: Any]]
    for i in chars.indices { chars[i]["voiceSamples"] = nil; chars[i]["selectedVoiceSample"] = nil }
    json["characters"] = chars
    var shots = json["shots"] as! [[String: Any]]
    for i in shots.indices {
        shots[i]["sounds"] = nil
        shots[i]["selectedSound"] = nil
        shots[i]["dialogue"] = (shots[i]["dialogue"] as! [[String: Any]]).map { d in
            var d = d
            d["takes"] = nil
            d["selectedTake"] = nil
            return d
        }
    }
    json["shots"] = shots
    let back = try JSONDecoder().decode(MovieProject.self, from: JSONSerialization.data(withJSONObject: json))
    #expect(back.shots[0].dialogue[0].takes.isEmpty && back.characters[0].voiceSamples.isEmpty)
    #expect(back.shots[0].sounds.isEmpty && back.music.isEmpty && back.mix == MovieMix() && back.lipSync)
    #expect(back.lipSyncEngine == .wanS2V)  // what movies from before the choice used
}

@Test func lipSyncEngineRoundTripsAndAnUnknownOneFallsBack() throws {
    var m = MovieProject()
    m.lipSyncEngine = .longcat
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONEncoder().encode(m)).lipSyncEngine == .longcat)
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m)) as! [String: Any]
    json["lipSyncEngine"] = "some-future-model"
    let back = try JSONDecoder().decode(MovieProject.self, from: JSONSerialization.data(withJSONObject: json))
    #expect(back.lipSyncEngine == .wanS2V)
    #expect(LipSyncEngine.allCases.map(\.task) == ["wan22-s2v", "longcat-avatar", "infinitetalk"])
}

@Test func voiceEngineRoundTripsAndAnUnknownOneFallsBack() throws {
    #expect(MovieProject().voiceEngine == .qwen3)
    var m = MovieProject()
    m.voiceEngine = .breeze
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONEncoder().encode(m)).voiceEngine == .breeze)
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m)) as! [String: Any]
    json["voiceEngine"] = "some-future-model"
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONSerialization.data(withJSONObject: json)).voiceEngine == .qwen3)
    json.removeValue(forKey: "voiceEngine")  // movies from before the choice
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONSerialization.data(withJSONObject: json)).voiceEngine == .qwen3)
    #expect(VoiceEngine.allCases.map(\.task) == ["qwen3-tts", "breeze-tts"])
    #expect(VoiceEngine.breeze.title.contains("non-commercial"))
}

@Test func breezeDesignsFromTheDescriptionAndDirectsLinesByTheirDelivery() throws {
    let c = MovieCharacter(id: "x", name: "X", look: "", voice: "a gruff old sailor")
    #expect(MovieVoices.TemplateName.sample(.breeze) != MovieVoices.TemplateName.sample(.qwen3))
    #expect(MovieVoices.TemplateName.line(.breeze) != MovieVoices.TemplateName.line(.qwen3))

    let sr = MovieVoices.sampleRender(c, engine: .breeze, seed: 1, server: "s")
    let sg = try MovieVoices.sampleTemplate(engine: .breeze).bind(sr.values)
    #expect(sg["voice"]?.type == "RemoteBreezeSpeech")
    #expect(sg["voice"]?.inputs["instruction"] == .value(.string("a gruff old sailor")))
    #expect(sg["voice"]?.inputs["text"] == .value(.string(MovieVoices.sampleText)))

    var l = DialogueLine(character: "x", text: " Stay. ", delivery: " soft, firm ")
    let lr = MovieVoices.lineRender(l, engine: .breeze, sample: URL(fileURLWithPath: "/m/s.wav"), seed: 2, server: "s")
    let lg = try MovieVoices.lineTemplate(engine: .breeze).bind(lr.values)
    #expect(lg["voice"]?.type == "RemoteBreezeSpeech" && lg["reference"]?.inputs["path"] == .value(.string("/m/s.wav")))
    #expect(lg["voice"]?.inputs["instruction"] == .value(.string("soft, firm")))
    #expect(lg["voice"]?.inputs["text"] == .value(.string("Stay.")))
    // Cloning needs the sample's exact words, which every sample says.
    #expect(lg["voice"]?.inputs["reference_text"] == .value(.string(MovieVoices.sampleText)))

    // Qwen3-TTS is not given the delivery (it can't use it while cloning).
    l.delivery = "shouting"
    let qr = MovieVoices.lineRender(l, sample: URL(fileURLWithPath: "/m/s.wav"), seed: 2, server: "s")
    #expect(qr.values["instruction"] == nil && qr.values["language"] == .string("Auto"))
    #expect(try MovieVoices.lineTemplate().bind(qr.values)["voice"]?.type == "RemoteTextToSpeech")
}

// MARK: layout

@Test func linesAreLaidOutAndShotsStretchToFit() {
    let a = line("one", 2), b = line("two", 1.5), c = line("unvoiced", nil)
    let shot = Shot(scene: "s", duration: 3, dialogue: [a, c, b], talking: true)
    let placed = MovieVoices.layout(shot)
    #expect(placed.map(\.line) == [a.id, b.id])
    #expect(placed[0].start == 0.3 && placed[0].seconds == 2)
    #expect(abs(placed[1].start - 2.55) < 1e-9)
    #expect(abs(MovieVoices.neededDuration(shot)! - 4.45) < 1e-9)
    #expect(MovieVoices.neededDuration(Shot(scene: "s", dialogue: [c])) == nil)

    var m = MovieProject()
    m.shots = [
        shot,                                                                        // 4.45 -> 4.5
        Shot(scene: "s", duration: 6, dialogue: [line("short", 1)]),                  // already long enough
        Shot(scene: "s", duration: 5, dialogue: [line("long", 9)]),                   // 9.7 > 7.5: capped
        Shot(scene: "s", duration: 5, dialogue: [line("long", 9)], talking: true),    // 9.7 <= 12 with lip sync
        Shot(scene: "s", duration: 5, dialogue: [line("huge", 13)], talking: true),   // 13.7 > 12
        Shot(scene: "s", duration: 2),
    ]
    let unfit = MovieVoices.fitDurations(&m)
    #expect(m.shots.map(\.duration) == [4.5, 6, 7.5, 9.7, 12, 2])
    #expect(unfit == [m.shots[2].id, m.shots[4].id])
    m.lipSync = false
    m.shots[3].duration = 5
    #expect(MovieVoices.fitDurations(&m).contains(m.shots[3].id) && m.shots[3].duration == 7.5)
}

// MARK: timeline

private func soundMovie() -> MovieProject {
    var m = MovieProject()
    m.characters = [MovieCharacter(id: "ann", name: "Ann", look: "", voice: "")]
    m.scenes = [MovieScene(id: "s1", heading: "INT. ROOM")]
    let l1 = line("Hello there.", 2, file: "audio/l1.wav")
    let l2 = line("  Second line.  ", 3, file: "audio/l2.wav")
    let fx = take("audio/fx.wav", seconds: 6)
    m.shots = [
        Shot(scene: "s1", dialogue: [l1, l2], transition: .cut, videos: [Take(file: "shots/0.mp4")]),
        Shot(scene: "s1", transition: .crossfade, videos: [Take(file: "shots/1.mp4")], sounds: [fx], selectedSound: fx.id),
    ]
    for i in m.shots.indices { m.shots[i].selectedVideo = m.shots[i].videos[0].id }
    return m
}

@Test func dialogueFollowsTheClipsTrimAndIsCutAtItsEnd() {
    var m = soundMovie()
    m.shots[0].trimStart = 1
    m.shots[0].trimEnd = 1.5
    let t = MovieTimeline(movie: m, durations: ["shots/0.mp4": 6, "shots/1.mp4": 4])
    // Clip 0: 1…4.5 of the file -> 3.5 s, then the crossfade clip starts 0.5 s early.
    #expect(t.clips[0].duration == 3.5 && t.clips[1].start == 3)
    let lines = t.sounds.filter { $0.kind == .dialogue }
    // Line 1 sits at 0.3…2.3 in the shot: 1.3 s of it after the trim, starting at the clip's start.
    #expect(abs(lines[0].start - 0) < 1e-9 && abs(lines[0].offset - 0.7) < 1e-9 && abs(lines[0].duration - 1.3) < 1e-9)
    // Line 2 at 2.55…5.55 in the shot -> 1.55…4.55 on the timeline, cut at the clip's end 3.5.
    #expect(abs(lines[1].start - 1.55) < 1e-9 && abs(lines[1].offset) < 1e-9 && abs(lines[1].duration - 1.95) < 1e-9)
    #expect(lines.allSatisfy { $0.label == "Ann" && $0.fadeIn == 0.02 && $0.fadeOut == 0.05 && $0.gain == 1 })
    #expect(t.subtitles.map(\.text) == ["Hello there.", "Second line."])
    #expect(t.subtitles.map(\.speaker) == ["Ann", "Ann"])
    #expect(abs(t.subtitles[1].end - 3.5) < 1e-9)
}

@Test func lineTakesGainAndMixApply() {
    var m = soundMovie()
    m.shots[0].dialogue[0].takes[0].gainDB = 6
    m.mix.dialogue = 0.5
    let t = MovieTimeline(movie: m, durations: ["shots/0.mp4": 6, "shots/1.mp4": 4])
    #expect(abs(t.sounds.first { $0.kind == .dialogue }!.gain - 0.5 * pow(10, 0.3)) < 1e-9)
}

@Test func duckingDipsTheMusicUnderMergedLines() {
    var m = MovieProject()
    m.scenes = [MovieScene(id: "s1", heading: "INT. ROOM")]
    m.characters = [MovieCharacter(id: "ann", name: "Ann", look: "", voice: "")]
    // Lines at 0.3…1.3, 1.55…2.55 (gap 0.25: merged), and 5.3…6.3 (far).
    let ls = [line("a", 1, file: "a.wav"), line("b", 1, file: "b.wav")]
    let far = line("c", 1, file: "c.wav")
    m.shots = [
        Shot(scene: "s1", duration: 5, dialogue: ls, videos: [Take(file: "v0.mp4")]),
        Shot(scene: "s1", duration: 5, dialogue: [far], videos: [Take(file: "v1.mp4")]),
    ]
    for i in m.shots.indices { m.shots[i].selectedVideo = m.shots[i].videos[0].id }
    let mt = take("music.wav", seconds: 20)
    m.music = [MusicCue(id: "s1", prompt: "p", takes: [mt], selected: mt.id)]
    m.mix.music = 1
    m.mix.duck = 0.5
    let t = MovieTimeline(movie: m, durations: ["v0.mp4": 5, "v1.mp4": 5])
    let music = t.sounds.first { $0.kind == .music }!
    #expect(music.start == 0 && music.duration == 10 && music.fadeIn == 0.5 && music.fadeOut == 1.5)
    let pts = t.volume(for: music)
    let expected: [(Double, Double)] = [
        (0, 0), (0.05, 0.1), (0.3, 0.3), (0.5, 0.5), (2.55, 0.5), (2.8, 1), (5.05, 1), (5.3, 0.5), (6.3, 0.5),
        (6.55, 1), (8.5, 1), (10, 0),
    ]
    #expect(pts.count == expected.count, "\(pts)")
    for (p, e) in zip(pts, expected) { #expect(abs(p.time - e.0) < 1e-9 && abs(p.gain - e.1) < 1e-9, "\(p) vs \(e)") }
}

@Test func overlappingEffectsGoToTwoLanes() {
    let m = soundMovie()
    let t = MovieTimeline(movie: m, durations: ["shots/0.mp4": 4, "shots/1.mp4": 4])
    // Add a second effect on the first shot, which the crossfade overlaps.
    var m2 = m
    let fx = take("audio/fx0.wav", seconds: 6)
    m2.shots[0].sounds = [fx]
    m2.shots[0].selectedSound = fx.id
    let t2 = MovieTimeline(movie: m2, durations: ["shots/0.mp4": 4, "shots/1.mp4": 4])
    let effects = t2.sounds.filter { $0.kind == .effects }.sorted { $0.start < $1.start }
    #expect(effects.map(\.track) == [0, 1] && effects[0].label == "Shot 1" && effects[1].label == "Shot 2")
    #expect(t.sounds.filter { $0.kind == .effects }.map(\.track) == [0])
    #expect(effects[1].fadeIn == 0.5 && effects[0].fadeIn == 0.05 && effects[0].fadeOut == 0.15)
}

@Test func srtHasHoursAndNumbers() {
    let subs = [
        MovieTimeline.Subtitle(start: 3723.5, end: 3725.0, speaker: "B", text: "Later."),
        MovieTimeline.Subtitle(start: 1.25, end: 2.5, speaker: "A", text: "Hello."),
    ]
    #expect(MovieSubtitles.srt(subs) == """
    1
    00:00:01,250 --> 00:00:02,500
    Hello.

    2
    01:02:03,500 --> 01:02:05,000
    Later.


    """)
    #expect(MovieSubtitles.srt([]) == "")
}

// MARK: sound templates

@Test func cuesKeepExistingOnesAndRendersRoundSeconds() throws {
    var m = MovieProject(style: "noir")
    m.scenes = [MovieScene(id: "a", heading: "A", mood: "tense"), MovieScene(id: "b", heading: "B"),
                MovieScene(id: "c", heading: "C", mood: "calm")]
    m.shots = [Shot(scene: "c", duration: 4.2), Shot(scene: "a", duration: 3), Shot(scene: "a", duration: 2.7)]
    m.music = [MusicCue(id: "c", prompt: "mine", takes: [take("m.wav")])]
    let cues = MovieSound.cues(for: m)
    #expect(cues.map(\.id) == ["a", "c"])
    #expect(cues[0].prompt == "tense, 1940s jazz, muted trumpet, double bass, smoky, instrumental")
    #expect(cues[1] == m.music[0])
    #expect(MovieSound.musicPrompt(for: m.scenes[1], style: .noir) == "1940s jazz, muted trumpet, double bass, smoky, instrumental")
    #expect(abs(MovieSound.sceneSeconds("a", in: m) - 5.7) < 1e-9)
    let r = MovieSound.musicRender(cues[0], seconds: 5.7, seed: 3, server: "gpu")
    #expect(r.values["seconds"] == .double(10) && r.values["lyrics"] == .string("[Instrumental]"))
    #expect(MovieSound.musicRender(cues[0], seconds: 20.2, seed: 3, server: "").values["seconds"] == .double(23))

    var shot = Shot(scene: "a", duration: 3.2, sfx: " door slam ")
    let fx = try #require(MovieSound.effectsRender(shot, seed: 1, server: "g"))
    #expect(fx.values["prompt"] == .string("door slam, sound effects only, no music, no speech"))
    #expect(fx.values["seconds"] == .double(3.5))
    shot.duration = 0.4
    #expect(MovieSound.effectsRender(shot, seed: 1, server: "")?.values["seconds"] == .double(1))
    shot.sfx = "  "
    #expect(MovieSound.effectsRender(shot, seed: 1, server: "") == nil)

    // Templates bind.
    let c = MovieCharacter(id: "x", name: "X", look: "", voice: " ")
    let sr = MovieVoices.sampleRender(c, seed: 1, server: "s")
    #expect(sr.values["description"] == .string("a clear, natural speaking voice"))
    _ = try MovieVoices.sampleTemplate().bind(sr.values)
    let lr = MovieVoices.lineRender(line(" hi ", 1), sample: URL(fileURLWithPath: "/m/s.wav"), seed: 2, server: "s")
    let g = try MovieVoices.lineTemplate().bind(lr.values)
    #expect(g["reference"]?.inputs["path"] == .value(.string("/m/s.wav")) && g["voice"]?.inputs["text"] == .value(.string("hi")))
    _ = try MovieSound.effectsTemplate().bind(fx.values)
    _ = try MovieSound.musicTemplate().bind(r.values)
    let tr = MovieShots.talkingRender(m.shots[0], movie: m, startImage: URL(fileURLWithPath: "/m/k.png"),
                                      voice: URL(fileURLWithPath: "/m/v.wav"), seed: 1, server: "s")
    let tg = try MovieShots.talkingTemplate().bind(tr.values)
    #expect(tg["voice"]?.inputs["path"] == .value(.string("/m/v.wav")))
    #expect(MovieShots.voiceTrackFile(for: m.shots[0]).hasPrefix("audio/shot-") && MovieShots.voiceTrackFile(for: m.shots[0]).hasSuffix("-voice.wav"))
}

// MARK: audio files

@Test func loudnessIgnoresSilenceAndAimsAtMinus20() throws {
    let folder = try tempFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    let url = folder.appendingPathComponent("sine.wav")
    try writeWAV(url, seconds: 2, amplitude: 0.1, silentHalf: true)
    let r = try MovieLoudness.measure(url)
    #expect(abs(r.seconds - 2) < 0.01)
    // 0.1 amplitude sine: RMS 0.0707 = -23 dBFS, so about +3 dB (the silent half is gated out).
    #expect(abs(r.gainDB - 3.01) < 0.3, "\(r.gainDB)")
    let quiet = folder.appendingPathComponent("silent.wav")
    try writeWAV(quiet, seconds: 1, amplitude: 0)
    #expect(try MovieLoudness.measure(quiet).gainDB == 0)
}

@Test func voiceTrackWritesTheLinesAt16kMono() throws {
    let folder = try tempFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    try writeWAV(folder.appendingPathComponent("audio/l.wav"), seconds: 1, amplitude: 0.5, rate: 24_000)
    let shot = Shot(scene: "s", dialogue: [line("x", 1, file: "audio/l.wav")], talking: true)
    let out = folder.appendingPathComponent("audio/track.wav")
    try MovieVoices.voiceTrack(shot, folder: folder, duration: 3, to: out)
    let file = try AVAudioFile(forReading: out)
    #expect(file.fileFormat.sampleRate == 16_000 && file.fileFormat.channelCount == 1 && file.length == 48_000)
    let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 48_000)!
    try file.read(into: buf)
    let d = buf.floatChannelData![0]
    func rms(_ a: Int, _ b: Int) -> Double { sqrt((a..<b).reduce(0.0) { $0 + Double(d[$1] * d[$1]) } / Double(b - a)) }
    #expect(rms(0, 4_000) < 0.001)               // lead-in 0.3 s = 4800 samples
    #expect(rms(6_000, 19_000) > 0.3)            // the line, 0.3…1.3 s
    #expect(rms(22_000, 48_000) < 0.001)
    #expect(throws: MovieAudioError.self) {
        try MovieVoices.voiceTrack(Shot(scene: "s"), folder: folder, duration: 3, to: out)
    }
}

@Test func eachSpeakerGetsATrackOnTheShotsTimeline() throws {
    let folder = try tempFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    try writeWAV(folder.appendingPathComponent("audio/a.wav"), seconds: 1, amplitude: 0.5, rate: 16_000)
    try writeWAV(folder.appendingPathComponent("audio/b.wav"), seconds: 1, amplitude: 0.5, rate: 16_000)
    let shot = Shot(scene: "s", dialogue: [line("hi", 1, character: "bo", file: "audio/a.wav"),
                                           line("hello", 1, character: "ann", file: "audio/b.wav"),
                                           line("bye", 0.5, character: "bo", file: "audio/a.wav")], talking: true)
    #expect(MovieVoices.speakers(shot) == ["bo", "ann"])  // in the order they first speak
    let out = folder.appendingPathComponent("audio/ann.wav")
    try MovieVoices.voiceTrack(shot, folder: folder, duration: 4, speaker: "ann", to: out)
    let file = try AVAudioFile(forReading: out)
    let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 64_000)!
    try file.read(into: buf)
    let d = buf.floatChannelData![0]
    func rms(_ a: Int, _ b: Int) -> Double { sqrt((a..<b).reduce(0.0) { $0 + Double(d[$1] * d[$1]) } / Double(b - a)) }
    #expect(file.length == 64_000)
    #expect(rms(5_000, 20_000) < 0.001)  // bo's first line (0.3…1.3 s) is not on Ann's track
    #expect(rms(26_000, 40_000) > 0.3)   // Ann's line, at its layout time (1.55…2.55 s)
    #expect(rms(46_000, 64_000) < 0.001)  // bo's last line, from 2.8 s
    #expect(MovieShots.voiceTrackFile(for: shot, speaker: 1).hasSuffix("-voice-2.wav"))
}

@Test func infiniteTalkCapsTalkingShotsAtAClipsLength() {
    let shot = Shot(scene: "s", duration: 3, dialogue: [line("long", 10)], talking: true)
    #expect(MovieVoices.maxDuration(shot, lipSync: true) == MovieVoices.maxTalking)
    #expect(MovieVoices.maxDuration(shot, lipSync: true, engine: .longcat) == MovieVoices.maxTalking)
    #expect(MovieVoices.maxDuration(shot, lipSync: true, engine: .infiniteTalk) == Shot.durationRange.upperBound)
    var m = MovieProject(shots: [shot], lipSyncEngine: .infiniteTalk)
    #expect(MovieVoices.fitDurations(&m) == [shot.id])  // 10.7 s of lines don't fit in 7.5
    #expect(m.shots[0].duration == Shot.durationRange.upperBound)
}

@Test func lipSyncTemplatesBindForEachEngine() throws {
    let movie = MovieProject(scenes: [MovieScene(id: "s", heading: "INT. ROOM")])
    let shot = Shot(scene: "s", duration: 4, talking: true)
    let folder = URL(fileURLWithPath: "/m")
    let start = URL(fileURLWithPath: "/m/k.png"), a = URL(fileURLWithPath: "/m/a.wav"), b = URL(fileURLWithPath: "/m/b.wav")

    // LongCat with one voice: the second Load Audio goes; with two, both feed the node.
    let one = MovieShots.longCatRender(shot, movie: movie, startImage: start, voices: [a], seed: 7, server: "gpu")
    let g1 = try MovieShots.longCatTemplate().bind(one.values, remove: one.unusedPictures)
    #expect(g1["voice2"] == nil && g1["video"]?.inputs["audio2"] == nil)
    #expect(g1["video"]?.inputs["server"] == .value(.string("gpu")))
    let two = MovieShots.longCatRender(shot, movie: movie, startImage: start, voices: [a, b], seed: 7, server: "gpu")
    let g2 = try MovieShots.longCatTemplate().bind(two.values, remove: two.unusedPictures)
    #expect(g2["voice2"]?.inputs["path"] == .value(.string("/m/b.wav")))
    #expect(g2["video"]?.inputs["audio2"] == .link(node: "voice2", output: "audio"))

    // InfiniteTalk: the ordinary clip into the dub, both on the server, no end picture for a shot without one.
    let ends = MovieShots.Ends(start: .keyframe("k.png"), end: nil)
    let dub = MovieShots.dubRender(shot, movie: movie, folder: folder, startImage: start, ends: ends, voice: a,
                                   seed: 9, server: "gpu")
    let g3 = try MovieShots.dubTemplate().bind(dub.values, remove: dub.unusedPictures)
    #expect(g3["end"] == nil && g3["video"]?.type == "RemoteWanImageToVideo")
    #expect(g3["talk"]?.inputs["video"] == .link(node: "video", output: "video"))
    #expect(g3["talk"]?.inputs["server"] == .value(.string("gpu")) && g3["video"]?.inputs["server"] == .value(.string("gpu")))
    #expect(g3["out"]?.inputs["video"] == .link(node: "talk", output: "video"))
    #expect(MovieShots.TemplateName.talking(.infiniteTalk) == "shot-infinitetalk")
    #expect(MovieShots.talkingTemplate(for: .longcat).graph["video"]?.type == "RemoteLongCatAvatar")
}

// MARK: composer

@Test func composerMixesSoundAndBurnsSubtitles() async throws {
    let folder = try tempFolder()
    defer { try? FileManager.default.removeItem(at: folder) }
    try FileManager.default.createDirectory(at: folder.appendingPathComponent("shots"), withIntermediateDirectories: true)
    for i in 0..<2 {
        try await writeClip(folder.appendingPathComponent("shots/\(i).mp4"), frames: 33, rgb: i == 0 ? (200, 30, 30) : (30, 30, 200))
    }
    try writeWAV(folder.appendingPathComponent("audio/l.wav"), seconds: 1)
    try writeWAV(folder.appendingPathComponent("audio/fx.wav"), seconds: 3, hz: 200)
    try writeWAV(folder.appendingPathComponent("audio/m.wav"), seconds: 6, hz: 330)

    var m = MovieProject()
    m.characters = [MovieCharacter(id: "ann", name: "Ann", look: "", voice: "")]
    m.scenes = [MovieScene(id: "s1", heading: "INT. ROOM")]
    let fx = take("audio/fx.wav", seconds: 3), mt = take("audio/m.wav", seconds: 6)
    m.shots = [
        Shot(scene: "s1", dialogue: [line("Hello there.", 1, file: "audio/l.wav")], videos: [Take(file: "shots/0.mp4")]),
        Shot(scene: "s1", transition: .crossfade, videos: [Take(file: "shots/1.mp4")], sounds: [fx], selectedSound: fx.id),
    ]
    for i in m.shots.indices { m.shots[i].selectedVideo = m.shots[i].videos[0].id }
    m.music = [MusicCue(id: "s1", prompt: "p", takes: [mt], selected: mt.id)]

    let durations = await MovieComposer.durations(of: ["shots/0.mp4", "shots/1.mp4"], in: folder)
    let timeline = MovieTimeline(movie: m, durations: durations)
    #expect(timeline.sounds.count == 3 && timeline.subtitles.count == 1)
    let comp = try await MovieComposer.build(timeline, folder: folder, size: (128, 96))
    #expect(comp.audioMix != nil && comp.subtitles.count == 1)

    let out = folder.appendingPathComponent("renders/film.mp4")
    try await MovieComposer.export(comp, to: out)
    let asset = AVURLAsset(url: out)
    #expect(try await asset.loadTracks(withMediaType: .video).count == 1)
    #expect(try await asset.loadTracks(withMediaType: .audio).count >= 1)
    let length = try await asset.load(.duration).seconds
    #expect(abs(length - timeline.duration) < 0.15, "\(length) vs \(timeline.duration)")

    let burned = folder.appendingPathComponent("renders/burned.mp4")
    try await MovieComposer.export(comp, to: burned, burnSubtitles: true)
    #expect(try await AVURLAsset(url: burned).loadTracks(withMediaType: .video).count == 1)

    let srt = folder.appendingPathComponent("renders/film.srt")
    try MovieComposer.writeSRT(comp.subtitles, to: srt)
    #expect(try String(contentsOf: srt, encoding: .utf8).contains("Hello there."))
}
