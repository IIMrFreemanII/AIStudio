import CoreGraphics
import Foundation
import ImageIO
import GraphRuntime
@testable import MovieCore
import Testing

private func spoken(_ text: String, _ character: String = "maya", delivery: String = "") -> DialogueLine {
    DialogueLine(character: character, text: text, delivery: delivery)
}

private func h3Movie() -> MovieProject {
    let sheet = Take(file: "cast/maya.png"), voice = Take(file: "audio/voice-maya.wav")
    var m = MovieProject(videoEngine: .minimaxH3)
    m.characters = [MovieCharacter(id: "maya", name: "Maya", look: "", voice: "young, clear, quiet",
                                   sheets: [sheet], selectedSheet: sheet.id, voiceSamples: [voice],
                                   selectedVoiceSample: voice.id),
                    MovieCharacter(id: "grandpa", name: "Grandpa", look: "", voice: "old, raspy")]
    return m
}

@Test func videoEngineRoundTripsDefaultsToWanAndSizesTheClips() throws {
    #expect(MovieProject().videoEngine == .wan22 && MovieProject().videoSize == (832, 480))
    var m = MovieProject(videoEngine: .minimaxH3)
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONEncoder().encode(m)).videoEngine == .minimaxH3)
    var json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(m)) as! [String: Any]
    json.removeValue(forKey: "videoEngine")  // movies from before the choice
    #expect(try JSONDecoder().decode(MovieProject.self, from: JSONSerialization.data(withJSONObject: json)).videoEngine == .wan22)
    #expect(m.videoSize == (1344, 768) && m.aspect.h3Aspect == "16:9")
    m.aspect = .vertical
    #expect(m.videoSize == (768, 1344) && m.aspect.h3Aspect == "9:16")
    #expect(VideoEngine.minimaxH3.makesSound && !VideoEngine.wan22.makesSound)
}

@Test func clipsRecordTheirEngineSoASwitchOffersTheMissingOnes() throws {
    var shot = Shot(scene: "s1")
    #expect(!shot.hasClip(by: .wan22) && !shot.hasClip(by: .minimaxH3))
    shot.videos = [Take(file: "shots/old.mp4")]  // before engines were recorded: Wan 2.2's
    #expect(shot.hasClip(by: .wan22) && !shot.hasClip(by: .minimaxH3))
    shot.videos.append(Take(file: "shots/h3.mp4", engine: VideoEngine.minimaxH3.rawValue))
    #expect(shot.hasClip(by: .minimaxH3))
    let back = try JSONDecoder().decode(Take.self, from: JSONEncoder().encode(shot.videos[1]))
    #expect(back.engine == "minimax-h3")
}

@Test func h3EstimatesLineTimesAndMakesRoomForThem() {
    var m = h3Movie()
    m.shots = [Shot(scene: "a", duration: 3, dialogue: [spoken("Stay."), spoken(" "),
                                                        spoken("The light has to stay on tonight, whatever happens out there.", "grandpa")]),
               Shot(scene: "a", duration: 3.2)]
    let times = MovieH3.lineTimes(m.shots[0])
    #expect(times.count == 2 && times[0].start == MovieH3.leadIn && times[0].seconds == 0.8)  // one word: the minimum
    #expect(abs(times[1].start - (MovieH3.leadIn + 0.8 + MovieH3.gap)) < 1e-9)
    #expect(MovieH3.speakers(m.shots[0]) == ["maya", "grandpa"])
    let need = MovieH3.neededDuration(m.shots[0])!
    #expect(MovieH3.fitDurations(&m).isEmpty)
    #expect(m.shots[0].duration >= need && m.shots[0].duration < need + 0.11 && m.shots[1].duration == 3.2)
    // H3 makes 5–15 s: a short shot asks for 5 (the edit plays its own length).
    #expect(MovieH3.clipSeconds(m.shots[1]) == 5 && MovieH3.clipSeconds(Shot(scene: "a", duration: 7.2)) == 7.5)
    m.shots[0].dialogue = [spoken(String(repeating: "word ", count: 60))]
    #expect(MovieH3.fitDurations(&m) == [m.shots[0].id] && m.shots[0].duration == 15)
}

@Test func h3PromptsFollowTheGuidesStructure() {
    let m = h3Movie()
    let shot = Shot(scene: "a", duration: 5, framing: "wide", camera: "slow push in", motionPrompt: "Maya steps forward",
                    dialogue: [spoken("Stay.", delivery: "soft, firm")], sfx: "howling wind, distant thunder")
    let p = MovieH3.prompt(for: shot, in: m, start: true, end: true)
    #expect(p.hasPrefix("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."))
    #expect(p.contains("integrated_multimodal_description: [Shot 1]") && p.contains("a wide shot"))
    #expect(p.contains("Maya steps forward.") && p.contains("Camera: slow push in."))
    #expect(p.contains("Maya, with a young, clear, quiet voice, (S1) says (soft, firm): <d>[English] Stay.</d>"))
    #expect(p.contains("ends on the composition of <Picture 2>"))
    #expect(p.contains("overall_soundscape: Howling wind, distant thunder.") && p.hasSuffix("non_diegetic_music: none"))
    #expect(!MovieH3.prompt(for: shot, in: m, start: false, end: false).contains("<Picture"))

    // References: the keyframe, Maya's sheet, Maya's voice; Grandpa (no sheet, no voice) by name.
    var talk = shot
    talk.dialogue.append(spoken("Go inside.", "grandpa"))
    #expect(MovieH3.usesReferences(talk, in: m))
    let refs = MovieH3.references(for: talk, in: m)
    #expect(refs.sheets == ["maya"] && refs.voices == ["maya"])
    let views = [MovieH3.SubjectPicture(id: "maya", view: .face), .init(id: "maya", view: .body)]
    let r = MovieH3.referencePrompt(for: talk, in: m, pictures: views, voices: refs.voices)
    for part in ["subject_definitions: <Picture 1> is the planned opening composition",
                 "<Subject 1> is Maya: their face and hair in <Picture 2>, and their whole figure and clothing in <Picture 3>.",
                 "<Picture 3>: reference - Maya's clothing and build are kept, never shown as a picture of its own.",
                 "in the setting of <Picture 1>, one continuous take.",
                 "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).",
                 "summary: One wide shot in the setting of <Picture 1> in which Maya and Grandpa speak.",
                 "<Audio 1>: reference - Maya speaks in this voice timbre",
                 "<Subject 1> (S1) says (soft, firm): <d>[English] Stay.</d>",
                 "Grandpa, with a old, raspy voice, (S2) says: <d>[English] Go inside.</d>",
                 "non_diegetic_music: none"] {
        #expect(r.contains(part), "\(part)")
    }
    // No voice for any speaker: not from references.
    #expect(!MovieH3.usesReferences(Shot(scene: "a", dialogue: [spoken("Hm.", "grandpa")]), in: m))
}

@Test func h3TemplatesBindAndDropWhatAShotDoesntUse() throws {
    let m = h3Movie()
    let folder = URL(fileURLWithPath: "/m")
    let shot = Shot(scene: "a", duration: 4, dialogue: [spoken("Stay.")])
    let start = URL(fileURLWithPath: "/m/k.png")
    let plain = MovieH3.videoRender(shot, movie: m, folder: folder, startImage: start,
                                    ends: MovieShots.Ends(start: .keyframe("k.png"), end: nil), seed: 3, server: "g")
    let g = try MovieH3.videoTemplate().bind(plain.values, remove: plain.unusedPictures)
    #expect(g["end"] == nil && g["video"]?.inputs["last_image"] == nil && g["start"]?.inputs["path"] == .value(.string("/m/k.png")))
    #expect(g["video"]?.inputs["seconds"] == .value(.double(5)) && g["video"]?.inputs["aspect"] == .value(.string("16:9")))
    let ended = MovieH3.videoRender(shot, movie: m, folder: folder, startImage: start,
                                    ends: MovieShots.Ends(start: .lastFrame(of: "c.mp4"), end: "k2.png"), seed: 3, server: "g")
    #expect(try MovieH3.videoTemplate().bind(ended.values, remove: ended.unusedPictures)["end"]?.inputs["path"]
            == .value(.string("/m/k2.png")))

    let refs = MovieH3.referencesRender(shot, movie: m, folder: folder, keyframe: start, seed: 4, server: "g")
    #expect(refs.unusedPictures == ["picture3", "picture4", "voice2", "voice3"])
    let rg = try MovieH3.referencesTemplate().bind(refs.values, remove: refs.unusedPictures)
    #expect(rg["picture2"]?.inputs["path"] == .value(.string("/m/cast/maya.png")))
    #expect(rg["voice1"]?.inputs["path"] == .value(.string("/m/audio/voice-maya.wav")))
    #expect(rg["picture3"] == nil && rg["video"]?.inputs["image3"] == nil && rg["video"]?.inputs["audio2"] == nil)
    #expect(MovieH3.work(shot, in: m) == MovieShots.work(width: 1344, height: 768, frames: 120))
}

@Test func h3CutsSheetsIntoViewsSoTheWholeSheetIsNeverAReference() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("h3-sheet-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir.appendingPathComponent("cast"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    // A 300×100 "sheet": three views side by side (red, green, blue thirds).
    let ctx = CGContext(data: nil, width: 300, height: 100, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    for (i, c) in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)].enumerated() {
        ctx.setFillColor(red: c.0, green: c.1, blue: c.2, alpha: 1)
        ctx.fill(CGRect(x: i * 100, y: 0, width: 100, height: 100))
    }
    try MovieComposer.writePNG(ctx.makeImage()!, to: dir.appendingPathComponent("cast/maya.png"))
    let panels = MovieH3.sheetPanels(dir.appendingPathComponent("cast/maya.png"))
    #expect(panels[.body]?.lastPathComponent == "maya-body.png" && panels[.face]?.lastPathComponent == "maya-face.png")
    func pixel(_ url: URL) -> (Int, Int) {  // width, and which colour is strongest at its centre (0 r, 1 g, 2 b)
        let img = CGImageSourceCreateImageAtIndex(CGImageSourceCreateWithURL(url as CFURL, nil)!, 0, nil)!
        let c = CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                          space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.draw(img, in: CGRect(x: -img.width / 2, y: -img.height / 2, width: img.width, height: img.height))
        let d = c.data!.assumingMemoryBound(to: UInt8.self)
        return (img.width, [0, 1, 2].max { d[$0] < d[$1] }!)
    }
    #expect(pixel(panels[.body]!).1 == 0 && pixel(panels[.face]!).1 == 2 && pixel(panels[.face]!).0 < 100)

    // One speaker: their face, then their whole figure; the prompt and the pictures agree.
    let m = h3Movie()
    let shot = Shot(scene: "a", duration: 4, dialogue: [spoken("Stay.")])
    let subjects = MovieH3.subjectPictures(for: shot, in: m, folder: dir)
    #expect(subjects.pictures == [.init(id: "maya", view: .face), .init(id: "maya", view: .body)])
    let refs = MovieH3.referencesRender(shot, movie: m, folder: dir, keyframe: dir.appendingPathComponent("k.png"),
                                        seed: 1, server: "g")
    #expect(refs.values["picture2"] == .string(dir.appendingPathComponent("cast/maya-face.png").path))
    #expect(refs.values["picture3"] == .string(dir.appendingPathComponent("cast/maya-body.png").path))
    #expect(refs.unusedPictures == ["picture4", "voice2", "voice3"])
}

@Test func h3TimelinePlaysEachClipsOwnSoundForTheShotsLength() {
    var m = h3Movie()
    let clip = Take(file: "shots/a.mp4"), clip2 = Take(file: "shots/b.mp4")
    var a = Shot(scene: "s1", duration: 3, dialogue: [spoken("Stay.")], talking: true)
    a.videos = [clip]; a.selectedVideo = clip.id
    var b = Shot(scene: "s1", duration: 6)
    b.videos = [clip2]; b.selectedVideo = clip2.id
    m.shots = [a, b]
    let mt = Take(file: "audio/m.wav", seconds: 30)
    m.music = [MusicCue(id: "s1", prompt: "p", takes: [mt], selected: mt.id)]
    let t = MovieTimeline(movie: m, durations: ["shots/a.mp4": 5.04, "shots/b.mp4": 6.04])
    // A 5 s H3 clip plays for its 3 s shot; the next starts there.
    #expect(t.clips.map(\.duration) == [3, 6] && t.clips[1].start == 3)
    let clips = t.sounds.filter { $0.kind == .clip }
    #expect(clips.map(\.file) == ["shots/a.mp4", "shots/b.mp4"] && clips.map(\.duration) == [3, 6])
    #expect(!t.sounds.contains { $0.kind == .dialogue || $0.kind == .effects })
    // The line's estimated time as a subtitle; the music dips under it.
    #expect(t.subtitles.count == 1 && t.subtitles[0].text == "Stay." && t.subtitles[0].start == MovieH3.leadIn)
    let music = t.sounds.first { $0.kind == .music }!
    let sub = t.subtitles[0]
    let during = t.volume(for: music).filter { $0.time >= sub.start - 1e-9 && $0.time <= sub.end + 1e-9 }
    #expect(!during.isEmpty && during.allSatisfy { $0.gain < music.gain * 0.5 })
    // The same movie with Wan 2.2: clips play whole, no clip sound.
    m.videoEngine = .wan22
    let w = MovieTimeline(movie: m, durations: ["shots/a.mp4": 5.04, "shots/b.mp4": 6.04])
    #expect(w.clips[0].duration == 5.04 && !w.sounds.contains { $0.kind == .clip })
}
