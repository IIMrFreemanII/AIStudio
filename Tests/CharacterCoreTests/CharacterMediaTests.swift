import Foundation
import GraphRuntime
import MovieCore
import Testing
@testable import CharacterCore

private func hero() -> Character { Character(name: "Ava", age: 28, gender: "woman", look: "red hair") }

private var utc: Calendar {
    var c = Calendar(identifier: .gregorian)
    c.timeZone = TimeZone(identifier: "UTC")!
    return c
}

@Suite struct CharacterMediaTests {
    @Test func voiceDescriptionSaysWhoSpeaks() {
        var c = hero()
        #expect(CharacterMedia.voiceDescription(c) == "A 28-year-old woman's voice")
        c.traits = CharacterOptions.toggle(c.traits, category: "voicePitch", option: "low")
        c.voice = "a little raspy"
        #expect(CharacterMedia.voiceDescription(c) == "A 28-year-old woman's voice: low-pitched, a little raspy")
        let render = CharacterMedia.voiceSampleRender(c, seed: 1, server: "s")
        #expect(render.values["description"] == .string("A 28-year-old woman's voice: low-pitched, a little raspy"))
    }

    @Test func askDetection() {
        #expect(MediaAsk.asked(.photo, in: "send me a selfie"))
        #expect(!MediaAsk.asked(.photo, in: "what a lovely picnic"))
        #expect(MediaAsk.asked(.video, in: "send a video") && !MediaAsk.asked(.talk, in: "send a video"))
        #expect(MediaAsk.asked(.talk, in: "send a video message"))
        #expect(MediaAsk.asked(.talk, in: "say it on camera") && MediaAsk.asked(.video, in: "say it on camera"))
        #expect(MediaAsk.asked(.voice, in: "can I hear your voice?"))
        for k in MediaKind.allCases { #expect(!MediaAsk.asked(k, in: "how was your day")) }
    }

    @Test func decide() {
        #expect(CharacterCost.decide(.photo, estimate: 0.01, autoLimit: 1, asked: false, initiativeLeft: 0)
            == .skip("No more unasked media today"))
        #expect(CharacterCost.decide(.photo, estimate: 0.01, autoLimit: 1, asked: true, initiativeLeft: 0) == .run)
        #expect(CharacterCost.decide(.video, estimate: 0.001, autoLimit: 1, asked: true, initiativeLeft: 0) == .ask)
        #expect(CharacterCost.decide(.talk, estimate: nil, autoLimit: 1, asked: false, initiativeLeft: 2) == .ask)
        #expect(CharacterCost.decide(.photo, estimate: 2, autoLimit: 1, asked: true, initiativeLeft: 0) == .ask)
        #expect(CharacterCost.decide(.photo, estimate: nil, autoLimit: 1, asked: true, initiativeLeft: 0) == .run)
        #expect(CharacterCost.decide(.voice, estimate: 0.5, autoLimit: 1, asked: false, initiativeLeft: 1) == .run)
        #expect(CharacterCost.alwaysAsks(.video) && CharacterCost.alwaysAsks(.talk))
        #expect(!CharacterCost.alwaysAsks(.photo) && !CharacterCost.alwaysAsks(.voice))
    }

    @Test func initiativeCounting() {
        let now = Date(timeIntervalSince1970: 1_799_971_200 + 12 * 3600)
        var m = CharacterMemory()
        m.ownMedia = [now, now.addingTimeInterval(-3600), now.addingTimeInterval(-86400)]
        #expect(m.ownMediaToday(now: now, calendar: utc) == 2)
        m.noteOwnMedia(at: now)
        #expect(m.ownMedia.count == 4)
        var old = CharacterMemory()
        old.ownMedia = [now.addingTimeInterval(-8 * 86400)]
        old.noteOwnMedia(at: now)
        #expect(old.ownMedia == [now])
        var c = hero()
        c.initiative = 1
        #expect(CharacterCost.initiativeLeft(c, memory: m, now: now, calendar: utc) == 0)
        c.initiative = 5
        #expect(CharacterCost.initiativeLeft(c, memory: m, now: now, calendar: utc) == 5 - m.ownMediaToday(now: now, calendar: utc))
    }

    @Test func jobLists() {
        let c = hero()
        #expect(CharacterMedia.jobs(.voice, character: c, text: "hi there", hasVoiceSample: false).count == 2)
        #expect(CharacterMedia.jobs(.voice, character: c, text: "hi there", hasVoiceSample: true).count == 1)
        #expect(CharacterMedia.jobs(.photo, character: c, text: "", hasVoiceSample: true).map(\.task) == ["sdxl"])
        #expect(CharacterMedia.jobs(.video, character: c, text: "", hasVoiceSample: true).map(\.task) == ["sdxl", "wan22-i2v"])
        #expect(CharacterMedia.jobs(.talk, character: c, text: "hello", hasVoiceSample: false).map(\.task)
            == ["sdxl", "qwen3-tts", "qwen3-tts", "longcat-avatar"])
        var chroma = hero()
        chroma.artModel = ArtModel(family: .chroma)
        #expect(CharacterMedia.jobs(.photo, character: chroma, text: "", hasVoiceSample: true)[0].task == "chroma")
        var breeze = hero()
        breeze.voiceEngine = .breeze
        #expect(CharacterMedia.jobs(.voice, character: breeze, text: "hi", hasVoiceSample: true)[0].task == "breeze-tts")
        #expect(CharacterMedia.speechSeconds("") == 1)
    }

    @Test func estimates() {
        let c = hero(), t = GPUTimings()
        let jobs = CharacterMedia.jobs(.talk, character: c, text: "hello there friend", hasVoiceSample: false)
        let run = jobs.reduce(0.0) { $0 + t.seconds(task: $1.task, megapixels: $1.work) * 0.001 }
        let warm = CharacterMedia.estimate(.talk, character: c, text: "hello there friend", hasVoiceSample: false,
                                           timings: t, price: { _ in 0.001 }, warm: { _ in true })
        #expect(abs(warm! - run) < 1e-9)
        let loads = Set(jobs.map(\.task)).reduce(0.0) { $0 + t.loadSeconds(task: $1) * 0.001 }
        let cold = CharacterMedia.estimate(.talk, character: c, text: "hello there friend", hasVoiceSample: false,
                                           timings: t, price: { _ in 0.001 }, warm: { _ in false })
        #expect(abs(cold! - (run + loads)) < 1e-9)
        let none = CharacterMedia.estimate(.talk, character: c, text: "hello", hasVoiceSample: false,
                                           timings: t, price: { $0 == "longcat-avatar" ? nil : 0.001 }, warm: { _ in true })
        #expect(none == nil)
    }

    @Test func rendersBind() throws {
        let c = hero()
        let img = URL(fileURLWithPath: "/tmp/a.png"), wav = URL(fileURLWithPath: "/tmp/a.wav")
        let v = CharacterMedia.videoRender(c, prompt: "p", start: img, seed: 3, server: "s")
        #expect(v.values["width"] == .int(480) && v.values["height"] == .int(832))
        #expect(v.values["length"] == .int(MovieShots.frames(for: 5)))
        #expect(v.values["start"] == .string("/tmp/a.png") && v.values["negative"] == .string(CharacterArt.negative(c)))
        #expect(v.unusedPictures == ["end"])
        _ = try CharacterMedia.videoTemplate().bind(v.values, optional: Set(v.values.keys), remove: v.unusedPictures)
        let t = CharacterMedia.talkRender(c, prompt: "p", start: img, voice: wav, seed: 3, server: "s")
        #expect(t.values["voice"] == .string("/tmp/a.wav") && t.unusedPictures == ["voice2"])
        _ = try CharacterMedia.talkTemplate().bind(t.values, optional: Set(t.values.keys), remove: t.unusedPictures)
        let s = CharacterMedia.voiceSampleRender(c, seed: 1, server: "s")
        _ = try CharacterMedia.voiceSampleTemplate(c).bind(s.values, optional: Set(s.values.keys), remove: s.unusedPictures)
        let l = CharacterMedia.voiceLineRender(c, text: "hi", sample: wav, seed: 1, server: "s")
        _ = try CharacterMedia.voiceLineTemplate(c).bind(l.values, optional: Set(l.values.keys), remove: l.unusedPictures)
        var b = hero()
        b.voiceEngine = .breeze
        let bl = CharacterMedia.voiceLineRender(b, text: "hi", sample: wav, seed: 1, server: "s")
        _ = try CharacterMedia.voiceLineTemplate(b).bind(bl.values, optional: Set(bl.values.keys), remove: bl.unusedPictures)
    }

    @Test func prompts() {
        let p = CharacterMedia.motionPrompt(hero(), request: "You wave at the camera.")
        #expect(p.contains("adult") && p.contains("28") && p.contains("Wave at the camera") && !p.contains("You ") && !p.contains(".."))
        #expect(!CharacterMedia.motionPrompt(Character(name: "X", age: 30, gender: "", look: ""), request: "waves").contains(".."))
        #expect(CharacterMedia.talkPrompt(hero()).hasPrefix("A 28-year-old adult woman talks to the camera"))
    }

    @Test func characterCodingWithNewKeys() throws {
        let json = #"{"id":"6E2C7A14-0000-4000-8000-000000000001","name":"Ava"}"#.data(using: .utf8)!
        let d = try JSONDecoder().decode(Character.self, from: json)
        #expect(d.voiceEngine == .qwen3 && d.voiceSamples.isEmpty && d.selectedVoiceSample == nil && d.media == MediaKind.allCases)
        var c = hero()
        c.voiceEngine = .breeze
        c.voiceSamples = ["a.wav"]
        c.selectedVoiceSample = "a.wav"
        c.media = [.photo, .voice]
        let back = try JSONDecoder().decode(Character.self, from: JSONEncoder().encode(c))
        #expect(back.voiceEngine == .breeze && back.voiceSamples == ["a.wav"] && back.selectedVoiceSample == "a.wav" && back.media == [.photo, .voice])
    }
}

@Suite struct SpokenWordsInstructionTests {
    @Test func spokenWordsAreNotRepeated() {
        #expect(MediaTags.instructions([.voice]).contains("don't also write them outside the tag"))
        #expect(MediaTags.instructions([.talk]).contains("don't also write them outside the tag"))
        #expect(!MediaTags.instructions([.photo, .video]).contains("outside the tag"))
    }
}
