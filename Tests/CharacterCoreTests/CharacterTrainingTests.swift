import Foundation
import Testing
@testable import CharacterCore

@Suite struct CharacterTrainingTests {
    @Test func requestsAreVariedDressedAndSayNothingAboutTheLook() {
        let r = CharacterTraining.requests(count: 40, gender: "woman")
        #expect(r.count == 40 && Set(r).count == 40)
        // Every framing, angle, outfit and place shows up.
        for word in ["close-up", "full-body", "profile", "three-quarter", "swimsuit", "blazer", "beach", "rooftop"] {
            #expect(r.contains { $0.contains(word) }, "\(word)")
        }
        // The person's look comes from the photo prompt; the captions leave it out, so it binds to the trigger.
        for r in r {
            for look in ["hair", "eyes", "skin", "slim", "curvy", "nude", "naked", "topless"] {
                #expect(!r.contains(look), "\(look) in \(r)")
            }
        }
        #expect(CharacterTraining.requests(count: 40, gender: "woman") == r)  // deterministic
        #expect(CharacterTraining.requests(count: 8, gender: "man").contains { $0.contains("swim shorts") }
                || CharacterTraining.requests(count: 40, gender: "man").contains { $0.contains("swim shorts") })
    }

    @Test func captionsStartWithTheTriggerAndWho() {
        var c = Character(name: "Mia-Rose 2", age: 28, gender: "woman", look: "")
        c.id = UUID(uuidString: "3F9A0000-0000-0000-0000-000000000000")!
        let t = CharacterTraining.trigger(for: c)
        #expect(t == "zqmiaros_3f9")
        #expect(CharacterTraining.caption(request: "upper-body shot, at the beach", trigger: t, gender: "woman")
                == "zqmiaros_3f9, a woman, upper-body shot, at the beach")
        #expect(CharacterTraining.caption(request: "x", trigger: t, gender: "").hasPrefix("zqmiaros_3f9, a person"))
        #expect(CharacterTraining.task(.sdxl) == "train-sdxl-lora" && CharacterTraining.task(.minimax) == "train-h3-image-lora")
        #expect(CharacterTraining.task(.chroma) == nil)
    }

    @Test func trainingSetRoundTripsAndCountsPicks() throws {
        let store = CharacterStore(root: FileManager.default.temporaryDirectory.appendingPathComponent("train-\(UUID())"))
        let id = UUID()
        #expect(store.training(id) == TrainingSet())
        var set = TrainingSet()
        set.pictures = (1...16).map { .init(file: "p\($0).png", request: "r\($0)") }
        set.family = .minimax
        set.trigger = "zqmia_3f9"
        #expect(set.canTrain)
        set.pictures[0].picked = false
        set.pictures[1].picked = false
        #expect(!set.canTrain && set.picked.count == 14)
        try store.saveTraining(set, for: id)
        #expect(store.training(id) == set)
    }

    @Test func zipArchiveIsReadable() throws {
        // Two stored entries; the CRC is zlib's (crc32("hello") = 0x3610A686).
        #expect(ZipArchive.crc32(Data("hello".utf8)) == 0x3610_A686)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("z-\(UUID()).zip")
        try ZipArchive.write([("01.txt", Data("hello".utf8)), ("02.txt", Data("world".utf8))], to: url)
        let data = try Data(contentsOf: url)
        #expect(data.prefix(4) == Data([0x50, 0x4B, 0x03, 0x04]))
        #expect(data.suffix(22).prefix(4) == Data([0x50, 0x4B, 0x05, 0x06]))  // end of central directory
        #expect(String(decoding: data, as: UTF8.self).contains("01.txthello"))
    }
}
