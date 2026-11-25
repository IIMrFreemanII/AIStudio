import Foundation
import Testing
@testable import CharacterCore

private func person(gender: String = "woman") -> Character {
    Character(name: "Ava", age: 28, gender: gender, date: Date(timeIntervalSince1970: 1_700_000_000))
}

@Suite struct CharacterOptionsTests {
    @Test func toggleSingleReplacesAndRemoves() {
        var t = CharacterOptions.toggle([:], category: "hairColor", option: "auburn")
        #expect(t == ["hairColor": ["auburn"]])
        t = CharacterOptions.toggle(t, category: "hairColor", option: "black")
        #expect(t == ["hairColor": ["black"]])
        t = CharacterOptions.toggle(t, category: "hairColor", option: "black")
        #expect(t["hairColor"] == nil && t.isEmpty)
    }

    @Test func toggleFullDropsOldest() {
        var t: [String: [String]] = [:]
        for o in ["playful", "shy", "confident", "caring"] { t = CharacterOptions.toggle(t, category: "traits", option: o) }
        #expect(t["traits"] == ["playful", "shy", "confident", "caring"])
        t = CharacterOptions.toggle(t, category: "traits", option: "calm")
        #expect(t["traits"] == ["shy", "confident", "caring", "calm"])
    }

    @Test func toggleUnknownCategoryUnchanged() {
        let t = ["traits": ["shy"]]
        #expect(CharacterOptions.toggle(t, category: "nope", option: "x") == t)
    }

    @Test func phrasesInCatalogOrder() {
        let t = ["hairStyle": ["curly"], "hairColor": ["auburn"], "eyes": ["green", "bogus"], "unknown": ["x"]]
        #expect(CharacterOptions.phrases(t, group: .look) == ["auburn hair", "curly hair", "green eyes"])
        let order = ["traits": ["calm", "playful"]]
        #expect(CharacterOptions.phrases(order, categories: ["traits"]) == ["playful", "calm"])
        #expect(CharacterOptions.phrases(["traits": ["x"]], categories: ["traits", "nope"]).isEmpty)
    }

    @Test func fullLookCombinations() {
        var c = person()
        #expect(c.fullLook.isEmpty)
        c.traits = ["hairColor": ["auburn"], "eyes": ["green"]]
        #expect(c.fullLook == "auburn hair, green eyes")
        c.look = "paint-stained sleeves."
        #expect(c.fullLook == "auburn hair, green eyes, paint-stained sleeves")
        c.traits = [:]
        c.look = "  freckles ; "
        #expect(c.fullLook == "freckles")
    }

    @Test func habitsEmptyWithoutPicks() {
        var c = person()
        #expect(c.habits.isEmpty)
        c.traits = ["rhythm": ["early-riser"], "habits": ["yoga"]]
        #expect(c.habits == "early riser, does yoga")
    }

    @Test func optionsByGender() throws {
        let extras = try #require(CharacterOptions.category("extras"))
        let man = CharacterOptions.options(extras, gender: " Man ").map(\.id)
        #expect(man.contains("beard") && !man.contains("bold-makeup"))
        let woman = CharacterOptions.options(extras, gender: "woman").map(\.id)
        #expect(woman.contains("bold-makeup") && !woman.contains("beard"))
        for g in ["non-binary", ""] {
            #expect(CharacterOptions.options(extras, gender: g) == extras.options)
        }
    }

    @Test func decodeWithoutTraitsAndRoundTrip() throws {
        let id = UUID()
        let old = try JSONDecoder().decode(Character.self, from: Data("{\"id\":\"\(id.uuidString)\",\"name\":\"X\"}".utf8))
        #expect(old.traits.isEmpty)
        var c = person()
        c.traits = ["traits": ["shy", "calm"], "hairColor": ["red"]]
        let data = try CharacterStore.encoder.encode(c)
        let back = try CharacterStore.decoder.decode(Character.self, from: data)
        #expect(back.traits == c.traits && back == c)
    }

    @Test func catalogIsConsistent() {
        let ids = CharacterOptions.all.map(\.id)
        #expect(Set(ids).count == ids.count)
        for cat in CharacterOptions.all {
            let o = cat.options.map(\.id)
            #expect(Set(o).count == o.count, "\(cat.id)")
            #expect(cat.max >= 1 && cat.options.count >= 2)
            for opt in cat.options {
                #expect(AgeGuard.youthTerm(in: opt.label) == nil, "\(cat.id)/\(opt.id)")
                #expect(AgeGuard.youthTerm(in: opt.phrase) == nil, "\(cat.id)/\(opt.id)")
            }
        }
        #expect(CharacterOptions.categories(.voice).map(\.id) == ["voiceTone", "voicePitch", "voicePace", "accent"])
    }

    @Test func promptsUsePicks() {
        var c = person()
        c.look = "paint-stained sleeves"
        c.traits = ["hairColor": ["auburn"], "rhythm": ["early-riser"], "hobbies": ["art"]]
        let p = CharacterPrompt.system(c, memory: CharacterMemory(), scene: nil)
        #expect(p.contains("Looks: auburn hair, paint-stained sleeves"))
        #expect(p.contains("Habits: early riser"))
        #expect(p.contains("Likes: art"))
        #expect(CharacterArt.portraitPrompt(c).contains("auburn hair"))
    }

    @Test func templatesAreConsistent() {
        for t in CharacterTemplates.all {
            #expect(!t.fullLook.isEmpty, "\(t.name)")
            for (cat, ids) in t.traits {
                let c = CharacterOptions.category(cat)
                #expect(c != nil, "\(t.name) \(cat)")
                for id in ids { #expect(c?.options.contains { $0.id == id } == true, "\(t.name) \(cat)/\(id)") }
                #expect(ids.count <= (c?.max ?? 0), "\(t.name) \(cat)")
            }
        }
    }
}
