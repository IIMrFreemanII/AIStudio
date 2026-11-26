import Testing
@testable import MovieCore

@Suite struct PromptCatalogTests {
    @Test func idsAreUniqueAndEveryChoiceHasWords() {
        let groups = PromptCatalog.groups
        #expect(Set(groups.map(\.id)).count == groups.count)
        for g in groups {
            #expect(Set(g.choices.map(\.id)).count == g.choices.count, "\(g.id)")
            #expect(g.choices.allSatisfy { !$0.words.isEmpty && !$0.label.isEmpty }, "\(g.id)")
            #expect(g.max >= 1 && g.choices.count >= (g.adult ? 5 : 8), "\(g.id) is rich enough")
        }
        #expect(groups.reduce(0) { $0 + $1.choices.count } >= 200)
    }

    @Test func videoAndSoundGroupsOnlyWhereTheyApply() {
        let image = Set(PromptCatalog.groups(for: .sdxl).map(\.id))
        #expect(!image.contains("movement") && !image.contains("sound") && image.contains("lighting"))
        let wan = Set(PromptCatalog.groups(for: .wan).map(\.id))
        #expect(wan.contains("movement") && !wan.contains("sound"))
        #expect(Set(PromptCatalog.groups(for: .ltx23).map(\.id)).isSuperset(of: ["movement", "sound"]))
    }

    @Test func picksFillTheSpecAndTheWritersPlaceThem() {
        var s = PromptSpec()
        s.subject = "a red fox"
        let picks = ["medium": ["film-35mm"], "shot": ["wide"], "lighting": ["golden-hour", "backlit"],
                     "movement": ["tracking"], "sound": ["waves"], "place": ["beach"]]
        PromptCatalog.apply(picks, to: &s, model: .sdxl)
        #expect(s.medium == "35mm film photo" && s.shot == "wide shot" && s.place == "a sandy beach")
        #expect(s.lighting == ["warm golden-hour sunlight", "backlit, glowing rim light"])
        #expect(s.camera.isEmpty && s.sound.isEmpty)  // no camera moves or sound for a still model
        let tags = PromptWriter.write(s, for: .sdxl).prompt
        #expect(tags.hasPrefix("35mm film photo, a red fox, wide shot, a sandy beach, warm golden-hour sunlight"))

        var v = PromptSpec()
        v.subject = "a red fox"
        PromptCatalog.apply(picks, to: &v, model: .ltx23)
        #expect(v.camera == ["tracking shot following the subject"] && v.sound == ["ocean waves"])
        // What the spec already has wins over a single-value pick.
        var typed = PromptSpec()
        typed.place = "my grandmother's garden"
        PromptCatalog.apply(picks, to: &typed, model: .sdxl)
        #expect(typed.place == "my grandmother's garden")
    }

    @Test func toggleFollowsEachGroupsLimit() {
        var p: [String: [String]] = [:]
        p = PromptCatalog.toggle(p, group: "shot", choice: "wide")
        p = PromptCatalog.toggle(p, group: "shot", choice: "close-up")
        #expect(p["shot"] == ["close-up"])
        for c in ["natural", "rim", "neon", "moonlight"] { p = PromptCatalog.toggle(p, group: "lighting", choice: c) }
        #expect(p["lighting"] == ["rim", "neon", "moonlight"])
        p = PromptCatalog.toggle(p, group: "lighting", choice: "neon")
        #expect(p["lighting"] == ["rim", "moonlight"])
        p = PromptCatalog.toggle(p, group: "shot", choice: "close-up")
        #expect(p["shot"] == nil)
        #expect(PromptCatalog.toggle(p, group: "nope", choice: "x") == p)
    }
}

@Suite struct PromptCatalogAdultTests {
    @Test func nudityOnlyWithTheAdultOptions() {
        #expect(!PromptCatalog.groups(for: .sdxl).contains { $0.id == "nudity" })
        #expect(PromptCatalog.groups(for: .sdxl, adult: true).contains { $0.id == "nudity" })
        let picks = ["nudity": ["topless"]]
        #expect(PromptCatalog.coverage(picks, adult: false) == .dressed)
        #expect(PromptCatalog.coverage(picks, adult: true) == .topless)
        var s = PromptSpec()
        PromptCatalog.apply(picks, to: &s, model: .sdxl)
        #expect(s.coverage == .dressed)
        PromptCatalog.apply(picks, to: &s, model: .sdxl, adult: true)
        #expect(s.coverage == .topless)
        // Every choice is a coverage.
        for c in PromptCatalog.group("nudity")!.choices { #expect(PromptSpec.Coverage(rawValue: c.id) != nil, "\(c.id)") }
    }
}
