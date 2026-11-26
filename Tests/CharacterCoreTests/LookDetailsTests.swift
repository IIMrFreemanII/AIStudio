import Foundation
import MovieCore
import Testing
@testable import CharacterCore

private typealias CO = CharacterOptions

/// The detailed face and body options, the 18+ details and what a picture's coverage writes.
@Suite struct LookDetailsTests {
    private func person(_ gender: String = "woman") -> Character {
        Character(name: "Ava", age: 28, gender: gender, date: Date(timeIntervalSince1970: 1_700_000_000))
    }

    @Test func everyLookCategoryHasASectionAndEveryOptionAHint() {
        for cat in CO.categories(.look) { #expect(CO.section(of: cat.id) != nil, "\(cat.id)") }
        #expect(CO.categories(.intimate).map(\.id) == CO.intimateCategories)
        #expect(CO.detailCategories.allSatisfy { CO.category($0)?.group == .look })
        for cat in CO.all where cat.group == .look || cat.group == .intimate {
            for o in cat.options { #expect(!CO.hint(category: cat.id, option: o.id).isEmpty, "\(cat.id)/\(o.id)") }
        }
        // Sections in order, each with its categories in catalog order.
        #expect(CO.categories(.face).first?.id == "eyes" && CO.categories(.style).map(\.id) == ["clothes"])
    }

    /// Adults only: no youth words and nothing minor-coded in any look phrase or picture text.
    @Test func nothingMinorCoded() {
        let banned = ["petite", "tiny", "flat chest", "flat-chested", "young", "youthful", "childlike", "girlish",
                      "boyish", "baby", "little girl", "little boy"]
        for cat in CO.all where cat.group == .look || cat.group == .intimate {
            for o in cat.options {
                for g in ["woman", "man"] {
                    let parts = CO.lookParts([cat.id: [o.id]], gender: g, coverage: .nude, adult: true)
                    let texts = [o.label, o.phrase] + (parts.details + parts.body + parts.outfit).flatMap { [$0.tag, $0.text] }
                        + parts.without
                    for t in texts {
                        #expect(AgeGuard.youthTerm(in: t) == nil, "\(cat.id)/\(o.id): \(t)")
                        #expect(!banned.contains { t.lowercased().contains($0) }, "\(cat.id)/\(o.id): \(t)")
                    }
                }
            }
        }
    }

    @Test func intimateDetailsOnlyWhenShownAndTurnedOn() {
        let traits = ["nipples": ["small"], "pubicHair": ["shaved"], "body": ["slim"], "clothes": ["casual"]]
        func body(_ c: PromptSpec.Coverage, adult: Bool) -> String {
            let p = CO.lookParts(traits, gender: "woman", coverage: c, adult: adult)
            return (p.body + p.outfit).map(\.tag).joined(separator: ", ")
        }
        #expect(!body(.dressed, adult: true).contains("nipples") && body(.dressed, adult: true).contains("casual"))
        #expect(body(.topless, adult: true).contains("small nipples") && !body(.topless, adult: true).contains("pubic"))
        #expect(body(.nude, adult: true).contains("small nipples") && body(.nude, adult: true).contains("shaved pubic"))
        #expect(!body(.nude, adult: true).contains("casual"), "no clothing picks when undressed")
        #expect(!body(.nude, adult: false).contains("nipples") && !body(.impliedNude, adult: true).contains("nipples"))
        // Never in the looks the chat model, voices and the creator read; in the system prompt only when turned on.
        var c = person()
        c.traits = traits
        #expect(!c.fullLook.contains("nipples"))
        let on = CharacterPrompt.system(c, memory: CharacterMemory(), scene: nil, adult: true)
        let off = CharacterPrompt.system(c, memory: CharacterMemory(), scene: nil)
        #expect(on.contains("Private details: small nipples, a shaved pubic area") && !off.contains("nipples"))
    }

    @Test func chatPhotosReadTheCoverageFromTheRequest() {
        #expect(CharacterArt.coverage("send me a nude selfie") == .nude)
        #expect(CharacterArt.coverage("Topless on the beach") == .topless)
        #expect(CharacterArt.coverage("in a red bikini by the pool") == .swimwear)
        #expect(CharacterArt.coverage("lying in bed, covered by a sheet") == .impliedNude)
        #expect(CharacterArt.coverage("in black lingerie") == .lingerie)
        #expect(CharacterArt.coverage("reading a book on the sofa") == .dressed)
        var c = person()
        c.traits = ["nipples": ["pink"], "clothes": ["elegant"]]
        c.artModel = ArtModel(family: .chroma)
        let on = CharacterArt.photoPrompt(c, request: "a nude photo in the bedroom", scene: nil,
                                          context: .init(adult: true))
        let off = CharacterArt.photoPrompt(c, request: "a nude photo in the bedroom", scene: nil)
        #expect(on.contains("light pink nipples") && on.contains("She is fully nude"))
        #expect(!off.contains("nipples") && off.contains(PromptSpec.adultBody), "the adult cue stays either way")
        #expect(!on.contains("cocktail dress") && !on.contains("elegant"))
        let dressed = CharacterArt.photoPrompt(c, request: "a selfie at dinner", scene: nil, context: .init(adult: true))
        #expect(!dressed.contains("nipples") && !dressed.contains(PromptSpec.adultBody))
    }

    @Test func surpriseRollsNoIntimateDetails() {
        for _ in 0..<20 {
            let c = CharacterSurprise.roll(person())
            #expect(CO.intimateCategories.allSatisfy { c.traits[$0] == nil })
        }
    }

    @Test func aRaceWritesItsLookAndFeatureChipsReplaceParts() {
        func parts(_ traits: [String: [String]]) -> CO.LookParts { CO.lookParts(traits, gender: "woman") }
        #expect(parts(["race": ["elf"]]).race == "an elf with long pointed elf ears")
        #expect(parts(["race": ["witch"]]).race == "a witch with an air of magic")
        // A picked feature chip replaces that part of the race's look; the other parts stay.
        let demon = parts(["race": ["demon"], "horns": ["antlers"]])
        #expect(demon.race == "a demon with a thin spade-tipped tail" && demon.details.map(\.text) == ["antlers"])
        // A human skin tone replaces the race's skin; a fantasy skin replaces the human one.
        let oni = parts(["race": ["oni"], "skin": ["tan"]])
        #expect(oni.race == "an oni with oni horns and small fangs" && oni.details.map(\.text) == ["tan skin"])
        #expect(parts(["skin": ["tan"], "fantasySkin": ["blue"]]).details.map(\.text) == ["blue skin"])
        #expect(parts(["hairColor": ["red"]]).race.isEmpty)
        // In a photo prompt the race comes first, right after who they are.
        var c = person()
        c.traits = ["race": ["kitsune"], "hairColor": ["red"]]
        c.artModel = ArtModel(family: .chroma)
        let prompt = CharacterArt.photoPrompt(c, request: "a selfie in the garden", scene: nil)
        #expect(prompt.contains("of a 28-year-old adult woman, a kitsune with fox ears and several fluffy fox tails")
                && prompt.contains("She has red hair"), "\(prompt)")
        // The chat model knows the race (it's part of their looks).
        #expect(c.fullLook.hasPrefix("a kitsune"))
        // Every race of every gender has a look entry or a phrase, and the finer heritages exist.
        #expect(["japanese", "nordic", "west-african", "polynesian"].allSatisfy { id in
            CO.category("heritage")!.options.contains { $0.id == id } })
    }
}
