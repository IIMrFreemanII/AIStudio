import AppKit
import Foundation
import MovieCore
import Testing
@testable import CharacterCore

@Suite struct CharacterOptionVisualsTests {
    private typealias CO = CharacterOptions

    @Test func everyOptionHasAGoodHint() {
        for cat in CO.all {
            for opt in cat.options {
                let h = CO.hint(category: cat.id, option: opt.id)
                #expect(!h.isEmpty && h.count <= 60 && !h.hasSuffix("."), "\(cat.id)/\(opt.id): \(h)")
                #expect(AgeGuard.youthTerm(in: h) == nil, "\(cat.id)/\(opt.id)")
            }
        }
        #expect(CO.hint(category: "nope", option: "x") == "")
    }

    @Test func visualsMatchTheirCategory() {
        let hex = try! NSRegularExpression(pattern: "^#[0-9A-Fa-f]{6}$")
        for cat in CO.all {
            for opt in cat.options {
                let v = CO.visual(category: cat.id, option: opt.id)
                let id = "\(cat.id)/\(opt.id)"
                if CO.swatchCategories.contains(cat.id) {
                    guard case .swatch(let h) = v else { Issue.record("\(id) not a swatch"); continue }
                    #expect(hex.firstMatch(in: h, range: NSRange(h.startIndex..., in: h)) != nil, "\(id) \(h)")
                } else if CO.symbolCategories.contains(cat.id) {
                    guard case .symbol(let name) = v else { Issue.record("\(id) not a symbol"); continue }
                    #expect(NSImage(systemSymbolName: name, accessibilityDescription: nil) != nil, "\(id) \(name)")
                } else if CO.photoCategories.contains(cat.id) {
                    #expect(v == .photo, "\(id)")
                } else {
                    #expect(v == .none, "\(id)")
                }
            }
        }
        for cat in ["height", "traits", "voiceTone"] {
            for opt in CO.category(cat)!.options { #expect(CO.visual(category: cat, option: opt.id) == .none) }
        }
        #expect(CO.visual(category: "nope", option: "x") == .none)
    }

    @Test func examplesCoverEveryPhotoOption() {
        let ex = CO.examples
        var expected: [String] = []
        for cat in CO.all where CO.photoCategories.contains(cat.id) {
            for opt in cat.options {
                for g in CO.exampleGenders(category: cat.id, option: opt.id) { expected.append("\(cat.id)/\(opt.id)/\(g)") }
            }
        }
        let got = ex.map { "\($0.category)/\($0.option)/\($0.gender)" }
        #expect(got == expected)
        #expect(Set(got).count == got.count)
        let sum = CO.all.filter { CO.photoCategories.contains($0.id) }
            .reduce(0) { n, cat in n + cat.options.reduce(0) { $0 + CO.exampleGenders(category: cat.id, option: $1.id).count } }
        #expect(ex.count == sum)
        #expect(CO.exampleGenders(category: "bust", option: "small") == ["woman"])
        #expect(CO.exampleGenders(category: "butt", option: "round") == ["woman", "man"])
    }

    @Test func examplePromptsAreSafe() {
        for e in CO.examples {
            let phrase = CO.category(e.category)!.options.first { $0.id == e.option }!.phrase
            let id = "\(e.category)/\(e.option)/\(e.gender)"
            for model in [PromptModel.minimaxImage, .sdxl, .chroma] {
                let p = CO.examplePrompt(category: e.category, option: e.option, gender: e.gender, model: model)
                // Tag prompts drop the phrase's leading article ("a sleek bob" → "sleek bob").
                let bare = phrase.replacingOccurrences(of: #"^(a|an) "#, with: "", options: .regularExpression)
                #expect(p.contains("adult") && p.lowercased().contains(bare.lowercased()), "\(id) \(model)")
                #expect(AgeGuard.youthTerm(in: p) == nil, "\(id) \(model)")
                #expect(!p.lowercased().contains("glamour"), "\(id) \(model)")
            }
            let spec = CO.exampleSpec(category: e.category, option: e.option, gender: e.gender)!
            #expect(!spec.outfit.isEmpty, "\(id)")
            let neg = CO.exampleNegative(category: e.category, option: e.option, gender: e.gender)
            if ["body", "bust", "butt", "figure"].contains(e.category) { #expect(neg.contains("nude"), "\(id)") }
            if e.category == "clothes" { #expect(!neg.contains("nude") && neg.contains("monochrome"), "\(id)") }
            // MiniMax takes no negative: it is told in words to draw no text.
            #expect(CO.examplePrompt(category: e.category, option: e.option, gender: e.gender)
                .hasSuffix("The image contains no text, letters or numbers.") || !spec.without.isEmpty, "\(id)")
        }
        // The weak options get a stronger description, and their opposite in the negative.
        let small = CO.examplePrompt(category: "bust", option: "small", gender: "woman")
        #expect(small.hasPrefix("A studio photo, full body, of a 21-year-old adult woman"))
        #expect(small.contains("She has small breasts, a flat chest") && small.contains("platinum blonde hair in a sleek bob"))
        #expect(small.contains("natural light makeup"))
        #expect(!CO.examplePrompt(category: "extras", option: "bold-makeup", gender: "woman").contains("light makeup"))
        #expect(CO.examplePrompt(category: "body", option: "average", gender: "man").contains("He wears only fitted black boxer briefs, shirtless."))
        #expect(CO.exampleNegative(category: "bust", option: "small", gender: "woman").contains("large breasts"))
        #expect(CO.examplePrompt(category: "body", option: "curvy", gender: "man").contains("stocky"))
        #expect(CO.examplePrompt(category: "body", option: "curvy", gender: "woman").contains("full rounded hips"))
        #expect(CO.examplePrompt(category: "extras", option: "tattoos", gender: "man").contains("waist-up"))
        // The man's own wording wins for options that came out alike (MiniMax has no negative prompt).
        #expect(CO.examplePrompt(category: "body", option: "average", gender: "man").contains("no visible abs at all"))
        #expect(CO.examplePrompt(category: "body", option: "average", gender: "man").hasSuffix("No visible abs and no muscle definition."))
        #expect(CO.examplePrompt(category: "body", option: "slim", gender: "man").contains("skinny"))
        #expect(CO.examplePrompt(category: "clothes", option: "streetwear", gender: "man").contains("hoodie"))
        #expect(CO.examplePrompt(category: "clothes", option: "boho", gender: "man").contains("linen shirt"))
        #expect(CO.examplePrompt(category: "clothes", option: "boho", gender: "woman").contains("linen") == false)
        // SDXL reads tags: the option weighted, the subject first.
        let sdxl = CO.examplePrompt(category: "body", option: "average", gender: "man", model: .sdxl)
        #expect(sdxl.hasPrefix("studio photo, a 30-year-old adult man, full body") && sdxl.contains("(an average build:1.3)") == false)
        #expect(sdxl.contains("(average build:1.3)"))
        #expect(CO.examplePrompt(category: "nope", option: "x", gender: "woman") == "")
        #expect(CO.examplePrompt(category: "bust", option: "nope", gender: "woman") == "")
    }

    @Test func lookPartsForCharacterPhotos() {
        let traits = ["hairColor": ["red"], "body": ["average"], "figure": ["toned-abs"], "clothes": ["boho"]]
        let parts = CO.lookParts(traits, gender: "woman")
        #expect(parts.details.map(\.tag) == ["red hair"])
        #expect(parts.body.first?.text.contains("soft untoned stomach") == true && parts.body.first?.weight == 1.2)
        #expect(parts.outfit.first?.tag == "boho style clothes" && parts.outfit.first?.text.contains("maxi dress") == true)
        // Toned abs asks for abs: the average body's "abs" and "six-pack" opposites are dropped, the rest stay.
        #expect(!parts.avoid.contains("abs") && !parts.avoid.contains("six-pack") && parts.avoid.contains("fitness model"))
        // An average body alone keeps "abs" out, even though its own words say "no visible abs".
        let man = CO.lookParts(["body": ["average"]], gender: "man")
        #expect(man.avoid.contains("abs") && man.without == ["no visible abs and no muscle definition"])
        // Two clothing styles: each by its tag (two full outfits would fight); none when the photo names an outfit.
        let two = CO.lookParts(["clothes": ["casual", "streetwear"]], gender: "man")
        #expect(two.outfit.map(\.text) == ["casual clothes", "streetwear"])
        #expect(CO.lookParts(traits, gender: "woman", withClothes: false).outfit.isEmpty)
    }

    @Test func seedsAndFilesAreStable() {
        let a = CO.exampleSeed(category: "hairStyle", gender: "woman")
        #expect(a == CO.exampleSeed(category: "hairStyle", gender: "woman"))
        #expect((0...0x7FFF_FFFF).contains(a))
        #expect(a != CO.exampleSeed(category: "hairStyle", gender: "man"))
        #expect(CO.exampleFile(category: "hairStyle", option: "long-wavy", gender: "woman") == "hairStyle-long-wavy-woman.jpg")
    }
}
