import AppKit
import Foundation
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
            let p = CO.examplePrompt(category: e.category, option: e.option, gender: e.gender)
            let phrase = CO.category(e.category)!.options.first { $0.id == e.option }!.phrase
            let id = "\(e.category)/\(e.option)/\(e.gender)"
            #expect(p.contains("adult") && p.contains(phrase) && p.hasSuffix("soft light."), "\(id)")
            #expect(AgeGuard.youthTerm(in: p) == nil, "\(id)")
            let neg = CO.exampleNegative(category: e.category, option: e.option, gender: e.gender)
            if ["body", "bust", "butt", "figure"].contains(e.category) {
                #expect(p.contains("wearing") && neg.contains("nude"), "\(id)")
            }
            if e.category == "clothes" { #expect(!neg.contains("nude") && neg.contains("monochrome"), "\(id)") }
        }
        // The weak options get a stronger description up front and their opposite in the negative.
        let small = CO.examplePrompt(category: "bust", option: "small", gender: "woman")
        #expect(small.hasPrefix("Glamour photo of a 21-year-old adult woman with small breasts, a flat chest"))
        #expect(small.contains("fair skin") && small.contains("platinum blonde hair in a sleek bob") && small.contains("light makeup"))
        #expect(CO.examplePrompt(category: "extras", option: "bold-makeup", gender: "woman").contains("light makeup") == false)
        #expect(CO.examplePrompt(category: "body", option: "average", gender: "man").hasPrefix("Glamour photo of a 30-year-old adult man"))
        #expect(CO.examplePrompt(category: "body", option: "average", gender: "man").contains("shirtless"))
        #expect(CO.exampleNegative(category: "bust", option: "small", gender: "woman").contains("large breasts"))
        #expect(CO.examplePrompt(category: "body", option: "curvy", gender: "man").contains("stocky"))
        #expect(CO.examplePrompt(category: "body", option: "curvy", gender: "woman").contains("full rounded hips"))
        #expect(CO.examplePrompt(category: "extras", option: "tattoos", gender: "man").contains("waist-up"))
        #expect(CO.examplePrompt(category: "nope", option: "x", gender: "woman") == "")
        #expect(CO.examplePrompt(category: "bust", option: "nope", gender: "woman") == "")
    }

    @Test func seedsAndFilesAreStable() {
        let a = CO.exampleSeed(category: "hairStyle", gender: "woman")
        #expect(a == CO.exampleSeed(category: "hairStyle", gender: "woman"))
        #expect((0...0x7FFF_FFFF).contains(a))
        #expect(a != CO.exampleSeed(category: "hairStyle", gender: "man"))
        #expect(CO.exampleFile(category: "hairStyle", option: "long-wavy", gender: "woman") == "hairStyle-long-wavy-woman.jpg")
    }
}
