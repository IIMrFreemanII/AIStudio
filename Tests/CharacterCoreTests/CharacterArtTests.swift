import Foundation
import GraphRuntime
import MovieCore
import Testing
@testable import CharacterCore

private func hero(style: ArtModel.Style = .photoreal) -> Character {
    var c = Character(name: "Ava", age: 28, gender: "woman", look: "red hair, green eyes")
    c.artModel = ArtModel(style: style)
    return c
}

private let cafe = Scenario(id: "x", title: "X", place: "a small cafe", outfit: "a blue dress", time: "sunny morning")

@Suite struct CharacterArtTests {
    @Test func photorealPrompt() {
        let p = CharacterArt.photoPrompt(hero(), request: "holding a latte", scene: cafe)
        for s in ["adult", "28", "red hair, green eyes", "holding a latte", "a small cafe", "a blue dress", "sunny morning"] {
            #expect(p.contains(s), "\(s) in \(p)")
        }
        #expect(!p.contains("Ava") && !p.contains("  "))
        #expect(p.hasPrefix("Photo of a 28-year-old adult woman"))
    }

    @Test func photorealWithoutScene() {
        let p = CharacterArt.photoPrompt(hero(), request: "smiling", scene: nil)
        #expect(p == "Photo of a 28-year-old adult woman, red hair, green eyes. smiling. Candid smartphone photo, natural light, sharp focus, detailed skin.")
    }

    @Test func captionDropsALeadingPronoun() {
        #expect(CharacterArt.caption("You on the sofa, smiling") == "On the sofa, smiling")
        #expect(CharacterArt.caption("I'm at the beach") == "At the beach")
        #expect(CharacterArt.caption("your favourite cafe at noon") == "your favourite cafe at noon")
        #expect(CharacterArt.caption("smiling") == "smiling")
        let p = CharacterArt.photoPrompt(hero(), request: "You holding a latte", scene: nil)
        #expect(p.contains(" Holding a latte.") && !p.contains("You"))
    }

    @Test func animePromptIsTags() {
        let p = CharacterArt.photoPrompt(hero(style: .anime), request: "waving", scene: cafe)
        #expect(p.hasPrefix("masterpiece, best quality, anime illustration, adult woman, 28 years old"))
        for s in ["red hair, green eyes", "waving", "a small cafe", "a blue dress", "sunny morning"] { #expect(p.contains(s)) }
        #expect(!p.contains("Ava") && !p.contains(",,") && !p.contains("  "))
    }

    @Test func portraitHasNoScene() {
        let p = CharacterArt.portraitPrompt(hero())
        #expect(p.contains("head-and-shoulders portrait") && p.contains("adult") && p.contains("28"))
        #expect(!p.contains("cafe") && !p.contains("wearing") && !p.contains("Ava"))
        #expect(CharacterArt.portraitPrompt(hero(style: .anime)).contains("plain background"))
    }

    @Test func negativeHasAgeGuard() {
        #expect(CharacterArt.negative(hero()).hasSuffix(AgeGuard.negative))
        #expect(CharacterArt.negative(hero()).hasPrefix("cartoon"))
        #expect(CharacterArt.negative(hero(style: .anime)).hasPrefix("lowres"))
    }

    @Test func checkRequests() throws {
        try CharacterArt.check("on the beach in a red dress")
        #expect(throws: CharacterArtError.youthTerm("teen")) { try CharacterArt.check("as a teen") }
        #expect(CharacterArtError.youthTerm("teen").description.contains("under 18"))
    }

    @Test func templateBindsWithFace() throws {
        var c = hero()
        c.loras = [CharacterLoRA(lora: "a", strength: 0.5), CharacterLoRA(lora: "b")]
        let r = CharacterArt.render(c, prompt: "p", seed: 7, server: "s", face: URL(fileURLWithPath: "/tmp/f.png"))
        #expect(r.unusedPictures.isEmpty)
        let t = CharacterArt.template(loras: 2)
        let g = try t.bind(r.values, optional: Set(r.values.keys), remove: r.unusedPictures)
        #expect(g["face"]?.type == "LoadImage" && g["face"]?.inputs["path"] == .value(.string("/tmp/f.png")))
        #expect(g["image"]?.inputs["face"] == .link(node: "face", output: "image"))
        #expect(g["lora2"]?.inputs["loras"] == .link(node: "lora1", output: "loras"))
        #expect(g["lora1"]?.inputs["loras"] == nil)
        #expect(g["image"]?.inputs["loras"] == .link(node: "lora2", output: "loras"))
        #expect(g["lora1"]?.inputs["lora"] == .value(.string("a")) && g["lora1"]?.inputs["strength"] == .value(.double(0.5)))
        #expect(g["lora2"]?.inputs["lora"] == .value(.string("b")) && g["lora2"]?.inputs["strength"] == .value(.double(0.8)))
        #expect(g["image"]?.inputs["model"] == .value(.string("")))
        #expect(g["image"]?.inputs["seed"] == .value(.int(7)))
        #expect(g["out"]?.inputs["image"] == .link(node: "image", output: "image"))
    }

    @Test func templateBindsWithoutFace() throws {
        let r = CharacterArt.render(hero(), prompt: "p", seed: 1, server: "s", face: nil)
        #expect(r.unusedPictures == ["face"] && r.values["face"] == nil)
        let g = try CharacterArt.template(loras: 0).bind(r.values, optional: Set(r.values.keys), remove: r.unusedPictures)
        #expect(g["face"] == nil && g["image"]?.inputs["face"] == nil && g["image"]?.inputs["loras"] == nil)
        #expect(CharacterArt.template(loras: 9).graph.nodes.filter { $0.type == "RemoteLoRA" }.count == CharacterArt.maxLoRAs)
    }

    @Test func loraDecodesOldLinkKey() throws {
        let old = try JSONDecoder().decode(CharacterLoRA.self, from: Data(#"{"link": "x", "strength": 0.5}"#.utf8))
        #expect(old == CharacterLoRA(lora: "x", strength: 0.5))
        let new = try JSONDecoder().decode(CharacterLoRA.self, from: Data(#"{"lora": "y"}"#.utf8))
        #expect(new == CharacterLoRA(lora: "y"))
        let out = String(decoding: try JSONEncoder().encode(old), as: UTF8.self)
        #expect(out.contains("\"lora\"") && !out.contains("link"))
    }
}
