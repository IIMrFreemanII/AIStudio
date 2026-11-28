import Foundation
import LLMClient
import MovieCore
import Testing
@testable import CharacterCore

@Suite struct PromptEngineTests {
    @Test func bundledRulesMatchTheDoc() throws {
        let doc = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("docs/prompting.md")
        let text = try String(contentsOf: doc, encoding: .utf8)
        #expect(PromptEngine.rulesText == text, "copy docs/prompting.md to Sources/CharacterCore/Resources/prompting.md")
    }

    @Test func everyModelHasItsRules() {
        for m in PromptModel.allCases {
            let r = PromptEngine.rules(for: m)
            #expect(r.contains(PromptEngine.heading(m).trimmingCharacters(in: .newlines)), "\(m) has no section")
            #expect(r.contains("| Model |"), "\(m) has no quick-reference row")
            #expect(r.contains("## Rules for the app's prompt writers"))
        }
        // Qwen-Image's section stops before Qwen-Image-Edit's.
        let q = PromptEngine.rules(for: .qwenImage)
        #expect(!q.contains("### Qwen-Image-Edit"))
        #expect(PromptEngine.quickReference(.qwenImage, in: PromptEngine.rulesText)?.contains("Qwen-Image-Edit") == false)
        #expect(PromptEngine.basics().hasPrefix("## The basics"))
    }

    @Test func sectionsEndAtTheNextHeadingOfTheirLevel() {
        let doc = "# T\n## A\none\n### A1\ntwo\n## B\nthree\n"
        #expect(PromptEngine.section("## A", in: doc) == "## A\none\n### A1\ntwo")
        #expect(PromptEngine.section("### A1", in: doc) == "### A1\ntwo")
        #expect(PromptEngine.section("## C", in: doc) == nil)
    }

    @Test func warnings() {
        let long = Array(repeating: "red", count: 90).joined(separator: ", ")
        #expect(PromptEngine.warnings(prompt: long, negative: "", model: .sdxl).contains { $0.contains("first 75") })
        #expect(PromptEngine.warnings(prompt: "a cat", negative: "", model: .sdxl).isEmpty)
        #expect(PromptEngine.warnings(prompt: "a cat", negative: "dog", model: .flux).contains { $0.contains("no negative") })
        #expect(PromptEngine.warnings(prompt: "a cat", negative: "dog", model: .chroma, fast: true).contains { $0.contains("fast mode") })
        #expect(PromptEngine.warnings(prompt: "a (red:1.3) cat", negative: "", model: .chroma).contains { $0.contains("weights") })
        #expect(PromptEngine.warnings(prompt: "(red:1.3) cat", negative: "", model: .sdxl).isEmpty)
        #expect(PromptEngine.warnings(prompt: "a cat", negative: "", model: .sdxl, triggerWords: ["ohwx"]).contains { $0.contains("ohwx") })
        #expect(PromptEngine.clipTokens("a red cat, sits") == 5)
        #expect(PromptEngine.clipTokens("standing") == 2)
    }

    @Test func specGetsTheAdultGuard() throws {
        let s = try PromptEngine.spec(from: .object(["subject": .string("a 30-year-old woman"), "pronoun": .string("she"),
                                                     "shot": .string("full body")]))
        #expect(s.subject == "a 30-year-old adult woman")
        #expect(s.avoid.contains("underage"))
        let t = try PromptEngine.spec(from: .object(["subject": .string("a woman"), "pronoun": .string("she")]))
        #expect(t.subject == "an adult woman")
        let thing = try PromptEngine.spec(from: .object(["subject": .string("a lighthouse")]))
        #expect(thing.subject == "a lighthouse")
        #expect(throws: PromptEngineError.self) {
            try PromptEngine.spec(from: .object(["subject": .string("a teen girl"), "pronoun": .string("she")]))
        }
        #expect(throws: PromptEngineError.self) {
            try PromptEngine.spec(from: .object(["subject": .string("a 16-year-old"), "pronoun": .string("she")]))
        }
    }

    @Test func rawNegativesGetTheGuard() {
        #expect(PromptEngine.guardedNegative("blurry", model: .sdxl) == "blurry, " + AgeGuard.negative)
        #expect(PromptEngine.guardedNegative("", model: .sdxl) == AgeGuard.negative)
        #expect(PromptEngine.guardedNegative("blurry", model: .flux) == "")
        #expect(PromptEngine.refusal(["a schoolgirl"]) != nil)
        #expect(PromptEngine.refusal(["a 30-year-old adult woman"]) == nil)
    }

    @Test func systemPromptCarriesTheRules() {
        let s = PromptEngine.systemPrompt(video: false, fixed: nil, character: nil)
        #expect(s.contains("draft_models") && s.contains("18") && s.contains("## The basics"))
        let fixed = PromptEngine.systemPrompt(video: true, fixed: .sdxl, character: "Mia")
        #expect(fixed.contains("fixed: sdxl") && fixed.contains("Mia") && fixed.contains("first frame"))
    }

    @Test func resultPicksTheBestTry() {
        let t = EngineTry(n: 1, model: .sdxl, prompt: "p", negative: "", seed: 1, file: "/tmp/a.png")
        let r = EngineResult(best: t, notes: "", checklist: ["x"], tries: [t], verdicts: [EngineVerdict(attempt: 1, pass: true)],
                             gpuUSD: 0.01, llmUSD: 0.02)
        let data = try! JSONEncoder().encode(r)
        #expect(try! JSONDecoder().decode(EngineResult.self, from: data) == r)
    }
}

@Suite struct PromptEngineLLMTests {
    let both = ["modal/qwen3-vl-8b-instruct", "modal/qwen3.8-27b-abliterated", "modal/qwen3.8-27b-abliterated-nvfp4"]

    @Test func theNVFP4CopyIsTheDefault() {
        #expect(PromptEngineLLM.pick(installed: both, saved: nil, upgraded: false) ==
                ("modal/qwen3.8-27b-abliterated-nvfp4", nil))
        #expect(PromptEngineLLM.pick(installed: ["modal/qwen3-vl-8b-instruct", "modal/qwen3.8-27b-abliterated"],
                                     saved: nil, upgraded: false) == ("modal/qwen3.8-27b-abliterated", nil))
        #expect(PromptEngineLLM.pick(installed: ["modal/qwen3-vl-8b-instruct"], saved: nil, upgraded: false) ==
                ("modal/qwen3-vl-8b-instruct", nil))
    }

    @Test func aSavedBF16PickMovesToTheCopyOnce() {
        #expect(PromptEngineLLM.pick(installed: both, saved: "modal/qwen3.8-27b-abliterated", upgraded: false) ==
                ("modal/qwen3.8-27b-abliterated-nvfp4", "modal/qwen3.8-27b-abliterated-nvfp4"))
        // picked again after the move: it stays
        #expect(PromptEngineLLM.pick(installed: both, saved: "modal/qwen3.8-27b-abliterated", upgraded: true) ==
                ("modal/qwen3.8-27b-abliterated", nil))
        // no copy on that server: unchanged
        #expect(PromptEngineLLM.pick(installed: ["modal/qwen3.8-27b-abliterated"], saved: "modal/qwen3.8-27b-abliterated",
                                     upgraded: false) == ("modal/qwen3.8-27b-abliterated", nil))
        #expect(PromptEngineLLM.pick(installed: both, saved: "modal/qwen3-vl-8b-instruct", upgraded: false) ==
                ("modal/qwen3-vl-8b-instruct", nil))
    }
}
