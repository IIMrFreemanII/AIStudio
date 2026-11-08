import MLX
import Testing
@testable import MLXDiffusionCore
@testable import QwenImage21

@Test func assistantPicksSystemPromptByModeAndImage() {
    let plain = PromptAssistant.messages(mode: .enhance, prompt: "  a cat ", hasImage: false)
    #expect(plain.system == PromptAssistant.enhanceSystem)
    #expect(plain.user == "a cat")
    #expect(PromptAssistant.messages(mode: .enhance, prompt: "make it winter", hasImage: true).system
        == PromptAssistant.editSystem)
    let describe = PromptAssistant.messages(mode: .describe, prompt: "", hasImage: true)
    #expect(describe.system == PromptAssistant.describeSystem)
    #expect(describe.user == "Describe this image as a prompt.")
    #expect(PromptAssistant.messages(mode: .describe, prompt: "moody", hasImage: true).user.hasSuffix("moody"))
    // Overrides win unless blank.
    #expect(PromptAssistant.messages(mode: .enhance, prompt: "x", hasImage: false,
                                     overrides: ["enhance": "Be brief."]).system == "Be brief.")
    #expect(PromptAssistant.messages(mode: .enhance, prompt: "x", hasImage: false,
                                     overrides: ["enhance": "  "]).system == PromptAssistant.enhanceSystem)
    #expect(PromptAssistant.messages(mode: .custom, prompt: "x", hasImage: false, instruction: "Translate.").system
        == "Translate.")
}

@Test func assistantCleansLabelsAndQuotes() {
    #expect(PromptAssistant.clean("  Prompt: \"A red fox in snow.\"\n") == "A red fox in snow.")
    #expect(PromptAssistant.clean("“A lighthouse”") == "A lighthouse")
    #expect(PromptAssistant.clean("It's a 'quoted' word") == "It's a 'quoted' word")
}

@Test func imagePadsExpandWithMRoPEPositions() {
    let pad = QwenImage21Pipeline.imagePadToken
    let vision = Qwen3VisionTower.Output(embeddings: MLXArray.zeros([6, 4]), deepstack: [MLXArray.zeros([6, 4])],
                                         gridH: 2, gridW: 3)
    let (ids, visual) = QwenImage21Pipeline.expandImagePads([10, pad, 11], visions: [vision])
    #expect(ids == [10] + Array(repeating: pad, count: 6) + [11])
    #expect(visual.runs.count == 1 && visual.runs[0].start == 1 && visual.runs[0].count == 6)
    let p = visual.positions
    #expect(p[0] == (0, 0, 0))
    // t fixed at 1 over the image; h, w follow the 2×3 grid.
    #expect(p[1] == (1, 1, 1))
    #expect(p[3] == (1, 1, 3))
    #expect(p[4] == (1, 2, 1))
    #expect(p[6] == (1, 2, 3))
    // Text after the image continues from 1 + max(grid): position 4 on every axis.
    #expect(p[7] == (4, 4, 4))
}
