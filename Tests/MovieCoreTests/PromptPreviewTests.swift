import GraphRuntime
import Testing
@testable import MovieCore

@Suite struct PromptPreviewTests {
    @Test func everyImageModelBindsAndVideoModelsHaveNone() throws {
        for m in PromptModel.allCases {
            guard let t = PromptPreview.template(m, negative: true) else {
                #expect(m.isVideo, "\(m) has no preview")
                continue
            }
            #expect(!m.isVideo)
            var w = PromptSpec()
            w.subject = "a red fox"
            w.avoid = ["blurry"]
            let written = PromptWriter.write(w, for: m)
            let values = PromptPreview.values(written, model: m, seed: 7, aspect: .portrait, server: "s", fineTune: "ft")
            let g = try t.bind(values)
            let node = try #require(g["image"])
            #expect(node.type == PromptPreview.nodeType(m))
            #expect(node.inputs["prompt"] == .value(.string(written.prompt)))
            #expect(node.inputs["seed"] == .value(.int(7)))
            #expect((node.inputs["negative"] != nil) == m.takesNegative, "\(m) negative slot")
            #expect((node.inputs["model"] != nil) == PromptPreview.takesFineTune(m), "\(m) fine-tune slot")
            #expect(g["out"]?.type == "PreviewImage")
        }
    }

    @Test func sizesAndFastModes() throws {
        #expect(PromptPreview.size(.sdxl, aspect: .portrait) == (832, 1216))
        #expect(PromptPreview.size(.sdxl, aspect: .landscape) == (1216, 832))
        #expect(PromptPreview.size(.sd15, aspect: .square) == (512, 512))
        // A negative needs guidance: the fast modes that skip it are turned off.
        let chroma = try #require(PromptPreview.template(.chroma, negative: true))
        #expect(chroma.graph["image"]?.inputs["fast"] == .value(.bool(false)))
        let plain = try #require(PromptPreview.template(.chroma, negative: false))
        #expect(plain.graph["image"]?.inputs["fast"] == nil)
        // MiniMax and FLUX read no negative.
        let h3 = try #require(PromptPreview.template(.minimaxImage, negative: true))
        #expect(h3.slots["negative"] == nil && h3.graph["image"]?.type == "RemoteH3Image")
        #expect(PromptPreview.task(.sdxlPony) == "sdxl" && PromptPreview.task(.wan) == nil)
    }
}
