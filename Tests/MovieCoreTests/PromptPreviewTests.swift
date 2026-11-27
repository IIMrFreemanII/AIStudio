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

    @Test func loRAsChainIntoTheNode() throws {
        let t = try #require(PromptPreview.template(.sdxlIllustrious, negative: true, loras: 2))
        #expect(t.graph["lora1"]?.type == "RemoteLoRA" && t.graph["lora1"]?.inputs["loras"] == nil)
        #expect(t.graph["lora2"]?.inputs["loras"] == .link(node: "lora1", output: "loras"))
        #expect(t.graph["image"]?.inputs["loras"] == .link(node: "lora2", output: "loras"))
        #expect(t.slots["lora2_strength"] == .init(node: "lora2", input: "strength"))
        // At most four, and none on MiniMax H3 Image, which takes no LoRAs.
        #expect(PromptPreview.template(.chroma, negative: false, loras: 9)?.graph.nodes.filter { $0.type == "RemoteLoRA" }.count == 4)
        let h3 = try #require(PromptPreview.template(.minimaxImage, negative: false, loras: 2))
        #expect(h3.graph["lora1"] == nil && h3.graph["image"]?.inputs["loras"] == nil)
        #expect(PromptPreview.template(.flux, negative: false)?.graph["lora1"] == nil)
        let v = PromptPreview.loraValues([("detail", 0.6), ("style", 1.0)], first: 2)
        #expect(v["lora2"] == .string("detail") && v["lora2_strength"] == .double(0.6) && v["lora3"] == .string("style"))
    }

    @Test func sizesAndFastModes() throws {
        #expect(PromptPreview.size(.sdxl, aspect: .portrait) == (832, 1216))
        #expect(PromptPreview.size(.sdxl, aspect: .landscape) == (1216, 832))
        #expect(PromptPreview.size(.chroma, aspect: .square) == (1024, 1024))
        #expect(PromptModel(rawValue: "sd15") == nil)  // removed on 2026-11-27
        // A negative needs guidance: the fast modes that skip it are turned off.
        let chroma = try #require(PromptPreview.template(.chroma, negative: true))
        #expect(chroma.graph["image"]?.inputs["fast"] == .value(.bool(false)))
        let plain = try #require(PromptPreview.template(.chroma, negative: false))
        #expect(plain.graph["image"]?.inputs["fast"] == .value(.bool(true)), "no negative: the node's fast default")
        // MiniMax and FLUX read no negative.
        let h3 = try #require(PromptPreview.template(.minimaxImage, negative: true))
        #expect(h3.slots["negative"] == nil && h3.graph["image"]?.type == "RemoteH3Image")
        #expect(PromptPreview.task(.sdxlPony) == "sdxl" && PromptPreview.task(.wan) == nil)
    }

    @Test func theFastSwitchKeepsTheFastModeWithANegative() throws {
        for m in PromptModel.allCases where PromptPreview.nodeType(m) != nil {
            let fast = try #require(PromptPreview.template(m, negative: true, fast: true))
            let full = try #require(PromptPreview.template(m, negative: true, fast: false))
            if PromptPreview.fastModels.contains(m) {
                #expect(fast.graph["image"]?.inputs["fast"] == .value(.bool(true)), "\(m) fast")
                #expect(full.graph["image"]?.inputs["fast"] == .value(.bool(false)), "\(m) full")
                #expect(fast.slots["negative"] != nil, "\(m) still sends its negative")
                #expect(PromptPreview.runsFast(m, negative: false, fast: false) == true)
            } else {
                #expect(fast.graph["image"]?.inputs["fast"] == nil && full == fast, "\(m) has no fast mode")
                #expect(PromptPreview.runsFast(m, negative: true, fast: true) == nil)
            }
        }
        #expect(PromptPreview.fastModels == [.chroma, .sd35, .qwenImage])
    }
}
