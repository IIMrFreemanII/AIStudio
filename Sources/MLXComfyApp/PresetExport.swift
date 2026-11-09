import Foundation
import GraphRuntime
import MLXComfyLibrary
import MLXDiffusionCore

/// `--export-presets <dir>` writes the built-in presets (`Sources/MLXComfyLibrary/Presets`), built from the graph
/// templates and the same Add to Workflow chains the app uses, so they never drift apart. Needs the models
/// installed, since image-to-image wiring depends on the model's family.
@MainActor
enum PresetExport {
    struct Spec {
        let id: String, name: String, category: Preset.Category, models: [String], modelName: String
        let summary: String, symbol: String
        let build: (AppModel) -> Graph?
    }

    static let placeholderImage = URL(fileURLWithPath: "/placeholder.png")

    static func run(_ model: AppModel, to dir: URL) -> Bool {
        func say(_ s: String) { print("[presets] \(s)"); fflush(stdout) }
        model.store?.refresh()
        let out = model.outputDirectory
        func path(_ folder: String) -> String { model.resolveModelFolder(folder) ?? folder }
        let qwen = path("qwen-turbo-mlx"), flux = path("flux-1-dev")

        /// Builds in a scratch tab so the chain helpers (which edit the active graph) can be used as they are.
        func edited(_ start: Graph, _ edit: (AppModel) -> Void) -> Graph {
            let doc = model.openTab(start, name: "export")
            edit(model)
            let g = model.graph
            model.closeTab(doc.id)
            return g
        }
        func clearPaths(_ g: Graph) -> Graph {
            var g = g
            for i in g.nodes.indices where ["LoadImage", "LoadMask"].contains(g.nodes[i].type) {
                g.nodes[i].inputs["path"] = .value(.string(""))
            }
            return g
        }
        func setPrompt(_ m: AppModel, _ text: String) {
            if let te = m.firstNode(ofType: "TextEncode") { m.setLiteral(.string(text), node: te.id, input: "prompt", undoManager: nil) }
        }
        let image = { (p: String) in GraphTemplates.textToImage(modelPath: p, outputDirectory: out) }

        let specs: [Spec] = [
            Spec(id: "qwen-txt2img", name: "Text to Image", category: .image, models: ["qwen-image-2.1-turbo"],
                 modelName: "Qwen-Image 2.1", summary: "Fast 8-step pictures; good at writing text into the image.",
                 symbol: "photo", build: { _ in image(qwen) }),
            Spec(id: "qwen-edit", name: "Edit an Image", category: .image, models: ["qwen-image-2.1-turbo"],
                 modelName: "Qwen-Image 2.1", summary: "Choose a picture and describe the change; the rest is kept.",
                 symbol: "wand.and.rays", build: { _ in
                     clearPaths(edited(image(qwen)) { m in
                         m.setReferenceImage(placeholderImage, undoManager: nil)
                         setPrompt(m, "Make it winter: snow on the ground and falling flakes")
                     })
                 }),
            Spec(id: "qwen-inpaint", name: "Inpaint", category: .image, models: ["qwen-image-2.1-turbo"],
                 modelName: "Qwen-Image 2.1", summary: "Repaint only the masked part of a picture (Paint Mask on the image node).",
                 symbol: "paintbrush", build: { _ in
                     clearPaths(edited(image(qwen)) { m in
                         m.setReferenceImage(placeholderImage, undoManager: nil)
                         m.setMask(placeholderImage, undoManager: nil)
                         m.groupAddedChain(.inpaint, undoManager: nil)
                         setPrompt(m, "A small wooden cabin with a lit window")
                     })
                 }),
            Spec(id: "qwen-outpaint", name: "Outpaint", category: .image, models: ["qwen-image-2.1-turbo"],
                 modelName: "Qwen-Image 2.1", summary: "Extend a picture past its borders; set the margins on Pad Image.",
                 symbol: "arrow.up.left.and.arrow.down.right", build: { _ in
                     clearPaths(edited(image(qwen)) { m in
                         m.setReferenceImage(placeholderImage, undoManager: nil)
                         m.setOutpaint(left: 256, top: 0, right: 256, bottom: 0, undoManager: nil)
                         m.groupAddedChain(.outpaint, undoManager: nil)
                         setPrompt(m, "Continue the scene naturally beyond the edges")
                     })
                 }),
            Spec(id: "qwen-hires", name: "Text to Image + Hi-res Fix", category: .image, models: ["qwen-image-2.1-turbo"],
                 modelName: "Qwen-Image 2.1", summary: "Generate, upscale 1.5× and refine the detail in a second pass.",
                 symbol: "plus.magnifyingglass", build: { _ in
                     edited(image(qwen)) { m in
                         m.setHiRes(true, undoManager: nil)
                         m.groupAddedChain(.hires, undoManager: nil)
                     }
                 }),
            Spec(id: "flux-txt2img", name: "Flux Text to Image", category: .image, models: ["flux-1-dev"],
                 modelName: "FLUX.1 [dev]", summary: "28 steps at guidance 3.5; add LoRAs from Add to Workflow.",
                 symbol: "photo.artframe", build: { _ in image(flux) }),
            Spec(id: "flux-img2img", name: "Flux Image to Image", category: .image, models: ["flux-1-dev"],
                 modelName: "FLUX.1 [dev]", summary: "Redraw a picture from the prompt; the Sampler's denoise sets how much.",
                 symbol: "photo.on.rectangle", build: { _ in
                     clearPaths(edited(image(flux)) { m in
                         m.setReferenceImage(placeholderImage, undoManager: nil)
                         setPrompt(m, "An oil painting of the same scene, thick brush strokes")
                     })
                 }),
            Spec(id: "flux-inpaint", name: "Flux Inpaint", category: .image, models: ["flux-1-dev"],
                 modelName: "FLUX.1 [dev]", summary: "Repaint only the masked part of a picture with Flux.",
                 symbol: "paintbrush.pointed", build: { _ in
                     clearPaths(edited(image(flux)) { m in
                         m.setReferenceImage(placeholderImage, undoManager: nil)
                         m.setMask(placeholderImage, undoManager: nil)
                         m.groupAddedChain(.inpaint, undoManager: nil)
                         setPrompt(m, "A small wooden cabin with a lit window")
                     })
                 }),
            Spec(id: "ltx-t2v", name: "Text to Video", category: .video, models: ["ltx-2.3"],
                 modelName: "LTX-2.3", summary: "5 s at 768×512 and 24 fps, with a matching soundtrack.",
                 symbol: "film", build: { _ in
                     GraphTemplates.textToVideo(modelPath: path("ltx-2.3"), outputDirectory: out, family: .ltx)
                 }),
            Spec(id: "ltx-i2v", name: "Image to Video", category: .video, models: ["ltx-2.3"],
                 modelName: "LTX-2.3", summary: "Animate a picture: it becomes the clip's first frame.",
                 symbol: "film.stack", build: { _ in
                     clearPaths(edited(GraphTemplates.textToVideo(modelPath: path("ltx-2.3"), outputDirectory: out, family: .ltx)) { m in
                         m.setFirstFrame(placeholderImage, undoManager: nil)
                     })
                 }),
            Spec(id: "wan-t2v", name: "Wan Text to Video", category: .video, models: ["wan-2.2-t2v"],
                 modelName: "Wan 2.2 T2V A14B", summary: "Silent 832×480 clips at 16 fps; 4 fast Lightning steps.",
                 symbol: "film", build: { _ in
                     GraphTemplates.textToVideo(modelPath: path("wan-2.2-t2v"), outputDirectory: out, family: .wan)
                 }),
            Spec(id: "wan-i2v", name: "Wan Image to Video", category: .video, models: ["wan-2.2-i2v"],
                 modelName: "Wan 2.2 I2V A14B", summary: "Animate a picture with Wan: it becomes the first frame.",
                 symbol: "film.stack", build: { _ in
                     clearPaths(edited(GraphTemplates.textToVideo(modelPath: path("wan-2.2-i2v"), outputDirectory: out, family: .wan)) { m in
                         m.setFirstFrame(placeholderImage, undoManager: nil)
                     })
                 }),
            Spec(id: "sound", name: "Sound Effect", category: .sound, models: ["stable-audio-open-small"],
                 modelName: "Stable Audio Open Small", summary: "Up to 11 s of sound from a description.",
                 symbol: "speaker.wave.2", build: { _ in
                     GraphTemplates.textToSound(modelPath: path("stable-audio-open-small"), outputDirectory: out)
                 }),
            Spec(id: "music", name: "Song", category: .music, models: ["ace-step-1.5"],
                 modelName: "ACE-Step 1.5", summary: "Music from style tags and lyrics, planned by the model.",
                 symbol: "music.note", build: { _ in
                     GraphTemplates.textToMusic(modelPath: path("ace-step-1.5"), outputDirectory: out)
                 }),
            Spec(id: "speech", name: "Text to Speech", category: .speech, models: ["kokoro-82m"],
                 modelName: "Kokoro 82M", summary: "Read text aloud in one of 28 English voices.",
                 symbol: "waveform", build: { _ in
                     GraphTemplates.textToSpeech(modelPath: path("kokoro-82m"), outputDirectory: out)
                 }),
        ]

        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        var ok = true
        for (i, spec) in specs.enumerated() {
            guard let g = spec.build(model) else { say("\(spec.id): could not build"); ok = false; continue }
            if spec.id.hasPrefix("flux-img2img"), g[AppModel.img2imgEncodeID] == nil {
                say("\(spec.id): no image-to-image chain (is FLUX.1 [dev] installed?)"); ok = false
            }
            let portable = Preset.portable(g, folderName: model.modelFolderName, outputDirectory: out)
            let p = Preset(id: spec.id, name: spec.name, category: spec.category, models: spec.models,
                           modelName: spec.modelName, summary: spec.summary, symbol: spec.symbol, order: (i + 1) * 10,
                           graph: portable)
            do {
                try e.encode(p).write(to: dir.appendingPathComponent("\(spec.id).json"), options: .atomic)
                say("wrote \(spec.id).json (\(g.nodes.count) nodes)")
            } catch {
                say("\(spec.id): \(error)"); ok = false
            }
        }
        return ok
    }
}
