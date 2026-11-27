import Foundation
import GraphRuntime
import AIStudioLibrary
import MLXDiffusionCore

/// `--export-presets <dir>` writes the built-in presets (`Sources/AIStudioLibrary/Presets`), built from the graph
/// templates and the same Add to Workflow chains the app uses, so they never drift apart. Needs the models
/// installed, since image-to-image wiring depends on the model's family.
@MainActor
enum PresetExport {
    struct Spec {
        let id: String, name: String, category: Preset.Category, models: [String], modelName: String
        let summary: String, symbol: String
        var needsServer: String? = nil
        let build: (AppModel) -> Graph?
    }

    /// A Cloud GPU node (optionally fed by Load Image) into a local Save node for its output (`video`, `image` or
    /// `audio`); `literals` are the node's preset values.
    /// The H3 presets' prompt, in the structure MiniMax's prompting guide describes.
    static let h3Prompt = "integrated_multimodal_description: [Shot 1] A wide shot of a lighthouse on a rocky headland at dusk; waves break against the rocks below as the camera slowly pushes in, and the lamp turns on, sweeping its beam across the water.\noverall_soundscape: Crashing surf, wind, gulls far off, a low mechanical hum as the lamp starts.\nnon_diegetic_music: none"

    static func cloudGraph(node type: String, literals: [String: JSONValue], startImage: Bool = false,
                           imageInput: String = "image", output: String, prefix: String? = nil,
                           outputDirectory out: String) -> Graph {
        var nodes: [GraphNode] = []
        var remote = literals.mapValues { NodeInput.value($0) }
        var x = 40.0
        if startImage {
            nodes.append(GraphNode(id: "first_frame", type: "LoadImage", inputs: ["path": .value(.string(""))],
                                   position: GraphPoint(x: 40, y: 140)))
            remote[imageInput] = .link(node: "first_frame", output: "image")
            x = 340
        }
        nodes.append(GraphNode(id: "cloud", type: type, inputs: remote, position: GraphPoint(x: x, y: 60)))
        let save = ["video": "SaveVideo", "image": "SaveImage", "audio": "SaveAudio"][output]!
        var inputs: [String: NodeInput] = [output: .link(node: "cloud", output: output), "directory": .value(.string(out)),
                                           "filename_prefix": .value(.string(prefix ?? output))]
        if output == "audio" { inputs["format"] = .value(.string("wav")) }
        nodes.append(GraphNode(id: "save", type: save, inputs: inputs, position: GraphPoint(x: x + 360, y: 120)))
        return Graph(nodes: nodes)
    }

    static let placeholderImage = URL(fileURLWithPath: "/placeholder.png")

    static let cloudLyrics = """
        [Verse]
        Morning light on the window pane
        Coffee steam and the sound of rain
        Every street is a song I know
        Humming softly, I let it go

        [Chorus]
        Oh, we're running with the open sky
        Paper planes on a summer high
        Every heartbeat a little louder
        We're alive, we're alive

        [Outro]
        We're alive, we're alive
        """

    /// `only`: just these preset ids (the others' files are left as they are).
    static func run(_ model: AppModel, to dir: URL, only: Set<String>? = nil) -> Bool {
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
            Spec(id: "cloud-wan-t2v", name: "Wan 2.2 Text to Video · Cloud GPU", category: .video, models: [],
                 modelName: "Wan 2.2 A14B on a GPU server", summary: "Wan 2.2 clips on an NVIDIA GPU, billed per second on Modal.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteWanTextToVideo",
                                literals: ["prompt": .string("A red fox trotting through fresh snow at sunrise, its breath steaming")],
                                output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-wan-i2v", name: "Wan 2.2 Image to Video · Cloud GPU", category: .video, models: [],
                 modelName: "Wan 2.2 A14B on a GPU server", summary: "Animate a picture on an NVIDIA GPU: it becomes the first frame.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteWanImageToVideo",
                                literals: ["prompt": .string("The camera slowly pushes in as the scene comes alive")],
                                startImage: true, output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-flux", name: "FLUX.1 dev · Cloud GPU", category: .image, models: [],
                 modelName: "FLUX.1 dev on a GPU server", summary: "FLUX.1 [dev] pictures on an NVIDIA GPU, billed per second on Modal.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteFlux",
                                literals: ["prompt": .string("A lighthouse on a rocky coast at dusk, warm light in the windows")],
                                output: "image", outputDirectory: out)
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
            // Appended, so the earlier presets keep their order numbers.
            Spec(id: "cloud-music", name: "Song · Cloud GPU", category: .music, models: [],
                 modelName: "ACE-Step 1.5 XL on a GPU server",
                 summary: "Songs up to 10 minutes from style tags and lyrics, planned by a 4B model, on an NVIDIA GPU.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteTextToMusic", literals: [
                        "tags": .string("uplifting indie pop, female vocals, acoustic guitar, warm piano, hand claps"),
                        "lyrics": .string(Self.cloudLyrics), "seconds": .double(60), "seed": .int(1),
                     ], output: "audio", prefix: "music", outputDirectory: out)
                 }),
            Spec(id: "cloud-sound", name: "Sound Effect · Cloud GPU", category: .sound, models: [],
                 modelName: "Stable Audio 3 on a GPU server",
                 summary: "Sound effects and ambience up to six minutes, 44.1 kHz stereo, on an NVIDIA GPU.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteTextToSound", literals: [
                        "prompt": .string("Heavy rain on a tin roof, distant rolling thunder, water dripping from the gutter"),
                        "seconds": .double(20), "seed": .int(1),
                     ], output: "audio", prefix: "sound", outputDirectory: out)
                 }),
            Spec(id: "cloud-speech", name: "Voice Clone & TTS · Cloud GPU", category: .speech, models: [],
                 modelName: "Qwen3-TTS on a GPU server",
                 summary: "Speech in 10 languages: a built-in voice, one designed from a description, or a clone of a clip (connect Load Audio to reference).",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteTextToSpeech", literals: [
                        "text": .string("Hello! This voice was made on a cloud GPU. Connect a short recording to the reference input, and I'll speak in that voice instead."),
                        "voice": .string("Ryan"), "seed": .int(1),
                     ], output: "audio", prefix: "speech", outputDirectory: out)
                 }),
            Spec(id: "cloud-flux-lora", name: "FLUX.1 dev + LoRA · Cloud GPU", category: .image, models: [],
                 modelName: "FLUX.1 dev on a GPU server",
                 summary: "FLUX.1 [dev] with your own LoRA: pick it in the LoRA node's menu, or add one there from a Civitai / Hugging Face link or a file (up to four with Add to Workflow).",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     var g = cloudGraph(node: "RemoteFlux",
                                        literals: ["prompt": .string("A portrait of a woman in a sunlit garden, in the style of your LoRA (add its trigger word)")],
                                        output: "image", outputDirectory: out)
                     for i in g.nodes.indices { g.nodes[i].position?.x += 300 }
                     g.nodes.insert(GraphNode(id: "lora1", type: "RemoteLoRA",
                                              inputs: ["lora": .value(.string("")), "path": .value(.string("")),
                                                       "strength": .value(.double(1))],
                                              position: GraphPoint(x: 40, y: 60)), at: 0)
                     if let i = g.nodes.firstIndex(where: { $0.id == "cloud" }) {
                         g.nodes[i].inputs["loras"] = .link(node: "lora1", output: "loras")
                     }
                     return g
                 }),
            Spec(id: "cloud-qwen-image", name: "Qwen-Image · Cloud GPU", category: .image, models: [],
                 modelName: "Qwen-Image (20B) on a GPU server",
                 summary: "The original 20B Qwen-Image on an NVIDIA GPU: crisp text in pictures and long prompts. Fast mode takes 8 steps; Qwen-Image fine-tunes and LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteQwenImage", literals: [
                        "prompt": .string("A cozy bookshop window on a rainy evening, a hand-painted sign reading \"Open Late\", warm light, reflections on the wet street"),
                     ], output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-sd35", name: "Stable Diffusion 3.5 Large · Cloud GPU", category: .image, models: [],
                 modelName: "SD 3.5 Large on a GPU server",
                 summary: "Stable Diffusion 3.5 Large on an NVIDIA GPU. Fast mode runs Large Turbo in 4 steps; SD 3.5 Large LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteSD35", literals: [
                        "prompt": .string("A lighthouse on a cliff at dusk, waves crashing below, dramatic clouds, detailed oil painting"),
                     ], output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-sdxl", name: "SDXL · Cloud GPU", category: .image, models: [],
                 modelName: "SDXL 1.0 on a GPU server",
                 summary: "SDXL 1.0 on an NVIDIA GPU. SDXL fine-tunes (Juggernaut, Pony, Illustrious…) and LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteSDXL", literals: [
                        "prompt": .string("A portrait of an old fisherman, weathered face, golden hour light, shallow depth of field, photo"),
                        "negative": .string("blurry, low quality, watermark"),
                     ], output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-qwen-edit", name: "Qwen-Image-Edit · Cloud GPU", category: .image, models: [],
                 modelName: "Qwen-Image-Edit 2511 on a GPU server",
                 summary: "Change a picture by telling it what to do, on an NVIDIA GPU: keeps faces, clothes and places. Up to three pictures (wire Load Image into image2 and image3: \"the woman from picture 1 sits in the room from picture 2\"). Fast mode takes 8 steps.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteQwenImageEdit", literals: [
                        "prompt": .string("Turn it into a snowy winter evening, warm light in the windows, keep everything else the same"),
                     ], startImage: true, imageInput: "image1", output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-wan-s2v", name: "Talking Clip · Cloud GPU", category: .video, models: [],
                 modelName: "Wan 2.2 S2V 14B on a GPU server",
                 summary: "A picture speaks: a lip-synced clip from a still and a speech recording (up to about 15 s), on an NVIDIA GPU. The clip carries the audio. Several minutes of GPU time per clip.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     Graph(nodes: [
                        GraphNode(id: "first_frame", type: "LoadImage", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 60)),
                        GraphNode(id: "speech", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 260)),
                        GraphNode(id: "cloud", type: "RemoteWanSpeechToVideo", inputs: [
                            "image": .link(node: "first_frame", output: "image"),
                            "audio": .link(node: "speech", output: "audio"),
                            "prompt": .value(.string("A person talking to the camera, natural expressions")),
                        ], position: GraphPoint(x: 340, y: 60)),
                        GraphNode(id: "show", type: "PreviewVideo", inputs: [
                            "video": .link(node: "cloud", output: "video")], position: GraphPoint(x: 700, y: 120)),
                     ])
                 }),
            Spec(id: "cloud-longcat-avatar", name: "Talking Clip, Fast · Cloud GPU", category: .video, models: [],
                 modelName: "LongCat-Video-Avatar 1.6 on a GPU server",
                 summary: "A picture speaks, in 8 steps: a lip-synced clip from a still and clean speech, on an NVIDIA GPU. Connect a second voice to audio 2 for a two-person scene (left and right). About a minute of GPU time per clip.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     Graph(nodes: [
                        GraphNode(id: "first_frame", type: "LoadImage", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 60)),
                        GraphNode(id: "speech", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 260)),
                        GraphNode(id: "cloud", type: "RemoteLongCatAvatar", inputs: [
                            "image": .link(node: "first_frame", output: "image"),
                            "audio": .link(node: "speech", output: "audio"),
                            "prompt": .value(.string("A person talking to the camera, natural expressions")),
                        ], position: GraphPoint(x: 340, y: 60)),
                        GraphNode(id: "show", type: "PreviewVideo", inputs: [
                            "video": .link(node: "cloud", output: "video")], position: GraphPoint(x: 700, y: 120)),
                     ])
                 }),
            Spec(id: "cloud-infinitetalk", name: "Lip-Sync a Clip · Cloud GPU", category: .video, models: [],
                 modelName: "InfiniteTalk on a GPU server",
                 summary: "Dub a clip: the speaker's lips, head and body follow new speech while the camera move stays, on an NVIDIA GPU. The result is as long as the speech. About a minute of GPU time for a short clip.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     Graph(nodes: [
                        GraphNode(id: "clip", type: "LoadVideo", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 60)),
                        GraphNode(id: "speech", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 260)),
                        GraphNode(id: "cloud", type: "RemoteInfiniteTalk", inputs: [
                            "video": .link(node: "clip", output: "video"),
                            "audio": .link(node: "speech", output: "audio"),
                            "prompt": .value(.string("A person talking, natural expressions")),
                        ], position: GraphPoint(x: 340, y: 60)),
                        GraphNode(id: "show", type: "PreviewVideo", inputs: [
                            "video": .link(node: "cloud", output: "video")], position: GraphPoint(x: 700, y: 120)),
                     ])
                 }),
            Spec(id: "cloud-breeze-tts", name: "Breeze TTS 2 · Cloud GPU (non-commercial)", category: .speech,
                 models: [], modelName: "Breeze TTS 2 on a GPU server",
                 summary: "Expressive English and Chinese speech: design a voice from a description, or connect Load Audio to reference (with its exact transcript) to clone and direct a voice. Research and non-commercial use only.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteBreezeSpeech", literals: [
                        "text": .string("(sigh) Welcome aboard. This voice was designed from a description, on a cloud GPU."),
                        "instruction": .string("A warm, thoughtful young woman with a clear voice and a calm, reflective delivery."),
                        "seed": .int(1),
                     ], output: "audio", prefix: "speech", outputDirectory: out)
                 }),
            Spec(id: "cloud-h3-t2v", name: "MiniMax H3 Video with Sound · Cloud GPU", category: .video, models: [],
                 modelName: "MiniMax H3 on a GPU server",
                 summary: "A 5–15 s clip with its own dialogue, effects and ambience from one prompt, on an H200. MiniMax H3's licence excludes the US, EU, UK and Korea without MiniMax's own licence.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteH3Video", literals: [
                        "prompt": .string(Self.h3Prompt), "seed": .int(1),
                     ], output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-h3-i2v", name: "MiniMax H3 Image to Video with Sound · Cloud GPU", category: .video, models: [],
                 modelName: "MiniMax H3 on a GPU server",
                 summary: "Animate a picture into a clip with its own sound: the picture is the first frame (connect a last frame too, if you like). MiniMax H3's licence excludes the US, EU, UK and Korea without MiniMax's own licence.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteH3Video", literals: [
                        "prompt": .string(Self.h3Prompt), "seed": .int(1),
                     ], startImage: true, output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-ltx23-t2v", name: "LTX-2.3 · Text to Video · Cloud GPU", category: .video, models: [],
                 modelName: "LTX-2.3 on a GPU server",
                 summary: "A clip with its own sound from one prompt. Fast mode runs the distilled model in 8 steps; LTX-2.3 fine-tunes and LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteLTX23", literals: [
                        "prompt": .string("A golden retriever runs along a beach at sunset, waves rolling in, its paws splashing; the sound of the surf and its happy barking"),
                        "seed": .int(1),
                     ], output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-ltx23-i2v", name: "LTX-2.3 · Image to Video · Cloud GPU", category: .video, models: [],
                 modelName: "LTX-2.3 on a GPU server",
                 summary: "Animate a picture into a clip with its own sound: the picture is the first frame.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteLTX23", literals: [
                        "prompt": .string("The scene comes to life: gentle camera push-in, natural motion and ambient sound"),
                        "seed": .int(1),
                     ], startImage: true, output: "video", outputDirectory: out)
                 }),
            Spec(id: "cloud-h3-references", name: "MiniMax H3 from References · Cloud GPU", category: .video, models: [],
                 modelName: "MiniMax H3 on a GPU server",
                 summary: "A clip from a reference picture and a voice clip: name them <Picture 1> and <Audio 1> in the prompt, and the character speaks in that voice. MiniMax H3's licence excludes the US, EU, UK and Korea without MiniMax's own licence.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     Graph(nodes: [
                        GraphNode(id: "picture", type: "LoadImage", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 60)),
                        GraphNode(id: "voice", type: "LoadAudio", inputs: ["path": .value(.string(""))],
                                  position: GraphPoint(x: 40, y: 300)),
                        GraphNode(id: "cloud", type: "RemoteH3References", inputs: [
                            "image1": .link(node: "picture", output: "image"),
                            "audio1": .link(node: "voice", output: "audio"),
                            "prompt": .value(.string("integrated_multimodal_description: [Shot 1] Medium close-up of <Picture 1> in soft window light; they look into the camera and speak warmly. <Picture 1> says: <d>[English] Hello! This clip was made from one picture and one voice.</d> (voice: <Audio 1>)\noverall_soundscape: A quiet room, faint birdsong outside.\nnon_diegetic_music: none")),
                            "seed": .value(.int(1)),
                        ], position: GraphPoint(x: 340, y: 60)),
                        GraphNode(id: "show", type: "PreviewVideo", inputs: [
                            "video": .link(node: "cloud", output: "video")], position: GraphPoint(x: 700, y: 120)),
                     ])
                 }),
            Spec(id: "cloud-h3-image", name: "MiniMax H3 Image · Cloud GPU", category: .image, models: [],
                 modelName: "MiniMax H3 Image on a GPU server",
                 summary: "MiniMax H3 Image on an NVIDIA H200: detailed, realistic pictures from long descriptions, in six steps. Wire Load Image into image1…image4 and name them in the prompt as <Picture 1>… to edit or combine pictures. Its licence excludes the US, EU, UK and Korea.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteH3Image", literals: [
                        "prompt": .string("A portrait photo of an old fisherman on a wooden pier at golden hour: deep wrinkles, a gray beard, a knitted navy cap, warm low sun from the left, the sea softly out of focus behind him, shallow depth of field"),
                     ], output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-chroma", name: "Chroma · Cloud GPU", category: .image, models: [],
                 modelName: "Chroma1-HD on a GPU server",
                 summary: "Chroma1-HD, a Flux-based open model, on an NVIDIA GPU. Fast mode runs Chroma1-Flash (16 steps); Chroma fine-tunes and LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteChroma", literals: [
                        "prompt": .string("A portrait of an old fisherman, weathered face, golden hour light, shallow depth of field, photo"),
                        "negative": .string("low quality, blurry, deformed, watermark"),
                     ], output: "image", outputDirectory: out)
                 }),
            Spec(id: "cloud-anima", name: "Anima · Cloud GPU", category: .image, models: [],
                 modelName: "Anima on a GPU server",
                 summary: "Anima: a 2B anime text-to-image model (Danbooru-style tags or natural language) on the cloud GPU. Anima fine-tunes and LoRAs from the library work too.",
                 symbol: "cloud", needsServer: "gpu", build: { _ in
                     cloudGraph(node: "RemoteAnima", literals: [
                        "prompt": .string("masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, red eyes, school uniform, cherry blossoms, park, smile, looking at viewer, upper body"),
                        "negative": .string("worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, chromatic aberration"),
                     ], output: "image", outputDirectory: out)
                 }),
        ]

        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        var ok = true
        for (i, spec) in specs.enumerated() where only?.contains(spec.id) ?? true {
            guard let g = spec.build(model) else { say("\(spec.id): could not build"); ok = false; continue }
            if spec.id.hasPrefix("flux-img2img"), g[AppModel.img2imgEncodeID] == nil {
                say("\(spec.id): no image-to-image chain (is FLUX.1 [dev] installed?)"); ok = false
            }
            let portable = Preset.portable(g, folderName: model.modelFolderName, outputDirectory: out)
            let p = Preset(id: spec.id, name: spec.name, category: spec.category, models: spec.models,
                           modelName: spec.modelName, summary: spec.summary, symbol: spec.symbol, order: (i + 1) * 10,
                           graph: portable, needsServer: spec.needsServer)
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
