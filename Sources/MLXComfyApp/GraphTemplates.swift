import Foundation
import GraphRuntime
import MLXDiffusionCore
import NodesStandard

enum GraphTemplates {
    /// The graph behind Simple mode: model → prompt → empty latent → sampler → decode → save.
    static func textToImage(modelPath: String, outputDirectory: String,
                            prompt: String = "A red fox standing in fresh snow at sunrise, detailed fur, photograph")
        -> Graph
    {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "prompt", type: "TextEncode",
                      inputs: ["model": link("model", "model"), "prompt": .value(.string(prompt))],
                      position: GraphPoint(x: 340, y: 40)),
            GraphNode(id: "latent", type: "EmptyLatent",
                      inputs: ["width": .value(.int(1024)), "height": .value(.int(1024))],
                      position: GraphPoint(x: 340, y: 300)),
            GraphNode(id: "sampler", type: "Sampler",
                      inputs: ["model": link("model", "model"), "conditioning": link("prompt", "conditioning"),
                               "latent": link("latent", "latent"), "seed": .value(.int(1)),
                               "steps": .value(.int(0))],
                      position: GraphPoint(x: 640, y: 120)),
            GraphNode(id: "decode", type: "VAEDecode",
                      inputs: ["model": link("model", "model"), "latent": link("sampler", "latent")],
                      position: GraphPoint(x: 940, y: 140)),
            GraphNode(id: "save", type: "SaveImage",
                      inputs: ["image": link("decode", "image"), "filename_prefix": .value(.string("mlxcomfy")),
                               "directory": .value(.string(outputDirectory))],
                      position: GraphPoint(x: 1220, y: 140)),
        ])
    }

    /// The graph behind the Video mode: model → prompt → empty clip → video sampler → decode → save (with sound).
    static func textToVideo(modelPath: String, outputDirectory: String, family: VideoFamily? = nil,
                            prompt: String = "A red fox trotting through fresh snow at sunrise, its breath steaming, birds calling in the distance")
        -> Graph
    {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        let wan = (family ?? VideoFamily.of(path: modelPath)) == .wan
        let (w, h, fps) = wan ? (832, 480, 16.0) : (768, 512, 24.0)
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "prompt", type: "TextEncode",
                      inputs: ["model": link("model", "model"), "prompt": .value(.string(prompt))],
                      position: GraphPoint(x: 340, y: 40)),
            GraphNode(id: "latent", type: "EmptyVideoLatent",
                      inputs: ["width": .value(.int(w)), "height": .value(.int(h)), "seconds": .value(.double(5)),
                               "fps": .value(.double(fps))],
                      position: GraphPoint(x: 340, y: 300)),
            GraphNode(id: "sampler", type: "VideoSampler",
                      inputs: ["model": link("model", "model"), "conditioning": link("prompt", "conditioning"),
                               "latent": link("latent", "latent"), "seed": .value(.int(1)), "steps": .value(.int(0))],
                      position: GraphPoint(x: 640, y: 120)),
            GraphNode(id: "decode", type: "VideoDecode",
                      inputs: ["model": link("model", "model"), "latent": link("sampler", "latent")],
                      position: GraphPoint(x: 940, y: 140)),
            GraphNode(id: "save", type: "SaveVideo",
                      inputs: ["video": link("decode", "video"), "audio": link("decode", "audio"),
                               "filename_prefix": .value(.string("video")),
                               "directory": .value(.string(outputDirectory))],
                      position: GraphPoint(x: 1220, y: 140)),
        ])
    }

    /// The graph behind the Sound mode: model → text to audio → save (WAV).
    static func textToSound(modelPath: String, outputDirectory: String,
                            prompt: String = "Rain on a tin roof with distant thunder, cozy and close") -> Graph {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "generate", type: "TextToAudio",
                      inputs: ["model": link("model", "model"), "prompt": .value(.string(prompt)),
                               "seconds": .value(.double(8)), "seed": .value(.int(1)), "steps": .value(.int(0))],
                      position: GraphPoint(x: 340, y: 100)),
            GraphNode(id: "save", type: "SaveAudio",
                      inputs: ["audio": link("generate", "audio"), "filename_prefix": .value(.string("sound")),
                               "directory": .value(.string(outputDirectory)), "format": .value(.string("wav"))],
                      position: GraphPoint(x: 680, y: 140)),
        ])
    }

    /// The graph behind the Music mode: model → text to music → save (WAV, 48 kHz stereo).
    static func textToMusic(modelPath: String, outputDirectory: String,
                            tags: String = "Upbeat synthwave with punchy drums, warm analog bass and a catchy lead melody",
                            lyrics: String = "[Instrumental]") -> Graph {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "generate", type: "TextToMusic",
                      inputs: ["model": link("model", "model"), "tags": .value(.string(tags)),
                               "lyrics": .value(.string(lyrics)), "seconds": .value(.double(30)),
                               "seed": .value(.int(1)), "steps": .value(.int(0)), "planner": .value(.bool(true)),
                               "bpm": .value(.int(0)), "key": .value(.string("auto")),
                               "time_signature": .value(.string("auto")), "language": .value(.string("auto"))],
                      position: GraphPoint(x: 340, y: 60)),
            GraphNode(id: "save", type: "SaveAudio",
                      inputs: ["audio": link("generate", "audio"), "filename_prefix": .value(.string("music")),
                               "directory": .value(.string(outputDirectory)), "format": .value(.string("wav"))],
                      position: GraphPoint(x: 700, y: 140)),
        ])
    }

    /// The graph behind the Speech mode: model → text to speech → save (WAV, 24 kHz mono).
    static func textToSpeech(modelPath: String, outputDirectory: String,
                             text: String = "Welcome to MLX Comfy. Everything you hear was generated on this Mac, from text to waveform, in a fraction of a second.",
                             voice: String = "af_heart") -> Graph {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "speak", type: "TextToSpeech",
                      inputs: ["model": link("model", "model"), "text": .value(.string(text)), "voice": .value(.string(voice)),
                               "speed": .value(.double(1)), "seed": .value(.int(1))],
                      position: GraphPoint(x: 340, y: 100)),
            GraphNode(id: "save", type: "SaveAudio",
                      inputs: ["audio": link("speak", "audio"), "filename_prefix": .value(.string("speech")),
                               "directory": .value(.string(outputDirectory)), "format": .value(.string("wav"))],
                      position: GraphPoint(x: 680, y: 140)),
        ])
    }

    /// Voice previews: the same model, a short sentence in the chosen voice, shown but not saved to history.
    static func voicePreview(modelPath: String, voice: String, speed: Double) -> Graph {
        let v = SpeechVoices.all.first { $0.id == voice }
        let text = "Hi, I'm \(v?.name ?? voice). This is how I sound when I read your text aloud."
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))]),
            GraphNode(id: "speak", type: "TextToSpeech",
                      inputs: ["model": link("model", "model"), "text": .value(.string(text)), "voice": .value(.string(voice)),
                               "speed": .value(.double(speed)), "seed": .value(.int(1))]),
            GraphNode(id: "preview", type: "PreviewAudio", inputs: ["audio": link("speak", "audio")]),
        ])
    }

    /// Video sizes: two-stage generation needs multiples of 64.
    static let videoSizePresets: [(name: String, width: Int, height: Int)] = [
        ("Landscape 3:2", 768, 512),
        ("Portrait 2:3", 512, 768),
        ("Landscape 16:9", 1024, 576),
        ("Portrait 9:16", 576, 1024),
        ("Square 1:1", 640, 640),
        ("Small 3:2 (fast)", 512, 320),
    ]

    /// Wan 2.2 sizes (multiples of 16; trained at 832×480 and 1280×720).
    static let wanSizePresets: [(name: String, width: Int, height: Int)] = [
        ("Landscape 16:9", 832, 480),
        ("Portrait 9:16", 480, 832),
        ("Square 1:1", 624, 624),
        ("Small 16:9 (fast)", 512, 288),
    ]

    static let sizePresets: [(name: String, width: Int, height: Int)] = [
        ("Square 1:1", 1024, 1024),
        ("Portrait 3:4", 896, 1184),
        ("Landscape 4:3", 1184, 896),
        ("Portrait 9:16", 768, 1344),
        ("Landscape 16:9", 1344, 768),
    ]
}

/// Which video model family a graph's model folder holds (the Video panel adapts its settings to it).
enum VideoFamily: String {
    case ltx, wan

    static func of(path: String) -> VideoFamily {
        guard !path.isEmpty else { return .ltx }
        return ModelRegistry.className(at: ModelPaths.resolve(path)) == "WanPipeline" ? .wan : .ltx
    }
}

/// Image model families the Image panel adapts to (read from the folder's `model_index.json` class name).
enum ImageFamily: String {
    case qwen, flux

    static func of(path: String) -> ImageFamily {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
        return ModelRegistry.className(at: url) == "FluxPipeline" ? .flux : .qwen
    }
}
