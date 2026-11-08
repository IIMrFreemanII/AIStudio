import ACEStep
import KokoroSpeech
import ArgumentParser
import Foundation
import GraphRuntime
import LTX2
import StableAudio
import MLX
import MLXDiffusionCore
import NodesStandard
import QwenImage21
import Tokenizers

@main
struct MLXComfyCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlxcomfy-cli",
        abstract: "MLXComfyUI engine command line.",
        subcommands: [Run.self, Generate.self, GenerateVideo.self, GenerateAudio.self, GenerateMusic.self, TTS.self, ProbeACEStep.self, ProbeStableAudio.self, Nodes.self, GraphFromPNG.self, Compare.self, VAERoundTrip.self,
                      FitPreview.self, MakeTestLoRA.self, Assist.self,
                      Tokenize.self])
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func memoryString() -> String {
    String(format: "active %.1f GB, peak %.1f GB", Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9)
}

func parseSize(_ s: String) throws -> (Int, Int) {
    let p = s.lowercased().split(separator: "x").compactMap { Int($0) }
    guard p.count == 2 else { throw ValidationError("size must look like 1024x1024") }
    return (p[0], p[1])
}

/// `node.input=value` overrides.
func applyOverrides(_ overrides: [String], to graph: inout Graph) throws {
    for o in overrides {
        guard let eq = o.firstIndex(of: "="), let dot = o[..<eq].firstIndex(of: ".") else {
            throw ValidationError("override '\(o)' must look like node.input=value")
        }
        try graph.setValue(JSONValue.parse(String(o[o.index(after: eq)...])), node: String(o[..<dot]),
                           input: String(o[o.index(after: dot)..<eq]))
    }
}

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Execute a graph JSON file.")

    @Argument(help: "Graph JSON (or a PNG saved by SaveImage).") var graph: String
    @Option(name: .customLong("set"), help: "Override a literal input: node.input=value (repeatable).")
    var overrides: [String] = []
    @Option(name: .customLong("then-set"),
            help: "Run a second time in the same process with these overrides, to exercise the cache.")
    var secondOverrides: [String] = []

    func run() async throws {
        let url = URL(fileURLWithPath: graph)
        var g: Graph
        if url.pathExtension.lowercased() == "png" {
            guard let json = ImageArray.readPNGText(url)[StandardNodes.graphMetadataKey] else {
                throw ValidationError("\(graph) carries no graph metadata")
            }
            g = try JSONDecoder().decode(Graph.self, from: Data(json.utf8))
        } else {
            g = try Graph.load(url)
        }
        try applyOverrides(overrides, to: &g)

        let executor = GraphExecutor(registry: StandardNodes.makeRegistry(), hooks: StandardNodes.mlxHooks)
        try await runOnce(executor, g, label: "run 1")
        if !secondOverrides.isEmpty {
            try applyOverrides(secondOverrides, to: &g)
            try await runOnce(executor, g, label: "run 2")
        }
    }

    func runOnce(_ executor: GraphExecutor, _ g: Graph, label: String) async throws {
        log("── \(label)")
        let result = try await executor.run(g) { event in
            switch event {
            case .started(let nodes): log("plan: \(nodes.joined(separator: " → "))")
            case .nodeStarted(let n): log("▶︎ \(n)")
            case .nodeCached(let n): log("✓ \(n) (cached)")
            case .nodeProgress(let n, let s, let t, _): log("  \(n) \(s)/\(t)")
            case .nodeFinished(let n, let secs, let peak):
                log(String(format: "✓ %@ %.2fs%@", n, secs, peak.map { String(format: ", peak %.1f GB", Double($0) / 1e9) } ?? ""))
            case .nodeFailed(let n, let e): log("✗ \(n): \(e)")
            case .finished(let secs): log(String(format: "done in %.1fs (%@)", secs, memoryString()))
            case .cancelled: log("cancelled")
            }
        }
        for (node, out) in result.outputs.sorted(by: { $0.key < $1.key }) {
            if case .object(let o)? = out["images"], let urls = o as? [URL] {
                urls.forEach { log("\(node) wrote \($0.path)") }
            }
        }
        log("executed \(result.executed.count), cached \(result.cached.count)")
    }
}

struct Nodes: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List available node types.")

    func run() async throws {
        for s in StandardNodes.makeRegistry().schemas {
            let ins = s.inputs.map { i in
                "\(i.name): \(i.type)" + (i.defaultValue.map { " = \($0)" } ?? "")
            }.joined(separator: ", ")
            let outs = s.outputs.map { "\($0.name): \($0.type)" }.joined(separator: ", ")
            print("\(s.type) [\(s.category)]\(s.isOutput ? " (output)" : "")\n  in:  \(ins)\n  out: \(outs)")
        }
    }
}

struct GraphFromPNG: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "graph-from-png", abstract: "Print the graph embedded in a saved PNG.")
    @Argument var png: String

    func run() async throws {
        guard let json = ImageArray.readPNGText(URL(fileURLWithPath: png))[StandardNodes.graphMetadataKey] else {
            throw ValidationError("no graph metadata in \(png)")
        }
        let g = try JSONDecoder().decode(Graph.self, from: Data(json.utf8))
        print(String(data: try g.jsonData(), encoding: .utf8)!)
    }
}

struct GenerateAudio: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate-audio", abstract: "Text to audio with an audio model (Stable Audio Open), no graph.")

    @Option(help: "Model folder (with an mlxcomfy.json manifest).") var model: String
    @Option(help: "Prompt text.") var prompt: String
    @Option(help: "Length in seconds.") var seconds: Double = 10
    @Option(help: "Random seed.") var seed: UInt64 = 0
    @Option(help: "Sampling steps (default: the model's).") var steps: Int?
    @Option(help: "Output path (.wav or .m4a).") var out = "out.wav"

    func run() async throws {
        _ = StandardNodes.makeRegistry()
        var t = Date()
        let loaded = try await ModelRegistry.shared.load(ModelPaths.resolve(model))
        guard let pipe = loaded as? AudioModel else { throw ValidationError("\(loaded.displayName) is not an audio model") }
        log(String(format: "loaded %@ in %.1fs (%@)", pipe.displayName, Date().timeIntervalSince(t), memoryString()))
        t = Date()
        let clip = try pipe.generateAudio(AudioRequest(prompt: prompt, seconds: seconds, seed: seed, steps: steps)) { i, n in
            log(String(format: "step %d/%d  %.2fs", i, n, Date().timeIntervalSince(t)))
        }
        let x = clip.samples
        log(String(format: "generated %.2f s at %d Hz, %d ch in %.1fs (%@); rms %.4f, peak %.4f", clip.duration,
                   clip.sampleRate, clip.channels, Date().timeIntervalSince(t), memoryString(),
                   sqrt((x * x).mean()).item(Float.self), abs(x).max().item(Float.self)))
        let url = URL(fileURLWithPath: out)
        try await MediaIO.writeAudio(clip, to: url, format: url.pathExtension.lowercased() == "m4a" ? .m4a : .wav,
                                     metadata: ["mlxcomfy.prompt": prompt, "mlxcomfy.seed": String(seed)])
        log("wrote \(url.path)")
    }
}

struct GenerateMusic: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate-music", abstract: "Text and lyrics to music with a music model (ACE-Step 1.5), no graph.")

    @Option(help: "Model folder (with an mlxcomfy.json manifest).") var model: String
    @Option(help: "Style: genre, instruments, mood, vocals.") var tags: String
    @Option(help: "Lyrics with [Verse]/[Chorus] markers, or [Instrumental]. A path to a text file also works.")
    var lyrics = "[Instrumental]"
    @Option(help: "Length in seconds (10–240).") var seconds: Double = 30
    @Option(help: "Random seed.") var seed: UInt64 = 0
    @Option(help: "Diffusion steps (default: the model's).") var steps: Int?
    @Option(help: "Tempo in BPM (default: auto).") var bpm: Int?
    @Option(help: "Key, e.g. \"A minor\" (default: auto).") var key = ""
    @Option(help: "Time signature: 2, 3, 4 or 6 (default: auto).") var timeSignature = ""
    @Option(help: "Vocal language code, e.g. en (default: auto).") var language = ""
    @Flag(help: "Skip the song-planner LM (faster, weaker structure).") var noPlanner = false
    @Option(help: "Output path (.wav or .m4a).") var out = "song.wav"

    func run() async throws {
        _ = StandardNodes.makeRegistry()
        var lyricText = lyrics
        if FileManager.default.fileExists(atPath: lyrics), let s = try? String(contentsOfFile: lyrics, encoding: .utf8) {
            lyricText = s
        }
        var t = Date()
        let loaded = try await ModelRegistry.shared.load(ModelPaths.resolve(model))
        guard let pipe = loaded as? AudioModel else { throw ValidationError("\(loaded.displayName) is not an audio model") }
        log(String(format: "loaded %@ in %.1fs (%@)", pipe.displayName, Date().timeIntervalSince(t), memoryString()))
        t = Date()
        let request = AudioRequest(prompt: tags, seconds: seconds, seed: seed, steps: steps, lyrics: lyricText, bpm: bpm,
                                   key: key, timeSignature: timeSignature, language: language, planner: !noPlanner)
        var lastLogged = -1
        let clip = try pipe.generateAudio(request) { i, n in
            let pct = n > 0 ? i * 20 / n : 0
            if pct != lastLogged || i == n {
                lastLogged = pct
                log(String(format: "progress %d/%d  %.2fs", i, n, Date().timeIntervalSince(t)))
            }
        }
        let x = clip.samples
        log(String(format: "generated %.2f s at %d Hz, %d ch in %.1fs (%@); rms %.4f, peak %.4f", clip.duration,
                   clip.sampleRate, clip.channels, Date().timeIntervalSince(t), memoryString(),
                   sqrt((x * x).mean()).item(Float.self), abs(x).max().item(Float.self)))
        let url = URL(fileURLWithPath: out)
        try await MediaIO.writeAudio(clip, to: url, format: url.pathExtension.lowercased() == "m4a" ? .m4a : .wav,
                                     metadata: ["mlxcomfy.prompt": tags, "mlxcomfy.seed": String(seed)])
        log("wrote \(url.path)")
    }
}

struct TTS: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tts", abstract: "Text to speech with a speech model (Kokoro), no graph.")

    @Option(help: "Model folder (with an mlxcomfy.json manifest).") var model: String
    @Option(help: "Text to speak, or a path to a text file.") var text: String
    @Option(help: "Voice, e.g. af_heart (American) or bf_emma (British).") var voice = "af_heart"
    @Option(help: "Speaking rate (0.5–2).") var speed: Double = 1
    @Option(help: "Seed for the vocoder's noise.") var seed: UInt64 = 0
    @Flag(help: "Print the phonemes instead of speaking.") var phonemes = false
    @Option(help: "Output path (.wav or .m4a).") var out = "speech.wav"

    func run() async throws {
        _ = StandardNodes.makeRegistry()
        var input = text
        if FileManager.default.fileExists(atPath: text), let s = try? String(contentsOfFile: text, encoding: .utf8) { input = s }
        var t = Date()
        let loaded = try await ModelRegistry.shared.load(ModelPaths.resolve(model))
        guard let pipe = loaded as? KokoroPipeline else { throw ValidationError("\(loaded.displayName) is not a speech model") }
        log(String(format: "loaded %@ in %.1fs (%@)", pipe.displayName, Date().timeIntervalSince(t), memoryString()))
        if phonemes {
            for chunk in try pipe.chunks(input, voice: voice) { print(chunk.phonemes) }
            return
        }
        t = Date()
        let clip = try pipe.generateAudio(AudioRequest(prompt: input, seed: seed, voice: voice, speed: speed), onStep: nil)
        let secs = Date().timeIntervalSince(t)
        log(String(format: "spoke %.2f s of audio at %d Hz in %.2fs (%.0f× real time, %@)", clip.duration, clip.sampleRate,
                   secs, clip.duration / max(secs, 1e-6), memoryString()))
        let url = URL(fileURLWithPath: out)
        try await MediaIO.writeAudio(clip, to: url, format: url.pathExtension.lowercased() == "m4a" ? .m4a : .wav,
                                     metadata: ["mlxcomfy.prompt": input, "mlxcomfy.voice": voice])
        log("wrote \(url.path)")
    }
}

/// Development check: the ACE-Step pipeline stages on a dump from the official pipeline (scratchpad ace_ref.py), with
/// the reference's noise: prompt tokens, conditioning, the sampler and the decode.
struct ProbeACEStep: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "probe-ace-step", shouldDisplay: false)
    @Option var model: String
    @Option var dump: String
    @Option var caption = "upbeat synthwave with punchy drums, warm analog bass and a catchy lead melody"
    @Option var seconds: Double = 20

    func run() async throws {
        let d = try MLX.loadArrays(url: URL(fileURLWithPath: dump))
        func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            guard a.shape == b.shape else { log("\(name): shape \(a.shape) vs ref \(b.shape)"); return }
            let diff = (a.asType(.float32) - b.asType(.float32))
            let rel = sqrt((diff * diff).sum()) / sqrt((b.asType(.float32) * b.asType(.float32)).sum())
            log(String(format: "%@: shape %@ max-abs %.3g rel-L2 %.3g", name, "\(a.shape)", abs(diff).max().item(Float.self),
                       rel.item(Float.self)))
        }
        var t = Date()
        let pipe = try await ACEStepPipeline(directory: ModelPaths.resolve(model))
        log(String(format: "loaded in %.1fs (%@)", Date().timeIntervalSince(t), memoryString()))
        let metas = ACEPrompts.metas(bpm: nil, timeSignature: "", keyScale: "", seconds: seconds)
        let textPrompt = ACEPrompts.caption(caption: caption, metas: metas)
        let lyricPrompt = ACEPrompts.lyrics("[Instrumental]", language: "")
        let ids = pipe.tokens(textPrompt, max: 256)
        let refIDs = d["te_ids0"]!.asType(.int32)
        log("caption ids equal: \(ids.shape == refIDs.shape && (ids .== refIDs).all().item(Bool.self)) (\(ids.dim(1)) tokens)")
        t = Date()
        let enc = pipe.encode(captionPrompt: textPrompt, lyricPrompt: lyricPrompt)
        eval(enc)
        log(String(format: "encode %.2fs", Date().timeIntervalSince(t)))
        report("cond_enc", enc, d["cond_enc"]!)
        let frames = d["cond_ctx"]!.dim(1)
        let ctx = concatenated([pipe.silenceFrames(frames), MLXArray.ones([1, frames, 64])], axis: -1)
        report("cond_ctx", ctx, d["cond_ctx"]!)
        t = Date()
        let lat = try pipe.sample(noise: d["dit_x0"]!, context: ctx, encoder: enc)
        log(String(format: "sample %.2fs (%@)", Date().timeIntervalSince(t), memoryString()))
        report("target_latents", lat, d["gen_target_latents"]!)
        t = Date()
        let audio = pipe.decode(lat, seconds: seconds)
        log(String(format: "decode %.2fs (%@)", Date().timeIntervalSince(t), memoryString()))
        report("audio", audio, d["audio"]!)
    }
}

/// Development check: runs the Stable Audio modules on inputs dumped from the MLX reference and prints the error.
struct ProbeStableAudio: AsyncParsableCommand {
    static let configuration = CommandConfiguration(commandName: "probe-stable-audio", shouldDisplay: false)
    @Option var model: String
    @Option var dump: String

    func run() async throws {
        let d = try MLX.loadArrays(url: URL(fileURLWithPath: dump))
        let dir = ModelPaths.resolve(model)
        func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
            let diff = (a.asType(.float32) - b.asType(.float32))
            let rel = sqrt((diff * diff).sum()) / sqrt((b.asType(.float32) * b.asType(.float32)).sum())
            log(String(format: "%@: shape %@ max-abs %.3g rel-L2 %.3g", name, "\(a.shape)", abs(diff).max().item(Float.self),
                       rel.item(Float.self)))
        }
        let store = try WeightStore(url: dir.appendingPathComponent("model.safetensors"))
        let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("model_config.json"))) as? [String: Any] ?? [:]
        let dit = try StableAudioDiT(store: store, config: StableAudioDiT.Config(modelConfig: cfg))
        var blockIndex = 0
        dit.trace = { name, a in
            let key = name == "block" ? "block\(blockIndex)" : name
            if name == "block" { blockIndex += 1 }
            if let r = d[key] { report(key, a, r) }
        }
        report("dit velocity", dit(d["x"]!, t: d["t"]!, cond: d["cond"]!, global: d["g"]!), d["v"]!)
        let dec = try OobleckDecoder(store: store, prefix: "pretransform.model.decoder.")
        report("oobleck audio", dec(d["lat"]!), d["audio"]!)
        let tok = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("t5"))
        log("swift ids: \(tok.encode(text: "128 BPM tech house drum loop, punchy kick"))")
        log("ref ids:   \(d["ids"]![0][0..<13].asArray(Int32.self))")
        let t5 = try T5Encoder(directory: dir.appendingPathComponent("t5"))
        report("t5", t5(ids: d["ids"]!, mask: d["am"]!), d["t5"]!)
    }
}

struct GenerateVideo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "generate-video", abstract: "Text/image-to-video with sound (LTX-2.3, no graph).")

    @Option(help: "Model folder (with an mlxcomfy.json manifest).") var model: String
    @Option(help: "Prompt text.") var prompt: String
    @Option(help: "Size WxH (multiples of 64).") var size = "768x512"
    @Option(help: "Length in seconds.") var seconds: Double = 5
    @Option(help: "Frames per second.") var fps: Double = 24
    @Option(help: "Random seed.") var seed: UInt64 = 0
    @Option(help: "Stage-1 steps (default: the distilled schedule's 8).") var steps: Int?
    @Option(help: "First frame image (image-to-video).") var image: String?
    @Option(help: "Output MP4 path.") var out = "out.mp4"
    @Option(help: "Also write a PNG contact sheet of 6 evenly spaced frames.") var contactSheet: String?
    @Option(help: "Fit the sampler-preview colour map to this clip and write it as Swift source to this path.")
    var fitPreview: String?
    @Option(help: "Save each sampler step's preview as <prefix>N.png.") var previewPrefix: String?

    func run() async throws {
        _ = StandardNodes.makeRegistry()  // registers the model loaders
        let (w, h) = try parseSize(size)
        var t = Date()
        let loaded = try await ModelRegistry.shared.load(ModelPaths.resolve(model))
        guard let pipe = loaded as? VideoDiffusionModel else { throw ValidationError("\(loaded.displayName) is not a video model") }
        log(String(format: "loaded %@ in %.1fs (%@)", pipe.displayName, Date().timeIntervalSince(t), memoryString()))
        t = Date()
        let cond = try pipe.encodeVideoPrompt(prompt)
        log(String(format: "prompt encoded in %.1fs (%@)", Date().timeIntervalSince(t), memoryString()))
        let first = try image.map { ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: $0))) }
        let latent = VideoLatent(width: w, height: h, frames: VideoLatent.frames(seconds: seconds, fps: fps), fps: fps)
        t = Date()
        let sampled = try pipe.sampleVideo(conditioning: cond, latent: latent, seed: seed, steps: steps, image: first) { s in
            log(String(format: "%@ step %d/%d  %.1fs", s.stage, s.index + 1, s.count, Date().timeIntervalSince(t)))
            if let p = previewPrefix, let rgb = s.preview {
                try? ImageArray.savePNG(rgb, to: URL(fileURLWithPath: "\(p)\(s.index + 1).png"))
            }
        }
        log(String(format: "sampled %dx%d, %d frames in %.1fs (%@)", sampled.width, sampled.height, sampled.frames,
                   Date().timeIntervalSince(t), memoryString()))
        t = Date()
        let clip = try pipe.decodeVideo(sampled)
        log(String(format: "decoded in %.1fs (%@)", Date().timeIntervalSince(t), memoryString()))
        if let path = fitPreview, let l = sampled.payload as? LTXVideoLatents {
            let c = LTXPreview.fit(tokens: l.video, frames: l.frames, height: l.height, width: l.width, pixels: clip.frames)
            let body = stride(from: 0, to: c.count, by: 8).map { i in
                "        " + c[i..<min(i + 8, c.count)].map { String(format: "%.6g", $0) }.joined(separator: ", ") + ","
            }.joined(separator: "\n")
            let src = "// Generated by `mlxcomfy-cli generate-video --fit-preview`: weights [128, 3] row-major, then bias [3].\n"
                + "enum LTXPreviewCoefficients {\n    static let values: [Float] = [\n\(body)\n    ]\n}\n"
            try src.write(toFile: path, atomically: true, encoding: .utf8)
            log("wrote preview coefficients to \(path)")
        }
        if let a = clip.audio {
            let x = a.samples
            log(String(format: "audio %d ch @ %d Hz: rms %.4f, peak %.4f", a.channels, a.sampleRate,
                       sqrt((x * x).mean()).item(Float.self), abs(x).max().item(Float.self)))
        }
        if let sheet = contactSheet {
            let n = clip.frameCount
            let picks = (0..<6).map { MLXArray(Int32(($0 * (n - 1)) / 5)) }
            let tiles = picks.map { clip.frames[$0] }
            let rows = [concatenated(Array(tiles[0..<3]), axis: 1), concatenated(Array(tiles[3..<6]), axis: 1)]
            try ImageArray.savePNG(concatenated(rows, axis: 0), to: URL(fileURLWithPath: sheet))
            log("wrote contact sheet \(sheet)")
        }
        let url = URL(fileURLWithPath: out)
        try await MediaIO.writeVideo(clip, audio: nil, to: url, metadata: [
            "mlxcomfy.prompt": prompt, "mlxcomfy.seed": String(seed), "mlxcomfy.model": pipe.displayName,
        ])
        log(String(format: "wrote %@ (%d frames, %.2f s, audio %.2f s)", url.path, clip.frameCount, clip.duration,
                   clip.audio?.duration ?? 0))
    }
}

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Text-to-image with Qwen-Image 2.1 (no graph).")

    @Option(help: "Model folder (diffusers layout).") var model = "qwen-turbo-mlx"
    @Option(help: "Prompt text.") var prompt: String
    @Option(help: "Output size WxH (default 1024x1024, or the condition image's aspect at ~1 MP).") var size: String?
    @Option(help: "Random seed.") var seed: UInt64 = 0
    @Option(help: "Override the checkpoint's step count.") var steps: Int?
    @Option(name: .customLong("image"), help: "Condition image for editing (repeatable).") var images: [String] = []
    @Option(name: .customLong("lora"), help: "LoRA file, optionally with strength: path[:strength] (repeatable).")
    var loras: [String] = []
    @Option(help: "Output PNG path.") var out = "out.png"
    @Flag(help: "Save every step's x0 estimate as out.stepN.png.") var saveSteps = false
    @Flag(help: "Expand the 8-bit transformer to bf16 at load (more memory, faster steps).") var dequantize = false

    func run() async throws {
        let (w, h) = try parseSize(size ?? "1024x1024")
        let t0 = Date()
        var pipe = try await QwenImage21Pipeline(directory: ModelPaths.resolve(model),
                                                 dequantizeTransformer: dequantize, log: log)
        log(String(format: "loaded in %.1fs (%@)", Date().timeIntervalSince(t0), memoryString()))
        for spec in loras {
            let parts = spec.split(separator: ":", maxSplits: 1).map(String.init)
            let strength = parts.count > 1 ? Float(parts[1]) ?? 1 : 1
            let (m, report) = try pipe.applyingLoRA(URL(fileURLWithPath: parts[0]), strength: strength)
            pipe = m as! QwenImage21Pipeline
            log("LoRA \(report) at strength \(strength)")
        }

        var req = QwenImage21Pipeline.Request(prompt: prompt, width: w, height: h, seed: seed, steps: steps)
        req.images = try images.map { ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: $0))) }
        req.sizeFromImage = size == nil
        var snapped = pipe.snapSize(width: w, height: h)
        if let first = req.images.last, size == nil {
            let d = QwenImage21Pipeline.dimensions(area: 1024 * 1024, aspect: Double(first.dim(1)) / Double(first.dim(0)))
            snapped = pipe.snapSize(width: d.width, height: d.height)
        }
        let outURL = URL(fileURLWithPath: out)
        let t1 = Date()
        let image = try pipe.generate(req) { step in
            log(String(format: "step %d/%d  sigma %.3f  %.2fs", step.index + 1, step.count, step.sigma,
                       Date().timeIntervalSince(t1)))
            if saveSteps {
                let img = pipe.decode(step.predicted, width: snapped.width, height: snapped.height)
                let u = outURL.deletingPathExtension().appendingPathExtension("step\(step.index + 1).png")
                try? ImageArray.savePNG(img, to: u)
            }
        }
        log(String(format: "generated in %.1fs (%@)", Date().timeIntervalSince(t1), memoryString()))
        try ImageArray.savePNG(image, to: outURL, text: [
            "mlxcomfy.prompt": prompt, "mlxcomfy.seed": String(seed), "mlxcomfy.model": "Qwen-Image-2.1",
        ])
        log("wrote \(outURL.path)")
    }
}

struct VAERoundTrip: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vae-roundtrip", abstract: "Encode and decode an image through the VAE.")

    @Option var model = "qwen-turbo-mlx"
    @Option var input: String
    @Option var out = "roundtrip.png"
    @Option(help: "Resize to this size (multiple of 32), e.g. 768x1024.") var size: String?
    @Option(help: "Tile the encoder and decoder: auto, on or off.") var tiled = "off"
    @Option(help: "Tile size in latent cells (16 px).") var tile = 64

    func run() async throws {
        let vae = try QwenImage21VAE(directory: ModelPaths.resolve(model).appendingPathComponent("vae"))
        let cg = try ImageArray.load(URL(fileURLWithPath: input))
        var w = cg.width / 32 * 32, h = cg.height / 32 * 32
        if let size { (w, h) = try parseSize(size) }
        let x = ImageArray.array(from: cg, width: w, height: h)
        let t = Date()
        let input = (x * 2 - 1).expandedDimensions(axis: 0)
        let tiles = QwenImage21Pipeline.shouldTile(VAETiling(rawValue: tiled) ?? .off, cells: (w / 16) * (h / 16))
        Memory.peakMemory = 0
        let z = tiles ? vae.encodeTiled(input, tile: tile) : vae.encode(input)
        eval(z)
        log("latents \(z.shape)  mean \(z.mean().item(Float.self))  std \(sqrt(z.variance().item(Float.self)))")
        let y = ((tiles ? vae.decodeTiled(normalizedLatents: z, tile: tile) : vae.decode(normalizedLatents: z))[0].asType(.float32) + 1) / 2
        eval(y)
        let diff = abs(y[0..., 0..., ..<3] - x[0..., 0..., ..<3])
        let mse = (diff * diff).mean().item(Float.self)
        log(String(format: "%.1fs  mean abs err %.4f  PSNR %.2f dB  (%@)", Date().timeIntervalSince(t),
                   diff.mean().item(Float.self), 10 * log10(1 / mse), memoryString()))
        try ImageArray.savePNG(y, to: URL(fileURLWithPath: out))
        log("wrote \(out)")
    }
}

struct Compare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Compare the pixels of two images.")
    @Argument var a: String
    @Argument var b: String
    @Option(help: "Only compare pixels where this mask image is black.") var outsideMask: String?

    func run() async throws {
        let x = ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: a)))
        let y = ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: b)))
        guard x.shape == y.shape else { print("different sizes: \(x.shape) vs \(y.shape)"); throw ExitCode(1) }
        var d = abs(x - y)
        if let outsideMask {
            let m = ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: outsideMask)))
            let keep = (try ImageArray.resized(m, width: x.dim(1), height: x.dim(0))[0..., 0..., ..<1] .< 0.01)
            d = MLX.where(keep, d, MLXArray(Float(0)))
            let inside = MLX.where(keep, MLXArray(Float(0)), abs(x - y))
            print(String(format: "inside mask: mean diff %.1f/255", inside.sum().item(Float.self) * 255
                / max(1, (1 - keep.asType(.float32)).sum().item(Float.self) * Float(x.dim(2)))))
        }
        let maxDiff = d.max().item(Float.self) * 255
        let differing = (d .> 0).asType(.float32).mean().item(Float.self) * 100
        print(String(format: "max diff %.0f/255, %.2f%% of values differ", maxDiff, differing))
        if maxDiff > 0 { throw ExitCode(1) }
        print("identical")
    }
}

struct FitPreview: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "fit-preview",
        abstract: "Fit a linear latent→RGB projection for fast step previews (prints Swift constants).")
    @Option var model = "qwen-turbo-mlx"
    @Argument var images: [String]

    func run() async throws {
        let vae = try QwenImage21VAE(directory: ModelPaths.resolve(model).appendingPathComponent("vae"))
        var xs: [MLXArray] = [], ys: [MLXArray] = []
        for path in images {
            let cg = try ImageArray.load(URL(fileURLWithPath: path))
            let scale = min(1, 1024 / Double(max(cg.width, cg.height)))
            let w = Int(Double(cg.width) * scale) / 32 * 32, h = Int(Double(cg.height) * scale) / 32 * 32
            let img = ImageArray.array(from: cg, width: w, height: h)
            let z = vae.encode((img * 2 - 1).expandedDimensions(axis: 0))[0]  // [h/16, w/16, 64]
            let lh = h / 16, lw = w / 16
            let rgb = img[0..., 0..., ..<3].reshaped(lh, 16, lw, 16, 3).mean(axes: [1, 3])
            xs.append(z.reshaped(-1, 64).asType(.float32))
            ys.append(rgb.reshaped(-1, 3))
            log("\(path): \(lh)x\(lw) latents")
        }
        let x = concatenated(xs, axis: 0)
        let X = concatenated([x, MLXArray.ones([x.dim(0), 1])], axis: 1)
        let Y = concatenated(ys, axis: 0)
        let xtx = matmul(X.T, X) + 1e-3 * MLXArray.eye(65)
        let W = matmul(MLXLinalg.inv(xtx, stream: .cpu), matmul(X.T, Y))  // [65, 3]
        let err = abs(matmul(X, W) - Y).mean().item(Float.self)
        log(String(format: "mean abs error %.4f (%.1f/255) over %d samples", err, err * 255, X.dim(0)))
        let vals = W.asArray(Float.self)
        print("static let previewWeights: [Float] = [  // 64 rows × RGB, then bias")
        for r in 0..<65 {
            print("    " + (0..<3).map { String(format: "%.6f", vals[r * 3 + $0]) }.joined(separator: ", ") + ",")
        }
        print("]")
    }
}

struct MakeTestLoRA: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "make-test-lora",
        abstract: "Write a synthetic PEFT-format LoRA over every block's attention projections (for testing).")
    @Option var out = "outputs/test-lora.safetensors"
    @Option var rank = 8
    @Option var layers = 32
    @Option(help: "Std-dev of the up matrices; 0 gives a no-op LoRA (like a freshly initialised one).") var upScale: Float = 0
    @Option var seed: UInt64 = 0

    func run() async throws {
        var arrays: [String: MLXArray] = [:]
        var key = MLXRandom.key(seed)
        for i in 0..<layers {
            for t in ["to_q", "to_k", "to_v", "to_out.0"] {
                let name = "transformer.transformer_blocks.\(i).attn.\(t)"
                let ks = MLXRandom.split(key: key, into: 3)
                key = ks[0]
                arrays[name + ".lora_A.weight"] = (MLXRandom.normal([rank, 4096], key: ks[1]) / sqrt(Float(4096)))
                    .asType(.bfloat16)
                arrays[name + ".lora_B.weight"] = (MLXRandom.normal([4096, rank], key: ks[2]) * upScale).asType(.bfloat16)
            }
        }
        try save(arrays: arrays, url: URL(fileURLWithPath: out))
        print("wrote \(out) (\(arrays.count) tensors)")
    }
}

struct Tokenize: AsyncParsableCommand {
    @Option var model = "qwen-turbo-mlx"
    @Argument var text: String

    func run() async throws {
        let url = ModelPaths.resolve(model).appendingPathComponent("processor")
        let tok = try await AutoTokenizer.from(modelFolder: url)
        print(tok.encode(text: text, addSpecialTokens: false))
    }
}

struct Assist: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Prompt assistant: enhance a prompt or describe an image with the model's language model.")

    @Option var model = "qwen-turbo-mlx"
    @Option(help: "The prompt to enhance, or extra guidance when describing.") var prompt = ""
    @Option(help: "Image to describe, or the image an edit instruction refers to (repeatable).") var image: [String] = []
    @Option(help: "enhance, describe or custom.") var mode = "enhance"
    @Option(help: "System instruction for custom mode.") var instruction = ""
    @Option var seed: UInt64 = 0
    @Option(help: "0 = greedy.") var temperature: Float = 0.7
    @Option var maxTokens = 320

    func run() async throws {
        guard let m = PromptAssistant.Mode(rawValue: mode) else { throw ValidationError("unknown mode \(mode)") }
        let t0 = Date()
        let pipe = try await QwenImage21Pipeline(directory: ModelPaths.resolve(model))
        log(String(format: "loaded in %.1fs (%@)", Date().timeIntervalSince(t0), memoryString()))
        let images = try image.map { ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: $0))) }
        let (system, user) = PromptAssistant.messages(mode: m, prompt: prompt, hasImage: !images.isEmpty,
                                                      instruction: instruction)
        let t = Date()
        var first: Date?
        var count = 0
        let text = try pipe.generateText(system: system, user: user, images: images,
                                         options: TextGenerationOptions(maxTokens: maxTokens,
                                                                        temperature: temperature, seed: seed)) { _ in
            if first == nil { first = Date() }
            count += 1
            return true
        }
        let decodeTime = first.map { Date().timeIntervalSince($0) } ?? 0
        log(String(format: "%d tokens, first after %.1fs, %.1f tok/s (%@)", count,
                   (first ?? Date()).timeIntervalSince(t), Double(max(count - 1, 0)) / max(decodeTime, 1e-3),
                   memoryString()))
        print(PromptAssistant.clean(text))
    }
}
