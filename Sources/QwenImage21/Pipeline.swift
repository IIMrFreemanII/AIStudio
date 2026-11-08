import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

/// Qwen-Image 2.1 (port of diffusers' `QwenImage21Pipeline`, KV-cache path), exposed as a ``DiffusionModel``.
public final class QwenImage21Pipeline: DiffusionModel {
    public struct Request {
        public var prompt: String
        /// Condition images (`[H, W, 4]` RGBA in `[0, 1]`) for editing; the output takes the last one's aspect
        /// ratio unless `width`/`height` are set explicitly.
        public var images: [MLXArray] = []
        public var sizeFromImage = true
        public var width = 1024
        public var height = 1024
        public var seed: UInt64 = 0
        /// `nil` → the checkpoint's `sample_sigmas` (or 40 linear steps if it has none).
        public var steps: Int? = nil

        public init(prompt: String, width: Int = 1024, height: Int = 1024, seed: UInt64 = 0, steps: Int? = nil) {
            self.prompt = prompt
            self.width = width
            self.height = height
            self.seed = seed
            self.steps = steps
        }
    }

    /// What ``encodePrompt(_:)`` returns inside its ``Conditioning``.
    public struct PromptPayload {
        public let embeddings: MLXArray
        public let prefix: QwenImage21Transformer.PrefixCache
    }

    public static let className = "QwenImage21Pipeline"
    public static let systemPrompt = "Comprehend and analyze the provided prompt."
    public static let vaeScale = 16
    public let family = "qwen-image-2.1"
    public var displayName: String {
        "Qwen-Image 2.1 (\(directory.lastPathComponent))" + (loras.isEmpty ? "" : " + \(loras.count) LoRA")
    }

    public let directory: URL
    /// LoRAs applied on top of the base weights, in order.
    public private(set) var loras: [LoRAReport] = []
    public let tokenizer: Tokenizer
    public let textEncoder: Qwen3TextEncoder
    public let transformer: QwenImage21Transformer
    public let vae: QwenImage21VAE
    public let sampleSigmas: [Float]?
    let schedulerTemplate: FlowMatchEuler
    let dropCount: Int

    /// Registers this pipeline's loader for its `model_index.json` class name.
    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await QwenImage21Pipeline(directory: url) }
    }

    public init(directory: URL, dequantizeTransformer: Bool = false,
                log: (String) -> Void = { _ in }) async throws {
        self.directory = directory
        let index = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("model_index.json"))) as? [String: Any] ?? [:]
        sampleSigmas = (index["sample_sigmas"] as? [Double])?.map(Float.init)

        var t = Date()
        tokenizer = try await AutoTokenizer.from(modelFolder: directory.appendingPathComponent("processor"))
        let sys = "<|im_start|>system\n\(Self.systemPrompt)<|im_end|>\n"
        dropCount = tokenizer.encode(text: sys, addSpecialTokens: false).count
        log(String(format: "tokenizer %.1fs (system prefix %d tokens)", Date().timeIntervalSince(t), dropCount))

        t = Date()
        textEncoder = try Qwen3TextEncoder(directory: directory.appendingPathComponent("text_encoder"))
        log(String(format: "text encoder %.1fs", Date().timeIntervalSince(t)))
        t = Date()
        transformer = try QwenImage21Transformer(directory: directory.appendingPathComponent("transformer"),
                                                 dequantize: dequantizeTransformer)
        log(String(format: "transformer %.1fs", Date().timeIntervalSince(t)))
        t = Date()
        vae = try QwenImage21VAE(directory: directory.appendingPathComponent("vae"))
        log(String(format: "vae %.1fs", Date().timeIntervalSince(t)))
        schedulerTemplate = try FlowMatchEuler(
            configURL: directory.appendingPathComponent("scheduler/scheduler_config.json"))
    }

    private init(copying o: QwenImage21Pipeline, transformer: QwenImage21Transformer, loras: [LoRAReport]) {
        directory = o.directory; tokenizer = o.tokenizer; textEncoder = o.textEncoder; vae = o.vae
        sampleSigmas = o.sampleSigmas; schedulerTemplate = o.schedulerTemplate; dropCount = o.dropCount
        self.transformer = transformer
        self.loras = loras
    }

    public func applyingLoRA(_ lora: URL, strength: Float) throws -> (model: DiffusionModel, report: LoRAReport) {
        let file = try LoRAFile(url: lora, moduleNames: transformer.linearNames)
        let (t, matched) = transformer.adding(file.adapters(strength: strength))
        let report = LoRAReport(file: lora.lastPathComponent, matchedLayers: matched,
                                unmatchedKeys: file.unmatchedKeys.count)
        return (QwenImage21Pipeline(copying: self, transformer: t, loras: loras + [report]), report)
    }

    // MARK: DiffusionModel

    public var maxConditionImages: Int { 4 }
    static let imagePadToken = 151655

    public func tokens(for prompt: String, imageCount: Int = 0) -> [Int] {
        let p = prompt.isEmpty ? " " : prompt
        var vision = ""
        for i in 0..<imageCount {
            vision += (i == 0 ? "" : " ") + "<image\(i + 1)><|vision_start|><|image_pad|><|vision_end|>"
        }
        let text = "<|im_start|>system\n\(Self.systemPrompt)<|im_end|>\n<|im_start|>user\n\(vision)\(p)<|im_end|>\n"
            + "<|im_start|>assistant\n"
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// `calculate_dimensions`: the size with `area` pixels and the given aspect ratio, rounded to multiples of 32.
    public static func dimensions(area: Int, aspect: Double) -> (width: Int, height: Int) {
        let w = (Double(area) * aspect).squareRoot()
        let h = w / aspect
        return (max(32, Int((w / 32).rounded()) * 32), max(32, Int((h / 32).rounded()) * 32))
    }

    /// A condition image prepared for both encoders.
    struct PreparedImage {
        let vision: Qwen3VisionTower.Output
        let latents: MLXArray  // [1, h*w, 64] normalised
        let latentH: Int
        let latentW: Int
        let width: Int
        let height: Int
    }

    /// A condition image resized to ~1 MP (same aspect, multiples of 32): RGBA, and RGB with alpha over white.
    func conditioningRGB(_ image: MLXArray) throws -> (rgba: MLXArray, rgb: MLXArray, width: Int, height: Int) {
        let (w, h) = Self.dimensions(area: 1024 * 1024, aspect: Double(image.dim(1)) / Double(image.dim(0)))
        let rgba: MLXArray
        if image.dim(0) == h && image.dim(1) == w && image.dim(2) == 4 {
            rgba = image.asType(.float32)
        } else {
            rgba = ImageArray.array(from: try ImageArray.cgImage(from: image, keepAlpha: true), width: w, height: h)
        }
        let alpha = rgba[0..., 0..., 3...]
        return (rgba, rgba[0..., 0..., ..<3] * alpha + (1 - alpha), w, h)
    }

    /// Resizes to ~1 MP (same aspect), then runs the vision tower (alpha over white) and the VAE (RGBA).
    func prepare(_ image: MLXArray) throws -> PreparedImage {
        let (rgba, rgb, w, h) = try conditioningRGB(image)
        let vision = textEncoder.vision(rgb)
        let z = vae.encode((rgba * 2 - 1).expandedDimensions(axis: 0))  // [1, h/16, w/16, 64]
        let lh = h / Self.vaeScale, lw = w / Self.vaeScale
        let latents = z.reshaped(1, lh * lw, 64)
        eval(latents)
        return PreparedImage(vision: vision, latents: latents, latentH: lh, latentW: lw, width: w, height: h)
    }

    public func encodePrompt(_ prompt: String, images: [MLXArray]) throws -> Conditioning {
        guard images.count <= maxConditionImages else { throw DiffusionError.tooManyImages(max: maxConditionImages) }
        if images.isEmpty {
            let h = textEncoder.hiddenStates(tokens: tokens(for: prompt))
            let embeds = h[0..., dropCount..., 0...]
            eval(embeds)
            let prefix = transformer.encodePrefix([.text(embeds)])
            return Conditioning(family: family, payload: PromptPayload(embeddings: embeds, prefix: prefix))
        }

        let prepared = try images.map(prepare)
        let (ids, visual) = Self.expandImagePads(tokens(for: prompt, imageCount: images.count),
                                                 visions: prepared.map(\.vision))
        let runs = visual.runs
        let hidden = textEncoder.hiddenStates(tokens: ids, visual: visual)
        eval(hidden)

        // Transformer prefix: encoder features for text, VAE latents in place of each image's slots.
        var segments: [QwenImage21Transformer.Segment] = []
        var pos = dropCount
        for (i, run) in runs.enumerated() {
            if run.start > pos { segments.append(.text(hidden[0..., pos ..< run.start, 0...])) }
            let p = prepared[i]
            segments.append(.image(p.latents, height: p.latentH, width: p.latentW))
            pos = run.start + run.count
        }
        if pos < ids.count { segments.append(.text(hidden[0..., pos..., 0...])) }
        let prefix = transformer.encodePrefix(segments)
        let embeds = hidden[0..., dropCount..., 0...]
        let last = prepared.last!
        return Conditioning(family: family, payload: PromptPayload(embeddings: embeds, prefix: prefix),
                            suggestedSize: (last.width, last.height))
    }

    /// Expands each `<|image_pad|>` to its image's merged-token count and lays out the mRoPE positions
    /// (`t` constant over an image, `h`/`w` its grid; text after it continues from the largest position + 1).
    static func expandImagePads(_ tokens: [Int], visions: [Qwen3VisionTower.Output])
        -> (ids: [Int], visual: Qwen3TextEncoder.VisualInputs)
    {
        var ids: [Int] = []
        var runs: [(start: Int, count: Int)] = []
        var positions: [(Int32, Int32, Int32)] = []
        var cursor: Int32 = 0
        var next = 0
        for t in tokens {
            if t == imagePadToken, next < visions.count {
                let v = visions[next]
                runs.append((ids.count, v.gridH * v.gridW))
                for r in 0..<v.gridH {
                    for c in 0..<v.gridW { positions.append((cursor, cursor + Int32(r), cursor + Int32(c))) }
                }
                ids += Array(repeating: t, count: v.gridH * v.gridW)
                cursor += Int32(max(v.gridH, v.gridW))
                next += 1
            } else {
                ids.append(t)
                positions.append((cursor, cursor, cursor))
                cursor += 1
            }
        }
        let visual = Qwen3TextEncoder.VisualInputs(
            runs: runs,
            embeddings: visions.isEmpty ? MLXArray.zeros([0, 1]) : concatenated(visions.map(\.embeddings), axis: 0),
            deepstack: (0 ..< (visions.first?.deepstack.count ?? 0)).map { k in
                concatenated(visions.map { $0.deepstack[k] }, axis: 0)
            },
            positions: positions)
        return (ids, visual)
    }

    public func snapSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let m = Self.vaeScale * 2
        return (max(m, width / m * m), max(m, height / m * m))
    }

    func latentSize(width: Int, height: Int) -> (h: Int, w: Int) {
        let s = snapSize(width: width, height: height)
        return (s.height / Self.vaeScale, s.width / Self.vaeScale)
    }

    public func noise(seed: UInt64, width: Int, height: Int) -> MLXArray {
        let (h, w) = latentSize(width: width, height: height)
        let key = MLXRandom.key(seed)
        // Same layout as the reference: [B, 1, C, H, W] noise packed to [B, H*W, C].
        let n = MLXRandom.normal([1, 64, h, w], key: key)
        return n.reshaped(1, 64, h * w).transposed(0, 2, 1)
    }

    public func makeScheduler(steps: Int?) -> FlowMatchEuler {
        var s = schedulerTemplate
        if let steps { s.setSteps(steps) } else if let sampleSigmas { s.setSigmas(sampleSigmas) } else { s.setSteps(40) }
        return s
    }

    public func velocity(_ latents: MLXArray, conditioning: Conditioning, sigma: Float, width: Int,
                         height: Int) throws -> MLXArray {
        guard let p = conditioning.payload as? PromptPayload else {
            throw DiffusionError.wrongConditioning(expected: family, got: conditioning.family)
        }
        let (h, w) = latentSize(width: width, height: height)
        return transformer.denoise(latents: latents, prefix: p.prefix, sigma: sigma, latentHeight: h, latentWidth: w)
    }

    /// `auto` tiles the VAE above this many latent cells (about 1.25 MP). Whole-image VAE passes peak at about
    /// 16 GB at 1536² and 28 GB at 2048², on top of the resident models; tiles of 64 cells stay near 8 GB at any size.
    /// Everything up to the standard presets (1024², 896×1184, 768×1344) is processed whole, exactly as before.
    public static var autoTileCells = 80 * 80

    public static func shouldTile(_ tiling: VAETiling, cells: Int) -> Bool {
        switch tiling {
        case .on: return true
        case .off: return false
        case .auto: return cells > autoTileCells
        }
    }

    public func decode(_ latents: MLXArray, width: Int, height: Int, tiling: VAETiling) -> MLXArray {
        let (h, w) = latentSize(width: width, height: height)
        let z = latents.reshaped(1, h, w, 64)
        let img = Self.shouldTile(tiling, cells: h * w) ? vae.decodeTiled(normalizedLatents: z)
            : vae.decode(normalizedLatents: z)
        let out = (img[0].asType(.float32) + 1) / 2
        eval(out)
        return out
    }

    public var latentScale: Int { Self.vaeScale }

    public func encodeImage(_ image: MLXArray, tiling: VAETiling) -> MLXArray {
        let (h, w) = (image.dim(0) / Self.vaeScale, image.dim(1) / Self.vaeScale)
        let input = (image.asType(.float32) * 2 - 1).expandedDimensions(axis: 0)
        let z = Self.shouldTile(tiling, cells: h * w) ? vae.encodeTiled(input) : vae.encode(input)
        let packed = z.reshaped(1, h * w, 64)
        eval(packed)
        return packed
    }

    public func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray? {
        let (h, w) = latentSize(width: width, height: height)
        return QwenImage21LatentPreview.rgb(latents, h: h, w: w)
    }

    // MARK: convenience

    /// Prompt → RGBA `[H, W, 4]` in `[0, 1]`, through the same ``DiffusionModel/sample`` loop the graph uses.
    public func generate(_ req: Request, onStep: ((SamplerStep) throws -> Void)? = nil) throws -> MLXArray {
        let cond = try encodePrompt(req.prompt, images: req.images)
        var size = snapSize(width: req.width, height: req.height)
        if req.sizeFromImage, let s = cond.suggestedSize { size = snapSize(width: s.width, height: s.height) }
        let out = try sample(conditioning: cond, latent: Latent(width: size.width, height: size.height),
                             seed: req.seed, steps: req.steps, onStep: onStep)
        return decode(out.data!, width: size.width, height: size.height)
    }
}
