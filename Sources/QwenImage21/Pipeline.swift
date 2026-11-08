import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

/// Qwen-Image 2.1 (port of diffusers' `QwenImage21Pipeline`, KV-cache path), exposed as a ``DiffusionModel``.
public final class QwenImage21Pipeline: DiffusionModel {
    public struct Request {
        public var prompt: String
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
    public var displayName: String { "Qwen-Image 2.1 (\(directory.lastPathComponent))" }

    public let directory: URL
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

    // MARK: DiffusionModel

    public func tokens(for prompt: String) -> [Int] {
        let p = prompt.isEmpty ? " " : prompt
        let text = "<|im_start|>system\n\(Self.systemPrompt)<|im_end|>\n<|im_start|>user\n\(p)<|im_end|>\n"
            + "<|im_start|>assistant\n"
        return tokenizer.encode(text: text, addSpecialTokens: false)
    }

    public func encodePrompt(_ prompt: String) throws -> Conditioning {
        let h = textEncoder.hiddenStates(tokens: tokens(for: prompt))
        let embeds = h[0..., dropCount..., 0...]
        eval(embeds)
        let prefix = transformer.encodePrefix([.text(embeds)])
        return Conditioning(family: family, payload: PromptPayload(embeddings: embeds, prefix: prefix))
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

    public func decode(_ latents: MLXArray, width: Int, height: Int) -> MLXArray {
        let (h, w) = latentSize(width: width, height: height)
        let img = vae.decode(normalizedLatents: latents.reshaped(1, h, w, 64))
        let out = (img[0].asType(.float32) + 1) / 2
        eval(out)
        return out
    }

    public func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray? {
        let (h, w) = latentSize(width: width, height: height)
        return QwenImage21LatentPreview.rgb(latents, h: h, w: w)
    }

    // MARK: convenience

    /// Prompt → RGBA `[H, W, 4]` in `[0, 1]`, through the same ``DiffusionModel/sample`` loop the graph uses.
    public func generate(_ req: Request, onStep: ((SamplerStep) throws -> Void)? = nil) throws -> MLXArray {
        let cond = try encodePrompt(req.prompt)
        let size = snapSize(width: req.width, height: req.height)
        let out = try sample(conditioning: cond, latent: Latent(width: size.width, height: size.height),
                             seed: req.seed, steps: req.steps, onStep: onStep)
        return decode(out.data!, width: size.width, height: size.height)
    }
}
