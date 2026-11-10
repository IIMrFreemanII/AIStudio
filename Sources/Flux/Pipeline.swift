import CLIPText
import FluxTransformer
import FluxVAE
import Foundation
import MLX
import MLXDiffusionCore
import T5Text
import Tokenizers

/// FLUX.1 [dev] (port of diffusers' `FluxPipeline` / `FluxImg2ImgPipeline` / `FluxInpaintPipeline` behaviour),
/// exposed as a ``DiffusionModel``: text-to-image, image-to-image (denoise < 1) and inpainting through the generic
/// sampling loop.
///
/// Loads the official diffusers folder. The bf16 transformer is quantized to 8-bit once and cached in
/// `.mlxcomfy-cache/`; a transformer stored already quantized (imported fine-tunes, see `ModelImport`) is used as
/// is. The T5-XXL encoder (9.5 GB) is loaded only while a prompt is encoded.
public final class FluxPipeline: DiffusionModel {
    /// What ``encodePrompt(_:images:)`` returns inside its ``Conditioning``.
    public struct PromptPayload {
        /// T5 hidden states `[1, 512, 4096]` (float32).
        public let context: MLXArray
        /// Pooled CLIP output `[1, 768]` (float32).
        public let pooled: MLXArray
    }

    public static let className = "FluxPipeline"
    public static let defaultSteps = 28
    public static let t5Length = 512
    public let family = "flux.1"
    public var displayName: String {
        "FLUX.1 (\(directory.lastPathComponent))" + (loras.isEmpty ? "" : " + \(loras.count) LoRA")
    }

    public let directory: URL
    public private(set) var loras: [LoRAReport] = []
    public let clipTokenizer: CLIPTokenizer
    public let clip: CLIPTextEncoder
    public let t5Tokenizer: Tokenizer
    public let transformer: FluxTransformer
    public let vae: FluxVAE
    let schedulerTemplate: FlowMatchEuler
    private let ropeLock = NSLock()
    private var ropeCache: [String: (MLXArray, MLXArray)] = [:]

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await FluxPipeline(directory: url) }
    }

    /// The quantized transformer cache inside a model folder.
    public static func transformerCacheURL(_ directory: URL, bits: Int = 8) -> URL {
        directory.appendingPathComponent(".mlxcomfy-cache/transformer-q\(bits).safetensors")
    }

    /// The transformer weights of a model folder: a pre-quantized `transformer/` as stored, otherwise the dense one
    /// quantized to `quantizeBits` through the cache (or as stored when `quantizeBits` is `nil`).
    public static func transformerStore(directory: URL, quantizeBits: Int? = 8,
                                        log: (String) -> Void = { _ in }) throws -> WeightStore {
        let tDir = directory.appendingPathComponent("transformer")
        guard let bits = quantizeBits else { return try WeightStore(directory: tDir) }
        return try WeightStore.quantizedCache(source: tDir, cacheURL: transformerCacheURL(directory, bits: bits),
                                              bits: bits, log: log)
    }

    /// - Parameter quantizeBits: 8 (default, cached) or `nil` to run the bf16 weights as stored (about 24 GB).
    public init(directory: URL, quantizeBits: Int? = 8, log: (String) -> Void = { _ in }) async throws {
        self.directory = directory
        var t = Date()
        clipTokenizer = try CLIPTokenizer(directory: directory.appendingPathComponent("tokenizer"))
        let clipDir = directory.appendingPathComponent("text_encoder")
        clip = try CLIPTextEncoder(store: WeightStore(directory: clipDir),
                                   config: CLIPTextEncoder.Config(url: clipDir.appendingPathComponent("config.json")))
        t5Tokenizer = try await AutoTokenizer.from(modelFolder: directory.appendingPathComponent("tokenizer_2"))
        log(String(format: "text encoders %.1fs", Date().timeIntervalSince(t)))

        t = Date()
        let tDir = directory.appendingPathComponent("transformer")
        let store = try Self.transformerStore(directory: directory, quantizeBits: quantizeBits, log: log)
        transformer = try FluxTransformer(store: store,
                                          config: FluxTransformer.Config(url: tDir.appendingPathComponent("config.json")))
        log(String(format: "transformer %.1fs", Date().timeIntervalSince(t)))
        vae = try FluxVAE(directory: directory.appendingPathComponent("vae"))
        schedulerTemplate = try FlowMatchEuler(
            configURL: directory.appendingPathComponent("scheduler/scheduler_config.json"))
    }

    private init(copying o: FluxPipeline, transformer: FluxTransformer, loras: [LoRAReport]) {
        directory = o.directory; clipTokenizer = o.clipTokenizer; clip = o.clip; t5Tokenizer = o.t5Tokenizer
        vae = o.vae; schedulerTemplate = o.schedulerTemplate
        self.transformer = transformer
        self.loras = loras
    }

    public func applyingLoRA(_ lora: URL, strength: Float) throws -> (model: DiffusionModel, report: LoRAReport) {
        let file = try transformer.loraFile(lora)
        let (t, matched) = transformer.adding(file.adapters(strength: strength))
        let report = LoRAReport(file: lora.lastPathComponent, matchedLayers: matched,
                                unmatchedKeys: file.unmatchedKeys.count)
        return (FluxPipeline(copying: self, transformer: t, loras: loras + [report]), report)
    }

    // MARK: prompt

    /// T5 ids: the prompt with `</s>`, truncated and padded with `<pad>` (0) to 512, as the reference pipeline does
    /// (the transformer attends to the padding too, so it is part of the conditioning).
    public func t5Tokens(_ prompt: String) -> [Int] { Self.t5Tokens(t5Tokenizer, prompt) }

    public static func t5Tokens(_ tokenizer: Tokenizer, _ prompt: String) -> [Int] {
        var ids = tokenizer.encode(text: prompt)
        if ids.count > Self.t5Length { ids = Array(ids.prefix(Self.t5Length - 1)) + [ids.last!] }
        return ids + Array(repeating: 0, count: Self.t5Length - ids.count)
    }

    /// T5 hidden states `[1, 512, 4096]` for each prompt. Loads the encoder for the call and frees it afterwards.
    public func t5Encode(_ prompts: [String]) throws -> [MLXArray] {
        var cfg = T5XXLEncoder.Config()
        cfg.vocabSize = 32_128
        cfg.sharedPositionBias = true
        let enc = try T5XXLEncoder(store: WeightStore(directory: directory.appendingPathComponent("text_encoder_2")),
                                   config: cfg, layout: .huggingFace)
        defer { Memory.clearCache() }
        return prompts.map { p in
            let ids = t5Tokens(p)
            let h = enc(MLXArray(ids.map { Int32($0) }, [1, ids.count]))
            eval(h)
            return h
        }
    }

    /// Pooled CLIP-L output `[1, 768]`.
    public func clipPooled(_ prompt: String) -> MLXArray {
        let p = clip.pooled(clipTokenizer.encode(prompt), eos: clipTokenizer.eos)
        eval(p)
        return p
    }

    public var maxConditionImages: Int { 0 }

    public func encodePrompt(_ prompt: String, images: [MLXArray]) throws -> Conditioning {
        guard images.isEmpty else { throw DiffusionError.tooManyImages(max: 0) }
        let context = try t5Encode([prompt])[0]
        return Conditioning(family: family, payload: PromptPayload(context: context, pooled: clipPooled(prompt)))
    }

    // MARK: sampling

    public static let vaeScale = 16  // pixels per packed latent token

    public func snapSize(width: Int, height: Int) -> (width: Int, height: Int) {
        let m = Self.vaeScale
        return (max(4 * m, width / m * m), max(4 * m, height / m * m))
    }

    func latentSize(width: Int, height: Int) -> (h: Int, w: Int) {
        let s = snapSize(width: width, height: height)
        return (s.height / Self.vaeScale, s.width / Self.vaeScale)
    }

    public func noise(seed: UInt64, width: Int, height: Int) -> MLXArray {
        let (h, w) = latentSize(width: width, height: height)
        return MLXRandom.normal([1, h * w, 64], key: MLXRandom.key(seed))
    }

    public func makeScheduler(steps: Int?) -> FlowMatchEuler {
        makeScheduler(steps: steps, width: 1024, height: 1024)
    }

    public func makeScheduler(steps: Int?, width: Int, height: Int) -> FlowMatchEuler {
        let (h, w) = latentSize(width: width, height: height)
        var s = schedulerTemplate
        s.setSteps(steps ?? Self.defaultSteps, imageTokens: h * w)
        return s
    }

    public var defaultGuidance: Float? { 3.5 }

    func rope(textLength: Int, h: Int, w: Int) -> (MLXArray, MLXArray) {
        let key = "\(textLength)x\(h)x\(w)"
        ropeLock.lock(); defer { ropeLock.unlock() }
        if let r = ropeCache[key] { return r }
        let r = transformer.rope(textLength: textLength, height: h, width: w)
        eval(r.0, r.1)
        ropeCache = [key: r]
        return r
    }

    public func velocity(_ latents: MLXArray, conditioning: Conditioning, sigma: Float, width: Int,
                         height: Int) throws -> MLXArray {
        try velocity(latents, conditioning: conditioning, sigma: sigma, width: width, height: height, guidance: nil)
    }

    public func velocity(_ latents: MLXArray, conditioning: Conditioning, sigma: Float, width: Int, height: Int,
                         guidance: Float?) throws -> MLXArray {
        guard let p = conditioning.payload as? PromptPayload else {
            throw DiffusionError.wrongConditioning(expected: family, got: conditioning.family)
        }
        let (h, w) = latentSize(width: width, height: height)
        let vec = transformer.conditioningVector(sigma: sigma, guidance: guidance ?? defaultGuidance!, pooled: p.pooled)
        return transformer(latents, context: p.context, vec: vec, rope: rope(textLength: p.context.dim(1), h: h, w: w))
    }

    // MARK: VAE

    /// Packed `[1, h·w, 64]` (channels `c·4 + dy·2 + dx`, as diffusers' `_pack_latents`) ↔ `[1, 2h, 2w, 16]`.
    public static func unpack(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        x.reshaped(1, h, w, 16, 2, 2).transposed(0, 1, 4, 2, 5, 3).reshaped(1, 2 * h, 2 * w, 16)
    }

    public static func pack(_ z: MLXArray) -> MLXArray {
        let (h, w) = (z.dim(1) / 2, z.dim(2) / 2)
        return z.reshaped(1, h, 2, w, 2, 16).transposed(0, 1, 3, 5, 2, 4).reshaped(1, h * w, 64)
    }

    public func decode(_ latents: MLXArray, width: Int, height: Int, tiling: VAETiling) -> MLXArray {
        let (h, w) = latentSize(width: width, height: height)
        let z = vae.denormalize(Self.unpack(latents.asType(.float32), h: h, w: w))
        let rgb = MLX.clip((vae.decode(z)[0] + 1) / 2, min: 0, max: 1)
        let out = concatenated([rgb, MLXArray.ones([rgb.dim(0), rgb.dim(1), 1])], axis: -1)
        eval(out)
        return out
    }

    public var latentScale: Int { Self.vaeScale }

    public func encodeImage(_ image: MLXArray, tiling: VAETiling) -> MLXArray {
        let rgb = image[0..., 0..., ..<3].asType(.float32)
        let z = vae.encode((rgb * 2 - 1).expandedDimensions(axis: 0))
        let packed = Self.pack(vae.normalize(z))
        eval(packed)
        return packed
    }

    public func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray? {
        let (h, w) = latentSize(width: width, height: height)
        return FluxLatentPreview.rgb(latents, h: h, w: w)
    }

    // MARK: convenience

    /// Prompt → RGBA `[H, W, 4]` in `[0, 1]`, through the same ``DiffusionModel/sample`` loop the graph uses.
    public func generate(prompt: String, width: Int = 1024, height: Int = 1024, seed: UInt64 = 0, steps: Int? = nil,
                         guidance: Float? = nil, onStep: ((SamplerStep) throws -> Void)? = nil) throws -> MLXArray {
        let cond = try encodePrompt(prompt, images: [])
        let size = snapSize(width: width, height: height)
        let out = try sample(conditioning: cond, latent: Latent(width: size.width, height: size.height), seed: seed,
                             steps: steps, guidance: guidance, onStep: onStep)
        return decode(out.data!, width: size.width, height: size.height)
    }
}
