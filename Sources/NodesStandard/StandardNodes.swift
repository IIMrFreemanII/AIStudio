import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore
import LTX2
import QwenImage21
import StableAudio

/// The built-in node library and the model loaders it relies on.
public enum StandardNodes {
    public static let all: [NodeImplementation.Type] = [
        LoadModelNode.self, TextEncodeNode.self, EmptyLatentNode.self, SamplerNode.self, VAEDecodeNode.self,
        SaveImageNode.self, PreviewImageNode.self, LoadImageNode.self,
        VAEEncodeNode.self, LoadMaskNode.self, SetLatentMaskNode.self, ImageCompositeMaskedNode.self,
        LoRANode.self,
        FeatherMaskNode.self, PadImageNode.self, ImageScaleNode.self, PromptAssistantNode.self,
        NoteNode.self, RerouteNode.self,
        TestPatternNode.self, TestToneNode.self, LoadVideoNode.self, LoadAudioNode.self, MixAudioNode.self,
        AddAudioToVideoNode.self, SaveVideoNode.self, SaveAudioNode.self, PreviewVideoNode.self, PreviewAudioNode.self,
        EmptyVideoLatentNode.self, VideoSamplerNode.self, VideoDecodeNode.self, TextToAudioNode.self,
    ]

    /// Registry with every standard node; also registers the bundled model loaders.
    public static func makeRegistry() -> NodeRegistry {
        QwenImage21Pipeline.register()
        LTX2Pipeline.register()
        StableAudioPipeline.register()
        let r = NodeRegistry()
        r.register(all)
        return r
    }

    /// Per-node peak-memory metrics from MLX.
    public static let mlxHooks = ExecutorHooks(
        resetPeakMemory: { Memory.peakMemory = 0 },
        peakMemory: { Memory.peakMemory })

    /// Graph key under which saved PNGs carry their workflow.
    public static let graphMetadataKey = "mlxcomfy.graph"
}

/// Sampler progress preview: the current clean-latent estimate plus what's needed to decode it.
public struct LatentPreview {
    public let model: DiffusionModel
    public let latents: MLXArray
    public let width: Int
    public let height: Int
}

func resolvePath(_ path: String) -> URL {
    let expanded = (path as NSString).expandingTildeInPath
    if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded).standardizedFileURL }
    return URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        .appendingPathComponent(expanded).standardizedFileURL
}

// MARK: - nodes

public struct LoadModelNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoadModel", displayName: "Load Model", category: "loaders",
        inputs: [InputSpec("path", .string, default: .string("qwen-turbo-mlx"), widget: .filePath)],
        outputs: [OutputSpec("model", .model)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let url = resolvePath(try inputs.string("path"))
        // Models live in the resource cache so they stay resident across runs and graph edits.
        let model: GenerativeModel = try await context.resources.get(
            "model:\(url.path)", estimatedBytes: ModelRegistry.weightBytes(at: url)) {
            try await ModelRegistry.shared.load(url)
        }
        return ["model": .object(model)]
    }
}

public struct TextEncodeNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "TextEncode", displayName: "Text Encode", category: "conditioning",
        inputs: [
            InputSpec("model", .model),
            InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true)),
            InputSpec("image", .image, optional: true),
            InputSpec("image2", .image, optional: true),
        ],
        outputs: [OutputSpec("conditioning", .conditioning)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let any: GenerativeModel = try inputs.object("model")
        if let video = any as? VideoDiffusionModel {
            return ["conditioning": .object(try video.encodeVideoPrompt(try inputs.string("prompt")))]
        }
        let model: DiffusionModel = try inputs.object("model")
        let images = ["image", "image2"].compactMap { inputs.optionalObject($0, as: MLXArray.self) }
        return ["conditioning": .object(try model.encodePrompt(try inputs.string("prompt"), images: images))]
    }
}

public struct EmptyLatentNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "EmptyLatent", displayName: "Empty Latent", category: "latent",
        inputs: [
            // 0 = take the size the conditioning suggests (the condition image's aspect ratio), else 1024.
            InputSpec("width", .int, default: .int(1024), widget: .number(min: 0, max: 2048, step: 32)),
            InputSpec("height", .int, default: .int(1024), widget: .number(min: 0, max: 2048, step: 32)),
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["latent": .object(Latent(width: try inputs.int("width"), height: try inputs.int("height")))]
    }
}

public struct SamplerNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "Sampler", displayName: "Sampler", category: "sampling",
        inputs: [
            InputSpec("model", .model),
            InputSpec("conditioning", .conditioning),
            InputSpec("latent", .latent),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 100, step: 1)),
            // < 1 with a VAE-encoded input latent: img2img strength. Inpainting (masked latent) uses 1.
            InputSpec("denoise", .float, default: .double(1), widget: .number(min: 0, max: 1, step: 0.05)),
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let cond: Conditioning = try inputs.object("conditioning")
        var latent: Latent = try inputs.object("latent")
        if latent.width <= 0 || latent.height <= 0 {
            let s = cond.suggestedSize ?? (1024, 1024)
            latent.width = s.width
            latent.height = s.height
        }
        let size = model.snapSize(width: latent.width, height: latent.height)
        latent.width = size.width
        latent.height = size.height
        let steps = try inputs.int("steps")
        let seed = UInt64(truncatingIfNeeded: try inputs.int("seed"))
        let denoise = Float(try inputs.float("denoise"))
        let out = try model.sample(conditioning: cond, latent: latent, seed: seed, steps: steps > 0 ? steps : nil,
                                   denoise: denoise) {
            step in
            context.progress(step: step.index + 1, total: step.count, preview: LatentPreview(
                model: model, latents: step.predicted, width: size.width, height: size.height))
        }
        return ["latent": .object(out)]
    }
}

public struct VAEDecodeNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "VAEDecode", displayName: "VAE Decode", category: "latent",
        inputs: [InputSpec("model", .model), InputSpec("latent", .latent), tilingInput],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let latent: Latent = try inputs.object("latent")
        guard let data = latent.data else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "latent is empty; connect a Sampler")
        }
        return ["image": .object(model.decode(data, width: latent.width, height: latent.height,
                                              tiling: try tiling(inputs)))]
    }
}

public struct SaveImageNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "SaveImage", displayName: "Save Image", category: "image",
        inputs: [
            InputSpec("image", .image),
            InputSpec("filename_prefix", .string, default: .string("mlxcomfy")),
            InputSpec("directory", .string, default: .string("outputs"), widget: .filePath),
        ],
        outputs: [], isOutput: true, alwaysExecute: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let image: MLXArray = try inputs.object("image")
        let dir = resolvePath(try inputs.string("directory"))
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let prefix = try inputs.string("filename_prefix")
        let url = nextFilename(in: dir, prefix: prefix)
        try ImageArray.savePNG(image, to: url, text: [
            StandardNodes.graphMetadataKey: context.graph.jsonString(),
            "mlxcomfy.node": context.nodeID,
        ])
        return ["images": .object([url]), "image": .object(image)]
    }

    func nextFilename(in dir: URL, prefix: String) -> URL {
        let existing = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        let used = existing.compactMap { name -> Int? in
            guard name.hasPrefix(prefix + "_"), name.hasSuffix(".png") else { return nil }
            return Int(name.dropFirst(prefix.count + 1).dropLast(4))
        }
        let n = (used.max() ?? 0) + 1
        return dir.appendingPathComponent(String(format: "%@_%05d.png", prefix, n))
    }
}

public struct PreviewImageNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "PreviewImage", displayName: "Preview Image", category: "image",
        inputs: [InputSpec("image", .image)], outputs: [], isOutput: true)
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["image": .object(try inputs.object("image", as: MLXArray.self))]
    }
}

public struct LoadImageNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoadImage", displayName: "Load Image", category: "image",
        inputs: [InputSpec("path", .string, default: .string(""), widget: .filePath)],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let cg = try ImageArray.load(resolvePath(try inputs.string("path")))
        return ["image": .object(ImageArray.array(from: cg))]
    }
}


// MARK: - img2img / inpainting

/// Size with about `megapixels` MP at the given aspect ratio, in multiples of 32.
func sizeFor(megapixels: Double, aspect: Double) -> (Int, Int) {
    let area = megapixels * 1024 * 1024
    let w = (area * aspect).squareRoot(), h = w / aspect
    return (max(32, Int((w / 32).rounded()) * 32), max(32, Int((h / 32).rounded()) * 32))
}

public struct VAEEncodeNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "VAEEncode", displayName: "VAE Encode", category: "latent",
        inputs: [
            InputSpec("model", .model),
            InputSpec("image", .image),
            // 0 keeps the image size (snapped to the model's multiple).
            InputSpec("megapixels", .float, default: .double(1), widget: .number(min: 0, max: 4, step: 0.25)),
            tilingInput,
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let image: MLXArray = try inputs.object("image")
        let mp = try inputs.float("megapixels")
        var (w, h) = (image.dim(1), image.dim(0))
        if mp > 0 { (w, h) = sizeFor(megapixels: mp, aspect: Double(w) / Double(h)) }
        let snapped = model.snapSize(width: w, height: h)
        let resized = try ImageArray.resized(image, width: snapped.width, height: snapped.height)
        let latents = model.encodeImage(resized, tiling: try tiling(inputs))
        return ["latent": .object(Latent(data: latents, width: snapped.width, height: snapped.height))]
    }
}

public struct LoadMaskNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoadMask", displayName: "Load Mask", category: "image",
        inputs: [
            InputSpec("path", .string, default: .string(""), widget: .filePath),
            InputSpec("channel", .string, default: .string("luminance"),
                      widget: .choice(["luminance", "alpha", "red"])),
            InputSpec("invert", .bool, default: .bool(false), widget: .toggle),
        ],
        outputs: [OutputSpec("mask", .mask)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let img = ImageArray.array(from: try ImageArray.load(resolvePath(try inputs.string("path"))))
        var m: MLXArray
        switch try inputs.string("channel") {
        case "alpha": m = img[0..., 0..., 3]
        case "red": m = img[0..., 0..., 0]
        default: m = 0.299 * img[0..., 0..., 0] + 0.587 * img[0..., 0..., 1] + 0.114 * img[0..., 0..., 2]
        }
        if try inputs.bool("invert") { m = 1 - m }
        return ["mask": .object(m)]
    }
}

/// Restricts sampling to a mask (white = regenerate), mapped onto the latent grid.
public struct SetLatentMaskNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "SetLatentMask", displayName: "Set Latent Mask", category: "latent",
        inputs: [
            InputSpec("model", .model),
            InputSpec("latent", .latent),
            InputSpec("mask", .mask),
            // hard: every latent the mask touches is fully regenerated (exact outside the mask after compositing).
            // soft: each latent follows the mask's average coverage, blending generated and original content.
            InputSpec("blend", .string, default: .string("hard"), widget: .choice(["hard", "soft"])),
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        var latent: Latent = try inputs.object("latent")
        let mask: MLXArray = try inputs.object("mask")
        guard latent.data != nil else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "mask needs an encoded image; connect VAE Encode")
        }
        let s = model.latentScale
        let m = try ImageArray.resized(mask, width: latent.width, height: latent.height)
        let (h, w) = (latent.height / s, latent.width / s)
        // Any coverage of a latent token regenerates it; the pixel composite restores exact edges afterwards.
        let cells = m.reshaped(h, s, w, s)
        let weights = try inputs.string("blend") == "soft"
            ? clip(cells.mean(axes: [1, 3]), min: 0, max: 1)
            : (cells.max(axes: [1, 3]) .> 0.02).asType(.float32)
        latent.noiseMask = weights.reshaped(1, h * w, 1)
        return ["latent": .object(latent)]
    }
}

/// `destination` with `source` pasted where `mask` is white (resized to the destination's size).
public struct ImageCompositeMaskedNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "ImageCompositeMasked", displayName: "Composite (Masked)", category: "image",
        inputs: [
            InputSpec("destination", .image),
            InputSpec("source", .image),
            InputSpec("mask", .mask),
        ],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let dst: MLXArray = try inputs.object("destination")
        let (h, w) = (dst.dim(0), dst.dim(1))
        let src = try ImageArray.resized(try inputs.object("source", as: MLXArray.self), width: w, height: h)
        let m = try ImageArray.resized(try inputs.object("mask", as: MLXArray.self), width: w, height: h)
            .expandedDimensions(axis: 2)
        let out = m * src + (1 - m) * dst
        eval(out)
        return ["image": .object(out)]
    }
}


// MARK: - LoRA

/// Applies a LoRA on top of a model. The result shares the base weights; chain several to stack LoRAs.
/// Connect it to both TextEncode and Sampler: the prompt's transformer prefix is computed by the LoRA'd model too.
public struct LoRANode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "LoRA", displayName: "LoRA", category: "loaders",
        inputs: [
            InputSpec("model", .model),
            InputSpec("path", .string, default: .string(""), widget: .filePath),
            InputSpec("strength", .float, default: .double(1), widget: .number(min: -2, max: 2, step: 0.05)),
        ],
        outputs: [OutputSpec("model", .model)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let strength = Float(try inputs.float("strength"))
        if strength == 0 { return ["model": .object(model)] }
        let (out, report) = try model.applyingLoRA(resolvePath(try inputs.string("path")), strength: strength)
        context.progress(step: 1, total: 1, preview: report.description)
        return ["model": .object(out)]
    }
}
