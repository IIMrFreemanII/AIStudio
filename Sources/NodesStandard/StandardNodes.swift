import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore
import QwenImage21

/// The built-in node library and the model loaders it relies on.
public enum StandardNodes {
    public static let all: [NodeImplementation.Type] = [
        LoadModelNode.self, TextEncodeNode.self, EmptyLatentNode.self, SamplerNode.self, VAEDecodeNode.self,
        SaveImageNode.self, PreviewImageNode.self, LoadImageNode.self,
    ]

    /// Registry with every standard node; also registers the bundled model loaders.
    public static func makeRegistry() -> NodeRegistry {
        QwenImage21Pipeline.register()
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
        let model: DiffusionModel = try await context.resources.get("model:\(url.path)") {
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
        ],
        outputs: [OutputSpec("conditioning", .conditioning)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        return ["conditioning": .object(try model.encodePrompt(try inputs.string("prompt")))]
    }
}

public struct EmptyLatentNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "EmptyLatent", displayName: "Empty Latent", category: "latent",
        inputs: [
            InputSpec("width", .int, default: .int(1024), widget: .number(min: 256, max: 2048, step: 32)),
            InputSpec("height", .int, default: .int(1024), widget: .number(min: 256, max: 2048, step: 32)),
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
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let cond: Conditioning = try inputs.object("conditioning")
        var latent: Latent = try inputs.object("latent")
        let size = model.snapSize(width: latent.width, height: latent.height)
        latent.width = size.width
        latent.height = size.height
        let steps = try inputs.int("steps")
        let seed = UInt64(truncatingIfNeeded: try inputs.int("seed"))
        let out = try model.sample(conditioning: cond, latent: latent, seed: seed, steps: steps > 0 ? steps : nil) {
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
        inputs: [InputSpec("model", .model), InputSpec("latent", .latent)],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model: DiffusionModel = try inputs.object("model")
        let latent: Latent = try inputs.object("latent")
        guard let data = latent.data else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "latent is empty; connect a Sampler")
        }
        return ["image": .object(model.decode(data, width: latent.width, height: latent.height))]
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
