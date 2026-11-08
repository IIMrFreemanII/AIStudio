import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore

// Nodes for video models (`VideoDiffusionModel`): an empty clip, the sampler and the decoder. Prompts go through the
// ordinary Text Encode node, which works for both image and video models.

func videoModel(_ inputs: NodeInputs, _ name: String = "model") throws -> VideoDiffusionModel {
    let model: GenerativeModel = try inputs.object(name)
    guard let video = model as? VideoDiffusionModel else {
        throw GraphError.nodeFailed(node: inputs.nodeID, message: "\(model.displayName) is not a video model")
    }
    return video
}

/// The size and length of the clip to generate.
public struct EmptyVideoLatentNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "EmptyVideoLatent", displayName: "Empty Video Latent", category: "latent",
        inputs: [
            InputSpec("width", .int, default: .int(768), widget: .number(min: 64, max: 1920, step: 64)),
            InputSpec("height", .int, default: .int(512), widget: .number(min: 64, max: 1920, step: 64)),
            InputSpec("seconds", .float, default: .double(5), widget: .number(min: 0.5, max: 20, step: 0.5)),
            InputSpec("fps", .float, default: .double(24), widget: .number(min: 8, max: 60, step: 1)),
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let fps = try inputs.float("fps")
        return ["latent": .object(VideoLatent(width: try inputs.int("width"), height: try inputs.int("height"),
                                              frames: VideoLatent.frames(seconds: try inputs.float("seconds"), fps: fps),
                                              fps: fps))]
    }
}

/// Generates a clip (and its soundtrack, for models that make one). An optional image pins the first frame.
public struct VideoSamplerNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "VideoSampler", displayName: "Video Sampler", category: "sampling",
        inputs: [
            InputSpec("model", .model),
            InputSpec("conditioning", .conditioning),
            InputSpec("latent", .latent),
            InputSpec("seed", .int, default: .int(0), widget: .seed),
            // 0 = the model's default schedule.
            InputSpec("steps", .int, default: .int(0), widget: .number(min: 0, max: 50, step: 1)),
            InputSpec("image", .image, optional: true),
        ],
        outputs: [OutputSpec("latent", .latent)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model = try videoModel(inputs)
        let cond: Conditioning = try inputs.object("conditioning")
        let latent: VideoLatent = try inputs.object("latent")
        let steps = try inputs.int("steps")
        let seed = UInt64(truncatingIfNeeded: try inputs.int("seed"))
        let image = inputs.optionalObject("image", as: MLXArray.self)
        let out = try model.sampleVideo(conditioning: cond, latent: latent, seed: seed, steps: steps > 0 ? steps : nil,
                                        image: image) { step in
            context.progress(step: step.index + 1, total: step.count, preview: step.preview)
        }
        return ["latent": .object(out)]
    }
}

/// Decodes a sampled clip into frames and, for models that make one, the soundtrack.
public struct VideoDecodeNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "VideoDecode", displayName: "Video Decode", category: "latent",
        inputs: [InputSpec("model", .model), InputSpec("latent", .latent)],
        outputs: [OutputSpec("video", .video), OutputSpec("audio", .audio)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let model = try videoModel(inputs)
        let latent: VideoLatent = try inputs.object("latent")
        guard latent.payload != nil else {
            throw GraphError.nodeFailed(node: inputs.nodeID, message: "latent is empty; connect a Video Sampler")
        }
        let clip = try model.decodeVideo(latent)
        // A silent clip still has an audio output: one sample of silence keeps downstream nodes simple.
        let audio = clip.audio ?? AudioClip(samples: MLXArray.zeros([1, 1]), sampleRate: 48000)
        return ["video": .object(clip), "audio": .object(audio)]
    }
}
