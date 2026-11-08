import Foundation
import MLX

/// Model-specific prompt conditioning (e.g. encoder features plus a cached transformer prefix).
/// Opaque to everything except the model that produced it.
public final class Conditioning {
    public let family: String
    public let payload: Any
    /// Output size the conditioning suggests (e.g. the aspect ratio of a condition image), if any.
    public let suggestedSize: (width: Int, height: Int)?
    public init(family: String, payload: Any, suggestedSize: (width: Int, height: Int)? = nil) {
        self.family = family
        self.payload = payload
        self.suggestedSize = suggestedSize
    }
}

/// Packed latents plus the pixel size they decode to.
///
/// `data == nil` is an empty latent: the sampler starts from pure noise. With `data`, the sampler starts from it
/// (img2img) at the strength given by its `denoise`, and `noiseMask` (`[1, h*w, 1]`, 1 = regenerate) restricts
/// generation to the masked area (inpainting).
public struct Latent {
    public var data: MLXArray?
    public var width: Int
    public var height: Int
    public var noiseMask: MLXArray?
    public init(data: MLXArray? = nil, width: Int, height: Int, noiseMask: MLXArray? = nil) {
        self.data = data
        self.width = width
        self.height = height
        self.noiseMask = noiseMask
    }
}

public struct SamplerStep {
    public let index: Int
    public let count: Int
    public let sigma: Float
    /// Clean-sample estimate after this step, in the model's packed latent layout.
    public let predicted: MLXArray
}

/// Anything a model folder loads into: image, video or audio models (and models that also write text).
public protocol GenerativeModel: AnyObject {
    /// Stable family identifier, e.g. `"qwen-image-2.1"`.
    var family: String { get }
    var displayName: String { get }
}

/// What an image diffusion model must provide for the generic graph nodes (TextEncode, Sampler, VAEDecode…).
public protocol DiffusionModel: GenerativeModel {

    /// Encodes a prompt, optionally with condition images (`[H, W, 4]` RGBA in `[0, 1]`) for editing/reference.
    func encodePrompt(_ prompt: String, images: [MLXArray]) throws -> Conditioning

    /// How many condition images the model accepts (0 = text only).
    var maxConditionImages: Int { get }

    /// Pixel size snapped to what the model accepts.
    func snapSize(width: Int, height: Int) -> (width: Int, height: Int)

    /// Initial noise for a latent of the given pixel size.
    func noise(seed: UInt64, width: Int, height: Int) -> MLXArray

    /// A scheduler configured for this model, with its default schedule (`steps == nil` → checkpoint default).
    func makeScheduler(steps: Int?) -> FlowMatchEuler

    /// Flow velocity prediction.
    func velocity(_ latents: MLXArray, conditioning: Conditioning, sigma: Float, width: Int, height: Int) throws
        -> MLXArray

    /// Packed latents → RGBA `[H, W, 4]` in `[0, 1]`.
    func decode(_ latents: MLXArray, width: Int, height: Int, tiling: VAETiling) -> MLXArray

    /// RGBA `[H, W, 4]` in `[0, 1]` (H, W already snapped) → packed latents.
    func encodeImage(_ image: MLXArray, tiling: VAETiling) -> MLXArray

    /// Pixels per latent token along each axis (for mapping masks to latents).
    var latentScale: Int { get }

    /// A model that shares this one's weights with a LoRA applied on top (stackable).
    func applyingLoRA(_ lora: URL, strength: Float) throws -> (model: DiffusionModel, report: LoRAReport)

    /// Cheap low-resolution RGB `[h, w, 3]` preview of packed latents, or `nil` if the model has none.
    func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray?
}

/// Whether the VAE processes an image in overlapping tiles. `auto` tiles only above a model-chosen size, so ordinary
/// sizes are decoded exactly as before.
public enum VAETiling: String, CaseIterable, Sendable {
    case auto, on, off
}

extension DiffusionModel {
    public func encodePrompt(_ prompt: String) throws -> Conditioning { try encodePrompt(prompt, images: []) }

    public func decode(_ latents: MLXArray, width: Int, height: Int) -> MLXArray {
        decode(latents, width: width, height: height, tiling: .auto)
    }

    public func encodeImage(_ image: MLXArray) -> MLXArray { encodeImage(image, tiling: .auto) }

    public func applyingLoRA(_ lora: URL, strength: Float) throws -> (model: DiffusionModel, report: LoRAReport) {
        throw LoRAError.unsupported(displayName)
    }

    public func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray? { nil }
}

public enum DiffusionError: Error, CustomStringConvertible {
    case wrongConditioning(expected: String, got: String)
    case unknownModel(URL, String?)
    case tooManyImages(max: Int)

    public var description: String {
        switch self {
        case .wrongConditioning(let e, let g): return "conditioning from '\(g)' cannot drive a '\(e)' model"
        case .unknownModel(let u, let c): return "no loader for model at \(u.path) (class \(c ?? "unknown"))"
        case .tooManyImages(let m): return m == 0 ? "this model does not take condition images" : "at most \(m) condition images"
        }
    }
}

extension DiffusionModel {
    /// Full sampling loop shared by the CLI and the Sampler node, so both produce identical results.
    ///
    /// - `denoise` < 1 (with source latents) starts part-way down the schedule, at the first sigma ≤ `denoise`.
    /// - With `latent.noiseMask`, the unmasked area is reset after every step to the source re-noised to the next
    ///   sigma, so only the masked area is generated, in context; at the final sigma it equals the source exactly.
    public func sample(conditioning: Conditioning, latent: Latent, seed: UInt64, steps: Int?, denoise: Float = 1,
                       onStep: ((SamplerStep) throws -> Void)? = nil) throws -> Latent {
        let scheduler = makeScheduler(steps: steps)
        let noise = noise(seed: seed, width: latent.width, height: latent.height)
        var start = 0
        var x: MLXArray
        if let source = latent.data {
            if denoise < 1 {
                start = scheduler.sigmas.firstIndex { $0 <= denoise } ?? scheduler.stepCount
                start = min(start, max(scheduler.stepCount - 1, 0))
            }
            let s0 = scheduler.sigmas[start]
            x = s0 * noise + (1 - s0) * source.asType(.float32)
        } else {
            x = noise
        }
        let mask = latent.noiseMask?.asType(.float32)
        for i in start..<scheduler.stepCount {
            try Task.checkCancellation()
            let sigma = scheduler.sigmas[i]
            let v = try velocity(x, conditioning: conditioning, sigma: sigma, width: latent.width,
                                 height: latent.height)
            var x0 = scheduler.predictX0(v, sample: x, index: i)
            x = scheduler.step(v, sample: x, index: i)
            if let mask, let source = latent.data?.asType(.float32) {
                let next = scheduler.sigmas[i + 1]
                x = mask * x + (1 - mask) * (next * noise + (1 - next) * source)
                x0 = mask * x0 + (1 - mask) * source
            }
            eval(x, x0)
            try onStep?(SamplerStep(index: i - start, count: scheduler.stepCount - start, sigma: sigma, predicted: x0))
        }
        return Latent(data: x, width: latent.width, height: latent.height)
    }
}

/// A request to a video model: prompt, size, length and optional first frame.
public struct VideoRequest: @unchecked Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var seconds: Double
    public var fps: Double
    public var seed: UInt64
    public var steps: Int?
    /// Optional first frame (`[H, W, 3|4]` in `[0, 1]`) for image-to-video.
    public var image: MLXArray?
    public init(prompt: String, width: Int = 768, height: Int = 512, seconds: Double = 5, fps: Double = 24,
                seed: UInt64 = 0, steps: Int? = nil, image: MLXArray? = nil) {
        self.prompt = prompt; self.width = width; self.height = height; self.seconds = seconds; self.fps = fps
        self.seed = seed; self.steps = steps; self.image = image
    }
}

/// A model that generates video (possibly with a soundtrack). Model ports refine this with finer-grained steps.
public protocol VideoModel: GenerativeModel {
    func generateVideo(_ request: VideoRequest, onStep: ((Int, Int) throws -> Void)?) throws -> VideoClip
}

/// Video latents in a model's own layout (an opaque payload), plus the clip they decode to.
///
/// `payload == nil` is an empty latent: the sampler starts from noise.
public struct VideoLatent: @unchecked Sendable {
    public var payload: Any?
    public var width: Int
    public var height: Int
    public var frames: Int
    public var fps: Double
    public init(payload: Any? = nil, width: Int, height: Int, frames: Int, fps: Double) {
        self.payload = payload; self.width = width; self.height = height; self.frames = frames; self.fps = fps
    }

    /// Frame count for a duration (the model snaps it to what it accepts).
    public static func frames(seconds: Double, fps: Double) -> Int { max(1, Int((seconds * fps).rounded()) + 1) }
}

/// Progress of a video sampler: `index` of `count` steps across all its stages (`stage` names the current one),
/// with an optional small RGB preview `[h, w, 3]` in `[0, 1]`.
public struct VideoSamplerStep {
    public let index: Int
    public let count: Int
    public let stage: String
    public let preview: MLXArray?
    public init(index: Int, count: Int, stage: String, preview: MLXArray?) {
        self.index = index; self.count = count; self.stage = stage; self.preview = preview
    }
}

/// A video model split into the stages the graph nodes drive: prompt encoding, sampling, decoding.
public protocol VideoDiffusionModel: VideoModel {
    func encodeVideoPrompt(_ prompt: String) throws -> Conditioning
    /// Size and frame count snapped to what the model accepts.
    func snapVideo(width: Int, height: Int, frames: Int) -> (width: Int, height: Int, frames: Int)
    /// Samples a clip; `image` (`[H, W, 3|4]` in `[0, 1]`) pins the first frame (image-to-video).
    func sampleVideo(conditioning: Conditioning, latent: VideoLatent, seed: UInt64, steps: Int?, image: MLXArray?,
                     onStep: ((VideoSamplerStep) throws -> Void)?) throws -> VideoLatent
    /// Latents → frames (and the soundtrack, for models that generate one).
    func decodeVideo(_ latent: VideoLatent) throws -> VideoClip
}

extension VideoDiffusionModel {
    /// One-call generation built from the stages.
    public func generateVideo(_ request: VideoRequest, onStep: ((Int, Int) throws -> Void)?) throws -> VideoClip {
        let cond = try encodeVideoPrompt(request.prompt)
        let latent = VideoLatent(width: request.width, height: request.height,
                                 frames: VideoLatent.frames(seconds: request.seconds, fps: request.fps), fps: request.fps)
        let out = try sampleVideo(conditioning: cond, latent: latent, seed: request.seed, steps: request.steps,
                                  image: request.image) { try onStep?($0.index + 1, $0.count) }
        return try decodeVideo(out)
    }
}

/// A request to an audio model. Which fields matter depends on the model (music uses lyrics and the musical fields,
/// speech a voice).
public struct AudioRequest: Sendable {
    public var prompt: String
    public var seconds: Double
    public var seed: UInt64
    public var steps: Int?
    public var lyrics: String
    public var voice: String
    public var speed: Double
    /// Music: tempo (nil = let the model decide), key such as "C major" and time signature such as "4" ("" = auto),
    /// vocal language code such as "en" ("" = auto), and whether the song planner (LM) runs first.
    public var bpm: Int?
    public var key: String
    public var timeSignature: String
    public var language: String
    public var planner: Bool
    public init(prompt: String, seconds: Double = 10, seed: UInt64 = 0, steps: Int? = nil, lyrics: String = "",
                voice: String = "", speed: Double = 1, bpm: Int? = nil, key: String = "", timeSignature: String = "",
                language: String = "", planner: Bool = true) {
        self.prompt = prompt; self.seconds = seconds; self.seed = seed; self.steps = steps; self.lyrics = lyrics
        self.voice = voice; self.speed = speed
        self.bpm = bpm; self.key = key; self.timeSignature = timeSignature; self.language = language
        self.planner = planner
    }
}

/// A model that generates audio: music, speech or sound effects.
public protocol AudioModel: GenerativeModel {
    func generateAudio(_ request: AudioRequest, onStep: ((Int, Int) throws -> Void)?) throws -> AudioClip
}

/// Maps a model folder to a loader, by the `_class_name` in a diffusers `model_index.json` or the `class` in an
/// `mlxcomfy.json` manifest (for repos without one).
public final class ModelRegistry {
    public typealias Loader = (URL) async throws -> GenerativeModel

    public static let shared = ModelRegistry()
    private var loaders: [String: Loader] = [:]
    private let lock = NSLock()

    public init() {}

    public func register(className: String, loader: @escaping Loader) {
        lock.lock(); defer { lock.unlock() }
        loaders[className] = loader
    }

    public static func className(at url: URL) -> String? {
        for (file, key) in [("model_index.json", "_class_name"), ("mlxcomfy.json", "class")] {
            guard let data = try? Data(contentsOf: url.appendingPathComponent(file)),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = obj[key] as? String else { continue }
            return name
        }
        return nil
    }

    /// Size of the weight files in a model folder (recursively): the estimate used for the memory budget.
    public static func weightBytes(at url: URL) -> Int {
        guard let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total = 0
        for case let f as URL in e where ["safetensors", "bin", "gguf", "npz"].contains(f.pathExtension.lowercased()) {
            total += (try? f.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        }
        return total
    }

    public func isRegistered(_ className: String) -> Bool {
        lock.lock(); defer { lock.unlock() }
        return loaders[className] != nil
    }

    public func load(_ url: URL) async throws -> GenerativeModel {
        let cls = Self.className(at: url)
        lock.lock()
        let loader = cls.flatMap { loaders[$0] }
        lock.unlock()
        guard let loader else { throw DiffusionError.unknownModel(url, cls) }
        return try await loader(url)
    }
}
