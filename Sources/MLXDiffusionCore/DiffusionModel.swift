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

/// What a diffusion model must provide for the generic graph nodes (TextEncode, Sampler, VAEDecode…) to drive it.
public protocol DiffusionModel: AnyObject {
    /// Stable family identifier, e.g. `"qwen-image-2.1"`.
    var family: String { get }
    var displayName: String { get }

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
    func decode(_ latents: MLXArray, width: Int, height: Int) -> MLXArray

    /// RGBA `[H, W, 4]` in `[0, 1]` (H, W already snapped) → packed latents.
    func encodeImage(_ image: MLXArray) -> MLXArray

    /// Pixels per latent token along each axis (for mapping masks to latents).
    var latentScale: Int { get }

    /// A model that shares this one's weights with a LoRA applied on top (stackable).
    func applyingLoRA(_ lora: URL, strength: Float) throws -> (model: DiffusionModel, report: LoRAReport)

    /// Cheap low-resolution RGB `[h, w, 3]` preview of packed latents, or `nil` if the model has none.
    func latentPreview(_ latents: MLXArray, width: Int, height: Int) -> MLXArray?
}

extension DiffusionModel {
    public func encodePrompt(_ prompt: String) throws -> Conditioning { try encodePrompt(prompt, images: []) }

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

/// Maps a diffusers `model_index.json` `_class_name` to a loader.
public final class ModelRegistry {
    public typealias Loader = (URL) async throws -> DiffusionModel

    public static let shared = ModelRegistry()
    private var loaders: [String: Loader] = [:]
    private let lock = NSLock()

    public init() {}

    public func register(className: String, loader: @escaping Loader) {
        lock.lock(); defer { lock.unlock() }
        loaders[className] = loader
    }

    public static func className(at url: URL) -> String? {
        let idx = url.appendingPathComponent("model_index.json")
        guard let data = try? Data(contentsOf: idx),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return obj["_class_name"] as? String
    }

    public func load(_ url: URL) async throws -> DiffusionModel {
        let cls = Self.className(at: url)
        lock.lock()
        let loader = cls.flatMap { loaders[$0] }
        lock.unlock()
        guard let loader else { throw DiffusionError.unknownModel(url, cls) }
        return try await loader(url)
    }
}
