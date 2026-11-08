import Foundation
import MLX

/// Model-specific prompt conditioning (e.g. encoder features plus a cached transformer prefix).
/// Opaque to everything except the model that produced it.
public final class Conditioning {
    public let family: String
    public let payload: Any
    public init(family: String, payload: Any) {
        self.family = family
        self.payload = payload
    }
}

/// Packed latents plus the pixel size they decode to. `data == nil` is an empty latent (sampler draws noise).
public struct Latent {
    public var data: MLXArray?
    public var width: Int
    public var height: Int
    public init(data: MLXArray? = nil, width: Int, height: Int) {
        self.data = data
        self.width = width
        self.height = height
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

    func encodePrompt(_ prompt: String) throws -> Conditioning

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
}

public enum DiffusionError: Error, CustomStringConvertible {
    case wrongConditioning(expected: String, got: String)
    case unknownModel(URL, String?)

    public var description: String {
        switch self {
        case .wrongConditioning(let e, let g): return "conditioning from '\(g)' cannot drive a '\(e)' model"
        case .unknownModel(let u, let c): return "no loader for model at \(u.path) (class \(c ?? "unknown"))"
        }
    }
}

extension DiffusionModel {
    /// Full sampling loop shared by the CLI and the Sampler node, so both produce identical results.
    public func sample(conditioning: Conditioning, latent: Latent, seed: UInt64, steps: Int?,
                       onStep: ((SamplerStep) throws -> Void)? = nil) throws -> Latent {
        let scheduler = makeScheduler(steps: steps)
        var x = latent.data ?? noise(seed: seed, width: latent.width, height: latent.height)
        for i in 0..<scheduler.stepCount {
            try Task.checkCancellation()
            let sigma = scheduler.sigmas[i]
            let v = try velocity(x, conditioning: conditioning, sigma: sigma, width: latent.width,
                                 height: latent.height)
            let x0 = scheduler.predictX0(v, sample: x, index: i)
            x = scheduler.step(v, sample: x, index: i)
            eval(x, x0)
            try onStep?(SamplerStep(index: i, count: scheduler.stepCount, sigma: sigma, predicted: x0))
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
