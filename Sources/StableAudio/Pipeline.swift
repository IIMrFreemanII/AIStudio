import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

/// Stable Audio Open Small: text → stereo 44.1 kHz audio (sound effects, ambience, loops), as stable-audio-tools'
/// `generate_diffusion_cond` with the model's `rf_denoiser` objective: T5 + seconds conditioning, a rectified-flow
/// DiT sampled with ping-pong steps on a linear schedule, and the Oobleck VAE decoder.
///
/// Folder layout (the catalog download): `model.safetensors`, `model_config.json`, the t5-base encoder in `t5/`, and
/// `aistudio.json` with `"class": "StableAudioPipeline"`.
public final class StableAudioPipeline: AudioModel, @unchecked Sendable {
    public static let className = "StableAudioPipeline"

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await StableAudioPipeline(directory: url) }
    }

    public var family: String { "stable-audio-open" }
    public var displayName: String { "Stable Audio Open Small (\(directory.lastPathComponent))" }

    public let directory: URL
    public let tokenizer: Tokenizer
    public let t5: T5Encoder
    public let dit: StableAudioDiT
    public let decoder: OobleckDecoder
    public let sampleRate: Int
    /// Audio samples the model generates per run (`sample_size`); requests are trimmed to their length.
    public let sampleSize: Int
    public let downsampling: Int
    let maxTokens: Int
    let secondsRange: (min: Double, max: Double)
    let secondsFreqs: MLXArray  // [128]
    let secondsLinear: Lin      // 257 → 768
    let pretransformScale: Float
    public let defaultSteps: Int
    public let objective: String
    let lock = NSLock()

    public init(directory: URL) async throws {
        self.directory = directory
        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: directory.appendingPathComponent("model_config.json"))) as? [String: Any] ?? [:]
        let model = cfg["model"] as? [String: Any] ?? [:]
        sampleRate = cfg["sample_rate"] as? Int ?? 44100
        sampleSize = cfg["sample_size"] as? Int ?? 524_288
        let pre = model["pretransform"] as? [String: Any] ?? [:]
        let preCfg = pre["config"] as? [String: Any] ?? [:]
        downsampling = preCfg["downsampling_ratio"] as? Int ?? 2048
        pretransformScale = Float(pre["scale"] as? Double ?? 1)
        let decCfg = ((preCfg["decoder"] as? [String: Any])?["config"] as? [String: Any]) ?? [:]
        let diffusion = model["diffusion"] as? [String: Any] ?? [:]
        objective = diffusion["diffusion_objective"] as? String ?? "rf_denoiser"
        defaultSteps = objective == "rf_denoiser" ? 8 : 50

        var tokens = 128
        var range = (min: 0.0, max: 256.0)
        for c in ((model["conditioning"] as? [String: Any])?["configs"] as? [[String: Any]]) ?? [] {
            let conf = c["config"] as? [String: Any] ?? [:]
            if c["id"] as? String == "prompt" { tokens = conf["max_length"] as? Int ?? tokens }
            if c["id"] as? String == "seconds_total" {
                range = (conf["min_val"] as? Double ?? 0, conf["max_val"] as? Double ?? 256)
            }
        }
        maxTokens = tokens
        secondsRange = range

        let t5Dir = directory.appendingPathComponent("t5")
        tokenizer = try await AutoTokenizer.from(modelFolder: t5Dir)
        t5 = try T5Encoder(directory: t5Dir)
        let store = try WeightStore(url: directory.appendingPathComponent("model.safetensors"))
        let sp = "conditioner.conditioners.seconds_total.embedder.embedding."
        secondsFreqs = try store.get(sp + "0.weights").asType(.float32)
        secondsLinear = try store.linear(sp + "1", dtype: .float32)
        dit = try StableAudioDiT(store: store, config: StableAudioDiT.Config(modelConfig: cfg))
        decoder = try OobleckDecoder(store: store, prefix: "pretransform.model.decoder.",
                                     strides: decCfg["strides"] as? [Int] ?? [2, 4, 4, 8, 8],
                                     finalTanh: decCfg["final_tanh"] as? Bool ?? false)
        store.remove(prefix: "pretransform.model.encoder.")
    }

    /// Longest clip the model makes.
    public var maxSeconds: Double { Double(sampleSize) / Double(sampleRate) }

    /// T5 tokens (zero-padded rows) and the seconds embedding: cross-attention `[1, L+1, 768]`, global `[1, 768]`.
    func conditioning(prompt: String, seconds: Double) -> (cross: MLXArray, global: MLXArray) {
        var ids = tokenizer.encode(text: prompt)  // ends with </s>
        if ids.count > maxTokens { ids = Array(ids.prefix(maxTokens - 1)) + [ids.last ?? 1] }
        let n = ids.count
        let padded = ids + [Int](repeating: 0, count: maxTokens - n)
        let mask = MLXArray((0..<maxTokens).map { Float($0 < n ? 1 : 0) }).reshaped(1, maxTokens)
        let text = t5(ids: MLXArray(padded.map { Int32($0) }).reshaped(1, maxTokens), mask: mask)
        // NumberConditioner: clamp, normalize to [0, 1], then [x, sin(2πxw), cos(2πxw)] → linear.
        let x = Float((min(max(seconds, secondsRange.min), secondsRange.max) - secondsRange.min)
                      / (secondsRange.max - secondsRange.min))
        let f = 2 * Float.pi * x * secondsFreqs
        let feats = concatenated([MLXArray([x]), sin(f), cos(f)], axis: 0).reshaped(1, -1)
        let secs = secondsLinear(feats)  // [1, 768]
        return (concatenated([text, secs.expandedDimensions(axis: 1)], axis: 1), secs)
    }

    /// Linear schedule from 1 to 0 (`build_schedule` without a distribution shift).
    static func schedule(steps: Int) -> [Float] { (0...steps).map { 1 - Float($0) / Float(steps) } }

    public func generateAudio(_ request: AudioRequest, onStep: ((Int, Int) throws -> Void)?) throws -> AudioClip {
        lock.lock(); defer { lock.unlock() }
        let seconds = min(max(request.seconds, 0.5), maxSeconds)
        let (cross, global) = conditioning(prompt: request.prompt, seconds: seconds)
        eval(cross, global)
        let steps = max(1, request.steps ?? defaultSteps)
        let t = Self.schedule(steps: steps)
        let key = MLXRandom.key(request.seed)
        let keys = MLXRandom.split(key: key, into: steps + 1)
        let length = sampleSize / downsampling
        var x = MLXRandom.normal([1, length, dit.config.ioChannels], key: keys[0])
        for i in 0..<steps {
            try Task.checkCancellation()
            let tc = MLXArray([t[i]])
            let v = dit(x, t: tc, cond: cross, global: global)
            let denoised = x - t[i] * v
            // Ping-pong: jump to the clean estimate, then re-noise to the next level with fresh noise.
            x = (1 - t[i + 1]) * denoised + t[i + 1] * MLXRandom.normal(x.shape, key: keys[i + 1])
            eval(x)
            try onStep?(i + 1, steps)
        }
        var audio = decoder(x * pretransformScale)[0]  // [N, 2]
        let keep = min(audio.dim(0), Int((seconds * Double(sampleRate)).rounded()))
        audio = audio[0..<keep].transposed(1, 0)  // [2, N]
        // Peak-normalize like the model card's example (to 0.98, leaving headroom for lossy encoding).
        audio = audio * (0.98 / maximum(abs(audio).max(), MLXArray(Float(1e-6))))
        eval(audio)
        Memory.clearCache()
        return AudioClip(samples: audio, sampleRate: sampleRate)
    }
}
