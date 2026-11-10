import Foundation
import MLX
import MLXDiffusionCore
import WanTransformer
import WanVAE

public enum WanError: Error, CustomStringConvertible {
    case missingFile(String, String), wrongConditioning(String), needsImage, emptyLatent
    public var description: String {
        switch self {
        case .missingFile(let f, let d): "\(f) not found in the model folder \(d)"
        case .wrongConditioning(let f): "the prompt was encoded by a different model (\(f)); connect Text Encode to the Wan model"
        case .needsImage: "Wan 2.2 I2V animates an image: connect a first-frame image to the Video Sampler"
        case .emptyLatent: "nothing to decode: connect the Video Sampler's latent"
        }
    }
}

/// Wan 2.2 A14B text-to-video and image-to-video, as the official `WanT2V` / `WanI2V`: umT5 prompt encoding, a
/// high-noise and a low-noise 14B expert switched at a timestep boundary, and the Wan 2.1 VAE. 16 fps, no sound.
///
/// Two sampling modes:
/// - **fast** (default when the Lightning LoRAs are in `lora/`): 4 Euler steps, guidance 1, shift 5, with each
///   expert's lightx2v LoRA attached;
/// - **full**: the official UniPC sampler (40 steps, the variant's shift and per-expert guidance, the default
///   negative prompt).
///
/// Folder layouts: the PocketAiHub T2V (`q4/{config.json, high_noise_model, low_noise_model}` + `common/{t5_encoder,
/// tokenizer.json, vae}`) or the flat mlx-video layout (all in one folder), plus `lora/**/{high,low}_noise_model
/// .safetensors` and `mlxcomfy.json` with `"class": "WanPipeline"`.
public final class WanPipeline: LoRAVideoModel, @unchecked Sendable {
    public static let className = "WanPipeline"

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await WanPipeline(directory: url) }
    }

    public var family: String { isI2V ? "wan-2.2-i2v" : "wan-2.2-t2v" }
    public var displayName: String { "Wan 2.2 \(isI2V ? "I2V" : "T2V") A14B (\(directory.lastPathComponent))" }

    /// Official sampling settings from `config.json`.
    public struct Settings: Sendable {
        public var boundary = 0.875, shift = 12.0, steps = 40
        public var guidanceLow = 3.0, guidanceHigh = 4.0
        public var fps = 16.0, maxFrames = 81
        public var negativePrompt = ""
    }

    public let directory: URL
    public let config: WanDiT.Config
    public let settings: Settings
    let high: WanDiT
    let low: WanDiT
    let vae: WanVAE
    let text: WanTextEncoder
    let lightning: (high: [String: LoRAAdapter], low: [String: LoRAAdapter])?
    /// Your LoRAs (LoRA nodes), per expert, on top of the shared weights.
    let userLoRAs: (high: [[String: LoRAAdapter]], low: [[String: LoRAAdapter]])
    var negativeContext: MLXArray?
    /// Shared by every LoRA view of the same weights: the experts' adapters are set per generation.
    let lock: NSLock

    public var isI2V: Bool { config.inDim > 16 }
    public var hasLightning: Bool { lightning != nil }

    struct Context { let context: MLXArray }

    public init(directory: URL) async throws {
        self.directory = directory
        lock = NSLock()
        userLoRAs = ([], [])
        let fm = FileManager.default
        func find(_ name: String) throws -> URL {
            for sub in ["", "common", "q4", "q8", "bf16"] {
                let u = sub.isEmpty ? directory.appendingPathComponent(name)
                    : directory.appendingPathComponent(sub).appendingPathComponent(name)
                if fm.fileExists(atPath: u.path) { return u }
            }
            throw WanError.missingFile(name, directory.lastPathComponent)
        }
        let configURL = try find("config.json")
        let modelDir = configURL.deletingLastPathComponent()
        let cfg = try WanDiT.Config(url: configURL)
        config = cfg
        let o = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        var s = Settings()
        s.boundary = o["boundary"] as? Double ?? s.boundary
        s.shift = o["sample_shift"] as? Double ?? s.shift
        s.steps = o["sample_steps"] as? Int ?? s.steps
        if let g = o["sample_guide_scale"] as? [Double], g.count == 2 { (s.guidanceLow, s.guidanceHigh) = (g[0], g[1]) }
        s.fps = Double(o["sample_fps"] as? Int ?? 16)
        s.maxFrames = o["frame_num"] as? Int ?? s.maxFrames
        s.negativePrompt = o["sample_neg_prompt"] as? String ?? ""
        settings = s
        let bits = (o["quantization"] as? [String: Any])?["bits"] as? Int

        func expert(_ name: String) throws -> WanDiT {
            let store = try WeightStore(url: modelDir.appendingPathComponent(name))
            if let bits { store.bitsOverride = bits }
            return try WanDiT(store: store, config: cfg)
        }
        high = try expert("high_noise_model.safetensors")
        low = try expert("low_noise_model.safetensors")
        vae = try WanVAE(url: try find("vae.safetensors"), encoder: cfg.inDim > 16)
        text = try WanTextEncoder(weights: try find("t5_encoder.safetensors"), tokenizerFile: try find("tokenizer.json"))

        // Lightning LoRAs: lora/**/high_noise_model.safetensors and low_noise_model.safetensors.
        var loraFiles: [String: URL] = [:]
        if let e = fm.enumerator(at: directory.appendingPathComponent("lora"), includingPropertiesForKeys: nil) {
            for case let u as URL in e where u.pathExtension == "safetensors" {
                if u.lastPathComponent.hasPrefix("high_noise") { loraFiles["high"] = u }
                if u.lastPathComponent.hasPrefix("low_noise") { loraFiles["low"] = u }
            }
        }
        if let h = loraFiles["high"], let l = loraFiles["low"] {
            let names = high.linearNames
            lightning = (try LoRAFile(url: h, moduleNames: names).adapters(strength: 1),
                         try LoRAFile(url: l, moduleNames: names).adapters(strength: 1))
        } else {
            lightning = nil
        }
    }

    private init(copying o: WanPipeline, userLoRAs: (high: [[String: LoRAAdapter]], low: [[String: LoRAAdapter]])) {
        directory = o.directory; config = o.config; settings = o.settings; high = o.high; low = o.low; vae = o.vae
        text = o.text; lightning = o.lightning; lock = o.lock; negativeContext = o.negativeContext
        self.userLoRAs = userLoRAs
    }

    // MARK: LoRA

    /// A view of this model with a LoRA on top: `lora` for both experts (`expert` nil), or for one; `companion` is the
    /// low-noise expert's file of a pair (then `lora` is the high-noise one's). The weights are shared.
    public func applyingLoRA(_ lora: URL, companion: URL?, expert: String?, strength: Float) throws
        -> (model: GenerativeModel, report: LoRAReport) {
        let names = high.linearNames
        let first = try Self.loraFile(lora, names: names)
        var (h, l) = userLoRAs
        var matched = 0, unmatched = first.unmatchedKeys.count
        let a = first.adapters(strength: strength)
        if let companion {
            let second = try Self.loraFile(companion, names: names)
            h.append(a)
            l.append(second.adapters(strength: strength))
            matched = first.pairs.count + second.pairs.count
            unmatched += second.unmatchedKeys.count
        } else {
            if expert != "low" { h.append(a); matched += first.pairs.count }
            if expert != "high" { l.append(a); matched += first.pairs.count }
        }
        let name = companion.map { "\(lora.lastPathComponent) + \($0.lastPathComponent)" } ?? lora.lastPathComponent
        let report = LoRAReport(file: name, matchedLayers: matched, unmatchedKeys: unmatched)
        return (WanPipeline(copying: self, userLoRAs: (h, l)), report)
    }

    /// A Wan LoRA file: the official names (kohya / ComfyUI / musubi: `blocks.N.self_attn.q`) or diffusers'
    /// (`blocks.N.attn1.to_q`), mapped onto the transformer's layers.
    static func loraFile(_ url: URL, names: Set<String>) throws -> LoRAFile {
        let (arrays, _) = try loadArraysAndMetadata(url: url)
        let (groups, other) = LoRAFile.groups(arrays)
        let underscored = Dictionary(names.map { ($0.replacingOccurrences(of: ".", with: "_"), $0) },
                                     uniquingKeysWith: { a, _ in a })
        var pairs: [String: LoRAFile.Pair] = [:]
        var unmatched = other
        for (name, pair) in groups {
            let official = officialName(name)
            guard let target = names.contains(official) ? official : underscored[name] else {
                unmatched += pair.keys; continue
            }
            pairs[target] = pair
        }
        return try LoRAFile(url: url, pairs: pairs, unmatchedKeys: unmatched, sample: Array(arrays.keys.sorted().prefix(3)))
    }

    /// diffusers' Wan layer names → the official ones (unchanged when already official).
    static func officialName(_ name: String) -> String {
        var n = name
        for (a, b) in [(".attn1.", ".self_attn."), (".attn2.", ".cross_attn."), (".to_out.0", ".o"), (".to_q", ".q"),
                       (".to_k", ".k"), (".to_v", ".v"), (".ffn.net.0.proj", ".ffn.0"), (".ffn.net.2", ".ffn.2")] {
            n = n.replacingOccurrences(of: a, with: b)
        }
        return n
    }

    // MARK: VideoDiffusionModel

    public func encodeVideoPrompt(_ prompt: String) throws -> Conditioning {
        lock.lock(); defer { lock.unlock() }
        let ctx = try text.encode([prompt])[0]
        return Conditioning(family: family, payload: Context(context: ctx))
    }

    public func snapVideo(width: Int, height: Int, frames: Int) -> (width: Int, height: Int, frames: Int) {
        func snap(_ v: Int) -> Int { max(256, min(1280, (v / 16) * 16)) }
        let k = max(1, min((settings.maxFrames - 1) / 4, Int((Double(frames - 1) / 4).rounded())))
        return (snap(width), snap(height), 4 * k + 1)
    }

    public func sampleVideo(conditioning: Conditioning, latent: VideoLatent, seed: UInt64, steps: Int?, image: MLXArray?,
                            onStep: ((VideoSamplerStep) throws -> Void)?) throws -> VideoLatent {
        try sampleVideo(conditioning: conditioning, latent: latent, seed: seed, steps: steps, image: image,
                        quality: .fast, onStep: onStep)
    }

    public func sampleVideo(conditioning: Conditioning, latent: VideoLatent, seed: UInt64, steps: Int?, image: MLXArray?,
                            quality: VideoQuality, onStep: ((VideoSamplerStep) throws -> Void)?) throws -> VideoLatent {
        guard let cond = conditioning.payload as? Context else {
            throw WanError.wrongConditioning(conditioning.family)
        }
        lock.lock(); defer { lock.unlock() }
        let (W, H, frames) = snapVideo(width: latent.width, height: latent.height, frames: latent.frames)
        let (T, h, w) = ((frames - 1) / 4 + 1, H / 8, W / 8)
        let fast = quality == .fast && lightning != nil

        let schedule = fast ? WanSchedule.lightning(steps: max(1, min(8, steps ?? 4)))
            : WanSchedule.official(steps: max(2, min(100, steps ?? settings.steps)), shift: settings.shift)
        // Lightning (fast) and your LoRAs, per expert.
        let highSets = (fast ? lightning.map { [$0.high] } ?? [] : []) + userLoRAs.high
        let lowSets = (fast ? lightning.map { [$0.low] } ?? [] : []) + userLoRAs.low
        if highSets.isEmpty { high.clearAdapters() } else { high.setAdapters(highSets) }
        if lowSets.isEmpty { low.clearAdapters() } else { low.setAdapters(lowSets) }
        let (gLow, gHigh) = fast ? (1.0, 1.0) : (settings.guidanceLow, settings.guidanceHigh)
        var negative: MLXArray?
        if gLow != 1 || gHigh != 1 {
            if negativeContext == nil { negativeContext = try text.encode([settings.negativePrompt])[0] }
            negative = negativeContext
        }

        // Image-to-video condition: 4 mask channels (first frame known) + the VAE latents of [image, zeros…].
        var y: MLXArray?
        if isI2V {
            guard let image else { throw WanError.needsImage }
            y = try i2vCondition(image: image, width: W, height: H, frames: frames)
        }

        var x = MLXRandom.normal([16, T, h, w], key: MLXRandom.key(seed))
        let rope = high.rope(frames: T / config.patch.t, height: h / config.patch.h, width: w / config.patch.w)
        var prepared: [ObjectIdentifier: ([(MLXArray, MLXArray)], [(MLXArray, MLXArray)]?)] = [:]
        func kv(_ m: WanDiT) -> ([(MLXArray, MLXArray)], [(MLXArray, MLXArray)]?) {
            if let p = prepared[ObjectIdentifier(m)] { return p }
            let c = m.crossKV(m.embedText(cond.context))
            let u = negative.map { m.crossKV(m.embedText($0)) }
            prepared = [ObjectIdentifier(m): (c, u)]  // the other expert's K/V are no longer needed
            return (c, u)
        }

        var unipc = WanUniPC(sigmas: schedule.sigmas)
        let n = schedule.timesteps.count
        for i in 0..<n {
            try Task.checkCancellation()
            let t = schedule.timesteps[i]
            let isHigh = t >= settings.boundary * 1000
            let m = isHigh ? high : low
            let (kc, ku) = kv(m)
            let input = y.map { concatenated([x, $0], axis: 0) } ?? x
            var v = m(input, timestep: Float(t), crossKV: kc, rope: rope)
            if let ku {
                let g = Float(isHigh ? gHigh : gLow)
                let vu = m(input, timestep: Float(t), crossKV: ku, rope: rope)
                v = vu + g * (v - vu)
            }
            let x0 = x - Float(schedule.sigmas[i]) * v
            x = fast ? x + Float(schedule.sigmas[i + 1] - schedule.sigmas[i]) * v : unipc.step(velocity: v, sample: x)
            eval(x, x0)
            try onStep?(VideoSamplerStep(index: i, count: n, stage: isHigh ? "high-noise expert" : "low-noise expert",
                                         preview: WanPreview.rgb(x0)))
        }
        prepared = [:]
        Memory.clearCache()
        return VideoLatent(payload: x, width: W, height: H, frames: frames, fps: settings.fps)
    }

    /// `[20, T, h, w]`: the mask (1 on the first frame, as 4 channels per latent frame) and the normalized VAE
    /// latents of the image followed by black frames.
    func i2vCondition(image: MLXArray, width W: Int, height H: Int, frames: Int) throws -> MLXArray {
        let rgb = image.dim(2) == 4 ? image[0..., 0..., 0..<3] : image
        let first = try Self.resizeAndCrop(rgb.asType(.float32), width: W, height: H) * 2 - 1  // [H, W, 3]
        let video = concatenated([first.expandedDimensions(axes: [0, 1]), MLXArray.zeros([1, frames - 1, H, W, 3])],
                                 axis: 1)
        let z = vae.encode(video)[0].transposed(3, 0, 1, 2)  // [16, T, h, w]
        let (T, h, w) = (z.dim(1), z.dim(2), z.dim(3))
        let mask = concatenated([MLXArray.ones([4, 1, h, w]), MLXArray.zeros([4, T - 1, h, w])], axis: 1)
        let y = concatenated([mask, z], axis: 0)
        eval(y)
        Memory.clearCache()
        return y
    }

    /// Scales `image` `[h, w, 3]` to cover `width × height` and crops the center.
    static func resizeAndCrop(_ image: MLXArray, width: Int, height: Int) throws -> MLXArray {
        let (ih, iw) = (Double(image.dim(0)), Double(image.dim(1)))
        let scale = max(Double(width) / iw, Double(height) / ih)
        let (rw, rh) = (max(width, Int((iw * scale).rounded())), max(height, Int((ih * scale).rounded())))
        let r = try ImageArray.resized(image, width: rw, height: rh)
        let x0 = (rw - width) / 2, y0 = (rh - height) / 2
        return r[y0 ..< y0 + height, x0 ..< x0 + width, 0...]
    }

    public func decodeVideo(_ latent: VideoLatent) throws -> VideoClip {
        guard let z = latent.payload as? MLXArray else { throw WanError.emptyLatent }
        lock.lock(); defer { lock.unlock() }
        let video = vae.decode(z.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))  // [1, F, H, W, 3]
        let frames = ((video[0] + 1) / 2).asType(.float32)
        eval(frames)
        Memory.clearCache()
        return VideoClip(frames: frames, fps: latent.fps)
    }
}
