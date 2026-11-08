import Foundation
import LTX2Audio
import LTX2Text
import LTX2Transformer
import LTX2VAE
import MLX
import MLXDiffusionCore

/// Sampled latents of a clip at full resolution: video tokens `[1, F*H*W, 128]` and audio tokens `[1, T, 128]`.
public struct LTXVideoLatents {
    public var video: MLXArray
    public var audio: MLXArray
    /// Latent grid (frames, height, width) of `video`.
    public var frames: Int
    public var height: Int
    public var width: Int
}

/// What `encodeVideoPrompt` returns: Gemma + connector features for the video and audio streams.
struct LTXPromptEmbeds {
    var video: MLXArray
    var audio: MLXArray
}

/// LTX-2.3 distilled: text/image-to-video with a synchronized soundtrack, in two stages
/// (8 steps at half resolution, 2× latent upscale, 3 refinement steps), as `ltx_pipelines_mlx.distilled`.
///
/// Folder layout (the catalog download): the `dgrauet/ltx-2.3-mlx-q4` files, the Gemma 3 12B text encoder in
/// `text_encoder/` (or a folder named by `text_encoder` in `mlxcomfy.json`), and `mlxcomfy.json` with
/// `"class": "LTX2Pipeline"`.
public final class LTX2Pipeline: VideoDiffusionModel, @unchecked Sendable {
    public static let className = "LTX2Pipeline"
    public static let family = "ltx-2.3"

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await LTX2Pipeline(directory: url) }
    }

    public var family: String { Self.family }
    public var displayName: String { "LTX-2.3 Distilled (\(directory.lastPathComponent))" }

    public let directory: URL
    public let textEncoder: LTXTextEncoder
    public let transformer: LTXTransformer
    public let x0: LTXX0Model
    public let videoEncoder: LTXVideoEncoder
    public let videoDecoder: LTXVideoDecoder
    public let upsampler: LTXLatentUpsampler
    public let audioDecoder: LTXAudioDecoder
    /// Serializes generations: the components keep no per-call state, but two concurrent runs would double memory.
    let lock = NSLock()

    public init(directory: URL, log: (String) -> Void = { _ in }) async throws {
        self.directory = directory
        var t = Date()
        func lap(_ what: String) {
            log(String(format: "%@ %.1fs", what, Date().timeIntervalSince(t)))
            t = Date()
        }
        textEncoder = try await LTXTextEncoder(gemmaDirectory: Self.textEncoderDirectory(in: directory),
                                               ltxDirectory: directory)
        lap("text encoder")
        let ts = try WeightStore(url: try Self.transformerFile(in: directory))
        ts.bitsOverride = Self.quantizationBits(in: directory)
        transformer = try LTXTransformer(store: ts, config: try LTXTransformerConfig(directory: directory))
        x0 = LTXX0Model(transformer)
        lap("transformer")
        videoEncoder = try LTXVideoEncoder(url: directory.appendingPathComponent("vae_encoder.safetensors"))
        videoDecoder = try LTXVideoDecoder(url: directory.appendingPathComponent("vae_decoder.safetensors"))
        upsampler = try LTXLatentUpsampler(
            url: directory.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors"),
            configURL: directory.appendingPathComponent("spatial_upscaler_x2_v1_1_config.json"))
        audioDecoder = try LTXAudioDecoder(directory: directory)
        lap("VAE, upscaler, audio")
    }

    static func textEncoderDirectory(in dir: URL) -> URL {
        let sub = dir.appendingPathComponent("text_encoder")
        if FileManager.default.fileExists(atPath: sub.appendingPathComponent("config.json").path) { return sub }
        if let data = try? Data(contentsOf: dir.appendingPathComponent("mlxcomfy.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let rel = obj["text_encoder"] as? String {
            return URL(fileURLWithPath: rel, relativeTo: dir).standardizedFileURL
        }
        return dir.deletingLastPathComponent().appendingPathComponent("gemma-3-12b-it-4bit")
    }

    static func transformerFile(in dir: URL) throws -> URL {
        for name in ["transformer-distilled-1.1", "transformer-distilled"] {
            let u = dir.appendingPathComponent(name + ".safetensors")
            if FileManager.default.fileExists(atPath: u.path) { return u }
        }
        throw WeightError.missing("transformer-distilled-1.1.safetensors in \(dir.path)")
    }

    static func quantizationBits(in dir: URL) -> Int? {
        guard let data = try? Data(contentsOf: dir.appendingPathComponent("quantize_config.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let q = obj["quantization"] as? [String: Any] else { return nil }
        return q["bits"] as? Int
    }

    /// The reference re-encodes conditioning images with H.264 (libx264, CRF 33, yuv420p) so they look like the
    /// compressed frames the model was trained on; we do the same round trip with VideoToolbox.
    static func videoLikeCompression(_ rgb: MLXArray) -> MLXArray { H264RoundTrip.apply(rgb) }

    // MARK: VideoDiffusionModel

    public func encodeVideoPrompt(_ prompt: String) throws -> Conditioning {
        lock.lock(); defer { lock.unlock() }
        let (v, a) = try textEncoder.encode(prompt)
        eval(v, a)
        return Conditioning(family: family, payload: LTXPromptEmbeds(video: v, audio: a))
    }

    /// Two-stage generation needs multiples of 64 (stage 1 runs at half size on a 32-pixel latent grid), and the
    /// causal VAE decodes 1 + 8k frames.
    public func snapVideo(width: Int, height: Int, frames: Int) -> (width: Int, height: Int, frames: Int) {
        let w = max(128, width / 64 * 64), h = max(128, height / 64 * 64)
        let k = max(1, Int((Double(frames - 1) / 8).rounded()))
        return (w, h, 8 * k + 1)
    }

    public func sampleVideo(conditioning: Conditioning, latent: VideoLatent, seed: UInt64, steps: Int?,
                            image: MLXArray?, onStep: ((VideoSamplerStep) throws -> Void)?) throws -> VideoLatent {
        guard let text = conditioning.payload as? LTXPromptEmbeds else {
            throw DiffusionError.wrongConditioning(expected: family, got: conditioning.family)
        }
        lock.lock(); defer { lock.unlock() }
        let (width, height, frames) = snapVideo(width: latent.width, height: latent.height, frames: latent.frames)
        let fps = latent.fps
        let sigmas1 = LTXSchedule.shortened(LTXSchedule.distilled, steps: steps, keepStart: true)
        let sigmas2 = LTXSchedule.stage2
        let total = sigmas1.count - 1 + sigmas2.count - 1
        let fpsD = Double(fps)
        // Stage 1: half resolution, pure noise.
        let (lf, lh, lw) = ltxVideoLatentShape(numFrames: frames, height: height / 2, width: width / 2)
        let audioT = ltxAudioTokenCount(numVideoFrames: frames, frameRate: fpsD)
        let audioPositions = ltxAudioPositions(numTokens: audioT)
        let rgb = image.map { img -> MLXArray in
            let c = img.dim(2) == 4 ? img[0..., 0..., 0..<3] : img
            return (clip(c.asType(.float32), min: 0, max: 1) * 255).round()
        }
        func conditions(h: Int, w: Int) -> [LTXFrameCondition] {
            guard let rgb else { return [] }
            let px = LTXImagePreprocess.conditioningPixels(rgb: rgb, height: h * 32, width: w * 32,
                                                          compress: Self.videoLikeCompression)
            let lat = videoEncoder.encode(px)
            eval(lat)
            return [LTXFrameCondition(latentFrame: 0, tokens: lat.reshaped(1, -1, 128))]
        }
        var step = 0
        func report(_ stage: String, _ x0: MLXArray, _ grid: (Int, Int, Int)) throws {
            step += 1
            guard let onStep else { return }
            let preview = LTXPreview.rgb(tokens: x0, frames: grid.0, height: grid.1, width: grid.2)
            try onStep(VideoSamplerStep(index: step - 1, count: total, stage: stage, preview: preview))
        }
        func run(_ v: LTXLatentState, _ a: LTXLatentState, _ sigmas: [Double], _ stage: String,
                 _ grid: (Int, Int, Int)) throws
            -> (video: MLXArray, audio: MLXArray)
        {
            try LTXSampler.denoise(video: v, audio: a, sigmas: sigmas, model: { [x0] vx, ax, sg, vt, at in
                x0(video: vx, audio: ax, sigma: sg, videoTimesteps: vt, audioTimesteps: at, videoText: text.video,
                   audioText: text.audio, videoPositions: v.positions, audioPositions: a.positions)
            }, onStep: { _, _, x in try report(stage, x, grid) })
        }

        let v1 = LTXSampler.noisedState(
            shape: [1, lf * lh * lw, 128], initial: nil,
            positions: ltxVideoPositions(frames: lf, height: lh, width: lw, frameRate: fpsD), seed: seed, sigma: 1,
            conditions: conditions(h: lh, w: lw), tokensPerFrame: lh * lw, scalarBlend: true)
        let a1 = LTXSampler.noisedState(shape: [1, audioT, 128], initial: nil, positions: audioPositions,
                                        seed: seed &+ 1, sigma: 1, scalarBlend: true)
        let out1 = try run(v1, a1, sigmas1, "Stage 1", (lf, lh, lw))

        // Upscale ×2 in latent space, then refine at full resolution.
        let half = out1.video[0..., 0..<(lf * lh * lw), 0...].reshaped(1, lf, lh, lw, 128)  // channels-last grid
        let up = upsampler.upsampleNormalized(half, encoder: videoEncoder)
        let (fh, fw) = (lh * 2, lw * 2)
        let s0 = sigmas2[0]
        let v2 = LTXSampler.noisedState(
            shape: [1, lf * fh * fw, 128], initial: up.reshaped(1, -1, 128),
            positions: ltxVideoPositions(frames: lf, height: fh, width: fw, frameRate: fpsD), seed: seed &+ 2,
            sigma: s0, conditions: conditions(h: fh, w: fw), tokensPerFrame: fh * fw, scalarBlend: true)
        let a2 = LTXSampler.noisedState(shape: out1.audio.shape, initial: out1.audio, positions: audioPositions,
                                        seed: seed &+ 2, sigma: s0, scalarBlend: false)
        let out2 = try run(v2, a2, sigmas2, "Stage 2", (lf, fh, fw))
        let tokens = out2.video[0..., 0..<(lf * fh * fw), 0...]
        eval(tokens, out2.audio)
        return VideoLatent(payload: LTXVideoLatents(video: tokens, audio: out2.audio, frames: lf, height: fh, width: fw),
                           width: fw * 32, height: fh * 32, frames: frames, fps: fps)
    }

    public func decodeVideo(_ latent: VideoLatent) throws -> VideoClip {
        guard let l = latent.payload as? LTXVideoLatents else {
            throw DiffusionError.wrongConditioning(expected: family, got: "latent")
        }
        lock.lock(); defer { lock.unlock() }
        let grid = l.video.reshaped(1, l.frames, l.height, l.width, 128)  // tokens are (F, H, W)-ordered, channels-last
        // The DiT stays resident, so the decode gets a fixed 8 GB budget rather than half of memory.
        let frames = videoDecoder.decodeFrames(grid, tiling: .auto(budgetBytes: 8 << 30, frameRate: Double(latent.fps)))
        eval(frames)
        Memory.clearCache()
        var audio = audioDecoder.decodeClip(tokens: l.audio)
        // Trim or pad the soundtrack to the clip's length so the two tracks end together.
        let seconds = Double(frames.dim(0)) / Double(latent.fps)
        audio = MediaOps.fit(audio, seconds: seconds)
        return VideoClip(frames: frames, fps: Double(latent.fps), audio: audio)
    }
}
