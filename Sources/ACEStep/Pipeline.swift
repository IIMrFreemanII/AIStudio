import ACEStepDiT
import ACEStepLM
import ACEStepText
import Foundation
import MLX
import MLXDiffusionCore
import StableAudio
import Tokenizers

/// ACE-Step 1.5 Turbo: text + lyrics → stereo 48 kHz music, as the official `generate_music` text-to-music path.
///
/// 1. Optional planner (the 5 Hz LM): writes the song's metadata (bpm, key, time signature, language, a richer
///    caption) and its audio codes; the codes are decoded into 25 Hz "LM hints" that replace the silence source.
/// 2. Qwen3-Embedding encodes the caption prompt; its token embeddings encode the lyrics; the condition encoder
///    packs lyrics, timbre (silence reference) and caption into the cross-attention sequence.
/// 3. The DiT samples 8 Euler steps (linear schedule, DCW correction) from noise, conditioned on that sequence and on
///    the context latents (source + chunk mask).
/// 4. The Oobleck VAE decodes 25 Hz latents to audio in overlapping chunks; the result is peak-normalized to −1 dBFS.
///
/// Folder layout (the catalog download of `ACE-Step/Ace-Step1.5`): `acestep-v15-turbo/` (DiT, condition encoder,
/// code detokenizer, `silence_latent.pt`), `vae/`, `Qwen3-Embedding-0.6B/`, `acestep-5Hz-lm-1.7B/`, and
/// `aistudio.json` with `"class": "ACEStepPipeline"`.
public final class ACEStepPipeline: AudioModel, @unchecked Sendable {
    public static let className = "ACEStepPipeline"

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await ACEStepPipeline(directory: url) }
    }

    public var family: String { "ace-step-1.5" }
    public var displayName: String { "ACE-Step 1.5 (\(directory.lastPathComponent))" }

    public let directory: URL
    public let sampleRate = 48_000
    /// Latent frames per second (48 kHz / 1920).
    public let latentRate = 25
    public static let minSeconds = 10.0, maxSeconds = 240.0
    public static let defaultSteps = 8

    let tokenizer: Tokenizer
    let textEncoder: Qwen3Model
    let condition: ACEConditionEncoder
    let dit: ACEDiT
    let codeDecoder: ACEAudioCodeDecoder
    let vae: OobleckDecoder
    /// `[1, 15000, 64]`: the latent of silence (source for text-to-music, timbre reference, hint padding).
    let silence: MLXArray
    let plannerDirectory: URL?
    let planner: ACEPlanner?
    let lock = NSLock()

    public init(directory: URL) async throws {
        self.directory = directory
        let ditDir = directory.appendingPathComponent("acestep-v15-turbo")
        let teDir = directory.appendingPathComponent("Qwen3-Embedding-0.6B")
        tokenizer = try await AutoTokenizer.from(modelFolder: teDir)
        textEncoder = try Qwen3Model(directory: teDir, dtype: .float32)
        let store = try WeightStore(url: ditDir.appendingPathComponent("model.safetensors"))
        condition = try ACEConditionEncoder(store: store, config: try ACEConditionEncoder.Config(url: ditDir.appendingPathComponent("config.json")))
        store.remove(prefix: "encoder.")
        codeDecoder = try ACEAudioCodeDecoder(store: store, config: try ACEAudioCodeDecoder.Config(url: ditDir.appendingPathComponent("config.json")))
        // fp32 compute on the bf16 weights: bf16 compute saves ~0.2 s per song for ~3000× the error.
        dit = try ACEDiT(store: store, config: try ACEDiT.Config(url: ditDir.appendingPathComponent("config.json")),
                         dtype: .float32)
        store.remove(prefix: "decoder.")
        let vaeStore = try WeightStore(url: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"))
        let vaeCfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("vae/config.json"))))
            as? [String: Any] ?? [:]
        vae = try OobleckDecoder(store: vaeStore, prefix: "decoder.",
                                 strides: vaeCfg["downsampling_ratios"] as? [Int] ?? [2, 4, 4, 6, 10],
                                 dtype: .float32, layout: .diffusers)
        let silenceURL = ditDir.appendingPathComponent("silence_latent.pt")
        let n = try TorchArchive.float32Count(url: silenceURL)
        silence = try TorchArchive.float32(url: silenceURL, shape: [1, 64, n / 64]).transposed(0, 2, 1)
        let lm = directory.appendingPathComponent("acestep-5Hz-lm-1.7B")
        if FileManager.default.fileExists(atPath: lm.appendingPathComponent("config.json").path) {
            plannerDirectory = lm
            planner = try await ACEPlanner(directory: lm)
        } else {
            plannerDirectory = nil
            planner = nil
        }
    }

    /// Whether the song planner (LM) is available in this folder.
    public var hasPlanner: Bool { planner != nil }

    /// Silence latents `[1, n, 64]` (tiled if `n` exceeds the stored 600 s).
    public func silenceFrames(_ n: Int) -> MLXArray {
        var s = silence
        while s.dim(1) < n { s = concatenated([s, silence], axis: 1) }
        return s[0..., 0..<n, 0...]
    }

    /// Token ids `[1, L]` (the tokenizer appends EOS, as in the reference), truncated to `max`.
    public func tokens(_ text: String, max: Int) -> MLXArray {
        var ids = tokenizer.encode(text: text)
        if ids.count > max { ids = Array(ids.prefix(max)) }
        return MLXArray(ids.map { Int32($0) }, [1, ids.count])
    }

    /// The cross-attention sequence for a caption prompt and lyric prompt.
    public func encode(captionPrompt: String, lyricPrompt: String) -> MLXArray {
        let textIDs = tokens(captionPrompt, max: 256)
        let lyricIDs = tokens(lyricPrompt, max: 2048)
        let text = textEncoder(textIDs)
        let lyrics = textEncoder.embed(lyricIDs)
        let (hidden, _) = condition(text: text, textMask: MLXArray.ones([1, textIDs.dim(1)]),
                                    lyrics: lyrics, lyricMask: MLXArray.ones([1, lyricIDs.dim(1)]),
                                    timbre: silence[0..., 0..<750, 0...], timbreOrder: [0])
        return hidden
    }

    public func generateAudio(_ request: AudioRequest, onStep: ((Int, Int) throws -> Void)?) throws -> AudioClip {
        lock.lock(); defer { lock.unlock() }
        let seconds = min(max(request.seconds, Self.minSeconds), Self.maxSeconds)
        let steps = max(1, min(20, request.steps ?? Self.defaultSteps))
        let lyrics = ACEPrompts.normalizedLyrics(request.lyrics)

        var frames = Int((seconds * Double(latentRate)).rounded())
        var textPrompt: String, lyricPrompt: String, source: MLXArray

        // Progress: planner codes (if any), DiT steps, then the decode.
        let codeCount = request.planner && hasPlanner ? Int(seconds * 5) : 0
        let total = codeCount + steps + 1
        if codeCount > 0, let planner {
            let planRequest = ACEPlanRequest(
                caption: request.prompt, lyrics: lyrics, duration: seconds, bpm: request.bpm,
                keyscale: request.key.isEmpty ? nil : request.key,
                timeSignature: request.timeSignature.isEmpty ? nil : request.timeSignature,
                language: request.language.isEmpty ? nil : request.language, seed: request.seed)
            let plan = try planner.plan(planRequest) { phase, i, _ in
                try Task.checkCancellation()
                try onStep?(phase == .codes ? min(i, codeCount) : 0, total)
            }
            // The official handler turns code-hinted text-to-music into its "cover" task: the hints become the
            // source latents and the instruction changes.
            let cond = ACEDiTConditioning.make(plan: plan, request: planRequest,
                                               hints: codeDecoder.hints(codes: plan.codes), silenceLatent: silence)
            textPrompt = cond.textPrompt
            lyricPrompt = cond.lyricsPrompt
            frames = cond.latentLength
            source = cond.srcLatents
        } else {
            let metas = ACEPrompts.metas(bpm: request.bpm, timeSignature: request.timeSignature, keyScale: request.key,
                                         seconds: seconds)
            textPrompt = ACEPrompts.caption(caption: request.prompt, metas: metas)
            lyricPrompt = ACEPrompts.lyrics(lyrics, language: request.language)
            source = silenceFrames(frames)
        }
        try Task.checkCancellation()

        let encoder = encode(captionPrompt: textPrompt, lyricPrompt: lyricPrompt)
        let context = concatenated([source, MLXArray.ones([1, frames, 64])], axis: -1)
        eval(encoder, context)

        let noise = MLXRandom.normal([1, frames, 64], key: MLXRandom.key(request.seed))
        let latents = try sample(noise: noise, context: context, encoder: encoder, steps: steps) { i in
            try Task.checkCancellation()
            try onStep?(codeCount + i, total)
        }
        let audio = decode(latents, seconds: seconds)
        try onStep?(total, total)
        return AudioClip(samples: audio, sampleRate: sampleRate)
    }

    /// The DiT's 8-step turbo sampler (linear schedule, Euler, DCW) from `noise` `[1, T, 64]`.
    public func sample(noise: MLXArray, context: MLXArray, encoder: MLXArray, steps: Int = defaultSteps,
                       onStep: ((Int) throws -> Void)? = nil) throws -> MLXArray {
        let latents = try ACESampler.sample(dit: dit, noise: noise, context: context, encoder: encoder, steps: steps,
                                            shift: 1, onStep: { i, _ in try onStep?(i) })
        eval(latents)
        Memory.clearCache()
        return latents
    }

    /// 25 Hz latents `[1, T, 64]` → stereo `[2, N]` trimmed to `seconds`, peak-normalized to −1 dBFS like the handler
    /// (which divides by the peak when it exceeds 1, then normalizes).
    public func decode(_ latents: MLXArray, seconds: Double) -> MLXArray {
        var audio = vae.decodeTiled(latents.asType(.float32), chunk: 256, overlap: 64)[0].transposed(1, 0)
        let keep = min(audio.dim(1), Int((seconds * Double(sampleRate)).rounded()))
        audio = audio[0..., 0..<keep]
        let target = Float(pow(10, -1.0 / 20))
        audio = audio * (target / maximum(abs(audio).max(), MLXArray(Float(1e-6))))
        eval(audio)
        Memory.clearCache()
        return audio
    }
}
