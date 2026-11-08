import Foundation
import MLX
import MLXDiffusionCore

/// LTX-2.3 audio decode, exactly as the distilled pipeline does it (`ltx_pipelines_mlx.utils.blocks.AudioDecoder`):
/// transformer audio tokens `[1, T, 128]` → audio-VAE latent `[1, 8, T, 16]` → stereo mel `[1, 2, 4T−3, 64]`
/// (bfloat16 VAE) → 16 kHz BigVGAN → bandwidth extension → 48 kHz stereo waveform.
///
/// T tokens give `(4T − 3) · 480` samples at 48 kHz (25 tokens/s; e.g. T = 126 → 240 480 samples ≈ 5.01 s).
public final class LTXAudioDecoder {
    public let vae: LTXAudioVAEDecoder
    public let vocoder: LTXVocoderWithBWE
    /// When true (the reference behaviour) the waveform is rounded to the mel dtype (bfloat16) before being
    /// returned as float32 — `VocoderWithBWE` casts its output back to its input dtype.
    public var roundOutputToMelDType: Bool

    /// Output sample rate (48 kHz for LTX-2.3).
    public var sampleRate: Int { vocoder.outputSampleRate }

    /// `directory` holds `audio_vae.safetensors`, `vocoder.safetensors` and (optionally) `embedded_config.json`.
    /// `vaeDType` is the audio-VAE compute dtype (reference: bfloat16); the vocoder always runs in float32.
    public init(directory: URL, vaeDType: DType = .bfloat16, roundOutputToMelDType: Bool = true) throws {
        let vaeStore = try WeightStore(url: directory.appendingPathComponent("audio_vae.safetensors"))
        vae = try LTXAudioVAEDecoder(store: vaeStore, dtype: vaeDType)
        let vocStore = try WeightStore(url: directory.appendingPathComponent("vocoder.safetensors"))
        var vocConfig: [String: Any]?
        let cfgURL = directory.appendingPathComponent("embedded_config.json")
        if let data = try? Data(contentsOf: cfgURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            vocConfig = obj["vocoder"] as? [String: Any]
        }
        vocoder = try LTXVocoderWithBWE(store: vocStore, config: vocConfig)
        self.roundOutputToMelDType = roundOutputToMelDType
    }

    /// Audio-VAE latent `[B, 8, T, 16]` → mel `[B, 2, 4T−3, 64]` (VAE dtype).
    public func mel(latent: MLXArray) -> MLXArray { vae.decode(latent: latent) }

    /// Audio-VAE latent `[B, 8, T, 16]` → waveform `[B, 2, N]` float32 at ``sampleRate`` (evaluated).
    /// The decode churns through many odd-sized buffers, so while it runs the MLX buffer-cache limit is lowered to
    /// ``decodeCacheLimit`` (restored afterwards) and the cache is cleared at the end unless
    /// ``clearsCacheAfterDecode`` is false.
    public func decode(latent: MLXArray) -> MLXArray {
        let savedLimit = Memory.cacheLimit
        if let decodeCacheLimit { Memory.cacheLimit = min(savedLimit, decodeCacheLimit) }
        defer { if decodeCacheLimit != nil { Memory.cacheLimit = savedLimit } }
        let m = vae.decode(latent: latent)
        eval(m)
        var w = vocoder(m)
        if roundOutputToMelDType { w = w.asType(m.dtype) }
        w = w.asType(.float32)
        eval(w)
        if clearsCacheAfterDecode { Memory.clearCache() }
        return w
    }

    /// See ``decode(latent:)``.
    public var clearsCacheAfterDecode = true
    /// Buffer-cache limit in bytes while decoding (nil leaves the global limit alone). See ``decode(latent:)``.
    public var decodeCacheLimit: Int? = 1 << 30

    /// Transformer audio tokens `[1, T, 128]` → (`samples` `[2, N]` float32, `sampleRate`).
    public func decode(tokens: MLXArray) -> (samples: MLXArray, sampleRate: Int) {
        let w = decode(latent: LTXAudioPatchifier.unpatchify(tokens))
        let s = w[0]
        eval(s)
        return (s, sampleRate)
    }

    /// ``decode(tokens:)`` wrapped as an ``AudioClip``.
    public func decodeClip(tokens: MLXArray) -> AudioClip {
        let (s, r) = decode(tokens: tokens)
        return AudioClip(samples: s, sampleRate: r)
    }

    /// Number of 48 kHz samples produced for `tokens` latent frames.
    public func sampleCount(forTokens tokens: Int) -> Int {
        vae.melFrames(forTokens: tokens) * vocoder.base.hopLength * vocoder.resampler.factor
    }
}
