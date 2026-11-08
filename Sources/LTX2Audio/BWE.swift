import Foundation
import MLX
import MLXDiffusionCore

/// 3× Hann-windowed-sinc upsampler (`HannSincResampler`, rolloff 0.99, 6 zero crossings, 43 taps). No weights.
public struct LTXHannSincResampler {
    public let factor: Int
    let width: Int
    let kernel: MLXArray  // [1, K, 1]

    public init(factor: Int = 3) {
        self.factor = factor
        let rolloff = 0.99
        let lpfw = 6.0
        width = Int((lpfw / rolloff).rounded(.up))  // 7
        let k = 2 * width * factor + 1
        var taps = [Float](repeating: 0, count: k)
        for i in 0 ..< k {
            let t = (Double(i) / Double(factor) - Double(width)) * rolloff
            let tc = min(max(t, -lpfw), lpfw)
            let w = pow(cos(tc * Double.pi / lpfw / 2), 2)
            let sinc = t == 0 ? 1.0 : sin(Double.pi * t) / (Double.pi * t)
            taps[i] = Float(sinc * w * rolloff / Double(factor))
        }
        kernel = MLXArray(taps).reshaped(1, k, 1)
    }

    /// `x`: `[N, T]` float32 → `[N, T · factor]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (n, t) = (x.dim(0), x.dim(1))
        let r = factor
        let k = kernel.dim(1)
        // Replicate-pad by `width`, zero-insert (conv_transpose style), full conv, ×ratio, slice.
        let xp = concatenated([repeated(x[0..., ..<1], count: width, axis: 1), x,
                               repeated(x[0..., (t - 1)...], count: width, axis: 1)], axis: 1)
        let tp = xp.dim(1)
        var parts = [xp]
        for _ in 1 ..< r { parts.append(MLXArray.zeros(like: xp)) }
        var z = stacked(parts, axis: 2).reshaped(n, tp * r)[0..., ..<((tp - 1) * r + 1)]
        z = padded(z.expandedDimensions(axis: -1), widths: [0, .init(k - 1), 0])
        var y = conv1d(z, kernel).squeezed(axis: -1) * Float(r)
        let padLeft = 2 * width * r
        let padRight = k - r
        y = y[0..., padLeft ..< (y.dim(1) - padRight)]
        return y[0..., ..<(t * r)]
    }
}

/// Causal log-magnitude mel spectrogram used as the BWE generator input (`MelSTFT`: n_fft 512, hop 80, 64 mels),
/// computed with the STFT basis and mel filterbank stored in the checkpoint.
public struct LTXMelSTFT {
    let forwardBasis: MLXArray  // [n_fft + 2, n_fft, 1]
    let melBasis: MLXArray      // [n_mels, n_fft/2 + 1]
    public let nFFT: Int
    public let hopLength: Int

    public init(store: WeightStore, prefix: String, hopLength: Int = 80) throws {
        forwardBasis = try store.get(prefix + "stft_fn.forward_basis").asType(.float32)
        melBasis = try store.get(prefix + "mel_basis").asType(.float32)
        nFFT = forwardBasis.dim(1)
        self.hopLength = hopLength
    }

    /// `wave`: `[N, T]` → log-mel `[N, frames, n_mels]`.
    public func callAsFunction(_ wave: MLXArray) -> MLXArray {
        let leftPad = max(0, nFFT - hopLength)
        let x = padded(wave.expandedDimensions(axis: -1), widths: [0, .init((leftPad, 0)), 0])
        let s = conv1d(x, forwardBasis, stride: hopLength)
        let bins = nFFT / 2 + 1
        let re = s[0..., 0..., ..<bins]
        let im = s[0..., 0..., bins...]
        let mag = sqrt(re * re + im * im + Float(1e-9))
        let mel = matmul(mag, melBasis.T)
        return log(maximum(mel, Float(1e-5)))
    }
}

/// Mel → 48 kHz stereo waveform (`VocoderWithBWE`): base BigVGAN to 16 kHz (tanh), causal mel of that signal,
/// BWE BigVGAN residual at 48 kHz, plus the 3× sinc-resampled base signal, clipped to [−1, 1].
/// Everything runs in float32 (the reference upcasts all vocoder weights).
public final class LTXVocoderWithBWE {
    public let base: LTXBigVGAN
    public let bwe: LTXBigVGAN
    public let melSTFT: LTXMelSTFT
    public let resampler: LTXHannSincResampler
    /// Rate of the base vocoder output.
    public let baseSampleRate: Int
    /// Rate of the final output.
    public let outputSampleRate: Int

    /// `store` holds `vocoder.safetensors`; `config` is the `vocoder` section of `embedded_config.json` (optional).
    public init(store: WeightStore, prefix: String = "vocoder.", config: [String: Any]? = nil) throws {
        let bweJSON = config?["bwe"] as? [String: Any]
        base = try LTXBigVGAN(store: store, prefix: prefix,
                              config: LTXBigVGANConfig(json: config?["vocoder"] as? [String: Any], fallback: .ltxBase))
        bwe = try LTXBigVGAN(store: store, prefix: prefix + "bwe_generator.",
                             config: LTXBigVGANConfig(json: bweJSON, fallback: .ltxBWE))
        melSTFT = try LTXMelSTFT(store: store, prefix: prefix + "mel_stft.",
                                 hopLength: bweJSON?["hop_length"] as? Int ?? 80)
        baseSampleRate = bweJSON?["input_sampling_rate"] as? Int ?? 16000
        outputSampleRate = bweJSON?["output_sampling_rate"] as? Int ?? 48000
        resampler = LTXHannSincResampler(factor: outputSampleRate / baseSampleRate)
        eval(base.arrays + bwe.arrays + [melSTFT.forwardBasis, melSTFT.melBasis, resampler.kernel])
    }

    /// Stereo mel `[B, 2, T, 64]` → `[B, 2, T · 160]` at 16 kHz (base vocoder only).
    public func baseWaveform(mel: MLXArray) -> MLXArray {
        let (b, c, t, m) = (mel.dim(0), mel.dim(1), mel.dim(2), mel.dim(3))
        // channel index = c·64 + m (the two mel channels are concatenated, not processed separately)
        let x = mel.asType(.float32).transposed(0, 2, 1, 3).reshaped(b, t, c * m)
        return base(x).transposed(0, 2, 1)
    }

    /// Stereo mel `[B, 2, T, 64]` → `[B, 2, T · 160 · 3]` at 48 kHz, float32 in [−1, 1].
    /// `taps` receives `wave16k`, `bwe_mel`, `bwe_residual`, `skip48k` for testing.
    public func callAsFunction(_ mel: MLXArray, taps: ((String, MLXArray) -> Void)? = nil) -> MLXArray {
        let (b, c, m) = (mel.dim(0), mel.dim(1), mel.dim(3))
        var w16 = baseWaveform(mel: mel)
        taps?("wave16k", w16)
        let length16 = w16.dim(2)
        let outLength = length16 * resampler.factor
        let hop = melSTFT.hopLength
        if length16 % hop != 0 {
            w16 = padded(w16, widths: [0, 0, .init((0, hop - length16 % hop))])
        }
        let flat = w16.reshaped(b * c, -1)
        let bm = melSTFT(flat)  // [B·2, frames, 64]
        taps?("bwe_mel", bm)
        let frames = bm.dim(1)
        let bx = bm.reshaped(b, c, frames, m).transposed(0, 2, 1, 3).reshaped(b, frames, c * m)
        let residual = bwe(bx).transposed(0, 2, 1)  // [B, 2, T48]
        taps?("bwe_residual", residual)
        let skip = resampler(flat).reshaped(b, c, -1)
        taps?("skip48k", skip)
        let n = min(skip.dim(2), residual.dim(2))
        let out = clip(skip[0..., 0..., ..<n] + residual[0..., 0..., ..<n], min: -1, max: 1)
        return out[0..., 0..., ..<min(n, outLength)]
    }
}
