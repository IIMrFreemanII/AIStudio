import Foundation
import MLX

/// Audio samples `[channels, frames]`, float32 in `[-1, 1]`.
public struct AudioClip: @unchecked Sendable {
    public var samples: MLXArray
    public var sampleRate: Int

    public init(samples: MLXArray, sampleRate: Int) {
        self.samples = samples.ndim == 1 ? samples.expandedDimensions(axis: 0) : samples
        self.sampleRate = sampleRate
    }

    public var channels: Int { samples.dim(0) }
    public var frameCount: Int { samples.dim(1) }
    public var duration: Double { Double(frameCount) / Double(sampleRate) }
}

/// Video frames `[T, H, W, 3]`, float32 in `[0, 1]`, with an optional soundtrack.
public struct VideoClip: @unchecked Sendable {
    public var frames: MLXArray
    public var fps: Double
    public var audio: AudioClip?

    public init(frames: MLXArray, fps: Double, audio: AudioClip? = nil) {
        self.frames = frames
        self.fps = fps
        self.audio = audio
    }

    public var frameCount: Int { frames.dim(0) }
    public var height: Int { frames.dim(1) }
    public var width: Int { frames.dim(2) }
    public var duration: Double { Double(frameCount) / fps }
}

/// Pure operations on clips (mixing, resampling, test signals).
public enum MediaOps {
    /// Linear-interpolation resampling (good enough for mixing; models produce audio at their own rate).
    public static func resample(_ clip: AudioClip, to rate: Int) -> AudioClip {
        guard rate != clip.sampleRate, clip.frameCount > 1 else { return clip }
        let n = max(1, Int((Double(clip.frameCount) * Double(rate) / Double(clip.sampleRate)).rounded()))
        let pos = MLXArray(Array(0..<n).map { Float($0) * Float(clip.sampleRate) / Float(rate) })
        let i0 = MLX.clip(floor(pos), min: 0, max: Float(clip.frameCount - 1)).asType(.int32)
        let i1 = minimum(i0 + 1, MLXArray(Int32(clip.frameCount - 1)))
        let t = pos - floor(pos)
        let x = clip.samples.asType(.float32)
        let out = x[0..., i0] * (1 - t) + x[0..., i1] * t
        return AudioClip(samples: out, sampleRate: rate)
    }

    /// Same channel count: mono is duplicated to stereo; more channels than wanted are averaged down.
    public static func channels(_ clip: AudioClip, _ count: Int) -> AudioClip {
        guard clip.channels != count else { return clip }
        if clip.channels == 1 { return AudioClip(samples: repeated(clip.samples, count: count, axis: 0), sampleRate: clip.sampleRate) }
        let mono = clip.samples.mean(axis: 0, keepDims: true)
        return AudioClip(samples: count == 1 ? mono : repeated(mono, count: count, axis: 0), sampleRate: clip.sampleRate)
    }

    /// `a` with `b` (× `gain`) added starting `offset` seconds in. The result is as long as both need.
    public static func mix(_ a: AudioClip, _ b: AudioClip, gain: Float = 1, offset: Double = 0) -> AudioClip {
        let bb = channels(resample(b, to: a.sampleRate), a.channels)
        let start = max(0, Int((offset * Double(a.sampleRate)).rounded()))
        let length = max(a.frameCount, start + bb.frameCount)
        let pa = padded(a.samples.asType(.float32), widths: [0, .init((0, length - a.frameCount))])
        let pb = padded(bb.samples.asType(.float32) * gain, widths: [0, .init((start, length - start - bb.frameCount))])
        return AudioClip(samples: MLX.clip(pa + pb, min: -1, max: 1), sampleRate: a.sampleRate)
    }

    /// `audio` placed `offset` seconds into a track exactly `seconds` long (trimmed or padded with silence).
    public static func fit(_ audio: AudioClip, seconds: Double, offset: Double = 0, gain: Float = 1) -> AudioClip {
        let total = max(1, Int((seconds * Double(audio.sampleRate)).rounded()))
        let silence = AudioClip(samples: MLXArray.zeros([audio.channels, total]), sampleRate: audio.sampleRate)
        let mixed = mix(silence, audio, gain: gain, offset: offset)
        return AudioClip(samples: mixed.samples[0..., ..<total], sampleRate: audio.sampleRate)
    }

    /// Short fades at both ends so a clip doesn't click.
    public static func fade(_ clip: AudioClip, seconds: Double = 0.01) -> AudioClip {
        let n = min(clip.frameCount / 2, Int(seconds * Double(clip.sampleRate)))
        guard n > 1 else { return clip }
        var env = [Float](repeating: 1, count: clip.frameCount)
        for i in 0..<n {
            let v = Float(i) / Float(n)
            env[i] = v
            env[clip.frameCount - 1 - i] = v
        }
        return AudioClip(samples: clip.samples * MLXArray(env), sampleRate: clip.sampleRate)
    }

    /// A sine tone (with short fades), `channels × frames`.
    public static func tone(frequency: Double, seconds: Double, sampleRate: Int, channels: Int = 2, amplitude: Float = 0.3)
        -> AudioClip
    {
        let n = max(1, Int(seconds * Double(sampleRate)))
        let t = MLXArray(Array(0..<n).map { Float($0) / Float(sampleRate) })
        let s = amplitude * sin(Float(2 * Double.pi * frequency) * t)
        return fade(AudioClip(samples: repeated(s.expandedDimensions(axis: 0), count: channels, axis: 0),
                              sampleRate: sampleRate))
    }

    /// Moving colour test pattern: a hue sweep with a white bar crossing left to right, `[T, H, W, 3]`.
    public static func testPattern(width: Int, height: Int, frames: Int, fps: Double) -> VideoClip {
        let x = MLXArray(Array(0..<width).map { Float($0) / Float(max(width - 1, 1)) }).reshaped(1, 1, width, 1)
        let y = MLXArray(Array(0..<height).map { Float($0) / Float(max(height - 1, 1)) }).reshaped(1, height, 1, 1)
        let t = MLXArray(Array(0..<frames).map { Float($0) / Float(max(frames, 1)) }).reshaped(frames, 1, 1, 1)
        let phase = Float(2 * Double.pi)
        let r = 0.5 + 0.5 * sin(phase * (x + t))
        let g = 0.5 + 0.5 * sin(phase * (y + t + 0.33))
        let b = 0.5 + 0.5 * sin(phase * (x + y - t + 0.66))
        var rgb = concatenated([r * MLXArray.ones([frames, height, width, 1]), g * MLXArray.ones([frames, height, width, 1]),
                                b * MLXArray.ones([frames, height, width, 1])], axis: 3)
        let bar = (abs(x - t) .< 0.02).asType(.float32)  // [T, 1, W, 1]
        rgb = rgb * (1 - bar) + bar
        eval(rgb)
        return VideoClip(frames: rgb, fps: fps)
    }
}
