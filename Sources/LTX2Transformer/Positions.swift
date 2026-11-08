import Foundation
import MLX

// MARK: - shapes and positions (ltx_core_mlx/utils/positions.py, components/patchifiers.py)

/// VAE compression factors and audio latent timing used by the LTX-2.3 pipelines.
public enum LTXLatentGeometry {
    public static let videoTemporalScale = 8
    public static let videoSpatialScale = 32
    public static let audioDownsampleFactor = 4
    public static let audioHopLength = 160
    public static let audioSampleRate = 16000
    /// 16000 / 160 / 4 = 25 audio latents per second.
    public static let audioLatentsPerSecond: Double = 25
}

/// Latent `(F, H, W)` for a pixel video: `F = ceil(frames / 8)`, `H = height / 32`, `W = width / 32`.
public func ltxVideoLatentShape(
    numFrames: Int, height: Int, width: Int, temporalCompression: Int = 8, spatialCompression: Int = 32
) -> (frames: Int, height: Int, width: Int) {
    ((numFrames + temporalCompression - 1) / temporalCompression, height / spatialCompression,
     width / spatialCompression)
}

/// Number of audio latent tokens for a video of `numVideoFrames` at `frameRate` (Python `round`, i.e. half-to-even).
public func ltxAudioTokenCount(numVideoFrames: Int, frameRate: Double = 24) -> Int {
    let duration = Double(numVideoFrames) / frameRate
    return Int((duration * LTXLatentGeometry.audioLatentsPerSecond).rounded(.toNearestOrEven))
}

/// RoPE positions of the video tokens, `[1, F*H*W, 3]` float32 (time in seconds, height px, width px).
///
/// Causal temporal positioning: latent frame `i` covers pixel frames `[max(0, 8i - 7), 8i + 1]`; the position is the
/// middle of that range divided by `frameRate`. Spatial positions are pixel-cell midpoints (`32h + 16`).
public func ltxVideoPositions(frames: Int, height: Int, width: Int, frameRate: Double = 24) -> MLXArray {
    let ts = Float(LTXLatentGeometry.videoTemporalScale)
    let ss = Float(LTXLatentGeometry.videoSpatialScale)
    let fps = Float(frameRate)
    let fMid: [Float] = (0 ..< frames).map { i in
        let idx = Float(i)
        let s = max(idx * ts + 1 - ts, 0)
        let e = max((idx + 1) * ts + 1 - ts, 0)
        return (s + e) / 2 / fps
    }
    var out = [Float]()
    out.reserveCapacity(frames * height * width * 3)
    for f in 0 ..< frames {
        for h in 0 ..< height {
            for w in 0 ..< width {
                out.append(fMid[f])
                out.append(Float(h) * ss + ss / 2)
                out.append(Float(w) * ss + ss / 2)
            }
        }
    }
    return MLXArray(out, [1, frames * height * width, 3])
}

/// RoPE positions of the audio tokens, `[1, T, 1]` float32 (middle of each latent's causal time span, seconds).
public func ltxAudioPositions(numTokens: Int) -> MLXArray {
    let d = Float(LTXLatentGeometry.audioDownsampleFactor)
    let hop = Float(LTXLatentGeometry.audioHopLength)
    let sr = Float(LTXLatentGeometry.audioSampleRate)
    let mids: [Float] = (0 ..< numTokens).map { i in
        let idx = Float(i)
        let s = max(idx * d + 1 - d, 0) * hop / sr
        let e = max((idx + 1) * d + 1 - d, 0) * hop / sr
        return (s + e) / 2
    }
    return MLXArray(mids, [1, numTokens, 1])
}

/// `[B, C, F, H, W]` video latent → `[B, F*H*W, C]` tokens.
public func ltxPatchifyVideo(_ latent: MLXArray) -> MLXArray {
    let (b, c, f, h, w) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3), latent.dim(4))
    return latent.transposed(0, 2, 3, 4, 1).reshaped(b, f * h * w, c)
}

/// `[B, F*H*W, C]` tokens → `[B, C, F, H, W]` video latent.
public func ltxUnpatchifyVideo(_ tokens: MLXArray, frames: Int, height: Int, width: Int) -> MLXArray {
    let (b, c) = (tokens.dim(0), tokens.dim(2))
    return tokens.reshaped(b, frames, height, width, c).transposed(0, 4, 1, 2, 3)
}

/// `[B, 8, T, 16]` audio latent → `[B, T, 128]` tokens.
public func ltxPatchifyAudio(_ latent: MLXArray) -> MLXArray {
    let (b, c1, t, c2) = (latent.dim(0), latent.dim(1), latent.dim(2), latent.dim(3))
    return latent.transposed(0, 2, 1, 3).reshaped(b, t, c1 * c2)
}

/// `[B, T, 128]` tokens → `[B, 8, T, 16]` audio latent.
public func ltxUnpatchifyAudio(_ tokens: MLXArray, channels: Int = 8, melBins: Int = 16) -> MLXArray {
    let (b, t) = (tokens.dim(0), tokens.dim(1))
    return tokens.reshaped(b, t, channels, melBins).transposed(0, 2, 1, 3)
}

// MARK: - RoPE (model/transformer/rope.py, SPLIT layout)

/// Per-head SPLIT-RoPE tables: `cos`/`sin` of shape `[B, heads, N, headDim / 2]`, float32.
public struct LTXRope {
    public let cos: MLXArray
    public let sin: MLXArray

    /// Log-spaced frequency grid `theta^linspace(0, 1, n) * pi/2` (`generate_freq_grid`).
    ///
    /// `linspace` is evaluated in float32 (as MLX does); with `doublePrecision` the power and scaling run in Double
    /// before the final cast to float32, matching the reference's CPU float64 path.
    static func freqGrid(theta: Double, numPosDims: Int, innerDim: Int, doublePrecision: Bool) -> [Float] {
        let n = innerDim / (2 * numPosDims)
        let t: [Float] = (0 ..< n).map { n == 1 ? 0 : Float($0) / Float(n - 1) }
        if doublePrecision {
            return t.map { Float(Foundation.pow(theta, Double($0)) * (Double.pi / 2)) }
        }
        return t.map { Foundation.pow(Float(theta), $0) * Float(Double.pi / 2) }
    }

    /// `positions`: `[B, N, P]` (P position axes); `maxPos`: per-axis normaliser.
    public init(positions: MLXArray, heads: Int, headDim: Int, theta: Double, maxPos: [Int], doublePrecision: Bool) {
        let (b, n, p) = (positions.dim(0), positions.dim(1), positions.dim(2))
        let inner = heads * headDim
        let grid = MLXArray(Self.freqGrid(theta: theta, numPosDims: p, innerDim: inner,
                                          doublePrecision: doublePrecision))
        let frac = stacked((0 ..< p).map { i in positions[0..., 0..., i].asType(.float32) / Float(maxPos[i]) },
                           axis: -1)
        let scaled = grid * (frac.expandedDimensions(axis: -1) * 2.0 - 1.0)  // [B, N, P, nf]
        var freqs = scaled.transposed(0, 1, 3, 2).reshaped(b, n, -1)
        let pad = inner / 2 - freqs.dim(2)
        if pad > 0 {
            freqs = concatenated([MLXArray.zeros([b, n, pad], type: Float.self), freqs], axis: -1)
        }
        let half = inner / (2 * heads)
        cos = MLX.cos(freqs).reshaped(b, n, heads, half).transposed(0, 2, 1, 3)
        sin = MLX.sin(freqs).reshaped(b, n, heads, half).transposed(0, 2, 1, 3)
    }
}

/// `apply_rope_split`: rotate the two halves of the head dimension. `x`: `[B, H, N, D]`.
private let ropeSplitCompiled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: false) { a in
    let (x, c0, s0) = (a[0], a[1], a[2])
    let c = c0.asType(x.dtype)
    let s = s0.asType(x.dtype)
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return [concatenated([x1 * c - x2 * s, x1 * s + x2 * c], axis: -1)]
}

func applyRopeSplit(_ x: MLXArray, _ r: LTXRope) -> MLXArray {
    ropeSplitCompiled([x, r.cos, r.sin])[0]
}
