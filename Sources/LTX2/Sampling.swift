import Foundation
import MLX

/// Sigma schedules and the joint audio-video Euler loop of the distilled LTX-2.3 pipeline
/// (`ltx_pipelines_mlx.scheduler`, `utils.samplers.denoise_loop`, `utils.helpers.create_noised_state`).
public enum LTXSchedule {
    /// Stage 1 (half resolution, from pure noise): 8 steps.
    public static let distilled: [Double] = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
    /// Stage 2 (full resolution refinement of the upscaled stage-1 result): 3 steps.
    public static let stage2: [Double] = [0.909375, 0.725, 0.421875, 0.0]

    /// A `steps`-step version of `table` that still ends at its last value (`shorten_schedule`).
    /// `keepStart` pins `table[0]` then takes the last `steps` sigmas (for a stage that starts from noise);
    /// otherwise the last `steps + 1` sigmas are kept (a refinement stage).
    public static func shortened(_ table: [Double], steps: Int?, keepStart: Bool) -> [Double] {
        guard let steps, steps > 0, steps < table.count - 1 else { return table }
        if keepStart { return [table[0]] + Array(table.suffix(steps)) }
        return Array(table.suffix(steps + 1))
    }
}

/// One modality's denoising state (`LatentState`): tokens `[1, N, 128]`, the clean tokens the conditioning pins,
/// and a per-token mask `[1, N, 1]` (1 = generate, 0 = keep).
public struct LTXLatentState {
    public var latent: MLXArray
    public var clean: MLXArray
    public var mask: MLXArray
    public var positions: MLXArray

    /// True when every token is generated (no conditioning): the model then takes one timestep for all tokens.
    public var isUniform: Bool { mask.min().item(Float.self) == 1 }
}

/// An image pinned at a latent frame (`VideoConditionByLatentIndex`): its tokens replace that frame's and are kept.
public struct LTXFrameCondition {
    public var latentFrame: Int
    /// `[1, H*W, 128]` encoded image tokens.
    public var tokens: MLXArray
    public var strength: Float = 1

    public init(latentFrame: Int, tokens: MLXArray, strength: Float = 1) {
        self.latentFrame = latentFrame; self.tokens = tokens; self.strength = strength
    }

    func apply(_ s: LTXLatentState, tokensPerFrame: Int) -> LTXLatentState {
        let start = latentFrame * tokensPerFrame, end = start + tokensPerFrame
        guard tokens.dim(1) >= tokensPerFrame, end <= s.latent.dim(1) else { return s }
        let t = tokens[0..., 0..<tokensPerFrame, 0...]
        func splice(_ a: MLXArray, _ mid: MLXArray) -> MLXArray {
            concatenated([a[0..., 0..<start, 0...], mid, a[0..., end..., 0...]], axis: 1)
        }
        // mx.full defaults to float32, so the spliced mask is promoted exactly as in the reference.
        let frameMask = MLXArray.full([s.mask.dim(0), tokensPerFrame, 1], values: MLXArray(1 - strength))
        var out = s
        out.latent = splice(s.latent, t)
        out.clean = splice(s.clean, t)
        out.mask = splice(s.mask, frameMask)
        return out
    }
}

public enum LTXSampler {
    /// `create_noised_state`. With `scalarBlend` (video, both stages) the noise is blended in before the
    /// conditioning is applied, as the reference's `legacy_scalar_blend`; otherwise (`noise_latent_state`, audio
    /// stage 2) the blend goes through the bf16 mask.
    public static func noisedState(shape: [Int], initial: MLXArray?, positions: MLXArray, seed: UInt64, sigma: Double,
                                   conditions: [LTXFrameCondition] = [], tokensPerFrame: Int = 0,
                                   scalarBlend: Bool) -> LTXLatentState {
        let latent = initial ?? MLXArray.zeros(shape, dtype: .bfloat16)
        let mask = MLXArray.ones([shape[0], shape[1], 1], dtype: .bfloat16)
        var s = LTXLatentState(latent: latent, clean: latent, mask: mask, positions: positions)
        MLXRandom.seed(seed)
        if scalarBlend {
            let noise = MLXRandom.normal(s.clean.shape).asType(.bfloat16)
            s.latent = noise * Float(sigma) + s.clean * Float(1 - sigma)
            for c in conditions { s = c.apply(s, tokensPerFrame: tokensPerFrame) }
        } else {
            for c in conditions { s = c.apply(s, tokensPerFrame: tokensPerFrame) }
            let noise = MLXRandom.normal(s.clean.shape).asType(s.clean.dtype)
            let scaled = s.mask * Float(sigma)
            s.latent = noise * scaled + s.clean * (1 - scaled)
        }
        return s
    }

    /// `x0 = model(video, audio, sigma, videoTimesteps, audioTimesteps)`.
    public typealias X0 = (_ video: MLXArray, _ audio: MLXArray, _ sigma: MLXArray, _ videoTimesteps: MLXArray?,
                           _ audioTimesteps: MLXArray?) throws -> (video: MLXArray, audio: MLXArray)

    /// The Euler loop over consecutive sigma pairs (`denoise_loop`). `onStep(index, count, videoX0)`.
    public static func denoise(video: LTXLatentState, audio: LTXLatentState, sigmas: [Double], model: X0,
                               onStep: ((Int, Int, MLXArray) throws -> Void)? = nil) throws
        -> (video: MLXArray, audio: MLXArray)
    {
        var vx = video.latent, ax = audio.latent
        let vUniform = video.isUniform, aUniform = audio.isUniform
        let pairs = Array(zip(sigmas.dropLast(), sigmas.dropFirst()))
        for (i, (sigma, next)) in pairs.enumerated() {
            try Task.checkCancellation()
            let s = MLXArray([Float(sigma)]).asType(.bfloat16)
            let vt = vUniform ? nil : (video.mask * Float(sigma)).squeezed(axis: -1)
            let at = aUniform ? nil : (audio.mask * Float(sigma)).squeezed(axis: -1)
            var (v0, a0) = try model(vx, ax, s, vt, at)
            v0 = v0 * video.mask + video.clean * (1 - video.mask)
            a0 = a0 * audio.mask + audio.clean * (1 - audio.mask)
            vx = euler(vx, v0, sigma, next)
            ax = euler(ax, a0, sigma, next)
            eval(vx, ax)
            try onStep?(i, pairs.count, v0)
        }
        return (vx, ax)
    }

    /// `x + (σ' − σ) · (x − x0) / σ` (`mlx_arsenal.diffusion.euler_step`).
    static func euler(_ x: MLXArray, _ x0: MLXArray, _ sigma: Double, _ next: Double) -> MLXArray {
        if sigma == 0 { return x0 }
        return x + Float(next - sigma) * ((x - x0) / Float(sigma))
    }
}
