import Foundation
import MLX

/// ACE-Step 1.5 turbo sampler (`AceStepConditionGenerationModel.generate_audio`, text-to-music path, `ode` + `euler`).
///
/// For each scheduled time `t_i` (descending from 1): `v = DiT(x, t_i)`; if not last, `x ← x − v·(t_i − t_{i+1})`
/// then DCW against `denoised = x_old − v·t_i`; on the last step `x0 = x − v·t_i` (DCW there is an exact no-op
/// because `x_next == denoised`). Timestep `r` passed to the DiT equals `t` (mean-flow `t − r = 0`).
public enum ACESampler {
    /// The schedule `generate_audio` builds when the pipeline passes `infer_steps` (it always does):
    /// `t_i = 1 − i/n` for `i in 0..<n` (n clamped to 20), then `t ← shift·t / (1 + (shift − 1)·t)` if `shift != 1`.
    /// The trailing 0 is implicit (the last step jumps straight to x0). With n = 8, shift = 1 this is
    /// `[1, .875, .75, .625, .5, .375, .25, .125]`, identical to the fixed `SHIFT_TIMESTEPS[1.0]` table.
    public static func schedule(steps: Int, shift: Float) -> [Float] {
        let n = max(1, min(steps, 20))
        return (0..<n).map { i in
            var t = 1.0 - Double(i) / Double(n)
            if shift != 1 { t = Double(shift) * t / (1 + (Double(shift) - 1) * t) }
            return Float(t)
        }
    }

    /// Runs the full turbo sampler from `noise` `[B, T, 64]` and returns the clean latents `[B, T, 64]` (float32).
    ///
    /// - Parameters:
    ///   - context: `[B, T, 128]` context latents from the condition encoder.
    ///   - encoder: `[B, S, 2048]` condition-encoder hidden states.
    ///   - timesteps: explicit schedule (overrides `steps`/`shift`; no trailing 0).
    ///   - onStep: called after each step with (completed steps, total); throw to cancel.
    ///   - trace: development hook, receives (step index, DiT input x, velocity) for each step.
    public static func sample(dit: ACEDiT, noise: MLXArray, context: MLXArray, encoder: MLXArray,
                              steps: Int = 8, shift: Float = 1, timesteps: [Float]? = nil,
                              dcw: DCWOptions = .turboDefault,
                              onStep: ((Int, Int) throws -> Void)? = nil,
                              trace: ((Int, MLXArray, MLXArray) -> Void)? = nil) throws -> MLXArray {
        let ts = timesteps ?? schedule(steps: steps, shift: shift)
        let cache = dit.crossCache(encoder: encoder)
        var x = noise.asType(.float32)
        for (i, t) in ts.enumerated() {
            let v = dit(x, t: t, context: context, encoder: encoder, cache: cache)
            trace?(i, x, v)
            let denoised = x - v * t
            if i == ts.count - 1 {
                x = denoised
            } else {
                x = dcw.apply(x - v * (t - ts[i + 1]), denoised: denoised, t: t)
            }
            eval(x)
            try onStep?(i + 1, ts.count)
        }
        return x
    }
}
