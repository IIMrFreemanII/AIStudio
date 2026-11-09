import Foundation
import MLX

/// Sigma schedules for Wan's flow matching (sigma 1 = noise, 0 = clean; the model's timestep is `1000·sigma`).
public enum WanSchedule {
    /// Shifted sigma: `shift·s / (1 + (shift−1)·s)`.
    static func shifted(_ s: Double, _ shift: Double) -> Double { shift * s / (1 + (shift - 1) * s) }

    /// The official UniPC/DPM schedule: `steps` values linearly spaced from the training schedule's largest sigma
    /// (999/1000) down to 0, shifted, then a final 0. Timesteps are the integer parts of `1000·sigma`.
    public static func official(steps: Int, shift: Double, trainSteps: Int = 1000) -> (sigmas: [Double], timesteps: [Double]) {
        let maxS = Double(trainSteps - 1) / Double(trainSteps)
        var sigmas = (0..<steps).map { i -> Double in
            let s = maxS - (maxS - 0) * Double(i) / Double(steps)
            return Double(Float(shifted(s, shift)))
        }
        sigmas.append(0)
        let timesteps = sigmas.dropLast().map { Double(Int(Float($0) * Float(trainSteps))) }
        return (sigmas, timesteps)
    }

    /// The Lightning (lightx2v) 4-step distillation schedule: Euler with shift 5 from sigma 1, i.e. timesteps
    /// 1000, 937.5, 833.3, 625 for 4 steps (fewer steps keep the same spacing).
    public static func lightning(steps: Int = 4, shift: Double = 5) -> (sigmas: [Double], timesteps: [Double]) {
        var sigmas = (0..<steps).map { shifted(1 - Double($0) / Double(steps), shift) }
        sigmas.append(0)
        return (sigmas, sigmas.dropLast().map { $0 * 1000 })
    }
}

/// UniPC multistep (bh2, order 2, predict-x0, lower order at the end) for flow matching, as the official
/// `FlowUniPCMultistepScheduler` with Wan's settings.
public struct WanUniPC {
    let sigmas: [Double]
    let order = 2
    var stepIndex = 0
    var lowerOrderNums = 0
    var outputs: [MLXArray?] = [nil, nil]  // x0 history, newest last
    var lastSample: MLXArray?
    var thisOrder = 1

    public init(sigmas: [Double]) { self.sigmas = sigmas }

    static func lambda(_ s: Double) -> Double {
        if s >= 1 { return -.infinity }
        if s <= 0 { return .infinity }
        return log((1 - s) / s)
    }

    /// One step from `sample` with the model's velocity `v` at the current sigma.
    public mutating func step(velocity v: MLXArray, sample: MLXArray) -> MLXArray {
        let i = stepIndex
        let x0 = sample - Float(sigmas[i]) * v
        var sample = sample
        if i > 0, let last = lastSample { sample = corrector(x0, last: last, order: thisOrder) }
        outputs = [outputs[1], x0]
        let n = sigmas.count - 1
        thisOrder = min(min(order, n - i), lowerOrderNums + 1)
        lastSample = sample
        let next = predictor(x0, sample: sample, order: thisOrder)
        if lowerOrderNums < order { lowerOrderNums += 1 }
        stepIndex += 1
        return next
    }

    func predictor(_ x0: MLXArray, sample: MLXArray, order: Int) -> MLXArray {
        let i = stepIndex
        let s0 = sigmas[i], st = sigmas[i + 1]
        if st == 0 { return x0 }
        let h = Self.lambda(st) - Self.lambda(s0), hh = -h
        let alphaT = 1 - st, phi1 = expm1(hh)
        let m0 = outputs[1]!
        var x = Float(st / s0) * sample - Float(alphaT * phi1) * m0
        if order >= 2, i - 1 >= 0, let m1 = outputs[0] {
            let rk = (Self.lambda(sigmas[i - 1]) - Self.lambda(s0)) / h
            if rk.isFinite {
                let d1 = (m1 - m0) / Float(rk)
                x = x - Float(alphaT * phi1 * 0.5) * d1
            }
        }
        return x
    }

    func corrector(_ x0: MLXArray, last: MLXArray, order: Int) -> MLXArray {
        let i = stepIndex
        let s0 = sigmas[i - 1], st = sigmas[i]
        if st == 0 { return last }  // unreachable in practice: the final step's sample is never corrected
        let h = Self.lambda(st) - Self.lambda(s0), hh = -h
        let alphaT = 1 - st, phi1 = expm1(hh), bh = phi1
        let m0 = outputs[1]!
        let base = Float(st / s0) * last - Float(alphaT * phi1) * m0
        let d1t = x0 - m0
        var rks: [Double] = [], d1s: [MLXArray] = []
        if order >= 2, i - 2 >= 0, let m1 = outputs[0] {
            let rk = (Self.lambda(sigmas[i - 2]) - Self.lambda(s0)) / h
            if rk.isFinite {
                rks.append(rk)
                d1s.append((m1 - m0) / Float(rk))
            }
        }
        rks.append(1)
        // rhos_c solves R·rho = b with R[j] = rks^j and b[j] = h_phi_k·j! / B(h).
        var rhos: [Double]
        if rks.count == 1 {
            rhos = [0.5]
        } else {
            var hPhiK = phi1 / hh - 1, fact = 1.0
            var b: [Double] = []
            for j in 1...rks.count {
                b.append(hPhiK * fact / bh)
                fact *= Double(j + 1)
                hPhiK = hPhiK / hh - 1 / fact
            }
            // 2×2 solve: [1 1; r 1]·rho = b
            let r = rks[0], det = 1 * 1 - 1 * r
            rhos = [(b[0] * 1 - 1 * b[1]) / det, (1 * b[1] - r * b[0]) / det]
        }
        var corr = MLXArray.zeros(d1t.shape, dtype: d1t.dtype)
        for (k, d) in d1s.enumerated() { corr = corr + Float(rhos[k]) * d }
        return base - Float(alphaT * bh) * (corr + Float(rhos.last!) * d1t)
    }
}
