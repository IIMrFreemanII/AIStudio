import Foundation
import MLX

/// Port of diffusers' `FlowMatchEulerDiscreteScheduler` for the configurations we use (no dynamic shifting,
/// no stochastic sampling, no Karras/beta/exponential sigma spacing).
public struct FlowMatchEuler: Sendable {
    public var numTrainTimesteps: Int = 1000
    public var shift: Float = 1.0

    /// Sampling sigmas including the terminal 0.
    public private(set) var sigmas: [Float] = []

    public init(numTrainTimesteps: Int = 1000, shift: Float = 1.0) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
    }

    /// Reads `scheduler_config.json`.
    public init(configURL: URL) throws {
        let data = try Data(contentsOf: configURL)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        self.numTrainTimesteps = obj["num_train_timesteps"] as? Int ?? 1000
        self.shift = Float(obj["shift"] as? Double ?? 1.0)
        if obj["use_dynamic_shifting"] as? Bool == true {
            FileHandle.standardError.write("warning: dynamic shifting is not implemented; ignoring\n".data(using: .utf8)!)
        }
    }

    /// Sets the schedule from explicit sigmas (excluding the terminal sigma, which is appended).
    public mutating func setSigmas(_ input: [Float]) {
        let shifted = input.map { s in shift * s / (1 + (shift - 1) * s) }
        sigmas = shifted + [0]
    }

    /// Default linear schedule `linspace(1, 1/n, n)` as the Qwen pipeline uses when no grid is configured.
    public mutating func setSteps(_ n: Int) {
        precondition(n > 0)
        let grid = (0..<n).map { i -> Float in
            n == 1 ? 1 : 1 + (Float(i) / Float(n - 1)) * (1 / Float(n) - 1)
        }
        setSigmas(grid)
    }

    public var stepCount: Int { max(sigmas.count - 1, 0) }

    /// Model-facing timestep for step `i` (`sigma * num_train_timesteps`).
    public func timestep(_ i: Int) -> Float { sigmas[i] * Float(numTrainTimesteps) }

    /// Euler step: `x_{i+1} = x_i + (σ_{i+1} − σ_i) · v`. Computed in float32.
    public func step(_ velocity: MLXArray, sample: MLXArray, index i: Int) -> MLXArray {
        let dt = sigmas[i + 1] - sigmas[i]
        return sample.asType(.float32) + dt * velocity.asType(.float32)
    }

    /// Prediction of the clean sample from the current one (used for previews).
    public func predictX0(_ velocity: MLXArray, sample: MLXArray, index i: Int) -> MLXArray {
        sample.asType(.float32) - sigmas[i] * velocity.asType(.float32)
    }
}
