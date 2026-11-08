import Foundation
import MLX
import MLXDiffusionCore

/// The Oobleck audio VAE decoder (stable-audio-tools `OobleckDecoder`): latents `[B, T, 64]` → stereo waveform
/// `[B, T * 2048, 2]`. Snake activations, weight-normalized convolutions (fused at load), transposed-conv upsampling.
/// Shared by Stable Audio Open and ACE-Step.
public final class OobleckDecoder {
    struct Conv {
        let weight: MLXArray  // MLX layout [out, k, in]
        let bias: MLXArray?
        let stride: Int, padding: Int, dilation: Int
        let transposed: Bool

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = transposed
                ? convTransposed1d(x, weight, stride: stride, padding: padding)
                : conv1d(x, weight, stride: stride, padding: padding, dilation: dilation)
            if let bias { y = y + bias }
            return y
        }
    }

    struct Snake {
        let alpha: MLXArray, invBeta: MLXArray
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let s = sin(alpha * x)
            return x + invBeta * s * s
        }
    }

    struct ResidualUnit {
        let snake1: Snake, conv1: Conv, snake2: Snake, conv2: Conv
        func callAsFunction(_ x: MLXArray) -> MLXArray { x + conv2(snake2(conv1(snake1(x)))) }
    }

    struct Block {
        let snake: Snake, up: Conv, res: [ResidualUnit]
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = up(snake(x))
            for r in res { h = r(h) }
            return h
        }
    }

    let inConv: Conv
    let blocks: [Block]
    let outSnake: Snake
    let outConv: Conv
    let finalTanh: Bool
    public let dtype: DType
    /// Samples per latent frame (the product of the strides).
    public let hopLength: Int

    /// `prefix`: key prefix of the decoder in `store` (e.g. `pretransform.model.decoder.`).
    public init(store s: WeightStore, prefix p: String, strides: [Int] = [2, 4, 4, 8, 8], finalTanh: Bool = false,
                dtype: DType = .float32) throws {
        self.finalTanh = finalTanh
        self.dtype = dtype
        hopLength = strides.reduce(1, *)
        func weight(_ k: String, transposed: Bool) throws -> MLXArray {
            var w: MLXArray
            if let g = s.optional(k + ".weight_g"), let v = s.optional(k + ".weight_v") {
                // weight_norm(dim=0): w = g · v / ‖v‖ over every axis but the first.
                let v32 = v.asType(.float32)
                let n = sqrt((v32 * v32).sum(axes: [1, 2], keepDims: true))
                w = g.asType(.float32) * v32 / n
            } else {
                w = try s.get(k + ".weight").asType(.float32)
            }
            // PyTorch Conv1d [out, in, k] / ConvTranspose1d [in, out, k] → MLX [out, k, in].
            return (transposed ? w.transposed(1, 2, 0) : w.transposed(0, 2, 1)).asType(dtype)
        }
        func conv(_ k: String, stride: Int = 1, padding: Int, dilation: Int = 1, transposed: Bool = false) throws -> Conv {
            Conv(weight: try weight(k, transposed: transposed), bias: s.optional(k + ".bias")?.asType(dtype),
                 stride: stride, padding: padding, dilation: dilation, transposed: transposed)
        }
        func snake(_ k: String) throws -> Snake {
            // alpha/beta are stored as logs (alpha_logscale).
            let a = exp(try s.get(k + ".alpha").asType(.float32))
            let b = exp(try s.get(k + ".beta").asType(.float32))
            return Snake(alpha: a.asType(dtype), invBeta: (1 / (b + 1e-9)).asType(dtype))
        }
        func residual(_ k: String, dilation: Int) throws -> ResidualUnit {
            ResidualUnit(snake1: try snake(k + ".layers.0"),
                         conv1: try conv(k + ".layers.1", padding: 3 * dilation, dilation: dilation),
                         snake2: try snake(k + ".layers.2"), conv2: try conv(k + ".layers.3", padding: 0))
        }
        inConv = try conv(p + "layers.0", padding: 3)
        let rev = Array(strides.reversed())
        blocks = try rev.enumerated().map { i, stride in
            let k = p + "layers.\(i + 1).layers."
            return Block(snake: try snake(k + "0"),
                         up: try conv(k + "1", stride: stride, padding: (stride + 1) / 2, transposed: true),
                         res: try [1, 3, 9].enumerated().map { j, d in try residual(k + "\(j + 2)", dilation: d) })
        }
        outSnake = try snake(p + "layers.\(strides.count + 1)")
        outConv = try conv(p + "layers.\(strides.count + 2)", padding: 3)
    }

    /// `latents`: `[B, T, 64]` → `[B, T * hopLength, channels]` float32.
    public func callAsFunction(_ latents: MLXArray) -> MLXArray {
        var h = inConv(latents.asType(dtype))
        for b in blocks {
            h = b(h)
            eval(h)
        }
        h = outConv(outSnake(h))
        if finalTanh { h = tanh(h) }
        return h.asType(.float32)
    }
}
