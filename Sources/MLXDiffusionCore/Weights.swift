import Foundation
import MLX

public enum WeightError: Error, CustomStringConvertible {
    case missing(String)
    case badShape(String, [Int])

    public var description: String {
        switch self {
        case .missing(let k): return "missing weight '\(k)'"
        case .badShape(let k, let s): return "unexpected shape \(s) for weight '\(k)'"
        }
    }
}

/// A flat name → tensor map loaded from one or more safetensors files.
///
/// Models pull their parameters out of it by key (`store["blocks.0.attn.to_q.weight"]`) instead of relying on
/// reflection, which keeps diffusers/transformers key names usable verbatim.
public final class WeightStore {
    public private(set) var arrays: [String: MLXArray]
    public let metadata: [String: String]

    public init(arrays: [String: MLXArray], metadata: [String: String] = [:]) {
        self.arrays = arrays
        self.metadata = metadata
    }

    public convenience init(url: URL) throws {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        self.init(arrays: arrays, metadata: metadata)
    }

    /// Loads every `*.safetensors` file in a directory (sharded checkpoints).
    public convenience init(directory: URL) throws {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        var all: [String: MLXArray] = [:]
        var meta: [String: String] = [:]
        for f in files {
            let (a, m) = try loadArraysAndMetadata(url: f)
            all.merge(a) { _, new in new }
            meta.merge(m) { _, new in new }
        }
        self.init(arrays: all, metadata: meta)
    }

    public var keys: [String] { Array(arrays.keys) }

    public func has(_ key: String) -> Bool { arrays[key] != nil }

    public func get(_ key: String) throws -> MLXArray {
        guard let a = arrays[key] else { throw WeightError.missing(key) }
        return a
    }

    public func optional(_ key: String) -> MLXArray? { arrays[key] }

    /// Bits used by MLX-quantized checkpoints (`quantization_level` metadata), defaulting to 8.
    public var quantizationBits: Int {
        if let s = metadata["quantization_level"], let b = Int(s) { return b }
        return 8
    }

    /// Builds a linear layer from `<prefix>.weight` (+ `.scales`/`.biases` when quantized, + `.bias`).
    public func linear(_ prefix: String, dtype: DType? = nil) throws -> Lin {
        let w = try get(prefix + ".weight")
        let bias = optional(prefix + ".bias")
        if let scales = optional(prefix + ".scales") {
            let bits = quantizationBits
            let inDim = w.dim(1) * 32 / bits
            let groupSize = inDim / scales.dim(1)
            var l = Lin(weight: w, bias: bias, scales: scales, qbiases: optional(prefix + ".biases"),
                        bits: bits, groupSize: groupSize)
            l.name = prefix
            return l
        }
        let wd = dtype.map { w.asType($0) } ?? w
        var l = Lin(weight: wd, bias: bias.map { b in dtype.map { b.asType($0) } ?? b })
        l.name = prefix
        return l
    }

    /// Same as ``linear(_:dtype:)`` but expands quantized weights to `dtype` (trades memory for matmul speed).
    public func denseLinear(_ prefix: String, dtype: DType = .bfloat16) throws -> Lin {
        let l = try linear(prefix)
        guard let scales = l.scales else { return l }
        let w = dequantized(l.weight, scales: scales, biases: l.qbiases, groupSize: l.groupSize, bits: l.bits)
            .asType(dtype)
        eval(w)
        var d = Lin(weight: w, bias: l.bias)
        d.name = prefix
        return d
    }

    /// Drops tensors (e.g. after a model has taken what it needs) so their memory can be reclaimed.
    public func remove(prefix: String) {
        for k in arrays.keys where k.hasPrefix(prefix) { arrays[k] = nil }
    }
}

/// Dense or affine-quantized linear layer (`y = x Wᵀ + b`).
public struct Lin {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let scales: MLXArray?
    public let qbiases: MLXArray?
    public let bits: Int
    public let groupSize: Int
    /// Checkpoint key prefix (e.g. `transformer_blocks.0.attn.to_q`), used to match LoRA weights.
    public var name: String = ""
    /// Low-rank side branches added to the output (LoRA), applied without touching the base weight.
    public var adapters: [LoRAAdapter] = []

    public init(weight: MLXArray, bias: MLXArray? = nil, scales: MLXArray? = nil, qbiases: MLXArray? = nil,
                bits: Int = 8, groupSize: Int = 64) {
        self.weight = weight
        self.bias = bias
        self.scales = scales
        self.qbiases = qbiases
        self.bits = bits
        self.groupSize = groupSize
    }

    public var isQuantized: Bool { scales != nil }

    public var outFeatures: Int { weight.dim(0) }
    public var inFeatures: Int { isQuantized ? weight.dim(1) * 32 / bits : weight.dim(1) }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y: MLXArray
        if let scales {
            y = quantizedMM(x, weight, scales: scales, biases: qbiases, transpose: true,
                            groupSize: groupSize, bits: bits)
        } else {
            y = matmul(x, weight.T)
        }
        if let bias { y = y + bias }
        for a in adapters {
            y = y + a.scale * matmul(matmul(x, a.down.T), a.up.T)
        }
        return y
    }

    /// A copy with `adapter` appended.
    public func adding(_ adapter: LoRAAdapter) -> Lin {
        var l = self
        l.adapters.append(adapter)
        return l
    }
}

/// 2D convolution stored in MLX layout (`[O, kh, kw, I]`), loaded from PyTorch layout (`[O, I, kh, kw]`).
public struct Conv2d {
    public let weight: MLXArray
    public let bias: MLXArray?
    public let stride: Int
    public let padding: Int

    public init(store: WeightStore, prefix: String, stride: Int = 1, padding: Int = 0, dtype: DType? = nil) throws {
        var w = try store.get(prefix + ".weight")
        if w.ndim == 5 { w = w.squeezed(axis: 2) }  // [O, I, 1, kh, kw] style 3D kernels with T=1
        guard w.ndim == 4 else { throw WeightError.badShape(prefix + ".weight", w.shape) }
        w = w.transposed(0, 2, 3, 1)
        var b = store.optional(prefix + ".bias")
        if let dtype {
            w = w.asType(dtype)
            b = b?.asType(dtype)
        }
        self.weight = w
        self.bias = b
        self.stride = stride
        self.padding = padding
    }

    public var outChannels: Int { weight.dim(0) }
    public var inChannels: Int { weight.dim(3) }

    /// `x`: `[B, H, W, C]`.
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv2d(x, weight, stride: .init(stride), padding: .init(padding))
        if let bias { y = y + bias }
        return y
    }
}

// MARK: - activations

public func siluAct(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

public func geluTanh(_ x: MLXArray) -> MLXArray {
    let c: Float = 0.7978845608028654  // sqrt(2/pi)
    return 0.5 * x * (1 + tanh(c * (x + 0.044715 * x * x * x)))
}
