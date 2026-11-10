import Foundation
import MLX
@testable import ModelImport

/// A temporary folder removed when the value goes away.
final class TempDir {
    let url: URL

    init(_ name: String = "import") {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    deinit { try? FileManager.default.removeItem(at: url) }

    func sub(_ path: String) -> URL {
        let u = url.appendingPathComponent(path)
        try! FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
        return u
    }
}

/// Deterministic random tensors.
struct RNG {
    var key: MLXArray

    init(_ seed: UInt64) { key = MLXRandom.key(seed) }

    mutating func normal(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let k = MLXRandom.split(key: key, into: 2)
        key = k[0]
        return MLXRandom.normal(shape, key: k[1]) * scale
    }

    /// A linear layer's weight `[out, in]` (scaled to keep activations O(1)) and bias.
    mutating func linear(_ name: String, _ out: Int, _ inp: Int, bias: Bool = true, into d: inout [String: MLXArray]) {
        d[name + ".weight"] = normal([out, inp], scale: 1 / Float(inp).squareRoot()).asType(.bfloat16)
        if bias { d[name + ".bias"] = normal([out], scale: 0.1).asType(.bfloat16) }
    }

    /// A norm weight around 1.
    mutating func norm(_ name: String, _ n: Int, into d: inout [String: MLXArray]) {
        d[name] = (1 + normal([n], scale: 0.2)).asType(.bfloat16)
    }
}

/// Writes `arrays` as a safetensors file; names in `fp8` are stored as `F8_E4M3` bytes, `tags` overrides others.
func writeSafetensors(_ arrays: [String: MLXArray], to url: URL, fp8: Set<String> = [], tags: [String: String] = [:],
                      metadata: [String: String] = [:]) throws {
    var plan = arrays.keys.sorted().map { k in
        fp8.contains(k) ? PlannedTensor(name: k, array: toFP8(arrays[k]!), dtypeTag: "F8_E4M3")
            : PlannedTensor(name: k, array: arrays[k]!, dtypeTag: tags[k])
    }
    try SafetensorsWriter.write(&plan, metadata: metadata, to: url)
}

/// The bf16 value an FP8 checkpoint stores for `a` (optionally with a ComfyUI `scale_weight`).
func fp8RoundTrip(_ a: MLXArray, scale: Float? = nil) -> MLXArray {
    guard let scale else { return fromFP8(toFP8(a), dtype: .bfloat16) }
    return (fromFP8(toFP8(a / scale), dtype: .float32) * MLXArray(scale)).asType(.bfloat16)
}

/// MLX affine quantization of the 2-D weights `should` picks, as the installed MLX folders store them.
func quantizeWeights(_ d: [String: MLXArray], bits: Int, groupSize: Int = 64,
                     should: (String, MLXArray) -> Bool) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (k, a) in d {
        guard should(k, a) else { out[k] = a; continue }
        let (w, s, b) = quantized(a, groupSize: groupSize, bits: bits)
        let base = String(k.dropLast(".weight".count))
        out[k] = w; out[base + ".scales"] = s; out[base + ".biases"] = b!
    }
    return out
}

func writeJSON(_ o: Any, _ url: URL) throws {
    try JSONSerialization.data(withJSONObject: o, options: [.prettyPrinted, .sortedKeys]).write(to: url)
}

/// A placeholder component folder (text encoder, VAE…) with one file in it.
func placeholder(_ dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data("x".utf8).write(to: dir.appendingPathComponent("weights.bin"))
}

func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// The error a throwing call raises, as its user-facing text.
func errorText(_ body: () throws -> Any) -> String? {
    do { _ = try body(); return nil } catch { return "\(error)" }
}

/// A folder's entries as the loaders list them (`contentsOfDirectory`, which does not follow a link to a folder).
func listing(_ dir: URL) -> [String]? {
    (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?.map(\.lastPathComponent).sorted()
}

/// Where a symbolic link points (nil when `url` is not one).
func linkTarget(_ url: URL) -> String? { try? FileManager.default.destinationOfSymbolicLink(atPath: url.path) }

/// Every hidden `.…partial-…` folder in `dir` (left behind by a failed import, which must not happen).
func partials(in dir: URL) -> [String] {
    ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.contains(".partial-") }
}
