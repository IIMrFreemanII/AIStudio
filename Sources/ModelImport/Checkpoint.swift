import Foundation
import MLX

public enum ModelImportError: Error, CustomStringConvertible, LocalizedError {
    case notSafetensors(String)
    case gguf(String)
    case bitsAndBytes(String)
    case alreadyQuantized(String)
    case unsupportedDtype(String, tensor: String)
    case unknownSchema(sample: [String])
    case wrongFamily(expected: String, found: String)
    case unsupportedVariant(String)
    case unexpectedTensors(sample: [String])
    case architectureMismatch(family: String, base: String, details: [String], hint: String?)
    case badBase(String)
    case destinationExists(String)
    case wanExpertUnknown(String)
    case notEnoughSpace(needed: Int64, available: Int64)
    case noTensors(String)

    public var description: String {
        switch self {
        case .notSafetensors(let f):
            "\(f) is not a .safetensors checkpoint (pickled .ckpt/.pt files are not supported)"
        case .gguf(let f):
            "\(f) is a GGUF file; only .safetensors checkpoints (bf16, fp16, fp32 or fp8) can be imported"
        case .bitsAndBytes(let f):
            "\(f) is NF4/FP4 (bitsandbytes) quantized; use the bf16, fp16 or fp8 version of this checkpoint"
        case .alreadyQuantized(let f):
            "\(f) is already quantized (MLX/GPTQ-style scales); use the bf16, fp16 or fp8 version of this checkpoint"
        case .unsupportedDtype(let d, let t):
            "tensor '\(t)' is stored as \(d); only BF16, F16, F32 and F8_E4M3 weights can be imported"
        case .unknownSchema(let s):
            "no diffusion transformer recognised in this file (keys look like \(s.joined(separator: ", ")))"
        case .wrongFamily(let e, let f):
            "this checkpoint is a \(f) model, not \(e)"
        case .unsupportedVariant(let m):
            m
        case .unexpectedTensors(let s):
            "unrecognised transformer tensors: \(s.joined(separator: ", "))"
        case .architectureMismatch(let family, let base, let details, let hint):
            "this checkpoint does not match the installed \(family) model '\(base)': "
                + details.joined(separator: "; ") + (hint.map { " (\($0))" } ?? "")
        case .badBase(let m):
            m
        case .destinationExists(let p):
            "\(p) already exists"
        case .wanExpertUnknown(let f):
            "\(f): say which Wan 2.2 expert this file is (high noise or low noise)"
        case .notEnoughSpace(let n, let a):
            "not enough disk space: the import needs \(ByteCountFormatter.string(fromByteCount: n, countStyle: .file)), "
                + "\(ByteCountFormatter.string(fromByteCount: a, countStyle: .file)) is free"
        case .noTensors(let f):
            "\(f) contains no tensors"
        }
    }

    public var errorDescription: String? { description }
}

/// How the transformer's tensors are named in a source checkpoint.
public enum KeySchema: String, Sendable, Codable {
    /// Black Forest Labs' original FLUX names (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`, `img_in`…).
    case bfl
    /// diffusers names (`transformer_blocks.N.attn.to_q`…; for Wan `blocks.N.attn1.to_q`, `condition_embedder.…`).
    case diffusers
    /// The official Wan repository's names (`blocks.N.self_attn.q`, `patch_embedding`, `text_embedding.0`…).
    case wan
}

/// Which model a checkpoint's transformer belongs to (as far as its key names tell).
public enum Architecture: String, Sendable, Codable {
    case flux1 = "flux.1"
    case qwenImage21 = "qwen-image-2.1"
    case wan22 = "wan-2.2"
}

/// The two Wan 2.2 A14B experts.
public enum WanExpert: String, Sendable, Codable, CaseIterable {
    case high, low

    public var fileName: String { "\(rawValue)_noise_model.safetensors" }

    /// Guesses the expert from a file or folder name (`…HighNoise…`, `high_noise_model`, `…_low_…`).
    public static func guess(_ name: String) -> WanExpert? {
        let n = name.lowercased().replacingOccurrences(of: "-", with: "_").replacingOccurrences(of: " ", with: "_")
        let high = n.contains("high") || n.contains("_hn") || n.hasPrefix("hn_")
        let low = n.contains("low") || n.contains("_ln") || n.hasPrefix("ln_")
        return high == low ? nil : high ? .high : .low
    }
}

/// Top-level names of the non-transformer parts of full checkpoints (VAE, text encoders), dropped on import.
let foreignRoots = ["vae", "first_stage_model", "text_encoders", "text_encoder", "text_encoder_2", "cond_stage_model",
                    "conditioner", "clip_l", "clip_g", "t5xxl", "t5", "umt5", "model_ema", "lora_te", "embedding_manager"]

/// The prefixes a transformer can sit under in a single-file checkpoint (ComfyUI's first).
let transformerPrefixes = ["model.diffusion_model.", "diffusion_model.", "transformer.", ""]

/// What a checkpoint's names say about it.
struct Layout {
    var architecture: Architecture
    var schema: KeySchema
    var prefix: String
    /// Full keys of the transformer's tensors (prefix included), and of everything else in the file.
    var transformerKeys: [String]
    var droppedKeys: [String]

    static func isBlock(_ k: String, _ field: String) -> Bool {
        // "blocks.<N>.<field>" with a numeric N
        guard k.hasPrefix("blocks.") else { return false }
        let rest = k.dropFirst("blocks.".count)
        guard let dot = rest.firstIndex(of: "."), Int(rest[..<dot]) != nil else { return false }
        return rest[rest.index(after: dot)...].hasPrefix(field)
    }

    static func classify(_ keys: [String]) throws -> (Architecture, KeySchema)? {
        if keys.contains(where: { $0.hasPrefix("double_blocks.") || $0.hasPrefix("single_blocks.") }) {
            return (.flux1, .bfl)
        }
        if keys.contains(where: { $0.hasPrefix("single_transformer_blocks.") }) { return (.flux1, .diffusers) }
        if keys.contains(where: { $0.hasPrefix("transformer_blocks.") }) {
            if keys.contains(where: { $0.contains(".img_mlp.gate_layer.") || $0.hasPrefix("modulation.") }) {
                return (.qwenImage21, .diffusers)
            }
            if keys.contains(where: { $0.contains(".txt_mlp.") || $0.contains(".img_mod.") }) {
                throw ModelImportError.unsupportedVariant(
                    "this is a Qwen-Image (v1 / 2512 / Edit) checkpoint; only Qwen-Image 2.1 fine-tunes can be imported")
            }
            if keys.contains(where: { $0.hasPrefix("x_embedder.") }) { return (.flux1, .diffusers) }
            return nil
        }
        if keys.contains(where: { isBlock($0, "self_attn.") }) { return (.wan22, .wan) }
        if keys.contains(where: { isBlock($0, "attn1.") }) { return (.wan22, .diffusers) }
        return nil
    }

    init(keys: [String], file: String) throws {
        guard !keys.isEmpty else { throw ModelImportError.noTensors(file) }
        for p in transformerPrefixes {
            let under = p.isEmpty
                ? keys.filter { k in !foreignRoots.contains(where: { k.hasPrefix($0 + ".") || k.hasPrefix($0 + "_") })
                    && !transformerPrefixes.dropLast().contains(where: { k.hasPrefix($0) }) }
                : keys.filter { $0.hasPrefix(p) }
            guard !under.isEmpty, let (arch, schema) = try Self.classify(under.map { String($0.dropFirst(p.count)) })
            else { continue }
            architecture = arch
            self.schema = schema
            prefix = p
            transformerKeys = under
            let set = Set(under)
            droppedKeys = keys.filter { !set.contains($0) }
            return
        }
        throw ModelImportError.unknownSchema(sample: Array(keys.sorted().prefix(4)))
    }

    /// Top-level names of the dropped parts (`vae`, `text_encoders`…), for reporting.
    var droppedComponents: [String] {
        Array(Set(droppedKeys.map { String($0.split(separator: ".").first ?? "") })).sorted()
    }
}

/// A source checkpoint (one `.safetensors` file, or a folder of shards) with its transformer located.
///
/// Tensors are loaded lazily: ``tensor(_:)`` returns an unevaluated bf16 array, read from disk only when the output
/// it feeds is written.
final class SourceCheckpoint {
    let files: [URL]
    let headers: [SafetensorsHeader]
    let tags: [String: String]
    let layout: Layout
    private var arrays: [String: MLXArray] = [:]
    private var loaded = false

    /// - Parameter urls: one file or the shards of one checkpoint.
    init(files urls: [URL]) throws {
        files = urls
        for u in urls where u.pathExtension.lowercased() == "gguf" { throw ModelImportError.gguf(u.lastPathComponent) }
        headers = try urls.map { try SafetensorsHeader(url: $0) }
        var tags: [String: String] = [:]
        for h in headers { for (k, e) in h.tensors { tags[k] = e.dtype } }
        self.tags = tags
        let name = urls.first?.lastPathComponent ?? "checkpoint"
        layout = try Layout(keys: Array(tags.keys), file: name)

        // Formats we refuse.
        let tk = layout.transformerKeys
        if tk.contains(where: { $0.hasSuffix(".absmax") || $0.contains("quant_map") || $0.contains("quant_state") }) {
            throw ModelImportError.bitsAndBytes(name)
        }
        if tk.contains(where: { $0.hasSuffix(".scales") || $0.hasSuffix(".qweight") || $0.hasSuffix(".qzeros") }) {
            throw ModelImportError.alreadyQuantized(name)
        }
        for k in tk where !Self.isAuxiliary(k) {
            let t = tags[k]!
            guard ["BF16", "F16", "F32", "F64", "F8_E4M3"].contains(t) else {
                if t == "U8" || t == "I8" { throw ModelImportError.bitsAndBytes(name) }
                throw ModelImportError.unsupportedDtype(t, tensor: k)
            }
        }
    }

    /// ComfyUI's scaled-fp8 bookkeeping: the `scaled_fp8` marker and per-layer `scale_weight` / `scale_input`.
    static func isAuxiliary(_ key: String) -> Bool {
        key.hasSuffix(".scale_weight") || key.hasSuffix(".scale_input") || key == "scaled_fp8"
            || key.hasSuffix(".scaled_fp8")
    }

    /// The transformer's tensor names with the prefix removed and the scaled-fp8 bookkeeping left out.
    var keys: [String] {
        layout.transformerKeys.filter { !Self.isAuxiliary($0) }.map { String($0.dropFirst(layout.prefix.count)) }
            .sorted(by: naturalLess)
    }

    /// The dtype tag the weights are mostly stored in (`BF16`, `F8_E4M3`…), for reporting.
    var weightDtype: String {
        var counts: [String: Int] = [:]
        for k in layout.transformerKeys where k.hasSuffix(".weight") { counts[tags[k]!, default: 0] += 1 }
        return counts.max { $0.value < $1.value }?.key ?? "BF16"
    }

    var transformerBytes: Int {
        let set = Set(layout.transformerKeys)
        return headers.reduce(0) { sum, h in sum + h.tensors.filter { set.contains($0.key) }.reduce(0) { $0 + $1.value.byteCount } }
    }

    func shape(_ key: String) -> [Int]? {
        let full = layout.prefix + key
        for h in headers { if let e = h.tensors[full] { return e.shape } }
        return nil
    }

    private func load() throws {
        guard !loaded else { return }
        for f in files { arrays.merge(try loadArrays(url: f)) { _, new in new } }
        loaded = true
    }

    /// The tensor `key` (prefix removed) as a lazy array: FP8 decoded (times its `scale_weight` when present) and
    /// floating-point tensors cast to bf16.
    func tensor(_ key: String) throws -> MLXArray {
        try load()
        let full = layout.prefix + key
        guard var a = arrays[full], let tag = tags[full] else { throw ModelImportError.unexpectedTensors(sample: [key]) }
        let scale = key.hasSuffix(".weight")
            ? arrays[layout.prefix + key.dropLast(".weight".count) + ".scale_weight"] : nil
        if tag == "F8_E4M3" {
            a = fromFP8(a, dtype: scale == nil ? .bfloat16 : .float32)
        }
        if let scale { a = a.asType(.float32) * scale.asType(.float32) }
        if a.dtype.isFloatingPoint, a.dtype != .bfloat16 { a = a.asType(.bfloat16) }
        return a
    }
}

/// Orders `blocks.2.x` before `blocks.10.x` (numeric path components compared as numbers).
func naturalLess(_ a: String, _ b: String) -> Bool {
    let pa = a.split(separator: "."), pb = b.split(separator: ".")
    for (x, y) in zip(pa, pb) where x != y {
        if let i = Int(x), let j = Int(y) { return i < j }
        return x < y
    }
    return pa.count < pb.count
}

extension DType {
    var isFloatingPoint: Bool { [.float16, .bfloat16, .float32, .float64].contains(self) }
}
