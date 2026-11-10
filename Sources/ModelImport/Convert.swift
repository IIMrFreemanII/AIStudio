import FluxTransformer
import Foundation
import MLX

/// Name mapping of each family: source tensors → (name, lazy array) in the names the app's loader reads, still
/// dense. ``quantize(_:bits:groupSize:should:)`` then turns the chosen weights into MLX affine-quantized triples.
enum Convert {
    typealias Named = (name: String, array: MLXArray)

    // MARK: FLUX.1

    /// FLUX: diffusers names as is; BFL names through ``FluxTransformer/bflTarget(_:)`` (fused `qkv` and `linear1`
    /// split by rows, the final modulation's halves swapped, the QK norms' `scale` → `weight`).
    static func flux(_ src: SourceCheckpoint) throws -> [Named] {
        guard src.layout.schema == .bfl else { return try src.keys.map { ($0, try src.tensor($0)) } }
        var out: [Named] = []
        var unknown: [String] = []
        for key in src.keys {
            let parts = key.split(separator: ".")
            let module = parts.dropLast().joined(separator: "."), param = String(parts.last ?? "")
            guard let target = FluxTransformer.bflTarget(module) else { unknown.append(key); continue }
            let a = try src.tensor(key)
            switch param {
            case "scale" where target.names.count == 1:
                out.append((target.names[0] + ".weight", a))
            case "weight", "bias":
                if target.names.count == 1 {
                    var t = a
                    if target.swapHalves {
                        let h = a.dim(0) / 2
                        t = concatenated([a[h...], a[..<h]], axis: 0)
                    }
                    out.append((target.names[0] + "." + param, t))
                    continue
                }
                // Fused projection: q, k, v (and in single blocks the MLP input after them), split by rows.
                let total = a.dim(0)
                var rows: [Int]
                if target.names.count == 3 {
                    guard total % 3 == 0 else { throw ModelImportError.unexpectedTensors(sample: [key]) }
                    rows = [total / 3, total / 3, total / 3]
                } else {
                    guard let hidden = src.shape(module + ".weight")?.last, total > 3 * hidden else {
                        throw ModelImportError.unexpectedTensors(sample: [key])
                    }
                    rows = [hidden, hidden, hidden, total - 3 * hidden]
                }
                var offset = 0
                for (name, r) in zip(target.names, rows) {
                    out.append((name + "." + param, a[offset ..< offset + r]))
                    offset += r
                }
            default:
                unknown.append(key)
            }
        }
        if !unknown.isEmpty { throw ModelImportError.unexpectedTensors(sample: Array(unknown.prefix(5))) }
        return out
    }

    // MARK: Qwen-Image 2.1

    /// Qwen-Image 2.1: the diffusers names are the loader's.
    static func qwen(_ src: SourceCheckpoint) throws -> [Named] {
        guard src.layout.schema == .diffusers else {
            throw ModelImportError.unknownSchema(sample: Array(src.keys.prefix(4)))
        }
        return try src.keys.map { ($0, try src.tensor($0)) }
    }

    // MARK: Wan 2.2

    /// diffusers' `WanTransformer3DModel` name → the official Wan name (the inverse of diffusers' conversion table).
    static func wanOfficialName(fromDiffusers k: String) -> String {
        let top: [(String, String)] = [
            ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
            ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
            ("condition_embedder.time_proj.", "time_projection.1."),
            ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
            ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
            ("proj_out.", "head.head."),
        ]
        for (a, b) in top where k.hasPrefix(a) { return b + k.dropFirst(a.count) }
        if k == "scale_shift_table" { return "head.modulation" }
        guard k.hasPrefix("blocks.") else { return k }
        var parts = k.split(separator: ".").map(String.init)  // blocks, N, module…
        guard parts.count >= 3 else { return k }
        let rest = parts[2...].joined(separator: ".")
        let blockMap: [(String, String)] = [
            ("attn1.to_q.", "self_attn.q."), ("attn1.to_k.", "self_attn.k."), ("attn1.to_v.", "self_attn.v."),
            ("attn1.to_out.0.", "self_attn.o."), ("attn1.norm_q.", "self_attn.norm_q."),
            ("attn1.norm_k.", "self_attn.norm_k."),
            ("attn2.to_q.", "cross_attn.q."), ("attn2.to_k.", "cross_attn.k."), ("attn2.to_v.", "cross_attn.v."),
            ("attn2.to_out.0.", "cross_attn.o."), ("attn2.norm_q.", "cross_attn.norm_q."),
            ("attn2.norm_k.", "cross_attn.norm_k."),
            ("norm2.", "norm3."), ("ffn.net.0.proj.", "ffn.0."), ("ffn.net.2.", "ffn.2."),
        ]
        if rest == "scale_shift_table" { return "blocks.\(parts[1]).modulation" }
        for (a, b) in blockMap where rest.hasPrefix(a) {
            parts = ["blocks", parts[1], b + rest.dropFirst(a.count)]
            return parts.joined(separator: ".")
        }
        return k
    }

    /// The official Wan name → the mlx-video name ``WanDiT`` loads (`text_embedding.2` → `text_embedding_1`,
    /// `ffn.0` → `ffn.fc1`, `patch_embedding` → `patch_embedding_proj`, …).
    static func wanMLXName(official k: String) -> String {
        let top: [(String, String)] = [
            ("text_embedding.0.", "text_embedding_0."), ("text_embedding.2.", "text_embedding_1."),
            ("time_embedding.0.", "time_embedding_0."), ("time_embedding.2.", "time_embedding_1."),
            ("time_projection.1.", "time_projection."), ("patch_embedding.", "patch_embedding_proj."),
        ]
        for (a, b) in top where k.hasPrefix(a) { return b + k.dropFirst(a.count) }
        if Layout.isBlock(k, "ffn.0.") { return k.replacingOccurrences(of: ".ffn.0.", with: ".ffn.fc1.") }
        if Layout.isBlock(k, "ffn.2.") { return k.replacingOccurrences(of: ".ffn.2.", with: ".ffn.fc2.") }
        return k
    }

    /// Wan 2.2 (official, ComfyUI or diffusers names) → the mlx-video schema; the Conv3d patch embedding
    /// `[dim, C, pt, ph, pw]` becomes the linear `[dim, C·pt·ph·pw]` the DiT applies to patchified latents.
    static func wan(_ src: SourceCheckpoint) throws -> [Named] {
        guard src.layout.schema == .wan || src.layout.schema == .diffusers else {
            throw ModelImportError.unknownSchema(sample: Array(src.keys.prefix(4)))
        }
        return try src.keys.map { key in
            let official = src.layout.schema == .diffusers ? wanOfficialName(fromDiffusers: key) : key
            var a = try src.tensor(key)
            if official == "patch_embedding.weight", a.ndim == 5 { a = a.reshaped(a.dim(0), -1) }
            return (wanMLXName(official: official), a)
        }
    }

    // MARK: quantization

    /// The default choice of weights to quantize (as `WeightStore.quantizedCache`): `*.weight` matrices whose
    /// input width divides into groups.
    static func defaultShouldQuantize(groupSize: Int) -> (String, [Int]) -> Bool {
        { name, shape in name.hasSuffix(".weight") && shape.count == 2 && shape[1] % groupSize == 0 && shape[0] % 32 == 0 }
    }

    /// Dense named tensors → the tensors to write: chosen weights as `weight` (uint32) + `scales` + `biases`.
    static func quantize(_ tensors: [Named], bits: Int?, groupSize: Int,
                         should: (String, [Int]) -> Bool) -> [PlannedTensor] {
        var out: [PlannedTensor] = []
        for t in tensors {
            guard let bits, should(t.name, t.array.shape) else {
                out.append(PlannedTensor(name: t.name, array: t.array))
                continue
            }
            let (wq, scales, biases) = quantized(t.array, groupSize: groupSize, bits: bits)
            let base = String(t.name.dropLast(".weight".count))
            out.append(PlannedTensor(name: t.name, array: wq))
            out.append(PlannedTensor(name: base + ".scales", array: scales))
            if let biases { out.append(PlannedTensor(name: base + ".biases", array: biases)) }
        }
        return out
    }

    // MARK: validation

    /// Differences between the tensors about to be written and the installed base model's (names and shapes).
    static func mismatches(_ planned: [(name: String, shape: [Int])], expected: [String: [Int]]) -> [String] {
        var details: [String] = []
        let have = Dictionary(planned.map { ($0.name, $0.shape) }, uniquingKeysWith: { a, _ in a })
        let missing = expected.keys.filter { have[$0] == nil }.sorted(by: naturalLess)
        let extra = have.keys.filter { expected[$0] == nil }.sorted(by: naturalLess)
        let shapes = expected.keys.filter { have[$0] != nil && have[$0] != expected[$0] }.sorted(by: naturalLess)
        if !missing.isEmpty {
            details.append("\(missing.count) tensor(s) missing, e.g. " + missing.prefix(3).joined(separator: ", "))
        }
        if !extra.isEmpty {
            details.append("\(extra.count) unexpected tensor(s), e.g. " + extra.prefix(3).joined(separator: ", "))
        }
        if !shapes.isEmpty {
            details.append("\(shapes.count) shape(s) differ, e.g. " + shapes.prefix(3).map {
                "\($0) \(have[$0]!) vs \(expected[$0]!)"
            }.joined(separator: ", "))
        }
        return details
    }
}
