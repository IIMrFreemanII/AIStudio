import Foundation
import MLX
import MLXDiffusionCore
@testable import ModelImport
import Testing
import WanTransformer

/// A tiny Wan 2.2 expert: dim 64 (2 heads × 32), FFN 128, two blocks; T2V (16 latent channels) by default.
enum TinyWan {
    static let dim = 64, ffn = 128, layers = 2

    static func config(inDim: Int = 16, bits: Int? = 4) -> [String: Any] {
        var c: [String: Any] = [
            "model_type": inDim > 16 ? "i2v" : "t2v", "patch_size": [1, 2, 2], "text_len": 8, "in_dim": inDim,
            "dim": dim, "ffn_dim": ffn, "freq_dim": 256, "text_dim": 64, "out_dim": 16, "num_heads": 2,
            "num_layers": layers, "eps": 1e-6, "boundary": 0.875, "sample_shift": 12.0, "sample_steps": 40,
        ]
        if let bits { c["quantization"] = ["bits": bits, "group_size": 64] }
        return c
    }

    /// Random weights in the mlx-video names `WanDiT` loads (dense bf16).
    static func weights(seed: UInt64, inDim: Int = 16) -> [String: MLXArray] {
        var r = RNG(seed)
        var w: [String: MLXArray] = [:]
        r.linear("patch_embedding_proj", dim, inDim * 4, into: &w)
        r.linear("text_embedding_0", dim, 64, into: &w)
        r.linear("text_embedding_1", dim, dim, into: &w)
        r.linear("time_embedding_0", dim, 256, into: &w)
        r.linear("time_embedding_1", dim, dim, into: &w)
        r.linear("time_projection", 6 * dim, dim, into: &w)
        for i in 0..<layers {
            let p = "blocks.\(i)."
            for a in ["self_attn", "cross_attn"] {
                for n in ["q", "k", "v", "o"] { r.linear(p + "\(a).\(n)", dim, dim, into: &w) }
                r.norm(p + "\(a).norm_q.weight", dim, into: &w)
                r.norm(p + "\(a).norm_k.weight", dim, into: &w)
            }
            r.norm(p + "norm3.weight", dim, into: &w)
            w[p + "norm3.bias"] = r.normal([dim], scale: 0.1).asType(.bfloat16)
            r.linear(p + "ffn.fc1", ffn, dim, into: &w)
            r.linear(p + "ffn.fc2", dim, ffn, into: &w)
            w[p + "modulation"] = r.normal([1, 6, dim], scale: 0.1).asType(.bfloat16)
        }
        r.linear("head.head", 64, dim, into: &w)
        w["head.modulation"] = r.normal([1, 2, dim], scale: 0.1).asType(.bfloat16)
        return w
    }

    /// Only the blocks' linear layers are quantized, as in the installed mlx-video folders.
    static func quantize(_ w: [String: MLXArray], bits: Int) -> [String: MLXArray] {
        quantizeWeights(w, bits: bits) { k, a in
            k.hasPrefix("blocks.") && k.hasSuffix(".weight") && a.ndim == 2
        }
    }

    /// The official Wan repository's names (Conv3d patch embedding, `ffn.0`, `text_embedding.2`…).
    static func official(_ w: [String: MLXArray]) -> [String: MLXArray] {
        var o: [String: MLXArray] = [:]
        for (k, a) in w {
            var n = k, v = a
            let renames = [("patch_embedding_proj.", "patch_embedding."), ("text_embedding_0.", "text_embedding.0."),
                           ("text_embedding_1.", "text_embedding.2."), ("time_embedding_0.", "time_embedding.0."),
                           ("time_embedding_1.", "time_embedding.2."), ("time_projection.", "time_projection.1.")]
            for (a, b) in renames where n.hasPrefix(a) { n = b + n.dropFirst(a.count) }
            n = n.replacingOccurrences(of: ".ffn.fc1.", with: ".ffn.0.").replacingOccurrences(of: ".ffn.fc2.", with: ".ffn.2.")
            if n == "patch_embedding.weight" { v = a.reshaped(dim, -1, 1, 2, 2) }
            o[n] = v
        }
        return o
    }

    /// diffusers' `WanTransformer3DModel` names, from the official ones (as diffusers' conversion script renames).
    static func diffusers(_ official: [String: MLXArray]) -> [String: MLXArray] {
        var o: [String: MLXArray] = [:]
        let top = [("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
                   ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
                   ("time_projection.1.", "condition_embedder.time_proj."),
                   ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
                   ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
                   ("head.head.", "proj_out.")]
        let block = [("self_attn.q.", "attn1.to_q."), ("self_attn.k.", "attn1.to_k."), ("self_attn.v.", "attn1.to_v."),
                     ("self_attn.o.", "attn1.to_out.0."), ("self_attn.norm_q.", "attn1.norm_q."),
                     ("self_attn.norm_k.", "attn1.norm_k."), ("cross_attn.q.", "attn2.to_q."),
                     ("cross_attn.k.", "attn2.to_k."), ("cross_attn.v.", "attn2.to_v."),
                     ("cross_attn.o.", "attn2.to_out.0."), ("cross_attn.norm_q.", "attn2.norm_q."),
                     ("cross_attn.norm_k.", "attn2.norm_k."), ("norm3.", "norm2."), ("ffn.0.", "ffn.net.0.proj."),
                     ("ffn.2.", "ffn.net.2.")]
        for (k, a) in official {
            var n = k
            for (x, y) in top where n.hasPrefix(x) { n = y + n.dropFirst(x.count) }
            if n == "head.modulation" { n = "scale_shift_table" }
            if n.hasPrefix("blocks.") {
                let parts = n.split(separator: ".", maxSplits: 2).map(String.init)
                var rest = parts[2]
                for (x, y) in block where rest.hasPrefix(x) { rest = y + rest.dropFirst(x.count) }
                if rest == "modulation" { rest = "scale_shift_table" }
                n = "blocks.\(parts[1]).\(rest)"
            }
            o[n] = a
        }
        return o
    }

    /// An installed-looking Wan 2.2 T2V folder (the PocketAiHub layout): `q4/` experts, `common/`, `lora/`.
    static func makeBase(_ dir: URL, high: [String: MLXArray], low: [String: MLXArray], bits: Int = 4) throws {
        let fm = FileManager.default
        let q = dir.appendingPathComponent("q\(bits)")
        try fm.createDirectory(at: q, withIntermediateDirectories: true)
        try writeJSON(config(bits: bits), q.appendingPathComponent("config.json"))
        try writeSafetensors(quantize(high, bits: bits), to: q.appendingPathComponent("high_noise_model.safetensors"))
        try writeSafetensors(quantize(low, bits: bits), to: q.appendingPathComponent("low_noise_model.safetensors"))
        let common = dir.appendingPathComponent("common")
        try fm.createDirectory(at: common, withIntermediateDirectories: true)
        for f in ["t5_encoder.safetensors", "vae.safetensors", "tokenizer.json"] {
            try Data("x".utf8).write(to: common.appendingPathComponent(f))
        }
        let lora = dir.appendingPathComponent("lora/Lightning")
        try fm.createDirectory(at: lora, withIntermediateDirectories: true)
        for f in ["high_noise_model.safetensors", "low_noise_model.safetensors"] {
            try Data("x".utf8).write(to: lora.appendingPathComponent(f))
        }
        try writeJSON(["class": "WanPipeline"], dir.appendingPathComponent("mlxcomfy.json"))
        try Data("readme".utf8).write(to: dir.appendingPathComponent("README.md"))
    }

    /// One expert of a model folder, loaded as `WanPipeline` does (config found first at the top, bits from it).
    static func expert(_ dir: URL, _ e: WanExpert) throws -> WanDiT {
        let fm = FileManager.default
        let modelDir = ["", "q4", "q8"].map { $0.isEmpty ? dir : dir.appendingPathComponent($0) }
            .first { fm.fileExists(atPath: $0.appendingPathComponent("config.json").path) }!
        let configURL = modelDir.appendingPathComponent("config.json")
        let o = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
        let store = try WeightStore(url: modelDir.appendingPathComponent(e.fileName))
        if let bits = (o["quantization"] as? [String: Any])?["bits"] as? Int { store.bitsOverride = bits }
        return try WanDiT(store: store, config: .init(url: configURL))
    }

    /// The reference: the given dense weights quantized as the base's and loaded the same way.
    static func reference(_ w: [String: MLXArray], bits: Int = 4) throws -> WanDiT {
        let store = WeightStore(arrays: quantize(w, bits: bits))
        store.bitsOverride = bits
        var c = WanDiT.Config()
        c.dim = dim; c.ffnDim = ffn; c.textDim = 64; c.textLen = 8; c.numHeads = 2; c.numLayers = layers
        return try WanDiT(store: store, config: c)
    }

    static func forward(_ m: WanDiT) -> MLXArray {
        var r = RNG(5)
        let x = r.normal([16, 1, 4, 4]), ctx = r.normal([5, 64])
        let out = m(x, timestep: 700, crossKV: m.crossKV(m.embedText(ctx)), rope: m.rope(frames: 1, height: 2, width: 2))
        eval(out)
        return out
    }
}

@Suite(.serialized) struct WanImportTests {
    let tmp = TempDir("wan-import")
    var models: URL { tmp.url.appendingPathComponent("models") }
    var base: URL { models.appendingPathComponent("wan-2.2-t2v") }
    let baseHigh = TinyWan.weights(seed: 1), baseLow = TinyWan.weights(seed: 2)

    init() throws { try TinyWan.makeBase(base, high: baseHigh, low: baseLow) }

    /// The folder resolves like the base (config at the top, shared parts through links) and each expert computes
    /// what its reference weights compute.
    func check(_ dest: URL, high: [String: MLXArray], low: [String: MLXArray]) throws {
        for (e, w) in [(WanExpert.high, high), (.low, low)] {
            let got = TinyWan.forward(try TinyWan.expert(dest, e))
            #expect(maxDiff(TinyWan.forward(try TinyWan.reference(w)), got) < 1e-4, "\(e) expert differs")
        }
        let fm = FileManager.default
        for f in ["common/t5_encoder.safetensors", "common/vae.safetensors", "common/tokenizer.json"] {
            #expect(fm.fileExists(atPath: dest.appendingPathComponent(f).path), "\(f) not reachable")
        }
        #expect(listing(dest.appendingPathComponent("common")) == ["t5_encoder.safetensors", "tokenizer.json", "vae.safetensors"])
        #expect(linkTarget(dest.appendingPathComponent("common/vae.safetensors")) == "../../wan-2.2-t2v/common/vae.safetensors")
        // The pipeline finds the Lightning LoRAs by enumerating lora/.
        let loras = (fm.enumerator(at: dest.appendingPathComponent("lora"), includingPropertiesForKeys: nil)?
            .compactMap { ($0 as? URL)?.lastPathComponent }.filter { $0.hasSuffix(".safetensors") } ?? []).sorted()
        #expect(loras == ["high_noise_model.safetensors", "low_noise_model.safetensors"])
        #expect(linkTarget(dest.appendingPathComponent("lora/Lightning/high_noise_model.safetensors"))
            == "../../../wan-2.2-t2v/lora/Lightning/high_noise_model.safetensors")
        #expect(!fm.fileExists(atPath: dest.appendingPathComponent("q4").path))
        #expect(ModelRegistry.className(at: dest) == "WanPipeline")
        let cfg = try JSONSerialization.jsonObject(with: Data(contentsOf: dest.appendingPathComponent("config.json")))
            as? [String: Any]
        #expect((cfg?["quantization"] as? [String: Any])?["bits"] as? Int == 4)
        #expect(partials(in: models).isEmpty)
    }

    /// One official bf16 expert (named …HighNoise…); the low-noise expert stays the base's, linked.
    @Test func officialHighExpertOnly() throws {
        let w = TinyWan.weights(seed: 3)
        let src = tmp.url.appendingPathComponent("wan22-ft-HighNoise.safetensors")
        try writeSafetensors(TinyWan.official(w), to: src)
        let info = try ModelImporter.inspect(src)
        #expect(info.architecture == .wan22 && info.schema == .wan && info.wanExperts == [.high])

        let dest = models.appendingPathComponent("wan-ft")
        let r = try ModelImporter.importTransformer(from: src, base: base, into: dest)
        #expect(r.importedExperts == [.high] && r.bits == 4 && r.family == "wan-2.2-t2v")
        #expect(linkTarget(dest.appendingPathComponent("low_noise_model.safetensors"))
            == "../wan-2.2-t2v/q4/low_noise_model.safetensors")
        // Stored like the base: same tensors, shapes and dtypes.
        let h = try SafetensorsHeader(url: dest.appendingPathComponent("high_noise_model.safetensors"))
        let b = try SafetensorsHeader(url: base.appendingPathComponent("q4/high_noise_model.safetensors"))
        #expect(h.tensors.mapValues { [$0.dtype] + $0.shape.map(String.init) }
            == b.tensors.mapValues { [$0.dtype] + $0.shape.map(String.init) })
        try check(dest, high: w, low: baseLow)
    }

    /// A ComfyUI-prefixed FP8 low-noise expert, given explicitly.
    @Test func comfyFP8LowExpert() throws {
        let w = TinyWan.weights(seed: 4)
        var file: [String: MLXArray] = [:]
        var fp8: Set<String> = []
        var ref = w
        for (k, a) in TinyWan.official(w) {
            file["model.diffusion_model." + k] = a
            if k.hasSuffix(".weight"), a.ndim >= 2 { fp8.insert("model.diffusion_model." + k) }
        }
        for (k, a) in w where k.hasSuffix(".weight") && a.ndim == 2 { ref[k] = fp8RoundTrip(a) }
        let src = tmp.url.appendingPathComponent("wan-low.safetensors")
        try writeSafetensors(file, to: src, fp8: fp8)

        let dest = models.appendingPathComponent("wan-fp8")
        let r = try ModelImporter.importWan(high: nil, low: src, base: base, into: dest)
        #expect(r.importedExperts == [.low] && r.sourceDtype == "F8_E4M3")
        try check(dest, high: baseHigh, low: ref)
    }

    /// A diffusers Wan 2.2 folder: `transformer/` (high noise) and `transformer_2/` (low noise), fp32.
    @Test func diffusersBothExperts() throws {
        let wh = TinyWan.weights(seed: 5), wl = TinyWan.weights(seed: 6)
        let folder = tmp.sub("Wan2.2-T2V-ft-Diffusers")
        for (sub, w) in [("transformer", wh), ("transformer_2", wl)] {
            let d = tmp.sub("Wan2.2-T2V-ft-Diffusers/\(sub)")
            try writeSafetensors(TinyWan.diffusers(TinyWan.official(w)).mapValues { $0.asType(.float32) },
                                 to: d.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        }
        let info = try ModelImporter.inspect(folder)
        #expect(info.schema == .diffusers && info.wanExperts == [.high, .low])
        let dest = models.appendingPathComponent("wan-hf")
        let r = try ModelImporter.importTransformer(from: folder, family: "wan-2.2", base: base, into: dest)
        #expect(r.importedExperts == [.high, .low])
        #expect(linkTarget(dest.appendingPathComponent("high_noise_model.safetensors")) == nil)
        try check(dest, high: wh, low: wl)
    }

    /// An I2V checkpoint onto the T2V base is refused with a hint; a file that names no expert asks for one.
    @Test func mismatchesAreExplained() throws {
        let src = tmp.url.appendingPathComponent("wan-i2v-high.safetensors")
        try writeSafetensors(TinyWan.official(TinyWan.weights(seed: 7, inDim: 36)), to: src)
        let e = errorText { try ModelImporter.importTransformer(from: src, base: base, into: models.appendingPathComponent("x")) }
        #expect(e?.contains("image-to-video (I2V)") == true)

        let plain = tmp.url.appendingPathComponent("wan-finetune.safetensors")
        try writeSafetensors(TinyWan.official(TinyWan.weights(seed: 8)), to: plain)
        let e2 = errorText { try ModelImporter.importTransformer(from: plain, base: base, into: models.appendingPathComponent("y")) }
        #expect(e2?.contains("high noise or low noise") == true)
        #expect(partials(in: models).isEmpty)
    }
}
