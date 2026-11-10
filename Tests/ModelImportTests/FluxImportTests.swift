import Flux
import FluxTransformer
import Foundation
import MLX
import MLXDiffusionCore
@testable import ModelImport
import Testing

/// A tiny FLUX.1 dev: 2 heads × 32, two double and two single blocks (widths multiple of 64 so they quantize).
enum TinyFlux {
    static let d = 64, mlp = 256, layers = 2, singles = 2

    static let config: [String: Any] = [
        "_class_name": "FluxTransformer2DModel", "in_channels": 64, "num_layers": layers,
        "num_single_layers": singles, "attention_head_dim": 32, "num_attention_heads": 2,
        "joint_attention_dim": 64, "pooled_projection_dim": 64, "guidance_embeds": true,
        "axes_dims_rope": [4, 14, 14], "patch_size": 1,
    ]

    /// Random weights in diffusers names (bf16).
    static func weights(seed: UInt64, guidance: Bool = true) -> [String: MLXArray] {
        var r = RNG(seed)
        var w: [String: MLXArray] = [:]
        r.linear("x_embedder", d, 64, into: &w)
        r.linear("context_embedder", d, 64, into: &w)
        r.linear("time_text_embed.timestep_embedder.linear_1", d, 256, into: &w)
        r.linear("time_text_embed.timestep_embedder.linear_2", d, d, into: &w)
        if guidance {
            r.linear("time_text_embed.guidance_embedder.linear_1", d, 256, into: &w)
            r.linear("time_text_embed.guidance_embedder.linear_2", d, d, into: &w)
        }
        r.linear("time_text_embed.text_embedder.linear_1", d, 64, into: &w)
        r.linear("time_text_embed.text_embedder.linear_2", d, d, into: &w)
        for i in 0..<layers {
            let p = "transformer_blocks.\(i)."
            r.linear(p + "norm1.linear", 6 * d, d, into: &w)
            r.linear(p + "norm1_context.linear", 6 * d, d, into: &w)
            for n in ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"] {
                r.linear(p + "attn." + n, d, d, into: &w)
            }
            for n in ["norm_q", "norm_k", "norm_added_q", "norm_added_k"] { r.norm(p + "attn.\(n).weight", 32, into: &w) }
            r.linear(p + "ff.net.0.proj", mlp, d, into: &w)
            r.linear(p + "ff.net.2", d, mlp, into: &w)
            r.linear(p + "ff_context.net.0.proj", mlp, d, into: &w)
            r.linear(p + "ff_context.net.2", d, mlp, into: &w)
        }
        for i in 0..<singles {
            let p = "single_transformer_blocks.\(i)."
            r.linear(p + "norm.linear", 3 * d, d, into: &w)
            for n in ["to_q", "to_k", "to_v"] { r.linear(p + "attn." + n, d, d, into: &w) }
            r.linear(p + "proj_mlp", mlp, d, into: &w)
            r.linear(p + "proj_out", d, d + mlp, into: &w)
            for n in ["norm_q", "norm_k"] { r.norm(p + "attn.\(n).weight", 32, into: &w) }
        }
        r.linear("norm_out.linear", 2 * d, d, into: &w)
        r.linear("proj_out", 64, d, into: &w)
        return w
    }

    /// The same weights in BFL's original names, written out independently of the importer's table: fused qkv
    /// and linear1, `(shift, scale)` final modulation, QK norm `scale`s.
    static func bfl(_ w: [String: MLXArray]) -> [String: MLXArray] {
        var o: [String: MLXArray] = [:]
        func copy(_ from: String, _ to: String) {
            for s in ["weight", "bias"] { if let a = w["\(from).\(s)"] { o["\(to).\(s)"] = a } }
        }
        func fuse(_ from: [String], _ to: String) {
            for s in ["weight", "bias"] { o["\(to).\(s)"] = concatenated(from.map { w["\($0).\(s)"]! }, axis: 0) }
        }
        copy("x_embedder", "img_in"); copy("context_embedder", "txt_in")
        copy("time_text_embed.timestep_embedder.linear_1", "time_in.in_layer")
        copy("time_text_embed.timestep_embedder.linear_2", "time_in.out_layer")
        copy("time_text_embed.guidance_embedder.linear_1", "guidance_in.in_layer")
        copy("time_text_embed.guidance_embedder.linear_2", "guidance_in.out_layer")
        copy("time_text_embed.text_embedder.linear_1", "vector_in.in_layer")
        copy("time_text_embed.text_embedder.linear_2", "vector_in.out_layer")
        copy("proj_out", "final_layer.linear")
        for s in ["weight", "bias"] {
            let a = w["norm_out.linear.\(s)"]!
            o["final_layer.adaLN_modulation.1.\(s)"] = concatenated([a[d...], a[..<d]], axis: 0)
        }
        for i in 0..<layers {
            let p = "transformer_blocks.\(i).", q = "double_blocks.\(i)."
            copy(p + "norm1.linear", q + "img_mod.lin"); copy(p + "norm1_context.linear", q + "txt_mod.lin")
            fuse(["to_q", "to_k", "to_v"].map { p + "attn." + $0 }, q + "img_attn.qkv")
            fuse(["add_q_proj", "add_k_proj", "add_v_proj"].map { p + "attn." + $0 }, q + "txt_attn.qkv")
            copy(p + "attn.to_out.0", q + "img_attn.proj"); copy(p + "attn.to_add_out", q + "txt_attn.proj")
            o[q + "img_attn.norm.query_norm.scale"] = w[p + "attn.norm_q.weight"]
            o[q + "img_attn.norm.key_norm.scale"] = w[p + "attn.norm_k.weight"]
            o[q + "txt_attn.norm.query_norm.scale"] = w[p + "attn.norm_added_q.weight"]
            o[q + "txt_attn.norm.key_norm.scale"] = w[p + "attn.norm_added_k.weight"]
            copy(p + "ff.net.0.proj", q + "img_mlp.0"); copy(p + "ff.net.2", q + "img_mlp.2")
            copy(p + "ff_context.net.0.proj", q + "txt_mlp.0"); copy(p + "ff_context.net.2", q + "txt_mlp.2")
        }
        for i in 0..<singles {
            let p = "single_transformer_blocks.\(i).", q = "single_blocks.\(i)."
            fuse(["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"].map { p + $0 }, q + "linear1")
            copy(p + "proj_out", q + "linear2"); copy(p + "norm.linear", q + "modulation.lin")
            o[q + "norm.query_norm.scale"] = w[p + "attn.norm_q.weight"]
            o[q + "norm.key_norm.scale"] = w[p + "attn.norm_k.weight"]
        }
        return o
    }

    /// An installed-looking FLUX.1 dev folder: dense diffusers transformer, text encoders, tokenizers, VAE, scheduler.
    static func makeBase(_ dir: URL, weights: [String: MLXArray]) throws {
        let t = dir.appendingPathComponent("transformer")
        try FileManager.default.createDirectory(at: t, withIntermediateDirectories: true)
        try writeSafetensors(weights, to: t.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try writeJSON(config, t.appendingPathComponent("config.json"))
        try writeJSON(["_class_name": "FluxPipeline"], dir.appendingPathComponent("model_index.json"))
        for c in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae", "scheduler", ".mlxcomfy-cache"] {
            try placeholder(dir.appendingPathComponent(c))
        }
        try Data("readme".utf8).write(to: dir.appendingPathComponent("README.md"))
    }

    /// A transformer from a folder exactly as `FluxPipeline` loads it.
    static func load(_ dir: URL) throws -> FluxTransformer {
        let store = try FluxPipeline.transformerStore(directory: dir)
        return try FluxTransformer(store: store, config: .init(url: dir.appendingPathComponent("transformer/config.json")))
    }

    /// The reference: the same weights in a plain diffusers folder, loaded (and 8-bit quantized) as the base is.
    static func reference(_ w: [String: MLXArray], in tmp: TempDir) throws -> FluxTransformer {
        let dir = tmp.sub("reference-\(UUID().uuidString.prefix(6))")
        try makeBase(dir, weights: w)
        return try load(dir)
    }

    static func forward(_ t: FluxTransformer) -> MLXArray {
        var r = RNG(99)
        let x = r.normal([1, 6, 64]), ctx = r.normal([1, 5, 64]), pooled = r.normal([1, 64])
        let vec = t.conditioningVector(sigma: 0.7, guidance: 3.5, pooled: pooled)
        let out = t(x, context: ctx, vec: vec, rope: t.rope(textLength: 5, height: 2, width: 3))
        eval(out)
        return out
    }
}

@Suite(.serialized) struct FluxImportTests {
    let tmp = TempDir("flux-import")
    var models: URL { tmp.url.appendingPathComponent("models") }
    var base: URL { models.appendingPathComponent("flux-1-dev") }

    init() throws {
        try TinyFlux.makeBase(base, weights: TinyFlux.weights(seed: 1))
    }

    /// Checks the assembled folder and that it computes what the fine-tune's own weights compute.
    func check(_ dest: URL, reference: [String: MLXArray]) throws {
        let fm = FileManager.default
        // Own 8-bit transformer, used as is (no quantized cache is built for it).
        let t = try TinyFlux.load(dest)
        #expect(!fm.fileExists(atPath: dest.appendingPathComponent(".mlxcomfy-cache").path))
        let header = try SafetensorsHeader(url: dest.appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        #expect(header.metadata["quantization_level"] == "8")
        #expect(header.tensors["transformer_blocks.0.attn.to_q.scales"] != nil)
        #expect(header.tensors["transformer_blocks.0.attn.to_q.weight"]?.dtype == "U32")

        let want = TinyFlux.forward(try TinyFlux.reference(reference, in: tmp))
        let got = TinyFlux.forward(t)
        #expect(maxDiff(want, got) < 1e-4, "converted transformer differs from the reference")
        // …and it is the fine-tune, not the base.
        #expect(maxDiff(TinyFlux.forward(try TinyFlux.load(base)), got) > 1e-2)

        // The base's other parts are linked file by file (relatively) in listable folders, model_index.json copied,
        // docs and caches left out.
        for c in ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "vae", "scheduler"] {
            #expect(listing(dest.appendingPathComponent(c)) == ["weights.bin"])
            #expect(linkTarget(dest.appendingPathComponent("\(c)/weights.bin")) == "../../flux-1-dev/\(c)/weights.bin")
            #expect(fm.fileExists(atPath: dest.appendingPathComponent("\(c)/weights.bin").path))
        }
        #expect(linkTarget(dest.appendingPathComponent("model_index.json")) == nil)
        #expect(ModelRegistry.className(at: dest) == "FluxPipeline")
        #expect(!fm.fileExists(atPath: dest.appendingPathComponent("README.md").path))
        let link = try JSONSerialization.jsonObject(with: Data(contentsOf: dest.appendingPathComponent(
            ModelImporter.linkManifestName))) as? [String: Any]
        #expect(link?["base"] as? String == "flux-1-dev" && link?["family"] as? String == "flux.1")
        #expect(partials(in: models).isEmpty)
    }

    @Test func bflSingleFileBF16() throws {
        let w = TinyFlux.weights(seed: 2)
        let src = tmp.url.appendingPathComponent("my-flux-finetune.safetensors")
        try writeSafetensors(TinyFlux.bfl(w), to: src)

        let info = try ModelImporter.inspect(src)
        #expect(info.architecture == .flux1 && info.schema == .bfl && info.prefix == "" && info.weightDtype == "BF16")

        var fractions: [Double] = []
        let dest = models.appendingPathComponent("my-flux")
        let r = try ModelImporter.importTransformer(from: src, base: base, into: dest,
                                                    sourceInfo: ["url": "https://example.com/m/1"]) {
            fractions.append($0.fraction)
        }
        #expect(r.schema == .bfl && r.bits == 8 && r.family == "flux.1")
        #expect(fractions.last == 1 && fractions == fractions.sorted())
        try check(dest, reference: w)
        let link = try JSONSerialization.jsonObject(with: Data(contentsOf: dest.appendingPathComponent(
            ModelImporter.linkManifestName))) as? [String: Any]
        #expect((link?["source"] as? [String: Any])?["url"] as? String == "https://example.com/m/1")
    }

    /// A ComfyUI full checkpoint: transformer under `model.diffusion_model.` in FP8 (some layers with ComfyUI's
    /// `scale_weight`), plus a VAE and text encoders that are dropped.
    @Test func comfyFullCheckpointFP8() throws {
        let w = TinyFlux.weights(seed: 3)
        let bfl = TinyFlux.bfl(w)
        // FP8-store the 2-D weights in BFL names, then derive what the reference (diffusers names) holds.
        var stored: [String: MLXArray] = [:], fp8: Set<String> = []
        let scaled = Set(["double_blocks.0.img_attn.qkv.weight", "single_blocks.1.linear1.weight", "img_in.weight"])
        var decoded: [String: MLXArray] = [:]
        for (k, a) in bfl {
            let key = "model.diffusion_model." + k
            guard k.hasSuffix(".weight"), a.ndim == 2 else { stored[key] = a; decoded[k] = a; continue }
            let s: Float? = scaled.contains(k) ? 0.5 : nil
            stored[key] = s.map { a / $0 } ?? a
            fp8.insert(key)
            if let s { stored[String(key.dropLast(".weight".count)) + ".scale_weight"] = MLXArray(s) }
            decoded[k] = fp8RoundTrip(a, scale: s)
        }
        stored["model.diffusion_model.scaled_fp8"] = MLXArray.zeros([0], dtype: .float32)
        stored["vae.decoder.conv_in.weight"] = MLXArray.zeros([8, 4, 3, 3], dtype: .bfloat16)
        stored["text_encoders.clip_l.transformer.text_model.embeddings.token_embedding.weight"] =
            MLXArray.zeros([10, 8], dtype: .bfloat16)
        // The reference holds the decoded values under diffusers names.
        let ref = Self.diffusersFromDecoded(decoded, original: w)

        let src = tmp.url.appendingPathComponent("flux-finetune-fp8.safetensors")
        try writeSafetensors(stored, to: src, fp8: fp8)
        let info = try ModelImporter.inspect(src)
        #expect(info.schema == .bfl && info.prefix == "model.diffusion_model." && info.weightDtype == "F8_E4M3")
        #expect(info.droppedComponents == ["text_encoders", "vae"])

        let dest = models.appendingPathComponent("my-flux-fp8")
        let r = try ModelImporter.importTransformer(from: src, family: "flux.1", base: base, into: dest)
        #expect(r.sourceDtype == "F8_E4M3" && r.droppedComponents == ["text_encoders", "vae"])
        try check(dest, reference: ref)
    }

    /// Diffusers weights for decoded BFL tensors: undoes the fusing exactly as the test's `bfl` did it.
    static func diffusersFromDecoded(_ decoded: [String: MLXArray], original w: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let d = TinyFlux.d
        for (k, a) in w {
            let parts = k.split(separator: ".").map(String.init)
            let s = parts.last!
            func from(_ bflName: String, rows: Range<Int>? = nil) -> MLXArray {
                let b = decoded["\(bflName).\(s)"]!
                return rows.map { b[$0] } ?? b
            }
            let name = parts.dropLast().joined(separator: ".")
            var v: MLXArray?
            if name.hasPrefix("transformer_blocks.") {
                let i = parts[1], q = "double_blocks.\(i)."
                let m = parts[2...].dropLast().joined(separator: ".")
                let map: [String: (String, Range<Int>?)] = [
                    "norm1.linear": ("img_mod.lin", nil), "norm1_context.linear": ("txt_mod.lin", nil),
                    "attn.to_q": ("img_attn.qkv", 0 ..< d), "attn.to_k": ("img_attn.qkv", d ..< 2 * d),
                    "attn.to_v": ("img_attn.qkv", 2 * d ..< 3 * d),
                    "attn.add_q_proj": ("txt_attn.qkv", 0 ..< d), "attn.add_k_proj": ("txt_attn.qkv", d ..< 2 * d),
                    "attn.add_v_proj": ("txt_attn.qkv", 2 * d ..< 3 * d),
                    "attn.to_out.0": ("img_attn.proj", nil), "attn.to_add_out": ("txt_attn.proj", nil),
                    "ff.net.0.proj": ("img_mlp.0", nil), "ff.net.2": ("img_mlp.2", nil),
                    "ff_context.net.0.proj": ("txt_mlp.0", nil), "ff_context.net.2": ("txt_mlp.2", nil),
                ]
                if let (b, r) = map[m] { v = from(q + b, rows: r) }
            } else if name.hasPrefix("single_transformer_blocks.") {
                let i = parts[1], q = "single_blocks.\(i)."
                let m = parts[2...].dropLast().joined(separator: ".")
                let map: [String: (String, Range<Int>?)] = [
                    "attn.to_q": ("linear1", 0 ..< d), "attn.to_k": ("linear1", d ..< 2 * d),
                    "attn.to_v": ("linear1", 2 * d ..< 3 * d), "proj_mlp": ("linear1", 3 * d ..< 3 * d + TinyFlux.mlp),
                    "proj_out": ("linear2", nil), "norm.linear": ("modulation.lin", nil),
                ]
                if let (b, r) = map[m] { v = from(q + b, rows: r) }
            } else {
                let map = ["x_embedder": "img_in", "context_embedder": "txt_in",
                           "time_text_embed.timestep_embedder.linear_1": "time_in.in_layer",
                           "time_text_embed.timestep_embedder.linear_2": "time_in.out_layer",
                           "time_text_embed.guidance_embedder.linear_1": "guidance_in.in_layer",
                           "time_text_embed.guidance_embedder.linear_2": "guidance_in.out_layer",
                           "time_text_embed.text_embedder.linear_1": "vector_in.in_layer",
                           "time_text_embed.text_embedder.linear_2": "vector_in.out_layer", "proj_out": "final_layer.linear"]
                if let b = map[name] { v = from(b) }
                if name == "norm_out.linear" {
                    let b = from("final_layer.adaLN_modulation.1")
                    v = concatenated([b[d...], b[..<d]], axis: 0)
                }
            }
            // QK norms are 1-D and not FP8-stored: unchanged.
            out[k] = v ?? a
        }
        return out
    }

    /// A diffusers model folder with the transformer in fp16 shards.
    @Test func diffusersFolderFP16Shards() throws {
        let w = TinyFlux.weights(seed: 4)
        let folder = tmp.sub("hf-flux-finetune")
        let t = tmp.sub("hf-flux-finetune/transformer")
        let keys = w.keys.sorted()
        let half = keys.count / 2
        let fp16 = w.mapValues { $0.asType(.float16) }
        try writeSafetensors(fp16.filter { keys[..<half].contains($0.key) },
                             to: t.appendingPathComponent("diffusion_pytorch_model-00001-of-00002.safetensors"))
        try writeSafetensors(fp16.filter { keys[half...].contains($0.key) },
                             to: t.appendingPathComponent("diffusion_pytorch_model-00002-of-00002.safetensors"))
        try writeJSON(TinyFlux.config, t.appendingPathComponent("config.json"))
        try writeJSON(["_class_name": "FluxPipeline"], folder.appendingPathComponent("model_index.json"))

        let info = try ModelImporter.inspect(folder)
        #expect(info.schema == .diffusers && info.files.count == 2 && info.weightDtype == "F16")
        let dest = models.appendingPathComponent("hf-flux")
        try ModelImporter.importTransformer(from: folder, base: base, into: dest)
        try check(dest, reference: fp16.mapValues { $0.asType(.bfloat16) })
    }

    /// FLUX.1 schnell has no guidance embedding: refused with a hint, and nothing is left behind.
    @Test func schnellIsRefused() throws {
        let src = tmp.url.appendingPathComponent("schnell.safetensors")
        try writeSafetensors(TinyFlux.bfl(TinyFlux.weights(seed: 5, guidance: false)), to: src)
        let dest = models.appendingPathComponent("schnell")
        let e = errorText { try ModelImporter.importTransformer(from: src, base: base, into: dest) }
        #expect(e?.contains("schnell") == true)
        #expect(!FileManager.default.fileExists(atPath: dest.path) && partials(in: models).isEmpty)
    }

    @Test func existingDestinationIsRefused() throws {
        let src = tmp.url.appendingPathComponent("ft.safetensors")
        try writeSafetensors(TinyFlux.bfl(TinyFlux.weights(seed: 6)), to: src)
        let e = errorText { try ModelImporter.importTransformer(from: src, base: base, into: base) }
        #expect(e?.contains("already exists") == true)
    }
}
