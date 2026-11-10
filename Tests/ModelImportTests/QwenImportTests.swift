import Foundation
import MLX
import MLXDiffusionCore
@testable import ModelImport
import QwenImage21
import Testing

/// A tiny Qwen-Image 2.1 transformer: 2 heads × 32, two blocks, no linear biases (as the released weights).
enum TinyQwen {
    static let d = 64

    static let config: [String: Any] = [
        "_class_name": "QwenImage21Transformer2DModel", "attention_head_dim": 32, "num_attention_heads": 2,
        "num_layers": 2, "axes_dims_rope": [4, 14, 14], "causal_condition": true, "context_in_dim": 64,
        "eps": 1e-6, "in_channels": 64, "out_channels": 64, "mlp_ratio": 3, "patch_size": 1,
    ]

    static func weights(seed: UInt64) -> [String: MLXArray] {
        var r = RNG(seed)
        var w: [String: MLXArray] = [:]
        r.norm("txt_in.text_norm.weight", 64, into: &w)
        w["txt_in.text_norm.weight"] = w["txt_in.text_norm.weight"]! - 1  // zero-centred
        r.linear("txt_in.in_layer", d, 64, bias: false, into: &w)
        r.linear("txt_in.out_layer", d, d, bias: false, into: &w)
        r.linear("img_in", d, 64, bias: false, into: &w)
        r.linear("time_text_embed.timestep_embedder.linear_1", d, 256, bias: false, into: &w)
        r.linear("time_text_embed.timestep_embedder.linear_2", d, d, bias: false, into: &w)
        r.linear("modulation.1", 4 * d, d, bias: false, into: &w)
        for i in 0..<2 {
            let p = "transformer_blocks.\(i)."
            for n in ["to_q", "to_k", "to_v", "to_out.0"] { r.linear(p + "attn." + n, d, d, bias: false, into: &w) }
            r.norm(p + "attn.norm_q.weight", 32, into: &w)
            r.norm(p + "attn.norm_k.weight", 32, into: &w)
            r.linear(p + "img_mlp.proj", 3 * d, d, bias: false, into: &w)
            r.linear(p + "img_mlp.gate_layer", 3 * d, d, bias: false, into: &w)
            r.linear(p + "img_mlp.out", d, 3 * d, bias: false, into: &w)
        }
        r.linear("norm_out.linear", d, d, bias: false, into: &w)
        r.linear("proj_out", 64, d, bias: false, into: &w)
        return w
    }

    /// Every 2-D weight 8-bit, as the installed `qwen-turbo-mlx` folder.
    static func quantize(_ w: [String: MLXArray]) -> [String: MLXArray] {
        quantizeWeights(w, bits: 8) { k, a in k.hasSuffix(".weight") && a.ndim == 2 }
    }

    /// An installed-looking Qwen-Image 2.1 folder (MLX 8-bit transformer).
    static func makeFolder(_ dir: URL, weights: [String: MLXArray]) throws {
        let t = dir.appendingPathComponent("transformer")
        try FileManager.default.createDirectory(at: t, withIntermediateDirectories: true)
        try writeSafetensors(quantize(weights), to: t.appendingPathComponent("model.safetensors"),
                             metadata: ["quantization_level": "8"])
        try writeJSON(config, t.appendingPathComponent("config.json"))
        try writeJSON(["_class_name": "QwenImage21Pipeline"], dir.appendingPathComponent("model_index.json"))
        for c in ["processor", "text_encoder", "vae", "scheduler"] { try placeholder(dir.appendingPathComponent(c)) }
    }

    static func forward(_ dir: URL) throws -> MLXArray {
        let t = try QwenImage21Transformer(directory: dir.appendingPathComponent("transformer"))
        var r = RNG(42)
        let prefix = t.encodePrefix([.text(r.normal([1, 5, 64]))])
        let out = t.denoise(latents: r.normal([1, 6, 64]), prefix: prefix, sigma: 0.6, latentHeight: 2, latentWidth: 3)
        eval(out)
        return out
    }
}

@Suite(.serialized) struct QwenImportTests {
    let tmp = TempDir("qwen-import")
    var models: URL { tmp.url.appendingPathComponent("models") }
    var base: URL { models.appendingPathComponent("qwen-turbo-mlx") }

    init() throws { try TinyQwen.makeFolder(base, weights: TinyQwen.weights(seed: 1)) }

    func check(_ dest: URL, reference: [String: MLXArray]) throws {
        let refDir = tmp.sub("reference")
        try TinyQwen.makeFolder(refDir, weights: reference)
        let got = try TinyQwen.forward(dest)
        #expect(maxDiff(try TinyQwen.forward(refDir), got) < 1e-4)
        #expect(maxDiff(try TinyQwen.forward(base), got) > 1e-2)
        // Stored like the base: same tensors, shapes and dtypes, 8-bit metadata.
        let h = try SafetensorsHeader(url: dest.appendingPathComponent("transformer/model.safetensors"))
        let b = try SafetensorsHeader(url: base.appendingPathComponent("transformer/model.safetensors"))
        #expect(h.tensors.mapValues { [$0.dtype] + $0.shape.map(String.init) }
            == b.tensors.mapValues { [$0.dtype] + $0.shape.map(String.init) })
        #expect(h.metadata["quantization_level"] == "8")
        for c in ["processor", "text_encoder", "vae", "scheduler"] {
            #expect(listing(dest.appendingPathComponent(c)) == ["weights.bin"])
            #expect(linkTarget(dest.appendingPathComponent("\(c)/weights.bin")) == "../../qwen-turbo-mlx/\(c)/weights.bin")
        }
        #expect(ModelRegistry.className(at: dest) == "QwenImage21Pipeline")
    }

    @Test func comfyPrefixedBF16() throws {
        let w = TinyQwen.weights(seed: 2)
        var file: [String: MLXArray] = [:]
        for (k, a) in w { file["model.diffusion_model." + k] = a }
        file["vae.decoder.conv_in.weight"] = MLXArray.zeros([4, 4], dtype: .bfloat16)
        let src = tmp.url.appendingPathComponent("qwen21-finetune.safetensors")
        try writeSafetensors(file, to: src)
        let info = try ModelImporter.inspect(src)
        #expect(info.architecture == .qwenImage21 && info.schema == .diffusers && info.droppedComponents == ["vae"])

        let dest = models.appendingPathComponent("qwen-ft")
        let r = try ModelImporter.importTransformer(from: src, family: "qwen-image-2.1", base: base, into: dest)
        #expect(r.bits == 8)
        try check(dest, reference: w)
    }

    @Test func diffusersFolderFP8() throws {
        let w = TinyQwen.weights(seed: 3)
        let folder = tmp.sub("qwen21-fp8/transformer")
        let fp8 = Set(w.keys.filter { $0.hasSuffix(".weight") && w[$0]!.ndim == 2 })
        try writeSafetensors(w, to: folder.appendingPathComponent("diffusion_pytorch_model.safetensors"), fp8: fp8)
        let dest = models.appendingPathComponent("qwen-fp8")
        let r = try ModelImporter.importTransformer(from: folder.deletingLastPathComponent(), base: base, into: dest)
        #expect(r.sourceDtype == "F8_E4M3")
        try check(dest, reference: Dictionary(uniqueKeysWithValues: w.map {
            ($0.key, fp8.contains($0.key) ? fp8RoundTrip($0.value) : $0.value)
        }))
    }

    /// A Qwen-Image (v1) checkpoint is told apart from 2.1.
    @Test func qwenImageV1IsRefused() throws {
        let src = tmp.url.appendingPathComponent("qwen-image.safetensors")
        try writeSafetensors(["transformer_blocks.0.img_mod.1.weight": MLXArray.zeros([4, 4]),
                              "transformer_blocks.0.txt_mlp.net.0.proj.weight": MLXArray.zeros([4, 4])], to: src)
        let e = errorText { try ModelImporter.importTransformer(from: src, base: base, into: models.appendingPathComponent("v1")) }
        #expect(e?.contains("Qwen-Image 2.1") == true)
    }
}
