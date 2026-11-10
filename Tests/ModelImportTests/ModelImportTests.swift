import Foundation
import MLX
@testable import ModelImport
import Testing

/// Files the importer refuses, each with a message the user can act on.
@Suite(.serialized) struct RefusalTests {
    let tmp = TempDir("import-refusals")

    func inspectError(_ name: String, _ write: (URL) throws -> Void) throws -> String? {
        let u = tmp.url.appendingPathComponent(name)
        try write(u)
        return errorText { try ModelImporter.inspect(u) }
    }

    @Test func gguf() throws {
        let e = try inspectError("flux1-dev-Q8_0.gguf") { try Data("GGUF\u{3}\0\0\0junk".utf8).write(to: $0) }
        #expect(e?.contains("GGUF") == true)
        let e2 = try inspectError("renamed.safetensors") { try Data("GGUF\u{3}\0\0\0junk".utf8).write(to: $0) }
        #expect(e2?.contains("GGUF") == true)
    }

    @Test func pickleIsNotSafetensors() throws {
        let e = try inspectError("model.safetensors") { try Data([0x80, 0x02, 0x7d, 0x71]).write(to: $0) }
        #expect(e?.contains("not a .safetensors") == true)
    }

    @Test func nf4() throws {
        let e = try inspectError("flux-nf4.safetensors") {
            try writeSafetensors([
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": MLXArray.zeros([96, 1], dtype: .uint8),
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight.absmax": MLXArray.zeros([3]),
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight.quant_map": MLXArray.zeros([16]),
            ], to: $0)
        }
        #expect(e?.contains("NF4") == true)
    }

    @Test func e5m2() throws {
        let e = try inspectError("flux-e5m2.safetensors") {
            try writeSafetensors(["double_blocks.0.img_attn.qkv.weight": MLXArray.zeros([6, 2], dtype: .uint8)],
                                 to: $0, tags: ["double_blocks.0.img_attn.qkv.weight": "F8_E5M2"])
        }
        #expect(e?.contains("F8_E5M2") == true)
    }

    @Test func unknownSchema() throws {
        let e = try inspectError("sdxl.safetensors") {
            try writeSafetensors(["model.diffusion_model.input_blocks.0.0.weight": MLXArray.zeros([4, 4]),
                                  "conditioner.embedders.0.weight": MLXArray.zeros([4])], to: $0)
        }
        #expect(e?.contains("no diffusion transformer recognised") == true)
    }

    /// A Wan checkpoint pointed at a FLUX base.
    @Test func wrongFamily() throws {
        let models = tmp.sub("models")
        let base = models.appendingPathComponent("flux-1-dev")
        try TinyFlux.makeBase(base, weights: TinyFlux.weights(seed: 1))
        let src = tmp.url.appendingPathComponent("wan.safetensors")
        try writeSafetensors(TinyWan.official(TinyWan.weights(seed: 1)), to: src)
        let e = errorText {
            try ModelImporter.importTransformer(from: src, family: "flux.1", base: base,
                                                into: models.appendingPathComponent("x"))
        }
        #expect(e?.contains("wan-2.2 model, not flux.1") == true)
        let e2 = errorText {
            try ModelImporter.importTransformer(from: src, family: "wan-2.2", base: base,
                                                into: models.appendingPathComponent("x"))
        }
        #expect(e2?.contains("is a flux.1 model, not wan-2.2") == true)
        #expect(partials(in: models).isEmpty)
    }

    @Test func expertNamesAreGuessed() {
        #expect(WanExpert.guess("Wan2.2-T2V-HighNoise-v1") == .high)
        #expect(WanExpert.guess("wan22_low_noise_fp8") == .low)
        #expect(WanExpert.guess("wan-finetune") == nil)
        #expect(WanExpert.guess("high-and-low") == nil)
    }
}
