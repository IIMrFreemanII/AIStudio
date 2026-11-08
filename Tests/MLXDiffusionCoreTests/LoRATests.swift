import Foundation
import MLX
import Testing
@testable import MLXDiffusionCore

@Test func adapterMatchesMergedWeight() {
    let w = MLXRandom.normal([6, 4], key: MLXRandom.key(1))
    let a = MLXRandom.normal([2, 4], key: MLXRandom.key(2))
    let b = MLXRandom.normal([6, 2], key: MLXRandom.key(3))
    let x = MLXRandom.normal([3, 4], key: MLXRandom.key(4))
    let lin = Lin(weight: w).adding(LoRAAdapter(down: a, up: b, scale: 0.5))
    let merged = Lin(weight: w + 0.5 * matmul(b, a))
    #expect(abs(lin(x) - merged(x)).max().item(Float.self) < 1e-4)
}

@Test func zeroUpLeavesOutputUnchanged() {
    let w = MLXRandom.normal([6, 4], key: MLXRandom.key(1))
    let x = MLXRandom.normal([3, 4], key: MLXRandom.key(4))
    let lin = Lin(weight: w).adding(LoRAAdapter(down: MLXRandom.normal([2, 4]), up: MLXArray.zeros([6, 2]), scale: 1))
    #expect(abs(lin(x) - Lin(weight: w)(x)).max().item(Float.self) == 0)
}

@Test func parsesCommonKeyFormats() throws {
    let modules: Set<String> = ["transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_out.0"]
    let dir = FileManager.default.temporaryDirectory
    func write(_ arrays: [String: MLXArray]) throws -> URL {
        let u = dir.appendingPathComponent("lora-\(UInt32.random(in: 0...UInt32.max)).safetensors")
        try save(arrays: arrays, url: u)
        return u
    }
    let down = MLXArray.ones([4, 8]), up = MLXArray.ones([8, 4])

    // diffusers / PEFT
    let peft = try LoRAFile(url: try write([
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": down,
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": up,
    ]), moduleNames: modules)
    #expect(peft.pairs.keys.sorted() == ["transformer_blocks.0.attn.to_q"])
    #expect(peft.adapters(strength: 1)["transformer_blocks.0.attn.to_q"]!.scale == 1)

    // kohya / Comfy with alpha (scale = strength * alpha / rank)
    let kohya = try LoRAFile(url: try write([
        "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_down.weight": down,
        "diffusion_model.transformer_blocks.0.attn.to_out.0.lora_up.weight": up,
        "diffusion_model.transformer_blocks.0.attn.to_out.0.alpha": MLXArray(Float(2)),
    ]), moduleNames: modules)
    #expect(kohya.adapters(strength: 0.5)["transformer_blocks.0.attn.to_out.0"]!.scale == 0.25)

    // lora_unet_ underscore naming, plus an unrelated key that should be ignored
    let unet = try LoRAFile(url: try write([
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": down,
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": up,
        "text_encoder.something.lora_down.weight": down,
    ]), moduleNames: modules)
    #expect(unet.pairs.keys.sorted() == ["transformer_blocks.0.attn.to_q"])
    #expect(unet.unmatchedKeys == ["text_encoder.something.lora_down.weight"])

    // nothing matches → error
    #expect(throws: LoRAError.self) {
        try LoRAFile(url: try write(["foo.lora_A.weight": down, "foo.lora_B.weight": up]), moduleNames: modules)
    }
}
