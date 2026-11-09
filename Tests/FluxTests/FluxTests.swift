import CLIPText
import Flux
@testable import FluxTransformer
import Foundation
import MLX
import MLXDiffusionCore
import Testing

/// The resolution-dependent schedule matches diffusers' Flux pipeline (values from `calculate_shift` + `time_shift`).
@Test func dynamicShiftScheduleMatchesTheReference() {
    var s = FlowMatchEuler()
    s.useDynamicShifting = true
    s.setSteps(28, imageTokens: 4096)  // 1024²
    #expect(abs(s.mu(imageTokens: 4096) - 1.15) < 1e-12)
    for (i, v) in [(0, 1.0), (1, 0.988408625125885), (13, 0.7846716046333313), (27, 0.10472089052200317)] as [(Int, Float)] {
        #expect(abs(s.sigmas[i] - v) < 1e-6)
    }
    s.setSteps(28, imageTokens: 1024)  // 512²
    for (i, v) in [(1, 0.9806559681892395), (13, 0.6841912865638733), (27, 0.06501959264278412)] as [(Int, Float)] {
        #expect(abs(s.sigmas[i] - v) < 1e-6)
    }
    #expect(s.sigmas.count == 29 && s.sigmas.last == 0)
}

/// Packing follows diffusers' `_pack_latents`: token (r, c) holds channels `c·4 + dy·2 + dx` of the 2×2 patch.
@Test func packingMatchesDiffusersLayout() {
    let z = MLXRandom.normal([1, 6, 8, 16], key: MLXRandom.key(3))  // NHWC
    let p = FluxPipeline.pack(z)
    #expect(p.shape == [1, 12, 64])
    // token (r=1, c=2) = index 1·4 + 2; channel 5·4 + 1·2 + 0 = z[0, 2+1, 4+0, 5]
    #expect(p[0, 6, 22].item(Float.self) == z[0, 3, 4, 5].item(Float.self))
    #expect(FluxPipeline.unpack(p, h: 3, w: 4).asArray(Float.self) == z.asArray(Float.self))
}

/// A tiny Flux transformer with random weights (2 heads × 8, one block of each kind).
func tinyTransformer() throws -> FluxTransformer {
    var c = FluxTransformer.Config()
    c.numLayers = 1; c.numSingleLayers = 1; c.headDim = 8; c.numHeads = 2; c.jointDim = 16; c.pooledDim = 12
    c.axesDims = [2, 2, 4]
    let d = c.dim, mlp = 4 * d
    var shapes: [String: [Int]] = [
        "x_embedder": [d, 64], "context_embedder": [d, 16], "norm_out.linear": [2 * d, d], "proj_out": [64, d],
        "time_text_embed.timestep_embedder.linear_1": [d, 256], "time_text_embed.timestep_embedder.linear_2": [d, d],
        "time_text_embed.guidance_embedder.linear_1": [d, 256], "time_text_embed.guidance_embedder.linear_2": [d, d],
        "time_text_embed.text_embedder.linear_1": [d, 12], "time_text_embed.text_embedder.linear_2": [d, d],
    ]
    let b = "transformer_blocks.0."
    for n in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
              "attn.add_v_proj", "attn.to_add_out"] { shapes[b + n] = [d, d] }
    shapes[b + "norm1.linear"] = [6 * d, d]; shapes[b + "norm1_context.linear"] = [6 * d, d]
    shapes[b + "ff.net.0.proj"] = [mlp, d]; shapes[b + "ff.net.2"] = [d, mlp]
    shapes[b + "ff_context.net.0.proj"] = [mlp, d]; shapes[b + "ff_context.net.2"] = [d, mlp]
    let s = "single_transformer_blocks.0."
    for n in ["attn.to_q", "attn.to_k", "attn.to_v"] { shapes[s + n] = [d, d] }
    shapes[s + "norm.linear"] = [3 * d, d]; shapes[s + "proj_mlp"] = [mlp, d]; shapes[s + "proj_out"] = [d, d + mlp]
    var arrays: [String: MLXArray] = [:]
    var key = MLXRandom.key(7)
    for (name, shape) in shapes.sorted(by: { $0.key < $1.key }) {
        let k = MLXRandom.split(key: key, into: 3)
        key = k[0]
        arrays[name + ".weight"] = MLXRandom.normal(shape, key: k[1]) / Float(shape[1]).squareRoot()
        arrays[name + ".bias"] = MLXRandom.normal([shape[0]], key: k[2]) * 0.1
    }
    for n in [b + "attn.norm_q", b + "attn.norm_k", b + "attn.norm_added_q", b + "attn.norm_added_k",
              s + "attn.norm_q", s + "attn.norm_k"] { arrays[n + ".weight"] = MLXArray.ones([8]) }
    return try FluxTransformer(store: WeightStore(arrays: arrays), config: c)
}

func tinyForward(_ t: FluxTransformer) -> MLXArray {
    let x = MLXRandom.normal([1, 6, 64], key: MLXRandom.key(1))
    let ctx = MLXRandom.normal([1, 5, 16], key: MLXRandom.key(2))
    let vec = t.conditioningVector(sigma: 0.7, guidance: 3.5, pooled: MLXRandom.normal([1, 12], key: MLXRandom.key(4)))
    let out = t(x, context: ctx, vec: vec, rope: t.rope(textLength: 5, height: 2, width: 3))
    eval(out)
    return out
}

/// A kohya/BFL LoRA with fused qkv (and fused qkv+MLP in single blocks) gives exactly the result of the same update
/// written per layer in diffusers/PEFT naming; the final modulation's (shift, scale) halves are swapped onto
/// diffusers' (scale, shift); text-encoder keys are reported as ignored.
@Test func fusedKohyaLoRAEqualsTheSplitPEFTLoRA() throws {
    let t = try tinyTransformer()
    let d = 16, mlp = 64, r = 4
    var key = MLXRandom.key(11)
    func rand(_ shape: [Int]) -> MLXArray {
        let k = MLXRandom.split(key: key, into: 2)
        key = k[0]
        return MLXRandom.normal(shape, key: k[1]) * 0.5
    }
    let qkvDown = rand([r, d]), qkvUp = rand([3 * d, r])
    let txtDown = rand([r, d]), txtUp = rand([3 * d, r])
    let l1Down = rand([r, d]), l1Up = rand([3 * d + mlp, r])
    let finDown = rand([r, d]), finUp = rand([2 * d, r])
    let kohya: [String: MLXArray] = [
        "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": qkvDown,
        "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight": qkvUp,
        "lora_unet_double_blocks_0_img_attn_qkv.alpha": MLXArray(Float(2)),
        "diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight": txtDown,
        "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight": txtUp,
        "lora_unet_single_blocks_0_linear1.lora_down.weight": l1Down,
        "lora_unet_single_blocks_0_linear1.lora_up.weight": l1Up,
        "lora_unet_final_layer_adaLN_modulation_1.lora_down.weight": finDown,
        "lora_unet_final_layer_adaLN_modulation_1.lora_up.weight": finUp,
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": rand([r, 8]),
        "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight": rand([8, r]),
    ]
    let b = "transformer.transformer_blocks.0.attn.", s = "transformer.single_transformer_blocks.0."
    var peft: [String: MLXArray] = [:]
    func put(_ name: String, _ down: MLXArray, _ up: MLXArray) {
        peft[name + ".lora_A.weight"] = down
        peft[name + ".lora_B.weight"] = up
    }
    // alpha 2 over rank 4 → scale 0.5 on the img qkv; folded into the up matrices of the PEFT copy
    put(b + "to_q", qkvDown, qkvUp[0 ..< d] * 0.5); put(b + "to_k", qkvDown, qkvUp[d ..< 2 * d] * 0.5)
    put(b + "to_v", qkvDown, qkvUp[(2 * d)...] * 0.5)
    put(b + "add_q_proj", txtDown, txtUp[0 ..< d]); put(b + "add_k_proj", txtDown, txtUp[d ..< 2 * d])
    put(b + "add_v_proj", txtDown, txtUp[(2 * d)...])
    put(s + "attn.to_q", l1Down, l1Up[0 ..< d]); put(s + "attn.to_k", l1Down, l1Up[d ..< 2 * d])
    put(s + "attn.to_v", l1Down, l1Up[(2 * d) ..< 3 * d]); put(s + "proj_mlp", l1Down, l1Up[(3 * d)...])
    put("transformer.norm_out.linear", finDown, concatenated([finUp[d...], finUp[..<d]], axis: 0))

    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("flux-lora-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let kURL = dir.appendingPathComponent("kohya.safetensors"), pURL = dir.appendingPathComponent("peft.safetensors")
    try save(arrays: kohya, url: kURL)
    try save(arrays: peft, url: pURL)

    let kFile = try t.loraFile(kURL), pFile = try t.loraFile(pURL)
    #expect(kFile.pairs.count == 11 && pFile.pairs.count == 11)
    #expect(kFile.unmatchedKeys.count == 2 && pFile.unmatchedKeys.isEmpty)
    let (kt, km) = t.adding(kFile.adapters(strength: 1, dtype: .float32))
    let (pt, pm) = t.adding(pFile.adapters(strength: 1, dtype: .float32))
    #expect(km == 11 && pm == 11)
    let base = tinyForward(t), a = tinyForward(kt), p = tinyForward(pt)
    #expect(abs(a - base).max().item(Float.self) > 1e-2)  // the LoRA changes the output…
    #expect(abs(a - p).max().item(Float.self) < 1e-3)  // …exactly as the split one does
}

/// The CLIP byte table is GPT-2's: 256 distinct printable characters, identity on printable ASCII.
@Test func clipByteTableIsReversible() {
    let t = CLIPTokenizer.bytesToUnicode()
    #expect(Set(t).count == 256)
    #expect(t[Int(UInt8(ascii: "a"))] == "a" && t[32] == "Ġ")
    #expect(CLIPTokenizer.clean("  A  Red\tFox ") == "a red fox")
    // As BERT's basic tokenizer: CJK ideographs become separate words (kana don't), control characters are dropped.
    #expect(CLIPTokenizer.clean("東京の夜景\u{200B}!") == "東 京 の 夜 景 !")
}
