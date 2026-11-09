import CLIPText
import Flux
import FluxTransformer
import FluxVAE
import Foundation
import MLX
import MLXDiffusionCore
import T5Text
import Tokenizers

// Development probe for the Flux port. Compares each module against tensors dumped by the reference (diffusers) for
// the same inputs and weights.
//
//   flux-probe tokens <model-dir> <prompt>        CLIP and T5 token ids
//   flux-probe text <work-dir> <model-dir>        CLIP pooled output and T5 hidden states   (text_ref.safetensors)
//   flux-probe dit <work-dir> <model-dir> [bf16]  transformer forward (dit_ref.safetensors); FLUX_LAYERS /
//                                                 FLUX_SINGLE_LAYERS truncate it, FLUX_FP32=1 computes in float32
//   flux-probe vae <work-dir> <model-dir>         VAE decode and encode                      (vae_ref.safetensors)
//   flux-probe sched [steps] [tokens]             the dynamic-shift sigmas

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    let a = a.asType(.float32), b = b.asType(.float32)
    guard a.shape == b.shape else { print("\(name): shape \(a.shape) vs reference \(b.shape)"); return }
    let d = abs(a - b)
    let rel = sqrt(((a - b) * (a - b)).sum()) / maximum(sqrt((b * b).sum()), MLXArray(Float(1e-12)))
    print(String(format: "%@: max abs %.3g, rel L2 %.3g, ref max %.3g", name, d.max().item(Float.self),
                 rel.item(Float.self), abs(b).max().item(Float.self)))
}

let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: flux-probe tokens|text|dit|vae|sched …"); exit(2) }

switch args[1] {
case "tokens":
    let dir = URL(fileURLWithPath: args[2])
    let clip = try CLIPTokenizer(directory: dir.appendingPathComponent("tokenizer"))
    print("clip", clip.encode(args[3]))
    let t5tok = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer_2"))
    print("t5", FluxPipeline.t5Tokens(t5tok, args[3]).filter { $0 != 0 })

case "text":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let ref = try loadArrays(url: work.appendingPathComponent("text_ref.safetensors"))
    let tok = try CLIPTokenizer(directory: dir.appendingPathComponent("tokenizer"))
    let prompt = try String(contentsOf: work.appendingPathComponent("prompt.txt"), encoding: .utf8)
    let ids = tok.encode(prompt)
    let refIDs = ref["clip_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init)
    print("clip ids match:", Array(refIDs.prefix(ids.count)) == ids, ids.count)
    let t5tok = try await AutoTokenizer.from(modelFolder: dir.appendingPathComponent("tokenizer_2"))
    let t5ids = FluxPipeline.t5Tokens(t5tok, prompt)
    print("t5 ids match:", ref["t5_ids"]!.asType(.int32).asArray(Int32.self).map(Int.init) == t5ids,
          t5ids.prefix(24).map { $0 })
    let clipDir = dir.appendingPathComponent("text_encoder")
    let clip = try CLIPTextEncoder(store: WeightStore(directory: clipDir),
                                   config: CLIPTextEncoder.Config(url: clipDir.appendingPathComponent("config.json")))
    report("clip pooled", clip.pooled(ids, eos: tok.eos), ref["pooled"]!)
    var cfg = T5XXLEncoder.Config()
    cfg.vocabSize = 32_128
    cfg.sharedPositionBias = true
    let t5 = try T5XXLEncoder(store: WeightStore(directory: dir.appendingPathComponent("text_encoder_2")), config: cfg,
                              layout: .huggingFace)
    let t0 = Date()
    let h = t5(ref["t5_ids"]!.asType(.int32))
    eval(h)
    print(String(format: "T5: %.2f s", Date().timeIntervalSince(t0)))
    report("t5 hidden", h, ref["t5_hidden"]!)

case "dit":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let ref = try loadArrays(url: work.appendingPathComponent("dit_ref.safetensors"))
    let tDir = dir.appendingPathComponent("transformer")
    let store = args.count > 4 && args[4] == "bf16" ? try WeightStore(directory: tDir)
        : try WeightStore.quantizedCache(source: tDir, cacheURL: FluxPipeline.transformerCacheURL(dir), log: { print($0) })
    var cfg = try FluxTransformer.Config(url: tDir.appendingPathComponent("config.json"))
    let env = ProcessInfo.processInfo.environment
    if let n = env["FLUX_LAYERS"].flatMap(Int.init) { cfg.numLayers = n }
    if let n = env["FLUX_SINGLE_LAYERS"].flatMap(Int.init) { cfg.numSingleLayers = n }
    let dit = try FluxTransformer(store: store, config: cfg, activations: env["FLUX_FP32"] != nil ? .float32 : .bfloat16)
    let x = ref["latents"]!, ctx = ref["context"]!, pooled = ref["pooled"]!
    let sigma = ref["sigma"]!.item(Float.self), guidance = ref["guidance"]!.item(Float.self)
    let h = ref["h"]!.item(Int32.self), w = ref["w"]!.item(Int32.self)
    let vec = dit.conditioningVector(sigma: sigma, guidance: guidance, pooled: pooled)
    if let rv = ref["vec"] { report("conditioning vector", vec, rv) }
    let rope = dit.rope(textLength: ctx.dim(1), height: Int(h), width: Int(w))
    let t0 = Date()
    let v = dit(x, context: ctx, vec: vec, rope: rope)
    eval(v)
    print(String(format: "forward: %.2f s", Date().timeIntervalSince(t0)))
    report("velocity", v, ref["v"]!)

case "vae":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let ref = try loadArrays(url: work.appendingPathComponent("vae_ref.safetensors"))
    let vae = try FluxVAE(directory: dir.appendingPathComponent("vae"))
    if let z = ref["z"], let img = ref["image"] {  // z [1, 16, h, w] raw, image [1, 3, H, W]
        let t0 = Date()
        let out = vae.decode(z.transposed(0, 2, 3, 1))
        eval(out)
        print(String(format: "decode: %.2f s, peak %.1f GB", Date().timeIntervalSince(t0), Double(Memory.peakMemory) / 1e9))
        report("decoded", out, img.transposed(0, 2, 3, 1))
    }
    if let input = ref["enc_in"], let mean = ref["enc_mean"] {
        let out = vae.encode(input.transposed(0, 2, 3, 1))
        eval(out)
        report("encoded mean", out, mean.transposed(0, 2, 3, 1))
    }

case "sched":
    let steps = args.count > 2 ? Int(args[2])! : 28
    let tokens = args.count > 3 ? Int(args[3])! : 4096
    var s = FlowMatchEuler()
    s.useDynamicShifting = true
    s.setSteps(steps, imageTokens: tokens)
    print("mu", s.mu(imageTokens: tokens))
    print(s.sigmas)

default:
    print("unknown command \(args[1])")
    exit(2)
}
