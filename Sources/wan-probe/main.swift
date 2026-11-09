import Foundation
import MLX
import MLXDiffusionCore
import T5Text
import Wan
import WanTransformer
import WanVAE

// Development probe for the Wan port. Compares each module against tensors dumped by the reference (mlx-video /
// official Wan2.2) for the same inputs and weights.
//
//   wan-probe tokens <model-dir> <prompt>          token ids of the cleaned prompt
//   wan-probe text <work-dir> <model-dir>          umT5: ids → hidden states        (text_ref.safetensors)
//   wan-probe dit <work-dir> <model-dir> [high|low] one expert forward              (dit_ref.safetensors)
//   wan-probe vae <work-dir> <model-dir>           VAE decode and encode            (vae_ref.safetensors)
//   wan-probe sched                                print the Lightning and official schedules

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    let a = a.asType(.float32), b = b.asType(.float32)
    guard a.shape == b.shape else { print("\(name): shape \(a.shape) vs reference \(b.shape)"); return }
    let d = abs(a - b)
    let rel = sqrt(((a - b) * (a - b)).sum()) / maximum(sqrt((b * b).sum()), MLXArray(Float(1e-12)))
    print(String(format: "%@: max abs %.3g, rel L2 %.3g, ref max %.3g", name, d.max().item(Float.self),
                 rel.item(Float.self), abs(b).max().item(Float.self)))
}

func find(_ name: String, _ dir: URL) -> URL {
    for sub in ["", "common", "q4", "q8"] {
        let u = sub.isEmpty ? dir.appendingPathComponent(name) : dir.appendingPathComponent(sub).appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: u.path) { return u }
    }
    fatalError("\(name) not found in \(dir.path)")
}

let args = CommandLine.arguments
guard args.count >= 2 else { print("usage: wan-probe tokens|text|dit|vae|sched …"); exit(2) }

switch args[1] {
case "tokens":
    let dir = URL(fileURLWithPath: args[2])
    let enc = try WanTextEncoder(weights: find("t5_encoder.safetensors", dir), tokenizerFile: find("tokenizer.json", dir))
    print(WanPrompt.clean(args[3]))
    print(enc.tokens(args[3]))

case "text":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let ref = try loadArrays(url: work.appendingPathComponent("text_ref.safetensors"))
    var tc = T5XXLEncoder.Config()
    if let n = ProcessInfo.processInfo.environment["T5_LAYERS"].flatMap(Int.init) { tc.numLayers = n }
    let enc = try T5XXLEncoder(store: WeightStore(url: find("t5_encoder.safetensors", dir)), config: tc)
    let t0 = Date()
    let out = enc(ref["ids"]!.asType(.int32))
    eval(out)
    print(String(format: "umT5: %.2f s", Date().timeIntervalSince(t0)))
    report("umT5 hidden", out[0], ref["hidden"]!)

case "dit":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let which = args.count > 4 ? args[4] : "low"
    let ref = try loadArrays(url: work.appendingPathComponent("dit_ref.safetensors"))
    let configURL = find("config.json", dir)
    let cfg = try WanDiT.Config(url: configURL)
    let o = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
    let store = try WeightStore(url: configURL.deletingLastPathComponent()
        .appendingPathComponent("\(which)_noise_model.safetensors"))
    if let bits = (o["quantization"] as? [String: Any])?["bits"] as? Int { store.bitsOverride = bits }
    let dit = try WanDiT(store: store, config: cfg)
    let x = ref["x"]!, t = ref["t"]!.item(Float.self), ctx = ref["context"]!
    let text = dit.embedText(ctx)
    report("text embedding", text, ref["text_emb"]!)
    let kv = dit.crossKV(text)
    let rope = dit.rope(frames: x.dim(1), height: x.dim(2) / 2, width: x.dim(3) / 2)
    let input = ref["y"].map { concatenated([x, $0], axis: 0) } ?? x
    let t0 = Date()
    let v = dit(input, timestep: t, crossKV: kv, rope: rope)
    eval(v)
    print(String(format: "forward: %.2f s", Date().timeIntervalSince(t0)))
    report("velocity (\(which))", v, ref["v"]!)

case "vae":
    let work = URL(fileURLWithPath: args[2]), dir = URL(fileURLWithPath: args[3])
    let ref = try loadArrays(url: work.appendingPathComponent("vae_ref.safetensors"))
    let vae = try WanVAE(url: find("vae.safetensors", dir), encoder: true)
    if let z = ref["z"], let video = ref["video"] {  // z [16, T, h, w], video [3, F, H, W]
        let t0 = Date()
        let out = vae.decode(z.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))
        eval(out)
        print(String(format: "decode: %.2f s", Date().timeIntervalSince(t0)))
        report("decoded video", out[0], video.transposed(1, 2, 3, 0))
    }
    if let input = ref["enc_in"], let mu = ref["enc_out"] {  // [3, F, H, W] → [16, T, h, w]
        let t0 = Date()
        let out = vae.encode(input.transposed(1, 2, 3, 0).expandedDimensions(axis: 0))
        eval(out)
        print(String(format: "encode: %.2f s", Date().timeIntervalSince(t0)))
        report("encoded latents", out[0], mu.transposed(1, 2, 3, 0))
    }

case "unipc":  // x' = v(x, i) = 0.3·x − 0.1·i·x0 with x0 = [1, −2, 0.5]; prints the final x
    let s = WanSchedule.official(steps: 10, shift: 5)
    var x = MLXArray([0.7, -1.3, 2.1] as [Float])
    let x0 = MLXArray([1, -2, 0.5] as [Float])
    var solver = WanUniPC(sigmas: s.sigmas)
    for i in 0..<10 { x = solver.step(velocity: 0.3 * x - Float(0.1 * Double(i)) * x0, sample: x) }
    print("timesteps", s.timesteps)
    print("final", x.asArray(Float.self))

case "vae-bench":  // decode random latents at a real size and report time and peak memory
    let dir = URL(fileURLWithPath: args[2])
    let vae = try WanVAE(url: find("vae.safetensors", dir), encoder: false)
    let (T, h, w) = (Int(args[3]) ?? 21, Int(args[4]) ?? 104, Int(args[5]) ?? 60)
    let z = MLXRandom.normal([1, T, h, w, 16], key: MLXRandom.key(0))
    eval(z)
    Memory.peakMemory = 0
    let t0 = Date()
    let out = vae.decode(z)
    eval(out)
    print(String(format: "decode %d latent frames at %dx%d: %.1f s, peak %.1f GB", T, w * 8, h * 8,
                 Date().timeIntervalSince(t0), Double(Memory.peakMemory) / 1e9))

case "dit-bench":  // one expert forward at a real size (I2V 480x832, 81 frames): load, warm up, time
    let dir = URL(fileURLWithPath: args[2])
    let configURL = find("config.json", dir)
    let cfg = try WanDiT.Config(url: configURL)
    let o = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] ?? [:]
    let store = try WeightStore(url: configURL.deletingLastPathComponent().appendingPathComponent("high_noise_model.safetensors"))
    if let bits = (o["quantization"] as? [String: Any])?["bits"] as? Int { store.bitsOverride = bits }
    let dit = try WanDiT(store: store, config: cfg)
    let (T, h, w) = (Int(args.count > 3 ? args[3] : "21")!, 104, 60)
    let x = MLXRandom.normal([cfg.inDim, T, h, w], key: MLXRandom.key(1))
    let kv = dit.crossKV(dit.embedText(MLXRandom.normal([20, 4096], key: MLXRandom.key(2)) * 0.2))
    let rope = dit.rope(frames: T, height: h / 2, width: w / 2)
    eval(x, rope.0, rope.1)
    for i in 0..<2 {
        let t0 = Date()
        let v = dit(x, timestep: 900, crossKV: kv, rope: rope)
        eval(v)
        print(String(format: "forward %d (%d tokens): %.1f s, peak %.1f GB", i, T * h * w / 4, Date().timeIntervalSince(t0),
                     Double(Memory.peakMemory) / 1e9))
    }

case "sched":
    print("lightning:", WanSchedule.lightning().timesteps, WanSchedule.lightning().sigmas)
    let o = WanSchedule.official(steps: 40, shift: 12)
    print("official T2V (40, shift 12): first", o.timesteps.prefix(6), "last", o.timesteps.suffix(4))

default:
    print("unknown command \(args[1])")
    exit(2)
}
