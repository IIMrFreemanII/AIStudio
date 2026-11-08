import Foundation
import LTX2Audio
import MLX
import MLXDiffusionCore

// Usage: ltx-probe-audio <work-dir-with-dumps> [model-dir]
let args = CommandLine.arguments
let work = URL(fileURLWithPath: args.count > 1 ? args[1] : FileManager.default.currentDirectoryPath)
let modelDir = args.count > 2 ? URL(fileURLWithPath: args[2]) :
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/MLXComfyUI/models/ltx-2.3-mlx-q4")

func report(_ name: String, _ ours: MLXArray, _ ref: MLXArray?) {
    guard let ref else { print("  \(name): no reference"); return }
    guard ours.shape == ref.shape else { print("  \(name): SHAPE MISMATCH ours \(ours.shape) ref \(ref.shape)"); return }
    let a = ours.asType(.float32), b = ref.asType(.float32)
    let d = abs(a - b)
    let maxAbs = d.max().item(Float.self)
    let refMax = abs(b).max().item(Float.self)
    let rel = (sqrt((d * d).sum()) / maximum(sqrt((b * b).sum()), Float(1e-12))).item(Float.self)
    print(String(format: "  %-14@ shape %@  maxAbs %.3e  (ref max %.3e)  relL2 %.3e", name as NSString,
                 "\(ours.shape)" as NSString, maxAbs, refMax, rel))
}

let t0 = Date()
let dec = try LTXAudioDecoder(directory: modelDir)
print(String(format: "loaded in %.2f s", Date().timeIntervalSince(t0)))

for tag in ["rand5s", "zeros5s", "rand_short"] {
    let url = work.appendingPathComponent("\(tag).safetensors")
    guard FileManager.default.fileExists(atPath: url.path) else { continue }
    let ref = try loadArrays(url: url)
    print("== \(tag)")
    let tokens = ref["tokens"]!
    let latent = LTXAudioPatchifier.unpatchify(tokens)
    var taps: [String: MLXArray] = [:]
    let mel = dec.vae.decode(latent: latent) { taps[$0] = $1 }
    eval(mel)
    for k in ["dec_conv_in", "dec_mid", "dec_up2", "dec_up1", "dec_up0"] where ref[k] != nil {
        report(k, taps[k]!, ref[k])
    }
    report("mel", mel, ref["mel"])
    // vocoder on the reference mel (isolates vocoder error) ...
    var vt: [String: MLXArray] = [:]
    let w48ref = dec.vocoder(ref["mel"]!) { vt[$0] = $1 }
    eval(w48ref)
    for k in ["wave16k", "bwe_mel", "bwe_residual", "skip48k"] where vt[k] != nil {
        report(k + "*", vt[k]!, ref[k])
    }
    report("wave48k*", w48ref.asType(.bfloat16), ref["wave48k"])
    // ... and the full chain from tokens
    let (samples, sr) = dec.decode(tokens: tokens)
    report("full wave48k", samples.expandedDimensions(axis: 0), ref["wave48k"])
    let rms = sqrt((samples * samples).mean()).item(Float.self)
    print(String(format: "  output [%d, %d] @ %d Hz, rms %.4f, peak %.4f", samples.dim(0), samples.dim(1), sr, rms,
                 abs(samples).max().item(Float.self)))
    taps = [:]; vt = [:]
    MLX.GPU.clearCache()
}
print("(* = vocoder fed the reference mel)")

// timing
for t in [126, 251] {
    let tokens = MLXRandom.normal([1, t, 128], key: MLXRandom.key(7)).asType(.bfloat16)
    var times: [Double] = []
    for _ in 0 ..< 3 {
        let s = Date()
        let (w, _) = dec.decode(tokens: tokens)
        eval(w)
        times.append(Date().timeIntervalSince(s))
    }
    print(String(format: "timing T=%d (%.2f s audio): %@ s  peak %.2f GB", t,
                 Double(dec.sampleCount(forTokens: t)) / Double(dec.sampleRate),
                 times.map { String(format: "%.3f", $0) }.joined(separator: ", "),
                 Double(MLX.GPU.peakMemory) / 1e9))
    MLX.GPU.clearCache()
}

// memory profile of one 5 s decode, stage by stage
do {
    func mem(_ s: String) {
        print(String(format: "  %-22@ active %.2f GB  cache %.2f GB  peak %.2f GB", s as NSString,
                     Double(Memory.activeMemory) / 1e9, Double(Memory.cacheMemory) / 1e9, Double(Memory.peakMemory) / 1e9))
    }
    MLX.GPU.clearCache()
    Memory.peakMemory = 0
    let tokens = MLXRandom.normal([1, 126, 128], key: MLXRandom.key(7)).asType(.bfloat16)
    eval(tokens)
    mem("start")
    let mel = dec.mel(latent: LTXAudioPatchifier.unpatchify(tokens)); eval(mel); mem("after VAE")
    let w16 = dec.vocoder.baseWaveform(mel: mel); eval(w16); mem("after base vocoder")
    let w = dec.vocoder(mel); eval(w); mem("after full vocoder")
    // BWE generator fed the reference BWE mel (isolates BWE-generator error from STFT error)
    if let ref = try? loadArrays(url: work.appendingPathComponent("rand5s.safetensors")), let bm = ref["bwe_mel"] {
        let x = bm.reshaped(1, 2, bm.dim(1), 64).transposed(0, 2, 1, 3).reshaped(1, bm.dim(1), 128)
        report("bwe_residual**", dec.vocoder.bwe(x).transposed(0, 2, 1), ref["bwe_residual"])
        print("(** = BWE generator fed the reference bwe_mel)")
    }
}
