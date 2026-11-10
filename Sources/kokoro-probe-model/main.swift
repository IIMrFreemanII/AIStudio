import Foundation
import Kokoro
import MLX
import MLXDiffusionCore

// Verification probe for Kokoro against dumps from scratchpad/work-kokoro/model/ref.py.
//   pth <ref.safetensors>                       loader check vs a torch.load flat dump
//   ref <case.safetensors> [f32|bf16]           per-module (isolated) + end-to-end comparison, reference noise
//   wav <case.safetensors> <out.safetensors> [seed|zero|refnoise] [f32|bf16]   synthesize the case's tokens
//   bench <case.safetensors> [f32|bf16] [runs]  timing / real-time factor
let args = CommandLine.arguments
let models = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/AIStudio/models/kokoro-82m")

func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
    let a = a.asType(.float32), b = b.asType(.float32)
    let d = a - b
    return sqrt((d * d).sum()).item(Float.self) / max(sqrt((b * b).sum()).item(Float.self), 1e-30)
}

/// Reference (torch layout) → channels-last layout of `like`.
func align(_ r: MLXArray, like m: MLXArray) -> MLXArray {
    if r.shape == m.shape { return r }
    if r.ndim == 3, r.transposed(0, 2, 1).shape == m.shape { return r.transposed(0, 2, 1) }
    if r.size == m.size { return r.reshaped(m.shape) }
    return r
}

let channelFirst: Set<String> = ["d_en", "en", "t_en", "asr", "dec_x", "har", "gen_x0", "gen_x1"]
func injected(_ ref: [String: MLXArray], _ names: [String]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for n in names {
        guard var r = ref[n] else { continue }
        if channelFirst.contains(n) { r = r.transposed(0, 2, 1) }
        if n == "dur_float" { r = r.reshaped(-1) }
        out[n] = r
    }
    return out
}

func dtypeArg(_ s: String?) -> DType { s == "bf16" ? .bfloat16 : .float32 }

func loadCase(_ path: String) throws -> (ref: [String: MLXArray], tokens: [Int], refS: MLXArray) {
    let ref = try loadArrays(url: URL(fileURLWithPath: path))
    let tokens = ref["ids"]!.asArray(Int32.self).map(Int.init)
    return (ref, tokens, ref["ref_s"]!)
}

func timed<T>(_ f: () throws -> T) rethrows -> (T, Double) {
    let t = Date()
    let r = try f()
    return (r, Date().timeIntervalSince(t))
}

let mode = args.count > 1 ? args[1] : ""
KokoroModel.profile = ProcessInfo.processInfo.environment["KOKORO_PROFILE"] != nil
switch mode {
case "pth":
    let t = Date()
    let w = try TorchPickle.load(url: models.appendingPathComponent("kokoro-v1_0.pth"))
    print("loaded \(w.count) tensors in \(String(format: "%.2f", Date().timeIntervalSince(t))) s")
    let ref = try loadArrays(url: URL(fileURLWithPath: args[2]))
    var worst: Float = 0, missing = 0
    for (k, v) in ref {
        guard let m = w[k] else { missing += 1; if missing < 5 { print("missing", k) }; continue }
        if m.shape != v.shape { print("shape", k, m.shape, v.shape); continue }
        worst = max(worst, abs(m.asType(.float32) - v).max().item(Float.self))
    }
    print("ref \(ref.count) tensors, missing \(missing), max abs diff \(worst)")
    let voice = try TorchPickle.load(url: models.appendingPathComponent("voices/af_heart.pt"))
    print("voice", voice.map { "\($0.key.isEmpty ? "<root>" : $0.key) \($0.value.shape)" })

case "ref":
    let (ref, tokens, refS) = try loadCase(args[2])
    let dt = dtypeArg(args.count > 3 ? args[3] : nil)
    let (model, tl) = try timed { try KokoroModel(directory: models, dtype: dt) }
    print("model loaded in \(String(format: "%.2f", tl)) s, dtype \(dt), \(tokens.count) tokens")
    let noise = KokoroNoise.given(ref["sine_noise"]!)
    var report: [(String, Float)] = []
    var dumped: [String: MLXArray] = [:]
    func tap(_ name: String, _ v: MLXArray) {
        dumped[name] = v
        guard let r = ref[name] else { return }
        let rr = align(r, like: v)
        guard rr.shape == v.shape else { print("  \(name): shape \(v.shape) vs ref \(r.shape)"); return }
        report.append((name, relL2(v, rr)))
    }
    // 1) isolated: every module gets the reference inputs
    var all = ["bert", "d_en", "d", "dur_float", "en", "F0_pred", "N_pred", "t_en", "asr", "dec_x"]
    if ProcessInfo.processInfo.environment["KOKORO_ISO_GEN"] != nil { all += ["har", "gen_x0"] }
    if ProcessInfo.processInfo.environment["KOKORO_ISO_HAR"] != nil { all += ["har_source"] }
    var inj = injected(ref, all)
    inj["pred_dur"] = ref["pred_dur"]!
    let iso = model.forward(tokens: tokens, refS: refS, noise: noise, inject: inj, tap: tap)
    print("isolated modules (rel-L2 vs reference inputs):")
    for (n, e) in report { print("  \(n.padding(toLength: 10, withPad: " ", startingAt: 0)) \(String(format: "%.3e", e))") }
    print(String(format: "  audio      %.3e", relL2(iso.audio, ref["audio"]!)))
    if let p = ProcessInfo.processInfo.environment["KOKORO_DUMP"] {
        dumped["audio"] = iso.audio
        try save(arrays: dumped, url: URL(fileURLWithPath: p))
    }
    // 2) end to end (reference noise only)
    report = []
    let refDur = ref["pred_dur"]!.asArray(Int32.self).map(Int.init)
    // durations first (a different total would not fit the reference noise)
    let dOnly = model.predictDurations(tokens: tokens, refS: refS)
    let diff = zip(dOnly.durations, refDur).filter { $0 != $1 }.count
    let e2eInject: [String: MLXArray] = dOnly.durations == refDur ? [:] : ["pred_dur": ref["pred_dur"]!]
    if !e2eInject.isEmpty { print("  durations differ at \(diff) tokens: end to end continues with the reference durations") }
    let ((audio, durs), te) = timed { model.forward(tokens: tokens, refS: refS, noise: noise, inject: e2eInject, tap: tap) }
    print("end to end (\(String(format: "%.2f", te)) s): durations \(diff == 0 ? "identical" : "DIFFER at \(diff) tokens") (Σ \(dOnly.durations.reduce(0, +)) vs \(refDur.reduce(0, +)))")
    _ = durs
    if let df = ref["dur_float"] {
        let f = df.reshaped(-1).asArray(Float.self)
        let m = f.map { abs($0 - $0.rounded(.down) - 0.5) }.min() ?? 1
        print(String(format: "  closest reference duration to a .5 tie: %.2e", m))
    }
    for (n, e) in report { print("  \(n.padding(toLength: 10, withPad: " ", startingAt: 0)) \(String(format: "%.3e", e))") }
    if audio.shape == ref["audio"]!.shape {
        print(String(format: "  audio      %.3e", relL2(audio, ref["audio"]!)))
    } else {
        print("  audio shape \(audio.shape) vs \(ref["audio"]!.shape)")
    }

case "wav":
    let (ref, tokens, refS) = try loadCase(args[2])
    let how = args.count > 4 ? args[4] : "0"
    let dt = dtypeArg(args.count > 5 ? args[5] : nil)
    let model = try KokoroModel(directory: models, dtype: dt)
    let noise: KokoroNoise = how == "zero" ? .zeros : how == "refnoise" ? .given(ref["sine_noise"]!)
        : .random(seed: UInt64(how) ?? 0)
    let (out, t) = timed { model.forward(tokens: tokens, refS: refS, noise: noise) }
    try save(arrays: ["audio": out.audio, "pred_dur": MLXArray(out.durations.map { Int32($0) })],
             url: URL(fileURLWithPath: args[3]))
    print("wrote \(out.audio.shape[0]) samples (\(String(format: "%.2f", Double(out.audio.shape[0]) / 24000)) s audio) in \(String(format: "%.2f", t)) s")

case "bench":
    let (_, tokens, refS) = try loadCase(args[2])
    let dt = dtypeArg(args.count > 3 ? args[3] : nil)
    let runs = args.count > 4 ? Int(args[4]) ?? 3 : 3
    let (model, tl) = try timed { try KokoroModel(directory: models, dtype: dt) }
    print("load \(String(format: "%.2f", tl)) s; \(tokens.count) tokens, dtype \(dt)")
    for r in 0 ..< runs {
        MLX.GPU.resetPeakMemory()
        let (out, t) = timed { model(tokens: tokens, refS: refS, seed: UInt64(r)) }
        let secs = Double(out.audio.shape[0]) / 24000
        print(String(format: "  run %d: %.3f s for %.2f s audio → RTF %.4f (%.0f× real time), peak GPU %.0f MB", r, t, secs,
                     t / secs, secs / t, Double(Memory.peakMemory) / 1e6))
    }

default:
    print("usage: kokoro-probe-model pth|ref|wav|bench …")
}
