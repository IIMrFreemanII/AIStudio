import ACEStepDiT
import Foundation
import MLX
import MLXDiffusionCore

// Usage: ace-probe-dit fwd|sample|euler|bench|dcw|layers <dump-dir> [f32w16|f32|bf16|bf16r] [model-dir] [ref.safetensors]
let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "fwd"
let work = URL(fileURLWithPath: args.count > 2 ? args[2] : FileManager.default.currentDirectoryPath)
// dtype arg: f32 | bf16 (bf16 matmuls, fp32 residual) | bf16r (bf16 residual too, torch-bf16-like)
let dtype: DType = (args.count > 3 && args[3].hasPrefix("bf16")) ? .bfloat16 : .float32
let residualDType: DType = (args.count > 3 && args[3] == "bf16r") ? .bfloat16 : .float32
// f32 = float32 compute on float32-converted weights; f32w16 = float32 compute on the checkpoint's bf16 weights
let weightDType: DType? = (args.count > 3 && args[3] == "f32") ? .float32 : nil
let modelDir = URL(fileURLWithPath: args.count > 4 ? args[4]
    : NSHomeDirectory() + "/Library/Application Support/AIStudio/models/ace-step-1.5/acestep-v15-turbo")

@discardableResult
func report(_ name: String, _ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.asType(.float32), y = b.asType(.float32)
    guard x.shape == y.shape else { print("\(name): SHAPE MISMATCH \(x.shape) vs \(y.shape)"); return .nan }
    let d = x - y
    let maxAbs = abs(d).max().item(Float.self)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    let refMax = abs(y).max().item(Float.self)
    print(String(format: "  %-22@ max-abs %.3e  (ref max %.3e)  rel-L2 %.3e", name as NSString, maxAbs, refMax, rel))
    return rel
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

let ref = (try? loadArrays(url: work.appendingPathComponent("dump.safetensors"))) ?? [:]
func r(_ k: String) -> MLXArray { guard let a = ref[k] else { fatalError("missing \(k) in dump") }; return a }

func loadDiT() throws -> ACEDiT {
    let t0 = Date()
    let store = try WeightStore(url: modelDir.appendingPathComponent("model.safetensors"))
    let cfg = try ACEDiT.Config(url: modelDir.appendingPathComponent("config.json"))
    let dit = try ACEDiT(store: store, config: cfg, dtype: dtype, weightDType: weightDType, residualDType: residualDType)
    store.remove(prefix: "")
    Memory.clearCache()
    print("loaded DiT (\(dtype), weights \(dit.weightDType), residual \(residualDType)) in \(String(format: "%.2f", Date().timeIntervalSince(t0)))s, active \(gb(Memory.activeMemory))")
    return dit
}

let ctx = ref["cond_ctx"] ?? MLXArray(0), enc = ref["cond_enc"] ?? MLXArray(0)

switch mode {
case "dcw":
    // DCW in isolation against a reference produced by dcw_ref.py (x, y, t -> out per mode).
    let d = try loadArrays(url: work.appendingPathComponent("dcw.safetensors"))
    let t = d["t"]!.item(Float.self)
    for (name, m) in [("low", DCWOptions.Mode.low), ("high", .high), ("double", .double), ("pix", .pix)] {
        for suffix in ["", "_odd"] {
            guard let want = d["out_\(name)\(suffix)"] else { continue }
            let o = DCWOptions(enabled: true, mode: m, scaler: 0.05, highScaler: 0.02)
                .apply(d["x\(suffix)"]!, denoised: d["y\(suffix)"]!, t: t)
            report("dcw \(name)\(suffix)", o, want)
        }
    }

case "euler":
    // One sampler update from the reference (x_i, v_i) vs the reference x_{i+1}: isolates Euler + DCW math.
    let ts = ACESampler.schedule(steps: 8, shift: 1)
    for i in 0..<ts.count {
        let x = r("dit_x\(i)"), v = r("dit_v\(i)"), t = ts[i]
        let denoised = x - v * t
        if i == ts.count - 1 {
            report("x0 from step \(i)", denoised, r("gen_target_latents"))
        } else {
            let xn = x - v * (t - ts[i + 1])
            report("x\(i + 1) (DCW)", DCWOptions.turboDefault.apply(xn, denoised: denoised, t: t), r("dit_x\(i + 1)"))
            report("x\(i + 1) (no DCW)", xn, r("dit_x\(i + 1)"))
        }
    }

case "fwd":
    let dit = try loadDiT()
    let cache = dit.crossCache(encoder: enc)
    var i = 0
    while let x = ref["dit_x\(i)"] {
        let t = r("dit_t\(i)").reshaped(-1)[0].item(Float.self)
        let t0 = Date()
        let v = dit(x, t: t, context: ctx, encoder: enc, cache: cache)
        eval(v)
        let dt = Date().timeIntervalSince(t0)
        report("step \(i) t=\(t) (\(String(format: "%.0f", dt * 1000)) ms)", v, r("dit_v\(i)"))
        i += 1
    }
    // No-cache path once (exercises encoder → K/V inside the call).
    let v0 = dit(r("dit_x0"), t: 1, context: ctx, encoder: enc, cache: nil)
    report("step 0 no-cache", v0, r("dit_v0"))

case "sample":
    let dit = try loadDiT()
    print("schedule:", ACESampler.schedule(steps: 8, shift: 1))
    Memory.peakMemory = 0
    var stepTimes: [Double] = []
    var last = Date()
    let t0 = Date()
    let out = try ACESampler.sample(
        dit: dit, noise: r("dit_x0"), context: ctx, encoder: enc, steps: 8, shift: 1,
        onStep: { k, n in
            let now = Date(); stepTimes.append(now.timeIntervalSince(last)); last = now
            _ = n; _ = k
        },
        trace: { i, x, v in
            report("x\(i)", x, r("dit_x\(i)"))
            report("v\(i)", v, r("dit_v\(i)"))
        })
    eval(out)
    let total = Date().timeIntervalSince(t0)
    report("final (gen_target)", out, r("gen_target_latents"))
    print("step ms (incl. compare):", stepTimes.map { String(format: "%.0f", $0 * 1000) }.joined(separator: " "),
          String(format: " total %.2fs", total), " peak \(gb(Memory.peakMemory))")
    // DCW off, for scale: how much DCW moves the result.
    let noDCW = try ACESampler.sample(dit: dit, noise: r("dit_x0"), context: ctx, encoder: enc, dcw: .off)
    report("no-DCW vs gen_target", noDCW, r("gen_target_latents"))

case "bench":
    let dit = try loadDiT()
    _ = try ACESampler.sample(dit: dit, noise: r("dit_x0"), context: ctx, encoder: enc, steps: 1)  // warm-up
    for _ in 0..<2 {
        Memory.peakMemory = 0
        var stepTimes: [Double] = []
        var last = Date()
        let t0 = Date()
        let out = try ACESampler.sample(dit: dit, noise: r("dit_x0"), context: ctx, encoder: enc,
                                        onStep: { _, _ in let now = Date(); stepTimes.append(now.timeIntervalSince(last)); last = now })
        eval(out)
        print(String(format: "8 steps: %.2fs", Date().timeIntervalSince(t0)),
              "per step ms:", stepTimes.map { String(format: "%.0f", $0 * 1000) }.joined(separator: " "),
              "peak \(gb(Memory.peakMemory))")
    }

case "layers":
    // Per-layer comparison against a standalone torch decoder run (work-ace/dit/dit_ref.py output, path in args[5];
    // holds x, t, ctx, enc, v, proj_in, layerN). Also covers odd T (dit_ref.py ... 499).
    let d = try loadArrays(url: URL(fileURLWithPath: args[5]))
    let dit = try loadDiT()
    dit.trace = { name, a in if let want = d[name], ["proj_in", "layer0", "layer5", "layer11", "layer17", "layer23"].contains(name) { report(name, a, want) } }
    let v = dit(d["x"]!, t: d["t"]!.reshaped(-1)[0].item(Float.self), context: d["ctx"]!, encoder: d["enc"]!)
    dit.trace = nil
    report("v", v, d["v"]!)

default:
    print("unknown mode \(mode)")
}
