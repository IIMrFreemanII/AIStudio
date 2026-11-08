import Foundation
import LTX2Transformer
import MLX
import MLXDiffusionCore

// Usage: ltx-probe-dit pos|cmp|bench <work-dir> [model-dir]
let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "cmp"
let work = URL(fileURLWithPath: args.count > 2 ? args[2] : FileManager.default.currentDirectoryPath)
let modelDir = URL(fileURLWithPath: args.count > 3 ? args[3]
    : NSHomeDirectory() + "/Library/Application Support/MLXComfyUI/models/ltx-2.3-mlx-q4")

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    let x = a.asType(.float32), y = b.asType(.float32)
    guard x.shape == y.shape else { print("\(name): SHAPE MISMATCH \(x.shape) vs \(y.shape)"); return }
    let d = x - y
    let maxAbs = abs(d).max().item(Float.self)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    let refMax = abs(y).max().item(Float.self)
    print(String(format: "%-28@ dtype %@/%@  max-abs %.3e  (ref max %.3e)  rel-L2 %.3e", name as NSString,
                 "\(a.dtype)" as NSString, "\(b.dtype)" as NSString, maxAbs, refMax, rel))
}

func load(_ name: String) throws -> [String: MLXArray] {
    try loadArrays(url: work.appendingPathComponent(name))
}

switch mode {
case "pos":
    let r = try load("pos.safetensors")
    let fhw = r["fhw"]!.asArray(Int32.self).map(Int.init)
    let s = ltxVideoLatentShape(numFrames: 121, height: 256, width: 384)
    print("latent shape swift \(s) ref \(fhw)")
    let na = ltxAudioTokenCount(numVideoFrames: 121, frameRate: 24)
    print("audio tokens swift \(na) ref \(r["audio_T"]!.item(Int32.self))")
    let vp = ltxVideoPositions(frames: s.frames, height: s.height, width: s.width, frameRate: 24)
    report("video positions", vp, r["video_positions"]!)
    print("  bit-exact:", arrayEqual(vp, r["video_positions"]!).item(Bool.self))
    let ap = ltxAudioPositions(numTokens: na)
    report("audio positions", ap, r["audio_positions"]!)
    print("  bit-exact:", arrayEqual(ap, r["audio_positions"]!).item(Bool.self))
    let vp2 = ltxVideoPositions(frames: 16, height: 16, width: 24, frameRate: 25)
    print("  25fps stage-2 bit-exact:", arrayEqual(vp2, r["video_positions_25"]!).item(Bool.self))
    let lat = r["latent"]!
    let tok = ltxPatchifyVideo(lat)
    print("patchify video exact:", arrayEqual(tok, r["latent_tok"]!).item(Bool.self),
          "roundtrip:", arrayEqual(ltxUnpatchifyVideo(tok, frames: lat.dim(2), height: lat.dim(3), width: lat.dim(4)), lat).item(Bool.self))
    let alat = r["alatent"]!
    let atok = ltxPatchifyAudio(alat)
    print("patchify audio exact:", arrayEqual(atok, r["alatent_tok"]!).item(Bool.self),
          "roundtrip:", arrayEqual(ltxUnpatchifyAudio(atok), alat).item(Bool.self))

case "cmp":
    var t0 = Date()
    let model = try LTXTransformer(directory: modelDir)
    print("init \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
    let x0m = LTXX0Model(model)
    for c in ["a", "b"] {
        guard FileManager.default.fileExists(atPath: work.appendingPathComponent("case_\(c)_in.safetensors").path)
        else { continue }
        let i = try load("case_\(c)_in.safetensors")
        let o = try load("case_\(c)_out.safetensors")
        print("== case \(c)")
        if c == "a" {
            // RoPE tables
            let vr = LTXRope(positions: i["video_positions"]!, heads: 32, headDim: 128, theta: 10000,
                             maxPos: [20, 2048, 2048], doublePrecision: true)
            report("rope video cos", vr.cos, o["rope_v_cos"]!)
            report("rope video sin", vr.sin, o["rope_v_sin"]!)
            let cr = LTXRope(positions: i["audio_positions"]!, heads: 32, headDim: 64, theta: 10000,
                             maxPos: [20], doublePrecision: true)
            report("rope audio-cross cos", cr.cos, o["rope_ac_cos"]!)
        }
        var hooks: [String: MLXArray] = [:]
        t0 = Date()
        let (v, a) = model(
            video: i["video"]!, audio: i["audio"]!, sigma: i["sigma"]!, videoTimesteps: i["video_timesteps"],
            audioTimesteps: i["audio_timesteps"], videoText: i["video_text"]!, audioText: i["audio_text"]!,
            videoPositions: i["video_positions"]!, audioPositions: i["audio_positions"]!,
            blockHook: { idx, vh, ah in
                if idx == 0 || idx == 47 { hooks["b\(idx)_v"] = vh; hooks["b\(idx)_a"] = ah; eval(vh, ah) }
            })
        eval(v, a)
        print("forward \(String(format: "%.2f", Date().timeIntervalSince(t0)))s")
        for k in ["b0_v", "b0_a", "b47_v", "b47_a"] { report("block " + k, hooks[k]!, o[k]!) }
        report("velocity video", v, o["v"]!)
        report("velocity audio", a, o["a"]!)
        let (xv, xa) = x0m(
            video: i["video"]!, audio: i["audio"]!, sigma: i["sigma"]!, videoTimesteps: i["video_timesteps"],
            audioTimesteps: i["audio_timesteps"], videoText: i["video_text"]!, audioText: i["audio_text"]!,
            videoPositions: i["video_positions"]!, audioPositions: i["audio_positions"]!)
        report("x0 video", xv, o["x0_v"]!)
        report("x0 audio", xa, o["x0_a"]!)
        if c == "b" {
            let n0 = 24
            let d = abs(xv[0..., ..<n0] - i["video"]![0..., ..<n0].asType(xv.dtype)).max().item(Float.self)
            print("  x0 of conditioning tokens == input: max diff \(d)")
        }
        MLX.Memory.clearCache()
    }

case "bench":
    let computeDType: DType? = args.count > 4 && args[4] == "bf16" ? .bfloat16 : nil
    let model = try LTXTransformer(directory: modelDir, computeDType: computeDType)
    let s = ltxVideoLatentShape(numFrames: 121, height: 256, width: 384)
    let nv = s.frames * s.height * s.width
    let na = ltxAudioTokenCount(numVideoFrames: 121, frameRate: 24)
    let nt = 1024
    MLXRandom.seed(0)
    let video = MLXRandom.normal([1, nv, 128])
    let audio = MLXRandom.normal([1, na, 128])
    let vText = MLXRandom.normal([1, nt, 4096])
    let aText = MLXRandom.normal([1, nt, 2048])
    let vpos = ltxVideoPositions(frames: s.frames, height: s.height, width: s.width)
    let apos = ltxAudioPositions(numTokens: na)
    let sigma = MLXArray([Float(0.9)]).asType(.bfloat16)
    // per-token video timesteps (I2V): first latent frame at 0
    var mask = [Float](repeating: 0.9, count: nv)
    for j in 0 ..< s.height * s.width { mask[j] = 0 }
    let vts = MLXArray(mask, [1, nv])
    print("video tokens \(nv), audio tokens \(na), text \(nt), computeDType \(String(describing: computeDType))")
    MLX.Memory.peakMemory = 0
    for run in 0 ..< 3 {
        let t0 = Date()
        let (v, a) = model(video: video, audio: audio, sigma: sigma, videoTimesteps: run == 2 ? vts : nil,
                           videoText: vText, audioText: aText, videoPositions: vpos, audioPositions: apos)
        eval(v, a)
        print(String(format: "run %d%@: %.2fs  peak %.2f GB", run, run == 2 ? " (per-token ts)" : "",
                     Date().timeIntervalSince(t0), Double(MLX.Memory.peakMemory) / 1e9))
    }

default:
    print("unknown mode \(mode)")
}
