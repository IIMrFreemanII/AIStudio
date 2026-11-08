import Foundation
import LTX2VAE
import MLX
import MLXDiffusionCore

// Verification probe for LTX2VAE against dumps written by $S/work-vae/ref_small.py.
// usage: ltx-probe-vae <work-dir> [dec] [tiled] [up] [enc] [auto] [bench] [bench-default]

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: ltx-probe-vae <work-dir> [dec|tiled|up|enc|auto|bench|bench-default]...")
    exit(2)
}
let work = URL(fileURLWithPath: args[1])
let what = Set(args.dropFirst(2).isEmpty ? ["dec", "tiled", "up", "enc", "auto"] : Array(args.dropFirst(2)))
let models = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/MLXComfyUI/models/ltx-2.3-mlx-q4")

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    guard a.shape == b.shape else {
        print(String(format: "  %-14@ SHAPE MISMATCH swift %@ ref %@", name as NSString, "\(a.shape)", "\(b.shape)"))
        return
    }
    let x = a.asType(.float32), y = b.asType(.float32)
    let d = abs(x - y)
    let maxAbs = d.max().item(Float.self)
    let meanAbs = d.mean().item(Float.self)
    let refMax = abs(y).max().item(Float.self)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    print(String(format: "  %-14@ %@ %@  maxabs %.4g (ref max %.3g)  meanabs %.3g  rel-L2 %.3g",
                 name as NSString, "\(a.shape)" as NSString, "\(a.dtype)" as NSString, maxAbs, refMax, meanAbs, rel))
}

func timed<T>(_ label: String, _ f: () throws -> T) rethrows -> T {
    let t = Date()
    let r = try f()
    print(String(format: "  [%@] %.3f s", label, Date().timeIntervalSince(t)))
    return r
}

func load(_ name: String) throws -> [String: MLXArray] {
    try loadArrays(url: work.appendingPathComponent(name))
}

func cat(_ parts: [MLXArray]) -> MLXArray { parts.count == 1 ? parts[0] : concatenated(parts, axis: 1) }

if what.contains("dec") || what.contains("tiled") || what.contains("big") || what.contains("bench") || what.contains("bench-default") {
    let dec = try LTXVideoDecoder(url: models.appendingPathComponent("vae_decoder.safetensors"))
    var traced: [String: MLXArray] = [:]

    if what.contains("dec") {
        print("== decoder (untiled), latent 1x3x4x6")
        let r = try load("dec_small.safetensors")
        let lat = r["latent"]!
        report("denorm", dec.denormalize(lat), r["denorm"]!)
        let y = timed("decode") { () -> MLXArray in let y = dec.decode(lat); eval(y); return y }
        report("decoded", y, r["decoded"]!)
        let y2 = timed("decode warm") { () -> MLXArray in let y = dec.decode(lat); eval(y); return y }
        _ = y2
        dec.debugTrace = { traced[$0] = $1 }
        _ = dec.decode(lat)
        dec.debugTrace = nil
        for k in ["conv_in"] + (0...8).map({ "up\($0)" }) { report(k, traced[k]!, r[k]!) }
        let fr = dec.decodeFrames(lat, tiling: .untiled)
        report("frames[0,1]", fr, clip((r["decoded"]![0].asType(.float32) + 1) / 2, min: 0, max: 1))
        MLX.GPU.clearCache()
    }

    if what.contains("tiled") {
        print("== decoder (tiled 192px/64, 24f/8), latent 1x5x6x10")
        let r = try load("dec_tiled.safetensors")
        let lat = r["latent"]!
        var chunks: [MLXArray] = []
        var firsts: [Int] = []
        timed("tiled decode") {
            dec.decodeTiled(lat, tiling: LTXVAETiling(
                spatial: LTXSpatialTiling(tileSizeInPixels: 192, tileOverlapInPixels: 64),
                temporal: LTXTemporalTiling(tileSizeInFrames: 24, tileOverlapInFrames: 8))) { c, f in
                chunks.append(c)
                firsts.append(f)
            }
        }
        print("  chunk starts \(firsts) shapes \(chunks.map { $0.shape })")
        var i = 0
        while let ref = r["chunk\(i)"] {
            if i < chunks.count { report("chunk\(i)", chunks[i], ref) } else { print("  missing chunk\(i)") }
            i += 1
        }
        report("untiled", dec.decode(lat), r["untiled"]!)
        MLX.GPU.clearCache()
    }

    if what.contains("big") {
        print("== big tiled decode vs reference: latent 1x16x16x24, frames=40/8 px=512/32")
        let r = try load("dec_big.safetensors")
        let t0 = Date()
        let frames = dec.decodeFrames(r["latent"]!, tiling: .fixed(LTXVAETiling(
            spatial: LTXSpatialTiling(tileSizeInPixels: 512, tileOverlapInPixels: 32),
            temporal: LTXTemporalTiling(tileSizeInFrames: 40, tileOverlapInFrames: 8))))
        print(String(format: "  %.2f s", Date().timeIntervalSince(t0)))
        report("frames(f16)", frames.asType(.float16), r["frames"]!)
        MLX.GPU.clearCache()
    }

    if what.contains("bench") || what.contains("bench-default") {
        let tiling: LTXVAETiling? = what.contains("bench-default")
            ? .default : LTXVAETiling.auto(frames: 16, height: 16, width: 24)
        print("== bench: latent 1x16x16x24 (121 frames 512x768), tiling \(tiling?.summary ?? "untiled")")
        MLXRandom.seed(5)
        let lat = MLXRandom.normal([1, 16, 16, 24, 128]).asType(.bfloat16)
        eval(lat)
        MLX.GPU.clearCache()
        MLX.GPU.resetPeakMemory()
        let base = MLX.GPU.activeMemory
        var n = 0
        let t0 = Date()
        var firstChunk: Double = 0
        dec.decodeFrames(lat, tiling: tiling.map { .fixed($0) } ?? .untiled, collect: false) { frames, first in
            if n == 0 { firstChunk = Date().timeIntervalSince(t0) }
            n += frames.dim(0)
            print(String(format: "  chunk @%d: %d frames  t=%.2fs", first, frames.dim(0), Date().timeIntervalSince(t0)))
        }
        let dt = Date().timeIntervalSince(t0)
        print(String(format: "  decoded %d frames in %.2f s (first chunk %.2f s), peak %.2f GB (weights/active before %.2f GB)",
                     n, dt, firstChunk, Double(MLX.GPU.peakMemory) / 1e9, Double(base) / 1e9))
    }
}

if what.contains("up") {
    print("== latent upsampler x2, latent 1x3x4x6")
    let r = try load("up_small.safetensors")
    let up = try LTXLatentUpsampler(url: models.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors"),
                                    configURL: models.appendingPathComponent("spatial_upscaler_x2_v1_1_config.json"))
    let enc = try LTXVideoEncoder(url: models.appendingPathComponent("vae_encoder.safetensors"))
    let lat = r["latent"]!
    report("denorm", enc.denormalizeLatent(lat), r["denorm"]!)
    let u = timed("upsample") { () -> MLXArray in let u = up(r["denorm"]!); eval(u); return u }
    report("up", u, r["up"]!)
    report("up_norm", up.upsampleNormalized(lat, encoder: enc), r["up_norm"]!)
    report("up_f32", up(lat.asType(.float32)), r["up_f32"]!)
    if let tr = try? load("up_trace.safetensors") {
        var got: [String: MLXArray] = [:]
        up.debugTrace = { got[$0] = $1 }
        eval(up(r["denorm"]!))
        up.debugTrace = nil
        for k in ["iconv", "inorm", "isilu", "rb0_c1", "rb0_n1", "rb0_s1", "rb0_c2", "rb0_n2", "rb0", "rb1", "rb2", "rb3",
                  "ups", "pb0_c1", "pb0_n1", "pb0_s1", "pb0_c2", "pb0_n2", "pb0_add", "pb0", "pb1", "pb2", "pb3", "final"] {
            if let a = got[k], let b = tr[k] { report(k, a, b) }
        }
    }
    MLX.GPU.clearCache()
}

if what.contains("enc") {
    print("== encoder")
    let r = try load("enc.safetensors")
    let enc = try LTXVideoEncoder(url: models.appendingPathComponent("vae_encoder.safetensors"))
    let img = URL(fileURLWithPath: "/Users/ndiahovets/ai/MLXComfyUI/outputs/fox.png")
    let rgb = try LTXImagePreprocess.loadRGB(img)
    report("raw rgb", rgb, r["raw"]!.asType(.float32))
    let px = LTXImagePreprocess.conditioningPixels(rgb: r["raw"]!.asType(.float32), height: 256, width: 384)
    report("pixels", px[0], r["pixels"]!)
    let lat = timed("encode") { () -> MLXArray in let l = enc.encode(r["pixels"]!.expandedDimensions(axis: 1)); eval(l); return l }
    report("latent", lat, r["latent"]!)
    let lat2 = try LTXImagePreprocess.encodeConditioning(url: img, height: 256, width: 384, encoder: enc)
    report("latent(file)", lat2, r["latent"]!)
    report("crf33 vs crf0", r["latent_crf33"]!, r["latent"]!)
    report("video latent", enc.encode(r["video"]!), r["video_latent"]!)
    MLX.GPU.clearCache()
}

if what.contains("auto") {
    print("== auto tiling (compare with ref_small.py auto)")
    for s in [(16, 16, 24), (16, 18, 32), (4, 8, 12), (31, 22, 38)] {
        for b: Int? in [nil, 8 << 30, 4 << 30] {
            let t = LTXVAETiling.auto(frames: s.0, height: s.1, width: s.2, budgetBytes: b)
            print("  \(s) \(b.map(String.init) ?? "None") \(t?.summary ?? "untiled")")
        }
    }
}

if what.contains("speed") {
    print("== speed (warm, typical pipeline sizes)")
    let enc = try LTXVideoEncoder(url: models.appendingPathComponent("vae_encoder.safetensors"))
    let up = try LTXLatentUpsampler(url: models.appendingPathComponent("spatial_upscaler_x2_v1_1.safetensors"))
    for (h, w) in [(256, 384), (512, 768)] {
        let px = (MLXRandom.uniform(low: -1, high: 1, [1, 1, h, w, 3])).asType(.bfloat16)
        eval(enc.encode(px))
        let t = Date()
        eval(enc.encode(px))
        print(String(format: "  encode 1 image %dx%d: %.3f s", w, h, Date().timeIntervalSince(t)))
    }
    let lat = MLXRandom.normal([1, 16, 8, 12, 128]).asType(.bfloat16)
    eval(up.upsampleNormalized(lat, encoder: enc))
    MLX.GPU.resetPeakMemory()
    let t = Date()
    let u = up.upsampleNormalized(lat, encoder: enc)
    print(String(format: "  upsample 1x16x8x12 -> %@: %.3f s, peak %.2f GB", "\(u.shape)", Date().timeIntervalSince(t),
                 Double(MLX.GPU.peakMemory) / 1e9))
}
