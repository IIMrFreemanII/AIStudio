import Foundation
import LTX2Text
import MLX
import MLXDiffusionCore
import Tokenizers

// Development probe for LTX2Text: compares the Swift port against tensors dumped by the Python reference
// (work-text/dump_ref.py).
//   ltx-probe-text tokens <workdir>        token ids vs tokens.json
//   ltx-probe-text full <workdir> [padded] hidden states, projections, connector outputs vs ref_p*.safetensors
//   ltx-probe-text bench "<prompt>"        timing + peak memory

let home = FileManager.default.homeDirectoryForCurrentUser
let models = home.appendingPathComponent("Library/Application Support/MLXComfyUI/models")
let gemmaDir = models.appendingPathComponent("gemma-3-12b-it-4bit")
let ltxDir = models.appendingPathComponent("ltx-2.3-mlx-q4")
let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "full"
let work = URL(fileURLWithPath: args.count > 2 ? args[2] : ".")

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    guard a.shape == b.shape else {
        print(String(format: "%-28@ SHAPE MISMATCH %@ vs %@", name as NSString, "\(a.shape)" as NSString,
                     "\(b.shape)" as NSString))
        return
    }
    let x = a.asType(.float32), y = b.asType(.float32)
    let d = x - y
    let maxAbs = abs(d).max().item(Float.self)
    let rel = (sqrt((d * d).sum()) / (sqrt((y * y).sum()) + 1e-12)).item(Float.self)
    let refMax = abs(y).max().item(Float.self)
    print(String(format: "%-28@ %@ max-abs %.3e  rel %.3e  (ref max %.3g)", name as NSString,
                 "\(a.dtype)/\(b.dtype)" as NSString, maxAbs, rel, refMax))
}

func gb(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }

switch mode {
case "tokens":
    let tok = try await AutoTokenizer.from(modelFolder: gemmaDir)
    let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: work.appendingPathComponent("tokens.json")))
        as! [String: Any]
    var ok = 0, total = 0
    for item in obj["items"] as! [[String: Any]] {
        let p = item["prompt"] as! String
        let ref = (item["ids"] as! [NSNumber]).map(\.intValue)
        let ids = tok.encode(text: p.trimmingCharacters(in: .whitespacesAndNewlines))
        total += 1
        if ids == ref { ok += 1 } else {
            print("MISMATCH \(p.debugDescription)\n  swift \(ids)\n  ref   \(ref)")
        }
        print("\(ids == ref ? "ok " : "BAD") n=\(ref.count) \(p.prefix(50).debugDescription)")
    }
    print("tokens: \(ok)/\(total) identical; pad id \(tok.convertTokenToId("<pad>") ?? -1)")

case "full":
    let padded = args.count > 3 && args[3] == "padded"
    var t = Date()
    let enc = try await LTXTextEncoder(gemmaDirectory: gemmaDir, ltxDirectory: ltxDir)
    enc.computePaddedRows = padded
    print(String(format: "load %.1fs  (padded rows: %@)", Date().timeIntervalSince(t), padded ? "yes" : "no"))
    print("gemma config:", enc.gemma.config)

    for (name, conn) in [("video", enc.connector.video), ("audio", enc.connector.audio)] {
        let f = work.appendingPathComponent("rope_\(name).safetensors")
        guard FileManager.default.fileExists(atPath: f.path) else { continue }
        let r = try loadArrays(url: f)
        let (c, s) = ltxConnectorRope(length: 1024, dim: conn.dim, heads: conn.heads, maxPos: conn.maxPos)
        report("rope.\(name).cos", c, r["cos"]!)
        report("rope.\(name).sin", s, r["sin"]!)
    }

    var i = 0
    while FileManager.default.fileExists(atPath: work.appendingPathComponent("ref_p\(i).safetensors").path) {
        let r = try loadArrays(url: work.appendingPathComponent("ref_p\(i).safetensors"))
        let refIds = r["ids"]!.asArray(Int32.self), refMask = r["mask"]!.asArray(Int32.self)
        let n = Int(refMask.reduce(0, +))
        let refTokens = Array(refIds.suffix(n)).map(Int.init)
        print("--- prompt \(i): \(n) tokens")
        let prompts = ["A cat walking on a sunny beach, cinematic lighting.", "",
                       "At 3:45 p.m., 2 robots (model X-1000) dance—wildly! Price: $19.99; 50% off? #test @home"]
        let ids = enc.tokenize(prompts[i])
        print("token ids", ids == refTokens ? "identical" : "DIFFER \(ids) vs \(refTokens)")
        let L = enc.maxLength

        // Gemma hidden states (valid rows)
        let hs = enc.hiddenStates(tokens: refTokens)
        for li in [0, 1, 5, 6, 24, 47, 48] {
            report("hs\(li)", hs[li], r["hs\(li)"]![0..., (L - n)...])
        }
        let x = ltxPerTokenRMSStack(hs, paddedLength: L)
        let (pv, pa) = enc.connector.project(x)
        report("proj_video (valid rows)", pv, r["proj_video"]![0..., (L - n)...])
        report("proj_audio (valid rows)", pa, r["proj_audio"]![0..., (L - n)...])
        let cin = enc.connector.video.fillWithRegisters(pv, length: L)
        report("conn_in_video", cin, r["conn_in_video"]!)
        // Connector alone, on the reference's own input
        report("video conn(ref input)", enc.connector.video(r["conn_in_video"]!), r["video_from_proj"]!)

        MLX.GPU.clearCache()
        Memory.peakMemory = 0
        t = Date()
        let (v, a) = enc.encode(tokens: refTokens)
        let dt = Date().timeIntervalSince(t)
        report("video (end to end)", v, r["video"]!)
        report("audio (end to end)", a, r["audio"]!)
        print(String(format: "encode %.2fs, peak %@ (active %@)", dt, gb(Memory.peakMemory) as NSString,
                     gb(Memory.activeMemory) as NSString))
        MLX.GPU.clearCache()
        i += 1
    }

case "layers":
    // Every Gemma layer of prompt 0 (padded/exact mode) vs layers_p0.safetensors (dump_layers.py).
    let enc = try await LTXTextEncoder(gemmaDirectory: gemmaDir, ltxDirectory: ltxDir)
    enc.computePaddedRows = !(args.count > 3 && args[3] == "fast")
    let ref = try loadArrays(url: work.appendingPathComponent("layers_p0.safetensors"))["all"]!
    let hs = enc.hiddenStates(tokens: enc.tokenize("A cat walking on a sunny beach, cinematic lighting."))
    for (i, h) in hs.enumerated() { report("layer \(i)", h[0], ref[i]) }

case "layer12":
    let g = try LTXGemma3Model(directory: gemmaDir)
    let r = try loadArrays(url: work.appendingPathComponent("layer12.safetensors"))
    let valid = [Int32](repeating: 0, count: 1012) + [Int32](repeating: 1, count: 12)
    let mask = ltxGemmaMask(valid: valid)
    for li in [12, 13, 14] {
        let out = g.debugLayer(li, r["l\(li).inp"]!.expandedDimensions(axis: 0), mask: mask)
        for k in ["xn", "attn", "h", "f", "gate", "up", "gelu", "mlp", "out"] {
            report("l\(li).\(k)", out[k]![0, (1012)...], r["l\(li).\(k)"]!)
        }
    }

case "conn":
    // Feature extractor + connectors only, on seeded random hidden states (dump_connector.py).
    let conn = try LTXTextConnector(url: ltxDir.appendingPathComponent("connector.safetensors"),
                                    ltxConfig: ltxDir.appendingPathComponent("embedded_config.json"))
    print("double precision rope:", conn.video.doublePrecisionRope)
    let r = try loadArrays(url: work.appendingPathComponent("conn_rand.safetensors"))
    let hv = r["hs_valid"]!
    let states = (0..<hv.dim(0)).map { hv[$0] }
    let t = Date()
    let padProj = args.count > 3 && args[3] == "padproj"
    var (pv, pa) = conn.project(ltxPerTokenRMSStack(states, paddedLength: 1024, keepPadding: padProj))
    if padProj { pv = pv[0..., (1024 - hv.dim(2))...]; pa = pa[0..., (1024 - hv.dim(2))...] }
    let v = conn.video(conn.video.fillWithRegisters(pv, length: 1024))
    let a = conn.audio(conn.audio.fillWithRegisters(pa, length: 1024))
    eval(v, a)
    print(String(format: "connector %.2fs", Date().timeIntervalSince(t)))
    report("video", v, r["video"]!)
    report("audio", a, r["audio"]!)
    let n = hv.dim(2)
    let mid = try loadArrays(url: work.appendingPathComponent("conn_rand_mid.safetensors"))
    report("stacked", ltxPerTokenRMSStack(states, paddedLength: 1024), mid["stacked_valid"]!)
    report("proj_video", pv, mid["proj_video"]![0..., (1024 - n)...])
    report("proj_audio", pa, mid["proj_audio"]![0..., (1024 - n)...])
    report("conn_in", conn.video.fillWithRegisters(pv, length: 1024), mid["conn_in"]!)
    report("block0(ref conn_in)", conn.video(mid["conn_in"]!, blockLimit: 1), mid["block0"]!)
    report("video(ref conn_in)", conn.video(mid["conn_in"]!), r["video"]!)
    report("video (token rows)", v[0..., ..<n], r["video"]![0..., ..<n])
    report("video (register rows)", v[0..., n...], r["video"]![0..., n...])
    for (name, c) in [("video", conn.video), ("audio", conn.audio)] {
        let f = work.appendingPathComponent("rope_\(name).safetensors")
        guard FileManager.default.fileExists(atPath: f.path) else { continue }
        let rr = try loadArrays(url: f)
        let (cc, ss) = ltxConnectorRope(length: 1024, dim: c.dim, heads: c.heads, maxPos: c.maxPos)
        report("rope.\(name).cos", cc, rr["cos"]!)
        report("rope.\(name).sin", ss, rr["sin"]!)
    }

case "bench":
    let prompt = args.count > 2 ? args[2] : "A cat walking on a sunny beach, cinematic lighting."
    var t = Date()
    let enc = try await LTXTextEncoder(gemmaDirectory: gemmaDir, ltxDirectory: ltxDir)
    enc.projectPaddedRows = !(args.count > 3 && args[3] == "noproj")
    print(String(format: "load %.1fs (projectPaddedRows %@)", Date().timeIntervalSince(t),
                 enc.projectPaddedRows ? "on" : "off"))
    for k in 0..<3 {
        MLX.GPU.clearCache()
        Memory.peakMemory = 0
        t = Date()
        let (v, a) = try enc.encode(prompt)
        print(String(format: "run %d: encode %.2fs  peak %@  -> video %@ audio %@", k, Date().timeIntervalSince(t),
                     gb(Memory.peakMemory) as NSString, "\(v.shape) \(v.dtype)" as NSString,
                     "\(a.shape)" as NSString))
    }

default:
    print("usage: ltx-probe-text tokens|full|bench <workdir|prompt> [padded]")
}
