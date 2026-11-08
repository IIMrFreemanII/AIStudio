import ACEStepText
import Foundation
import MLX
import MLXDiffusionCore

// Verification probe for ACEStepText against dumps written by scratchpad/ace/ace_ref.py.
// usage: ace-probe-text <dump-dir> [qwen] [cond] ...

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: ace-probe-text <dump-dir> [qwen] [cond]...")
    exit(2)
}
let work = URL(fileURLWithPath: args[1])
let what = Set(args.dropFirst(2).isEmpty ? ["qwen"] : Array(args.dropFirst(2)))
let models = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/MLXComfyUI/models/ace-step-1.5")

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    guard a.shape == b.shape else {
        print(String(format: "  %-14@ SHAPE MISMATCH swift %@ ref %@", name as NSString, "\(a.shape)", "\(b.shape)"))
        return
    }
    let x = a.asType(.float32), y = b.asType(.float32)
    let d = abs(x - y)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    print(String(format: "  %-14@ %@  maxabs %.4g (ref max %.3g)  rel-L2 %.3g", name as NSString, "\(a.shape)" as NSString,
                 d.max().item(Float.self), abs(y).max().item(Float.self), rel))
}

let dump = try loadArrays(url: work.appendingPathComponent("dump.safetensors"))

if what.contains("qwen") {
    for dt in [DType.float32, .bfloat16] {
        let enc = try Qwen3Model(directory: models.appendingPathComponent("Qwen3-Embedding-0.6B"), dtype: dt)
        let ids = dump["te_ids0"]!.asType(.int32)
        let t = Date()
        let out = enc(ids)
        eval(out)
        print("qwen3-embedding \(dt) (\(String(format: "%.2f", Date().timeIntervalSince(t))) s)")
        report("hidden", out, dump["te_out0"]!)
        if let lyr = dump["gen_lyric_hidden_states"] {
            print("  (lyric embeddings use embed_tokens; checked in the condition encoder probe)", lyr.shape)
        }
    }
}

if what.contains("cond") {
    // Lyric ids for "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>" (the tokenizer appends another EOS).
    let lyricIDs: [Int32] = [2, 54964, 198, 16088, 271, 2, 15953, 2216, 198, 58, 56324, 278, 60, 151643, 151643]
    let te = try Qwen3Model(directory: models.appendingPathComponent("Qwen3-Embedding-0.6B"), dtype: .float32)
    let lyrEmb = te.embed(MLXArray(lyricIDs, [1, lyricIDs.count]))
    print("lyric embeddings (Qwen3Model.embed)")
    report("lyric_emb", lyrEmb, dump["gen_lyric_hidden_states"]!)
    let dit = models.appendingPathComponent("acestep-v15-turbo")
    let store = try WeightStore(url: dit.appendingPathComponent("model.safetensors"))
    let cfg = try ACEConditionEncoder.Config(url: dit.appendingPathComponent("config.json"))
    let order = dump["gen_refer_audio_order_mask"]!.asType(.int32).asArray(Int32.self).map(Int.init)
    for dt in [DType.float32, .bfloat16] {
        let enc = try ACEConditionEncoder(store: store, config: cfg, dtype: dt)
        func run() -> (MLXArray, MLXArray) {
            let r = enc(text: dump["gen_text_hidden_states"]!, textMask: dump["gen_text_attention_mask"]!,
                        lyrics: dump["gen_lyric_hidden_states"]!, lyricMask: dump["gen_lyric_attention_mask"]!,
                        timbre: dump["gen_refer_audio_acoustic_hidden_states_packed"]!, timbreOrder: order)
            eval(r.hidden, r.mask)
            return r
        }
        var t = Date()
        var (h, m) = run()
        let first = Date().timeIntervalSince(t)
        t = Date()
        for _ in 0..<3 { (h, m) = run() }
        let warm = Date().timeIntervalSince(t) / 3
        print("condition encoder \(dt) (first \(String(format: "%.3f", first)) s, warm \(String(format: "%.3f", warm)) s)")
        report("cond_enc", h, dump["cond_enc"]!)
        report("cond_enc_mask", m, dump["cond_enc_mask"]!)
        let ref = dump["cond_enc"]!, nl = lyricIDs.count
        report(" lyric part", h[0..., 0 ..< nl], ref[0..., 0 ..< nl])
        report(" timbre part", h[0..., nl ..< nl + 1], ref[0..., nl ..< nl + 1])
        report(" text part", h[0..., (nl + 1)...], ref[0..., (nl + 1)...])
        // Padded, batched case (B=2, 3 refs ordered [0,1,1]) from scratchpad/work-ace/cond/pad_ref.py, if present.
        let padURL = work.deletingLastPathComponent().appendingPathComponent("cond/pad.safetensors")
        if let p = try? loadArrays(url: padURL) {
            let ord = p["order"]!.asType(.int32).asArray(Int32.self).map(Int.init)
            let r = enc(text: p["text"]!, textMask: p["tmask"]!, lyrics: p["lyr"]!, lyricMask: p["lmask"]!,
                        timbre: p["ref"]!, timbreOrder: ord)
            let valid = p["mask"]!.expandedDimensions(axis: -1)
            report("padded (valid)", r.hidden * valid, p["out"]! * valid)
            report("padded (all)", r.hidden, p["out"]!)
            report("padded mask", r.mask, p["mask"]!)
        }
        if let n = enc.nullConditionEmb { print("  null_condition_emb", n.shape) }
    }
}
