import ACEStepLM
import ACEStepText
import Foundation
import MLX
import MLXDiffusionCore
import Tokenizers

// Verification probe for ACEStepLM against references in scratchpad/work-ace (think20 = official thinking run,
// lm/ = prompts.json, logits.safetensors, greedy.json written by work-ace/lm/{prompts,greedy}.py).
// usage: ace-probe-lm <work-ace-dir> [decoder] [prompts] [logits] [greedy] [sample] [bf16|fp32]

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: ace-probe-lm <work-ace-dir> [decoder] [prompts] [logits] [greedy] [sample]")
    exit(2)
}
let work = URL(fileURLWithPath: args[1])
let what = Set(args.dropFirst(2))
let models = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent("Library/Application Support/MLXComfyUI/models/ace-step-1.5")
let lmDir = models.appendingPathComponent("acestep-5Hz-lm-1.7B")
let ditDir = models.appendingPathComponent("acestep-v15-turbo")
let dtypes: [DType] = what.contains("fp32") ? [.float32] : what.contains("bf16") ? [.bfloat16] : [.float32, .bfloat16]

func report(_ name: String, _ a: MLXArray, _ b: MLXArray) {
    guard a.shape == b.shape else {
        print("  \(name) SHAPE MISMATCH swift \(a.shape) ref \(b.shape)")
        return
    }
    let x = a.asType(.float32), y = b.asType(.float32)
    let d = abs(x - y)
    let rel = (sqrt((d * d).sum()) / sqrt((y * y).sum())).item(Float.self)
    print(String(format: "  %-24@ %@  maxabs %.4g (ref max %.3g)  rel-L2 %.3g", name as NSString,
                 "\(a.shape)" as NSString, d.max().item(Float.self), abs(y).max().item(Float.self), rel))
}

func json(_ name: String) throws -> Any { try JSONSerialization.jsonObject(with: Data(contentsOf: work.appendingPathComponent(name))) }

func parseCodes(_ s: String) -> [Int] {
    s.components(separatedBy: "<|audio_code_").dropFirst().compactMap { Int($0.prefix { $0.isNumber }) }
}

let think20 = (try json("think20/lm.json") as! [[String: Any]])[0]
let kw = think20["kwargs"] as! [String: String]
let refCodes = parseCodes(think20["audio_codes"] as! String)

func baseRequest(seed: UInt64 = 42) -> ACEPlanRequest {
    ACEPlanRequest(caption: kw["caption"]!, lyrics: kw["lyrics"]!, duration: 20, language: "en", seed: seed)
}

// MARK: (a) code decoder
if what.contains("decoder") {
    let store = try WeightStore(url: ditDir.appendingPathComponent("model.safetensors"))
    let cfg = try ACEAudioCodeDecoder.Config(url: ditDir.appendingPathComponent("config.json"))
    let dump = try loadArrays(url: work.appendingPathComponent("think20/dump.safetensors"))
    print("codes from lm.json: \(refCodes.count)")
    for dt in [DType.float32, .bfloat16] {
        let dec = try ACEAudioCodeDecoder(store: store, config: cfg, dtype: dt)
        var t = Date()
        var h = dec.hints(codes: refCodes)
        eval(h)
        let first = Date().timeIntervalSince(t)
        t = Date()
        h = dec.hints(codes: refCodes)
        eval(h)
        print("code decoder \(dt) (first \(String(format: "%.3f", first)) s, warm \(String(format: "%.3f", Date().timeIntervalSince(t))) s)")
        report("lm_hints", h, dump["lm_hints"]!)
        if dt == .float32 {
            // DiT-input merge with the reference plan's metadata.
            let meta = ACEPlanMetadata(bpm: 130, caption: "x", duration: 20, keyscale: "D major", language: "en",
                                       timeSignature: "4")
            let plan = ACEPlan(metadata: meta, cotText: "", cot: "", codes: refCodes)
            let c = ACEDiTConditioning.make(plan: plan, request: baseRequest(), hints: h, silenceLatent: dump["silence_latent"]!)
            report("src_latents", c.srcLatents, dump["gen_src_latents"]!)
            report("precomputed_hints", c.lmHints, dump["gen_precomputed_lm_hints_25Hz"]!)
            print("  metas: \(c.metas.debugDescription) T=\(c.latentLength) isCover=\(c.isCover)")
        }
    }
}

// MARK: (b) prompts, ids, FSM tables, YAML
if what.contains("prompts") {
    let P = try json("lm/prompts.json") as! [String: Any]
    let prompts = P["prompts"] as! [String: String]
    let refIDs = P["ids"] as! [String: [Int]]
    let t = Date()
    let tok = try await AutoTokenizer.from(modelFolder: lmDir)
    print("tokenizer load \(String(format: "%.2f", Date().timeIntervalSince(t))) s (python: ~10 s)")
    let refMeta = P["metadata"] as! [String: Any]
    let meta = ACEPlanMetadata(bpm: refMeta["bpm"] as? Int, caption: refMeta["caption"] as? String,
                               duration: refMeta["duration"] as? Int, keyscale: refMeta["keyscale"] as? String,
                               language: refMeta["language"] as? String, timeSignature: refMeta["timesignature"] as? String)
    let cot = ACEPlannerText.formatCoT(meta)
    print("cot text identical: \(cot == P["cot_text"] as! String)")
    let r = baseRequest()
    let mine: [String: String] = [
        "cot": ACEPlannerText.cotPrompt(caption: r.caption, lyrics: r.lyrics),
        "codes": ACEPlannerText.codesPrompt(caption: r.caption, lyrics: r.lyrics, cot: cot),
        "codes_uncond": ACEPlannerText.codesUnconditionalPrompt(),
        "cot_uncond": ACEPlannerText.cotUnconditionalPrompt(lyrics: r.lyrics),
    ]
    for k in ["cot", "codes", "codes_uncond", "cot_uncond"] {
        let ids = tok.encode(text: mine[k]!, addSpecialTokens: false)
        print("  \(k): string identical \(mine[k]! == prompts[k]!), ids identical \(ids == refIDs[k]!) (\(ids.count) tokens)")
    }
    // Constrained-decoding tables.
    let F = P["fsm"] as! [String: Any]
    let tables = try ACEPlanner.constraintTables(tokenizer: tok)
    for name in ["bpm", "duration", "timesig", "keyscale", "language"] {
        let refTree = F[name] as! [String: [Int]]
        var refConv: [[Int]: [Int]] = [:]
        for (k, v) in refTree { refConv[k.isEmpty ? [] : k.split(separator: " ").map { Int($0)! }] = v }
        print("  prefix tree \(name): identical \(refConv == tables.trees[name]!) (\(refConv.count) prefixes)")
    }
    let refFixed = F["fixed"] as! [String: [[Int]]]
    let names = ["THINK_TAG": "<think>", "NEWLINE_AFTER_THINK": "\n", "BPM_NAME": "bpm:", "CAPTION_NAME": "caption:",
                 "DURATION_NAME": "duration:", "KEYSCALE_NAME": "keyscale:", "LANGUAGE_NAME": "language:",
                 "TIMESIG_NAME": "timesignature:", "THINK_END_TAG": "</think>"]
    let fixedOK = names.allSatisfy { refFixed[$0.key]!.map { $0[0] } == tables.fixed[$0.value]! }
    print("  fixed-string tokens identical: \(fixedOK)")
    var userOK = true
    for (s, ids) in F["user"] as! [String: [Int]] where tok.encode(text: s, addSpecialTokens: false) != ids { userOK = false }
    print("  user-field injections identical: \(userOK)")
    // Parse of a raw CoT output.
    let raw = "<think>\nbpm: 130\ncaption: An energetic synth-pop track driven by a punchy four-on-the-floor drum machine\n  beat and a pulsing\nduration: 20\nkeyscale: D major\nlanguage: en\ntimesignature: 4\n<|im_end|>"
    let pm = ACEPlannerText.parse(raw)
    let pe = P["parse_example"] as! [String: Any]
    print("parse example ok: \(pm.bpm == pe["bpm"] as? Int && pm.caption == pe["caption"] as? String && pm.duration == pe["duration"] as? Int && pm.keyscale == pe["keyscale"] as? String && pm.language == pe["language"] as? String && pm.timeSignature == pe["timesignature"] as? String)")
    // YAML dumps.
    var yamlOK = 0, yamlN = 0
    for (cap, ref) in P["yaml"] as! [String: String] {
        let s = ACEPlannerText.formatCoT(ACEPlanMetadata(bpm: 120, caption: cap, keyscale: "B♭ minor", language: "no", timeSignature: "4"))
        let want = "<think>\n\(ref)\n</think>"
        yamlN += 1
        if s == want { yamlOK += 1 } else { print("  YAML MISMATCH for \(cap.debugDescription):\n    swift \(s.debugDescription)\n    ref   \(want.debugDescription)") }
    }
    print("yaml dumps identical: \(yamlOK)/\(yamlN)")
    // DiT text prompt vs the reference text-encoder ids.
    let dump = try loadArrays(url: work.appendingPathComponent("think20/dump.safetensors"))
    let teTok = try await AutoTokenizer.from(modelFolder: models.appendingPathComponent("Qwen3-Embedding-0.6B"))
    let refTE = dump["te_ids0"]!.asType(.int32).asArray(Int32.self).map(Int.init)
    let plan = ACEPlan(metadata: meta, cotText: "", cot: cot, codes: refCodes)
    let cond = ACEDiTConditioning.make(plan: plan, request: r, hints: MLXArray.zeros([1, 500, 64]),
                                       silenceLatent: dump["silence_latent"]!)
    let teIDs = teTok.encode(text: cond.textPrompt)
    print("DiT text prompt ids identical to te_ids0: \(teIDs == refTE) (\(teIDs.count) vs \(refTE.count))")
    if teIDs != refTE { print("  ref decoded: \(teTok.decode(tokens: refTE).debugDescription)") }
    print("DiT lyrics prompt: \(cond.lyricsPrompt.debugDescription)")
}

// MARK: (c) logits for fixed prompts, (d) greedy, (e) sampled
if what.contains("logits") || what.contains("greedy") || what.contains("sample") || what.contains("user") || what.contains("long") {
    let P = try json("lm/prompts.json") as! [String: Any]
    let refIDs = P["ids"] as! [String: [Int]]
    let ref = try loadArrays(url: work.appendingPathComponent("lm/logits.safetensors"))
    let greedy = try json("lm/greedy.json") as! [String: Any]
    for dt in dtypes {
        let planner = try await ACEPlanner(directory: lmDir, dtype: dt)
        print("planner \(dt): tokenizer \(String(format: "%.2f", planner.tokenizerLoadSeconds)) s, model \(String(format: "%.2f", planner.modelLoadSeconds)) s")
        let m = planner.model
        if what.contains("logits") {
            let ids = refIDs["cot"]!
            let h = m(MLXArray(ids.map(Int32.init)).reshaped(1, -1))
            let l = m.logits(h[0, (ids.count - 8)...])
            report("cot logits last 8", l, ref["cot_logits_last8"]!)
            let a1 = l.argMax(axis: -1).asArray(Int32.self), a2 = ref["cot_logits_last8"]!.argMax(axis: -1).asArray(Int32.self)
            print("  argmax equal: \(a1 == a2)")
            // CFG batch: left-padded uncond row through the same masking the planner uses.
            let bi = (P["batch_ids"] as! [[Int]]).flatMap { $0.map(Int32.init) }
            let bm = (P["batch_mask"] as! [[Int]])
            let L = bm[0].count
            var add = [Float](repeating: 0, count: 2 * L * L)
            for b in 0..<2 { for i in 0..<L { for j in 0..<L where j > i || (bm[b][j] == 0 && j != i) { add[(b * L + i) * L + j] = -.infinity } } }
            let hb = m(MLXArray(bi, [2, L]), mask: MLXArray(add, [2, 1, L, L]))
            let lb = m.logits(hb[0..., -1])
            report("codes batch last (cond+uncond)", lb, ref["codes_batch_logits_last"]!)
            report("uncond padded vs alone(ref)", lb[1], ref["codes_uncond_alone_last"]!)
        }
        if what.contains("greedy") {
            var r = baseRequest()
            r.temperature = 0
            let t = Date()
            let plan = try planner.plan(r)
            let dtSec = Date().timeIntervalSince(t)
            let cap = greedy["captured"] as! [[String: Any]]
            let refCoT = cap[0]["gen"] as! [Int]
            let refCoTText = planner.decode(refCoT)
            let mineIDs = planner.encode(plan.cotText)
            var firstDiff = -1
            let a = Array(plan.cotText.unicodeScalars), b = Array(refCoTText.unicodeScalars)
            for i in 0..<min(a.count, b.count) where a[i] != b[i] { firstDiff = i; break }
            if firstDiff < 0 && a.count != b.count { firstDiff = min(a.count, b.count) }
            print("greedy \(dt) (\(String(format: "%.1f", dtSec)) s): CoT \(plan.cotTokens) tokens (ref \(refCoT.count)), text identical \(plan.cotText == refCoTText)\(firstDiff >= 0 ? ", first diff at char \(firstDiff)" : "")")
            if firstDiff >= 0 {
                print("  swift: \(String(String.UnicodeScalarView(a[max(0, firstDiff - 40)..<min(a.count, firstDiff + 60)])).debugDescription)")
                print("  ref:   \(String(String.UnicodeScalarView(b[max(0, firstDiff - 40)..<min(b.count, firstDiff + 60)])).debugDescription)")
            }
            _ = mineIDs
            let refG = parseCodes(greedy["audio_codes"] as! String)
            var match = 0
            while match < min(plan.codes.count, refG.count) && plan.codes[match] == refG[match] { match += 1 }
            let same = zip(plan.codes, refG).filter { $0 == $1 }.count
            print("  codes \(plan.codes.count) (ref \(refG.count)); identical prefix \(match), equal positions \(same)/\(refG.count)")
            print("  metadata: \(plan.metadata)")
            if plan.cotText != refCoTText {
                // Codes phase alone, conditioned on the reference CoT.
                let rm = greedy["metadata"] as! [String: Any]
                let meta = ACEPlanMetadata(bpm: rm["bpm"] as? Int, caption: rm["caption"] as? String, duration: rm["duration"] as? Int,
                                           keyscale: rm["keyscale"] as? String, language: rm["language"] as? String,
                                           timeSignature: rm["timesignature"] as? String)
                let c = try planner.codes(r, metadata: meta)
                var p = 0
                while p < min(c.count, refG.count) && c[p] == refG[p] { p += 1 }
                print("  codes phase with the reference CoT: identical prefix \(p), equal positions \(zip(c, refG).filter { $0 == $1 }.count)/\(refG.count)")
            }
        }
        if what.contains("sample") {
            for seed in [UInt64(42), 7] {
                let r = baseRequest(seed: seed)
                var lastPhase = ""
                let plan = try planner.plan(r) { phase, n, total in
                    let p = "\(phase)"
                    if p != lastPhase { lastPhase = p }
                }
                print("sampled \(dt) seed \(seed): CoT \(plan.cotTokens) tok in \(String(format: "%.2f", plan.cotSeconds)) s (\(String(format: "%.1f", Double(plan.cotTokens) / plan.cotSeconds)) tok/s), codes \(plan.codes.count) in \(String(format: "%.2f", plan.codesSeconds)) s (\(String(format: "%.1f", Double(plan.codes.count) / plan.codesSeconds)) tok/s, CFG batch 2)")
                print("  metadata: \(plan.metadata)")
                print("  codes[0..<10]: \(Array(plan.codes.prefix(10))) range \(plan.codes.min()!)…\(plan.codes.max()!) unique \(Set(plan.codes).count)")
                print("  cot block:\n\(plan.cot)")
            }
        }
        if what.contains("user") {
            // Greedy with user bpm + keyscale injected (reference: lm/greedy_user.py).
            let g = try json("lm/greedy_user.json") as! [String: Any]
            var r = ACEPlanRequest(caption: "dark ambient drone with distant choir", lyrics: "[Instrumental]", duration: 12,
                                   bpm: 72, keyscale: "F# minor", seed: 1)
            r.temperature = 0
            let plan = try planner.plan(r)
            let refCoT = planner.decode(((g["cot"] as! [[String: Any]])[0]["gen"] as! [Int]))
            let refG = parseCodes(g["audio_codes"] as! String)
            var p = 0
            while p < min(plan.codes.count, refG.count) && plan.codes[p] == refG[p] { p += 1 }
            print("user-metas greedy \(dt): CoT identical \(plan.cotText == refCoT) (\(plan.cotTokens) tokens), codes \(plan.codes.count) (ref \(refG.count)) identical prefix \(p)")
            print("  swift cot: \(plan.cotText.debugDescription)")
            if plan.cotText != refCoT { print("  ref cot:   \(refCoT.debugDescription)") }
            // All four metas given: phase 1 is skipped.
            var r2 = r
            r2.timeSignature = "3"
            let p2 = try planner.plan(r2)
            print("  all metas given: cotTokens \(p2.cotTokens), cot \(p2.cot.debugDescription), codes \(p2.codes.count)")
        }
        if what.contains("long") {
            for secs in [60.0, 180.0] {
                var r = baseRequest(seed: 3)
                r.duration = secs
                let plan = try planner.plan(r)
                print("long \(dt) \(Int(secs)) s: CoT \(plan.cotTokens) tok \(String(format: "%.2f", plan.cotSeconds)) s, codes \(plan.codes.count) in \(String(format: "%.2f", plan.codesSeconds)) s (\(String(format: "%.1f", Double(plan.codes.count) / plan.codesSeconds)) tok/s); duration in CoT \(plan.metadata.duration ?? -1), peak mem \(Memory.peakMemory / 1_000_000) MB")
            }
        }
    }
}
