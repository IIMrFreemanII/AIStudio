import Foundation
import MLX

/// What the DiT receives after a thinking-mode planner run, reproducing `acestep/inference.py generate_music`
/// (`_update_metadata_from_lm`, caption/language override) and the handler's conditioning
/// (`_resolve_generate_music_task` → "cover" because audio codes are present, `prepare_batch_data` /
/// `_dict_to_meta_string`, `_prepare_target_latents_and_wavs`, `_build_chunk_masks_and_src_latents`,
/// `_prepare_precomputed_lm_hints`, `_prepare_text_conditioning_inputs`).
public struct ACEDiTConditioning {
    /// Caption inside the text prompt: the LM's rewrite when `useCotCaption`, else the user's.
    public var caption: String
    public var lyrics: String
    /// Vocal language of the lyric prompt. The official code reads `metadata["vocal_language"]`, a key the LM never
    /// writes, so this is always the user's language (default "unknown"); the LM's `language:` is not used.
    public var language: String
    public var bpm: Int?
    public var keyscale: String?
    public var timeSignature: String?
    public var duration: Double
    /// `- bpm: …\n- timesignature: …\n- keyscale: …\n- duration: N seconds\n`.
    public var metas: String
    /// The cover-task instruction (audio codes switch text2music to "cover").
    public var instruction: String
    /// `SFT_GEN_PROMPT` text for the caption encoder (tokenized with truncation at 256 tokens).
    public var textPrompt: String
    /// Lyric-encoder text (`# Languages\n{lang}\n\n# Lyric\n{lyrics}<|endoftext|>`, truncated at 2048 tokens).
    public var lyricsPrompt: String
    /// `is_covers` = true (code hints present).
    public var isCover: Bool
    /// Latent frames T = max(128, 5·codes) at 25 Hz; chunk masks are all ones over T.
    public var latentLength: Int
    /// `precomputed_lm_hints_25Hz` `[1, T, 64]`: decoded hints, silence-padded to T.
    public var lmHints: MLXArray
    /// `src_latents` `[1, T, 64]`: for a code-hinted item the target latents are the decoded hints (same tensor).
    public var srcLatents: MLXArray

    public static let coverInstruction = "Generate audio semantic tokens based on the given conditions:"

    /// `SFT_GEN_PROMPT.format(instruction, caption, metas)`.
    public static func textPrompt(instruction: String, caption: String, metas: String) -> String {
        let instr = instruction.hasSuffix(":") ? instruction : instruction + ":"
        return "# Instruction\n\(instr)\n\n# Caption\n\(caption)\n\n# Metas\n\(metas)<|endoftext|>\n"
    }

    /// `_format_lyrics`.
    public static func lyricsPrompt(lyrics: String, language: String) -> String {
        "# Languages\n\(language)\n\n# Lyric\n\(lyrics)<|endoftext|>"
    }

    /// `_dict_to_meta_string(_build_metadata_dict(bpm, key_scale, time_signature, duration))`.
    public static func metas(bpm: Int?, keyscale: String?, timeSignature: String?, duration: Double?) -> String {
        let b = bpm.map { $0 != 0 ? String($0) : "N/A" } ?? "N/A"
        let k = keyscale.flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0 } ?? "N/A"
        let t = timeSignature.flatMap {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || $0 == "N/A" ? nil : $0
        } ?? "N/A"
        let d = duration.map { "\(Int($0)) seconds" } ?? "30 seconds"
        return "- bpm: \(b)\n- timesignature: \(t)\n- keyscale: \(k)\n- duration: \(d)\n"
    }

    /// Silence latent frames `[n, 64]` (tiled when the stored latent is shorter; `_get_silence_latent_slice`).
    static func silence(_ silenceLatent: MLXArray, _ n: Int) -> MLXArray {
        let s = silenceLatent.ndim == 3 ? silenceLatent[0] : silenceLatent
        let avail = s.dim(0)
        if n <= avail { return s[..<n] }
        let reps = (n + avail - 1) / avail
        return tiled(s, repetitions: [reps, 1])[..<n]
    }

    /// Builds the DiT inputs.
    /// - Parameters:
    ///   - plan: the planner result (its `codes` must be non-empty).
    ///   - request: the request that produced it (user caption/lyrics/metas/language/duration).
    ///   - hints: `ACEAudioCodeDecoder.hints(codes: plan.codes)` `[1, 5·N, 64]`.
    ///   - silenceLatent: the DiT checkpoint's silence latent `[1, S, 64]` or `[S, 64]`.
    public static func make(plan: ACEPlan, request: ACEPlanRequest, hints: MLXArray,
                            silenceLatent: MLXArray) -> ACEDiTConditioning {
        let m = plan.metadata
        // _update_metadata_from_lm: user values win, LM fills the gaps.
        let bpm = request.bpm.flatMap { $0 > 0 ? $0 : nil } ?? m.bpm.flatMap { $0 > 0 ? $0 : nil }
        let userKey = request.keyscale.flatMap { $0.isEmpty ? nil : $0 }
        let keyscale = userKey ?? m.keyscale.flatMap { $0 == "N/A" || $0.isEmpty ? nil : $0 }
        let userTS = request.timeSignature.flatMap { $0.isEmpty ? nil : $0 }
        let timeSignature = userTS ?? m.timeSignature.flatMap { $0 == "N/A" || $0.isEmpty ? nil : $0 }
        var duration = request.duration
        if duration <= 0, let d = m.duration { duration = Double(d) }
        let caption = request.useCotCaption ? (m.caption ?? request.caption) : request.caption
        let language = request.language.flatMap { $0.isEmpty ? nil : $0 } ?? "unknown"
        let metas = metas(bpm: bpm, keyscale: keyscale, timeSignature: timeSignature, duration: duration)

        let frames = hints.dim(1)
        let T = max(128, frames)
        var h = hints.asType(.float32)
        if frames < T { h = concatenated([h, silence(silenceLatent, T - frames).asType(.float32)[.newAxis]], axis: 1) }
        return ACEDiTConditioning(
            caption: caption, lyrics: request.lyrics, language: language, bpm: bpm, keyscale: keyscale,
            timeSignature: timeSignature, duration: duration, metas: metas, instruction: coverInstruction,
            textPrompt: textPrompt(instruction: coverInstruction, caption: caption, metas: metas),
            lyricsPrompt: lyricsPrompt(lyrics: request.lyrics, language: language),
            isCover: true, latentLength: T, lmHints: h, srcLatents: h)
    }
}
