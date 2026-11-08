import ACEStep
import Testing

/// The DiT's caption prompt matches the official handler's text byte for byte (taken from a reference run: 20 s,
/// no planner; the tokenizer then appends one more `<|endoftext|>`).
@Test func captionPromptMatchesTheReference() {
    let metas = ACEPrompts.metas(bpm: nil, timeSignature: "", keyScale: nil, seconds: 20)
    let prompt = ACEPrompts.caption(caption: "upbeat synthwave with punchy drums, warm analog bass and a catchy lead melody",
                                    metas: metas)
    #expect(prompt == "# Instruction\nFill the audio semantic mask based on the given conditions:\n\n# Caption\n"
        + "upbeat synthwave with punchy drums, warm analog bass and a catchy lead melody\n\n# Metas\n- bpm: N/A\n"
        + "- timesignature: N/A\n- keyscale: N/A\n- duration: 20 seconds\n<|endoftext|>\n")
    #expect(ACEPrompts.metas(bpm: 130, timeSignature: "4", keyScale: "D major", seconds: 20.4)
        == "- bpm: 130\n- timesignature: 4\n- keyscale: D major\n- duration: 20 seconds\n")
}

@Test func lyricPromptHasALanguageHeader() {
    #expect(ACEPrompts.lyrics("[Instrumental]", language: "") == "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>")
    #expect(ACEPrompts.lyrics("[Verse]\nla", language: "en") == "# Languages\nen\n\n# Lyric\n[Verse]\nla<|endoftext|>")
    #expect(ACEPrompts.normalizedLyrics("  \n") == "[Instrumental]")
}
