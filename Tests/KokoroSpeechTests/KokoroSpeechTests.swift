import KokoroG2P
@testable import KokoroSpeech
import Testing

/// Long text is cut like `KPipeline.en_tokenize`: no chunk over 510 phonemes, and cuts fall after sentence ends.
@Test func longTextIsChunkedAtSentenceEnds() throws {
    let g2p = try EnglishG2P()
    let sentence = "The committee reviewed every proposal carefully before announcing its final decision to the public. "
    let text = String(repeating: sentence, count: 12)
    let chunks = KokoroPipeline.split(g2p.phonemize(text).tokens)
    #expect(chunks.count > 1)
    for c in chunks {
        #expect(c.phonemes.unicodeScalars.count <= 510)
        #expect(c.phonemes.hasSuffix("."))
    }
    // Nothing is lost: the chunks' words add back up to the input's.
    let words = chunks.map { $0.text.split(separator: " ").count }.reduce(0, +)
    #expect(words == text.split(separator: " ").count)
}

@Test func shortTextIsOneChunk() throws {
    let g2p = try EnglishG2P()
    let chunks = KokoroPipeline.split(g2p.phonemize("Hello there, world!").tokens)
    #expect(chunks.count == 1 && chunks[0].text == "Hello there, world!")
}
