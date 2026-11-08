import Foundation
import Kokoro
import KokoroG2P
import MLX
import MLXDiffusionCore

/// Kokoro-82M text-to-speech, as the official `KPipeline` for English (lang codes "a" and "b"):
/// split the text at line breaks, phonemize each paragraph (pure-Swift misaki port; British for `b*` voices),
/// cut it into chunks of at most 510 phonemes at the best punctuation (`en_tokenize` / `waterfall_last`), speak each
/// chunk with the voice row picked by its phoneme count, and concatenate. Mono 24 kHz.
///
/// Folder layout (the catalog download of `hexgrad/Kokoro-82M`): `config.json`, `kokoro-v1_0.pth`, `voices/*.pt`,
/// and `mlxcomfy.json` with `"class": "KokoroPipeline"`.
public final class KokoroPipeline: AudioModel, @unchecked Sendable {
    public static let className = "KokoroPipeline"

    public static func register(in registry: ModelRegistry = .shared) {
        registry.register(className: className) { url in try await KokoroPipeline(directory: url) }
    }

    public var family: String { "kokoro-82m" }
    public var displayName: String { "Kokoro 82M (\(directory.lastPathComponent))" }

    public let directory: URL
    public let model: KokoroModel
    public static let maxPhonemes = KokoroModel.maxPhonemes
    var g2p: [Bool: EnglishG2P] = [:]
    let lock = NSLock()

    public init(directory: URL) async throws {
        self.directory = directory
        model = try KokoroModel(directory: directory)
    }

    /// Voices in this folder (`voices/*.pt`), sorted.
    public var installedVoices: [String] {
        let dir = directory.appendingPathComponent("voices")
        return ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".pt") }.map { String($0.dropLast(3)) }.sorted()
    }

    func phonemizer(british: Bool) throws -> EnglishG2P {
        if let g = g2p[british] { return g }
        let g = try EnglishG2P(british: british)
        g2p[british] = g
        return g
    }

    /// A piece of text and the phonemes Kokoro speaks for it (≤ 510 phonemes).
    public struct Chunk: Sendable {
        public var text: String
        public var phonemes: String
    }

    /// The chunks `text` is spoken in with `voice` (its first letter picks American or British pronunciation).
    public func chunks(_ text: String, voice: String) throws -> [Chunk] {
        let g2p = try phonemizer(british: voice.hasPrefix("b"))
        var out: [Chunk] = []
        // KPipeline splits on runs of newlines first.
        for paragraph in text.trimmingCharacters(in: .whitespacesAndNewlines)
            .components(separatedBy: .newlines) where !paragraph.trimmingCharacters(in: .whitespaces).isEmpty {
            let tokens = g2p.phonemize(paragraph).tokens
            out += Self.split(tokens).filter { !$0.phonemes.isEmpty }
        }
        return out
    }

    /// Python `len()` of a phoneme string: Unicode scalars, not grapheme clusters.
    static func count(_ s: String) -> Int { s.unicodeScalars.count }

    static func phonemes(_ tokens: ArraySlice<Token>) -> String {
        tokens.map { ($0.phonemes ?? "") + ($0.whitespace.isEmpty ? "" : " ") }.joined()
            .trimmingCharacters(in: .whitespaces)
    }

    static func text(_ tokens: ArraySlice<Token>) -> String {
        tokens.map { $0.text + $0.whitespace }.joined().trimmingCharacters(in: .whitespaces)
    }

    /// `KPipeline.waterfall_last`: the cut after the last sentence end, else clause break, else comma/dash, whose
    /// remainder still fits; otherwise everything.
    static func waterfallLast(_ tokens: ArraySlice<Token>, nextCount: Int) -> Int {
        let waterfall: [Set<String>] = [["!", ".", "?", "…"], [":", ";"], [",", "—"]]
        let bumps: Set<String> = [")", "”"]
        let base = tokens.startIndex
        for w in waterfall {
            guard var z = tokens.indices.last(where: { w.contains(tokens[$0].phonemes ?? "") }) else { continue }
            z += 1
            if z < tokens.endIndex, bumps.contains(tokens[z].phonemes ?? "") { z += 1 }
            if nextCount - count(phonemes(tokens[base..<z])) <= maxPhonemes { return z - base }
        }
        return tokens.count
    }

    /// `KPipeline.en_tokenize`.
    static func split(_ tokens: [Token]) -> [Chunk] {
        var out: [Chunk] = []
        var tks: [Token] = []
        var pcount = 0
        for var t in tokens {
            t.phonemes = t.phonemes ?? ""
            var next = t.phonemes! + (t.whitespace.isEmpty ? "" : " ")
            let nextCount = pcount + count(String(next.reversed().drop(while: { $0.isWhitespace }).reversed()))
            if nextCount > maxPhonemes {
                let z = waterfallLast(tks[...], nextCount: nextCount)
                out.append(Chunk(text: text(tks[..<z]), phonemes: phonemes(tks[..<z])))
                tks = Array(tks[z...])
                pcount = count(phonemes(tks[...]))
                if tks.isEmpty { next = String(next.drop(while: { $0.isWhitespace })) }
            }
            tks.append(t)
            pcount += count(next)
        }
        if !tks.isEmpty { out.append(Chunk(text: text(tks[...]), phonemes: phonemes(tks[...]))) }
        return out.map { c in
            // KPipeline truncates (with a warning) if a chunk is still too long, e.g. one enormous word.
            count(c.phonemes) > maxPhonemes ? Chunk(text: c.text, phonemes: String(String.UnicodeScalarView(c.phonemes.unicodeScalars.prefix(maxPhonemes)))) : c
        }
    }

    public func generateAudio(_ request: AudioRequest, onStep: ((Int, Int) throws -> Void)?) throws -> AudioClip {
        lock.lock(); defer { lock.unlock() }
        let voice = request.voice.isEmpty ? "af_heart" : request.voice
        let pack = try model.loadVoice(voice)
        let pieces = try chunks(request.prompt, voice: voice)
        let speed = Float(min(max(request.speed, 0.5), 2))
        var audio: [MLXArray] = []
        for (i, c) in pieces.enumerated() {
            try Task.checkCancellation()
            try onStep?(i, pieces.count)
            let (a, _) = model(tokens: model.tokens(c.phonemes), refS: KokoroModel.refS(pack, phonemes: c.phonemes),
                               speed: speed, seed: request.seed &+ UInt64(i))
            eval(a)
            audio.append(a)
        }
        try onStep?(pieces.count, pieces.count)
        let samples = audio.isEmpty ? MLXArray.zeros([1, KokoroModel.sampleRate / 4]) : concatenated(audio, axis: 0).reshaped(1, -1)
        Memory.clearCache()
        return AudioClip(samples: samples, sampleRate: KokoroModel.sampleRate)
    }
}
