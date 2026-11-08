import Foundation
import KokoroG2P

// kokoro-probe-g2p — development probe for the KokoroG2P module.
//
//   kokoro-probe-g2p <sentences.txt> [--gb] [--no-fallback] [--unk X]
//       Phonemize one sentence per line; print JSON lines {text, ps, tokens:[[text,tag,ws,phonemes,fallback]]}.
//   kokoro-probe-g2p --lts <words.txt> [--gb] [--mode full|exclude|rules]
//       Print the OOV fallback pronunciation of one word per line as JSON lines {word, ps}.
//   kokoro-probe-g2p --bench <sentences.txt> [--gb] [--repeat N]
//       Lexicon load time and phonemize throughput.
let args = CommandLine.arguments
func flag(_ f: String) -> Bool { args.contains(f) }
func value(_ f: String) -> String? {
    guard let i = args.firstIndex(of: f), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func readLines(_ path: String) throws -> [String] {
    try String(contentsOfFile: path, encoding: .utf8).split(separator: "\n", omittingEmptySubsequences: true)
        .map(String.init).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
}
func json(_ v: some Encodable) -> String {
    let enc = JSONEncoder()
    enc.outputFormatting = [.withoutEscapingSlashes]
    return String(data: try! enc.encode(v), encoding: .utf8)!
}
func stderr(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

let british = flag("--gb")

if let path = value("--lts") {
    let g2p = try EnglishG2P(british: british)
    let mode: EnglishG2P.FallbackMode = switch value("--mode") ?? "full" {
    case "exclude": .excludingWord
    case "rules": .rulesOnly
    default: .full
    }
    struct Rec: Encodable { let word: String; let ps: String? }
    var out = ""
    for w in try readLines(path) { out += json(Rec(word: w, ps: g2p.fallbackPhonemes(for: w, mode: mode))) + "\n" }
    FileHandle.standardOutput.write(out.data(using: .utf8)!)
    exit(0)
}

if let path = value("--bench") {
    let lines = try readLines(path)
    let repeatN = Int(value("--repeat") ?? "5") ?? 5
    var t = Date()
    let g2p = try EnglishG2P(british: british)
    let tInit = Date().timeIntervalSince(t)
    t = Date()
    g2p.preload()
    let tPre = Date().timeIntervalSince(t)
    t = Date()
    let g2p2 = try EnglishG2P(british: british)
    g2p2.preload()
    let tSecond = Date().timeIntervalSince(t)
    _ = g2p.phonemize(lines[0])  // warm up NLTagger
    t = Date()
    var chars = 0, words = 0
    for _ in 0..<repeatN {
        for l in lines {
            let (_, toks) = g2p.phonemize(l)
            chars += l.count
            words += toks.count
        }
    }
    let dt = Date().timeIntervalSince(t)
    print(String(format: "init (gold) %.1f ms, preload (silver+rest) %.1f ms, second instance %.2f ms", tInit * 1000, tPre * 1000, tSecond * 1000))
    print(String(format: "%d sentences x %d: %.1f ms total, %.3f ms/sentence, %.0f tokens/s, %.0f chars/s",
                 lines.count, repeatN, dt * 1000, dt * 1000 / Double(lines.count * repeatN), Double(words) / dt, Double(chars) / dt))
    exit(0)
}

guard args.count >= 2 else {
    print("usage: kokoro-probe-g2p <sentences.txt> [--gb] [--no-fallback] [--unk X] | --lts <words.txt> [--mode full|exclude|rules] | --bench <sentences.txt>")
    exit(1)
}
let t0 = Date()
let g2p = try EnglishG2P(british: british, unknownMarker: value("--unk") ?? "", useFallback: !flag("--no-fallback"))
let tLoad = Date().timeIntervalSince(t0)
let lines = try readLines(args[1])
struct Rec: Encodable { let text: String; let ps: String; let kokoro: String; let tokens: [[String?]] }
let t1 = Date()
var out = ""
for line in lines {
    let (ps, toks) = g2p.phonemize(line)
    let rec = Rec(text: line, ps: ps, kokoro: EnglishG2P.pipelinePhonemes(toks),
                  tokens: toks.map { [$0.text, $0.tag, $0.whitespace, $0.phonemes, $0.isFallback ? "1" : "0"] })
    out += json(rec) + "\n"
}
let tRun = Date().timeIntervalSince(t1)
FileHandle.standardOutput.write(out.data(using: .utf8)!)
stderr(String(format: "load %.1f ms, %d lines in %.1f ms", tLoad * 1000, lines.count, tRun * 1000))
