import Foundation
import LLMClient
import MovieCore

/// The prompt engine's pure side: the rule book (docs/prompting.md, bundled), checks on a written prompt, the shapes
/// the agent reports in (checklist, tries, verdicts, the result), and its system prompt. The tools that call these
/// live in the app (`PromptTools`), shared by the headless engine and the Studio agent.
public enum PromptEngine {
    // MARK: the rule book

    /// docs/prompting.md as shipped in the app ("" when the resource is missing).
    public static let rulesText: String = {
        guard let url = Bundle.module.url(forResource: "prompting", withExtension: "md"),
              let s = try? String(contentsOf: url, encoding: .utf8) else { return "" }
        return s
    }()

    /// The heading of `model`'s section in the rule book.
    public static func heading(_ model: PromptModel) -> String {
        switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious: "### SDXL"
        case .sd35: "### Stable Diffusion 3.5"
        case .flux: "### FLUX.1 dev"
        case .chroma: "### Chroma"
        case .anima: "### Anima"
        case .qwenImage: "### Qwen-Image\n"
        case .minimaxImage: "### MiniMax H3 Image"
        case .wan: "### Wan 2.2 T2V"
        case .ltx23: "### LTX-2.3"
        case .minimaxVideo: "### MiniMax H3 video"
        }
    }

    /// The text from the line starting with `heading` up to the next heading of the same or a higher level.
    public static func section(_ heading: String, in doc: String) -> String? {
        let level = heading.prefix(while: { $0 == "#" }).count
        let lines = doc.components(separatedBy: "\n")
        let want = heading.hasSuffix("\n") ? String(heading.dropLast()) : heading
        guard let start = lines.firstIndex(where: { heading.hasSuffix("\n") ? $0 == want : $0.hasPrefix(want) })
        else { return nil }
        var end = lines.count
        for i in (start + 1)..<lines.count {
            let l = lines[i]
            let hashes = l.prefix(while: { $0 == "#" }).count
            if hashes > 0, hashes <= level, l.dropFirst(hashes).hasPrefix(" ") { end = i; break }
        }
        return lines[start..<end].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The quick-reference table's row for `model` (with the header), nil when it has none.
    public static func quickReference(_ model: PromptModel, in doc: String) -> String? {
        guard let table = section("## Quick reference", in: doc) else { return nil }
        let rows = table.components(separatedBy: "\n").filter { $0.hasPrefix("|") }
        guard rows.count > 2 else { return nil }
        let key: String = switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious: "| SDXL"
        case .sd35: "| SD 3.5"
        case .flux: "| FLUX"
        case .chroma: "| Chroma"
        case .anima: "| Anima"
        case .qwenImage: "| Qwen-Image |"
        case .minimaxImage: "| MiniMax H3 Image"
        case .wan: "| Wan 2.2 T2V"
        case .ltx23: "| LTX-2.3"
        case .minimaxVideo: "| MiniMax H3 video"
        }
        guard let row = rows.first(where: { $0.hasPrefix(key) }) else { return nil }
        return rows[0] + "\n" + rows[1] + "\n" + row
    }

    /// What the agent reads before writing for `model`: its quick-reference row, its section, and the writers' rules.
    public static func rules(for model: PromptModel, doc: String = rulesText) -> String {
        var parts = ["Rules for \(model.title) (\(model.rawValue)), from docs/prompting.md."]
        if let q = quickReference(model, in: doc) { parts.append(q) }
        if let s = section(heading(model), in: doc) { parts.append(s) }
        if let w = section("## Rules for the app's prompt writers", in: doc) { parts.append(w) }
        return parts.joined(separator: "\n\n")
    }

    /// The basics every prompt follows (token limits, tags vs sentences, negatives, weights, trigger words).
    public static func basics(doc: String = rulesText) -> String { section("## The basics", in: doc) ?? "" }

    // MARK: checks

    /// About how many CLIP tokens `text` is: words of up to 6 letters count one, longer ones a token per 6 letters,
    /// every digit and punctuation mark one. Close enough to warn about the first 75-token chunk.
    public static func clipTokens(_ text: String) -> Int {
        var n = 0
        var word = 0
        func flush() { if word > 0 { n += (word + 5) / 6; word = 0 } }
        for ch in text {
            if ch.isLetter { word += 1; continue }
            flush()
            if ch.isWhitespace { continue }
            n += 1
        }
        flush()
        return n
    }

    /// About how many tokens a T5 or LLM encoder reads: 1.3 per word.
    public static func wordTokens(_ text: String) -> Int {
        Int((Double(text.split(whereSeparator: \.isWhitespace).count) * 1.3).rounded(.up))
    }

    /// The encoder limit we send to, in tokens (nil: not published).
    public static func limit(_ model: PromptModel) -> Int? {
        switch model {
        case .sdxl, .sdxlPony, .sdxlIllustrious: 75
        case .sd35: 256
        case .flux, .chroma, .anima, .qwenImage, .wan: 512
        case .ltx23: 1024
        case .minimaxImage: 4096
        case .minimaxVideo: nil
        }
    }

    /// Warnings about a prompt for `model`: past the limit (or SDXL's first chunk), a negative the model or its fast
    /// mode ignores, weights on a model that reads them as text, LoRA trigger words missing, LTX past 200 words.
    public static func warnings(prompt: String, negative: String, model: PromptModel, fast: Bool = false,
                                triggerWords: [String] = []) -> [String] {
        var out: [String] = []
        let tags = model.dialect == .tags
        let tokens = tags ? clipTokens(prompt) : wordTokens(prompt)
        if let limit = limit(model), tokens > limit {
            out.append(tags
                ? "About \(tokens) CLIP tokens: the server reads past 75 in later chunks, which weigh less. Keep the subject, framing, action and place in the first 75."
                : "About \(tokens) tokens, past this model's \(limit): the rest is cut.")
        }
        if model == .ltx23, prompt.split(whereSeparator: \.isWhitespace).count > 200 {
            out.append("Over 200 words: LTX-2.3's guide advises 200 or fewer.")
        }
        let hasNegative = !negative.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasNegative, !model.takesNegative {
            out.append("\(model.title) reads no negative prompt: write exclusions as plain sentences in the prompt instead.")
        } else if hasNegative, fast, PromptPreview.fastModels.contains(model) {
            out.append("The fast mode skips guidance, so the negative is ignored: draft with fast off to use it.")
        }
        if !model.takesWeights, prompt.range(of: #"\([^()]+:\s*\d(\.\d+)?\)"#, options: .regularExpression) != nil {
            out.append("\(model.title) reads (word:1.3) weights as plain text: drop the brackets and numbers.")
        }
        let lower = prompt.lowercased()
        let missing = triggerWords.filter { !$0.isEmpty && !lower.contains($0.lowercased()) }
        if !missing.isEmpty { out.append("LoRA trigger words missing: \(missing.joined(separator: ", ")).") }
        return out
    }

    /// Why a prompt can't be drawn (a youth term), else nil. The negative isn't checked: `AgeGuard.negative` lives there.
    public static func refusal(_ texts: [String]) -> String? {
        for t in texts { if let term = AgeGuard.youthTerm(in: t) { return "it describes someone under 18 (“\(term)”)" } }
        return nil
    }

    /// The agent's `spec` argument as a `PromptSpec`, with the adult guard applied: a person's subject states an adult
    /// age, and `AgeGuard.negative` is kept out. Throws when the JSON doesn't fit or a field describes a minor.
    public static func spec(from json: JSONValue) throws -> PromptSpec {
        let data = try JSONEncoder().encode(json)
        var s = try JSONDecoder().decode(PromptSpec.self, from: data)
        let fields = [s.subject, s.action, s.place, s.text, s.medium, s.shot]
            + (s.details + s.body + s.outfit).flatMap { [$0.tag, $0.text] }
            + s.time + s.camera + s.lighting + s.mood + s.composition
        if let why = refusal(fields) { throw PromptEngineError.refused(why) }
        if s.pronoun != .it, !s.subject.lowercased().contains("adult") {
            // "a 30-year-old woman" → "a 30-year-old adult woman"; no age → "an adult woman".
            if s.subject.range(of: #"\d+-year-old "#, options: .regularExpression) != nil {
                s.subject = s.subject.replacingOccurrences(of: #"(\d+-year-old) "#, with: "$1 adult ", options: .regularExpression)
            } else {
                s.subject = "an adult " + s.subject.replacingOccurrences(of: #"^(a|an)\s+"#, with: "", options: .regularExpression)
            }
        }
        for term in AgeGuard.negative.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) })
            where !s.avoid.contains(term) {
            s.avoid.append(term)
        }
        return s
    }

    /// A raw prompt's negative with `AgeGuard.negative` in it (for models that read one).
    public static func guardedNegative(_ negative: String, model: PromptModel) -> String {
        guard model.takesNegative else { return "" }
        let n = negative.trimmingCharacters(in: .whitespacesAndNewlines)
        if n.lowercased().contains("underage") { return n }
        return n.isEmpty ? AgeGuard.negative : n + ", " + AgeGuard.negative
    }

    // MARK: the system prompt

    /// The engine's instructions. `fixed`: the model the result must use (a node's target); `character`: a saved
    /// character drafts use (face and LoRAs); `studio`: the Studio agent's variant, which only adds these tools to its
    /// own instructions.
    public static func systemPrompt(video: Bool, fixed: PromptModel?, character: String?) -> String {
        var s = """
        You are AIStudio's prompt engine. You turn the user's idea into a \(video ? "video clip" : "picture") that \
        shows exactly what they asked for, by writing the prompt the way the chosen model reads it, drafting, looking \
        at the draft and fixing what's wrong.

        How to work:
        1. Call set_checklist with what a passing result must show: framing, how many people, pose, place, light, \
        outfit, anything the idea insists on. Each item is something you can see in a picture.
        2. \(fixed.map { "The model is fixed: \($0.rawValue) (\($0.title))." } ?? "Call draft_models, and pick the installed model that fits the idea best (style, price, what it does well). You may try a second model when the first one can't do it.")
        3. Call model_rules for the model, and follow its rules.
        4. Call write_prompt (a spec, or a raw prompt when the rules call for wording the spec can't express). Read \
        its warnings and fix them before drafting.
        5. Call draft. The user sees a cost sheet and may decline it.
        6. Look at the draft carefully, item by item, and call verdict with what passes, what fails and the one change \
        you'll make next. Prefer one change per try (the framing words, a weight, the negative, the seed, a LoRA, the \
        model), so each try shows what that change does.
        7. Repeat 4–6 until every checklist item passes, then call finish with the best try. If the user declines a \
        cost sheet, call finish at once with the best try so far.

        Rules you never break:
        - Every person is an adult of 18 or more, and looks it: state the adult age in the prompt. Fail any draft \
        where someone looks under 18, and fix it.
        - Never name or draw a real person, or ask for a real person's likeness.
        - Explicit adult content is allowed when the idea asks for it; never add it when it doesn't.
        - Describe what is visible. Labels about the picture ("masterpiece", "glamour photo") only where the model's \
        rules ask for them.
        """
        if let character { s += "\n- Drafts draw \(character), with their face reference and LoRAs: don't describe a different face." }
        if video {
            s += "\n\nVideo: draft the first frame as a still first (it's cheap), and render the clip only once the still passes."
        }
        s += "\n\nThe basics of prompting, from docs/prompting.md:\n\n" + basics()
        return s
    }

    /// What the Studio agent adds to its instructions with the prompt tools.
    public static let studioNote = """
    Prompt tools: before writing a prompt for an image or video node, call model_rules for its model and write the \
    prompt with write_prompt, which follows the model's rules and checks the age guard and the token limit. To get a \
    picture right, set_checklist, draft (with its own cost sheet), look at it, verdict, and draft again with one change.
    """
}

public enum PromptEngineError: Error, CustomStringConvertible, Equatable {
    case refused(String)
    public var description: String {
        switch self {
        case .refused(let why): "Refused: \(why)."
        }
    }
}

// MARK: what the engine reports

/// A LoRA one try draws with.
public struct EngineLoRA: Codable, Equatable, Hashable, Sendable {
    public var id: String
    public var strength: Double
    public var triggerWords: [String]

    public init(id: String, strength: Double = 0.8, triggerWords: [String] = []) {
        self.id = id
        self.strength = strength
        self.triggerWords = triggerWords
    }
}

/// One draft: the settings, the file it made and what it cost.
public struct EngineTry: Codable, Equatable, Identifiable, Sendable {
    public var n: Int
    public var model: PromptModel
    public var prompt: String
    public var negative: String
    public var loras: [EngineLoRA]
    public var seed: Int
    public var aspect: PromptPreview.Aspect
    public var fast: Bool
    /// The library fine-tune ("" = the base model).
    public var variant: String
    /// The picture or clip (an absolute path).
    public var file: String
    public var costUSD: Double?
    public var seconds: Double?
    public var id: Int { n }

    public init(n: Int, model: PromptModel, prompt: String, negative: String, loras: [EngineLoRA] = [], seed: Int,
                aspect: PromptPreview.Aspect = .portrait, fast: Bool = false, variant: String = "", file: String,
                costUSD: Double? = nil, seconds: Double? = nil) {
        self.n = n
        self.model = model
        self.prompt = prompt
        self.negative = negative
        self.loras = loras
        self.seed = seed
        self.aspect = aspect
        self.fast = fast
        self.variant = variant
        self.file = file
        self.costUSD = costUSD
        self.seconds = seconds
    }
}

/// The agent's judgement of one try against the checklist.
public struct EngineVerdict: Codable, Equatable, Sendable {
    public var attempt: Int
    public var pass: Bool
    public var issues: [String]
    /// What it will change next ("" when it passed).
    public var change: String

    public init(attempt: Int, pass: Bool, issues: [String] = [], change: String = "") {
        self.attempt = attempt
        self.pass = pass
        self.issues = issues
        self.change = change
    }
}

/// How a run ended: the chosen try, the others, and what everything cost.
public struct EngineResult: Codable, Equatable, Sendable {
    public var best: EngineTry?
    public var notes: String
    public var checklist: [String]
    public var tries: [EngineTry]
    public var verdicts: [EngineVerdict]
    public var gpuUSD: Double
    public var llmUSD: Double

    public init(best: EngineTry?, notes: String, checklist: [String], tries: [EngineTry], verdicts: [EngineVerdict],
                gpuUSD: Double, llmUSD: Double) {
        self.best = best
        self.notes = notes
        self.checklist = checklist
        self.tries = tries
        self.verdicts = verdicts
        self.gpuUSD = gpuUSD
        self.llmUSD = llmUSD
    }
}

/// Which LLM the prompt engine runs on, from the installed vision models ("server/model" keys).
public enum PromptEngineLLM {
    /// The engine's own model: the abliterated Qwen3.8-27B, its NVFP4 copy first (≈ 1.7× the decode speed on the
    /// same H100, 2026-11-28).
    public static let preferred = ["qwen3.8-27b-abliterated-nvfp4", "qwen3.8-27b-abliterated"]
    /// Picks saved before a faster copy existed, moved to it once (``UpgradeKey`` remembers that it ran).
    public static let upgrades = ["qwen3.8-27b-abliterated": "qwen3.8-27b-abliterated-nvfp4"]

    /// The model to use and, when a saved pick was moved to its faster copy, the new pick to save.
    public static func pick(installed: [String], saved: String?, upgraded: Bool) -> (model: String?, save: String?) {
        func model(_ key: String) -> String { String(key.split(separator: "/", maxSplits: 1).last ?? "") }
        if let saved, installed.contains(saved) {
            if !upgraded, let to = upgrades[model(saved)] {
                let server = saved.split(separator: "/", maxSplits: 1).first.map(String.init) ?? ""
                if installed.contains("\(server)/\(to)") { return ("\(server)/\(to)", "\(server)/\(to)") }
            }
            return (saved, nil)
        }
        for id in preferred {
            if let k = installed.first(where: { model($0) == id }) { return (k, nil) }
        }
        return (installed.first, nil)
    }
}
