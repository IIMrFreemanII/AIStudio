import Agent
import CharacterCore
import Foundation
import LLMClient
import MovieCore
import NodesRemote

/// What the prompt tools record and show while an agent works: the checklist, the drafts, the verdicts and the end.
/// The headless engine shows it in the Prompt Builder; the Studio agent keeps one per reply.
@MainActor @Observable
final class PromptEngineRecord {
    var checklist: [String] = []
    var tries: [EngineTry] = []
    var verdicts: [EngineVerdict] = []
    var result: EngineResult?
    /// What's happening now ("Drafting try 2…").
    var status = ""
    var gpuUSD = 0.0
    var llmUSD = 0.0

    func verdict(_ n: Int) -> EngineVerdict? { verdicts.last { $0.attempt == n } }

    func finish(best n: Int?, notes: String) {
        let best = n.flatMap { k in tries.first { $0.n == k } } ?? tries.last
        result = EngineResult(best: best, notes: notes, checklist: checklist, tries: tries, verdicts: verdicts,
                              gpuUSD: gpuUSD, llmUSD: llmUSD)
    }
}

/// Where the prompt tools run: the app, the record, the model the result must use, the character drafts draw, and
/// the cost sheet (each host shows it its own way and says whether the user pressed Run).
@MainActor
protocol PromptToolHost: AnyObject {
    var app: AppModel? { get }
    var record: PromptEngineRecord { get }
    /// The model every draft must use (a node's target), nil: the agent picks.
    var fixedModel: PromptModel? { get }
    /// The saved character drafts draw (their face and LoRAs).
    var characterID: UUID? { get }
    /// Shows the cost sheet and waits: true when the user pressed Run.
    func approve(_ estimate: CostEstimate) async throws -> Bool
}

/// The prompt tools: the rule book, writing and checking a prompt, drafting it on the GPU behind a cost sheet, and
/// recording the checklist, verdicts and the end. Shared by the headless prompt engine and the Studio agent.
enum PromptTools {
    static let names: Set<String> = ["set_checklist", "draft_models", "model_rules", "write_prompt", "draft", "verdict",
                                     "finish"]

    /// `finish`: only the engine ends with it (the Studio agent ends its reply instead).
    static func tools(_ host: PromptToolHost, finish: Bool = true) -> [any AgentTool] {
        let models: LLMClient.JSONValue = .object(["type": .string("string"),
                                                   "enum": .array(PromptModel.allCases.map { .string($0.rawValue) })])
        let loras: LLMClient.JSONValue = .object([
            "type": .string("array"),
            "description": .string("Library LoRAs, up to \(PromptPreview.maxLoRAs): their trigger words go first in the prompt."),
            "items": .object(["type": .string("object"), "properties": .object([
                "id": .object(["type": .string("string")]),
                "strength": .object(["type": .string("number"), "description": .string("0–1.5, default 0.8")]),
            ]), "required": .array([.string("id")])]),
        ])
        func obj(_ props: [String: LLMClient.JSONValue], _ required: [String]) -> LLMClient.JSONValue {
            .object(["type": .string("object"), "properties": .object(props), "required": .array(required.map { .string($0) })])
        }
        func str(_ d: String) -> LLMClient.JSONValue { .object(["type": .string("string"), "description": .string(d)]) }

        var out: [any AgentTool] = [
            HelperTool(name: "set_checklist", effect: .read,
                       description: "Sets what a passing result must show: visible things, one per item (\"whole body in frame, head to toe, shoes visible\", \"one person\", \"standing\", \"sunlit kitchen\"). Call it first; call it again to change the list.",
                       parameters: obj(["items": .object(["type": .string("array"), "items": .object(["type": .string("string")])])], ["items"]),
                       preview: { _ in ToolPreview(kind: .read, title: "Setting the checklist") },
                       run: { args in
                           let items = (args.raw["items"] as? [Any])?.compactMap { ($0 as? String)?.trimmingCharacters(in: .whitespaces) }
                               .filter { !$0.isEmpty } ?? []
                           guard !items.isEmpty else { throw ToolError("give at least one item") }
                           await MainActor.run { host.record.checklist = items }
                           return ToolOutput("Checklist set: \(items.count) items.")
                       }),
            HelperTool(name: "draft_models", effect: .read,
                       description: "The installed picture (or video) models on the GPU server: price per draft, how long one takes, how each reads a prompt, and the library fine-tunes and LoRAs (with trigger words) that fit each.",
                       parameters: obj(["kind": .object(["type": .string("string"), "enum": .array([.string("image"), .string("video")])])], []),
                       preview: { _ in ToolPreview(kind: .read, title: "Listing the models") },
                       run: { args in
                           let video = args.string("kind") == "video"
                           return ToolOutput(try await MainActor.run { try modelList(host, video: video) })
                       }),
            HelperTool(name: "model_rules", effect: .read,
                       description: "How a model reads a prompt: its section of the app's prompting rules (encoder and limit, tags or sentences, negatives, weights, what worked and what didn't). Read it before writing for that model.",
                       parameters: obj(["model": models], ["model"]),
                       preview: { a in ToolPreview(kind: .read, title: "Reading the rules for \(a.string("model") ?? "a model")") },
                       run: { args in
                           let m = try model(args)
                           return ToolOutput(PromptEngine.rules(for: m))
                       }),
            HelperTool(name: "write_prompt", effect: .read,
                       description: """
                       Writes the prompt for a model. Give `spec`, the description as fields, and the app's writer lays it out the way the model reads it (tags, sentences, a paragraph); or give `prompt` (and `negative`) written yourself when the rules call for wording the spec can't hold. Returns the prompt, the negative, its length and warnings. Spec fields: medium, subject (with count and adult age: "a 30-year-old adult woman"), pronoun (she/he/they/it), shot (framing), action, details/body/outfit (arrays of {tag, text, weight}), place, time, camera, lighting, mood, palette, composition (arrays of strings), text, avoid (the negative), without (exclusions as words, for models with no negative), triggerWords.
                       """,
                       parameters: obj(["model": models, "spec": .object(["type": .string("object")]),
                                        "prompt": str("A prompt written yourself instead of a spec."),
                                        "negative": str("With `prompt`: the negative."), "loras": loras,
                                        "fast": .object(["type": .string("boolean"), "description": .string("The draft will use the fast mode.")])],
                                       ["model"]),
                       preview: { a in ToolPreview(kind: .read, title: "Writing the prompt for \(a.string("model") ?? "a model")") },
                       run: { args in
                           let m = try model(args)
                           try await MainActor.run { try checkFixed(host, m) }
                           let words = await MainActor.run { triggerWords(host, loraArgs(args)) }
                           let written = try write(args, model: m, triggerWords: words)
                           let warnings = PromptEngine.warnings(prompt: written.prompt, negative: written.negative, model: m,
                                                                fast: args.bool("fast"), triggerWords: words)
                           let length = m.dialect == .tags ? "about \(PromptEngine.clipTokens(written.prompt)) CLIP tokens"
                               : "\(written.prompt.split(whereSeparator: \.isWhitespace).count) words"
                           var text = "Prompt (\(length)):\n\(written.prompt)"
                           text += m.takesNegative ? "\n\nNegative:\n\(written.negative)" : "\n\n(\(m.title) reads no negative.)"
                           text += warnings.isEmpty ? "\n\nNo warnings." : "\n\nWarnings:\n" + warnings.map { "- \($0)" }.joined(separator: "\n")
                           return ToolOutput(text)
                       }),
            HelperTool(name: "draft", effect: .studio,
                       description: "Draws one picture of a written prompt on the GPU server, so you can look at it. The user sees a cost sheet first and may decline. Returns the picture. Pass the prompt and negative as write_prompt returned them.",
                       parameters: obj(["model": models, "prompt": str("The prompt."), "negative": str("The negative (\"\" for none)."),
                                        "loras": loras,
                                        "aspect": .object(["type": .string("string"), "enum": .array(PromptPreview.Aspect.allCases.map { .string($0.rawValue) })]),
                                        "seed": .object(["type": .string("integer"), "description": .string("Keep it to see what a prompt change does; change it to see variety.")]),
                                        "fast": .object(["type": .string("boolean"), "description": .string("Chroma, SD 3.5, Qwen-Image: the fast mode (ignores the negative).")]),
                                        "variant": str("A library fine-tune id from draft_models (\"\" = the base model).")],
                                       ["model", "prompt"]),
                       preview: { a in ToolPreview(kind: .run, title: "Drafting on \(a.string("model") ?? "a model")") },
                       run: { args in try await draft(host, args) }),
            HelperTool(name: "verdict", effect: .read,
                       description: "Records your judgement of a draft against the checklist: pass or not, the items that fail, and the one change you'll make next.",
                       parameters: obj(["attempt": .object(["type": .string("integer")]), "pass": .object(["type": .string("boolean")]),
                                        "issues": .object(["type": .string("array"), "items": .object(["type": .string("string")])]),
                                        "change": str("What you'll change next (\"\" when it passed).")],
                                       ["attempt", "pass"]),
                       preview: { a in ToolPreview(kind: .read, title: "Judging try \(a.int("attempt") ?? 0)") },
                       run: { args in
                           guard let n = args.int("attempt") else { throw ToolError("missing \"attempt\"") }
                           let issues = (args.raw["issues"] as? [Any])?.compactMap { $0 as? String } ?? []
                           let v = EngineVerdict(attempt: n, pass: args.bool("pass"), issues: issues, change: args.string("change") ?? "")
                           try await MainActor.run {
                               guard host.record.tries.contains(where: { $0.n == n }) else { throw ToolError("there's no try \(n)") }
                               host.record.verdicts.append(v)
                           }
                           return ToolOutput(v.pass ? "Try \(n) passes." : "Try \(n) noted: \(issues.count) issues.")
                       }),
        ]
        if finish {
            out.append(HelperTool(name: "finish", effect: .read,
                                  description: "Ends the run with the best try (the one that passes, or the closest). Notes: what worked, in a sentence or two.",
                                  parameters: obj(["best": .object(["type": .string("integer")]), "notes": str("What worked.")], ["best"]),
                                  preview: { _ in ToolPreview(kind: .read, title: "Finishing") },
                                  run: { args in
                                      let n = args.int("best")
                                      await MainActor.run { host.record.finish(best: n, notes: args.string("notes") ?? "") }
                                      return ToolOutput("Finished. Reply with one line saying which try you chose and why; no more tools.")
                                  }))
        }
        return out
    }

    // MARK: the pieces

    static func model(_ args: ToolArgs) throws -> PromptModel {
        guard let s = args.string("model"), let m = PromptModel(rawValue: s) else {
            throw ToolError("\"model\" must be one of: \(PromptModel.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        return m
    }

    @MainActor
    static func checkFixed(_ host: PromptToolHost, _ m: PromptModel) throws {
        if let f = host.fixedModel, f != m { throw ToolError("the model is fixed to \(f.rawValue) for this run") }
    }

    static func loraArgs(_ args: ToolArgs) -> [(id: String, strength: Double)] {
        ((args.raw["loras"] as? [Any]) ?? []).compactMap { any in
            guard let d = any as? [String: Any], let id = d["id"] as? String, !id.isEmpty else { return nil }
            let s = (d["strength"] as? Double) ?? (d["strength"] as? Int).map(Double.init) ?? 0.8
            return (id, min(max(s, 0), 1.5))
        }
    }

    /// The picked LoRAs' trigger words, from the library.
    @MainActor
    static func triggerWords(_ host: PromptToolHost, _ loras: [(id: String, strength: Double)]) -> [String] {
        var words: [String] = []
        for l in loras { for w in host.app?.store?.library.item(l.id)?.triggerWords ?? [] where !words.contains(w) { words.append(w) } }
        return words
    }

    /// The prompt from a spec (through `PromptWriter`) or as given, with the age guard: refused on a youth term,
    /// `AgeGuard.negative` in the negative where the model reads one.
    static func write(_ args: ToolArgs, model m: PromptModel, triggerWords: [String]) throws -> WrittenPrompt {
        if let spec = args.raw["spec"] as? [String: Any] {
            let data = try JSONSerialization.data(withJSONObject: spec)
            let json = try JSONDecoder().decode(LLMClient.JSONValue.self, from: data)
            var s: PromptSpec
            do { s = try PromptEngine.spec(from: json) } catch let e as PromptEngineError { throw ToolError(e.description) }
            for w in triggerWords where !s.triggerWords.contains(w) { s.triggerWords.append(w) }
            return PromptWriter.write(s, for: m)
        }
        guard var prompt = args.string("prompt")?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
            throw ToolError("give a spec or a prompt")
        }
        let negative = args.string("negative") ?? ""
        if let why = PromptEngine.refusal([prompt]) { throw ToolError("Refused: \(why).") }
        let missing = triggerWords.filter { !prompt.lowercased().contains($0.lowercased()) }
        if !missing.isEmpty, m.dialect != .motion { prompt = missing.joined(separator: ", ") + ", " + prompt }
        return WrittenPrompt(prompt: prompt, negative: PromptEngine.guardedNegative(negative, model: m))
    }

    /// The installed models of a kind, with prices, times, their dialect and what fits them.
    @MainActor
    static func modelList(_ host: PromptToolHost, video: Bool) throws -> String {
        guard let app = host.app else { throw ToolError("the app isn't available") }
        guard let server = app.connections.gpuServers.first?.id,
              case .gpuOnline(let info)? = app.connections.status[server] else {
            throw ToolError("the GPU server isn't online")
        }
        let timings = GPUTimings.load(app.gpuTimingsURL)
        let lib = app.store?.library
        var out: [String] = []
        let candidates = PromptModel.allCases.filter { $0.isVideo == video && (host.fixedModel == nil || host.fixedModel == $0) }
        for m in candidates {
            guard let task = PromptPreview.task(m) ?? (video ? videoTask(m) : nil), let t = info.task(task), t.installed else { continue }
            let price = t.pricePerSecond ?? MovieCost.price(gpu: t.gpu) ?? 0
            let secs = timings.seconds(task: task, megapixels: 1)
            let warm = timings.isWarm(task: task, idleSeconds: Double(info.idleSeconds ?? 120))
            var line = "- \(m.rawValue): \(m.title). \(dialect(m)). ≈ \(CostEstimate.format(secs * price)) and \(Int(secs)) s a draft on \(t.gpu)"
            line += warm ? " (loaded now)" : " (+ ≈ \(Int(timings.loadSeconds(task: task))) s to load)"
            line += m.takesNegative ? "; reads a negative" : "; no negative"
            if m.takesWeights { line += "; (word:1.3) weights work" }
            if m.drawsText { line += "; draws legible text" }
            out.append(line)
            if let type = PromptPreview.nodeType(m), let family = RemoteNodes.libraryFamilies[type], let lib {
                let tunes = PromptPreview.takesFineTune(m) ? lib.models(for: family) : []
                for e in tunes {
                    let fits = m == .sdxl || m == .sdxlPony || m == .sdxlIllustrious ? PromptModel.sdxl(baseModel: e.baseModel) == m : true
                    if fits { out.append("    fine-tune \(e.id): \(e.title)" + (e.baseModel.map { " (base \($0))" } ?? "")) }
                }
                for e in lib.loras(for: family).prefix(30) {
                    out.append("    LoRA \(e.id): \(e.title)"
                               + (e.triggerWords.isEmpty ? "" : "; trigger words: \(e.triggerWords.prefix(8).joined(separator: ", "))"))
                }
            }
        }
        if out.isEmpty { return "No \(video ? "video" : "picture") models are installed on the GPU server." }
        if let c = host.characterID.flatMap({ app.character($0) }) {
            out.append("\nDrafts draw \(c.name) with their face reference and LoRAs on \(c.artModel.family) models.")
        }
        return "Installed \(video ? "video" : "picture") models (pick by `model`):\n" + out.joined(separator: "\n")
    }

    static func dialect(_ m: PromptModel) -> String {
        switch m.dialect {
        case .tags: "Reads tags"
        case .sentences: "Reads sentences"
        case .narrative: "Reads one detailed paragraph"
        case .motion: "Reads a paragraph of what moves"
        }
    }

    /// Video models' server tasks (their clips come in a later step; listed for picking).
    static func videoTask(_ m: PromptModel) -> String? {
        switch m {
        case .wan: "wan22-i2v"
        case .ltx23: "ltx23"
        case .minimaxVideo: "h3-video"
        default: nil
        }
    }

    /// One draft: the cost sheet, then the picture through the Prompt Builder's preview path (character face and
    /// LoRAs included), kept with the builder's previews.
    @MainActor
    static func draft(_ host: PromptToolHost, _ args: ToolArgs) async throws -> ToolOutput {
        guard let app = host.app else { throw ToolError("the app isn't available") }
        let m = try model(args)
        try checkFixed(host, m)
        guard PromptPreview.nodeType(m) != nil else { throw ToolError("\(m.title) makes video: draft its first frame on a picture model") }
        guard let prompt = args.string("prompt"), !prompt.isEmpty else { throw ToolError("missing \"prompt\"") }
        if let why = PromptEngine.refusal([prompt]) { throw ToolError("Refused: \(why).") }
        let written = WrittenPrompt(prompt: prompt, negative: PromptEngine.guardedNegative(args.string("negative") ?? "", model: m))
        guard let server = app.connections.gpuServers.first?.id else { throw ToolError("no GPU server") }

        var d = PromptBuilderDraft()
        d.model = m
        d.isPerson = false
        d.characterID = host.characterID
        d.loras = loraArgs(args).prefix(PromptPreview.maxLoRAs).map {
            PromptBuilderLoRA(id: $0.id, strength: $0.strength, triggerWords: app.store?.library.item($0.id)?.triggerWords ?? [])
        }
        d.variant = args.string("variant") ?? ""
        d.aspect = args.string("aspect").flatMap(PromptPreview.Aspect.init(rawValue:)) ?? .portrait
        d.seed = args.int("seed") ?? Int.random(in: 0..<Int(Int32.max))
        d.fast = args.bool("fast")
        for l in d.loras where app.store?.library.item(l.id) == nil { throw ToolError("there's no library LoRA \(l.id)") }

        let record = host.record
        let n = (record.tries.map(\.n).max() ?? 0) + 1
        await app.connections.test(server)
        let price = app.promptPreviewEstimate(d, written: written)
        let task = app.previewTask(d) ?? m.rawValue
        var gpu = "GPU"
        if case .gpuOnline(let info)? = app.connections.status[server] { gpu = info.task(task)?.gpu ?? gpu }
        var notes = ["The prompt engine's try \(n). Billing is per GPU second; the GPU stays up a little after the job."]
        if record.llmUSD > 0 { notes.append("The engine's language model has used \(CostEstimate.format(record.llmUSD)) so far.") }
        if price == nil { notes.append("The price isn't known: the server didn't give one for \(task).") }
        let estimate = CostEstimate(title: "Draft try \(n) on \(m.title)",
                                    lines: [CostEstimate.Line(label: "\(m.title) picture" + (price?.warm == false ? " (cold start)" : ""),
                                                              count: 1, gpu: gpu, seconds: 0, usd: price?.usd ?? 0)],
                                    notes: notes)
        record.status = "Try \(n): waiting for the cost sheet"
        guard try await host.approve(estimate) else {
            record.status = "Try \(n) declined"
            return ToolOutput("The user declined the cost sheet for try \(n): nothing ran. Call finish now with the best try so far"
                              + (record.tries.isEmpty ? " (there is none: finish with best 0)." : "."))
        }
        record.status = "Drafting try \(n)…"
        let started = Date()
        let (template, render, _) = try await app.previewJob(d, server: server, written: written)
        let (outputs, jobs) = try await app.runHeadlessGraph(template: template, render: render, title: "Prompt engine try \(n)",
                                                             progress: { record.status = "Try \(n): \($0)" })
        let shot = try app.keepPreview(d, written: written, outputs: outputs, jobs: jobs)
        let url = app.promptPreviewURL(shot)
        let t = EngineTry(n: n, model: m, prompt: written.prompt, negative: written.negative,
                          loras: d.loras.map { EngineLoRA(id: $0.id, strength: $0.strength, triggerWords: $0.triggerWords) },
                          seed: d.seed, aspect: d.aspect, fast: d.fast, variant: d.variant, file: url.path,
                          costUSD: shot.costUSD, seconds: Date().timeIntervalSince(started))
        record.tries.append(t)
        record.gpuUSD += shot.costUSD ?? 0
        record.status = "Try \(n): looking"
        guard let image = MediaFrames.image(url), let data = GraphHelper.dataURL(image, maxSide: 1024) else {
            return ToolOutput("Try \(n) is made (\(url.lastPathComponent)) but its picture couldn't be read.", isError: true)
        }
        let checklist = record.checklist.isEmpty ? "" : "\nCheck it against the checklist:\n" + record.checklist.map { "- \($0)" }.joined(separator: "\n")
        return ToolOutput("Try \(n): seed \(d.seed), \(CostEstimate.format(shot.costUSD ?? 0)). The picture follows. Look at it closely, then call verdict." + checklist,
                          images: [data])
    }
}
