import Agent
import CharacterCore
import Foundation
import LLMClient
import MovieCore

extension SmokeTest {
    /// `--smoke-prompt-engine-fake <url>`: against a GPU server run with `AISTUDIO_FAKE_MODELS=1`, a scripted model
    /// drives the prompt tools: checklist, models, rules, write (with the age guard's refusal), two drafts behind cost
    /// sheets the test presses, verdicts and finish; then a second run whose sheet is declined ends with no try.
    static func runPromptEngineFake(_ model: AppModel, url: String) async -> Bool {
        var ok = true
        func say(_ s: String) { print("[smoke-prompt-engine] \(s)"); fflush(stdout) }
        func check(_ good: Bool, _ what: String) { if good { say("ok \(what)") } else { say("FAIL \(what)"); ok = false } }

        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else { say("FAIL server not reachable"); return false }
        let key = LLMKey(serverID: conn.id, modelID: "scripted")

        // The pure side: rules for every model, warnings, the guard.
        check(PromptModel.allCases.allSatisfy { PromptEngine.rules(for: $0).contains("###") }, "the rule book has every model's section")
        check(PromptEngine.warnings(prompt: "a photo", negative: "blurry", model: .flux).contains { $0.contains("no negative") },
              "FLUX's negative is flagged")

        for decline in [false, true] {
            let s = PromptEngineSession(app: model, idea: "a woman standing in a sunlit kitchen, whole body in frame", llm: key)
            let script = ScriptedEngineModel()
            s.clientOverride = script
            s.start()
            var sheets = 0
            let deadline = Date().addingTimeInterval(120)
            while s.running || s.record.result == nil, Date() < deadline {
                if let g = model.promptEngineGate {
                    sheets += 1
                    model.promptEngineGate = nil
                    if !decline { g.run() }
                    say("cost sheet \(sheets): \(g.estimate.title) ≈ \(CostEstimate.format(g.estimate.usd)) → \(decline ? "Cancel" : "Run")")
                }
                try? await Task.sleep(for: .milliseconds(100))
            }
            await s.wait()
            for step in s.steps { say("  step: \(step)") }
            let r = s.record
            if decline {
                check(sheets == 1 && r.tries.isEmpty, "a declined sheet draws nothing")
                check(r.result != nil && r.result?.best == nil, "the run ends with no try")
            } else {
                check(r.checklist.count == 3, "the checklist is set")
                check(script.refusedYouth, "write_prompt refuses a youth term")
                check(script.wrotePrompt.contains("adult") && script.wrotePrompt.contains("full body"),
                      "write_prompt lays out the spec with an adult subject")
                check(sheets == 2 && r.tries.count == 2, "two drafts, each behind a cost sheet")
                check(r.tries.allSatisfy { FileManager.default.fileExists(atPath: $0.file) }, "the drafts are saved")
                check(r.tries.allSatisfy { $0.negative.contains("underage") }, "the drafts' negatives carry the age guard")
                check(r.verdicts.map(\.pass) == [false, true], "verdicts recorded")
                check(r.result?.best?.n == 2, "finished with try 2")
                check(script.sawImage, "the model was shown the draft")
                for t in r.tries { model.deletePromptPreview(model.promptPreviews().first { model.promptPreviewURL($0).path == t.file }!) }
            }
        }
        say(ok ? "PASS" : "FAILED")
        return ok
    }

    /// `--smoke-prompt-engine "<idea>" [--engine-llm <model id>] [--engine-model <prompt model>]
    /// [--engine-auto-run] [--smoke-real-out <dir>]`: a real run on the first saved GPU server. Without
    /// `--engine-auto-run` it stops at the first cost sheet; with it, every sheet is pressed (only after the user
    /// approved the spend). Prints the steps, tries, verdicts and costs, and copies the drafts to the out folder.
    static func runPromptEngineReal(_ model: AppModel, idea: String) async -> Bool {
        func say(_ s: String) { print("[smoke-prompt-engine] \(s)"); fflush(stdout) }
        guard let server = await savedGPUServer(model) else { say("no saved GPU server with a token"); return false }
        await model.llm.refresh(server)
        let wanted = argument("--engine-llm") ?? "qwen3.8-27b-abliterated"
        let key = LLMKey(serverID: server, modelID: wanted)
        guard model.llm.model(key)?.installed == true else { say("\(wanted) isn't installed"); return false }
        let fixed = argument("--engine-model").flatMap(PromptModel.init(rawValue:))
        let s = PromptEngineSession(app: model, idea: idea, fixedModel: fixed, llm: key)
        s.autoRun = CommandLine.arguments.contains("--engine-auto-run")
        s.start()
        let started = Date()
        var lastStatus = "", lastSteps = 0
        while s.running || s.record.result == nil {
            if s.record.status != lastStatus { lastStatus = s.record.status; say("status: \(lastStatus)") }
            while lastSteps < s.steps.count { say("step: \(s.steps[lastSteps])"); lastSteps += 1 }
            if let g = model.promptEngineGate {
                say("cost sheet: \(g.estimate.title) ≈ \(CostEstimate.format(g.estimate.usd)); stopping (no --engine-auto-run)")
                model.promptEngineGate = nil
            }
            if Date().timeIntervalSince(started) > 3600 { say("an hour passed: stopping"); s.stop() }
            try? await Task.sleep(for: .milliseconds(300))
        }
        await s.wait()
        let r = s.record
        say("checklist: \(r.checklist.joined(separator: " | "))")
        let out = argument("--smoke-real-out").map { URL(fileURLWithPath: $0) }
        if let out { try? FileManager.default.createDirectory(at: out, withIntermediateDirectories: true) }
        for t in r.tries {
            let v = r.verdict(t.n)
            say("try \(t.n) [\(t.model.rawValue), seed \(t.seed), \(CostEstimate.format(t.costUSD ?? 0))]: \(v.map { $0.pass ? "PASS" : "fail: \($0.issues.joined(separator: "; ")) → \($0.change)" } ?? "no verdict")")
            say("  prompt: \(t.prompt)")
            if !t.negative.isEmpty { say("  negative: \(t.negative)") }
            if let out {
                try? FileManager.default.copyItem(at: URL(fileURLWithPath: t.file),
                                                  to: out.appendingPathComponent("try-\(t.n)." + (t.file as NSString).pathExtension))
            }
        }
        say("result: try \(r.result?.best?.n.description ?? "none") — \(r.result?.notes ?? "")")
        if !s.reply.isEmpty { say("reply: \(s.reply)") }
        if let e = s.error { say("error: \(e)") }
        say(String(format: "spent: GPU $%.3f, LLM $%.3f, %.0f s", r.gpuUSD, r.llmUSD, Date().timeIntervalSince(started)))
        return s.error == nil
    }
}

extension SmokeTest {
    /// The user's first saved GPU server, added to this automated run (which starts with none; its token stays in
    /// memory), once it answers.
    static func savedGPUServer(_ model: AppModel) async -> String? {
        guard let data = try? Data(contentsOf: ConnectionStore.defaultFileURL),
              let saved = try? JSONDecoder().decode([ServerConnection].self, from: data),
              let conn = saved.first(where: { $0.kind == .gpu }), let token = ConnectionToken.read(conn.id) else { return nil }
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else { return nil }
        return conn.id
    }

    /// `--smoke-engine-judge <folder> [--engine-llm <model id>]`: shows the engine's LLM each PNG in the folder and
    /// asks the checklist questions (people, standing, feet in frame, the lowest body part shown), to see how well it
    /// judges framing. One request per picture, no tools; prints the answers and the cost.
    static func runEngineJudge(_ model: AppModel, folder: String) async -> Bool {
        func say(_ s: String) { print("[engine-judge] \(s)"); fflush(stdout) }
        guard let server = await savedGPUServer(model) else { say("no saved GPU server with a token"); return false }
        await model.llm.refresh(server)
        let id = argument("--engine-llm") ?? "qwen3.8-27b-abliterated"
        guard let client = model.llm.client(server), model.llm.model(LLMKey(serverID: server, modelID: id))?.installed == true
        else { say("\(id) isn't installed"); return false }
        let dir = URL(fileURLWithPath: folder)
        let files = ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        let question = """
        Look at this picture and answer with only JSON: {"people": number of people, "standing": true/false, \
        "feet_in_frame": true if both feet (or shoes) are fully inside the picture, "lowest_visible": the lowest body \
        part that is visible, "framing": "close-up" / "head and shoulders" / "waist up" / "thighs up" / "knees up" / \
        "full body"}.
        """
        var total = 0.0
        let started = Date()
        for f in files {
            guard let image = MediaFrames.image(f), let data = GraphHelper.dataURL(image, maxSide: 1024) else {
                say("\(f.lastPathComponent): can't read"); continue
            }
            var req = ChatRequest(model: id, messages: [ChatMessage(role: .user, parts: [.image(url: data), .text(question)])])
            req.maxTokens = 2000
            req.temperature = 0.2
            var acc = ReplyAccumulator()
            let t0 = Date()
            do {
                for try await e in client.chat.stream(req) { acc.add(e) }
            } catch {
                say("\(f.lastPathComponent): \(error.localizedDescription)"); return false
            }
            total += acc.stats?.costUSD ?? 0
            let answer = acc.content.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "\n", with: " ")
            say(String(format: "%@ (%.0f s): %@", f.deletingPathExtension().lastPathComponent, Date().timeIntervalSince(t0), answer))
        }
        say(String(format: "spent: LLM $%.3f in %.0f s", total, Date().timeIntervalSince(started)))
        return true
    }
}

/// The fake-server smoke's model: one tool call per turn, in the engine's order, reading the last tool result.
final class ScriptedEngineModel: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var _refused = false, _wrote = "", _sawImage = false
    var refusedYouth: Bool { lock.withLock { _refused } }
    var wrotePrompt: String { lock.withLock { _wrote } }
    var sawImage: Bool { lock.withLock { _sawImage } }

    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        let tools = request.messages.filter { $0.role == .tool }
        let last = tools.last?.text ?? ""
        let image = request.messages.last.map { $0.role == .user && $0.parts.contains { if case .image = $0 { true } else { false } } } ?? false
        lock.withLock {
            if image { _sawImage = true }
            if tools.count == 4 { _refused = last.contains("Refused") }
            if tools.count == 5 { _wrote = last }
        }
        let spec: [String: Any] = ["subject": "a 30-year-old woman", "pronoun": "she", "shot": "full body",
                                   "action": "standing", "place": "a sunlit kitchen"]
        let prompt = "a 30-year-old adult woman standing in a sunlit kitchen, full body, head to toe"
        let steps: [(String, [String: Any])] = [
            ("set_checklist", ["items": ["whole body in frame", "one person", "sunlit kitchen"]]),
            ("draft_models", ["kind": "image"]),
            ("model_rules", ["model": "sdxl"]),
            ("write_prompt", ["model": "sdxl", "prompt": "a teen girl in a kitchen"]),
            ("write_prompt", ["model": "sdxl", "spec": spec]),
            ("draft", ["model": "sdxl", "prompt": prompt, "negative": "blurry", "seed": 1]),
            ("verdict", ["attempt": 1, "pass": false, "issues": ["cut at the knees"], "change": "another seed"]),
            ("draft", ["model": "sdxl", "prompt": prompt, "negative": "blurry", "seed": 2]),
            ("verdict", ["attempt": 2, "pass": true]),
            ("finish", ["best": 2, "notes": "the second seed framed the whole body"]),
        ]
        let declined = last.contains("declined")
        return AsyncThrowingStream { c in
            if declined {
                c.yield(.toolCall(index: 0, id: "s\(tools.count)", name: "finish", arguments: #"{"best":0,"notes":"declined"}"#))
            } else if tools.count < steps.count, !last.hasPrefix("Finished") {
                let (name, args) = steps[tools.count]
                let json = String(data: (try? JSONSerialization.data(withJSONObject: args)) ?? Data(), encoding: .utf8) ?? "{}"
                c.yield(.toolCall(index: 0, id: "s\(tools.count)", name: name, arguments: json))
            } else {
                c.yield(.content("Chose try 2: its framing shows the whole body."))
            }
            c.finish()
        }
    }
}
