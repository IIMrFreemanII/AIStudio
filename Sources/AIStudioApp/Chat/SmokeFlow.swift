import Agent
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-flow <url>` (a server with `AISTUDIO_FAKE_LLM=1`): in a coding chat on a fresh git repository, "run a
    /// flow" makes the fake call `flow` with its smoke script (two parallel reviewers with a schema, verifyEach, one
    /// edit agent). The run needs the user's approval, must finish with both reviewers overlapping in time, parsed
    /// findings, one verifier per finding and a branch with flow.txt; its result must reach the chat and the agent
    /// must carry on. Resuming the run reuses every agent (nothing runs again). A slow flow is stopped mid-run.
    static func runFlow(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-flow] \(s)"); fflush(stdout) }
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/agent" }?.id
        if mid == nil {
            do { mid = try await client.install(LLMInstallRequest(url: "fake/agent")).id } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            for _ in 0..<100 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        guard let mid else { say("no model"); return false }
        let key = LLMKey(serverID: conn.id, modelID: mid)

        let repo = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-flow-\(UUID().uuidString.prefix(8))")
            .standardizedFileURL.resolvingSymlinksInPath()
        try? FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        _ = await ShellRunner().run("git init -q -b main && git config user.email smoke@test && git config user.name Smoke"
                                    + " && echo '# Smoke' > README.md && git add -A && git commit -qm init", cwd: repo)
        let doc = model.newCodingChat(project: repo, model: key)
        var ok = true
        func check(_ c: Bool, _ s: String) { say((c ? "✓ " : "✗ ") + s); if !c { ok = false } }
        var asked: [String] = []
        let answerer = Task { @MainActor in
            while !Task.isCancelled {
                if let r = doc.permission { asked.append(r.call.name); doc.answer(.allow) }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        var runIDs: [String] = []
        func wait(_ seconds: Double, until done: @MainActor () -> Bool) async -> Bool {
            let start = Date()
            while !done() {
                if Date().timeIntervalSince(start) > seconds { return false }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return true
        }
        func finish(_ result: Bool) async -> Bool {
            answerer.cancel()
            for id in runIDs {
                if let c = model.flowController(id) { c.stop() }
                try? FileManager.default.removeItem(at: FlowRegistry.folder.appendingPathComponent(id + ".json"))
            }
            model.deleteChat(doc.id)
            _ = await ShellRunner().run("git worktree list --porcelain | awk '/^worktree /{print $2}' | tail -n +2 | "
                                        + "while read w; do git worktree remove --force \"$w\"; done", cwd: repo)
            try? FileManager.default.removeItem(at: repo)
            say(result ? "OK" : "FAILED")
            return result
        }

        // 1. The agent starts the smoke flow; the user approves it.
        doc.draft = "run a flow"
        model.send(doc)
        guard await wait(30, until: { !doc.flowRunByCall.isEmpty }),
              let runID = doc.flowRunByCall.values.first, let flow = model.flowController(runID) else {
            say("✗ no flow started (asked: \(asked), last: \(doc.transcript.turns.last?.text.prefix(200) ?? "none"))")
            return await finish(false)
        }
        runIDs.append(runID)
        check(asked.contains("flow"), "the flow asked for approval (\(asked))")
        check(flow.run.meta.name == "smoke-flow", "the run is “smoke-flow” (\(flow.run.meta.name))")
        guard await wait(90, until: { flow.run.status != .running }) else {
            say("✗ the flow didn't finish in 90 s: \(flow.run.agents.map { "\($0.label)=\($0.status.rawValue)" })")
            return await finish(false)
        }
        let run = flow.run
        check(run.status == .done, "the flow finished (\(run.status.rawValue)\(run.error.map { ": \($0)" } ?? ""))")
        check(run.agents.count == 5, "5 agents ran (\(run.agents.map(\.label)))")
        let reviews = run.agents.filter { $0.label.hasPrefix("review:") }
        if reviews.count == 2, let e0 = reviews[0].endedAt, let e1 = reviews[1].endedAt {
            check(reviews[1].startedAt < e0 && reviews[0].startedAt < e1, "the two reviewers ran at the same time")
        } else {
            check(false, "two reviewers finished (\(reviews.map { "\($0.label)=\($0.status.rawValue)" }))")
        }
        check(reviews.allSatisfy { $0.json != nil }, "the reviewers' findings were parsed from JSON")
        let verifiers = run.agents.filter { $0.label.hasPrefix("verify:") }
        check(verifiers.count == 2 && verifiers.allSatisfy { $0.phase == "Verify" && $0.json != nil },
              "one verifier per finding, in the Verify phase (\(verifiers.map(\.label)))")
        let result = run.result ?? ""
        check(result.contains("\"findings\":2") && result.contains("\"real\":2"), "the script's result (\(result))")
        let edit = run.agents.first { $0.mode == .edit }
        if let branch = edit?.branch {
            let shown = await ShellRunner().run("git show \(branch):flow.txt", cwd: repo).output
            check(shown == "made by a flow\n", "the edit agent's branch \(branch) has flow.txt (\(shown.prefix(60)))")
            check(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("flow.txt").path),
                  "the main checkout is untouched")
        } else {
            check(false, "the edit agent made a branch (\(edit?.text?.prefix(120) ?? edit?.error ?? "no edit agent"))")
        }

        // 2. The result reaches the chat and the agent carries on.
        let delivered = await wait(30) {
            guard let i = doc.transcript.turns.lastIndex(where: { $0.flowEvent?.runID == runID }) else { return false }
            return !doc.isStreaming && doc.transcript.turns.count > i + 1 && doc.transcript.turns.last?.role == .assistant
        }
        check(delivered, "the result came back to the chat and the agent replied")

        // 3. Resume: every agent is reused from the finished run.
        do {
            let again = try model.startFlow(script: run.script, meta: run.meta, args: run.args, chatID: doc.id, resumeFrom: runID)
            runIDs.append(again.run.id)
            if await wait(30, until: { again.run.status != .running }) {
                check(again.run.status == .done && again.run.agents.count == 5
                      && again.run.agents.allSatisfy { $0.status == .cached },
                      "resume reused all 5 agents (\(again.run.agents.map { $0.status.rawValue }))")
            } else {
                check(false, "the resumed run finished")
            }
        } catch {
            check(false, "resume started (\(error.localizedDescription))")
        }

        // 4. Stop: a slow flow is stopped while its agents work.
        let slow = """
        export const meta = { name: 'slow-flow', description: 'Three slow agents', phases: [{ title: 'Wait' }] }
        await parallel([1, 2, 3].map(i => () => agent(`Wait a while. sleep 20`, { label: `slow:${i}`, phase: 'Wait' })))
        return 'finished'
        """
        do {
            let s = try model.startFlow(script: slow, meta: try FlowScript.meta(slow), args: nil, chatID: doc.id)
            runIDs.append(s.run.id)
            _ = await wait(10) { s.run.agents.count == 3 }
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            let stoppedAt = Date()
            s.stop()
            let stopped = await wait(5) { s.run.status == .stopped }
            check(stopped && Date().timeIntervalSince(stoppedAt) < 5, "Stop ended the slow flow (\(s.run.status.rawValue))")
            check(s.run.agents.allSatisfy { $0.status == .stopped }, "its agents show as stopped (\(s.run.agents.map { $0.status.rawValue }))")
        } catch {
            check(false, "the slow flow started (\(error.localizedDescription))")
        }
        return await finish(ok)
    }
}
