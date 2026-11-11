import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-chat <url> [--smoke-token-file <file>] [--llm <link or id>] [--prompt <text>]`: adds the GPU server,
    /// installs the model if it isn't there (default `fake/tiny`, for a server run with `AISTUDIO_FAKE_LLM=1`), opens a
    /// chat, sends two messages and checks the replies, their stats, the saved transcript, and the engine's state.
    static func runChat(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-chat] \(s)"); fflush(stdout) }
        var token = "smoke"  // a local server without auth takes any token
        if let path = argument("--smoke-token-file") {
            token = (try? String(contentsOfFile: path, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        }
        let link = argument("--llm") ?? "fake/tiny"
        let prompt = argument("--prompt") ?? "hello there"
        let fake = link.hasPrefix("fake/")
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else {
            say("server not reachable: \(model.connections.status[conn.id].map { "\($0)" } ?? "?")"); return false
        }
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        if let e = llm.errors[conn.id] { say("listing failed: \(e)"); return false }

        // The model: installed already (by id), else installed from the link.
        var mid = llm.listings[conn.id]?.models.first { $0.id == link || $0.repo == link }?.id
        if mid == nil {
            do {
                let found = try await client.inspect(url: link)
                say("\(found.facts.repo): \(found.facts.modelType), \(found.facts.weightBytes / 1_000_000_000) GB, "
                    + "recommended \(found.plan.recommended?.label ?? "none")")
                let st = try await client.install(LLMInstallRequest(url: link))
                mid = st.id ?? found.suggestedId
            } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            let deadline = Date().addingTimeInterval(fake ? 30 : 6 * 3600)
            while Date() < deadline {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                if let st = llm.listings[conn.id]?.installs[mid!], st.state == "failed" {
                    say("install failed: \(st.error ?? "?")"); return false
                }
                try? await Task.sleep(nanoseconds: fake ? 200_000_000 : 10_000_000_000)
            }
        }
        guard let mid, llm.model(LLMKey(serverID: conn.id, modelID: mid))?.installed == true else {
            say("model didn't install"); return false
        }
        let key = LLMKey(serverID: conn.id, modelID: mid)
        say("model \(mid) on \(llm.model(key)!.gpuLabel)")

        // Two messages: the second checks the history goes along.
        let doc = model.newChat(model: key)
        var ok = true
        for (n, text) in [prompt, "and once more"].enumerated() {
            doc.draft = text
            model.send(doc)
            let start = Date()
            while doc.isStreaming, Date().timeIntervalSince(start) < (fake ? 30 : 3600) {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            guard let reply = doc.transcript.turns.last, reply.role == .assistant else { say("no reply"); return false }
            if let e = reply.error { say("reply \(n + 1) failed: \(e)"); ok = false; continue }
            let s = reply.stats
            say("reply \(n + 1) in \(String(format: "%.1f", Date().timeIntervalSince(start))) s: \(reply.text.prefix(200))")
            say("  thinking \(reply.reasoning?.count ?? 0) chars, \(s?.completionTokens ?? 0) tokens, "
                + "\(s?.tokPerSec.map { String(format: "%.0f tok/s", $0) } ?? "? tok/s"), "
                + "first token \(s?.ttftMs.map { String(format: "%.1f s", $0 / 1000) } ?? "?"), "
                + "\(s?.costUSD.map { String(format: "$%.4f", $0) } ?? "$?")")
            if reply.text.isEmpty { say("empty reply"); ok = false }
            if s == nil { say("no stats"); ok = false }
            if fake {
                if reply.text != "You said: \(text)" { say("unexpected reply"); ok = false }
                if reply.reasoning != "Thinking about it." { say("thinking not split out"); ok = false }
            }
        }
        if doc.title != ChatTranscript.title(from: prompt) { say("title is \(doc.title)"); ok = false }
        guard let saved = model.chatStore.load(doc.id), saved.turns.count == 4 else {
            say("transcript not saved"); return false
        }
        if !model.chatSummaries.contains(where: { $0.id == doc.id }) { say("not in the Chats panel"); ok = false }
        await llm.refreshStatus(key)
        say("engine: \(llm.status(key)?.state ?? "?") \(llm.status(key)?.gpu ?? "") uptime "
            + "\(Int(llm.status(key)?.uptime ?? 0)) s, \(String(format: "$%.3f", llm.status(key)?.costUSD ?? 0))")
        if llm.status(key)?.state != "ready" { say("engine not ready after the replies"); ok = false }

        // Closing and reopening the tab brings the conversation back.
        model.closeChat(doc.id)
        model.openChat(doc.id)
        if model.activeChat?.transcript.turns.count != 4 { say("reopened chat lost turns"); ok = false }
        model.closeChat(doc.id)

        if fake {
            await llm.stop(key)
            if llm.status(key)?.state != "cold" { say("stop didn't stop it"); ok = false }
            model.deleteChat(doc.id)
            await llm.remove(key)
            if llm.model(key) != nil { say("model not removed"); ok = false }
        } else {
            say("left \(mid) running: it stops by itself after the idle time (or Shut Down GPU in Servers)")
        }
        say(ok ? "OK" : "FAILED")
        return ok
    }
}
