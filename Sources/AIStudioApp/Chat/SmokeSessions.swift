import Agent
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-sessions <url>` (a server with `AISTUDIO_FAKE_LLM=1`): a coding chat in a fresh git repository is told
    /// to "start a session to create hello.txt". The fake model calls `start_session`; the child must run hidden in its
    /// own worktree on an `aistudio/…` branch and write hello.txt there (not in the main checkout). Its report must come
    /// back to the parent as a session-event turn, and the parent must carry on by itself. Then the child's branch is
    /// merged into the main checkout, and a fork from the first message is checked.
    static func runSessions(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-sessions] \(s)"); fflush(stdout) }
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

        let repo = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-sessions-\(UUID().uuidString.prefix(8))")
            .standardizedFileURL.resolvingSymlinksInPath()
        try? FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        _ = await ShellRunner().run("git init -q -b main && git config user.email smoke@test && git config user.name Smoke"
                                    + " && echo '# Smoke' > README.md && git add -A && git commit -qm init", cwd: repo)
        let parent = model.newCodingChat(project: repo, model: key)
        var ok = true
        func fail(_ s: String) { say(s); ok = false }

        // Every session's questions are allowed (the parent's start_session, the child's write and command).
        let answerer = Task { @MainActor in
            while !Task.isCancelled {
                for d in model.chats where d.permission != nil { d.answer(.allow) }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        defer { answerer.cancel() }
        func wait(_ seconds: Double, until done: @MainActor () -> Bool) async -> Bool {
            let start = Date()
            while !done() {
                if Date().timeIntervalSince(start) > seconds { return false }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return true
        }

        var childID: UUID?
        func finish(_ result: Bool) -> Bool {
            answerer.cancel()
            for id in [parent.id] + (childID.map { [$0] } ?? []) { model.deleteChat(id) }
            Task { _ = await ShellRunner().run("git worktree prune", cwd: repo) }
            try? FileManager.default.removeItem(at: repo)
            say(result ? "OK" : "FAILED")
            return result
        }

        // Switching between chats on a stopped model leaves it stopped: no start, and no keepalive that could wake it.
        await llm.stop(key)
        let other = model.newChat(model: key)
        for _ in 0..<7 {
            model.switchToPrevious()
            await llm.tick()
        }
        model.openChat(parent.id)
        await llm.refreshAll()
        await llm.refreshStatus(key)
        if llm.status(key)?.isActive == true {
            fail("switching chats started the model (\(llm.status(key)?.state ?? "?"))")
        } else {
            say("switching between chats leaves a stopped model stopped (\(llm.status(key)?.state ?? "?"))")
        }
        model.deleteChat(other.id)
        model.activateChat(parent.id)

        parent.draft = "start a session to create hello.txt"
        model.send(parent)

        // The child: hidden, in its own worktree.
        _ = await wait(30) {
            childID = model.chats.first { $0.transcript.parentID == parent.id }?.id
                ?? model.chatStore.list().first { $0.parentID == parent.id }?.id
            return childID != nil
        }
        guard let cid = childID else { fail("no child session started"); return finish(false) }
        if let live = model.chats.first(where: { $0.id == cid }), live.isRecent { fail("the child is in the recent list") }

        // Its report reaches the parent, which carries on by itself.
        let reported = await wait(90) {
            !parent.isStreaming && parent.transcript.turns.contains { $0.sessionEvent?.childID == cid }
                && parent.transcript.turns.last?.role == .assistant
                && parent.transcript.turns.last?.text.contains("hi from the agent") == true
        }
        let event = parent.transcript.turns.last { $0.sessionEvent != nil }?.sessionEvent
        say("event: \(event.map { "\($0.status) on \($0.branch ?? "-"): \($0.report.prefix(60))" } ?? "none")")
        say("parent: \(parent.transcript.turns.last?.text.prefix(100) ?? "none")")
        if !reported { fail("the parent didn't get the report and carry on") }

        // The child's work is in its worktree only; once done and hidden, it's unloaded.
        _ = await wait(10) { !model.chats.contains { $0.id == cid } }
        if model.chats.contains(where: { $0.id == cid }) { fail("the finished hidden child is still loaded") }
        guard let child = model.chatStore.load(cid), let wt = child.worktree else {
            fail("the child has no worktree"); return finish(false)
        }
        if !wt.branch.hasPrefix(model.worktreeBranchPrefix) { fail("branch \(wt.branch)") }
        let inTree = (try? String(contentsOfFile: wt.path + "/hello.txt", encoding: .utf8)) ?? ""
        if inTree != "hi from the agent\n" { fail("hello.txt in the worktree: \"\(inTree)\"") }
        if FileManager.default.fileExists(atPath: repo.appendingPathComponent("hello.txt").path) {
            fail("hello.txt leaked into the main checkout")
        }
        if (child.depth ?? 0) != 1 { fail("child depth \(child.depth ?? 0)") }

        // Merge the child's branch back.
        _ = await ShellRunner().run("git add -A && git commit -qm hello", cwd: URL(fileURLWithPath: wt.path))
        do {
            _ = try await GitWorktree.merge(branch: wt.branch, into: repo)
            if !FileManager.default.fileExists(atPath: repo.appendingPathComponent("hello.txt").path) {
                fail("merge didn't bring hello.txt")
            }
        } catch { fail("merge: \(error.localizedDescription)") }

        // Fork the parent from its first message.
        if let first = parent.transcript.turns.first {
            let before = Set(model.chats.map(\.id))
            model.fork(parent, at: first.id)
            _ = await wait(10) { model.chats.contains { !before.contains($0.id) } }
            if let f = model.chats.first(where: { !before.contains($0.id) }) {
                if f.transcript.turns.count != 1 || f.transcript.turns.first?.text != first.text
                    || f.transcript.parentID != parent.id {
                    fail("fork: \(f.transcript.turns.count) turns, parent \(String(describing: f.transcript.parentID))")
                }
                model.deleteChat(f.id)
            } else { fail("no fork opened") }
        }
        // A forked session keeps the history but not the message that started it (its brief takes its place, or it
        // would read "hand this off" and hand off again), and says it shares the folder and its branch.
        if let first = parent.transcript.turns.first(where: { $0.role == .user }) {
            let ask = "Hand the cleanup off to a new session."
            parent.transcript.append(ChatTurn(role: .user, text: ask))
            do {
                let info = try await model.startChild(SessionSpec(title: "Smoke fork", prompt: "Say hi.", context: .fork, worktree: false),
                                                      parent: parent)
                if let c = model.chats.first(where: { $0.id == info.id }) {
                    let texts = c.transcript.turns.filter { $0.role == .user }.map(\.text)
                    if !texts.contains(first.text) { fail("the forked session lost the parent's history") }
                    if texts.contains(ask) { fail("the forked session still has the message that started it") }
                    if texts.last?.contains("You share this folder") != true || texts.last?.contains("git branch") != true {
                        fail("the shared-folder note: \(texts.last ?? "none")")
                    }
                    _ = await wait(30) { !c.isStreaming }
                    model.deleteChat(c.id)
                } else { fail("no forked session") }
            } catch { fail("forked session: \(error.localizedDescription)") }
        }
        say("child branch \(wt.branch)")
        return finish(ok)
    }
}
