import Agent
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-git <url>` (a server with `AISTUDIO_FAKE_LLM=1`): a coding chat in a fresh git repository creates
    /// hello.txt; the chat's git state must list it as untracked (+1) after the run. Reverting it removes it; a tracked
    /// file's revert restores its content. Then hello.txt is written again, the commit message is drafted by the model
    /// (the fake's canned message), and committed: the status is clean and the commit has the co-author line. With no
    /// remote, pull requests aren't offered.
    static func runGit(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-git] \(s)"); fflush(stdout) }
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

        let repo = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-git-\(UUID().uuidString.prefix(8))")
            .standardizedFileURL.resolvingSymlinksInPath()
        try? FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        _ = await ShellRunner().run("git init -q -b main && git config user.email smoke@test && git config user.name Smoke"
                                    + " && echo '# Smoke' > README.md && git add -A && git commit -qm init", cwd: repo)
        // Reverted new files are deleted, not sent to the user's Trash.
        GitActions.trash = { try FileManager.default.removeItem(at: $0) }
        let doc = model.newCodingChat(project: repo, model: key)
        var ok = true
        func fail(_ s: String) { say("✗ " + s); ok = false }
        func check(_ c: Bool, _ s: String) { if c { say("✓ " + s) } else { fail(s) } }

        let answerer = Task { @MainActor in
            while !Task.isCancelled {
                if doc.permission != nil { doc.answer(.allow) }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        func wait(_ seconds: Double, until done: @MainActor () -> Bool) async -> Bool {
            let start = Date()
            while !done() {
                if Date().timeIntervalSince(start) > seconds { return false }
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            return true
        }
        func finish(_ result: Bool) -> Bool {
            answerer.cancel()
            model.deleteChat(doc.id)
            try? FileManager.default.removeItem(at: repo)
            say(result ? "OK" : "FAILED")
            return result
        }
        func hello() -> GitChange? { doc.git?.status.files.first { $0.path == "hello.txt" } }

        // The agent creates hello.txt; the run's end refreshes the git state by itself.
        doc.draft = "create hello.txt"
        model.send(doc)
        let ran = await wait(30) { !doc.isStreaming && FileManager.default.fileExists(atPath: repo.appendingPathComponent("hello.txt").path) }
        guard ran else { fail("the agent didn't create hello.txt"); return finish(false) }
        _ = await wait(10) { hello() != nil && !doc.gitRefreshing }
        guard let made = hello() else { fail("hello.txt isn't in the chat's git state: \(String(describing: doc.git?.status.files))"); return finish(false) }
        check(made.kind == .untracked && made.added == 1, "hello.txt is listed as untracked, +1 (\(made.kind), \(made.added ?? -1))")
        check(doc.git?.status.branch == "main" && doc.git?.base == "main", "branch and base are main")
        check(doc.git?.ghReady == false, "no remote: pull requests aren't offered")

        // Reverting: the new file goes; a tracked file gets its content back.
        check(await model.revertFile(doc, made), "revert hello.txt")
        check(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("hello.txt").path), "hello.txt is gone")
        let readme = repo.appendingPathComponent("README.md")
        try? "# Changed\n".write(to: readme, atomically: true, encoding: .utf8)
        await model.refreshGit(doc)
        if let change = doc.git?.status.files.first(where: { $0.path == "README.md" }) {
            check(change.kind == .modified, "README.md is modified")
            check(await model.revertFile(doc, change), "revert README.md")
            check((try? String(contentsOf: readme, encoding: .utf8)) == "# Smoke\n", "README.md is back to its committed text")
        } else {
            fail("README.md isn't listed after changing it")
        }
        check(doc.git?.status.files.isEmpty == true, "nothing left to commit")

        // Write it again, draft the message, commit.
        try? "hi from the agent\n".write(to: repo.appendingPathComponent("hello.txt"), atomically: true, encoding: .utf8)
        await model.refreshGit(doc)
        check(hello() != nil, "hello.txt is listed again")
        await model.draftCommitMessage(doc)
        check(doc.commitDraft == "Add hello.txt\n\nCreates hello.txt with a greeting from the agent.",
              "the model drafted the commit message (\(doc.commitDraft.prefix(60)))")
        check(await model.commit(doc, push: false), "commit")
        check(doc.git?.status.files.isEmpty == true, "the status is clean after committing")
        check(doc.commitDraft.isEmpty, "the commit box is cleared")
        let message = await ShellRunner().run("git log -1 --format=%B", cwd: repo).output
        check(message.hasPrefix("Add hello.txt"), "the commit has the drafted message")
        check(message.contains("Co-Authored-By: AIStudio ("), "the commit has the co-author line")
        check(doc.git?.status.head.isEmpty == false, "HEAD moved to the new commit")
        return finish(ok)
    }
}
