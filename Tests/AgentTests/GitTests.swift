import Foundation
import LLMClient
import Testing
@testable import Agent

private func sh(_ cmd: String, _ cwd: URL) async -> String {
    let r = await ShellRunner().run(cmd, cwd: cwd)
    return r.output
}

private func makeRepo() async throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("git-" + UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    _ = await sh("git init -q -b main && git config user.email t@t && git config user.name t && echo base > a.txt && git add -A && git commit -q -m init", dir)
    return dir.standardizedFileURL.resolvingSymlinksInPath()
}

struct GitTests {
    @Test func safetyTable() {
        let risky: [(String, String)] = [
            ("git push --force origin main", "Force-pushing rewrites the remote branch's history."),
            ("git push -f", "Force-pushing rewrites the remote branch's history."),
            ("git push --force-with-lease origin x", "Force-pushing rewrites the remote branch's history."),
            ("git push --force-if-includes", "Force-pushing rewrites the remote branch's history."),
            ("git push origin +main", "Force-pushing rewrites the remote branch's history."),
            ("echo hi && git push --force origin main", "Force-pushing rewrites the remote branch's history."),
            ("git commit --no-verify -m x", "It skips the repository's hooks."),
            ("git rebase --no-verify main", "It skips the repository's hooks."),
            ("git push origin x", "It sends commits to the remote."),
            ("git -C repo push", "It sends commits to the remote."),
            ("git reset --hard HEAD~1", "It discards uncommitted changes."),
            ("git clean -fd", "It deletes untracked files."),
            ("git clean -xdf", "It deletes untracked files."),
            ("git clean --force", "It deletes untracked files."),
            ("git checkout -- a.txt", "It discards changes to files."),
            ("git checkout .", "It discards changes to files."),
            ("git checkout -f main", "It discards changes to files."),
            ("git restore a.txt", "It discards changes to files."),
            ("git restore --staged --worktree a.txt", "It discards changes to files."),
            ("git branch -D feat", "It deletes a branch even if it isn't merged."),
            ("git branch --delete --force feat", "It deletes a branch even if it isn't merged."),
            ("git stash drop", "It deletes stashed changes."),
            ("git stash clear", "It deletes stashed changes."),
            ("git rebase main", "It rewrites commit history."),
            ("git filter-branch --all", "It rewrites commit history."),
            ("git commit --amend", "It rewrites the last commit."),
            ("git config user.name bob", "It changes git settings."),
            ("git config --global user.name bob", "It changes git settings."),
            ("gh pr create --title x", "It opens a pull request on GitHub."),
            ("gh pr merge 3 --squash", "It merges a pull request on GitHub."),
            ("gh pr close 3", "It closes a pull request on GitHub."),
        ]
        for (command, why) in risky { #expect(GitSafety.risk(command) == why, "\(command)") }
        let safe = ["git status", "git diff HEAD", "git log --oneline", "git add -A", "git commit -m x", "git branch -d done",
                    "git checkout main", "git switch -c feat", "git restore --staged a.txt", "git config --get user.name",
                    "git config user.name", "git -C sub status", "gh pr view", "ls", "git stash", "echo git push"]
        for command in safe { #expect(GitSafety.risk(command) == nil, "\(command)") }
    }

    @Test func riskyCommandsAskEvenWhenAllowed() {
        let rules = AllowRules(commandPrefixes: ["git push", "git status"])
        for mode in [PermissionMode.ask, .acceptEdits] {
            #expect(PermissionEngine.decide(effect: .execute, inProject: true, command: "git push origin x", mode: mode, rules: rules) == .ask)
            #expect(PermissionEngine.decide(effect: .execute, inProject: true, command: "git status", mode: mode, rules: rules) == .allow)
        }
        #expect(PermissionEngine.decide(effect: .execute, inProject: true, command: "git push origin x", mode: .bypass, rules: rules) == .allow)
    }

    @Test func parseStatusFixture() {
        let out = [
            "# branch.oid 0123456789abcdef0123456789abcdef01234567",
            "# branch.head main",
            "# branch.upstream origin/main",
            "# branch.ab +2 -1",
            "1 M. N... 100644 100644 100644 aaa bbb staged.txt",
            "1 .M N... 100644 100644 100644 aaa bbb unstaged.txt",
            "1 A. N... 000000 100644 100644 000 bbb added.txt",
            "1 .D N... 100644 100644 000000 aaa bbb deleted.txt",
            "2 R. N... 100644 100644 100644 aaa bbb R100 new name.txt",
            "old.txt",
            "u UU N... 100644 100644 100644 100644 aaa bbb ccc conflict.txt",
            "? untracked file.txt",
            "! ignored.log",
        ].joined(separator: "\0") + "\0"
        let s = Git.parseStatus(out)
        #expect(s.branch == "main" && s.head == "0123456" && s.upstream == "origin/main" && s.ahead == 2 && s.behind == 1)
        #expect(s.files == [
            GitChange(path: "staged.txt", kind: .modified, staged: true),
            GitChange(path: "unstaged.txt", kind: .modified, staged: false),
            GitChange(path: "added.txt", kind: .added, staged: true),
            GitChange(path: "deleted.txt", kind: .deleted, staged: false),
            GitChange(path: "new name.txt", oldPath: "old.txt", kind: .renamed, staged: true),
            GitChange(path: "conflict.txt", kind: .conflicted, staged: true),
            GitChange(path: "untracked file.txt", kind: .untracked, staged: false),
        ])
        let detached = Git.parseStatus("# branch.oid (initial)\0# branch.head (detached)\0")
        #expect(detached.branch == nil && detached.head == "" && detached.files.isEmpty && detached.ahead == 0)
    }

    @Test func parseNumstatCases() {
        let out = "3\t1\ta.txt\0-\t-\timg.png\0" + "2\t0\t\0old name.txt\0new name.txt\0"
        let r = Git.parseNumstat(out)
        #expect(r["a.txt"]?.added == 3 && r["a.txt"]?.removed == 1)
        #expect(r["img.png"] != nil && r["img.png"]?.added == nil && r["img.png"]?.removed == nil)
        #expect(r["new name.txt"]?.added == 2 && r["new name.txt"]?.removed == 0)
        #expect(r["old name.txt"] == nil && r.count == 3)
    }

    @Test func parsePRCounts() throws {
        let json = """
        {"number": 7, "url": "https://github.com/o/r/pull/7", "title": "Add x", "state": "OPEN", "isDraft": true,
         "reviewDecision": "", "mergeStateStatus": "",
         "statusCheckRollup": [
          {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS"},
          {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SKIPPED"},
          {"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "FAILURE"},
          {"__typename": "CheckRun", "status": "IN_PROGRESS", "conclusion": ""},
          {"__typename": "StatusContext", "state": "PENDING"}
         ]}
        """
        let pr = try #require(Git.parsePR(Data(json.utf8)))
        #expect(pr.number == 7 && pr.state == "OPEN" && pr.draft && pr.title == "Add x")
        #expect(pr.passed == 2 && pr.failed == 1 && pr.pending == 2)
        #expect(pr.review == nil && pr.mergeState == nil)
        #expect(Git.parsePR(Data("nope".utf8)) == nil)
    }

    @Test func trailerCases() {
        let t = "Co-Authored-By: X"
        #expect(Git.withTrailer("Add x\n\n", trailer: nil) == "Add x")
        #expect(Git.withTrailer("Add x\n", trailer: "") == "Add x")
        #expect(Git.withTrailer("Add x\n", trailer: t) == "Add x\n\n" + t)
        #expect(Git.withTrailer("Add x\n\n" + t + "\n", trailer: t) == "Add x\n\n" + t)
        #expect(Git.coAuthorLine(model: "m") == "Co-Authored-By: AIStudio (m) <noreply@aistudio.local>")
    }

    @Test func diffRowsFromGitOutput() {
        let diff = """
        diff --git a/x b/x
        index 1111111..2222222 100644
        --- a/x
        +++ b/x
        @@ -1,3 +1,4 @@ func f() -> Int
         one
        +two
         three
        -four
        +five
        \\ No newline at end of file
        """
        let rows = DiffNumbers.rows(diff)
        #expect(rows.allSatisfy { !$0.gap })
        #expect(rows.map(\.text) == ["one", "two", "three", "four", "five"])
        #expect(rows.map(\.number) == [1, 2, 3, 3, 4])
        #expect(rows.map(\.sign) == [nil, "+", nil, "-", "+"])
        #expect(DiffNumbers.starts("@@ -3,4 +3,5 @@ func f() -> Int") == (3, 3))
        #expect(DiffNumbers.starts("@@ -3 +4 @@") == (3, 4))
        #expect(DiffNumbers.starts("@@") == (nil, nil))
    }

    @Test func commitOnlyThePathsGiven() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        try "changed\n".write(to: repo.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try "new\n".write(to: repo.appendingPathComponent("keep me.txt"), atomically: true, encoding: .utf8)
        try "staged\n".write(to: repo.appendingPathComponent("staged.txt"), atomically: true, encoding: .utf8)
        _ = await sh("git add staged.txt", repo)  // staged earlier: still left out
        _ = try await Git.commit(repo, message: "Only the new file", paths: ["keep me.txt"])
        #expect(await sh("git show --name-only --format= HEAD", repo).trimmingCharacters(in: .whitespacesAndNewlines) == "keep me.txt")
        let left = try #require(await Git.status(repo)).files.map(\.path).sorted()
        #expect(left == ["a.txt", "staged.txt"])
        do {
            _ = try await Git.commit(repo, message: "Nothing", paths: ["keep me.txt"])
            Issue.record("expected Nothing to commit")
        } catch { #expect((error as? ToolError)?.message == "Nothing to commit.") }
    }

    @Test func realRepoRoundTrip() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let first = await sh("git rev-parse HEAD", repo).trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(await Git.defaultBranch(repo) == "main")
        #expect(await Git.ghReady(repo) == false)
        #expect(await Git.recentLog(repo).count == 1)

        try "base\nmore\n".write(to: repo.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try "x\ny\n".write(to: repo.appendingPathComponent("new file.txt"), atomically: true, encoding: .utf8)
        let st = try #require(await Git.status(repo))
        #expect(st.branch == "main" && st.head.count == 7)
        let a = try #require(st.files.first { $0.path == "a.txt" })
        let n = try #require(st.files.first { $0.path == "new file.txt" })
        #expect(a.kind == .modified && a.added == 1 && a.removed == 0)
        #expect(n.kind == .untracked && n.added == 2 && n.removed == 0)
        for c in [a, n] {
            let d = await Git.diff(repo, change: c)
            #expect(!d.isEmpty)
            #expect(DiffNumbers.rows(d).contains { $0.sign == "+" })
        }

        let msg = Git.withTrailer("Add things", trailer: Git.coAuthorLine(model: "m"))
        let hash = try await Git.commit(repo, message: msg)
        #expect(!hash.isEmpty)
        #expect(try #require(await Git.status(repo)).files.isEmpty)
        #expect(await sh("git log -1 --format=%B", repo).contains(Git.coAuthorLine(model: "m")))
        let committed = await Git.changes(repo, base: first).map(\.path).sorted()
        #expect(committed == ["a.txt", "new file.txt"])
        do {
            _ = try await Git.commit(repo, message: "Empty")
            Issue.record("expected Nothing to commit")
        } catch { #expect((error as? ToolError)?.message == "Nothing to commit.") }

        try "changed\n".write(to: repo.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try "tmp\n".write(to: repo.appendingPathComponent("tmp.txt"), atomically: true, encoding: .utf8)
        let st2 = try #require(await Git.status(repo))
        try await Git.revert(repo, change: try #require(st2.files.first { $0.path == "a.txt" }))
        #expect(try String(contentsOf: repo.appendingPathComponent("a.txt"), encoding: .utf8) == "base\nmore\n")
        try await Git.revert(repo, change: try #require(st2.files.first { $0.path == "tmp.txt" }),
                             trash: { try FileManager.default.removeItem(at: $0) })
        #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("tmp.txt").path))

        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("git-out-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        #expect(await Git.status(outside) == nil)
    }

    @Test func systemPromptGitParagraph() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let p = await ProjectContext.systemPrompt(root: repo, mode: .ask, coAuthor: "Co-Authored-By: X")
        #expect(p.contains("Git: commit only when the user asks"))
        #expect(p.contains("Default branch: main"))
        #expect(p.contains("Recent commits:"))
        #expect(p.contains("Co-Authored-By: X"))
        let plain = await ProjectContext.systemPrompt(root: repo, mode: .ask)
        #expect(plain.contains("Git: commit only") && !plain.contains("End each commit message"))

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("git-np-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let q = await ProjectContext.systemPrompt(root: dir, mode: .ask, coAuthor: "Co-Authored-By: X")
        #expect(q.contains("Not a git repository.") && !q.contains("Git: commit only"))
    }

    @Test func draftsCleanTheReply() async throws {
        let commit = ScriptedModel([[.content("```\nAdd x\n\nBody\n```"), .finish(reason: "stop")]])
        let m = try await GitDrafts.commitMessage(client: commit, model: "m", diff: "d", log: ["abc one"])
        #expect(m == "Add x\n\nBody")
        let user = commit.requests[0].messages.last?.text ?? ""
        #expect(user.hasPrefix("Recent commits:\nabc one\n\nDiff:\nd\n\n") && user.hasSuffix("Write the commit message now."))

        let pr = ScriptedModel([[.content("Title: Add x\n\n## Summary\n- y"), .finish(reason: "stop")]])
        let r = try await GitDrafts.pullRequest(client: pr, model: "m", diff: "d", log: [], branch: "feat", base: "main")
        #expect(r.title == "Add x" && r.body == "## Summary\n- y")

        let empty = ScriptedModel([[.content("  \n"), .finish(reason: "stop")]])
        await #expect(throws: ToolError.self) {
            _ = try await GitDrafts.commitMessage(client: empty, model: "m", diff: "d", log: [])
        }
    }
}
