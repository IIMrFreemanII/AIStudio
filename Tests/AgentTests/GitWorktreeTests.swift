import Foundation
import Testing
@testable import Agent

private func sh(_ cmd: String, _ cwd: URL) async -> String {
    let r = await ShellRunner().run(cmd, cwd: cwd)
    return r.output
}

private func makeRepo() async throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("gwt-" + UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    _ = await sh("git init -q -b main && git config user.email t@t && git config user.name t && echo base > a.txt && git add -A && git commit -q -m init", dir)
    return dir.standardizedFileURL.resolvingSymlinksInPath()
}

private func same(_ a: URL?, _ b: URL) -> Bool {
    a?.standardizedFileURL.resolvingSymlinksInPath().path == b.standardizedFileURL.resolvingSymlinksInPath().path
}

struct GitWorktreeTests {
    @Test func repoRootAndCurrentBranch() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let sub = repo.appendingPathComponent("x/y", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        #expect(same(await GitWorktree.repoRoot(sub), repo))
        #expect(await GitWorktree.currentBranch(repo) == "main")
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("gwt-out-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        #expect(await GitWorktree.repoRoot(outside) == nil)
        _ = await sh("git checkout -q --detach", repo)
        let detached = await GitWorktree.currentBranch(repo)
        #expect(detached != nil && detached != "HEAD" && detached != "main")
    }

    @Test func slugCases() {
        #expect(GitWorktree.slug("Fix the Bug!") == "fix-the-bug")
        #expect(GitWorktree.slug("  --Hello,   World--  ") == "hello-world")
        #expect(GitWorktree.slug("Добавить кнопку") == "session")
        #expect(GitWorktree.slug("café au lait") == "caf-au-lait")
        #expect(GitWorktree.slug("") == "session")
        #expect(GitWorktree.slug("!!!") == "session")
        let long = GitWorktree.slug(String(repeating: "abcd efgh ", count: 20))
        #expect(long.count <= 40)
        #expect(!long.hasSuffix("-") && !long.hasPrefix("-"))
        #expect(GitWorktree.slug(String(repeating: "a", count: 39) + " b") == String(repeating: "a", count: 39))
    }

    @Test func createMakesWorktreeAndExcludeOnce() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let first = try await GitWorktree.create(repo: repo, branch: "session/fix-bug")
        #expect(first.branch == "session/fix-bug")
        #expect(same(first.path, repo.appendingPathComponent(".aistudio/worktrees/fix-bug")))
        #expect(FileManager.default.fileExists(atPath: first.path.path))
        #expect(await GitWorktree.currentBranch(first.path) == "session/fix-bug")

        let second = try await GitWorktree.create(repo: repo, branch: "session/fix-bug")
        #expect(second.branch == "session/fix-bug-2")
        #expect(same(second.path, repo.appendingPathComponent(".aistudio/worktrees/fix-bug-2")))
        #expect(await GitWorktree.currentBranch(second.path) == "session/fix-bug-2")

        let exclude = try String(contentsOf: repo.appendingPathComponent(".git/info/exclude"), encoding: .utf8)
        #expect(exclude.components(separatedBy: "\n").filter { $0 == ".aistudio/worktrees/" }.count == 1)
        // Worktrees don't show up as untracked in the main checkout.
        #expect((await sh("git status --porcelain", repo)).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test func createSuffixesWhenOnlyBranchExists() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        _ = await sh("git branch taken", repo)
        let r = try await GitWorktree.create(repo: repo, branch: "taken")
        #expect(r.branch == "taken-2")
        #expect(r.path.lastPathComponent == "taken-2")
    }

    @Test func statusAndUnmerged() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let wt = try await GitWorktree.create(repo: repo, branch: "feat")
        _ = await sh("echo one > b.txt && git add -A && git commit -q -m one && echo two > c.txt", wt.path)
        let st = try await GitWorktree.status(wt.path, base: "main")
        #expect(st == WorktreeStatus(branch: "feat", ahead: 1, behind: 0, dirty: 1))
        #expect(await GitWorktree.unmergedCommits(branch: "feat", base: "main", repo: repo) == 1)
        #expect(await GitWorktree.unmergedCommits(branch: "nope", base: "main", repo: repo) == 0)
        _ = await sh("echo m > m.txt && git add -A && git commit -q -m m", repo)
        let st2 = try await GitWorktree.status(wt.path, base: "main")
        #expect(st2.ahead == 1 && st2.behind == 1)
    }

    @Test func mergeSuccess() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let wt = try await GitWorktree.create(repo: repo, branch: "feat")
        _ = await sh("echo one > b.txt && git add -A && git commit -q -m one", wt.path)
        let out = try await GitWorktree.merge(branch: "feat", into: repo)
        #expect(!out.isEmpty)
        #expect(FileManager.default.fileExists(atPath: repo.appendingPathComponent("b.txt").path))
        #expect(await GitWorktree.unmergedCommits(branch: "feat", base: "main", repo: repo) == 0)
    }

    @Test func mergeRefusedWhenMainDirty() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let wt = try await GitWorktree.create(repo: repo, branch: "feat")
        _ = await sh("echo one > b.txt && git add -A && git commit -q -m one", wt.path)
        _ = await sh("echo changed > a.txt", repo)
        await #expect(throws: ToolError.self) { try await GitWorktree.merge(branch: "feat", into: repo) }
        #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("b.txt").path))
    }

    @Test func mergeConflictIsAborted() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let wt = try await GitWorktree.create(repo: repo, branch: "feat")
        _ = await sh("echo theirs > a.txt && git commit -q -am theirs", wt.path)
        _ = await sh("echo ours > a.txt && git commit -q -am ours", repo)
        do {
            _ = try await GitWorktree.merge(branch: "feat", into: repo)
            Issue.record("expected a conflict")
        } catch let e as ToolError {
            #expect(e.message.contains("a.txt"))
            #expect(e.message.contains("Nothing was merged"))
        }
        #expect((await sh("git status --porcelain", repo)).trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent(".git/MERGE_HEAD").path))
        #expect(try String(contentsOf: repo.appendingPathComponent("a.txt"), encoding: .utf8) == "ours\n")
    }

    @Test func removeWithAndWithoutBranch() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        let a = try await GitWorktree.create(repo: repo, branch: "keep")
        try await GitWorktree.remove(a.path, repo: repo, branch: a.branch, force: false, deleteBranch: false)
        #expect(!FileManager.default.fileExists(atPath: a.path.path))
        #expect((await sh("git branch --list keep", repo)).contains("keep"))

        let b = try await GitWorktree.create(repo: repo, branch: "drop")
        _ = await sh("echo x > dirty.txt", b.path)
        await #expect(throws: ToolError.self) {
            try await GitWorktree.remove(b.path, repo: repo, branch: b.branch, force: false, deleteBranch: true)
        }
        try await GitWorktree.remove(b.path, repo: repo, branch: b.branch, force: true, deleteBranch: true)
        #expect(!FileManager.default.fileExists(atPath: b.path.path))
        #expect(!(await sh("git branch --list drop", repo)).contains("drop"))
    }

    @Test func canCreatePRNeedsRemote() async throws {
        let repo = try await makeRepo()
        defer { try? FileManager.default.removeItem(at: repo) }
        #expect(await GitWorktree.canCreatePR(repo) == false)
    }
}
