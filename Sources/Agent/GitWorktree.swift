import Foundation

/// Where a session's worktree stands against its base branch.
public struct WorktreeStatus: Equatable, Sendable {
    public var branch: String
    public var ahead: Int
    public var behind: Int
    public var dirty: Int

    public init(branch: String, ahead: Int, behind: Int, dirty: Int) {
        self.branch = branch
        self.ahead = ahead
        self.behind = behind
        self.dirty = dirty
    }
}

/// git helpers for sessions that work in their own worktree and branch.
public enum GitWorktree {
    private static func git(_ args: String, in dir: URL, timeout: TimeInterval = 120) async -> ShellRunner.Result {
        await ShellRunner().run("git " + args, cwd: dir, timeout: timeout)
    }

    private static func clip(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.count > 500 ? String(t.prefix(500)) + "…" : t
    }

    /// Runs git and returns its output; throws "git <subcommand> failed: …" on a non-zero exit.
    @discardableResult
    private static func gitChecked(_ sub: String, _ args: String, in dir: URL) async throws -> String {
        let r = await git(sub + (args.isEmpty ? "" : " " + args), in: dir)
        guard r.status == 0 else { throw ToolError("git \(sub) failed: \(clip(r.output))") }
        return r.output
    }

    private static func resolved(_ url: URL) -> URL { url.standardizedFileURL.resolvingSymlinksInPath() }

    /// The repository's top folder (`git rev-parse --show-toplevel`), nil when `folder` isn't in one.
    public static func repoRoot(_ folder: URL) async -> URL? {
        let r = await git("rev-parse --show-toplevel", in: folder)
        guard r.status == 0 else { return nil }
        let path = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { return nil }
        return resolved(URL(fileURLWithPath: path, isDirectory: true))
    }

    /// The checked-out branch (`git rev-parse --abbrev-ref HEAD`); a detached HEAD gives its short hash.
    public static func currentBranch(_ repo: URL) async -> String? {
        let r = await git("rev-parse --abbrev-ref HEAD", in: repo)
        guard r.status == 0 else { return nil }
        let name = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        if name == "HEAD" {
            let h = await git("rev-parse --short HEAD", in: repo)
            guard h.status == 0 else { return nil }
            let hash = h.output.trimmingCharacters(in: .whitespacesAndNewlines)
            return hash.isEmpty ? nil : hash
        }
        return name
    }

    /// A branch-safe name from a title: lowercase ASCII letters and digits, anything else becomes "-", runs
    /// collapsed, "-" trimmed at both ends, at most 40 characters (cut, then trimmed again); empty gives "session".
    public static func slug(_ title: String) -> String {
        var out = ""
        for ch in title.unicodeScalars {
            let v = ch.value
            let isLetter = v >= 97 && v <= 122
            let isUpper = v >= 65 && v <= 90
            let isDigit = v >= 48 && v <= 57
            if isLetter || isDigit {
                out.unicodeScalars.append(ch)
            } else if isUpper {
                out.unicodeScalars.append(Unicode.Scalar(v + 32)!)
            } else if !out.hasSuffix("-") {
                out.append("-")
            }
        }
        let dash = CharacterSet(charactersIn: "-")
        var s = out.trimmingCharacters(in: dash)
        if s.count > 40 { s = String(s.prefix(40)).trimmingCharacters(in: dash) }
        return s.isEmpty ? "session" : s
    }

    private static func branchExists(_ name: String, in repo: URL) async -> Bool {
        await git("show-ref --verify --quiet \(shellQuote("refs/heads/" + name))", in: repo).status == 0
    }

    /// Creates a worktree on a new branch from `base` (default HEAD) at `<repo>/.aistudio/worktrees/<last path
    /// component of branch>`. A name clash (folder or branch) suffixes "-2", "-3"… to both.
    public static func create(repo: URL, branch: String, base: String? = nil) async throws -> (path: URL, branch: String) {
        let repo = resolved(repo)
        let parent = repo.appendingPathComponent(".aistudio/worktrees", isDirectory: true)
        try await ensureExcluded(repo: repo)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)

        var n = 1
        var name = branch
        var folder = parent
        while true {
            name = n == 1 ? branch : "\(branch)-\(n)"
            let last = name.split(separator: "/").last.map(String.init) ?? name
            folder = parent.appendingPathComponent(last, isDirectory: true)
            let taken = FileManager.default.fileExists(atPath: folder.path)
            if !taken, !(await branchExists(name, in: repo)) { break }
            n += 1
            if n > 1000 { throw ToolError("git worktree add failed: no free name for \(branch)") }
        }
        var args = "add -b \(shellQuote(name)) \(shellQuote(folder.path))"
        if let base, !base.isEmpty { args += " \(shellQuote(base))" }
        try await gitChecked("worktree", args, in: repo)
        return (resolved(folder), name)
    }

    /// Makes sure `.aistudio/worktrees/` is a line of the repo's `info/exclude`.
    private static func ensureExcluded(repo: URL) async throws {
        let out = try await gitChecked("rev-parse", "--git-common-dir", in: repo)
        let common = out.trimmingCharacters(in: .whitespacesAndNewlines)
        let dir = URL(fileURLWithPath: common, relativeTo: repo).standardizedFileURL
        let infoDir = dir.appendingPathComponent("info", isDirectory: true)
        let file = infoDir.appendingPathComponent("exclude")
        let line = ".aistudio/worktrees/"
        var text = (try? String(contentsOf: file, encoding: .utf8)) ?? ""
        if text.components(separatedBy: "\n").contains(where: { $0.trimmingCharacters(in: .whitespaces) == line }) { return }
        if !text.isEmpty, !text.hasSuffix("\n") { text += "\n" }
        text += line + "\n"
        try FileManager.default.createDirectory(at: infoDir, withIntermediateDirectories: true)
        try text.write(to: file, atomically: true, encoding: .utf8)
    }

    /// Commits ahead of / behind `base`, and the number of changed or untracked files.
    public static func status(_ worktree: URL, base: String) async throws -> WorktreeStatus {
        let counts = try await gitChecked("rev-list", "--left-right --count \(shellQuote(base))...HEAD", in: worktree)
        let parts = counts.split(whereSeparator: { $0 == "\t" || $0 == " " || $0 == "\n" }).compactMap { Int($0) }
        guard parts.count == 2 else { throw ToolError("git rev-list failed: unexpected output \(clip(counts))") }
        let porcelain = try await gitChecked("status", "--porcelain", in: worktree)
        let dirty = porcelain.split(separator: "\n", omittingEmptySubsequences: true).count
        let branch = await currentBranch(worktree) ?? "HEAD"
        return WorktreeStatus(branch: branch, ahead: parts[1], behind: parts[0], dirty: dirty)
    }

    /// Commits on `branch` that `base` doesn't have; 0 on error.
    public static func unmergedCommits(branch: String, base: String, repo: URL) async -> Int {
        let r = await git("rev-list --count \(shellQuote(base))..\(shellQuote(branch))", in: repo)
        guard r.status == 0 else { return 0 }
        return Int(r.output.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
    }

    /// Merges `branch` into the repo's checked-out branch (`--no-ff --no-edit`). Refuses when the main checkout has
    /// uncommitted changes to tracked files. On a conflict the merge is aborted and the conflicted files are named.
    public static func merge(branch: String, into repo: URL) async throws -> String {
        let dirty = try await gitChecked("status", "--porcelain --untracked-files=no", in: repo)
        if !dirty.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            throw ToolError("The main checkout has uncommitted changes to tracked files: commit or stash them before merging \(branch).")
        }
        let r = await git("merge --no-ff --no-edit \(shellQuote(branch))", in: repo)
        if r.status == 0 { return r.output }
        let conflicted = await git("diff --name-only --diff-filter=U", in: repo)
        let files = conflicted.output.split(separator: "\n").map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        _ = await git("merge --abort", in: repo)
        if files.isEmpty { throw ToolError("git merge failed: \(clip(r.output))") }
        throw ToolError("Merging \(branch) conflicts in: \(files.joined(separator: ", ")). Nothing was merged.")
    }

    /// `git worktree remove [--force] <path>` in `repo`; also `git branch -D` when `deleteBranch` is true.
    public static func remove(_ worktree: URL, repo: URL, branch: String?, force: Bool, deleteBranch: Bool) async throws {
        try await gitChecked("worktree", "remove\(force ? " --force" : "") \(shellQuote(worktree.path))", in: repo)
        if deleteBranch, let branch, !branch.isEmpty {
            try await gitChecked("branch", "-D \(shellQuote(branch))", in: repo)
        }
    }

    /// Whether the `gh` command is available and the repo has a remote.
    public static func canCreatePR(_ repo: URL) async -> Bool {
        let r = await ShellRunner().run("command -v gh >/dev/null 2>&1 && git remote | grep -q .", cwd: repo, timeout: 20)
        return r.status == 0
    }
}
