import Foundation
import LLMClient

/// What happened to a file.
public enum GitChangeKind: String, Codable, Sendable { case modified, added, deleted, renamed, untracked, conflicted }

/// One changed file in a repository.
public struct GitChange: Equatable, Sendable, Codable, Identifiable {
    /// Repo-relative, as git prints it.
    public var path: String
    /// The previous path (renames only).
    public var oldPath: String?
    public var kind: GitChangeKind
    /// The index has changes for it (porcelain X column is not ".").
    public var staged: Bool
    /// Lines added / removed, from numstat; nil for binary or unknown.
    public var added: Int?
    public var removed: Int?
    public var id: String { path }

    public init(path: String, oldPath: String? = nil, kind: GitChangeKind, staged: Bool = false, added: Int? = nil,
                removed: Int? = nil) {
        self.path = path
        self.oldPath = oldPath
        self.kind = kind
        self.staged = staged
        self.added = added
        self.removed = removed
    }
}

/// The branch, its upstream and the changed files of a working tree.
public struct GitStatus: Equatable, Sendable {
    /// nil when HEAD is detached.
    public var branch: String?
    /// Short oid, or "" before the first commit.
    public var head: String
    public var upstream: String?
    public var ahead: Int
    public var behind: Int
    public var files: [GitChange]

    public init(branch: String? = nil, head: String = "", upstream: String? = nil, ahead: Int = 0, behind: Int = 0,
                files: [GitChange] = []) {
        self.branch = branch
        self.head = head
        self.upstream = upstream
        self.ahead = ahead
        self.behind = behind
        self.files = files
    }
}

/// A GitHub pull request as `gh pr view` reports it.
public struct PRInfo: Equatable, Sendable, Codable {
    public var number: Int
    public var url: String
    public var title: String
    /// "OPEN", "MERGED" or "CLOSED".
    public var state: String
    public var draft: Bool
    /// gh's reviewDecision ("APPROVED", "CHANGES_REQUESTED", "REVIEW_REQUIRED"), nil when empty.
    public var review: String?
    public var passed: Int
    public var failed: Int
    public var pending: Int
    /// gh's mergeStateStatus, nil when empty.
    public var mergeState: String?

    public init(number: Int, url: String, title: String, state: String, draft: Bool = false, review: String? = nil,
                passed: Int = 0, failed: Int = 0, pending: Int = 0, mergeState: String? = nil) {
        self.number = number
        self.url = url
        self.title = title
        self.state = state
        self.draft = draft
        self.review = review
        self.passed = passed
        self.failed = failed
        self.pending = pending
        self.mergeState = mergeState
    }
}

/// git and GitHub CLI helpers for the changes panel, commits and pull requests.
public enum Git {
    public enum MergeMethod: String, Sendable, CaseIterable { case squash, rebase, merge }

    /// Runs `git <args>` in dir.
    static func run(_ args: String, in dir: URL, timeout: TimeInterval = 120) async -> ShellRunner.Result {
        await ShellRunner().run("git " + args, cwd: dir, timeout: timeout)
    }

    private static let prFields = "number,url,title,state,isDraft,reviewDecision,statusCheckRollup,mergeStateStatus"

    private static func tail(_ s: String) -> String {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.count > 500 ? "…" + String(t.suffix(500)) : t
    }

    /// The repository's top folder, or `root` when it can't be found (paths git prints are relative to it).
    private static func top(_ root: URL) async -> URL {
        let r = await run("rev-parse --show-toplevel 2>/dev/null", in: root, timeout: 20)
        let path = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
        guard r.status == 0, !path.isEmpty else { return root }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    // MARK: parsers

    /// Reads `git status --porcelain=v2 --branch -z` output.
    public static func parseStatus(_ output: String) -> GitStatus {
        var status = GitStatus()
        let fields = output.components(separatedBy: "\0")
        var i = 0
        while i < fields.count {
            let f = fields[i]
            i += 1
            if f.hasPrefix("# ") {
                let parts = f.dropFirst(2).split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
                guard parts.count == 2 else { continue }
                let value = String(parts[1])
                switch parts[0] {
                case "branch.oid": status.head = value == "(initial)" ? "" : String(value.prefix(7))
                case "branch.head": status.branch = value == "(detached)" ? nil : value
                case "branch.upstream": status.upstream = value
                case "branch.ab":
                    for p in value.split(separator: " ") {
                        if p.hasPrefix("+") { status.ahead = Int(p.dropFirst()) ?? 0 }
                        if p.hasPrefix("-") { status.behind = Int(p.dropFirst()) ?? 0 }
                    }
                default: break
                }
                continue
            }
            guard let tag = f.first else { continue }
            func rest(_ n: Int) -> (xy: String, path: String)? {
                let p = f.split(separator: " ", maxSplits: n, omittingEmptySubsequences: false)
                guard p.count == n + 1 else { return nil }
                return (String(p[1]), String(p[n]))
            }
            switch tag {
            case "1":
                guard let (xy, path) = rest(8) else { continue }
                status.files.append(GitChange(path: path, kind: kind(xy), staged: xy.first != "."))
            case "2":
                guard let (xy, path) = rest(9) else { continue }
                let orig = i < fields.count ? fields[i] : ""
                i += 1
                status.files.append(GitChange(path: path, oldPath: orig.isEmpty ? nil : orig, kind: .renamed,
                                              staged: xy.first != "."))
            case "u":
                guard let (xy, path) = rest(10) else { continue }
                status.files.append(GitChange(path: path, kind: .conflicted, staged: xy.first != "."))
            case "?":
                status.files.append(GitChange(path: String(f.dropFirst(2)), kind: .untracked))
            default:
                continue  // "!" ignored files, anything unknown
            }
        }
        return status
    }

    private static func kind(_ xy: String) -> GitChangeKind {
        if xy.contains("U") { return .conflicted }
        if xy.contains("A") { return .added }
        if xy.contains("D") { return .deleted }
        return .modified
    }

    /// Reads `git diff --numstat -z` output, keyed by (new) path; binary files have nil counts.
    public static func parseNumstat(_ output: String) -> [String: (added: Int?, removed: Int?)] {
        var out: [String: (added: Int?, removed: Int?)] = [:]
        let fields = output.components(separatedBy: "\0")
        var i = 0
        while i < fields.count {
            let parts = fields[i].split(separator: "\t", maxSplits: 2, omittingEmptySubsequences: false)
            guard parts.count == 3 else { i += 1; continue }
            let a = Int(parts[0]), r = Int(parts[1])
            if parts[2].isEmpty {
                // A rename: the old and new paths follow as their own fields.
                if i + 2 < fields.count { out[fields[i + 2]] = (a, r) }
                i += 3
            } else {
                out[String(parts[2])] = (a, r)
                i += 1
            }
        }
        return out
    }

    /// Reads `gh pr view --json number,url,title,state,isDraft,reviewDecision,statusCheckRollup,mergeStateStatus`.
    public static func parsePR(_ json: Data) -> PRInfo? {
        guard let obj = try? JSONSerialization.jsonObject(with: json) as? [String: Any],
              let number = obj["number"] as? Int, let url = obj["url"] as? String else { return nil }
        var passed = 0, failed = 0, pending = 0
        for item in obj["statusCheckRollup"] as? [[String: Any]] ?? [] {
            if item["__typename"] as? String == "StatusContext" {
                switch item["state"] as? String ?? "" {
                case "PENDING", "EXPECTED": pending += 1
                case "FAILURE", "ERROR": failed += 1
                default: passed += 1
                }
            } else {
                if (item["status"] as? String) != "COMPLETED" { pending += 1; continue }
                switch item["conclusion"] as? String ?? "" {
                case "FAILURE", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE": failed += 1
                default: passed += 1
                }
            }
        }
        func nonEmpty(_ key: String) -> String? {
            guard let s = obj[key] as? String, !s.isEmpty else { return nil }
            return s
        }
        return PRInfo(number: number, url: url, title: obj["title"] as? String ?? "", state: obj["state"] as? String ?? "",
                      draft: obj["isDraft"] as? Bool ?? false, review: nonEmpty("reviewDecision"), passed: passed,
                      failed: failed, pending: pending, mergeState: nonEmpty("mergeStateStatus"))
    }

    /// `message` without trailing whitespace, then a blank line and `trailer` (unless the message has that line).
    public static func withTrailer(_ message: String, trailer: String?) -> String {
        var body = message
        while let last = body.last, last.isWhitespace || last.isNewline { body.removeLast() }
        guard let trailer, !trailer.isEmpty else { return body }
        if body.components(separatedBy: "\n").contains(where: { $0.trimmingCharacters(in: .whitespaces) == trailer }) {
            return body
        }
        return body + "\n\n" + trailer
    }

    /// The trailer for commits the agent writes.
    public static func coAuthorLine(model: String) -> String {
        "Co-Authored-By: AIStudio (\(model)) <noreply@aistudio.local>"
    }

    // MARK: reading

    /// The working tree's branch and changed files with line counts; nil when `root` isn't in a git repo.
    public static func status(_ root: URL) async -> GitStatus? {
        let r = await run("status --porcelain=v2 --branch -z --untracked-files=all 2>/dev/null", in: root, timeout: 30)
        guard r.status == 0 else { return nil }
        var status = parseStatus(r.output)
        let numstat = await run((status.head.isEmpty ? "diff --cached" : "diff HEAD") + " --numstat -z 2>/dev/null", in: root,
                                timeout: 30)
        let counts = numstat.status == 0 ? parseNumstat(numstat.output) : [:]
        let base = await top(root)
        for i in status.files.indices {
            if status.files[i].kind == .untracked {
                status.files[i].added = lineCount(base.appendingPathComponent(status.files[i].path))
                status.files[i].removed = 0
            } else if let c = counts[status.files[i].path] {
                status.files[i].added = c.added
                status.files[i].removed = c.removed
            }
        }
        return status
    }

    /// Lines in a text file: "\n" count, plus one when the last line has no newline. nil for binary data or > 2 MB.
    private static func lineCount(_ url: URL) -> Int? {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? nil
        guard let size, size <= 2_000_000, let data = try? Data(contentsOf: url),
              let text = String(data: data, encoding: .utf8) else { return nil }
        if text.isEmpty { return 0 }
        let newlines = text.utf8.reduce(0) { $1 == 10 ? $0 + 1 : $0 }
        return newlines + (text.hasSuffix("\n") ? 0 : 1)
    }

    private static func mergeBase(_ root: URL, _ base: String) async -> String? {
        let r = await run("merge-base \(shellQuote(base)) HEAD 2>/dev/null", in: root, timeout: 30)
        let mb = r.output.trimmingCharacters(in: .whitespacesAndNewlines)
        return r.status == 0 && !mb.isEmpty ? mb : nil
    }

    /// Everything that differs from the merge base of `base` and HEAD, working tree and untracked files included.
    /// Empty on any error.
    public static func changes(_ root: URL, base: String) async -> [GitChange] {
        guard let mb = await mergeBase(root, base) else { return [] }
        let names = await run("diff \(shellQuote(mb)) --name-status -M -z 2>/dev/null", in: root, timeout: 30)
        guard names.status == 0 else { return [] }
        let numstat = await run("diff \(shellQuote(mb)) --numstat -M -z 2>/dev/null", in: root, timeout: 30)
        let counts = numstat.status == 0 ? parseNumstat(numstat.output) : [:]
        var out: [GitChange] = []
        let fields = names.output.components(separatedBy: "\0")
        var i = 0
        while i + 1 < fields.count {
            let code = fields[i]
            guard let letter = code.first else { i += 1; continue }
            var change: GitChange
            if letter == "R" || letter == "C" {
                guard i + 2 < fields.count else { break }
                change = GitChange(path: fields[i + 2], oldPath: fields[i + 1], kind: letter == "R" ? .renamed : .added)
                i += 3
            } else {
                let kind: GitChangeKind
                switch letter {
                case "A": kind = .added
                case "D": kind = .deleted
                case "U": kind = .conflicted
                default: kind = .modified
                }
                change = GitChange(path: fields[i + 1], kind: kind)
                i += 2
            }
            if let c = counts[change.path] { change.added = c.added; change.removed = c.removed }
            out.append(change)
        }
        if let st = await status(root) {
            for f in st.files where f.kind == .untracked && !out.contains(where: { $0.path == f.path }) { out.append(f) }
        }
        return out
    }

    /// Unified diff text for one file: against HEAD (`base` nil) or against the merge base with `base`.
    /// "" on error.
    public static func diff(_ root: URL, change: GitChange, base: String? = nil) async -> String {
        let dir = await top(root)
        if change.kind == .untracked {
            let r = await run("diff --no-index -- /dev/null \(shellQuote(change.path))", in: dir, timeout: 30)
            return r.status == 0 || r.status == 1 ? r.output : ""
        }
        var paths = shellQuote(change.path)
        if let old = change.oldPath, !old.isEmpty { paths = shellQuote(old) + " " + paths }
        let from: String
        if let base {
            guard let mb = await mergeBase(dir, base) else { return "" }
            from = "diff \(shellQuote(mb)) -M"
        } else {
            let hasHead = await run("rev-parse --verify -q HEAD", in: dir, timeout: 20).status == 0
            from = hasHead ? "diff HEAD -M" : (change.staged ? "diff --cached -M" : "diff")
        }
        let r = await run("--literal-pathspecs \(from) --no-ext-diff -- \(paths) 2>/dev/null", in: dir, timeout: 30)
        return r.status == 0 ? r.output : ""
    }

    /// The branch pull requests target: origin's HEAD, else main or master if present, else the current branch.
    public static func defaultBranch(_ root: URL) async -> String {
        let sym = await run("symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null", in: root, timeout: 20)
        var name = sym.output.trimmingCharacters(in: .whitespacesAndNewlines)
        if sym.status == 0, !name.isEmpty {
            if name.hasPrefix("origin/") { name.removeFirst("origin/".count) }
            return name
        }
        for candidate in ["main", "master"] {
            if await run("show-ref --verify --quiet refs/heads/\(candidate)", in: root, timeout: 20).status == 0 {
                return candidate
            }
        }
        let cur = await run("branch --show-current 2>/dev/null", in: root, timeout: 20)
        let current = cur.output.trimmingCharacters(in: .whitespacesAndNewlines)
        return cur.status == 0 && !current.isEmpty ? current : "main"
    }

    /// The last `count` commits as one-line summaries; [] on error or before the first commit.
    public static func recentLog(_ root: URL, count: Int = 5) async -> [String] {
        let r = await run("log --oneline -n \(max(1, count)) 2>/dev/null", in: root, timeout: 20)
        guard r.status == 0 else { return [] }
        return r.output.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
    }

    // MARK: changing

    /// Undoes a file's changes (modified, deleted: restored from HEAD; added and untracked: removed through `trash`;
    /// renamed: the old file is restored and the new one removed). Conflicted files throw.
    public static func revert(_ root: URL, change: GitChange,
                              trash: (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }) async throws {
        let dir = await top(root)
        func restore(_ args: String, _ path: String) async throws {
            let r = await run("--literal-pathspecs restore \(args) -- \(shellQuote(path))", in: dir)
            guard r.status == 0 else { throw ToolError("git restore failed: \(tail(r.output))") }
        }
        func trashFile(_ path: String) throws {
            let url = dir.appendingPathComponent(path)
            if FileManager.default.fileExists(atPath: url.path) { try trash(url) }
        }
        switch change.kind {
        case .conflicted:
            throw ToolError("Resolve the conflict in \(change.path) first.")
        case .modified, .deleted:
            try await restore("--source=HEAD --staged --worktree", change.path)
        case .added:
            // Before the first commit there is no HEAD to restore the index from.
            let r = await run("--literal-pathspecs restore --staged -- \(shellQuote(change.path))", in: dir)
            if r.status != 0 {
                let rm = await run("--literal-pathspecs rm --cached -q -f -- \(shellQuote(change.path))", in: dir)
                guard rm.status == 0 else { throw ToolError("git restore failed: \(tail(r.output))") }
            }
            try trashFile(change.path)
        case .untracked:
            try trashFile(change.path)
        case .renamed:
            if let old = change.oldPath, !old.isEmpty { try await restore("--source=HEAD --staged --worktree", old) }
            _ = await run("--literal-pathspecs restore --staged -- \(shellQuote(change.path))", in: dir)
            try trashFile(change.path)
        }
    }

    /// Commits `paths` (nil: everything, with what was staged before) with `message`; returns the short hash. With
    /// paths, only those go in, whatever else is staged.
    public static func commit(_ root: URL, message: String, paths: [String]? = nil) async throws -> String {
        let dir = await top(root)
        let add: String
        if let paths, !paths.isEmpty {
            add = "--literal-pathspecs add -A -- " + paths.map(shellQuote).joined(separator: " ")
        } else {
            add = "add -A"
        }
        let a = await run(add, in: dir)
        guard a.status == 0 else { throw ToolError("git add failed: \(tail(a.output))") }
        // With paths, only those go in, even when other files were staged before.
        let only = paths.map { $0.isEmpty ? "" : " -- " + $0.map(shellQuote).joined(separator: " ") } ?? ""
        let pathspec = only.isEmpty ? "" : "--literal-pathspecs "
        if await run(pathspec + "diff --cached --quiet" + only, in: dir, timeout: 30).status == 0 { throw ToolError("Nothing to commit.") }
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("aistudio-commit-\(UUID().uuidString).txt")
        try message.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        let c = await run(pathspec + "commit -F \(shellQuote(file.path))" + only, in: dir, timeout: 180)
        guard c.status == 0 else { throw ToolError("git commit failed: \(tail(c.output))") }
        let h = await run("rev-parse --short HEAD", in: dir, timeout: 20)
        return h.output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `git push -u origin <branch>`.
    public static func push(_ root: URL, branch: String) async throws {
        let r = await run("push -u origin \(shellQuote(branch))", in: root, timeout: 180)
        guard r.status == 0 else { throw ToolError("git push failed: \(tail(r.output))") }
    }

    // MARK: GitHub CLI

    /// Whether `gh` is installed and signed in, and the repo has an origin remote. Only the active account counts:
    /// plain `gh auth status` fails when any other signed-in account has a stale token (`--active` is gh ≥ 2.40;
    /// older ones fall back to the plain check).
    public static func ghReady(_ root: URL) async -> Bool {
        let r = await ShellRunner().run("command -v gh >/dev/null 2>&1 && { gh auth status --active >/dev/null 2>&1 || gh auth status >/dev/null 2>&1; } && git remote get-url origin >/dev/null 2>&1",
                                        cwd: root, timeout: 20)
        return r.status == 0
    }

    /// The pull request for `branch`; nil when there is none or gh fails.
    public static func pullRequest(_ root: URL, branch: String) async -> PRInfo? {
        let r = await ShellRunner().run("gh pr view \(shellQuote(branch)) --json \(prFields) 2>/dev/null", cwd: root, timeout: 30)
        guard r.status == 0, let data = r.output.data(using: .utf8) else { return nil }
        return parsePR(data)
    }

    /// Opens a pull request and returns its URL.
    public static func createPullRequest(_ root: URL, title: String, body: String, base: String, head: String,
                                         draft: Bool) async throws -> String {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("aistudio-pr-\(UUID().uuidString).md")
        try body.write(to: file, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: file) }
        let cmd = "gh pr create --title \(shellQuote(title)) --body-file \(shellQuote(file.path)) --base \(shellQuote(base)) --head \(shellQuote(head))"
            + (draft ? " --draft" : "")
        let r = await ShellRunner().run(cmd, cwd: root, timeout: 120)
        guard r.status == 0 else { throw ToolError("gh pr create failed: \(tail(r.output))") }
        let url = r.output.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            .last(where: { $0.hasPrefix("https://") })
        guard let url else { throw ToolError("gh pr create gave no URL: \(tail(r.output))") }
        return url
    }

    /// `gh pr merge <number> --<method>`.
    public static func mergePullRequest(_ root: URL, number: Int, method: MergeMethod) async throws {
        let r = await ShellRunner().run("gh pr merge \(number) --\(method.rawValue)", cwd: root, timeout: 120)
        guard r.status == 0 else { throw ToolError("gh pr merge failed: \(tail(r.output))") }
    }
}

/// Model-written commit messages and pull requests.
public enum GitDrafts {
    static let commitSystem = "You write git commit messages. Reply with the message only: a summary line under 72 characters in the imperative mood, matching the style of the recent commits, then a blank line and a short body saying what changed and why when that isn't obvious. No code fences, no quotes, no trailers."
    static let prSystem = "You write GitHub pull requests. Reply with the title on the first line (under 70 characters, no prefix), then a blank line, then a Markdown body with a '## Summary' section of bullets and a '## Test plan' section. No code fences around the whole reply."

    /// A commit message for `diff` in the style of `log` (recent one-line commits). One model call, no tools.
    public static func commitMessage(client: any ChatStreaming, model: String, diff: String, log: [String]) async throws -> String {
        let reply = try await ask(client, model, system: commitSystem, log: log, diff: diff, ending: "Write the commit message now.")
        let text = clean(reply)
        if text.isEmpty { throw ToolError("the model returned an empty message") }
        return text
    }

    /// A pull request title and body for the branch's `diff` against `base`.
    public static func pullRequest(client: any ChatStreaming, model: String, diff: String, log: [String], branch: String,
                                   base: String) async throws -> (title: String, body: String) {
        let reply = try await ask(client, model, system: prSystem, log: log, diff: diff,
                                  ending: "Branch \(branch) into \(base).\n\nWrite the pull request now.")
        var lines = clean(reply).components(separatedBy: "\n")
        while let first = lines.first, first.trimmingCharacters(in: .whitespaces).isEmpty { lines.removeFirst() }
        guard var title = lines.first?.trimmingCharacters(in: .whitespaces) else {
            throw ToolError("the model returned an empty message")
        }
        lines.removeFirst()
        if title.lowercased().hasPrefix("title:") { title = String(title.dropFirst(6)) }
        while title.hasPrefix("#") { title.removeFirst() }
        title = title.trimmingCharacters(in: .whitespaces)
        if title.isEmpty { throw ToolError("the model returned an empty message") }
        return (title, lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines))
    }

    private static func ask(_ client: any ChatStreaming, _ model: String, system: String, log: [String], diff: String,
                            ending: String) async throws -> String {
        let clipped = diff.count > 24_000 ? String(diff.prefix(24_000)) + "\n[diff clipped]" : diff
        let user = "Recent commits:\n\(log.joined(separator: "\n"))\n\nDiff:\n\(clipped)\n\n\(ending)"
        let request = ChatRequest(model: model, messages: [.system(system), .user(user)])
        var acc = ReplyAccumulator(inlineThinking: true)
        for try await e in client.stream(request) { acc.add(e) }
        acc.finish()
        return acc.content
    }

    /// Trims the reply and drops a ``` fence that wraps all of it.
    private static func clean(_ reply: String) -> String {
        var text = reply.trimmingCharacters(in: .whitespacesAndNewlines)
        if text.hasPrefix("```"), text.hasSuffix("```"), text.count >= 6 {
            var lines = text.components(separatedBy: "\n")
            if lines.count >= 2 {
                lines.removeFirst()
                lines.removeLast()
                text = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return text
    }
}
