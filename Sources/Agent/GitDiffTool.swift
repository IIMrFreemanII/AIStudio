import Foundation
import LLMClient

/// `git diff` for agents that can't run commands (a review subagent): the uncommitted changes, or everything since a
/// branch or commit.
public struct GitDiffTool: AgentTool {
    public init() {}
    public let name = "git_diff"
    public let description = "Show what changed in the project's git repository: the uncommitted changes against HEAD (untracked files are listed), or with `base` everything since that branch or commit (from where they diverged, uncommitted changes included). `stat` gives only the changed files with line counts; `path` limits it to a file or folder."
    public var parameters: JSONValue {
        schema(["base": ("string", "Branch or commit to compare with, e.g. main (default: HEAD, the uncommitted changes)"),
                "path": ("string", "File or folder to limit the diff to"),
                "stat": ("boolean", "Only the changed files with line counts")], required: [])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let path = args.string("path").flatMap { $0.isEmpty ? nil : $0 }
        let base = args.string("base").flatMap { $0.isEmpty ? nil : $0 }
        return ToolPreview(kind: .read, title: "Diff" + (base.map { " since \($0)" } ?? "") + (path.map { " of \($0)" } ?? ""),
                           paths: [ctx.resolve(path ?? ".")])
    }

    /// A branch, tag or commit as git spells one (`main`, `origin/main`, `HEAD~2`, a hash); never an option.
    static func isRef(_ s: String) -> Bool {
        s.range(of: #"^[A-Za-z0-9_][A-Za-z0-9_./~^@{}-]*$"#, options: .regularExpression) != nil && !s.contains("..")
    }

    /// git's empty tree: what a repository without commits is compared with.
    static let emptyTree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let dir = ctx.root
        guard await Git.run("rev-parse --is-inside-work-tree 2>/dev/null", in: dir, timeout: 20).status == 0 else {
            return ToolOutput("The project isn't a git repository.", isError: true)
        }
        var from = "HEAD"
        if let base = args.string("base"), !base.isEmpty, base != "HEAD" {
            guard Self.isRef(base) else { return ToolOutput("\(base) isn't a branch or commit name.", isError: true) }
            let mb = await Git.run("merge-base \(shellQuote(base)) HEAD 2>/dev/null", in: dir, timeout: 30)
            let sha = mb.output.trimmingCharacters(in: .whitespacesAndNewlines)
            guard mb.status == 0, !sha.isEmpty else {
                return ToolOutput("\(base) isn't a branch or commit this branch shares history with.", isError: true)
            }
            from = sha
        } else if await Git.run("rev-parse --verify -q HEAD", in: dir, timeout: 20).status != 0 {
            from = Self.emptyTree
        }
        var pathspec = ""
        if let p = args.string("path"), !p.isEmpty {
            let url = ctx.resolve(p)
            guard ctx.inProject(url) else { return ToolOutput("\(p) is outside the project.", isError: true) }
            pathspec = " -- " + shellQuote(url.path)
        }
        let stat = args.bool("stat") ? " --stat=200" : ""
        let diff = await Git.run("-c core.quotepath=off --no-pager diff --no-ext-diff --no-color -M\(stat) \(from)\(pathspec) 2>&1",
                                 in: dir, timeout: 60)
        guard diff.status == 0 else { return ToolOutput("git diff failed: \(diff.output)", isError: true) }
        var out = diff.output.trimmingCharacters(in: .newlines)
        let untracked = await Git.run("-c core.quotepath=off ls-files --others --exclude-standard\(pathspec) 2>/dev/null",
                                      in: dir, timeout: 30)
        let files = untracked.status == 0 ? untracked.output.split(separator: "\n") : []
        if !files.isEmpty {
            out += (out.isEmpty ? "" : "\n\n") + "Untracked files (new, not in the diff; read them with read_file):\n"
                + files.joined(separator: "\n")
        }
        return ToolOutput(clip(out.isEmpty ? "(no changes)" : out))
    }
}
