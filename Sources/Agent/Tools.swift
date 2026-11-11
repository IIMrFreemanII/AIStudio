import Foundation
import LLMClient

/// A tool call's arguments (the model's JSON), with typed getters.
public struct ToolArgs: @unchecked Sendable {
    public let raw: [String: Any]

    public init(json: String) throws {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { raw = [:]; return }
        guard let obj = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any] else {
            throw ToolError("the arguments aren't a JSON object: \(trimmed.prefix(200))")
        }
        raw = obj
    }

    public init(_ raw: [String: Any]) { self.raw = raw }

    public func string(_ key: String) -> String? { raw[key] as? String }
    public func int(_ key: String) -> Int? { (raw[key] as? Int) ?? (raw[key] as? Double).map(Int.init) ?? (raw[key] as? String).flatMap(Int.init) }
    public func bool(_ key: String) -> Bool { (raw[key] as? Bool) ?? ((raw[key] as? String) == "true") }

    public func required(_ key: String) throws -> String {
        guard let s = string(key) else { throw ToolError("missing \"\(key)\"") }
        return s
    }
}

public struct ToolError: LocalizedError, Equatable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

/// What a call will do, shown on its card and in a permission request.
public struct ToolPreview: Equatable, Sendable {
    public enum Kind: String, Codable, Sendable { case read, diff, command, todo, graph, run }
    public var kind: Kind
    /// "Read Sources/App.swift", "Edit README.md", "Run swift build".
    public var title: String
    /// A unified diff, a command, or nothing.
    public var detail: String?
    public var added = 0
    public var removed = 0
    /// Every path the call touches, resolved (for the in-project check).
    public var paths: [URL] = []
    public var command: String?
    /// The host a `.network` call reads from (for the permission rules and "Always allow").
    public var domain: String?

    public init(kind: Kind, title: String, detail: String? = nil, paths: [URL] = [], command: String? = nil) {
        self.kind = kind
        self.title = title
        self.detail = detail
        self.paths = paths
        self.command = command
    }
}

/// A tool's result: the text the model gets, whether it failed, and images for it to look at.
public struct ToolOutput: Equatable, Sendable {
    public var text: String
    public var isError: Bool
    /// `data:` URLs (JPEG), sent to a vision model after the tool results.
    public var images: [String]
    /// What the card shows once the call is done, in place of the preview's detail (an answered question's summary).
    public var detail: String?
    public init(_ text: String, isError: Bool = false, images: [String] = [], detail: String? = nil) {
        self.text = text
        self.isError = isError
        self.images = images
        self.detail = detail
    }
}

/// What tools share during a run: the project folder, which files were read (and when), the to-do list.
public final class ToolContext: @unchecked Sendable {
    public let root: URL
    private let lock = NSLock()
    private var stamps: [String: Date] = [:]
    private var _todos: [Todo] = []
    /// Streams a running command's output to its card: (call id, chunk).
    public var onOutput: (@Sendable (String, String) -> Void)?
    /// Asks the user the questions of an `ask_user` call (call id, questions) and waits: one answer per question, or
    /// nil when they skipped them (or nobody can answer: no UI).
    public var askUser: (@Sendable (String, [UserQuestion]) async -> [UserAnswer]?)?
    /// Shows an `exit_plan` plan for approval (call id, plan markdown) and waits for the answer (nil: no UI).
    public var approvePlan: (@Sendable (String, String) async -> PlanAnswer)?
    /// Saves files before the agent changes them, so the user can rewind (nil: no checkpoints).
    public var checkpoints: CheckpointStore?
    /// The user message the current reply answers: what checkpoints are filed under.
    public var turnID: UUID?
    /// Commands left running in the background (`run_command` with `background`); outlive one reply.
    public var jobs: BackgroundJobs?
    /// Folders outside the project that may be read without asking (the skills' folders).
    public var readRoots: [URL] = []
    private var _callID = ""
    private var _mode = PermissionMode.ask
    /// The call being run. Calls run at the same time (`task`) each see their own, through `ToolCallScope`.
    public var callID: String {
        get { ToolCallScope.id ?? lock.withLock { _callID } }
        set { lock.withLock { _callID = newValue } }
    }
    /// The permission mode now: the loop starts it from its config, and `exit_plan` changes it.
    public var mode: PermissionMode {
        get { lock.withLock { _mode } }
        set { lock.withLock { _mode = newValue } }
    }

    public struct Todo: Codable, Equatable, Sendable {
        public var content: String
        public var status: String  // pending, in_progress, completed

        public init(content: String, status: String) {
            self.content = content
            self.status = status
        }
    }

    public init(root: URL) { self.root = root.standardizedFileURL.resolvingSymlinksInPath() }

    public var todos: [Todo] { lock.withLock { _todos } }
    func setTodos(_ t: [Todo]) { lock.withLock { _todos = t } }

    /// A path from the model, relative to the project or absolute.
    public func resolve(_ path: String) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        let url = expanded.hasPrefix("/") ? URL(fileURLWithPath: expanded) : root.appendingPathComponent(expanded)
        return url.standardizedFileURL.resolvingSymlinksInPath()
    }

    public func inProject(_ url: URL) -> Bool { Self.contains(root, url) }

    /// Reads of `url` are free: it's in the project or in one of `readRoots`.
    public func canRead(_ url: URL) -> Bool { inProject(url) || readRoots.contains { Self.contains($0, url) } }

    static func contains(_ folder: URL, _ url: URL) -> Bool {
        let f = folder.standardizedFileURL.resolvingSymlinksInPath().path
        let r = f.hasSuffix("/") ? f : f + "/"
        return url.path == f || url.path.hasPrefix(r)
    }

    /// The path as shown: relative to the project when inside it.
    public func display(_ url: URL) -> String {
        inProject(url) ? String(url.path.dropFirst(root.path.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/")) : url.path
    }

    func markRead(_ url: URL) { lock.withLock { stamps[url.path] = Self.modified(url) } }

    /// Edits need a fresh read: the file was read in this chat and hasn't changed on disk since.
    func checkFresh(_ url: URL) throws {
        let stamp = lock.withLock { stamps[url.path] }
        guard let stamp else { throw ToolError("read \(display(url)) first (read_file), then edit it") }
        if let now = Self.modified(url), now > stamp {
            throw ToolError("\(display(url)) changed on disk since it was read: read it again before editing")
        }
    }

    static func modified(_ url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }
}

/// The id of the tool call a task runs, for calls run at the same time (see `ToolContext.callID`).
public enum ToolCallScope {
    @TaskLocal public static var id: String?
}

/// The user's answer to an `exit_plan` plan.
public enum PlanAnswer: Equatable, Sendable {
    /// Go ahead, in this mode (`.acceptEdits` or `.ask`).
    case approve(PermissionMode)
    /// Not yet: keep planning, with the user's note (if any).
    case keepPlanning(String?)
}

public protocol AgentTool: Sendable {
    var name: String { get }
    var description: String { get }
    var parameters: JSONValue { get }
    var effect: ToolEffect { get }
    func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview
    func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput
}

extension AgentTool {
    public var spec: ToolSpec { ToolSpec(name: name, description: description, parameters: parameters) }
}

func schema(_ props: [String: (String, String)], required: [String], extra: [String: JSONValue] = [:]) -> JSONValue {
    var p: [String: JSONValue] = [:]
    for (k, (type, desc)) in props { p[k] = .object(["type": .string(type), "description": .string(desc)]) }
    for (k, v) in extra { p[k] = v }
    return .object(["type": .string("object"), "properties": .object(p), "required": .array(required.map(JSONValue.string))])
}

/// Long output for the model: the head and the tail, with how much was left out.
func clip(_ s: String, max: Int = 30_000) -> String {
    guard s.count > max else { return s }
    let half = max / 2
    return String(s.prefix(half)) + "\n… [\(s.count - max) characters left out] …\n" + String(s.suffix(half))
}

// MARK: the coding tools

public struct ReadFileTool: AgentTool {
    public init() {}
    public let name = "read_file"
    public let description = "Read a text file. Returns numbered lines (up to 2000; use offset and limit for more). Read a file before editing it."
    public var parameters: JSONValue {
        schema(["path": ("string", "File path, relative to the project or absolute"),
                "offset": ("integer", "First line to read (1-based)"), "limit": ("integer", "How many lines")],
               required: ["path"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = ctx.resolve(try args.required("path"))
        return ToolPreview(kind: .read, title: "Read \(ctx.display(url))", paths: [url])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let url = ctx.resolve(try args.required("path"))
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
            throw ToolError("\(ctx.display(url)) doesn't exist")
        }
        if isDir.boolValue {
            let names = (try? FileManager.default.contentsOfDirectory(atPath: url.path))?.sorted() ?? []
            return ToolOutput("\(ctx.display(url)) is a folder:\n" + names.prefix(500).joined(separator: "\n"))
        }
        let data = try Data(contentsOf: url)
        if data.count > 20_000_000 { throw ToolError("\(ctx.display(url)) is \(data.count / 1_000_000) MB: too big to read") }
        guard let text = String(data: data, encoding: .utf8) else { throw ToolError("\(ctx.display(url)) isn't a text file") }
        ctx.markRead(url)
        let lines = text.components(separatedBy: "\n")
        let start = max(1, args.int("offset") ?? 1)
        let count = min(max(1, args.int("limit") ?? 2000), 2000)
        guard start <= max(1, lines.count) else { return ToolOutput("(the file has \(lines.count) lines)") }
        let end = min(lines.count, start - 1 + count)
        var out = ""
        for i in (start - 1)..<end {
            var line = lines[i]
            if line.count > 2000 { line = String(line.prefix(2000)) + "…" }
            out += String(format: "%6d\t", i + 1) + line + "\n"
        }
        if end < lines.count { out += "… (\(lines.count - end) more lines; use offset \(end + 1))\n" }
        return ToolOutput(out.isEmpty ? "(empty file)" : out)
    }
}

public struct ListFilesTool: AgentTool {
    public init() {}
    public let name = "list_files"
    public let description = "List the project's files, optionally matching a glob such as \"**/*.swift\" or \"Sources/*\" (git-tracked and untracked files, without ignored ones)."
    public var parameters: JSONValue {
        schema(["pattern": ("string", "Glob to match, e.g. **/*.py"), "path": ("string", "Folder to list (default: the project)")],
               required: [])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = ctx.resolve(args.string("path") ?? ".")
        return ToolPreview(kind: .read, title: "List \(args.string("pattern") ?? ctx.display(url).ifEmpty("files"))", paths: [url])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let dir = ctx.resolve(args.string("path") ?? ".")
        var files = await Self.files(in: dir)
        if let pattern = args.string("pattern"), !pattern.isEmpty {
            files = files.filter { Glob.match(pattern, $0) }
        }
        files.sort()
        let shown = files.prefix(500).joined(separator: "\n")
        return ToolOutput(files.isEmpty ? "(no files match)" : shown + (files.count > 500 ? "\n… \(files.count - 500) more" : ""))
    }

    /// Paths relative to `dir`: git's view when it's a repository, else a walk skipping build and dependency folders.
    static func files(in dir: URL) async -> [String] {
        let git = await ShellRunner().run("git ls-files --cached --others --exclude-standard 2>/dev/null", cwd: dir, timeout: 20)
        if git.status == 0, !git.output.isEmpty {
            return git.output.split(separator: "\n").map(String.init)
        }
        let skip: Set<String> = [".git", ".build", "node_modules", "build", "DerivedData", ".venv", "__pycache__", ".next"]
        var out: [String] = []
        guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                                     options: [.skipsHiddenFiles]) else { return [] }
        for case let url as URL in e {
            if skip.contains(url.lastPathComponent) { e.skipDescendants(); continue }
            if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true { continue }
            out.append(String(url.standardizedFileURL.path.dropFirst(dir.standardizedFileURL.path.count + 1)))
            if out.count > 20_000 { break }
        }
        return out
    }
}

public struct SearchTool: AgentTool {
    public init() {}
    public let name = "search"
    public let description = "Search file contents with a regular expression (ripgrep syntax). Returns matching lines as path:line:text."
    public var parameters: JSONValue {
        schema(["pattern": ("string", "Regular expression"), "path": ("string", "Folder or file (default: the project)"),
                "glob": ("string", "Only files matching this glob, e.g. *.swift"),
                "ignore_case": ("boolean", "Case-insensitive")], required: ["pattern"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = ctx.resolve(args.string("path") ?? ".")
        return ToolPreview(kind: .read, title: "Search \(try args.required("pattern"))", paths: [url])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let pattern = try args.required("pattern")
        let target = ctx.resolve(args.string("path") ?? ".")
        let rg = await ShellRunner().run("command -v rg", cwd: ctx.root, timeout: 5)
        if rg.status == 0 {
            var cmd = "rg --line-number --no-heading --color never --max-columns 300 -e \(shellQuote(pattern))"
            if args.bool("ignore_case") { cmd += " -i" }
            if let g = args.string("glob") { cmd += " -g \(shellQuote(g))" }
            cmd += " -- \(shellQuote(target.path))"
            let r = await ShellRunner().run(cmd, cwd: ctx.root, timeout: 60)
            if r.status > 1 { throw ToolError(r.output.trimmingCharacters(in: .whitespacesAndNewlines)) }
            let lines = r.output.split(separator: "\n").map { line -> String in
                line.hasPrefix(ctx.root.path + "/") ? String(line.dropFirst(ctx.root.path.count + 1)) : String(line)
            }
            return ToolOutput(lines.isEmpty ? "(no matches)" : lines.prefix(300).joined(separator: "\n")
                              + (lines.count > 300 ? "\n… \(lines.count - 300) more matches" : ""))
        }
        // No ripgrep: a plain scan.
        let re = try NSRegularExpression(pattern: pattern, options: args.bool("ignore_case") ? [.caseInsensitive] : [])
        var files = await ListFilesTool.files(in: target)
        if let g = args.string("glob") { files = files.filter { Glob.match(g, $0) } }
        var out: [String] = []
        for f in files {
            let url = target.appendingPathComponent(f)
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            for (i, line) in text.components(separatedBy: "\n").enumerated()
            where re.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil {
                out.append("\(ctx.display(url)):\(i + 1):\(line.prefix(300))")
                if out.count >= 300 { break }
            }
            if out.count >= 300 { break }
        }
        return ToolOutput(out.isEmpty ? "(no matches)" : out.joined(separator: "\n"))
    }
}

public struct EditFileTool: AgentTool {
    public init() {}
    public let name = "edit_file"
    public let description = "Replace exact text in a file. old_string must match the file exactly (whitespace included) and only once, unless replace_all is true. Read the file first."
    public var parameters: JSONValue {
        schema(["path": ("string", "File path"), "old_string": ("string", "The exact text to replace"),
                "new_string": ("string", "The text to put instead"),
                "replace_all": ("boolean", "Replace every occurrence")], required: ["path", "old_string", "new_string"])
    }
    public let effect = ToolEffect.edit

    func apply(_ args: ToolArgs, _ ctx: ToolContext) throws -> (URL, String, String) {
        let url = ctx.resolve(try args.required("path"))
        let old = try args.required("old_string"), new = try args.required("new_string")
        guard old != new else { throw ToolError("old_string and new_string are the same") }
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { throw ToolError("can't read \(ctx.display(url))") }
        let count = text.components(separatedBy: old).count - 1
        if old.isEmpty || count == 0 { throw ToolError("old_string isn't in \(ctx.display(url)) (it must match exactly)") }
        if count > 1, !args.bool("replace_all") {
            throw ToolError("old_string is in \(ctx.display(url)) \(count) times: add context to make it unique, or set replace_all")
        }
        let updated = args.bool("replace_all") ? text.replacingOccurrences(of: old, with: new)
            : text.replacingCharacters(in: text.range(of: old)!, with: new)
        return (url, text, updated)
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = ctx.resolve(try args.required("path"))
        var p = ToolPreview(kind: .diff, title: "Edit \(ctx.display(url))", paths: [url])
        if let r = try? apply(args, ctx) {
            p.detail = UnifiedDiff.text(r.1, r.2)
            (p.added, p.removed) = UnifiedDiff.counts(r.1, r.2)
        }
        return p
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let url = ctx.resolve(try args.required("path"))
        try ctx.checkFresh(url)
        let (_, before, after) = try apply(args, ctx)
        if let cp = ctx.checkpoints, let turn = ctx.turnID { try await cp.save(url, turn: turn) }
        try after.write(to: url, atomically: true, encoding: .utf8)
        ctx.markRead(url)
        let (a, r) = UnifiedDiff.counts(before, after)
        return ToolOutput("Edited \(ctx.display(url)) (+\(a) −\(r)).")
    }
}

public struct WriteFileTool: AgentTool {
    public init() {}
    public let name = "write_file"
    public let description = "Create a file, or replace a whole file with new content (read an existing file first). Prefer edit_file for changes to existing files."
    public var parameters: JSONValue {
        schema(["path": ("string", "File path"), "content": ("string", "The file's full content")], required: ["path", "content"])
    }
    public let effect = ToolEffect.edit

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = ctx.resolve(try args.required("path"))
        let new = try args.required("content")
        let old = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        let exists = FileManager.default.fileExists(atPath: url.path)
        var p = ToolPreview(kind: .diff, title: "\(exists ? "Write" : "Create") \(ctx.display(url))", paths: [url])
        p.detail = UnifiedDiff.text(old, new)
        (p.added, p.removed) = UnifiedDiff.counts(old, new)
        return p
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let url = ctx.resolve(try args.required("path"))
        let content = try args.required("content")
        let exists = FileManager.default.fileExists(atPath: url.path)
        if exists { try ctx.checkFresh(url) }
        if let cp = ctx.checkpoints, let turn = ctx.turnID { try await cp.save(url, turn: turn) }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try content.write(to: url, atomically: true, encoding: .utf8)
        ctx.markRead(url)
        return ToolOutput("\(exists ? "Wrote" : "Created") \(ctx.display(url)) (\(content.components(separatedBy: "\n").count) lines).")
    }
}

public struct RunCommandTool: AgentTool {
    public init() {}
    public let name = "run_command"
    public let description = "Run a shell command (zsh) in the project folder and return its output and exit status. No interactive programs. Default timeout 120 s, at most 600. With background true it keeps running (a dev server, a watcher) and returns at once with a job id."
    public var parameters: JSONValue {
        schema(["command": ("string", "The command"), "timeout": ("integer", "Seconds before it is stopped"),
                "background": ("boolean", "Leave it running (a dev server, a watcher) and return at once with a job id. Read its output with command_output, end it with stop_command.")],
               required: ["command"])
    }
    public let effect = ToolEffect.execute

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let cmd = try args.required("command")
        let first = cmd.split(separator: "\n").first.map(String.init) ?? cmd
        return ToolPreview(kind: .command, title: (args.bool("background") ? "Run in background: " : "Run ") + first,
                           detail: cmd, paths: [ctx.root], command: cmd)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let cmd = try args.required("command")
        if args.bool("background") { return try await runInBackground(cmd, ctx) }
        let timeout = TimeInterval(min(600, max(1, args.int("timeout") ?? 120)))
        let id = ctx.callID
        let sink = ctx.onOutput
        let r = await ShellRunner().run(cmd, cwd: ctx.root, timeout: timeout) { chunk in sink?(id, chunk) }
        var head = "exit status \(r.status)"
        if r.timedOut { head = "stopped after \(Int(timeout)) s (timeout)" }
        if r.cancelled { head = "stopped (cancelled)" }
        let body = r.output.isEmpty ? "(no output)" : clip(r.output)
        return ToolOutput("\(head)\n\(body)", isError: r.status != 0 || r.timedOut)
    }

    private func runInBackground(_ cmd: String, _ ctx: ToolContext) async throws -> ToolOutput {
        guard let jobs = ctx.jobs else {
            return ToolOutput("Background commands aren't available here: run it without background.", isError: true)
        }
        let job = try jobs.start(cmd, cwd: ctx.root)
        for _ in 0..<30 where job.isRunning {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let output = job.readNew()
        let status = job.status
        var text = "Started \(job.id) in the background."
        if let status { text += " It already ended with exit status \(status)." }
        text += "\n" + (output.isEmpty ? "(no output yet)" : output)
        text += "\nRead more with command_output (id \"\(job.id)\"); end it with stop_command."
        return ToolOutput(text, isError: (status ?? 0) != 0)
    }
}

public struct TodoWriteTool: AgentTool {
    public init() {}
    public let name = "todo_write"
    public let description = "Keep a to-do list for multi-step work: send the whole list each time, with each item's status (pending, in_progress, completed)."
    public var parameters: JSONValue {
        .object(["type": .string("object"), "required": .array([.string("todos")]), "properties": .object([
            "todos": .object(["type": .string("array"), "items": .object([
                "type": .string("object"), "required": .array([.string("content"), .string("status")]),
                "properties": .object([
                    "content": .object(["type": .string("string")]),
                    "status": .object(["type": .string("string"), "enum": .array([.string("pending"), .string("in_progress"), .string("completed")])]),
                ]),
            ])]),
        ])])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        ToolPreview(kind: .todo, title: "Update the to-do list", detail: Self.list(args).map { "\($0.status == "completed" ? "☑" : $0.status == "in_progress" ? "▶" : "☐") \($0.content)" }.joined(separator: "\n"))
    }

    static func list(_ args: ToolArgs) -> [ToolContext.Todo] {
        (args.raw["todos"] as? [[String: Any]] ?? []).compactMap { d in
            guard let c = d["content"] as? String else { return nil }
            return ToolContext.Todo(content: c, status: d["status"] as? String ?? "pending")
        }
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let todos = Self.list(args)
        ctx.setTodos(todos)
        let done = todos.filter { $0.status == "completed" }.count
        return ToolOutput("To-do list updated: \(done) of \(todos.count) done.")
    }
}

/// The coding agent's tools.
public enum CodingTools {
    public static let all: [any AgentTool] = [ReadFileTool(), ListFilesTool(), SearchTool(), EditFileTool(),
                                              WriteFileTool(), RunCommandTool(), CommandOutputTool(), StopCommandTool(),
                                              TodoWriteTool()]
}

// MARK: helpers

func shellQuote(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

extension String {
    func ifEmpty(_ other: String) -> String { isEmpty ? other : self }
}

/// Globs for file lists: `*` within a folder, `**` across folders, `?`, `[…]`. A pattern without a slash matches
/// the file name anywhere.
enum Glob {
    static func match(_ pattern: String, _ path: String) -> Bool {
        if !pattern.contains("/") {
            return fnmatch(pattern, (path as NSString).lastPathComponent, 0) == 0
        }
        var p = pattern
        if p.hasPrefix("./") { p.removeFirst(2) }
        if fnmatch(p, path, FNM_PATHNAME) == 0 { return true }
        // `**/` matches zero or more folders.
        if p.contains("**") {
            let collapsed = p.replacingOccurrences(of: "**/", with: "")
            if fnmatch(collapsed, path, FNM_PATHNAME) == 0 { return true }
            return fnmatch(p.replacingOccurrences(of: "**", with: "*"), path, 0) == 0
        }
        return false
    }
}
