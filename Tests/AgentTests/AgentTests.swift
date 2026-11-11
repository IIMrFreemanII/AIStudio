import Foundation
import Testing
@testable import Agent
@testable import LLMClient

/// A model that answers from a script: one list of events per request. Records the requests.
final class ScriptedModel: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var replies: [[ChatEvent]]
    private(set) var requests: [ChatRequest] = []
    var repeatLast = false

    init(_ replies: [[ChatEvent]]) { self.replies = replies }

    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        let events: [ChatEvent] = lock.withLock {
            requests.append(request)
            if replies.count == 1, repeatLast { return replies[0] }
            return replies.isEmpty ? [.content("(no more script)")] : replies.removeFirst()
        }
        return AsyncThrowingStream { c in
            for e in events { c.yield(e) }
            c.finish()
        }
    }
}

final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [String] = []
    private var _asks: [PermissionRequest] = []
    var events: [String] { lock.withLock { _events } }
    var asks: [PermissionRequest] { lock.withLock { _asks } }
    func add(_ e: AgentEvent) {
        let s: String
        switch e {
        case .assistantStarted: s = "assistant"
        case .delta: return
        case .assistantFinished(_, _, let calls): s = "finished(\(calls.map(\.name).joined(separator: ",")))"
        case .toolStarted(let call, _): s = "start \(call.name)"
        case .toolOutput: return
        case .toolFinished(_, _, let status): s = "end \(status.rawValue)"
        }
        lock.withLock { _events.append(s) }
    }
    func ask(_ r: PermissionRequest) { lock.withLock { _asks.append(r) } }
}

func tempProject() throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("agent-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir.appendingPathComponent("Sources"), withIntermediateDirectories: true)
    try "line one\nline two\nline three\n".write(to: dir.appendingPathComponent("Sources/a.txt"), atomically: true, encoding: .utf8)
    try "# Readme\n".write(to: dir.appendingPathComponent("README.md"), atomically: true, encoding: .utf8)
    return dir
}

func call(_ name: String, _ args: [String: Any], id: String = "c1") -> ChatEvent {
    let json = String(decoding: try! JSONSerialization.data(withJSONObject: args), as: UTF8.self)
    return .toolCall(index: 0, id: id, name: name, arguments: json)
}

struct PermissionTests {
    @Test func theMatrix() {
        func d(_ e: ToolEffect, _ m: PermissionMode, outside: Bool = false, rules: AllowRules = AllowRules(),
               cmd: String? = nil) -> PermissionDecision {
            PermissionEngine.decide(effect: e, inProject: !outside, command: cmd, mode: m, rules: rules)
        }
        #expect(d(.read, .ask) == .allow && d(.read, .ask, outside: true) == .ask && d(.read, .bypass, outside: true) == .allow)
        #expect(d(.edit, .ask) == .ask && d(.edit, .acceptEdits) == .allow && d(.edit, .acceptEdits, outside: true) == .ask)
        #expect(d(.edit, .ask, rules: AllowRules(edits: true)) == .allow)
        #expect(d(.execute, .acceptEdits, cmd: "ls") == .ask && d(.execute, .bypass, cmd: "rm -rf x") == .allow)
        if case .deny = d(.edit, .plan) {} else { Issue.record("plan mode must refuse edits") }
        if case .deny = d(.execute, .plan, cmd: "ls") {} else { Issue.record("plan mode must refuse commands") }
        let git = AllowRules(commandPrefixes: ["git status", "ls"])
        #expect(d(.execute, .ask, rules: git, cmd: "git status --short") == .allow)
        #expect(d(.execute, .ask, rules: git, cmd: "git status && ls -la") == .allow)
        #expect(d(.execute, .ask, rules: git, cmd: "git status; rm -rf /") == .ask)
        #expect(d(.execute, .ask, rules: git, cmd: "ls > out.txt") == .ask)
        #expect(d(.execute, .ask, rules: git, cmd: "ls $(rm x)") == .ask)
    }

    @Test func prefixes() {
        #expect(AllowRules.prefix(of: "git commit -m x") == "git commit")
        #expect(AllowRules.prefix(of: "swift build -c release") == "swift build")
        #expect(AllowRules.prefix(of: "ls -la") == "ls")
        #expect(AllowRules.prefix(of: "npm --version") == "npm")
    }
}

struct ToolTests {
    @Test func readNumbersLinesAndPages() async throws {
        let ctx = ToolContext(root: try tempProject())
        let all = try await ReadFileTool().run(ToolArgs(["path": "Sources/a.txt"]), ctx)
        #expect(all.text.contains("     1\tline one") && all.text.contains("     3\tline three"))
        let page = try await ReadFileTool().run(ToolArgs(["path": "Sources/a.txt", "offset": 2, "limit": 1]), ctx)
        #expect(page.text.hasPrefix("     2\tline two") && page.text.contains("more lines; use offset 3"))
        await #expect(throws: ToolError.self) { _ = try await ReadFileTool().run(ToolArgs(["path": "nope.txt"]), ctx) }
    }

    @Test func editNeedsAFreshReadAndAUniqueMatch() async throws {
        let root = try tempProject()
        let ctx = ToolContext(root: root)
        let edit = EditFileTool()
        let args = ToolArgs(["path": "Sources/a.txt", "old_string": "line two", "new_string": "line 2"])
        await #expect(throws: ToolError.self) { _ = try await edit.run(args, ctx) }  // not read yet
        _ = try await ReadFileTool().run(ToolArgs(["path": "Sources/a.txt"]), ctx)
        let p = try edit.preview(args, ctx)
        #expect(p.kind == .diff && p.added == 1 && p.removed == 1 && p.detail?.contains("-line two") == true)
        _ = try await edit.run(args, ctx)
        let text = try String(contentsOf: root.appendingPathComponent("Sources/a.txt"), encoding: .utf8)
        #expect(text == "line one\nline 2\nline three\n")
        // "line" is there three times.
        await #expect(throws: ToolError.self) {
            _ = try await edit.run(ToolArgs(["path": "Sources/a.txt", "old_string": "line", "new_string": "row"]), ctx)
        }
        _ = try await edit.run(ToolArgs(["path": "Sources/a.txt", "old_string": "line", "new_string": "row", "replace_all": true]), ctx)
        // Changed on disk after the last read: refused.
        try await Task.sleep(nanoseconds: 1_100_000_000)
        try "someone else\n".write(to: root.appendingPathComponent("Sources/a.txt"), atomically: true, encoding: .utf8)
        await #expect(throws: ToolError.self) {
            _ = try await edit.run(ToolArgs(["path": "Sources/a.txt", "old_string": "someone", "new_string": "x"]), ctx)
        }
    }

    @Test func writeCreatesFoldersAndNeedsAReadToReplace() async throws {
        let root = try tempProject()
        let ctx = ToolContext(root: root)
        let out = try await WriteFileTool().run(ToolArgs(["path": "new/dir/b.swift", "content": "let b = 1\n"]), ctx)
        #expect(out.text.hasPrefix("Created new/dir/b.swift"))
        await #expect(throws: ToolError.self) {
            _ = try await WriteFileTool().run(ToolArgs(["path": "README.md", "content": "x"]), ctx)
        }
        #expect(ctx.display(ctx.resolve("../outside.txt")).hasPrefix("/") && !ctx.inProject(ctx.resolve("../outside.txt")))
    }

    @Test func listAndSearch() async throws {
        let ctx = ToolContext(root: try tempProject())
        let swift = try await ListFilesTool().run(ToolArgs(["pattern": "**/*.txt"]), ctx)
        #expect(swift.text == "Sources/a.txt")
        let all = try await ListFilesTool().run(ToolArgs([:]), ctx)
        #expect(all.text.contains("README.md") && all.text.contains("Sources/a.txt"))
        let found = try await SearchTool().run(ToolArgs(["pattern": "line t[a-z]+"]), ctx)
        #expect(found.text.contains("Sources/a.txt:2:line two") && found.text.contains("Sources/a.txt:3:line three"))
    }

    @Test func todos() async throws {
        let ctx = ToolContext(root: try tempProject())
        let out = try await TodoWriteTool().run(ToolArgs(["todos": [["content": "a", "status": "completed"],
                                                                      ["content": "b", "status": "pending"]]]), ctx)
        #expect(out.text == "To-do list updated: 1 of 2 done." && ctx.todos.count == 2)
    }

    @Test func diffText() {
        let t = UnifiedDiff.text("a\nb\nc\nd\ne\nf\ng\nh\n", "a\nb\nc\nD\ne\nf\ng\nh\n", context: 1)
        #expect(t == "@@\n c\n-d\n+D\n e")
        #expect(UnifiedDiff.counts("", "x\ny") == (2, 1))
    }
}

struct ShellTests {
    @Test func outputAndStatus() async throws {
        let dir = try tempProject()
        let r = await ShellRunner().run("echo hi; ls Sources; echo err >&2; exit 3", cwd: dir)
        #expect(r.status == 3 && r.output.contains("hi\na.txt\nerr") && !r.timedOut)
    }

    @Test func aTimeoutEndsTheWholeGroup() async throws {
        let start = Date()
        let r = await ShellRunner().run("sleep 30 & sleep 30; echo never", cwd: FileManager.default.temporaryDirectory, timeout: 1)
        #expect(r.timedOut && !r.output.contains("never"))
        #expect(Date().timeIntervalSince(start) < 8)
    }

    @Test func cancelStopsIt() async throws {
        let task = Task { await ShellRunner().run("sleep 30", cwd: FileManager.default.temporaryDirectory) }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()
        let r = await task.value
        #expect(r.cancelled)
    }
}

struct LoopTests {
    @Test func toolCallAskedAllowedRunAndAnswered() async throws {
        let root = try tempProject()
        let model = ScriptedModel([
            [.reasoning("I'll create it."), call("write_file", ["path": "hello.txt", "content": "hi\n"]), .finish(reason: "tool_calls")],
            [.content("Created hello.txt."), .finish(reason: "stop")],
        ])
        let rec = Recorder()
        let outcome = try await AgentLoop.run(client: model, messages: [.user("make hello.txt")],
                                              config: AgentConfig(model: "m"), context: ToolContext(root: root),
                                              ask: { r in rec.ask(r); return .allow }, events: { rec.add($0) })
        #expect(outcome == .finished)
        #expect(rec.events == ["assistant", "finished(write_file)", "start write_file", "end done", "assistant", "finished()"])
        #expect(rec.asks.count == 1 && rec.asks[0].preview.title == "Create hello.txt" && rec.asks[0].effect == .edit)
        #expect(try String(contentsOf: root.appendingPathComponent("hello.txt"), encoding: .utf8) == "hi\n")
        // The second request carries the call and its result; tools went with both.
        let second = model.requests[1].messages
        #expect(second.map(\.role) == [.user, .assistant, .tool] && second[1].toolCalls.first?.name == "write_file")
        #expect(second[2].toolCallID == "c1" && second[2].text.hasPrefix("Created hello.txt"))
        #expect(model.requests.allSatisfy { $0.tools.count == CodingTools.all.count })
    }

    @Test func declinedAndPlanModeCallsDontRun() async throws {
        let root = try tempProject()
        let script: [[ChatEvent]] = [[call("run_command", ["command": "touch made.txt"])], [.content("OK")]]
        let model = ScriptedModel(script)
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: AgentConfig(model: "m"),
                                    context: ToolContext(root: root), ask: { _ in .deny }, events: { _ in })
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("made.txt").path))
        #expect(model.requests[1].messages.last?.text.contains("declined") == true)

        let plan = ScriptedModel(script)
        let rec = Recorder()
        var config = AgentConfig(model: "m")
        config.mode = .plan
        _ = try await AgentLoop.run(client: plan, messages: [.user("x")], config: config, context: ToolContext(root: root),
                                    ask: { r in rec.ask(r); return .allow }, events: { rec.add($0) })
        #expect(rec.asks.isEmpty && rec.events.contains("end denied"))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("made.txt").path))
    }

    @Test func alwaysAllowRemembersTheCommandPrefix() async throws {
        let root = try tempProject()
        let model = ScriptedModel([[call("run_command", ["command": "ls Sources"])],
                                   [call("run_command", ["command": "ls -la"], id: "c2")], [.content("done")]])
        let rec = Recorder()
        let saved = Recorder()
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: AgentConfig(model: "m"),
                                    context: ToolContext(root: root), ask: { r in rec.ask(r); return .allowAlways },
                                    rulesChanged: { rules in saved.ask(PermissionRequest(
                                        call: ToolCall(id: "", name: rules.commandPrefixes.joined(separator: ","), arguments: ""),
                                        preview: ToolPreview(kind: .command, title: ""), effect: .execute, inProject: true)) },
                                    events: { _ in })
        #expect(rec.asks.count == 1)  // the second ls didn't ask
        #expect(saved.asks.first?.call.name == "ls")
        #expect(model.requests[1].messages.last?.text.contains("a.txt") == true)
    }

    @Test func stepLimitAndUnknownTools() async throws {
        let model = ScriptedModel([[call("nope", [:])]])
        model.repeatLast = true
        var config = AgentConfig(model: "m")
        config.maxSteps = 3
        let rec = Recorder()
        let outcome = try await AgentLoop.run(client: model, messages: [.user("x")], config: config,
                                              context: ToolContext(root: try tempProject()), ask: { _ in .allow },
                                              events: { rec.add($0) })
        #expect(outcome == .stepLimit && model.requests.count == 3 && rec.events.filter { $0 == "end failed" }.count == 3)
    }

    @Test func systemPromptHasTheProjectInstructions() async throws {
        let root = try tempProject()
        try "Use tabs.".write(to: root.appendingPathComponent("AGENTS.md"), atomically: true, encoding: .utf8)
        let p = await ProjectContext.systemPrompt(root: root, mode: .plan)
        #expect(p.contains(root.path) && p.contains("Use tabs.") && p.contains("plan mode"))
    }
}
