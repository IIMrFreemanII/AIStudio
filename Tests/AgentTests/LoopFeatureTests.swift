import Foundation
import Testing
@testable import Agent
@testable import LLMClient

private final class ModeLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _modes: [PermissionMode] = []
    var modes: [PermissionMode] { lock.withLock { _modes } }
    func add(_ m: PermissionMode) { lock.withLock { _modes.append(m) } }
}

private func planConfig() -> AgentConfig {
    var c = AgentConfig(model: "m", tools: CodingTools.all + [ExitPlanTool()])
    c.mode = .plan
    return c
}

struct PlanModeTests {
    @Test func approvedPlanSwitchesTheModeMidReply() async throws {
        let root = try tempProject()
        let model = ScriptedModel([
            [call("exit_plan", ["plan": "Do it"], id: "p1")],
            [call("read_file", ["path": "Sources/a.txt"], id: "r1"),
             .toolCall(index: 1, id: "e1", name: "edit_file",
                       arguments: "{\"path\":\"Sources/a.txt\",\"old_string\":\"line two\",\"new_string\":\"line 2\"}")],
            [.content("done")],
        ])
        let ctx = ToolContext(root: root)
        ctx.approvePlan = { _, _ in .approve(.acceptEdits) }
        let rec = Recorder()
        let log = ModeLog()
        _ = try await AgentLoop.run(client: model, messages: [.user("go")], config: planConfig(), context: ctx,
                                    ask: { r in rec.ask(r); return .allow }, modeChanged: { log.add($0) },
                                    events: { rec.add($0) })
        #expect(rec.asks.isEmpty)
        #expect(log.modes == [.acceptEdits])
        #expect(try String(contentsOf: root.appendingPathComponent("Sources/a.txt"), encoding: .utf8).contains("line 2"))
        #expect(model.requests[0].tools.contains { $0.name == "exit_plan" })
        #expect(!model.requests[1].tools.contains { $0.name == "exit_plan" })
    }

    @Test func keepPlanningStaysInPlanMode() async throws {
        let model = ScriptedModel([[call("exit_plan", ["plan": "Do it"])], [.content("ok")]])
        let ctx = ToolContext(root: try tempProject())
        ctx.approvePlan = { _, _ in .keepPlanning("shorter") }
        let log = ModeLog()
        _ = try await AgentLoop.run(client: model, messages: [.user("go")], config: planConfig(), context: ctx,
                                    ask: { _ in .allow }, modeChanged: { log.add($0) }, events: { _ in })
        #expect(log.modes.isEmpty && ctx.mode == .plan)
        #expect(model.requests[1].messages.last?.text.contains("shorter") == true)
        #expect(model.requests[1].tools.contains { $0.name == "exit_plan" })
    }

    @Test func noApprovalHandler() async throws {
        let model = ScriptedModel([[call("exit_plan", ["plan": "Do it"])], [.content("ok")]])
        let ctx = ToolContext(root: try tempProject())
        _ = try await AgentLoop.run(client: model, messages: [.user("go")], config: planConfig(), context: ctx,
                                    ask: { _ in .allow }, events: { _ in })
        #expect(model.requests[1].messages.last?.text.contains("can't be approved") == true)
        #expect(ctx.mode == .plan)
    }
}

struct CompactionTests {
    private func run(limit: Int?, usage: Int) async throws -> (ScriptedModel, Recorder) {
        let model = ScriptedModel([
            [.usage(promptTokens: usage, completionTokens: 10), call("read_file", ["path": "Sources/a.txt"])],
            [.content("SUMMARY")],
            [.content("done")],
        ])
        var config = AgentConfig(model: "m")
        config.contextLimit = limit
        let rec = Recorder()
        _ = try await AgentLoop.run(client: model, messages: [.system("sys"), .user("go")], config: config,
                                    context: ToolContext(root: try tempProject()), ask: { _ in .allow },
                                    events: { rec.add($0) })
        return (model, rec)
    }

    @Test func compactsNearTheLimit() async throws {
        let (model, rec) = try await run(limit: 1000, usage: 900)
        #expect(model.requests.count == 3)
        #expect(model.requests[1].tools.isEmpty)
        let m = model.requests[2].messages
        #expect(m.count == 2 && m[0].role == .system && m[1].role == .user)
        #expect(m[1].text.hasPrefix(ChatTurn.summaryPrefix) && m[1].text.contains("SUMMARY"))
        #expect(rec.events.contains("compacted"))
    }

    @Test func noCompactionBelowTheThresholdOrWithoutALimit() async throws {
        for (limit, usage) in [(1000 as Int?, 100), (nil, 900)] {
            let (model, rec) = try await run(limit: limit, usage: usage)
            #expect(model.requests.count == 2 && !rec.events.contains("compacted"))
        }
    }

    @Test func trimKeepsTheLastSix() {
        var msgs: [ChatMessage] = [.user("x")]
        for i in 0..<8 { msgs.append(ChatMessage(role: .tool, parts: [.text(String(repeating: "a", count: 300))], toolCallID: "\(i)")) }
        msgs.append(ChatMessage(role: .tool, parts: [.text("short")], toolCallID: "s"))
        let out = Compactor.trim(msgs)
        let tools = out.filter { $0.role == .tool }
        #expect(tools.prefix(3).allSatisfy { $0.text == "[output removed to save context]" })
        #expect(tools.dropFirst(3).prefix(5).allSatisfy { $0.text.count == 300 })
        #expect(tools.last?.text == "short")
        #expect(Compactor.estimate([.user(String(repeating: "a", count: 400))]) == 100)
    }
}

/// Answers by request: a subagent's (system starts with "You are a research subagent") sleeps, then answers.
private final class TaskModel: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [ChatRequest] = []
    private var parentCalls = 0
    var requests: [ChatRequest] { lock.withLock { _requests } }

    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        lock.withLock { _requests.append(request) }
        let isSub = request.messages.first?.text.hasPrefix("You are a research subagent") == true
        let parentFirst = !isSub && lock.withLock { () -> Bool in parentCalls += 1; return parentCalls == 1 }
        return AsyncThrowingStream { c in
            Task {
                if isSub {
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    c.yield(.content("found " + (request.messages.last?.text ?? "")))
                } else if parentFirst {
                    c.yield(call("task", ["description": "first", "prompt": "A"], id: "t1"))
                    c.yield(.toolCall(index: 1, id: "t2", name: "task", arguments: "{\"description\":\"second\",\"prompt\":\"B\"}"))
                } else {
                    c.yield(.content("done"))
                }
                c.finish()
            }
        }
    }
}

struct TaskToolTests {
    @Test func parallelSubagents() async throws {
        let model = TaskModel()
        var config = AgentConfig(model: "m", tools: CodingTools.all + [TaskTool(client: model, model: "m")])
        config.mode = .ask
        let start = Date()
        _ = try await AgentLoop.run(client: model, messages: [.user("go")], config: config,
                                    context: ToolContext(root: try tempProject()), ask: { _ in .allow }, events: { _ in })
        #expect(Date().timeIntervalSince(start) < 0.6)
        let parentSecond = model.requests.last { $0.messages.first?.text.hasPrefix("You are a research subagent") == false }!
        let tools = parentSecond.messages.filter { $0.role == .tool }
        #expect(tools.map(\.toolCallID) == ["t1", "t2"])
        #expect(tools.map(\.text) == ["found A", "found B"])
        let subs = model.requests.filter { $0.messages.first?.text.hasPrefix("You are a research subagent") == true }
        #expect(subs.count == 2)
        #expect(subs.allSatisfy { Set($0.tools.map(\.name)) == ["read_file", "list_files", "search"] })
    }
}
