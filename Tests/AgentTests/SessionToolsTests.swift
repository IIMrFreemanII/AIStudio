import Foundation
import Testing
@testable import Agent
@testable import LLMClient

final class FakeSessionHost: SessionHost, @unchecked Sendable {
    private let lock = NSLock()
    let depth: Int
    private var _children: [SessionInfo]
    private var _started: [SessionSpec] = []
    private var _proposed: [SessionSpec] = []
    private var _handedOff: [SessionSpec] = []
    private var _sent: [(String, UUID)] = []
    var started: [SessionSpec] { lock.withLock { _started } }
    var proposed: [SessionSpec] { lock.withLock { _proposed } }
    var handedOff: [SessionSpec] { lock.withLock { _handedOff } }
    var sent: [(String, UUID)] { lock.withLock { _sent } }

    init(depth: Int = 0, children: [SessionInfo] = []) {
        self.depth = depth
        _children = children
    }

    func propose(_ spec: SessionSpec, callID: String) async -> String {
        lock.withLock { _proposed.append(spec) }
        return "Proposed “\(spec.title)”."
    }
    func start(_ spec: SessionSpec, callID: String) async throws -> SessionInfo {
        lock.withLock { _started.append(spec) }
        return SessionInfo(id: FakeSessionHost.fixedID, title: spec.title, status: "working", branch: spec.worktree ? "agent/x" : nil)
    }
    func children() async -> [SessionInfo] { lock.withLock { _children } }
    func send(_ text: String, to id: UUID) async throws { lock.withLock { _sent.append((text, id)) } }
    func handOff(_ spec: SessionSpec, callID: String) async throws -> SessionInfo {
        lock.withLock { _handedOff.append(spec) }
        return SessionInfo(id: FakeSessionHost.fixedID, title: spec.title, status: "working")
    }

    static let fixedID = UUID(uuidString: "11111111-2222-3333-4444-555555555555")!
}

private func ctx(_ host: (any SessionHost)?) throws -> ToolContext {
    let c = ToolContext(root: try tempProject())
    c.sessions = host
    return c
}

private func child(_ status: String, _ n: Int = 0) -> SessionInfo {
    SessionInfo(id: UUID(), title: "c\(n)", status: status)
}

struct SessionToolsTests {
    let brief: [String: Any] = ["title": "Fix login", "prompt": "Fix the login bug"]

    @Test func proposeAsksTheHost() async throws {
        let host = FakeSessionHost()
        let out = try await ProposeSessionTool().run(ToolArgs(brief.merging(["context": "fork", "worktree": false]) { $1 }), try ctx(host))
        #expect(out.text == "Proposed “Fix login”." && !out.isError)
        #expect(host.proposed == [SessionSpec(title: "Fix login", prompt: "Fix the login bug", context: .fork, worktree: false)])
        let p = try ProposeSessionTool().preview(ToolArgs(brief), try ctx(host))
        #expect(p.kind == .read && p.title == "Propose session “Fix login”" && p.detail == "Fix the login bug")
    }

    @Test func defaultsAndMissingArguments() async throws {
        let host = FakeSessionHost()
        _ = try await StartSessionTool().run(ToolArgs(brief), try ctx(host))
        #expect(host.started == [SessionSpec(title: "Fix login", prompt: "Fix the login bug", context: .fresh, worktree: true)])
        await #expect(throws: ToolError.self) { try await StartSessionTool().run(ToolArgs(["title": "x"]), try ctx(host)) }
        await #expect(throws: ToolError.self) { try await StartSessionTool().run(ToolArgs(["title": " ", "prompt": "p"]), try ctx(host)) }
        await #expect(throws: ToolError.self) { try await ProposeSessionTool().run(ToolArgs(["prompt": "p"]), try ctx(host)) }
    }

    @Test func startReportsTheSession() async throws {
        let host = FakeSessionHost()
        let out = try await StartSessionTool().run(ToolArgs(brief), try ctx(host))
        #expect(!out.isError)
        #expect(out.text.hasPrefix("Started session “Fix login” (11111111-2222-3333-4444-555555555555) on branch agent/x. "))
        let p = try StartSessionTool().preview(ToolArgs(brief), try ctx(host))
        #expect(p.kind == .run && p.title == "Start session “Fix login”")
    }

    @Test func noHostMeansNoSessions() async throws {
        let c = try ctx(nil)
        let tools: [(any AgentTool, [String: Any])] = [
            (ProposeSessionTool(), brief), (StartSessionTool(), brief), (ListSessionsTool(), [:]),
            (SendToSessionTool(), ["id": UUID().uuidString, "message": "hi"]),
            (HandOffTool(), ["title": "t", "remaining_plan": "p"]),
        ]
        for (tool, args) in tools {
            let out = try await tool.run(ToolArgs(args), c)
            #expect(out == ToolOutput("Sessions aren't available here.", isError: true), "\(tool.name)")
        }
    }

    @Test func depthLimit() async throws {
        let host = FakeSessionHost(depth: SessionLimits.maxDepth)
        let out = try await StartSessionTool().run(ToolArgs(brief), try ctx(host))
        #expect(out.isError && out.text == "Sessions this deep can't start more sessions: do the work here.")
        #expect(host.started.isEmpty)
        let ok = try await StartSessionTool().run(ToolArgs(brief), try ctx(FakeSessionHost(depth: SessionLimits.maxDepth - 1)))
        #expect(!ok.isError)
    }

    @Test func workingLimitCountsWaitingToo() async throws {
        let full = FakeSessionHost(children: [child("working", 1), child("waiting", 2), child("working", 3), child("waiting", 4), child("done", 5)])
        let out = try await StartSessionTool().run(ToolArgs(brief), try ctx(full))
        #expect(out.isError && out.text == "4 sessions are already running: wait for one to finish (list_sessions).")
        #expect(full.started.isEmpty)
        let room = FakeSessionHost(children: [child("working", 1), child("waiting", 2), child("working", 3), child("done", 4), child("failed", 5)])
        #expect(try await !StartSessionTool().run(ToolArgs(brief), try ctx(room)).isError)
    }

    @Test func listSessions() async throws {
        let id = UUID()
        let long = String(repeating: "x", count: 500)
        let host = FakeSessionHost(children: [
            SessionInfo(id: id, title: "A", status: "done", branch: "agent/a", lastReport: "All fixed."),
            SessionInfo(id: FakeSessionHost.fixedID, title: "B", status: "working"),
            SessionInfo(id: UUID(), title: "C", status: "idle", lastReport: long),
        ])
        let out = try await ListSessionsTool().run(ToolArgs([:]), try ctx(host))
        let lines = out.text.components(separatedBy: "\n")
        #expect(lines[0] == "- A [done] id \(id.uuidString) branch agent/a" && lines[1] == "  All fixed.")
        #expect(lines[2] == "- B [working] id \(FakeSessionHost.fixedID.uuidString)" && lines[3] == "  (no report yet)")
        #expect(!out.text.contains(long) && out.text.contains("characters left out"))
        let none = try await ListSessionsTool().run(ToolArgs([:]), try ctx(FakeSessionHost()))
        #expect(none.text == "No sessions started from this chat.")
    }

    @Test func sendToSession() async throws {
        let kid = child("idle")
        let host = FakeSessionHost(children: [kid])
        let c = try ctx(host)
        let ok = try await SendToSessionTool().run(ToolArgs(["id": kid.id.uuidString, "message": "go on"]), c)
        #expect(ok.text == "Sent. Its reply will arrive here as a message." && !ok.isError)
        #expect(host.sent.count == 1 && host.sent[0].0 == "go on" && host.sent[0].1 == kid.id)
        let unknown = UUID().uuidString
        let a = try await SendToSessionTool().run(ToolArgs(["id": unknown, "message": "x"]), c)
        #expect(a.isError && a.text == "There is no session \(unknown) started from this chat.")
        let b = try await SendToSessionTool().run(ToolArgs(["id": "nope", "message": "x"]), c)
        #expect(b.isError && b.text == "There is no session nope started from this chat.")
        #expect(host.sent.count == 1)
    }

    @Test func handOffForksIntoAWorktree() async throws {
        let host = FakeSessionHost()
        let args = ToolArgs(["title": "Finish refactor", "remaining_plan": "1. do X\n2. test"])
        let out = try await HandOffTool().run(args, try ctx(host))
        #expect(host.handedOff == [SessionSpec(title: "Finish refactor", prompt: "1. do X\n2. test", context: .fork, worktree: true)])
        #expect(out.text == "Handed off to session “Finish refactor” (11111111-2222-3333-4444-555555555555). Stop working on it here and end your reply in one line.")
        let p = try HandOffTool().preview(args, try ctx(host))
        #expect(p.kind == .run && p.title == "Hand off to “Finish refactor”")
        await #expect(throws: ToolError.self) { try await HandOffTool().run(ToolArgs(["title": "t"]), try ctx(host)) }
    }

    @Test func allToolsAreRegistered() {
        #expect(SessionTools.all.map(\.name) == ["propose_session", "start_session", "list_sessions", "send_to_session", "hand_off"])
    }
}

struct SessionPermissionTests {
    private func run(_ tool: String, mode: PermissionMode, rules: AllowRules = AllowRules()) async throws -> (Recorder, FakeSessionHost, ScriptedModel) {
        let host = FakeSessionHost()
        let context = ToolContext(root: try tempProject())
        context.sessions = host
        let model = ScriptedModel([[call(tool, ["title": "T", "prompt": "P"])], [.content("ok")]])
        var config = AgentConfig(model: "m", tools: SessionTools.all, mode: mode, rules: rules)
        config.maxSteps = 5
        let rec = Recorder()
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: config, context: context,
                                    ask: { r in rec.ask(r); return .allow }, events: { rec.add($0) })
        return (rec, host, model)
    }

    @Test func startAsksInAskMode() async throws {
        let (rec, host, _) = try await run("start_session", mode: .ask)
        #expect(rec.asks.count == 1 && rec.asks[0].preview.kind == .run && rec.asks[0].effect == .run)
        #expect(host.started.count == 1)
    }

    @Test func startIsRefusedInPlanMode() async throws {
        let (rec, host, _) = try await run("start_session", mode: .plan)
        #expect(rec.asks.isEmpty && rec.events.contains("end denied"))
        #expect(host.started.isEmpty)
    }

    @Test func startRunsInBypassAndWithRunRule() async throws {
        let (rec, host, _) = try await run("start_session", mode: .bypass)
        #expect(rec.asks.isEmpty && host.started.count == 1)
        let (rec2, host2, _) = try await run("start_session", mode: .ask, rules: AllowRules(runs: true))
        #expect(rec2.asks.isEmpty && host2.started.count == 1)
    }

    @Test func proposeNeverAsks() async throws {
        for mode in [PermissionMode.ask, .plan, .acceptEdits, .bypass] {
            let (rec, host, _) = try await run("propose_session", mode: mode)
            #expect(rec.asks.isEmpty && host.proposed.count == 1, "\(mode)")
        }
    }
}
