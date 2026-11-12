import Foundation
import LLMClient
import Testing
@testable import Agent

private func sh(_ cmd: String, _ cwd: URL) async -> String {
    await ShellRunner().run(cmd, cwd: cwd).output
}

private func makeRepo() async throws -> URL {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent("flow-" + UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    _ = await sh("git init -q -b main && git config user.email t@t && git config user.name t && echo base > a.txt && git add -A && git commit -q -m init", dir)
    return dir.standardizedFileURL.resolvingSymlinksInPath()
}

private func json(_ s: String) -> JSONValue { try! JSONDecoder().decode(JSONValue.self, from: Data(s.utf8)) }

private final class FakeHost: FlowHost, @unchecked Sendable {
    private let lock = NSLock()
    private var _calls: [(script: String, meta: FlowMeta, args: JSONValue?, callID: String)] = []
    var calls: [(script: String, meta: FlowMeta, args: JSONValue?, callID: String)] { lock.withLock { _calls } }
    func start(script: String, meta: FlowMeta, args: JSONValue?, callID: String) async throws -> String {
        lock.withLock { _calls.append((script, meta, args, callID)) }
        return "r1"
    }
    func limits() async -> (concurrency: Int, maxAgents: Int) { (4, 30) }
}

private final class Usages: @unchecked Sendable {
    private let lock = NSLock()
    private var _v: [FlowUsage] = []
    private var _steps: [String] = []
    func add(_ e: FlowAgentEvent) {
        lock.withLock {
            switch e {
            case .usage(let u): _v.append(u)
            case .step(let s): _steps.append(s)
            }
        }
    }
    var usages: [FlowUsage] { lock.withLock { _v } }
    var steps: [String] { lock.withLock { _steps } }
}

private let goodScript = """
export const meta = { name: 'x', description: 'Does x', phases: [{title: 'A'}, {title: 'B'}] }
const r = await agent('hi', {mode: 'edit'})
return r
"""

struct FlowAgentsTests {
    @Test func extractJSON() {
        #expect(FlowJSON.extract("Here:\n```json\n{\"a\": 1}\n```\nDone") == json("{\"a\":1}"))
        #expect(FlowJSON.extract("Sure. {\"a\": [1, 2]} thanks") == json("{\"a\":[1,2]}"))
        #expect(FlowJSON.extract("{\"a\":1} then {\"b\":\"}\"}") == json("{\"b\":\"}\"}"))
        #expect(FlowJSON.extract("```\n[1,2]\n```") == json("[1,2]"))
        #expect(FlowJSON.extract("no json here") == nil)
    }

    @Test func validateJSON() {
        let schema = json("""
        {"type":"object","required":["findings"],"properties":{"findings":{"type":"array","items":{"type":"object",
        "required":["title"],"properties":{"title":{"type":"string"},"sev":{"enum":["low","high"]},"n":{"type":"integer"}}}}}}
        """)
        #expect(FlowJSON.validate(json("{\"findings\":[{\"title\":\"a\",\"sev\":\"low\",\"extra\":1}]}"), schema: schema) == nil)
        #expect(FlowJSON.validate(json("{}"), schema: schema)?.contains("findings") == true)
        let deep = FlowJSON.validate(json("{\"findings\":[{\"title\":\"a\"},{\"title\":3}]}"), schema: schema)
        #expect(deep == "findings[1].title: expected string")
        #expect(FlowJSON.validate(json("{\"findings\":[{\"title\":\"a\",\"sev\":\"mid\"}]}"), schema: schema)?.contains("findings[0].sev") == true)
        #expect(FlowJSON.validate(json("{\"findings\":[{\"title\":\"a\",\"n\":1.5}]}"), schema: schema)?.contains("findings[0].n") == true)
        #expect(FlowJSON.validate(.number(1.5), schema: json("{\"type\":\"number\"}")) == nil)
        #expect(FlowJSON.validate(.number(2), schema: json("{\"type\":\"number\"}")) == nil)
        #expect(FlowJSON.validate(.null, schema: json("{\"type\":[\"string\",\"null\"]}")) == nil)
        #expect(FlowJSON.validate(.array([]), schema: json("{\"type\":\"object\"}")) != nil)
    }

    @Test func policy() {
        func req(_ e: ToolEffect, inProject: Bool = true, command: String? = nil) -> PermissionRequest {
            var p = ToolPreview(kind: .command, title: "t")
            p.command = command
            return PermissionRequest(call: ToolCall(id: "1", name: "x", arguments: ""), preview: p, effect: e, inProject: inProject)
        }
        #expect(FlowPolicy.answer(req(.execute, command: "swift build")) == .allow)
        #expect(FlowPolicy.answer(req(.execute, command: "git push origin main")) == .deny)
        #expect(FlowPolicy.answer(req(.execute, inProject: false, command: "ls")) == .deny)
        #expect(FlowPolicy.answer(req(.edit)) == .allow)
        #expect(FlowPolicy.answer(req(.edit, inProject: false)) == .deny)
        #expect(FlowPolicy.answer(req(.read, inProject: false)) == .deny)
        #expect(FlowPolicy.answer(req(.network)) == .allow)
        #expect(FlowPolicy.answer(req(.run)) == .deny)
        #expect(FlowPolicy.answer(req(.orchestrate)) == .deny)
        #expect(FlowPolicy.answer(req(.graphEdit)) == .deny)
    }

    @Test func orchestrateDecisions() {
        func d(_ m: PermissionMode) -> PermissionDecision {
            PermissionEngine.decide(effect: .orchestrate, inProject: true, mode: m, rules: AllowRules(runs: true))
        }
        #expect(d(.ask) == .ask && d(.acceptEdits) == .ask)
        if case .deny = d(.plan) {} else { Issue.record("plan should deny") }
        #expect(d(.bypass) == .allow)
    }

    @Test func store() throws {
        let root = try tempProject()
        let personal = FileManager.default.temporaryDirectory.appendingPathComponent("flows-" + UUID().uuidString)
        FlowStore.personalOverride = personal
        defer { FlowStore.personalOverride = nil }
        func script(_ d: String) -> String { "export const meta = { name: 'n', description: '\(d)', phases: [] }\nreturn 1" }
        try FlowStore.save("review", script: script("project review"), root: root, personal: false)
        try FlowStore.save("review", script: script("personal review"), root: root, personal: true)
        try FlowStore.save("audit", script: script("personal audit"), root: root, personal: true)
        try "no meta here".write(to: personal.appendingPathComponent("broken.js"), atomically: true, encoding: .utf8)
        let list = FlowStore.list(root: root)
        #expect(list.map(\.name) == ["audit", "review"])
        #expect(list[1].meta.description == "project review" && !list[1].personal && list[0].personal)
        #expect(FlowStore.load("review", root: root)?.contains("project review") == true)
        #expect(FlowStore.load("audit", root: root)?.contains("personal audit") == true)
        #expect(FlowStore.load("nope", root: root) == nil)
        let section = FlowStore.promptSection(list)
        #expect(section == "Saved flows (run with flow {name}):\n- audit: personal audit\n- review: project review")
        #expect(FlowStore.promptSection([]) == "")
    }

    @Test func tool() async throws {
        let root = try tempProject()
        let ctx = ToolContext(root: root)
        let tool = FlowTool()
        let args = ToolArgs(["script": goodScript])
        let p = try tool.preview(args, ctx)
        #expect(p.title == "Flow: x")
        let detail = p.detail ?? ""
        #expect(detail.contains("Phases: A → B") && detail.contains("Up to 30 agents, 4 at a time"))
        #expect(detail.contains("Edit agents work in their own git worktrees") && detail.contains("Does x"))
        #expect(throws: ToolError.self) { try tool.preview(ToolArgs(["script": "return 1"]), ctx) }
        #expect(throws: ToolError.self) { try tool.preview(ToolArgs(["name": "missing"]), ctx) }

        let none = try await tool.run(args, ctx)
        #expect(none.isError)

        let host = FakeHost()
        ctx.flows = host
        ctx.callID = "call-9"
        let out = try await tool.run(ToolArgs(["script": goodScript, "args": ["k": 1]]), ctx)
        #expect(!out.isError && out.text.contains("started (run r1)"))
        #expect(host.calls.count == 1 && host.calls[0].meta.name == "x" && host.calls[0].callID == "call-9")
        #expect(host.calls[0].args == json("{\"k\":1}"))

        try FlowStore.save("saved", script: goodScript, root: root, personal: false)
        let named = try await tool.run(ToolArgs(["name": "saved"]), ctx)
        #expect(named.text.contains("started") && host.calls.count == 2 && host.calls[1].script == goodScript)
    }

    private func runner(_ model: ScriptedModel, root: URL, runID: String = "t1") -> AgentFlowRunner {
        AgentFlowRunner(client: model, model: "m", root: root, flowName: "demo", runID: runID, codeGraph: false)
    }

    @Test func readAgent() async throws {
        let root = try tempProject()
        let seen = Usages()
        let model = ScriptedModel([[.content("Found it."), .usage(promptTokens: 10, completionTokens: 5), .finish(reason: "stop")]])
        let r = try await runner(model, root: root).run(FlowAgentSpec(id: "a1", label: "L", prompt: "look"), events: { seen.add($0) })
        #expect(r.text == "Found it." && r.json == nil && r.branch == nil)
        #expect(r.usage.promptTokens == 10 && r.usage.completionTokens == 5)
        #expect(seen.usages.last?.promptTokens == 10)
        #expect(model.requests[0].messages.last?.text.contains("look") == true)
        #expect(!model.requests[0].tools.contains { $0.name == "write_file" })

        let schema = json("{\"type\":\"object\",\"required\":[\"ok\"],\"properties\":{\"ok\":{\"type\":\"boolean\"}}}")
        let m2 = ScriptedModel([[.content("```json\n{\"ok\":true}\n```")]])
        let r2 = try await runner(m2, root: root).run(FlowAgentSpec(id: "a2", label: "L", prompt: "p", schema: schema), events: { _ in })
        #expect(r2.json == json("{\"ok\":true}"))
        #expect(m2.requests[0].messages.last?.text.contains("Reply with only JSON") == true)

        let m3 = ScriptedModel([[.content("I think yes")], [.content("{\"ok\":false}")]])
        let r3 = try await runner(m3, root: root).run(FlowAgentSpec(id: "a3", label: "L", prompt: "p", schema: schema), events: { _ in })
        #expect(r3.json == json("{\"ok\":false}") && m3.requests.count == 2)
        #expect(m3.requests[1].messages.last?.text.contains("wasn't valid JSON") == true)

        let m4 = ScriptedModel([[.content("nope")], [.content("still nope")]])
        await #expect(throws: ToolError.self) {
            _ = try await runner(m4, root: root).run(FlowAgentSpec(id: "a4", label: "L", prompt: "p", schema: schema), events: { _ in })
        }
    }

    @Test func editAgent() async throws {
        let repo = try await makeRepo()
        let model = ScriptedModel([[call("write_file", ["path": "flow.txt", "content": "hi\n"]), .finish(reason: "tool_calls")],
                                   [.content("Wrote flow.txt.")]])
        let seen = Usages()
        let r = try await runner(model, root: repo).run(
            FlowAgentSpec(id: "a1", label: "Add File!", prompt: "write it", mode: .edit), events: { seen.add($0) })
        let branch = try #require(r.branch)
        #expect(branch.hasPrefix("aistudio/flow-t1-add-file"))
        #expect(seen.steps.count == 1)
        let subject = await sh("git -C '\(repo.path)' log '\(branch)' --format=%s -1", repo)
        #expect(subject.contains("Flow demo"))
        let wt = repo.appendingPathComponent(".aistudio/worktrees/" + (branch.split(separator: "/").last.map(String.init) ?? ""))
        #expect(FileManager.default.fileExists(atPath: wt.appendingPathComponent("flow.txt").path))
        #expect(!FileManager.default.fileExists(atPath: repo.appendingPathComponent("flow.txt").path))

        let idle = ScriptedModel([[.content("Nothing to do.")]])
        let r2 = try await runner(idle, root: repo, runID: "t2").run(
            FlowAgentSpec(id: "a2", label: "Idle", prompt: "nothing", mode: .edit), events: { _ in })
        #expect(r2.branch == nil && r2.text.contains("No changes"))

        let plain = try tempProject()
        await #expect(throws: ToolError.self) {
            _ = try await runner(idle, root: plain).run(FlowAgentSpec(id: "a3", label: "x", prompt: "p", mode: .edit), events: { _ in })
        }
    }
}
