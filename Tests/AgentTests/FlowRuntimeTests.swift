import Foundation
import LLMClient
import Testing
@testable import Agent

/// Runs agents without a model: sleeps the delay in the prompt ("sleep 0.3"), then answers.
private final class FakeRunner: FlowAgentRunner, @unchecked Sendable {
    private let lock = NSLock()
    private var starts: [String: Date] = [:]
    private var ends: [String: Date] = [:]
    private var specs: [FlowAgentSpec] = []
    private var active = 0
    private(set) var maxActive = 0

    var calls: Int { lock.lock(); defer { lock.unlock() }; return specs.count }
    func start(_ id: String) -> Date? { lock.lock(); defer { lock.unlock() }; return starts[id] }
    func end(_ id: String) -> Date? { lock.lock(); defer { lock.unlock() }; return ends[id] }
    func spec(_ id: String) -> FlowAgentSpec? { lock.lock(); defer { lock.unlock() }; return specs.first { $0.id == id } }

    struct Failure: LocalizedError { var errorDescription: String? { "it broke" } }

    func run(_ spec: FlowAgentSpec, events: @escaping @Sendable (FlowAgentEvent) -> Void) async throws -> FlowAgentResult {
        lock.lock()
        specs.append(spec)
        starts[spec.id] = Date()
        active += 1
        maxActive = max(maxActive, active)
        lock.unlock()
        defer { lock.lock(); active -= 1; ends[spec.id] = Date(); lock.unlock() }

        events(.step("Read x"))
        if let r = spec.prompt.range(of: #"sleep ([0-9.]+)"#, options: .regularExpression),
           let secs = Double(spec.prompt[r].dropFirst(6)) {
            try await Task.sleep(nanoseconds: UInt64(secs * 1_000_000_000))
        }
        if spec.prompt.range(of: #"\bfail\b"#, options: .regularExpression) != nil { throw Failure() }
        var json: JSONValue?
        if case .object(let o)? = spec.schema, case .object(let props)? = o["properties"] {
            json = props["findings"] != nil
                ? .object(["findings": .array([.object(["title": .string("t")])])])
                : .object(["real": .bool(true), "reason": .string("ok")])
        }
        let text = spec.mode == .edit ? "done: \(spec.prompt)\nPASSED" : "done: \(spec.prompt)"
        return FlowAgentResult(text: text, json: json, branch: spec.mode == .edit ? "b-\(spec.id)" : nil)
    }
}

private final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [FlowRunEvent] = []
    func add(_ e: FlowRunEvent) { lock.lock(); items.append(e); lock.unlock() }
    var all: [FlowRunEvent] { lock.lock(); defer { lock.unlock() }; return items }
    var specs: [FlowAgentSpec] { all.compactMap { if case .agentStarted(let s) = $0 { return s } else { return nil } } }
    var finishedIDs: [String: FlowAgentResult] {
        var out: [String: FlowAgentResult] = [:]
        for case .agentFinished(let id, let r) in all { out[id] = r }
        return out
    }
    var failures: [String: String] {
        var out: [String: String] = [:]
        for case .agentFailed(let id, let e) in all { out[id] = e }
        return out
    }
}

private func exec(_ script: String, args: JSONValue? = nil, runner: FakeRunner = FakeRunner(), concurrency: Int = 8,
                  maxAgents: Int = 50, cache: [String: FlowAgentResult] = [:], events: Events = Events())
    async -> (status: FlowStatus, result: String?, error: String?) {
    let rt = FlowRuntime(script: script, args: args, runner: runner, concurrency: concurrency, maxAgents: maxAgents,
                         cache: cache, onEvent: { events.add($0) })
    return await rt.run()
}

@Suite struct FlowRuntimeTests {
    @Test func parallelAgentsOverlap() async {
        let runner = FakeRunner()
        let r = await exec("""
        return await parallel([() => agent('one sleep 0.3'), () => agent('two sleep 0.3')]);
        """, runner: runner)
        #expect(r.status == .done)
        #expect(r.result?.contains("done: one sleep 0.3") == true)
        #expect(r.result?.contains("done: two sleep 0.3") == true)
        let s2 = runner.start("a2")!, e1 = runner.end("a1")!
        #expect(s2 < e1)
    }

    @Test func pipelineItemsAdvanceIndependently() async {
        let runner = FakeRunner()
        let r = await exec("""
        const out = await pipeline([1, 2],
          (n) => agent('s1 item' + n + (n === 2 ? ' sleep 0.6' : ' sleep 0.05')),
          (prev, item) => agent('s2 item' + item));
        return out;
        """, runner: runner)
        #expect(r.status == .done)
        // a1 = item 1 stage 1, a2 = item 2 stage 1, a3 = item 1 stage 2.
        #expect(runner.spec("a3")?.prompt == "s2 item1")
        #expect(runner.start("a3")! < runner.end("a2")!)
    }

    @Test func concurrencyCapLimitsRunningAgents() async {
        let runner = FakeRunner()
        let r = await exec("""
        return (await parallel([1,2,3,4].map(i => () => agent('n' + i + ' sleep 0.15')))).length;
        """, runner: runner, concurrency: 2)
        #expect(r.result == "4")
        #expect(runner.maxActive == 2)
    }

    @Test func maxAgentsRejectsTheExtraAgent() async {
        let runner = FakeRunner()
        let r = await exec("""
        const out = [];
        for (let i = 0; i < 3; i++) {
          try { out.push(await agent('q' + i)); } catch (e) { out.push('ERR ' + e.message); }
        }
        return out;
        """, runner: runner, maxAgents: 2)
        #expect(r.status == .done)
        #expect(r.result?.contains("ERR agent limit reached (2)") == true)
        #expect(runner.calls == 2)
    }

    @Test func phasesAreReportedBeforeAgents() async {
        let events = Events()
        let r = await exec("""
        phase('Review');
        await agent('a');
        await agent('b', { phase: 'Fix' });
        log('hi');
        """, events: events)
        #expect(r.status == .done)
        #expect(r.result == nil)
        let all = events.all
        guard case .phase("Review") = all[0] else { Issue.record("first event: \(all[0])"); return }
        guard case .agentStarted(let a1) = all[1] else { Issue.record("second event: \(all[1])"); return }
        #expect(a1.phase == "Review")
        let specs = events.specs
        #expect(specs[1].phase == "Fix")
        let phases = all.compactMap { e -> String? in if case .phase(let p) = e { return p } else { return nil } }
        #expect(phases == ["Review", "Fix"])
        #expect(all.contains { if case .log("hi") = $0 { return true } else { return false } })
    }

    @Test func verifyEachAttachesVerdicts() async {
        let events = Events()
        let r = await exec("""
        const found = await agent('find', { schema: { type: 'object', properties: { findings: { type: 'array' } } } });
        return await verifyEach(found.findings.concat(found.findings));
        """, events: events)
        #expect(r.status == .done)
        let data = Data((r.result ?? "").utf8)
        let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
        #expect(arr?.count == 2)
        #expect(arr?.first?["title"] as? String == "t")
        let verdict = arr?.first?["verdict"] as? [String: Any]
        #expect(verdict?["real"] as? Bool == true)
        #expect(events.specs.map(\.label) == ["agent 1", "verify:1", "verify:2"])
    }

    @Test func implementAndCheckReturnsBranchAndVerdict() async {
        let r = await exec("return await implementAndCheck('x', { check: 'swift test' });")
        #expect(r.status == .done)
        let obj = (try? JSONSerialization.jsonObject(with: Data((r.result ?? "").utf8))) as? [String: Any]
        #expect(obj?["branch"] as? String == "b-a1")
        #expect(obj?["passed"] as? Bool == true)
        let missing = await exec("return await implementAndCheck('x');")
        #expect(missing.status == .failed)
    }

    @Test func failingAgentFailsTheRun() async {
        let events = Events()
        let r = await exec("await agent('please fail', { label: 'boomer' });", events: events)
        #expect(r.status == .failed)
        #expect(r.error?.contains("boomer") == true)
        #expect(events.failures["a1"]?.contains("it broke") == true)
    }

    @Test func scriptErrorsReportTheLine() async {
        let syntax = await exec("const x = ;")
        #expect(syntax.status == .failed)
        #expect(syntax.error?.contains("line") == true)
        let thrown = await exec("const a = 1;\nawait agent('x');\nthrow new Error('boom');")
        #expect(thrown.status == .failed)
        #expect(thrown.error?.contains("boom") == true)
        #expect(thrown.error?.contains("line 3") == true)
        let hygiene = await exec("fetch('http://x');")
        #expect(hygiene.status == .failed)
    }

    @Test func stopCancelsRunningAgents() async {
        let runner = FakeRunner()
        let events = Events()
        let rt = FlowRuntime(script: "return await parallel([() => agent('a sleep 5'), () => agent('b sleep 5')]);",
                             runner: runner, concurrency: 4, maxAgents: 10, onEvent: { events.add($0) })
        let t0 = Date()
        let task = Task { await rt.run() }
        try? await Task.sleep(nanoseconds: 300_000_000)
        rt.stop()
        rt.stop()
        let r = await task.value
        #expect(r.status == .stopped)
        #expect(Date().timeIntervalSince(t0) < 1.5)
        #expect(events.failures["a1"] == FlowErrors.stopped)
        #expect(events.failures["a2"] == FlowErrors.stopped)
    }

    @Test func resumeReusesFinishedAgents() async {
        let script = "const a = await agent('one'); const b = await agent('two', { mode: 'edit' }); return [a, b];"
        var run = FlowRun(chatID: nil, meta: FlowMeta(name: "t", description: ""), script: script, maxAgents: 10, concurrency: 2)
        let first = FakeRunner()
        let events = Events()
        let r1 = await exec(script, runner: first, events: events)
        for e in events.all { run.apply(e) }
        #expect(r1.status == .done)
        #expect(first.calls == 2)

        let cache = run.cache(branchExists: { _ in true })
        let second = FakeRunner()
        let events2 = Events()
        let r2 = await exec(script, runner: second, cache: cache, events: events2)
        #expect(r2.status == .done)
        #expect(r2.result == r1.result)
        #expect(second.calls == 0)
        #expect(events2.finishedIDs.count == 2)
        #expect(events2.finishedIDs.values.allSatisfy { $0.cached })
    }

    @Test func argsAreVisible() async {
        let r = await exec("return args.n * 2;", args: .object(["n": .number(21)]))
        #expect(r.result == "42")
        let none = await exec("return args;")
        #expect(none.result == "null")
    }

    @Test func metaBlockIsAccepted() async {
        let r = await exec("""
        export const meta = { name: 'x', description: 'y', phases: [{title:'A'}] }
        return meta.name;
        """)
        #expect(r.status == .done)
        #expect(r.result == "\"x\"")
    }
}
