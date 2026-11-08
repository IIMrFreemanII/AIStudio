import Foundation
import Testing
@testable import GraphRuntime

/// Counts executions per node type so tests can tell cache hits from re-runs.
final class Counter: @unchecked Sendable {
    static let shared = Counter()
    private let lock = NSLock()
    private var counts: [String: Int] = [:]
    func hit(_ k: String) { lock.lock(); counts[k, default: 0] += 1; lock.unlock() }
    func get(_ k: String) -> Int { lock.lock(); defer { lock.unlock() }; return counts[k] ?? 0 }
    func reset() { lock.lock(); counts = [:]; lock.unlock() }
}

struct ConstNode: NodeImplementation {
    static let schema = NodeSchema(type: "Const", category: "test",
                                   inputs: [InputSpec("value", .int, default: .int(0))],
                                   outputs: [OutputSpec("out", .int)])
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        Counter.shared.hit("Const")
        return ["out": .int(try i.int("value"))]
    }
}

struct AddNode: NodeImplementation {
    static let schema = NodeSchema(type: "Add", category: "test",
                                   inputs: [InputSpec("a", .int), InputSpec("b", .int, default: .int(0))],
                                   outputs: [OutputSpec("sum", .int)])
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        Counter.shared.hit("Add")
        return ["sum": .int(try i.int("a") + i.int("b"))]
    }
}

struct TextNode: NodeImplementation {
    static let schema = NodeSchema(type: "Text", category: "test", inputs: [],
                                   outputs: [OutputSpec("text", .string)])
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["text": .string("hi")]
    }
}

/// Reports `steps` progress events, checking for cancellation between them.
struct SlowNode: NodeImplementation {
    static let schema = NodeSchema(type: "Slow", category: "test",
                                   inputs: [InputSpec("a", .int), InputSpec("steps", .int, default: .int(5))],
                                   outputs: [OutputSpec("out", .int)])
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let n = try i.int("steps")
        for s in 1...n {
            try context.checkCancellation()
            try await Task.sleep(nanoseconds: 20_000_000)
            context.progress(step: s, total: n)
        }
        Counter.shared.hit("Slow")
        return ["out": .int(try i.int("a"))]
    }
}

struct SinkNode: NodeImplementation {
    static let schema = NodeSchema(type: "Sink", category: "test", inputs: [InputSpec("in", .int)], outputs: [],
                                   isOutput: true)
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        Counter.shared.hit("Sink")
        return ["result": .int(try i.int("in"))]
    }
}

struct WriterNode: NodeImplementation {
    static let schema = NodeSchema(type: "Writer", category: "test", inputs: [InputSpec("in", .int)], outputs: [],
                                   isOutput: true, alwaysExecute: true)
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        Counter.shared.hit("Writer")
        return [:]
    }
}

func registry() -> NodeRegistry {
    let r = NodeRegistry()
    r.register([ConstNode.self, AddNode.self, TextNode.self, SlowNode.self, SinkNode.self, WriterNode.self])
    return r
}

func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }

/// c1 ─┐
///     ├─ add ── sink
/// c2 ─┘
func sumGraph(_ a: Int = 2, _ b: Int = 3) -> Graph {
    Graph(nodes: [
        GraphNode(id: "c1", type: "Const", inputs: ["value": .value(.int(a))]),
        GraphNode(id: "c2", type: "Const", inputs: ["value": .value(.int(b))]),
        GraphNode(id: "add", type: "Add", inputs: ["a": link("c1", "out"), "b": link("c2", "out")]),
        GraphNode(id: "sink", type: "Sink", inputs: ["in": link("add", "sum")]),
    ])
}

func sinkResult(_ r: RunResult) -> Int? {
    if case .int(let v)? = r.outputs["sink"]?["result"] { return v }
    return nil
}

// Tests touch a shared counter, so run them one at a time.
@Suite(.serialized) struct ExecutorTests {

    @Test func runsAndReturnsOutputs() async throws {
        Counter.shared.reset()
        let r = try await GraphExecutor(registry: registry()).run(sumGraph())
        #expect(sinkResult(r) == 5)
        #expect(r.executed == ["c1", "c2", "add", "sink"])
    }

    @Test func secondRunIsFullyCached() async throws {
        Counter.shared.reset()
        let ex = GraphExecutor(registry: registry())
        try await ex.run(sumGraph())
        let r = try await ex.run(sumGraph())
        #expect(sinkResult(r) == 5)
        #expect(r.executed.isEmpty)
        #expect(Set(r.cached) == ["c1", "c2", "add", "sink"])
        #expect(Counter.shared.get("Add") == 1)
    }

    @Test func changingOneWidgetRerunsOnlyDownstream() async throws {
        Counter.shared.reset()
        let ex = GraphExecutor(registry: registry())
        try await ex.run(sumGraph(2, 3))
        let r = try await ex.run(sumGraph(2, 10))
        #expect(sinkResult(r) == 12)
        #expect(r.executed == ["c2", "add", "sink"])
        #expect(r.cached == ["c1"])
    }

    @Test func positionsAndTitlesDoNotInvalidateCache() async throws {
        let ex = GraphExecutor(registry: registry())
        try await ex.run(sumGraph())
        var g = sumGraph()
        g.nodes[0].position = GraphPoint(x: 100, y: 200)
        g.nodes[2].title = "Renamed"
        let r = try await ex.run(g)
        #expect(r.executed.isEmpty)
    }

    @Test func alwaysExecuteNodesRerun() async throws {
        Counter.shared.reset()
        var g = sumGraph()
        g.nodes.append(GraphNode(id: "w", type: "Writer", inputs: ["in": link("add", "sum")]))
        let ex = GraphExecutor(registry: registry())
        try await ex.run(g)
        let r = try await ex.run(g)
        #expect(r.executed == ["w"])
        #expect(Counter.shared.get("Writer") == 2)
    }

    @Test func onlyNodesFeedingOutputsRun() async throws {
        var g = sumGraph()
        g.nodes.append(GraphNode(id: "orphan", type: "Const", inputs: ["value": .value(.int(9))]))
        let r = try await GraphExecutor(registry: registry()).run(g)
        #expect(!r.executed.contains("orphan"))
    }

    @Test func defaultsFillMissingLiterals() async throws {
        let g = Graph(nodes: [
            GraphNode(id: "c", type: "Const"),
            GraphNode(id: "add", type: "Add", inputs: ["a": link("c", "out")]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("add", "sum")]),
        ])
        let r = try await GraphExecutor(registry: registry()).run(g)
        #expect(sinkResult(r) == 0)
    }

    @Test func detectsCycles() {
        let g = Graph(nodes: [
            GraphNode(id: "a", type: "Add", inputs: ["a": link("b", "sum")]),
            GraphNode(id: "b", type: "Add", inputs: ["a": link("a", "sum")]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("b", "sum")]),
        ])
        #expect(throws: GraphError.self) { try GraphExecutor(registry: registry()).validate(g) }
        do {
            _ = try GraphExecutor(registry: registry()).validate(g)
        } catch let GraphError.cycle(ids) {
            #expect(Set(ids) == ["a", "b", "sink"])
        } catch { Issue.record("unexpected \(error)") }
    }

    @Test func rejectsTypeMismatch() {
        let g = Graph(nodes: [
            GraphNode(id: "t", type: "Text"),
            GraphNode(id: "add", type: "Add", inputs: ["a": link("t", "text")]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("add", "sum")]),
        ])
        #expect(throws: GraphError.typeMismatch(node: "add", input: "a", expected: .int, got: .string)) {
            try GraphExecutor(registry: registry()).validate(g)
        }
    }

    @Test func rejectsMissingRequiredInputAndUnknownTypes() {
        let missing = Graph(nodes: [GraphNode(id: "add", type: "Add"),
                                    GraphNode(id: "sink", type: "Sink", inputs: ["in": link("add", "sum")])])
        #expect(throws: GraphError.missingInput(node: "add", input: "a")) {
            try GraphExecutor(registry: registry()).validate(missing)
        }
        let unknown = Graph(nodes: [GraphNode(id: "x", type: "Nope")])
        #expect(throws: GraphError.unknownNodeType(node: "x", type: "Nope")) {
            try GraphExecutor(registry: registry()).validate(unknown)
        }
        let badLink = Graph(nodes: [GraphNode(id: "sink", type: "Sink", inputs: ["in": link("c", "out")])])
        #expect(throws: GraphError.unknownNode("c")) { try GraphExecutor(registry: registry()).validate(badLink) }
    }

    @Test func rejectsGraphWithoutOutputs() async {
        let g = Graph(nodes: [GraphNode(id: "c", type: "Const")])
        await #expect(throws: GraphError.noOutputs) { try await GraphExecutor(registry: registry()).run(g) }
    }

    @Test func cancellationStopsMidNodeAndLaterNodesDoNotRun() async throws {
        Counter.shared.reset()
        let g = Graph(nodes: [
            GraphNode(id: "c", type: "Const", inputs: ["value": .value(.int(1))]),
            GraphNode(id: "slow", type: "Slow", inputs: ["a": link("c", "out"), "steps": .value(.int(50))]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("slow", "out")]),
        ])
        let ex = GraphExecutor(registry: registry())
        let sawCancelled = Counter()
        let task = Task {
            try await ex.run(g) { e in
                if case .cancelled = e { sawCancelled.hit("cancelled") }
            }
        }
        try await Task.sleep(nanoseconds: 100_000_000)
        task.cancel()
        let outcome = await task.result
        guard case .failure(let error) = outcome else { Issue.record("run should have been cancelled"); return }
        #expect(error is CancellationError)
        #expect(Counter.shared.get("Slow") == 0)
        #expect(Counter.shared.get("Sink") == 0)
        #expect(sawCancelled.get("cancelled") == 1)

        // A cancelled node is not cached; the next run executes it.
        let r = try await ex.run(g)
        #expect(r.executed.contains("slow"))
        #expect(r.cached == ["c"])
    }

    @Test func progressEventsAreForwarded() async throws {
        let g = Graph(nodes: [
            GraphNode(id: "c", type: "Const"),
            GraphNode(id: "slow", type: "Slow", inputs: ["a": link("c", "out"), "steps": .value(.int(3))]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("slow", "out")]),
        ])
        let steps = Counter()
        try await GraphExecutor(registry: registry()).run(g) { e in
            if case .nodeProgress("slow", _, 3, _) = e { steps.hit("p") }
        }
        #expect(steps.get("p") == 3)
    }

    @Test func graphJSONRoundTrips() throws {
        var g = sumGraph()
        g.nodes[0].position = GraphPoint(x: 1, y: 2)
        try g.setValue(.string("x"), node: "c1", input: "label")
        let back = try JSONDecoder().decode(Graph.self, from: g.jsonData())
        #expect(back == g)
        #expect(back["add"]?.inputs["a"] == link("c1", "out"))
    }

    @Test func literalCoercion() throws {
        #expect(JSONValue.parse("5") == .int(5))
        #expect(JSONValue.parse("0.5") == .double(0.5))
        #expect(JSONValue.parse("true") == .bool(true))
        #expect(JSONValue.parse("a fox") == .string("a fox"))
        let g = Graph(nodes: [GraphNode(id: "c", type: "Const", inputs: ["value": .value(.string("nope"))])])
        #expect(throws: GraphError.self) { try GraphExecutor(registry: registry()).validate(g) }
    }
}

actor CreationCounter {
    var count = 0
    func bump() { count += 1 }
}

@Test func concurrentResourceGetsCreateOnce() async throws {
    let cache = ResourceCache()
    let counter = CreationCounter()
    async let a: Int = cache.get("model") {
        await counter.bump()
        try await Task.sleep(nanoseconds: 100_000_000)
        return 7
    }
    async let b: Int = cache.get("model") {
        await counter.bump()
        try await Task.sleep(nanoseconds: 100_000_000)
        return 8
    }
    let (x, y) = try await (a, b)
    #expect(x == y)
    #expect(await counter.count == 1)
    // Later calls hit the cache.
    let z: Int = try await cache.get("model") { 9 }
    #expect(z == x)
}

@Test func failedResourceCreationIsRetried() async throws {
    struct Boom: Error {}
    let cache = ResourceCache()
    await #expect(throws: Boom.self) { let _: Int = try await cache.get("m") { throw Boom() } }
    let v: Int = try await cache.get("m") { 3 }
    #expect(v == 3)
}
