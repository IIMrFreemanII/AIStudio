import Foundation
import Testing
@testable import GraphRuntime

func editorRegistry() -> NodeRegistry {
    let r = registry()
    r.register([NoteNode.self, RerouteNode.self])
    return r
}

func socketType(_ g: Graph, _ r: NodeRegistry) -> (GraphNode, String, Bool) -> SocketType? {
    { node, socket, isInput in
        let s = r.schema(node.type) ?? g.subgraphSchema(node)
        return isInput ? s?.input(socket)?.type : s?.output(socket)?.type
    }
}

// Part of the serialized executor suite: these run counted test nodes too.
extension ExecutorTests {
    @Test func collapsingKeepsResultAndMapsPorts() async throws {
        let r = editorRegistry()
        let g = sumGraph(2, 3)
        let (c, inst) = try #require(g.collapsing(["add"], name: "Adder", typeOf: socketType(g, r)))
        #expect(c.nodes.map(\.id) == ["c1", "c2", "adder", "sink"])
        #expect(inst == "adder")
        let def = try #require(c.subgraphDef(for: c[inst]!)).def
        #expect(def.inputs.map(\.name) == ["a", "b"])
        #expect(def.inputs.map(\.type) == [.int, .int])
        #expect(def.outputs.map(\.name) == ["sum"])
        #expect(c["sink"]?.inputs["in"] == link("adder", "sum"))
        #expect(c[inst]?.inputs["a"] == link("c1", "out"))
        // The flattened graph runs the inner node under a prefixed id.
        let flat = try c.flattened()
        #expect(flat.nodes.map(\.id).contains("adder/add"))
        let res = try await GraphExecutor(registry: r).run(c)
        #expect(sinkResult(res) == 5)
        #expect(res.executed.contains("adder/add"))
    }

    @Test func oneSourceFeedingTwoInnerInputsIsOnePort() throws {
        let r = editorRegistry()
        let g = Graph(nodes: [
            GraphNode(id: "c", type: "Const", inputs: ["value": .value(.int(4))]),
            GraphNode(id: "add", type: "Add", inputs: ["a": link("c", "out"), "b": link("c", "out")]),
            GraphNode(id: "sink", type: "Sink", inputs: ["in": link("add", "sum")]),
        ])
        let (c, inst) = try #require(g.collapsing(["add"], name: "Double", typeOf: socketType(g, r)))
        let def = try #require(c.subgraphDef(for: c[inst]!)).def
        #expect(def.inputs.count == 1)
        #expect(def.inputs[0].targets.count == 2)
    }

    @Test func subgraphNodesHitTheCacheOfTheirExpandedForm() async throws {
        let r = editorRegistry()
        let exec = GraphExecutor(registry: r)
        let g = sumGraph(2, 3)
        _ = try await exec.run(g)
        let (c, _) = try #require(g.collapsing(["c1", "add"], name: "Part", typeOf: socketType(g, r)))
        let res = try await exec.run(c)
        #expect(sinkResult(res) == 5)
        // Signatures ignore node ids, so nothing reruns. (No shared counter: suites run in parallel.)
        #expect(res.executed.isEmpty)
        #expect(res.cached.contains("part/add"))
    }

    @Test func expandingRestoresAnEquivalentGraph() async throws {
        let r = editorRegistry()
        let g = sumGraph(2, 3)
        let (c, inst) = try #require(g.collapsing(["c1", "add"], name: "Part", typeOf: socketType(g, r)))
        let (e, inner) = try #require(c.expanding(inst))
        #expect(Set(inner) == ["c1", "add"])
        #expect(e.subgraphs == nil)
        #expect(e["add"]?.inputs["a"] == link("c1", "out"))
        #expect(e["add"]?.inputs["b"] == link("c2", "out"))
        #expect(e["sink"]?.inputs["in"] == link("add", "sum"))
        #expect(sinkResult(try await GraphExecutor(registry: r).run(e)) == 5)
    }

    @Test func nestedSubgraphsRun() async throws {
        let r = editorRegistry()
        let g = sumGraph(2, 3)
        let (c1, a) = try #require(g.collapsing(["add"], name: "Inner", typeOf: socketType(g, r)))
        let (c2, b) = try #require(c1.collapsing([a, "c2"], name: "Outer", typeOf: socketType(c1, r)))
        #expect(c2.subgraphs?.count == 2)
        let flat = try c2.flattened()
        #expect(flat.nodes.map(\.id).contains("\(b)/\(a)/add"))
        #expect(sinkResult(try await GraphExecutor(registry: r).run(c2)) == 5)
        // Survives a JSON round trip.
        let back = try JSONDecoder().decode(Graph.self, from: try c2.jsonData())
        #expect(sinkResult(try await GraphExecutor(registry: r).run(back)) == 5)
    }

    @Test func cycleThroughASubgraphIsRejected() throws {
        let r = editorRegistry()
        let g = sumGraph(2, 3)
        var (c, inst) = try #require(g.collapsing(["add"], name: "Adder", typeOf: socketType(g, r)))
        // Feed the instance's own output back into its input.
        var n = c[inst]!
        n.inputs["b"] = link(inst, "sum")
        c[inst] = n
        #expect(throws: GraphError.self) { try GraphExecutor(registry: r).validate(c) }
    }

    @Test func reroutesAreSplicedOutAndTypeChecked() async throws {
        let r = editorRegistry()
        var g = sumGraph(2, 3)
        g.nodes.append(GraphNode(id: "r1", type: "Reroute", inputs: ["in": link("c1", "out")]))
        g.nodes.append(GraphNode(id: "r2", type: "Reroute", inputs: ["in": link("r1", "out")]))
        g["add"]!.inputs["a"] = link("r2", "out")
        #expect(g.resolvedType("r2", "out") { r.schema($0.type) } == .int)
        #expect(sinkResult(try await GraphExecutor(registry: r).run(g)) == 5)
        #expect(try g.flattened().nodes.allSatisfy { $0.type != "Reroute" })

        // STRING through a reroute into an INT input still fails validation.
        var bad = g
        bad.nodes.append(GraphNode(id: "t", type: "Text"))
        bad["r1"]!.inputs["in"] = link("t", "text")
        #expect(throws: GraphError.typeMismatch(node: "add", input: "a", expected: .int, got: .string)) {
            try GraphExecutor(registry: r).validate(bad)
        }

        // A reroute with nothing connected leaves the input missing.
        var dangling = g
        dangling["r1"]!.inputs["in"] = nil
        #expect(throws: GraphError.missingInput(node: "add", input: "a")) {
            try GraphExecutor(registry: r).validate(dangling)
        }
    }

    @Test func notesNeverRun() async throws {
        let r = editorRegistry()
        var g = sumGraph(2, 3)
        g.nodes.append(GraphNode(id: "note", type: "Note", inputs: ["text": .value(.string("remember"))]))
        let res = try await GraphExecutor(registry: r).run(g)
        #expect(!res.executed.contains("note"))
        #expect(sinkResult(res) == 5)
    }

    @Test func groupsAndSubgraphsAreOptionalInJSON() throws {
        let old = try JSONDecoder().decode(Graph.self, from: Data(#"{"version":1,"nodes":[]}"#.utf8))
        #expect(old.groups == nil && old.subgraphs == nil)
        var g = sumGraph()
        g.groups = [GraphGroup(id: "g1", title: "Inputs", x: 0, y: 0, width: 300, height: 200)]
        let back = try JSONDecoder().decode(Graph.self, from: try g.jsonData())
        #expect(back.groups == g.groups)
        #expect(!sumGraph().jsonString().contains("groups"))
        #expect(g.groups![0].contains(x: 10, y: 10))
        #expect(!g.groups![0].contains(x: 310, y: 10))
    }

    @Test func socketTypeAnyAcceptsEverything() {
        #expect(SocketType.any.accepts(.image))
        #expect(SocketType.image.accepts(.any))
        #expect(!SocketType.image.accepts(.mask))
    }
}

extension ExecutorTests {
    @Test func setValueReachesIntoSubgraphs() async throws {
        let r = editorRegistry()
        let g = sumGraph(2, 3)
        var (c, inst) = try #require(g.collapsing(["c1", "add"], name: "Part", typeOf: socketType(g, r)))
        try c.setValue(.int(10), node: "\(inst)/c1", input: "value")
        #expect(sinkResult(try await GraphExecutor(registry: r).run(c)) == 13)
        #expect(throws: GraphError.self) { try c.setValue(.int(1), node: "\(inst)/nope", input: "value") }
    }
}
