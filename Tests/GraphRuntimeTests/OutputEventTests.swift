import Foundation
import Testing
@testable import GraphRuntime

struct GraphReaderNode: NodeImplementation {
    static let schema = NodeSchema(type: "GraphReader", category: "test", inputs: [InputSpec("in", .int)], outputs: [],
                                   isOutput: true)
    func execute(_ i: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        ["nodes": .int(context.graph.nodes.count), "tag": .string(context.graph.metadata?["tag"] ?? "")]
    }
}

@Suite(.serialized) struct OutputEventTests {
    @Test func outputNodesAnnounceTheirValuesAsTheyFinish() async throws {
        let r = registry()
        r.register(GraphReaderNode.self)
        let ex = GraphExecutor(registry: r)
        var outputs: [String] = []
        var order: [String] = []
        try await ex.run(sumGraph()) { e in
            switch e {
            case .nodeOutput(let n, let values):
                outputs.append(n)
                order.append("output:\(n)")
                if case .int(let v)? = values["result"] { #expect(v == 5) }
            case .nodeFinished(let n, _, _): order.append("finished:\(n)")
            case .finished: order.append("done")
            default: break
            }
        }
        #expect(outputs == ["sink"])
        #expect(order.suffix(3) == ["finished:sink", "output:sink", "done"])
        // Cached output nodes announce their values too.
        var cachedOutputs: [String] = []
        try await ex.run(sumGraph()) { e in if case .nodeOutput(let n, _) = e { cachedOutputs.append(n) } }
        #expect(cachedOutputs == ["sink"])
    }

    @Test func nodesSeeTheMetadataGraphWhenGiven() async throws {
        let r = registry()
        r.register(GraphReaderNode.self)
        var g = Graph(nodes: [
            GraphNode(id: "c", type: "Const", inputs: ["value": .value(.int(1))]),
            GraphNode(id: "read", type: "GraphReader", inputs: ["in": link("c", "out")]),
        ])
        var original = g
        original.metadata = ["tag": "original"]
        original.nodes.append(GraphNode(id: "note", type: "Const"))
        g.metadata = ["tag": "rewritten"]
        let result = try await GraphExecutor(registry: r).run(g, metadataGraph: original)
        #expect(result.outputs["read"]?["tag"].map { "\($0)" } == "\(Value.string("original"))")
        let plain = try await GraphExecutor(registry: r).run(g)
        #expect(plain.outputs["read"]?["tag"].map { "\($0)" } == "\(Value.string("rewritten"))")
    }
}
