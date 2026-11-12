import Foundation
import Testing
@testable import Agent

private let root = URL(fileURLWithPath: "/abs")

private let queryTiny = """
Graph: /abs/graphify-out/graph.json (6 nodes) | Traversal: BFS depth=2 | Start: ['greet()'] | 5 nodes found

NODE greet() [src=Sources/Greeter.swift loc=L1 community=0]
NODE .main() [src=Sources/Main.swift loc=L2 community=1]
NODE String [src= loc= community=0]
EDGE greet() --references [EXTRACTED context=return_type]--> String at=Sources/Greeter.swift:L1
EDGE .main() --calls [INFERRED context=call]--> greet() at=/abs/Sources/Main.swift:L3
EDGE greet() <--calls [INFERRED context=call]-- .main() at=Sources/Main.swift:L3
"""

private let queryReal = """
Graph: graphify-out/graph.json (9541 nodes) | Traversal: BFS depth=2 | Start: ['Engine', 'engine()', 'PermissionEngine', '.decide()', 'PermissionMode'] | 303 nodes found

[!] TRUNCATED: showing 30 of 303 nodes (~800-token budget). The answer may be among the 273 cut nodes — raise the token budget (CLI: --budget) or narrow the query (e.g. context_filter=['call'], or get_node for a specific symbol).

NODE Engine [src=Sources/AIStudioApp/Engine.swift loc=L45 community=Engine]
NODE engine() [src=server/gpu/tests/test_llm.py loc=L229 community=test_llm.py]
NODE PermissionEngine [src=Sources/Agent/Permissions.swift loc=L96 community=AllowRules]
NODE .decide() [src=Sources/Agent/Permissions.swift loc=L99 community=AllowRules]
NODE PermissionMode [src=Sources/Agent/Permissions.swift loc=L16 community=tempProject]
NODE AppModel [src=Sources/AIStudioApp/AppModel.swift loc=L66 community=AppModel]
NODE Sendable [src= loc= community=Sendable]
NODE String [src= loc= community=String]
NODE ChatDocument [src=Sources/AIStudioApp/Chat/ChatModel.swift loc=L10 community=ChatDocument]
NODE Codable [src= loc= community=Sendable]
NODE ToolContext [src=Sources/Agent/Tools.swift loc=L77 community=ToolContext]
NODE GraphExecutor [src=Sources/GraphRuntime/Executor.swift loc=L72 community=GraphExecutor]
... (truncated — 273 more nodes cut by ~800-token budget. Narrow with context_filter=['call'] or use get_node for a specific symbol)
"""

private let explainRaw = """
Node: PermissionEngine
  ID:        sources_agent_permissions_permissionengine
  Source:    Sources/Agent/Permissions.swift L96
  Type:      code
  Community: AllowRules
  Degree:    2

Connections (2):
  --> .decide() [method] [EXTRACTED] Sources/Agent/Permissions.swift:L99
  <-- Permissions.swift [contains] [EXTRACTED] Sources/Agent/Permissions.swift:L96
  --> Sendable [protocol] [INFERRED]
"""

private let pathRaw = """
Shortest path (4 hops):
  ChatView --references [EXTRACTED]--> ChatDocument --references [EXTRACTED]--> AllowRules <--contains [EXTRACTED]-- Permissions.swift --contains [EXTRACTED]--> PermissionEngine
"""

private let affectedRaw = """
Affected nodes for GitSafety
Relations: calls, indirect_call, references, imports
Depth: 2
- .decide() [calls] Sources/Agent/Permissions.swift:L111
- .safetyTable() [calls] Tests/AgentTests/GitTests.swift:L53
"""

private func tmpDir(_ name: String) throws -> URL {
    let d = FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    return d.standardizedFileURL.resolvingSymlinksInPath()
}

private func write(_ text: String, _ url: URL) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try text.write(to: url, atomically: true, encoding: .utf8)
}

struct CodeGraphTests {
    @Test func compactQuery() {
        let out = GraphOutput.compact(queryTiny, root: root)
        #expect(out.hasPrefix("Start: greet() (5 nodes)"))
        #expect(out.contains("greet()  Sources/Greeter.swift:1"))
        #expect(!out.contains("String"))
        #expect(out.contains("greet() --references") == false)
        #expect(out.contains(".main() --calls--> greet()  Sources/Main.swift:3"))
        #expect(out.contains("greet() <--calls-- .main()  Sources/Main.swift:3"))
        #expect(!out.contains("community=") && !out.contains("[EXTRACTED") && !out.contains("[INFERRED") && !out.contains("/abs/"))
    }

    @Test func compactTruncation() {
        let out = GraphOutput.compact(queryReal, root: root)
        #expect(out.hasSuffix("… 273 more; narrow the question or raise budget"))
        #expect(!out.contains("TRUNCATED") && !out.contains("(truncated"))
        #expect(!out.contains("Sendable") && !out.contains("Codable"))
        #expect(out.contains("Start: Engine, engine(), PermissionEngine, .decide(), PermissionMode (303 nodes)"))
        #expect(Double(out.count) <= Double(queryReal.count) * 0.6, "\(out.count) of \(queryReal.count)")
    }

    @Test func compactExplainPathAffected() {
        let e = GraphOutput.compact(explainRaw, root: root)
        #expect(e.contains("Node: PermissionEngine"))
        #expect(e.contains("Source: Sources/Agent/Permissions.swift:96"))
        #expect(!e.contains("ID:") && !e.contains("Type:") && !e.contains("Community:") && !e.contains("Degree:"))
        #expect(e.contains("--> .decide() [method] Sources/Agent/Permissions.swift:99"))
        #expect(!e.contains("EXTRACTED") && !e.contains("Sendable"))
        let p = GraphOutput.compact(pathRaw, root: root)
        #expect(p.contains("ChatView --references--> ChatDocument"))
        #expect(!p.contains("EXTRACTED"))
        let a = GraphOutput.compact(affectedRaw, root: root)
        #expect(!a.contains("Relations:"))
        #expect(a.contains("- .decide() [calls] Sources/Agent/Permissions.swift:111"))
    }

    @Test func hubs() {
        let raw = "God nodes (most connected):\n  1. AppModel - 533 edges\n  2. ChatDocument - 169 edges\n  3. Graph - 158 edges\n"
        #expect(GraphOutput.hubsLine(raw) == "Main hubs: AppModel, ChatDocument, Graph")
        #expect(GraphOutput.hubsLine("nothing") == "")
    }

    @Test func validation() throws {
        let tool = CodeGraphTool(graphs: ProjectGraphs(cacheRoot: try tmpDir("cg-cache"), binary: "/nonexistent"))
        let ctx = ToolContext(root: try tmpDir("cg-root"))
        #expect(throws: ToolError.self) { try tool.preview(ToolArgs(["action": "bogus"]), ctx) }
        #expect(throws: ToolError.self) { try tool.preview(ToolArgs(["action": "query"]), ctx) }
        #expect(throws: ToolError.self) { try tool.preview(ToolArgs(["action": "path", "target": "A"]), ctx) }
        #expect(try tool.preview(ToolArgs(["action": "hubs"]), ctx).title == "Map: hubs")
        #expect(try tool.preview(ToolArgs(["action": "path", "target": "A", "to": "B"]), ctx).title == "Map: path A → B")
        let p = try tool.preview(ToolArgs(["action": "explain", "target": "X"]), ctx)
        #expect(p.title == "Map: explain X" && p.kind == .read && p.paths == [ctx.root])
    }

    @Test func missingBinaryIsAnError() async throws {
        let tool = CodeGraphTool(graphs: ProjectGraphs(cacheRoot: try tmpDir("cg-cache"), binary: "/nonexistent/graphify"))
        let ctx = ToolContext(root: try tmpDir("cg-root"))
        await #expect(throws: ToolError.self) { try await tool.run(ToolArgs(["action": "hubs"]), ctx) }
    }

    @Test func readIsAllowedInAskAndPlan() {
        for m in [PermissionMode.ask, .plan] {
            #expect(PermissionEngine.decide(effect: CodeGraphTool().effect, inProject: true, command: nil, mode: m, rules: AllowRules()) == .allow)
        }
    }

    @Test func systemPromptMentionsTheTool() async throws {
        let dir = try tmpDir("cg-prompt")
        let with = await ProjectContext.systemPrompt(root: dir, mode: .ask, projectMap: "Main hubs: A, B")
        #expect(with.contains("code_graph") && with.contains("Main hubs: A, B"))
        let without = await ProjectContext.systemPrompt(root: dir, mode: .ask)
        #expect(!without.contains("code_graph"))
        #expect(TaskTool.systemPrompt(root: dir, codeGraph: true).contains("code_graph"))
        #expect(!TaskTool.systemPrompt(root: dir).contains("code_graph"))
    }

    @Test func realGraphify() async throws {
        let cache = try tmpDir("cg-cache")
        let graphs = ProjectGraphs(cacheRoot: cache)
        guard let bin = await graphs.binary() else { return }
        let project = try tmpDir("cg-project")
        try write("func greet() -> String {\n    \"hi\"\n}\n", project.appendingPathComponent("Sources/Greeter.swift"))
        try write("struct App {\n    func main() {\n        print(greet())\n    }\n}\n", project.appendingPathComponent("Sources/Main.swift"))
        let tool = CodeGraphTool(graphs: graphs)
        let ctx = ToolContext(root: project)

        let q = try await tool.run(ToolArgs(["action": "query", "target": "greet"]), ctx)
        #expect(q.text.contains("Sources/Greeter.swift:1"), "\(q.text)")
        #expect(!FileManager.default.fileExists(atPath: project.appendingPathComponent("graphify-out").path))
        let a = try await tool.run(ToolArgs(["action": "affected", "target": "greet()"]), ctx)
        #expect(a.text.contains("Sources/Main.swift:3"), "\(a.text)")
        let miss = try await tool.run(ToolArgs(["action": "explain", "target": "Nonexistent123"]), ctx)
        #expect(miss.text.hasPrefix("No match in the project map"))

        try write("func wave() -> String {\n    \"bye\"\n}\n", project.appendingPathComponent("Sources/Wave.swift"))
        await graphs.markStale(root: project)
        let w = try await tool.run(ToolArgs(["action": "query", "target": "wave"]), ctx)
        #expect(w.text.contains("Sources/Wave.swift:1"), "\(w.text)")

        // A project with its own graph is refreshed in place and never touches the cache.
        let own = try tmpDir("cg-own")
        try write("func greet() -> String {\n    \"hi\"\n}\n", own.appendingPathComponent("Sources/Greeter.swift"))
        let seed = await ShellRunner().run("\(shellQuote(bin)) update \(shellQuote(own.path))", cwd: own, timeout: 120)
        #expect(seed.status == 0, "\(seed.output)")
        let cacheBefore = (try? FileManager.default.contentsOfDirectory(atPath: cache.path))?.count ?? 0
        let ownGraphs = ProjectGraphs(cacheRoot: try tmpDir("cg-cache2"), binary: bin)
        let g = try await ownGraphs.graphFile(root: own)
        #expect(g.path == own.appendingPathComponent("graphify-out/graph.json").path)
        let empty = (try? FileManager.default.contentsOfDirectory(atPath: (try tmpDir("cg-unused")).path))?.isEmpty ?? true
        #expect(empty && cacheBefore >= 1)
        await ownGraphs.markStale(root: own)
        #expect(try await ownGraphs.graphFile(root: own) == g)
    }
}
