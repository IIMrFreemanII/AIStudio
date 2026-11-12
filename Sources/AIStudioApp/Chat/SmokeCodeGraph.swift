import Agent
import Foundation
import LLMClient

extension SmokeTest {
    /// `--smoke-code-graph <url>` (a server with `AISTUDIO_FAKE_LLM=1`): a coding chat in a small Swift project is asked
    /// to "map the project". The fake calls code_graph (query "greet"), then read_file on the location it returned.
    /// code_graph must run without a permission prompt, return Greeter.swift with its line, and build the graph in the
    /// app's cache, not in the project. After a new file and markStale, the next query finds it. Passes with a note
    /// when graphify isn't installed.
    static func runCodeGraph(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-code-graph] \(s)"); fflush(stdout) }
        guard ProjectGraphs.findBinary() != nil else { say("skipped: graphify not installed"); say("OK"); return true }
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/agent" }?.id
        if mid == nil {
            do { mid = try await client.install(LLMInstallRequest(url: "fake/agent")).id } catch {
                say("install failed: \(error.localizedDescription)"); return false
            }
            for _ in 0..<100 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        guard let mid else { say("no model"); return false }
        let key = LLMKey(serverID: conn.id, modelID: mid)

        let project = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-graph-\(UUID().uuidString.prefix(8))")
            .standardizedFileURL.resolvingSymlinksInPath()
        let sources = project.appendingPathComponent("Sources")
        try? FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
        try? "func greet() -> String {\n    \"hi\"\n}\n".write(to: sources.appendingPathComponent("Greeter.swift"), atomically: true, encoding: .utf8)
        try? "struct App {\n    func main() {\n        print(greet())\n    }\n}\n"
            .write(to: sources.appendingPathComponent("Main.swift"), atomically: true, encoding: .utf8)

        let doc = model.newCodingChat(project: project, model: key)
        var ok = true
        func check(_ c: Bool, _ s: String) { say((c ? "✓ " : "✗ ") + s); if !c { ok = false } }
        var asked: [String] = []
        let answerer = Task { @MainActor in
            while !Task.isCancelled {
                if let r = doc.permission { asked.append(r.call.name); doc.answer(.allow) }
                try? await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        func finish(_ result: Bool) async -> Bool {
            answerer.cancel()
            model.deleteChat(doc.id)
            // The cached graph lives in <cache>/code-graphs/<hash>/graphify-out/graph.json.
            if let g = await ProjectGraphs.shared.existingGraph(root: project), g.path.contains("/code-graphs/") {
                try? FileManager.default.removeItem(at: g.deletingLastPathComponent().deletingLastPathComponent())
            }
            try? FileManager.default.removeItem(at: project)
            say(result ? "OK" : "FAILED")
            return result
        }

        check(model.codeGraphEnabled, "coding chats get code_graph")
        doc.draft = "map the project"
        model.send(doc)
        let start = Date()
        while doc.isStreaming || doc.transcript.turns.last?.role != .assistant {
            if Date().timeIntervalSince(start) > 60 { say("✗ no reply in 60 s"); return await finish(false) }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let tools = doc.transcript.turns.filter { $0.role == .tool }
        check(tools.map(\.toolStatus) == ["done", "done"],
              "code_graph then read_file ran (\(tools.map { "\($0.toolTitle ?? "?"): \($0.toolStatus ?? "?")" }))")
        check(asked.isEmpty, "no permission prompts (\(asked))")
        let graphText = tools.first?.text ?? ""
        check(tools.first?.toolTitle == "Map: query greet", "the card reads “Map: query greet” (\(tools.first?.toolTitle ?? "none"))")
        check(graphText.contains("Sources/Greeter.swift:1"), "the graph points at Sources/Greeter.swift:1 (\(graphText.prefix(120)))")
        check(!graphText.contains("community=") && !graphText.contains(project.path), "the answer is compacted")
        check(tools.last?.toolTitle?.contains("Greeter.swift") == true, "the agent read the file it was pointed to (\(tools.last?.toolTitle ?? "none"))")
        check(!FileManager.default.fileExists(atPath: project.appendingPathComponent("graphify-out").path), "nothing written into the project")
        let graph = await ProjectGraphs.shared.existingGraph(root: project)
        check(graph?.path.contains("/Caches/AIStudio/code-graphs/") == true, "the graph is in the app's cache (\(graph?.path ?? "none"))")

        // A file the graph hasn't seen: after markStale the next query rebuilds first.
        try? "func wave() -> String {\n    \"bye\"\n}\n".write(to: sources.appendingPathComponent("Wave.swift"), atomically: true, encoding: .utf8)
        await ProjectGraphs.shared.markStale(root: project)
        let wave = try? await CodeGraphTool().run(ToolArgs(["action": "query", "target": "wave"]), ToolContext(root: project))
        check(wave?.text.contains("Sources/Wave.swift:1") == true, "after an edit the map finds the new function (\(wave?.text.prefix(80) ?? "error"))")
        let hubs = await ProjectGraphs.shared.hubs(root: project)
        check(hubs.hasPrefix("Main hubs: "), "the prompt's hubs line (\(hubs))")
        return await finish(ok)
    }
}
