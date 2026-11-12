import Foundation
import Testing
@testable import Agent
@testable import LLMClient

struct EffortTests {
    private func json(_ v: JSONValue) throws -> String {
        String(decoding: try JSONEncoder().encode(v), as: UTF8.self)
    }

    @Test func toolsDeclareEffort() throws {
        #expect(try json(TaskTool(client: ScriptedModel([]), model: "m").parameters).contains("\"effort\""))
        #expect(try json(StartSessionTool().parameters).contains("\"effort\""))
        #expect(try json(ProposeSessionTool().parameters).contains("\"effort\""))
    }

    @Test func taskPassesEffortToTheSubagent() async throws {
        let dir = try tempProject()
        for (given, parent, want) in [("high", "low", "high"), (nil, "low", "low"), (nil, nil, nil)] as [(String?, String?, String?)] {
            let model = ScriptedModel([[.content("report")]])
            let tool = TaskTool(client: model, model: "m", reasoningEffort: parent)
            var args: [String: Any] = ["description": "d", "prompt": "find it"]
            if let given { args["effort"] = given }
            _ = try await tool.run(ToolArgs(args), ToolContext(root: dir))
            #expect(model.requests.first?.reasoningEffort == want)
        }
    }

    @Test func sessionSpecParsesEffort() async throws {
        let host = FakeSessionHost()
        let c = ToolContext(root: try tempProject())
        c.sessions = host
        let brief: [String: Any] = ["title": "T", "prompt": "P", "effort": " high "]
        _ = try await StartSessionTool().run(ToolArgs(brief), c)
        _ = try await StartSessionTool().run(ToolArgs(["title": "T", "prompt": "P"]), c)
        #expect(host.started.map(\.effort) == ["high", nil])
    }

    @Test func modelDecodesEfforts() throws {
        let base = #""id":"m","name":"M","repo":"o/m","gpu":"H100","count":1,"context":8192"#
        let with = try JSONDecoder().decode(LLMModel.self, from: Data("{\(base),\"efforts\":[\"off\",\"on\"]}".utf8))
        let none = try JSONDecoder().decode(LLMModel.self, from: Data("{\(base),\"efforts\":[]}".utf8))
        let old = try JSONDecoder().decode(LLMModel.self, from: Data("{\(base)}".utf8))
        #expect(with.efforts == ["off", "on"] && none.efforts == [] && old.efforts == nil)
    }

    @Test func turnEffortRoundTrips() throws {
        var t = ChatTurn(role: .assistant, text: "hi")
        t.effort = "medium"
        let back = try JSONDecoder().decode(ChatTurn.self, from: try JSONEncoder().encode(t))
        #expect(back.effort == "medium")
        var plain = t; plain.effort = nil
        #expect(try JSONDecoder().decode(ChatTurn.self, from: try JSONEncoder().encode(plain)).effort == nil)
    }
}
