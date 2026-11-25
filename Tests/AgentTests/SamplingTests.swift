import Foundation
import Testing
@testable import Agent
@testable import LLMClient

struct SamplingTests {
    @Test func configSamplingReachesTheRequest() async throws {
        let model = ScriptedModel([[.content("OK"), .finish(reason: "stop")]])
        var config = AgentConfig(model: "m")
        config.temperature = 0.7
        config.topP = 0.9
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: config,
                                    context: ToolContext(root: try tempProject()),
                                    ask: { _ in .allow }, events: { _ in })
        #expect(model.requests.first?.temperature == 0.7)
        #expect(model.requests.first?.topP == 0.9)
    }

    @Test func unsetSamplingLeavesTheServerDefault() async throws {
        let model = ScriptedModel([[.content("OK"), .finish(reason: "stop")]])
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: AgentConfig(model: "m"),
                                    context: ToolContext(root: try tempProject()),
                                    ask: { _ in .allow }, events: { _ in })
        #expect(model.requests.first?.temperature == nil)
        #expect(model.requests.first?.topP == nil)
    }

    @Test func summarizeUsesTheGivenInstructions() async throws {
        let model = ScriptedModel([[.content("A summary"), .finish(reason: "stop")]])
        let summary = try await Compactor.summarize(client: model, model: "m", messages: [.user("hello")], limit: nil,
                                                    system: "Summarize the scene.")
        #expect(summary == "A summary")
        #expect(model.requests.first?.messages.first?.text == "Summarize the scene.")
    }
}
