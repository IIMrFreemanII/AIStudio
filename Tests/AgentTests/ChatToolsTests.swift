import Foundation
import Testing
@testable import Agent
@testable import LLMClient

struct ChatToolsTests {
    static let askArgs: [String: Any] = ["questions": [[
        "question": "Which database?", "header": "Database",
        "options": [["label": "SQLite (Recommended)", "description": "One file"], ["label": "Postgres"]],
    ], [
        "question": "Which extras?", "multiSelect": true, "options": ["Redis", "Search", "Auth"],
    ]]]

    @Test func questionsAreCheckedAndAnswersSummed() throws {
        let qs = try UserQuestion.parse(ToolArgs(Self.askArgs))
        #expect(qs.count == 2 && qs[0].label == "Database" && qs[1].label == "Which extras?" && qs[1].multiSelect)
        #expect(qs[0].options[0].description == "One file" && qs[1].options.map(\.label) == ["Redis", "Search", "Auth"])
        #expect(throws: ToolError.self) { try UserQuestion.parse(ToolArgs(["questions": []])) }
        #expect(throws: ToolError.self) {
            try UserQuestion.parse(ToolArgs(["questions": [["question": "Q?", "options": ["only one"]]]]))
        }
        #expect(throws: ToolError.self) {
            try UserQuestion.parse(ToolArgs(["questions": Array(repeating: ["question": "Q?", "options": ["a", "b"]], count: 5)]))
        }
        let summary = UserAnswer.summary([UserAnswer(picked: ["SQLite (Recommended)"]),
                                          UserAnswer(picked: ["Redis", "Auth"], other: "a queue")], for: qs)
        #expect(summary == "Database: SQLite (Recommended)\nWhich extras?: Redis, Auth, Other: \"a queue\"")
        #expect(UserAnswer(other: "  ").isEmpty && UserAnswer().summary == "(no answer)")
    }

    @Test func askUserWaitsForTheAnswerOrASkip() async throws {
        let ctx = ToolContext(root: FileManager.default.temporaryDirectory)
        ctx.callID = "c9"
        let tool = AskUserTool()
        let none = try await tool.run(ToolArgs(Self.askArgs), ctx)
        #expect(none.isError && none.text.contains("can't be asked"))

        let asked = LockedBox<String>()
        ctx.askUser = { id, qs in
            asked.set(id + ":\(qs.count)")
            return [UserAnswer(picked: ["Postgres"]), UserAnswer(other: "nothing")]
        }
        let out = try await tool.run(ToolArgs(Self.askArgs), ctx)
        #expect(asked.value == "c9:2" && !out.isError)
        #expect(out.text == "The user answered:\nDatabase: Postgres\nWhich extras?: Other: \"nothing\"")
        #expect(out.detail == "Database: Postgres\nWhich extras?: Other: \"nothing\"")

        ctx.askUser = { _, _ in nil }
        let skipped = try await tool.run(ToolArgs(Self.askArgs), ctx)
        #expect(skipped.text.contains("skipped") && skipped.detail == "Skipped")
    }

    @Test func chatToolsRunInEveryModeWithoutAPermissionCard() async throws {
        for mode in [PermissionMode.ask, .plan, .acceptEdits, .bypass] {
            let model = ScriptedModel([[call("ask_user", Self.askArgs, id: "q1")],
                                       [call("show_widget", ["title": "T", "html": "<b>hi</b>"], id: "w1")],
                                       [.content("ok")]])
            let ctx = ToolContext(root: FileManager.default.temporaryDirectory)
            ctx.askUser = { _, _ in [UserAnswer(picked: ["SQLite (Recommended)"]), UserAnswer(picked: ["Redis"])] }
            let rec = Recorder()
            var config = AgentConfig(model: "m", tools: ChatTools.all, mode: mode)
            config.maxSteps = 5
            _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: config, context: ctx,
                                        ask: { r in rec.ask(r); return .deny }, events: { rec.add($0) })
            #expect(rec.asks.isEmpty)
            #expect(rec.events.filter { $0.hasPrefix("end") } == ["end done", "end done"])
            #expect(model.requests[1].messages.last?.text.contains("Database: SQLite") == true)
            #expect(model.requests[2].messages.last?.text.contains("Shown to the user") == true)
        }
    }

    @Test func widgetPagePutsThePolicyFirst() {
        let page = WidgetPage.html(body: "<!DOCTYPE html><html><head><script>evil()</script></head><body>x</body></html>",
                                   title: "A <b>", theme: ":root{--mg-bg:red}")
        let csp = page.range(of: "Content-Security-Policy")!
        #expect(page.hasPrefix("<!doctype html><html><head><meta http-equiv=\"Content-Security-Policy\""))
        #expect(page.range(of: "evil()")!.lowerBound > csp.lowerBound)
        #expect(page.contains("connect-src 'none'") && page.contains("https://cdn.jsdelivr.net"))
        #expect(page.contains("window.aistudio") && page.contains("ResizeObserver") && page.contains("--mg-bg:red"))
        #expect(page.contains("<title>A &lt;b&gt;</title>") && !page.contains("<!DOCTYPE html>"))
        let saved = WidgetPage.html(body: "x", title: "t", theme: "", bridge: false)
        #expect(!saved.contains("messageHandlers") && saved.contains("sendPrompt: function () {}"))
    }

    @Test func promptsMentionToolsOnlyWhenTheModelCallsThem() {
        let with = ChatPrompt.system(user: "Be brief.", tools: true)
        let without = ChatPrompt.system(user: nil, tools: false)
        #expect(with.hasPrefix("Be brief.") && with.contains("ask_user") && with.contains("<followups>"))
        #expect(!without.contains("ask_user") && !without.contains("show_widget") && without.contains("<followups>"))
        #expect(ChatPrompt.section(tools: false) == ChatPrompt.followUps)
    }
}

final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T?
    var value: T? { lock.withLock { _value } }
    func set(_ v: T) { lock.withLock { _value = v } }
}
