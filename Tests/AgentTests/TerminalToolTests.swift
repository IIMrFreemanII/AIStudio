import Foundation
import Testing
@testable import Agent
@testable import LLMClient

struct TerminalToolTests {
    @Test func readsTheChosenTerminalWithinTheLimit() async throws {
        let ctx = ToolContext(root: FileManager.default.temporaryDirectory)
        let tool = ReadTerminalTool()

        let none = try await tool.run(ToolArgs([:]), ctx)
        #expect(none.isError && none.text.contains("no terminal"))

        let asked = LockedBox<String>()
        ctx.readTerminal = { which, lines in
            asked.set("\(which ?? "-"):\(lines)")
            return which == "closed" ? nil : "Terminal 1 of 1: “zsh”\n\n$ make\nerror: boom"
        }
        let out = try await tool.run(ToolArgs([:]), ctx)
        #expect(!out.isError && out.text.hasSuffix("error: boom") && asked.value == "-:200")

        _ = try await tool.run(ToolArgs(["lines": 5000, "terminal": "2"]), ctx)
        #expect(asked.value == "2:1000", "at most 1000 lines (\(asked.value ?? ""))")
        _ = try await tool.run(ToolArgs(["lines": 0]), ctx)
        #expect(asked.value == "-:1")

        let closed = try await tool.run(ToolArgs(["terminal": "closed"]), ctx)
        #expect(!closed.isError && closed.text.contains("No terminal is open"))

        let preview = try tool.preview(ToolArgs(["terminal": "zsh", "lines": 50]), ctx)
        #expect(preview.title == "Read terminal “zsh” (50 lines)" && preview.paths.isEmpty)
    }

    /// It only reads: free in every mode, plan mode included.
    @Test func itNeverAsks() {
        for mode in PermissionMode.allCases {
            #expect(PermissionEngine.decide(effect: ReadTerminalTool().effect, inProject: true, mode: mode,
                                            rules: AllowRules()) == .allow)
        }
    }
}
