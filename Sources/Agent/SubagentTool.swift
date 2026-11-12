import Foundation
import LLMClient

/// Hands a research question to a read-only subagent with its own conversation; only its report comes back.
public struct TaskTool: AgentTool {
    let client: any ChatStreaming
    let model: String
    let tools: [any AgentTool]
    let reasoningEffort: String?
    let inlineThinking: Bool

    public init(client: any ChatStreaming, model: String, tools: [any AgentTool] = TaskTool.defaultTools,
                reasoningEffort: String? = nil, inlineThinking: Bool = false) {
        self.client = client
        self.model = model
        self.tools = tools
        self.reasoningEffort = reasoningEffort
        self.inlineThinking = inlineThinking
    }

    public static var defaultTools: [any AgentTool] { [ReadFileTool(), ListFilesTool(), SearchTool()] }

    public let name = "task"
    public let description = "Hand a self-contained research question to a subagent that reads and searches on its own (read-only) and reports back. Use it for broad searches across many files, so their contents don't fill this conversation. Its prompt must be complete: it doesn't see this conversation. Several task calls in one message run at the same time."
    public var parameters: JSONValue {
        schema(["description": ("string", "A 3–6 word label"), "prompt": ("string", "The full task for the subagent")],
               required: ["description", "prompt"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        ToolPreview(kind: .read, title: "Task: \(try args.required("description"))", detail: try args.required("prompt"))
    }

    static func systemPrompt(root: URL, codeGraph: Bool = false) -> String {
        let day = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate])
        return "You are a research subagent working in the project at \(root.path) on the user's Mac. Answer the task below using the read-only tools; you can't edit files, run commands or ask the user. Finish with a concise report of what you found, with file paths and line numbers.\nToday: \(day)" + (codeGraph ? "\nUse code_graph first to locate code (it returns file:line), then read only those lines." : "")
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let parent = ctx.callID
        let prompt = try args.required("prompt")
        let sub = ToolContext(root: ctx.root)
        sub.readRoots = ctx.readRoots
        var config = AgentConfig(model: model, tools: tools, mode: .plan)
        config.maxSteps = 30
        config.reasoningEffort = reasoningEffort
        config.inlineThinking = inlineThinking
        let onOutput = ctx.onOutput
        let last = LastText()
        let outcome = try await AgentLoop.run(
            client: client, messages: [.system(TaskTool.systemPrompt(root: ctx.root, codeGraph: tools.contains { $0.name == "code_graph" })), .user(prompt)], config: config,
            context: sub, ask: { _ in .deny },
            events: { e in
                switch e {
                case .toolStarted(_, let preview): onOutput?(parent, "• \(preview.title)\n")
                case .assistantFinished(let content, _, _): last.set(content)
                default: break
                }
            })
        var final = last.value
        if outcome == .stepLimit { final += "\n(The subagent stopped after 30 steps.)" }
        return ToolOutput(clip(final.isEmpty ? "(The subagent returned nothing.)" : final, max: 20_000))
    }
}

private final class LastText: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    func set(_ s: String) { if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lock.withLock { text = s } } }
    var value: String { lock.withLock { text } }
}
