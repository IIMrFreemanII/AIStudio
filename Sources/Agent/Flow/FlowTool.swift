import Foundation
import LLMClient

/// Starts a multi-agent flow in the background; the user approves each run.
public struct FlowTool: AgentTool {
    public init() {}

    public let name = "flow"
    public let description = "Run a multi-agent flow: a JavaScript script (Claude Code workflow format) that starts many agents in parallel or in stages and combines their results. Use it only for big jobs with independent parts (review then verify, research fan-outs, several changes at once); the user approves each run. It runs in the background: its result arrives in this chat when it finishes."
    public var parameters: JSONValue {
        schema(["script": ("string", "The flow script"), "name": ("string", "A saved flow's name, instead of script"),
                "args": ("object", "Values the script reads as `args`")], required: [])
    }
    public let effect = ToolEffect.orchestrate

    private func resolve(_ args: ToolArgs, _ ctx: ToolContext) throws -> (script: String, meta: FlowMeta) {
        let script: String
        if let s = args.string("script"), !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            script = s
        } else if let name = args.string("name"), !name.isEmpty {
            guard let s = FlowStore.load(name, root: ctx.root) else { throw ToolError("no saved flow named \(name)") }
            script = s
        } else {
            throw ToolError("give either \"script\" or \"name\"")
        }
        do {
            return (script, try FlowScript.meta(script))
        } catch {
            throw ToolError(error.localizedDescription)
        }
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (script, meta) = try resolve(args, ctx)
        var lines: [String] = []
        if !meta.description.isEmpty { lines.append(meta.description) }
        if !meta.phases.isEmpty { lines.append("Phases: " + meta.phases.joined(separator: " → ")) }
        lines.append("Up to \(ctx.flowLimits.maxAgents) agents, \(ctx.flowLimits.concurrency) at a time")
        if FlowScript.editAgentMentions(script) > 0 { lines.append("Edit agents work in their own git worktrees and branches.") }
        lines.append("")
        lines.append(script)
        return ToolPreview(kind: .run, title: "Flow: \(meta.name)", detail: lines.joined(separator: "\n"), paths: [ctx.root])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        guard let host = ctx.flows else { return ToolOutput("Flows aren't available here.", isError: true) }
        let (script, meta) = try resolve(args, ctx)
        var flowArgs: JSONValue?
        if let raw = args.raw["args"], JSONSerialization.isValidJSONObject(["a": raw]),
           let data = try? JSONSerialization.data(withJSONObject: raw, options: [.fragmentsAllowed]) {
            flowArgs = try? JSONDecoder().decode(JSONValue.self, from: data)
        }
        let id = try await host.start(script: script, meta: meta, args: flowArgs, callID: ctx.callID)
        return ToolOutput("Flow “\(meta.name)” started (run \(id)). It runs in the background; its result arrives in this chat as a message when it finishes. Don't wait or poll for it; tell the user it's running.")
    }
}
