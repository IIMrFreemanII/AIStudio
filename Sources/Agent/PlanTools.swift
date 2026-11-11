import Foundation
import LLMClient

/// Plan mode's way out: shows the finished plan for approval; if approved the mode switches mid-reply.
public struct ExitPlanTool: AgentTool {
    public init() {}
    public let name = "exit_plan"
    public let description = "Present your finished plan for the user to approve (plan mode only). If they approve, the mode switches and you go ahead and carry it out in this same reply; otherwise keep planning with their note."
    public var parameters: JSONValue { schema(["plan": ("string", "The plan, in Markdown")], required: ["plan"]) }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        ToolPreview(kind: .read, title: "Plan ready for approval", detail: try args.required("plan"))
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let plan = try args.required("plan")
        if ctx.mode != .plan { return ToolOutput("You're not in plan mode: go ahead.") }
        guard let approve = ctx.approvePlan else {
            return ToolOutput("The plan can't be approved right now: stop and present it to the user.")
        }
        switch await approve(ctx.callID, plan) {
        case .approve(let m):
            ctx.mode = m
            return ToolOutput("The user approved the plan. The mode is now \(m.title): carry out the plan now.",
                              detail: "Approved — \(m.title)")
        case .keepPlanning(let note):
            return ToolOutput("The user wants to keep planning." + (note.map { " Their note: \($0)" } ?? "")
                              + " Revise the plan, then call exit_plan again.",
                              detail: "Keep planning" + (note.map { ": \($0)" } ?? ""))
        }
    }
}
