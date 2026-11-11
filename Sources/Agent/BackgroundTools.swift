import Foundation
import LLMClient

/// The message for an id that isn't a background command of this chat.
private func unknownJob(_ id: String, _ ctx: ToolContext) -> ToolOutput {
    let running = ctx.jobs?.running.map(\.id) ?? []
    let tail = running.isEmpty ? "None are running." : "Running: \(running.joined(separator: ", "))."
    return ToolOutput("There is no background command \(id). \(tail)", isError: true)
}

/// Reads the new output of a background command.
public struct CommandOutputTool: AgentTool {
    public init() {}
    public let name = "command_output"
    public let description = "Read the new output of a background command (since the last read) and whether it's still running."
    public var parameters: JSONValue {
        schema(["id": ("string", "The job id, like job1"),
                "wait": ("integer", "Seconds (0 to 60) to wait for new output or for it to end. Default 0.")],
               required: ["id"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let id = try args.required("id")
        return ToolPreview(kind: .command, title: "Output of \(id)", paths: [ctx.root])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let id = try args.required("id")
        guard let job = ctx.jobs?.job(id) else { return unknownJob(id, ctx) }
        let wait = min(60, max(0, args.int("wait") ?? 0))
        let deadline = Date().addingTimeInterval(TimeInterval(wait))
        var output = job.readNew()
        while output.isEmpty, job.isRunning, Date() < deadline {
            try? await Task.sleep(nanoseconds: 200_000_000)
            if Task.isCancelled { break }
            output = job.readNew()
        }
        if output.isEmpty, !job.isRunning { output = job.readNew() }
        let head: String
        if let status = job.status {
            head = "\(id): ended with exit status \(status)"
        } else {
            head = "\(id): running (\(Int(Date().timeIntervalSince(job.started))) s)"
        }
        return ToolOutput(head + "\n" + (output.isEmpty ? "(no new output)" : output))
    }
}

/// Stops a background command and everything it started.
public struct StopCommandTool: AgentTool {
    public init() {}
    public let name = "stop_command"
    public let description = "Stop a background command and everything it started."
    public var parameters: JSONValue { schema(["id": ("string", "The job id, like job1")], required: ["id"]) }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let id = try args.required("id")
        return ToolPreview(kind: .command, title: "Stop \(id)", paths: [ctx.root])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let id = try args.required("id")
        guard let job = ctx.jobs?.job(id) else { return unknownJob(id, ctx) }
        job.stop()
        for _ in 0..<30 where job.isRunning {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let output = job.readNew()
        return ToolOutput("Stopped \(id)." + (output.isEmpty ? "" : "\n" + output))
    }
}
