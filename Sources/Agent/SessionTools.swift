import Foundation
import LLMClient

/// What a new session starts with.
public struct SessionSpec: Equatable, Sendable {
    public enum Context: String, Sendable { case fresh, fork }
    public var title: String
    public var prompt: String
    public var context: Context
    public var worktree: Bool

    public init(title: String, prompt: String, context: Context = .fresh, worktree: Bool = true) {
        self.title = title
        self.prompt = prompt
        self.context = context
        self.worktree = worktree
    }
}

/// A session this chat started.
public struct SessionInfo: Equatable, Sendable {
    public var id: UUID
    public var title: String
    /// "working", "waiting", "done", "failed", "idle".
    public var status: String
    public var branch: String?
    public var lastReport: String?

    public init(id: UUID, title: String, status: String, branch: String? = nil, lastReport: String? = nil) {
        self.id = id
        self.title = title
        self.status = status
        self.branch = branch
        self.lastReport = lastReport
    }
}

/// What the app provides to one session's tools: it creates and tracks that session's children.
public protocol SessionHost: Sendable {
    /// How deep this session is (0: started by the user).
    var depth: Int { get }
    /// Shows a "New session" card the user can start; returns at once with a line for the model.
    func propose(_ spec: SessionSpec, callID: String) async -> String
    func start(_ spec: SessionSpec, callID: String) async throws -> SessionInfo
    /// This session's children, oldest first.
    func children() async -> [SessionInfo]
    func send(_ text: String, to id: UUID) async throws
    func handOff(_ spec: SessionSpec, callID: String) async throws -> SessionInfo
}

public enum SessionLimits {
    public static let maxWorking = 4
    public static let maxDepth = 2
}

private let noSessions = ToolOutput("Sessions aren't available here.", isError: true)

private func sessionSpec(_ args: ToolArgs) throws -> SessionSpec {
    let title = (args.string("title") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    let prompt = (args.string("prompt") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    if title.isEmpty { throw ToolError("missing \"title\"") }
    if prompt.isEmpty { throw ToolError("missing \"prompt\"") }
    let context = args.string("context").flatMap(SessionSpec.Context.init(rawValue:)) ?? .fresh
    let worktree = args.raw["worktree"] == nil ? true : args.bool("worktree")
    return SessionSpec(title: title, prompt: prompt, context: context, worktree: worktree)
}

private func sessionParameters() -> JSONValue {
    schema([
        "title": ("string", "Under 60 characters"),
        "prompt": ("string", "The complete brief: it doesn't see this conversation unless context is fork"),
        "worktree": ("boolean", "Work in its own git worktree and branch (when the project is a git repository)"),
    ], required: ["title", "prompt"], extra: [
        "context": .object(["type": .string("string"), "enum": .array([.string("fresh"), .string("fork")]),
                            "description": .string("fresh (default): a new conversation; fork: a copy of this one")]),
    ])
}

public struct ProposeSessionTool: AgentTool {
    public init() {}
    public let name = "propose_session"
    public let description = "Suggest a separate session for a task of its own you noticed (a follow-up, an unrelated fix). The user sees a card and decides; it doesn't start by itself."
    public var parameters: JSONValue { sessionParameters() }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let spec = try sessionSpec(args)
        return ToolPreview(kind: .read, title: "Propose session “\(spec.title)”", detail: spec.prompt)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let spec = try sessionSpec(args)
        guard let host = ctx.sessions else { return noSessions }
        return ToolOutput(await host.propose(spec, callID: ctx.callID))
    }
}

public struct StartSessionTool: AgentTool {
    public init() {}
    public let name = "start_session"
    public let description = "Start a separate session now for an independent task that should run in parallel: it gets the brief, works on its own (in its own worktree by default) and reports back here when it ends a reply."
    public var parameters: JSONValue { sessionParameters() }
    public let effect = ToolEffect.run

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let spec = try sessionSpec(args)
        return ToolPreview(kind: .run, title: "Start session “\(spec.title)”", detail: spec.prompt)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let spec = try sessionSpec(args)
        guard let host = ctx.sessions else { return noSessions }
        if host.depth >= SessionLimits.maxDepth {
            return ToolOutput("Sessions this deep can't start more sessions: do the work here.", isError: true)
        }
        let working = await host.children().filter { $0.status == "working" || $0.status == "waiting" }.count
        if working >= SessionLimits.maxWorking {
            return ToolOutput("\(working) sessions are already running: wait for one to finish (list_sessions).", isError: true)
        }
        let info = try await host.start(spec, callID: ctx.callID)
        return ToolOutput("Started session “\(info.title)” (\(info.id.uuidString))\(info.branch.map { " on branch \($0)" } ?? ""). It works in the background; its report arrives here as a message when it ends a reply. Don't wait for it: carry on or end your reply.")
    }
}

public struct ListSessionsTool: AgentTool {
    public init() {}
    public let name = "list_sessions"
    public let description = "List the sessions this chat started: their status, branch and last report."
    public var parameters: JSONValue { schema([:], required: []) }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        ToolPreview(kind: .read, title: "List sessions")
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        guard let host = ctx.sessions else { return noSessions }
        let children = await host.children()
        if children.isEmpty { return ToolOutput("No sessions started from this chat.") }
        let blocks = children.map { c -> String in
            let report = c.lastReport.map { clip($0, max: 300) } ?? "(no report yet)"
            return "- \(c.title) [\(c.status)] id \(c.id.uuidString)\(c.branch.map { " branch \($0)" } ?? "")\n  \(report)"
        }
        return ToolOutput(blocks.joined(separator: "\n"))
    }
}

public struct SendToSessionTool: AgentTool {
    public init() {}
    public let name = "send_to_session"
    public let description = "Send a follow-up message to a session this chat started; it replies in the background and reports back here."
    public var parameters: JSONValue {
        schema(["id": ("string", "The session's id (from list_sessions)"), "message": ("string", "The message")],
               required: ["id", "message"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let id = try args.required("id")
        let message = try args.required("message")
        // The title is only known to the app (and preview is synchronous): show the id.
        return ToolPreview(kind: .read, title: "Message “\(id)”", detail: message)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let id = try args.required("id")
        let message = try args.required("message")
        guard let host = ctx.sessions else { return noSessions }
        guard let uuid = UUID(uuidString: id.trimmingCharacters(in: .whitespaces)),
              await host.children().contains(where: { $0.id == uuid }) else {
            return ToolOutput("There is no session \(id) started from this chat.", isError: true)
        }
        try await host.send(message, to: uuid)
        return ToolOutput("Sent. Its reply will arrive here as a message.")
    }
}

public struct HandOffTool: AgentTool {
    public init() {}
    public let name = "hand_off"
    public let description = "Move the task you're in the middle of into its own session (a copy of this conversation plus your remaining plan), when the user asks about something else mid-task. Then end your reply in one line."
    public var parameters: JSONValue {
        schema(["title": ("string", "Under 60 characters"),
                "remaining_plan": ("string", "What's left to do: steps, files, what done looks like")],
               required: ["title", "remaining_plan"])
    }
    public let effect = ToolEffect.run

    private func spec(_ args: ToolArgs) throws -> SessionSpec {
        let title = (args.string("title") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let plan = (args.string("remaining_plan") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if title.isEmpty { throw ToolError("missing \"title\"") }
        if plan.isEmpty { throw ToolError("missing \"remaining_plan\"") }
        return SessionSpec(title: title, prompt: plan, context: .fork, worktree: true)
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let s = try spec(args)
        return ToolPreview(kind: .run, title: "Hand off to “\(s.title)”", detail: s.prompt)
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let s = try spec(args)
        guard let host = ctx.sessions else { return noSessions }
        let info = try await host.handOff(s, callID: ctx.callID)
        return ToolOutput("Handed off to session “\(info.title)” (\(info.id.uuidString)). Stop working on it here and end your reply in one line.")
    }
}

public enum SessionTools {
    public static var all: [any AgentTool] {
        [ProposeSessionTool(), StartSessionTool(), ListSessionsTool(), SendToSessionTool(), HandOffTool()]
    }
}
