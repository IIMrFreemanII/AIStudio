import Foundation
import LLMClient

/// Anything that streams chat completions (the GPU server's client, or a script in tests).
public protocol ChatStreaming: Sendable {
    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error>
}

extension OpenAIChatClient: ChatStreaming {}

/// How a tool call ended.
public enum ToolStatus: String, Codable, Sendable {
    case done, failed, denied
}

/// What the loop reports as it goes, in order.
public enum AgentEvent: Sendable {
    /// A model request starts (a new assistant message).
    case assistantStarted
    case delta(ChatEvent)
    /// The model's message is complete; its tool calls run next (if any).
    case assistantFinished(content: String, reasoning: String, toolCalls: [ToolCall])
    case toolStarted(call: ToolCall, preview: ToolPreview)
    case toolOutput(callID: String, chunk: String)
    case toolFinished(callID: String, output: ToolOutput, status: ToolStatus)
}

/// A call that needs the user's answer.
public struct PermissionRequest: Sendable {
    public var call: ToolCall
    public var preview: ToolPreview
    public var effect: ToolEffect
    public var inProject: Bool

    public init(call: ToolCall, preview: ToolPreview, effect: ToolEffect, inProject: Bool) {
        self.call = call
        self.preview = preview
        self.effect = effect
        self.inProject = inProject
    }
}

public enum PermissionAnswer: Sendable, Equatable {
    case allow
    case deny
    /// Allow, and the same kind from now on in this chat (all edits, or this command's prefix).
    case allowAlways
}

public enum AgentOutcome: Sendable, Equatable {
    /// The model answered without asking for more tools.
    case finished
    /// It kept calling tools past the step limit.
    case stepLimit
}

public struct AgentConfig: Sendable {
    public var model: String
    public var tools: [any AgentTool]
    public var mode: PermissionMode
    public var rules: AllowRules
    public var maxSteps = 50
    public var reasoningEffort: String?
    /// Split `<think>` out of the content (servers without a reasoning parser).
    public var inlineThinking = false
    /// The model takes images: tools' images are sent to it (else it's told it can't see them).
    public var vision = false

    public init(model: String, tools: [any AgentTool] = CodingTools.all, mode: PermissionMode = .ask,
                rules: AllowRules = AllowRules()) {
        self.model = model
        self.tools = tools
        self.mode = mode
        self.rules = rules
    }
}

/// The agent: ask the model, run the tools it calls (checking permissions, asking the user when the rules say so),
/// send back the results, repeat until it answers without tools. Cancelling the task stops it (a running command
/// is ended).
public enum AgentLoop {
    public static func run(client: any ChatStreaming, messages initial: [ChatMessage], config: AgentConfig,
                           context ctx: ToolContext,
                           ask: @escaping @Sendable (PermissionRequest) async -> PermissionAnswer,
                           rulesChanged: @escaping @Sendable (AllowRules) async -> Void = { _ in },
                           events: @escaping @Sendable (AgentEvent) async -> Void) async throws -> AgentOutcome {
        var messages = initial
        var rules = config.rules
        let byName = Dictionary(config.tools.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        let specs = config.tools.map(\.spec)
        for _ in 0..<config.maxSteps {
            try Task.checkCancellation()
            var request = ChatRequest(model: config.model, messages: messages)
            request.tools = specs
            request.reasoningEffort = config.reasoningEffort
            await events(.assistantStarted)
            var acc = ReplyAccumulator(inlineThinking: config.inlineThinking)
            for try await e in client.stream(request) {
                acc.add(e)
                await events(.delta(e))
            }
            acc.finish()
            let calls = acc.tools.calls
            await events(.assistantFinished(content: acc.content, reasoning: acc.reasoning, toolCalls: calls))
            messages.append(ChatMessage(role: .assistant, parts: [.text(acc.content)], toolCalls: calls))
            if calls.isEmpty { return .finished }

            var images: [String] = []
            for call in calls {
                try Task.checkCancellation()
                var (output, status) = await runCall(call, byName: byName, mode: config.mode, rules: &rules, ctx: ctx,
                                                     ask: ask, rulesChanged: rulesChanged, events: events)
                if !output.images.isEmpty {
                    if config.vision { images += output.images }
                    else { output.text += "\n(\(output.images.count) image(s) left out: this model can't see images.)" }
                }
                await events(.toolFinished(callID: call.id, output: output, status: status))
                messages.append(ChatMessage(role: .tool, parts: [.text(output.text)], toolCallID: call.id))
            }
            // Tool messages are text only in the OpenAI API: images follow as a user message.
            if !images.isEmpty {
                messages.append(ChatMessage(role: .user, parts: [.text("The images from the tool results above, in order:")]
                                            + images.map { .image(url: $0) }))
            }
        }
        return .stepLimit
    }

    static func runCall(_ call: ToolCall, byName: [String: any AgentTool], mode: PermissionMode, rules: inout AllowRules,
                        ctx: ToolContext, ask: @Sendable (PermissionRequest) async -> PermissionAnswer,
                        rulesChanged: @Sendable (AllowRules) async -> Void,
                        events: @Sendable (AgentEvent) async -> Void) async -> (ToolOutput, ToolStatus) {
        guard let tool = byName[call.name] else {
            await events(.toolStarted(call: call, preview: ToolPreview(kind: .read, title: call.name)))
            return (ToolOutput("There is no tool named \(call.name). The tools are: \(byName.keys.sorted().joined(separator: ", ")).",
                               isError: true), .failed)
        }
        let args: ToolArgs
        do { args = try ToolArgs(json: call.arguments) } catch {
            await events(.toolStarted(call: call, preview: ToolPreview(kind: .read, title: call.name)))
            return (ToolOutput("\(error.localizedDescription)", isError: true), .failed)
        }
        let preview: ToolPreview
        do { preview = try tool.preview(args, ctx) } catch {
            await events(.toolStarted(call: call, preview: ToolPreview(kind: .read, title: call.name)))
            return (ToolOutput("\(call.name): \(error.localizedDescription)", isError: true), .failed)
        }
        await events(.toolStarted(call: call, preview: preview))
        let inProject = preview.paths.allSatisfy(ctx.inProject)
        switch PermissionEngine.decide(effect: tool.effect, inProject: inProject, command: preview.command, mode: mode,
                                       rules: rules) {
        case .deny(let why):
            return (ToolOutput(why, isError: true), .denied)
        case .ask:
            let answer = await ask(PermissionRequest(call: call, preview: preview, effect: tool.effect, inProject: inProject))
            if Task.isCancelled { return (ToolOutput("Stopped.", isError: true), .denied) }
            switch answer {
            case .deny:
                return (ToolOutput("The user declined this \(call.name) call. Don't try it again; ask them what they'd like instead.",
                                   isError: true), .denied)
            case .allowAlways:
                if tool.effect == .edit, inProject { rules.edits = true }
                if tool.effect == .run { rules.runs = true }
                if tool.effect == .execute, let cmd = preview.command {
                    let prefix = AllowRules.prefix(of: cmd)
                    if !prefix.isEmpty, !rules.commandPrefixes.contains(prefix) { rules.commandPrefixes.append(prefix) }
                }
                await rulesChanged(rules)
            case .allow:
                break
            }
        case .allow:
            break
        }
        ctx.callID = call.id
        do {
            let out = try await tool.run(args, ctx)
            return (out, out.isError ? .failed : .done)
        } catch is CancellationError {
            return (ToolOutput("Stopped.", isError: true), .failed)
        } catch {
            return (ToolOutput("\(call.name) failed: \(error.localizedDescription)", isError: true), .failed)
        }
    }
}

/// The system prompt for a coding chat: what the agent is, where it works, the project's instructions.
public enum ProjectContext {
    public static func systemPrompt(root: URL, mode: PermissionMode, date: Date = Date()) async -> String {
        let git = await ShellRunner().run("git status --short --branch 2>/dev/null | head -40", cwd: root, timeout: 10)
        var instructions = ""
        for name in ["AGENTS.md", "CLAUDE.md", ".claude/CLAUDE.md"] {
            if let text = try? String(contentsOf: root.appendingPathComponent(name), encoding: .utf8) {
                instructions = "\n\n# Project instructions (\(name))\n\n" + String(text.prefix(20_000))
                break
            }
        }
        let modeLine: String
        switch mode {
        case .plan: modeLine = "You are in plan mode: only read and search. Don't edit files or run commands; finish with a concrete plan."
        case .ask: modeLine = "The user approves each edit and command, and may decline one: then ask what they want instead."
        case .acceptEdits: modeLine = "Edits in the project are applied without asking; the user approves each command."
        case .bypass: modeLine = "Edits and commands run without asking. Be careful: nothing is undone automatically."
        }
        let day = ISO8601DateFormatter.string(from: date, timeZone: .current, formatOptions: [.withFullDate])
        return """
        You are a coding agent working in the user's project on their Mac. Use the tools to read, search, edit \
        and run things; don't guess file contents. Read a file before editing it, and keep edits small and exact. \
        Run the project's build or tests after changes when you can, and report what you verified. When you're \
        done, say briefly what you changed. \(modeLine)

        Project folder: \(root.path)
        Today: \(day)
        Shell: zsh on macOS (no interactive commands; stdin is closed)
        \(git.status == 0 && !git.output.isEmpty ? "Git status:\n\(git.output)" : "Not a git repository.")\(instructions)
        """
    }
}
