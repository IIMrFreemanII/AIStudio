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
    /// The conversation so far was replaced by this summary (the context was filling up): later requests start from
    /// it.
    case compacted(summary: String)
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
    /// The model's context in tokens: when set, the loop compacts the conversation as it nears the limit.
    public var contextLimit: Int?
    /// The share of `contextLimit` that triggers compaction.
    public var compactAt = 0.75

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
                           modeChanged: @escaping @Sendable (PermissionMode) async -> Void = { _ in },
                           events: @escaping @Sendable (AgentEvent) async -> Void) async throws -> AgentOutcome {
        var messages = initial
        var rules = config.rules
        let byName = Dictionary(config.tools.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        ctx.mode = config.mode
        for _ in 0..<config.maxSteps {
            try Task.checkCancellation()
            var request = ChatRequest(model: config.model, messages: messages)
            request.tools = config.tools.filter { $0.name != "exit_plan" || ctx.mode == .plan }.map(\.spec)
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

            let appendedFrom = messages.count
            var images: [String] = []
            var results = [(output: ToolOutput, status: ToolStatus)?](repeating: nil, count: calls.count)
            var i = 0
            while i < calls.count {
                try Task.checkCancellation()
                var j = i
                if calls[i].name == "task" { while j < calls.count, calls[j].name == "task" { j += 1 } }
                if j - i >= 2 {
                    let group = Array(i..<j)
                    let modeBefore = ctx.mode
                    let rulesNow = rules
                    await withTaskGroup(of: (Int, ToolOutput, ToolStatus).self) { g in
                        for k in group {
                            let call = calls[k]
                            g.addTask {
                                var own = rulesNow
                                let (o, st) = await ToolCallScope.$id.withValue(call.id) {
                                    await runCall(call, byName: byName, rules: &own, ctx: ctx, ask: ask,
                                                  rulesChanged: rulesChanged, events: events)
                                }
                                return (k, o, st)
                            }
                        }
                        for await (k, o, st) in g { results[k] = (o, st) }
                    }
                    if ctx.mode != modeBefore { await modeChanged(ctx.mode) }
                    i = j
                } else {
                    let modeBefore = ctx.mode
                    let call = calls[i]
                    let (o, st) = await ToolCallScope.$id.withValue(call.id) {
                        await runCall(call, byName: byName, rules: &rules, ctx: ctx, ask: ask,
                                      rulesChanged: rulesChanged, events: events)
                    }
                    results[i] = (o, st)
                    if ctx.mode != modeBefore { await modeChanged(ctx.mode) }
                    i += 1
                }
            }
            for (k, call) in calls.enumerated() {
                guard var (output, status) = results[k] else { continue }
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

            if let limit = config.contextLimit {
                let used = (acc.promptTokens ?? Compactor.estimate(Array(messages[..<(appendedFrom - 1)])))
                    + (acc.completionTokens ?? 0) + Compactor.estimate(Array(messages[appendedFrom...]))
                let threshold = config.compactAt * Double(limit)
                if Double(used) > threshold {
                    let before = Compactor.estimate(messages)
                    messages = Compactor.trim(messages)
                    // The server's token count is the better measure; trimming only takes off what it saved.
                    let usedAfterTrim = used - max(0, before - Compactor.estimate(messages))
                    if Double(usedAfterTrim) > threshold {
                        do {
                            let summary = try await Compactor.summarize(client: client, model: config.model, messages: messages,
                                                                        limit: limit)
                            var fresh: [ChatMessage] = []
                            if let first = messages.first, first.role == .system { fresh.append(first) }
                            fresh.append(.user(ChatTurn.summaryPrefix + summary + "\n\nContinue the task from where it left off."))
                            messages = fresh
                            await events(.compacted(summary: summary))
                        } catch is CancellationError {
                            throw CancellationError()
                        } catch {
                            // Carry on with the trimmed messages.
                        }
                    }
                }
            }
        }
        return .stepLimit
    }

    static func runCall(_ call: ToolCall, byName: [String: any AgentTool], rules: inout AllowRules,
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
        // Reads may also go to the skills' folders without asking.
        let inProject = preview.paths.allSatisfy(tool.effect == .read ? ctx.canRead : ctx.inProject)
        switch PermissionEngine.decide(effect: tool.effect, inProject: inProject, command: preview.command,
                                       domain: preview.domain, mode: ctx.mode, rules: rules) {
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
                if tool.effect == .network, let d = preview.domain?.lowercased(), !d.isEmpty, !rules.domains.contains(d) {
                    rules.domains.append(d)
                }
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
        case .plan: modeLine = "You are in plan mode: only read and search (web_fetch is allowed). Don't edit files or run commands. When the plan is ready, call exit_plan with it: the user approves it, or asks for changes."
        case .ask: modeLine = "The user approves each edit and command, and may decline one: then ask what they want instead."
        case .acceptEdits: modeLine = "Edits in the project are applied without asking; the user approves each command."
        case .bypass: modeLine = "Edits and commands run without asking. Edits can be rewound from the chat; what commands do can't."
        }
        let day = ISO8601DateFormatter.string(from: date, timeZone: .current, formatOptions: [.withFullDate])
        return """
        You are a coding agent working in the user's project on their Mac. Use the tools to read, search, edit \
        and run things; don't guess file contents. Read a file before editing it, and keep edits small and exact. \
        Run the project's build or tests after changes when you can, and report what you verified. When you're \
        done, say briefly what you changed. \(modeLine)

        Sessions: for an independent task that should run in parallel, start_session starts a separate session \
        (its own worktree by default) that reports back here; propose_session only suggests one. Pieces of your \
        current task stay here.

        Project folder: \(root.path)
        Today: \(day)
        Shell: zsh on macOS (no interactive commands; stdin is closed)
        \(git.status == 0 && !git.output.isEmpty ? "Git status:\n\(git.output)" : "Not a git repository.")\(instructions)
        """
    }
}

/// Keeping a long conversation inside the model's context: estimate its size, drop old tool output, summarize.
public enum Compactor {
    /// Rough tokens: characters of text and tool-call arguments / 4, plus 1,000 per image or video.
    public static func estimate(_ messages: [ChatMessage]) -> Int {
        var chars = 0
        var media = 0
        for m in messages {
            for p in m.parts {
                switch p {
                case .text(let t): chars += t.count
                case .image, .video: media += 1
                }
            }
            for c in m.toolCalls { chars += c.arguments.count }
        }
        return chars / 4 + media * 1_000
    }

    static let removed = "[output removed to save context]"

    /// Old tool output (all but the last `keepLast` tool messages) over 200 characters is dropped.
    public static func trim(_ messages: [ChatMessage], keepLast: Int = 6) -> [ChatMessage] {
        let toolIndexes = messages.indices.filter { messages[$0].role == .tool }
        let old = Set(toolIndexes.dropLast(keepLast))
        var out = messages
        for i in old where out[i].text.count > 200 { out[i].parts = [.text(removed)] }
        return out
    }

    static let summarySystem = "You compact a conversation between a user and a coding agent so the agent can continue with less context. Write a summary that keeps everything needed to continue: the user's requests and preferences, decisions made, files read and changed (paths and what changed), commands run and their results, errors, and the exact current state: what was being done and what comes next. Use short headed sections and bullets. Don't add anything new."

    public static func summarize(client: any ChatStreaming, model: String, messages: [ChatMessage],
                                 limit: Int?) async throws -> String {
        var parts: [String] = []
        for m in messages {
            switch m.role {
            case .system: continue
            case .user: parts.append("## User\n" + clip(m.text, max: 6_000))
            case .assistant:
                var s = "## Assistant\n" + clip(m.text, max: 6_000)
                for c in m.toolCalls { s += "\nCalls: \(c.name)(\(clip(c.arguments, max: 500)))" }
                parts.append(s)
            case .tool: parts.append("## Tool result\n" + clip(m.text, max: 2_000))
            }
        }
        let render = clip(parts.joined(separator: "\n\n"), max: max((limit ?? 32_768) * 2, 20_000))
        let request = ChatRequest(model: model, messages: [.system(summarySystem), .user(render + "\n\nWrite the summary now.")])
        var acc = ReplyAccumulator(inlineThinking: true)
        for try await e in client.stream(request) { acc.add(e) }
        acc.finish()
        let summary = acc.content.trimmingCharacters(in: .whitespacesAndNewlines)
        if summary.isEmpty { throw ToolError("the summary came back empty") }
        return summary
    }
}
