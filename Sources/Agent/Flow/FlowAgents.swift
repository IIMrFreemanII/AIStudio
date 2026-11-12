import Foundation
import LLMClient

/// What an edit agent may do without a person: commands only inside the worktree and without risky git, edits in
/// the project, reads in the project, web reads; never `run`, `orchestrate` or `graphEdit`.
public enum FlowPolicy {
    public static func answer(_ r: PermissionRequest) -> PermissionAnswer {
        switch r.effect {
        case .execute: return r.inProject && GitSafety.risk(r.preview.command ?? "") == nil ? .allow : .deny
        case .edit, .read: return r.inProject ? .allow : .deny
        case .network: return .allow
        case .run, .orchestrate, .graphEdit: return .deny
        }
    }
}

/// Runs each flow agent with the AgentLoop: read agents in plan mode on the project, edit agents in their own
/// git worktree and branch.
public struct AgentFlowRunner: FlowAgentRunner {
    let client: any ChatStreaming
    let model: String
    let root: URL
    let flowName: String
    let runID: String
    let codeGraph: Bool
    let reasoningEffort: String?
    let inlineThinking: Bool
    let contextLimit: Int?

    public init(client: any ChatStreaming, model: String, root: URL, flowName: String, runID: String, codeGraph: Bool,
                reasoningEffort: String? = nil, inlineThinking: Bool = false, contextLimit: Int? = nil) {
        self.client = client
        self.model = model
        self.root = root
        self.flowName = flowName
        self.runID = runID
        self.codeGraph = codeGraph
        self.reasoningEffort = reasoningEffort
        self.inlineThinking = inlineThinking
        self.contextLimit = contextLimit
    }

    static func slug(_ s: String) -> String {
        var out = ""
        var lastDash = false
        for ch in s.lowercased() {
            if ch.isASCII, ch.isLetter || ch.isNumber {
                out.append(ch)
                lastDash = false
            } else if !lastDash {
                out.append("-")
                lastDash = true
            }
        }
        let t = String(out.drop { $0 == "-" }.prefix(30))
        return String(t.reversed().drop { $0 == "-" }.reversed())
    }

    /// Text, usage and the last reply, gathered from the loop's events (which arrive from a Sendable closure).
    private final class Collector: @unchecked Sendable {
        private let lock = NSLock()
        private var _text = ""
        private var _usage = FlowUsage()
        func setText(_ s: String) { if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lock.withLock { _text = s } } }
        func resetText() { lock.withLock { _text = "" } }
        var text: String { lock.withLock { _text } }
        var usage: FlowUsage { lock.withLock { _usage } }
        func update(_ f: (inout FlowUsage) -> Void) -> FlowUsage { lock.withLock { f(&_usage); return _usage } }
    }

    public func run(_ spec: FlowAgentSpec, events: @escaping @Sendable (FlowAgentEvent) -> Void) async throws -> FlowAgentResult {
        var tools: [any AgentTool]
        var config: AgentConfig
        var system: String
        var workdir = root
        var worktree: (path: URL, branch: String, base: String, repo: URL)?
        let ask: @Sendable (PermissionRequest) async -> PermissionAnswer

        switch spec.mode {
        case .read:
            tools = TaskTool.defaultTools + (codeGraph ? [CodeGraphTool()] : []) + [WebFetchTool()]
            config = AgentConfig(model: model, tools: tools, mode: .plan)
            config.maxSteps = spec.maxSteps ?? 30
            system = "You are one agent in a multi-agent flow (\(flowName)), working in the project at \(root.path). Do only the task below, with the read-only tools; you can't edit files, run commands or ask anyone. Finish with a concise report, citing file paths and line numbers."
            if codeGraph { system += "\nUse code_graph first to locate code (it returns file:line), then read only those lines." }
            ask = { _ in .deny }
        case .edit:
            guard let repo = await GitWorktree.repoRoot(root) else { throw ToolError("edit agents need a git repository") }
            let base = await GitWorktree.currentBranch(repo) ?? "HEAD"
            let made = try await GitWorktree.create(repo: repo, branch: "aistudio/flow-\(runID)-\(Self.slug(spec.label))", base: base)
            worktree = (made.path, made.branch, base, repo)
            workdir = made.path
            tools = CodingTools.all + (codeGraph ? [CodeGraphTool()] : [])
            config = AgentConfig(model: model, tools: tools, mode: .acceptEdits)
            config.maxSteps = spec.maxSteps ?? 60
            system = "You are one agent in a multi-agent flow (\(flowName)), working in a git worktree at \(made.path.path) on branch \(made.branch). Make the change the task asks for, run what checks it needs, and commit your work on this branch with a short message. You can't push, ask anyone, or start other agents. Finish with a concise report of what you changed."
            ask = { FlowPolicy.answer($0) }
        }
        config.reasoningEffort = spec.effort ?? reasoningEffort
        config.inlineThinking = inlineThinking
        config.contextLimit = contextLimit

        var user = spec.prompt
        if let schema = spec.schema {
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            let text = (try? enc.encode(schema)).map { String(decoding: $0, as: UTF8.self) } ?? "{}"
            user += "\n\nReply with only JSON matching this schema (no other text):\n" + text
        }
        let base: [ChatMessage] = [.system(system), .user(user)]
        let collector = Collector()
        let client = self.client
        let ctx = ToolContext(root: workdir)

        func loop(_ messages: [ChatMessage], _ config: AgentConfig) async throws -> AgentOutcome {
            try await AgentLoop.run(client: client, messages: messages, config: config, context: ctx, ask: ask, events: { e in
                switch e {
                case .assistantFinished(let content, _, _):
                    collector.setText(content)
                case .toolStarted(_, let preview):
                    events(.step(preview.title))
                case .delta(.usage(let p, let c)):
                    events(.usage(collector.update { $0.promptTokens += p; $0.completionTokens += c }))
                case .delta(.stats(let s)):
                    events(.usage(collector.update {
                        $0.seconds += s.seconds
                        $0.costUSD += s.costUSD ?? 0
                        if let r = s.usdPerSecond { $0.usdPerSecond = r }
                    }))
                default: break
                }
            })
        }

        let outcome = try await loop(base, config)
        var text = collector.text
        var json: JSONValue?
        if let schema = spec.schema {
            func problem(_ t: String) -> (JSONValue?, String?) {
                guard let v = FlowJSON.extract(t) else { return (nil, "no JSON found") }
                if let p = FlowJSON.validate(v, schema: schema) { return (nil, p) }
                return (v, nil)
            }
            var (value, issue) = problem(text)
            if let first = issue {
                var retry = config
                retry.maxSteps = 3
                collector.resetText()
                _ = try await loop(base + [.assistant(text),
                                           .user("That reply wasn't valid JSON for the schema: \(first). Reply with only the JSON.")], retry)
                text = collector.text
                (value, issue) = problem(text)
            }
            if let issue { throw ToolError("the reply didn't match the schema: \(issue)") }
            json = value
        }
        if outcome == .stepLimit { text += "\n(The agent stopped after \(config.maxSteps) steps.)" }

        var branch: String?
        if let w = worktree {
            if let st = await Git.status(w.path), !st.files.isEmpty {
                _ = try? await Git.commit(w.path, message: "Flow \(flowName): \(spec.label)")
            }
            if await GitWorktree.unmergedCommits(branch: w.branch, base: w.base, repo: w.repo) > 0 {
                branch = w.branch
            } else {
                text += "\n(No changes were made.)"
            }
        }
        return FlowAgentResult(text: text, json: json, branch: branch, usage: collector.usage)
    }
}
