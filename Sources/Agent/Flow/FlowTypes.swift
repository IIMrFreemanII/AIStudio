import CryptoKit
import Foundation
import LLMClient

// Flows: a JavaScript script (Claude Code's workflow format) that starts many agents. The runtime (FlowRuntime) runs
// the script; a FlowAgentRunner runs each agent; the app (a FlowHost) shows the run and delivers its result.

public enum FlowAgentMode: String, Codable, Sendable {
    /// Reads and searches the project (plan mode); the default.
    case read
    /// Edits and runs commands in its own git worktree and branch.
    case edit
}

/// One `agent(prompt, opts)` call.
public struct FlowAgentSpec: Codable, Sendable, Equatable {
    /// "a1", "a2", … in the order the script called agent().
    public var id: String
    public var label: String
    public var phase: String?
    public var prompt: String
    /// A JSON Schema subset (type, properties, required, items, enum): the agent's reply is parsed into it.
    public var schema: JSONValue?
    public var mode: FlowAgentMode
    public var maxSteps: Int?
    /// Thinking effort for this agent ("off"/"on", or "low"/"medium"/"high"; nil: the chat's). The server maps it to
    /// the model's own control.
    public var effort: String?
    /// What resume matches finished agents on: the prompt and the options that change the result.
    public var cacheKey: String

    public init(id: String, label: String, phase: String? = nil, prompt: String, schema: JSONValue? = nil,
                mode: FlowAgentMode = .read, maxSteps: Int? = nil, effort: String? = nil) {
        self.id = id
        self.label = label
        self.phase = phase
        self.prompt = prompt
        self.schema = schema
        self.mode = mode
        self.maxSteps = maxSteps
        self.effort = effort
        self.cacheKey = Self.cacheKey(prompt: prompt, schema: schema, mode: mode, maxSteps: maxSteps, effort: effort)
    }

    /// SHA-256 (hex) of the prompt, the schema as sorted-key JSON, the mode, maxSteps and effort.
    public static func cacheKey(prompt: String, schema: JSONValue?, mode: FlowAgentMode, maxSteps: Int?,
                                effort: String? = nil) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys]
        let schemaText = schema.flatMap { try? enc.encode($0) }.map { String(decoding: $0, as: UTF8.self) } ?? "null"
        var parts = [prompt, schemaText, mode.rawValue, maxSteps.map(String.init) ?? "-"]
        if let effort { parts.append(effort) }  // absent: the same key as before efforts existed
        let text = parts.joined(separator: "\u{1F}")
        return SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }
}

/// Tokens and time one agent used (summed over its requests).
public struct FlowUsage: Codable, Sendable, Equatable {
    public var promptTokens = 0
    public var completionTokens = 0
    /// Request wall time as the server reports it (parallel requests overlap: don't sum across agents for cost).
    public var seconds = 0.0
    /// What the server billed for these requests (overcounts when requests ran in parallel).
    public var costUSD = 0.0
    public var usdPerSecond: Double?

    public init(promptTokens: Int = 0, completionTokens: Int = 0, seconds: Double = 0, costUSD: Double = 0,
                usdPerSecond: Double? = nil) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.seconds = seconds
        self.costUSD = costUSD
        self.usdPerSecond = usdPerSecond
    }

    public static func + (a: FlowUsage, b: FlowUsage) -> FlowUsage {
        FlowUsage(promptTokens: a.promptTokens + b.promptTokens, completionTokens: a.completionTokens + b.completionTokens,
                  seconds: a.seconds + b.seconds, costUSD: a.costUSD + b.costUSD, usdPerSecond: b.usdPerSecond ?? a.usdPerSecond)
    }
}

public struct FlowAgentResult: Codable, Sendable, Equatable {
    /// The agent's final reply.
    public var text: String
    /// The reply parsed against the spec's schema (nil without a schema).
    public var json: JSONValue?
    /// An edit agent's branch (nil for read agents, or when it changed nothing).
    public var branch: String?
    public var usage: FlowUsage
    /// Reused from an earlier run (resume): the runner wasn't called.
    public var cached = false

    public init(text: String, json: JSONValue? = nil, branch: String? = nil, usage: FlowUsage = FlowUsage(), cached: Bool = false) {
        self.text = text
        self.json = json
        self.branch = branch
        self.usage = usage
        self.cached = cached
    }
}

/// Progress from a running agent.
public enum FlowAgentEvent: Sendable, Equatable {
    /// A tool call started: its card title ("Read Sources/App.swift").
    case step(String)
    /// The agent's usage so far (a running total, not a delta).
    case usage(FlowUsage)
}

/// Runs one agent. The runtime only knows this protocol (tests inject a fake).
public protocol FlowAgentRunner: Sendable {
    /// Runs the agent to its final reply. Throws on failure; cancelling the task stops it.
    func run(_ spec: FlowAgentSpec, events: @escaping @Sendable (FlowAgentEvent) -> Void) async throws -> FlowAgentResult
}

/// The script's `export const meta = { name, description, phases }`.
public struct FlowMeta: Codable, Sendable, Equatable {
    public var name: String
    public var description: String
    /// Phase titles, in order.
    public var phases: [String]

    public init(name: String, description: String, phases: [String] = []) {
        self.name = name
        self.description = description
        self.phases = phases
    }
}

public enum FlowStatus: String, Codable, Sendable { case running, done, failed, stopped }

public enum FlowAgentStatus: String, Codable, Sendable { case running, done, failed, cached, stopped }

/// What the runtime reports while a script runs.
public enum FlowRunEvent: Sendable {
    /// `phase(title)` was called, or an agent with a new `phase` option started.
    case phase(String)
    case log(String)
    case agentStarted(FlowAgentSpec)
    case agentEvent(id: String, FlowAgentEvent)
    case agentFinished(id: String, FlowAgentResult)
    case agentFailed(id: String, error: String)
}

/// One agent of a run, as the app shows and saves it.
public struct FlowAgentRecord: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var label: String
    public var phase: String?
    public var mode: FlowAgentMode
    public var status: FlowAgentStatus
    /// Step titles, oldest first (the last 50).
    public var steps: [String] = []
    public var text: String?
    public var json: JSONValue?
    public var branch: String?
    public var error: String?
    public var usage = FlowUsage()
    public var startedAt: Date
    public var endedAt: Date?
    public var cacheKey: String

    public init(spec: FlowAgentSpec, startedAt: Date = Date()) {
        id = spec.id
        label = spec.label
        phase = spec.phase
        mode = spec.mode
        status = .running
        self.startedAt = startedAt
        cacheKey = spec.cacheKey
    }
}

/// A flow run: saved to `<Application Support>/AIStudio/flow-runs/<id>.json`.
public struct FlowRun: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var chatID: UUID?
    public var meta: FlowMeta
    public var script: String
    public var args: JSONValue?
    public var status: FlowStatus = .running
    public var startedAt = Date()
    public var endedAt: Date?
    public var currentPhase: String?
    public var agents: [FlowAgentRecord] = []
    public var logs: [String] = []
    /// The script's return value as JSON text.
    public var result: String?
    public var error: String?
    /// The run whose finished agents this one reused.
    public var resumedFrom: String?
    public var maxAgents: Int
    public var concurrency: Int

    public init(id: String = String(UUID().uuidString.prefix(8)).lowercased(), chatID: UUID?, meta: FlowMeta, script: String,
                args: JSONValue? = nil, maxAgents: Int, concurrency: Int, resumedFrom: String? = nil) {
        self.id = id
        self.chatID = chatID
        self.meta = meta
        self.script = script
        self.args = args
        self.maxAgents = maxAgents
        self.concurrency = concurrency
        self.resumedFrom = resumedFrom
    }

    /// Finished agents' results by cache key, for resuming from this run (edit agents only if `branchExists`).
    public func cache(branchExists: (String) -> Bool) -> [String: FlowAgentResult] {
        var out: [String: FlowAgentResult] = [:]
        for a in agents where a.status == .done || a.status == .cached {
            if a.mode == .edit, let b = a.branch, !branchExists(b) { continue }
            out[a.cacheKey] = FlowAgentResult(text: a.text ?? "", json: a.json, branch: a.branch, usage: a.usage)
        }
        return out
    }

    /// Applies a runtime event (the app's live record and the saved one both use this).
    public mutating func apply(_ event: FlowRunEvent, now: Date = Date()) {
        switch event {
        case .phase(let p):
            currentPhase = p
        case .log(let s):
            logs.append(s)
            if logs.count > 200 { logs.removeFirst(logs.count - 200) }
        case .agentStarted(let spec):
            agents.append(FlowAgentRecord(spec: spec, startedAt: now))
            if let p = spec.phase { currentPhase = p }
        case .agentEvent(let id, let e):
            guard let i = agents.firstIndex(where: { $0.id == id }) else { return }
            switch e {
            case .step(let s):
                agents[i].steps.append(s)
                if agents[i].steps.count > 50 { agents[i].steps.removeFirst() }
            case .usage(let u):
                agents[i].usage = u
            }
        case .agentFinished(let id, let r):
            guard let i = agents.firstIndex(where: { $0.id == id }) else { return }
            agents[i].status = r.cached ? .cached : .done
            agents[i].text = r.text
            agents[i].json = r.json
            agents[i].branch = r.branch
            agents[i].usage = r.usage
            agents[i].endedAt = now
        case .agentFailed(let id, let error):
            guard let i = agents.firstIndex(where: { $0.id == id }) else { return }
            agents[i].status = error == FlowErrors.stopped ? .stopped : .failed
            agents[i].error = error
            agents[i].endedAt = now
        }
    }
}

public enum FlowErrors {
    /// The error text of agents cancelled by Stop.
    public static let stopped = "stopped"
}

/// What a chat provides to the `flow` tool: starts runs in the background.
public protocol FlowHost: Sendable {
    /// Starts a run and returns its id at once; the result reaches the chat as an event when the run ends.
    func start(script: String, meta: FlowMeta, args: JSONValue?, callID: String) async throws -> String
    /// The defaults the approval card shows: agents at a time, and the most agents one run may start.
    func limits() async -> (concurrency: Int, maxAgents: Int)
}
