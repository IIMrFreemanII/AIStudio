import AIStudioProtocol
import Agent
import AppKit
import Foundation
import LLMClient
import Observation

// Flows in the app: a FlowController runs one script for a chat (its live record is what the card and the panel
// show), the model starts them and delivers their results to the chat, and AppFlowHost is what the `flow` tool calls.

extension FlowRun {
    /// The last price per second any agent's usage reported.
    var usdPerSecond: Double? { agents.reversed().lazy.compactMap { $0.usage.usdPerSecond }.first }

    /// Run time × the GPU's price (parallel agents share the GPU), nil while no price is known.
    func estimatedCost(now: Date = Date()) -> Double? {
        guard let rate = usdPerSecond else { return nil }
        return max(0, (endedAt ?? now).timeIntervalSince(startedAt)) * rate
    }

    var failedAgents: Int { agents.filter { $0.status == .failed }.count }
    var doneAgents: Int { agents.filter { $0.status == .done || $0.status == .cached }.count }
    /// Branches of edit agents that changed something, in order.
    var branches: [String] {
        var seen = Set<String>()
        return agents.compactMap(\.branch).filter { seen.insert($0).inserted }
    }
    var isFinished: Bool { status != .running }
}

/// One flow run: the live record, and the runtime that fills it.
@MainActor @Observable
final class FlowController: Identifiable {
    var run: FlowRun
    let chatID: UUID
    /// The agent row open in the panel.
    var expandedAgent: String?
    /// Loaded from disk: it can be looked at, not stopped.
    let readOnly: Bool
    @ObservationIgnored private(set) var runtime: FlowRuntime?
    @ObservationIgnored private weak var model: AppModel?
    @ObservationIgnored private var launch: Launch?
    @ObservationIgnored private var session: URLSession?

    nonisolated var id: String { runID }
    nonisolated let runID: String

    struct Launch {
        var runner: any FlowAgentRunner
        var cache: [String: FlowAgentResult]
        var session: URLSession?
    }

    init(run: FlowRun, chatID: UUID, readOnly: Bool = false) {
        self.run = run
        self.chatID = chatID
        self.readOnly = readOnly
        runID = run.id
    }

    func prepare(model: AppModel, runner: any FlowAgentRunner, cache: [String: FlowAgentResult], session: URLSession?) {
        self.model = model
        launch = Launch(runner: runner, cache: cache, session: session)
    }

    /// Batches the runtime's events (they arrive on its queue, many per second) into one main-actor hop per 100 ms.
    private final class EventBuffer: @unchecked Sendable {
        private let lock = NSLock()
        private var events: [FlowRunEvent] = []
        private var scheduled = false
        private let drain: @Sendable ([FlowRunEvent]) -> Void

        init(drain: @escaping @Sendable ([FlowRunEvent]) -> Void) { self.drain = drain }

        func push(_ e: FlowRunEvent) {
            lock.lock()
            events.append(e)
            let first = !scheduled
            scheduled = true
            lock.unlock()
            if first {
                Task.detached { [self] in
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    flush()
                }
            }
        }

        func take() -> [FlowRunEvent] {
            lock.lock()
            defer { lock.unlock() }
            let out = events
            events = []
            scheduled = false
            return out
        }

        func flush() {
            let out = take()
            if !out.isEmpty { drain(out) }
        }
    }

    func start() {
        guard let launch, runtime == nil, !readOnly else { return }
        session = launch.session
        let buffer = EventBuffer { [weak self] events in
            Task { @MainActor in self?.apply(events) }
        }
        let rt = FlowRuntime(script: run.script, args: run.args, runner: launch.runner, concurrency: run.concurrency,
                             maxAgents: run.maxAgents, cache: launch.cache, onEvent: { buffer.push($0) })
        runtime = rt
        self.launch = nil
        FlowRegistry.save(run)
        Task { @MainActor in
            let outcome = await rt.run()
            apply(buffer.take())
            finish(outcome)
        }
    }

    func stop() { runtime?.stop() }

    private func apply(_ events: [FlowRunEvent]) {
        for e in events { run.apply(e) }
    }

    private func finish(_ o: (status: FlowStatus, result: String?, error: String?)) {
        run.status = o.status
        run.result = o.result
        run.error = o.error
        run.endedAt = Date()
        for i in run.agents.indices where run.agents[i].status == .running {
            run.agents[i].status = o.status == .stopped ? .stopped : .failed
            run.agents[i].endedAt = run.endedAt
        }
        session?.finishTasksAndInvalidate()
        session = nil
        FlowRegistry.save(run)
        model?.flowFinished(self)
    }
}

/// Runs by id: the ones started in this session, and finished ones read back from disk.
@MainActor
enum FlowRegistry {
    static var controllers: [String: FlowController] = [:]

    nonisolated static var folder: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("AIStudio/flow-runs", isDirectory: true)
    }

    nonisolated static func save(_ run: FlowRun) {
        let url = folder.appendingPathComponent(run.id + ".json")
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        if let data = try? enc.encode(run) { try? data.write(to: url, options: .atomic) }
    }

    static func load(_ id: String) -> FlowRun? {
        guard !id.isEmpty, !id.contains("/"),
              let data = try? Data(contentsOf: folder.appendingPathComponent(id + ".json")) else { return nil }
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        return try? dec.decode(FlowRun.self, from: data)
    }
}

extension AppModel {
    // MARK: limits and client

    /// Agents at a time and the most agents one run may start, from Settings (0: sized to the GPU's KV cache).
    func flowLimits(for doc: ChatDocument) -> (concurrency: Int, maxAgents: Int) {
        let d = UserDefaults.standard
        var concurrency = d.integer(forKey: "flowConcurrency")
        if concurrency <= 0 {
            if let kv = doc.modelKey.flatMap({ llm.status($0)?.kvTokens }), kv > 0 {
                concurrency = min(16, max(2, kv / 24_000))
            } else {
                concurrency = 4
            }
        }
        let maxAgents = d.integer(forKey: "flowMaxAgents")
        return (concurrency, maxAgents > 0 ? maxAgents : 30)
    }

    /// A client for one run: its own URLSession, so many agents don't queue behind the app's other requests.
    private func flowClient(_ serverID: String, concurrency: Int) -> (client: LLMServerClient, session: URLSession)? {
        guard let c = llm.connections.connection(serverID), c.kind == .gpu, let url = c.url else { return nil }
        let cfg = URLSessionConfiguration.default
        cfg.httpMaximumConnectionsPerHost = max(6, concurrency + 2)
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 24 * 3600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        let session = URLSession(configuration: cfg)
        let http = HTTPService(baseURL: url, token: ConnectionToken.read(serverID) ?? "", session: session)
        return (LLMServerClient(http: http), session)
    }

    private static func branchExists(_ branch: String, in root: URL) -> Bool {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        p.arguments = ["rev-parse", "--verify", "--quiet", "refs/heads/" + branch]
        p.currentDirectoryURL = root
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return false }
        p.waitUntilExit()
        return p.terminationStatus == 0
    }

    // MARK: starting

    /// Starts a run for the chat. With `resumeFrom`, the agents that finished in that run (and whose branches still
    /// exist) aren't run again.
    @discardableResult
    func startFlow(script: String, meta: FlowMeta, args: JSONValue?, chatID: UUID, resumeFrom: String? = nil) throws -> FlowController {
        guard let doc = chats.first(where: { $0.id == chatID }) ?? loadSession(chatID), let root = doc.projectURL else {
            throw ToolError("Flows run in coding chats.")
        }
        guard let key = doc.modelKey else { throw ToolError("Choose a model for this chat.") }
        let limits = flowLimits(for: doc)
        guard let made = flowClient(key.serverID, concurrency: limits.concurrency) else {
            throw ToolError(llm.servers.isEmpty ? "Add a GPU server in Servers first." : "The chat's GPU server isn't available.")
        }
        var cache: [String: FlowAgentResult] = [:]
        if let resumeFrom, let old = flowController(resumeFrom)?.run {
            cache = old.cache(branchExists: { Self.branchExists($0, in: root) })
        }
        let run = FlowRun(chatID: chatID, meta: meta, script: script, args: args, maxAgents: limits.maxAgents,
                          concurrency: limits.concurrency, resumedFrom: resumeFrom)
        let runner = AgentFlowRunner(client: made.client.chat, model: key.modelID, root: root, flowName: meta.name, runID: run.id,
                                     codeGraph: codeGraphEnabled, reasoningEffort: doc.transcript.reasoningEffort,
                                     inlineThinking: llm.model(key)?.reasoningParser == nil, contextLimit: contextLimit(for: key))
        let controller = FlowController(run: run, chatID: chatID)
        controller.prepare(model: self, runner: runner, cache: cache, session: made.session)
        FlowRegistry.controllers[run.id] = controller
        controller.start()
        return controller
    }

    /// The live controller of a run, or a read-only one loaded from disk.
    func flowController(_ runID: String) -> FlowController? {
        if let c = FlowRegistry.controllers[runID] { return c }
        guard let run = FlowRegistry.load(runID) else { return nil }
        let c = FlowController(run: run, chatID: run.chatID ?? UUID(), readOnly: true)
        FlowRegistry.controllers[runID] = c
        return c
    }

    // MARK: delivering the result

    /// Called when a run ends: tells the chat (queued while it is streaming).
    func flowFinished(_ c: FlowController) {
        let run = c.run
        let text = run.result ?? run.error ?? (run.status == .stopped ? "Stopped." : "No result.")
        let event = FlowEvent(runID: run.id, name: run.meta.name, status: run.status.rawValue, summary: String(text.prefix(4000)),
                              agents: run.agents.count, failed: run.failedAgents, branches: run.branches)
        guard let doc = loadSession(c.chatID) else { return }
        if doc.isStreaming || doc.compacting {
            doc.pendingFlowEvents.append(event)
        } else {
            deliverFlow([event], to: doc)
        }
    }

    /// Adds the results to the chat (one turn each) and, when the agent spoke last, lets it carry on (at most 5 times
    /// in a row without the user).
    func deliverFlow(_ events: [FlowEvent], to doc: ChatDocument) {
        let lastTurn = doc.transcript.turns.last
        let agentSpokeLast = lastTurn.map { ($0.role == .assistant || $0.role == .tool) && $0.error == nil } ?? false
        for e in events {
            var t = ChatTurn(role: .user, text: "")
            t.flowEvent = e
            doc.transcript.append(t)
        }
        doc.transcript.updated = Date()
        saveChat(doc)
        if agentSpokeLast, doc.autoContinues < 5 {
            doc.autoContinues += 1
            reply(doc)
        } else {
            sessionChanged(doc)
            unloadIfHidden(doc)
        }
    }

    // MARK: branches, worktrees, saving

    /// Asks, then merges an edit agent's branch into the chat's project (its checked-out branch).
    func mergeFlowBranch(_ branch: String, chat doc: ChatDocument) {
        guard let project = doc.projectURL else { return }
        presenter.alert("Merge \(branch)?", message: "It merges into the branch checked out in \(project.lastPathComponent).", actions: [
            MGAlertAction(label: "Merge") { [self] in
                Task { @MainActor in
                    do {
                        let repo = await GitWorktree.repoRoot(project) ?? project
                        let out = try await GitWorktree.merge(branch: branch, into: repo)
                        presenter.alert("Merged \(branch)", message: String(out.prefix(1000)), actions: [.cancel("OK")])
                    } catch {
                        presenter.alert("Couldn't merge", message: error.localizedDescription, actions: [.cancel("OK")])
                    }
                    refreshGitSoon(doc)
                }
            },
            .cancel(),
        ])
    }

    /// Removes the worktrees of a run's edit agents (their branches stay).
    func removeFlowWorktrees(_ run: FlowRun, chat doc: ChatDocument) {
        guard let project = doc.projectURL else { return }
        let branches = run.branches
        guard !branches.isEmpty else { return }
        presenter.alert("Remove the worktrees of this flow?",
                        message: "Uncommitted changes in them are lost. Their branches (\(branches.joined(separator: ", "))) are kept.",
                        actions: [
            .destructive("Remove Worktrees") { [self] in
                Task { @MainActor in
                    guard let repo = await GitWorktree.repoRoot(project) else { return }
                    let list = await ShellRunner().run("git worktree list --porcelain", cwd: repo, timeout: 20)
                    var path: String?
                    var found: [String: String] = [:]
                    for line in list.output.split(separator: "\n", omittingEmptySubsequences: false) {
                        if line.hasPrefix("worktree ") { path = String(line.dropFirst(9)) }
                        else if line.hasPrefix("branch refs/heads/"), let path { found[String(line.dropFirst(18))] = path }
                    }
                    var removed = 0
                    var problems: [String] = []
                    for b in branches {
                        guard let p = found[b] else { continue }
                        do {
                            try await GitWorktree.remove(URL(fileURLWithPath: p), repo: repo, branch: b, force: true, deleteBranch: false)
                            removed += 1
                        } catch {
                            problems.append("\(b): \(error.localizedDescription)")
                        }
                    }
                    if !problems.isEmpty {
                        presenter.alert("Couldn't remove all worktrees", message: problems.joined(separator: "\n"), actions: [.cancel("OK")])
                    } else if removed == 0 {
                        presenter.alert("No worktrees left", message: "They were already removed.", actions: [.cancel("OK")])
                    }
                    refreshGitSoon(doc)
                }
            },
            .cancel(),
        ])
    }

    /// Saves the run's script as a flow, in the project (`.aistudio/flows`) or the personal folder.
    func saveFlow(_ run: FlowRun, personal: Bool, chat doc: ChatDocument) {
        var name = ""
        var lastDash = true
        for ch in run.meta.name.lowercased() {
            if ch.isASCII, ch.isLetter || ch.isNumber { name.append(ch); lastDash = false }
            else if !lastDash { name.append("-"); lastDash = true }
        }
        while name.hasSuffix("-") { name.removeLast() }
        if name.isEmpty { name = "flow-" + run.id }
        do {
            let url = try FlowStore.save(name, script: run.script, root: doc.projectURL, personal: personal)
            presenter.alert("Flow saved", message: "\(url.path)\n\nThe agent knows it by the name “\(name)”.", actions: [.cancel("OK")])
        } catch {
            presenter.alert("Couldn't save the flow", message: error.localizedDescription, actions: [.cancel("OK")])
        }
    }

    // MARK: running a saved flow from the menu

    /// Confirms, then runs a saved flow and opens the panel.
    func runSavedFlow(_ saved: SavedFlow, in doc: ChatDocument) {
        guard let script = try? String(contentsOf: saved.url, encoding: .utf8) else { return }
        let limits = flowLimits(for: doc)
        presenter.alert("Run flow “\(saved.meta.name)”?",
                        message: saved.meta.description + "\nUp to \(limits.maxAgents) agents, \(limits.concurrency) at a time.",
                        actions: [
            MGAlertAction(label: "Run") { [self] in
                do {
                    let c = try startFlow(script: script, meta: saved.meta, args: nil, chatID: doc.id)
                    openFlowPanel(c.run.id, in: doc)
                } catch {
                    presenter.alert("Couldn't start the flow", message: error.localizedDescription, actions: [.cancel("OK")])
                }
            },
            .cancel(),
        ])
    }

    /// Shows a run in the right-hand pane (the changes pane gives way).
    func openFlowPanel(_ runID: String, in doc: ChatDocument) {
        doc.changesOpen = false
        doc.flowPanelRunID = runID
    }

    /// Starts the run again, reusing what the agents of `old` finished.
    func resumeFlow(_ old: FlowRun, in doc: ChatDocument) {
        do {
            let c = try startFlow(script: old.script, meta: old.meta, args: old.args, chatID: doc.id, resumeFrom: old.id)
            openFlowPanel(c.run.id, in: doc)
        } catch {
            presenter.alert("Couldn't resume the flow", message: error.localizedDescription, actions: [.cancel("OK")])
        }
    }
}

/// What the `flow` tool of a coding chat talks to: the app, for the chat that owns it.
final class AppFlowHost: FlowHost, @unchecked Sendable {
    private weak var model: AppModel?
    private let chatID: UUID

    init(model: AppModel, chatID: UUID) {
        self.model = model
        self.chatID = chatID
    }

    func start(script: String, meta: FlowMeta, args: JSONValue?, callID: String) async throws -> String {
        try await MainActor.run {
            guard let model else { throw ToolError("The app is gone.") }
            let c = try model.startFlow(script: script, meta: meta, args: args, chatID: chatID)
            model.chats.first(where: { $0.id == chatID })?.flowRunByCall[callID] = c.run.id
            return c.run.id
        }
    }

    func limits() async -> (concurrency: Int, maxAgents: Int) {
        await MainActor.run {
            guard let model, let doc = model.chats.first(where: { $0.id == chatID }) else { return (4, 30) }
            return model.flowLimits(for: doc)
        }
    }
}
