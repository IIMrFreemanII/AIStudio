import Foundation
import JavaScriptCore
import LLMClient

/// Runs a flow script in JavaScriptCore. Each `agent()` call goes to the injected runner; progress goes to `onEvent`.
/// The JSContext is touched only on `queue`; events reach `onEvent` one at a time.
public final class FlowRuntime: @unchecked Sendable {
    private let script: String
    private let args: JSONValue?
    private let runner: any FlowAgentRunner
    private let concurrency: Int
    private let maxAgents: Int
    private let cache: [String: FlowAgentResult]
    private let onEvent: @Sendable (FlowRunEvent) -> Void

    private let queue = DispatchQueue(label: "aistudio.flow.runtime")
    private let emitLock = NSLock()
    private let stateLock = NSLock()

    // Guarded by stateLock.
    private var outcome: (status: FlowStatus, result: String?, error: String?)?
    private var continuation: CheckedContinuation<(status: FlowStatus, result: String?, error: String?), Never>?
    private var stopped = false

    // Confined to `queue`.
    private var context: JSContext?
    private var agentCount = 0
    private var startedCount = 0
    private var running = 0
    private var currentPhase: String?
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    private var waiting: [String] = []

    private struct Entry {
        var spec: FlowAgentSpec
        var kind: String
        var resolve: JSValue
        var reject: JSValue
        var task: Task<Void, Never>?
    }

    private enum Outcome: Sendable {
        case success(FlowAgentResult)
        case failure(String)
    }

    public init(script: String, args: JSONValue? = nil, runner: any FlowAgentRunner, concurrency: Int, maxAgents: Int,
                cache: [String: FlowAgentResult] = [:], onEvent: @escaping @Sendable (FlowRunEvent) -> Void) {
        self.script = script
        self.args = args
        self.runner = runner
        self.concurrency = max(1, concurrency)
        self.maxAgents = maxAgents
        self.cache = cache
        self.onEvent = onEvent
    }

    /// Runs the script to the end. result: the script's return value as JSON text (nil when it returns undefined).
    public func run() async -> (status: FlowStatus, result: String?, error: String?) {
        queue.async { self.start() }
        return await withCheckedContinuation { c in
            stateLock.lock()
            if let o = outcome {
                stateLock.unlock()
                c.resume(returning: o)
            } else {
                continuation = c
                stateLock.unlock()
            }
        }
    }

    /// Stops the run: running agents are cancelled, pending agent() promises reject, and run() returns .stopped.
    public func stop() {
        stateLock.lock()
        let already = stopped
        stopped = true
        stateLock.unlock()
        if already { return }
        queue.async { self.cleanUpAfterStop() }
    }

    private var isStopped: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return stopped
    }

    // MARK: Events and outcome

    private func emit(_ event: FlowRunEvent) {
        emitLock.lock()
        onEvent(event)
        emitLock.unlock()
    }

    private func finish(_ status: FlowStatus, result: String? = nil, error: String? = nil) {
        stateLock.lock()
        guard outcome == nil else { stateLock.unlock(); return }
        let o = (status: status, result: result, error: error)
        outcome = o
        let c = continuation
        continuation = nil
        stateLock.unlock()
        c?.resume(returning: o)
    }

    // MARK: Script

    private func start() {
        if isStopped { cleanUpAfterStop(); return }
        guard let ctx = JSContext() else {
            finish(.failed, error: "JavaScript isn't available.")
            return
        }
        context = ctx
        ctx.exceptionHandler = { [weak self] _, e in
            self?.finish(.failed, error: Self.describe(e))
        }
        install(in: ctx)

        let argsText: String
        if let args, let data = try? JSONEncoder().encode(args) { argsText = String(decoding: data, as: UTF8.self) } else { argsText = "null" }
        ctx.setObject(argsText as NSString, forKeyedSubscript: "__argsJSON" as NSString)
        ctx.evaluateScript("var args = JSON.parse(__argsJSON);")

        ctx.evaluateScript(FlowPrelude.source)
        if outcomeIsSet { return }

        // Same line as the script's first line, so reported line numbers match the script.
        let wrapped = "(async () => { " + FlowScript.body(script) + "\n})()"
        let promise = ctx.evaluateScript(wrapped)
        if outcomeIsSet { return }
        guard let promise, promise.isObject else {
            finish(.failed, error: "The script didn't start.")
            return
        }

        let onDone: @convention(block) (JSValue) -> Void = { [weak self] value in
            guard let self, let ctx = self.context else { return }
            let s = ctx.objectForKeyedSubscript("__stringify").call(withArguments: [value])
            if self.outcomeIsSet { return }
            let text: String? = (s == nil || s!.isUndefined || s!.isNull) ? nil : s!.toString()
            self.finish(.done, result: text)
        }
        let onError: @convention(block) (JSValue) -> Void = { [weak self] err in
            self?.finish(.failed, error: Self.describe(err))
        }
        promise.invokeMethod("then", withArguments: [
            JSValue(object: onDone, in: ctx) as Any, JSValue(object: onError, in: ctx) as Any,
        ])
    }

    private var outcomeIsSet: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return outcome != nil
    }

    private static func describe(_ e: JSValue?) -> String {
        guard let e else { return "Script error" }
        var message = e.toString() ?? "Script error"
        if e.isObject, let m = e.forProperty("message"), m.isString, let s = m.toString() { message = s }
        if e.isObject, let line = e.forProperty("line"), line.isNumber { message += " (line \(Int(line.toInt32())))" }
        return message
    }

    private func install(in ctx: JSContext) {
        let phaseFn: @convention(block) (String) -> Void = { [weak self] title in
            guard let self, !self.isStopped else { return }
            self.currentPhase = title
            self.emit(.phase(title))
        }
        let logFn: @convention(block) (String) -> Void = { [weak self] msg in
            guard let self, !self.isStopped else { return }
            self.emit(.log(msg))
        }
        let agentFn: @convention(block) (String, String, JSValue, JSValue) -> Void = { [weak self] prompt, optsJSON, resolve, reject in
            self?.startAgent(prompt: prompt, optsJSON: optsJSON, resolve: resolve, reject: reject)
        }
        ctx.setObject(phaseFn, forKeyedSubscript: "__phase" as NSString)
        ctx.setObject(logFn, forKeyedSubscript: "__log" as NSString)
        ctx.setObject(agentFn, forKeyedSubscript: "__agent" as NSString)
    }

    // MARK: Agents (all on `queue`)

    private struct Opts: Decodable {
        var label: String?
        var phase: String?
        var schema: JSONValue?
        var mode: String?
        var maxSteps: Int?
        var effort: String?
    }

    private func startAgent(prompt: String, optsJSON: String, resolve: JSValue, reject: JSValue) {
        if isStopped {
            reject.call(withArguments: [FlowErrors.stopped])
            return
        }
        agentCount += 1
        let n = agentCount
        let opts = (try? JSONDecoder().decode(Opts.self, from: Data(optsJSON.utf8))) ?? Opts()
        let id = "a\(n)"
        let mode = FlowAgentMode(rawValue: opts.mode ?? "read") ?? .read
        let schema: JSONValue? = { if case .null? = opts.schema { return nil } else { return opts.schema } }()
        let spec = FlowAgentSpec(id: id, label: opts.label ?? "agent \(n)", phase: opts.phase ?? currentPhase, prompt: prompt,
                                 schema: schema, mode: mode, maxSteps: opts.maxSteps, effort: opts.effort)
        let kind = schema != nil ? "json" : (mode == .edit ? "edit" : "text")

        if let cached = cache[spec.cacheKey] {
            emitPhaseIfNew(opts.phase)
            emit(.agentStarted(spec))
            var r = cached
            r.cached = true
            emit(.agentFinished(id: id, r))
            Self.resolve(resolve, kind: kind, result: r)
            return
        }
        if startedCount >= maxAgents {
            reject.call(withArguments: ["agent limit reached (\(maxAgents))"])
            return
        }
        startedCount += 1
        emitPhaseIfNew(opts.phase)
        emit(.agentStarted(spec))
        entries[id] = Entry(spec: spec, kind: kind, resolve: resolve, reject: reject, task: nil)
        order.append(id)
        waiting.append(id)
        pump()
    }

    private func emitPhaseIfNew(_ p: String?) {
        guard let p, p != currentPhase else { return }
        currentPhase = p
        emit(.phase(p))
    }

    private func pump() {
        while running < concurrency, !waiting.isEmpty, !isStopped {
            let id = waiting.removeFirst()
            guard var entry = entries[id] else { continue }
            running += 1
            let spec = entry.spec
            let runner = self.runner
            entry.task = Task { [weak self] in
                let outcome: Outcome
                do {
                    let r = try await runner.run(spec) { e in
                        guard let self, !self.isStopped else { return }
                        self.emit(.agentEvent(id: spec.id, e))
                    }
                    outcome = .success(r)
                } catch {
                    outcome = .failure(error.localizedDescription)
                }
                guard let self else { return }
                self.queue.async { self.complete(id, outcome) }
            }
            entries[id] = entry
        }
    }

    private func complete(_ id: String, _ outcome: Outcome) {
        guard let entry = entries.removeValue(forKey: id) else { return } // already settled (stopped)
        order.removeAll { $0 == id }
        running -= 1
        switch outcome {
        case .success(let r):
            emit(.agentFinished(id: id, r))
            Self.resolve(entry.resolve, kind: entry.kind, result: r)
        case .failure(let message):
            emit(.agentFailed(id: id, error: message))
            entry.reject.call(withArguments: ["\(entry.spec.label): \(message)"])
        }
        pump()
    }

    private static func resolve(_ fn: JSValue, kind: String, result: FlowAgentResult) {
        var kind = kind
        var json: Any = NSNull()
        if kind == "json" {
            if let j = result.json, let data = try? JSONEncoder().encode(j) { json = String(decoding: data, as: UTF8.self) } else { kind = "text" }
        }
        fn.call(withArguments: [kind, result.text, json, result.branch.map { $0 as Any } ?? NSNull()])
    }

    private func cleanUpAfterStop() {
        let ids = order
        let all = ids.compactMap { entries[$0] }
        entries = [:]
        order = []
        waiting = []
        for (id, e) in zip(ids, all) {
            e.task?.cancel()
            emit(.agentFailed(id: id, error: FlowErrors.stopped))
        }
        finish(.stopped, error: FlowErrors.stopped)
        for e in all { e.reject.call(withArguments: [FlowErrors.stopped]) }
    }
}
