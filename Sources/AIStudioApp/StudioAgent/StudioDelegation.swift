import Agent
import Foundation
import LLMClient

/// One step a worker took, for its task row in the panel's plan (live; saved in the `delegate` result's text).
struct StudioWorkerStep: Identifiable, Equatable {
    var id: String
    var title: String
    var status: String  // running, done, failed, denied
}

extension StudioController {
    /// A worker's step started or ended.
    func workerStep(task: String, id: String, title: String? = nil, status: String) {
        var list = workerSteps[task] ?? []
        if let i = list.firstIndex(where: { $0.id == id }) {
            list[i].status = status
            if let title { list[i].title = title }
        } else {
            list.append(StudioWorkerStep(id: id, title: title ?? "", status: status))
        }
        workerSteps[task] = list
    }
}

/// The architect's side of a reply: its plan (approved by the user on the plan card) and the workers it hands tasks
/// to. A worker is its own agent loop on the same model and GPU with less thinking, a brief instead of the chat, and
/// a smaller set of studio tools; its actions carry its own pointer (`StudioActorScope`).
@MainActor
final class StudioWorkers {
    let session: StudioSession
    let client: any ChatStreaming
    let model: String
    let effort: String?
    let inlineThinking: Bool
    let vision: Bool
    /// Each task's last brief and report, for a retry with the architect's feedback.
    private var history: [String: [ChatMessage]] = [:]

    init(session: StudioSession, client: any ChatStreaming, model: String, effort: String?, inlineThinking: Bool, vision: Bool) {
        self.session = session
        self.client = client
        self.model = model
        self.effort = effort
        self.inlineThinking = inlineThinking
        self.vision = vision
    }

    var studio: StudioController { session.studio }

    /// The tools a worker gets: reading, editing, running and looking. Not planning or delegating, navigating on its
    /// own, approving, exporting, deleting, creating movies, or the click-and-type fallback (it needs the screen).
    static let workerToolNames: Set<String> = [
        "studio_state", "studio_list", "movie_get", "movie_edit", "movie_run", "movie_choose_take", "look_at",
        "get_graph", "edit_graph", "set_prompt", "run_workflow", "get_errors", "list_models",
    ]

    func tools() -> [any AgentTool] {
        StudioTools.all(session).filter { Self.workerToolNames.contains($0.name) }
    }

    // MARK: the plan card

    /// Shows the plan card and waits for the user: Approve, Change… (a note) or Reject.
    func plan(_ raw: Any?, callID: String) async throws -> String {
        let plan: StudioPlan
        do { plan = try StudioPlan.parse(raw) } catch let r as StudioPlan.Refusal { throw ToolError(r.description) }
        if let old = studio.plan, old.running > 0 { throw ToolError("workers are still running the current plan: wait for them first") }
        let answer: StudioPlanAnswer = await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<StudioPlanAnswer, Never>) in
                var done = false
                studio.pendingPlan = (plan, callID, { a in
                    guard !done else { return }
                    done = true
                    cont.resume(returning: a)
                })
                studio.cursor(.architect).state = .waiting
                studio.cursor(.architect).waiting = "Your plan: waiting for you to approve it"
            }
        } onCancel: {
            Task { @MainActor [weak self] in self?.studio.answerPlan(.cancelled) }
        }
        studio.cursor(.architect).waiting = nil
        studio.cursor(.architect).state = .working
        switch answer {
        case .approved:
            studio.plan = plan
            studio.workerSteps = [:]
            let ready = plan.tasks.filter { plan.blockers($0.id).isEmpty }.map(\.id)
            return "The user approved the plan (\(plan.tasks.count) tasks). Hand out the ready ones now in one delegate call with tasks: [\(ready.map { "\"\($0)\"" }.joined(separator: ", "))] — they run at the same time, up to \(StudioPlan.maxWorkers). Check each worker's report (look_at, movie_get) before marking its stage done or delegating what depends on it."
        case .change(let note):
            return "The user wants the plan changed: “\(note)”. Make a new plan with studio_plan."
        case .rejected:
            return "The user rejected the plan. Nothing was started: ask them what they'd like instead."
        case .cancelled:
            throw CancellationError()
        }
    }

    // MARK: delegating

    /// Runs one approved task on a worker (or again with `feedback`), and returns its report.
    func delegate(task id: String, feedback: String?, callID: String) async throws -> String {
        guard studio.plan != nil else { throw ToolError("there's no approved plan: make one with studio_plan first") }
        // A free worker (several delegate calls may arrive at once).
        while studio.plan?.hasFreeWorker == false { try await Task.sleep(for: .milliseconds(200)) }
        let actor: StudioActor
        do {
            guard var p = studio.plan else { throw ToolError("the plan is gone") }
            actor = try p.start(id, retry: feedback != nil)
            studio.plan = p
        } catch let r as StudioPlan.Refusal {
            throw ToolError(r.description)
        }
        guard let task = studio.plan?.task(id) else { throw ToolError("no task \(id)") }
        let c = studio.cursor(actor)
        c.spent = 0
        if feedback == nil { studio.workerSteps[id] = [] }
        await studio.handOff(to: actor, task: id)

        var messages: [ChatMessage]
        if let feedback, let before = history[id] {
            messages = before + [.user("The architect checked your work and wants it redone: \(feedback)\nFix it, then report again.")]
        } else {
            messages = [.system(Self.workerPrompt(actor, task: task, app: session.app, vision: vision)), .user(task.brief)]
        }
        var config = AgentConfig(model: model, tools: tools(), mode: .acceptEdits)
        config.maxSteps = 30
        config.reasoningEffort = effort
        config.inlineThinking = inlineThinking
        config.vision = vision
        let context = ToolContext(root: session.app?.library.root ?? FileManager.default.temporaryDirectory)
        let report = WorkerReport()
        let prefix = callID + "/"
        let studio = self.studio
        var outcome: AgentOutcome = .finished
        var failure: Error?
        do {
            outcome = try await StudioActorScope.$actor.withValue(actor) {
                try await StudioActorScope.$prefix.withValue(prefix) {
                    try await AgentLoop.run(client: client, messages: messages, config: config, context: context,
                                            ask: { _ in .deny }, events: { e in
                        switch e {
                        case .toolStarted(let call, let preview):
                            await MainActor.run { studio.workerStep(task: id, id: prefix + call.id, title: preview.title, status: "running") }
                        case .toolFinished(let callID, let output, let status):
                            await MainActor.run {
                                studio.workerStep(task: id, id: prefix + callID, status: output.isError && status == .done ? "failed" : status.rawValue)
                            }
                        case .assistantFinished(let content, _, _):
                            report.set(content)
                        default: break
                        }
                    })
                }
            }
        } catch {
            failure = error
        }
        let text = report.value.trimmingCharacters(in: .whitespacesAndNewlines)
        let ok = failure == nil && outcome == .finished && !text.isEmpty && !Self.couldntFinish(text)
        history[id] = messages + (text.isEmpty ? [] : [.assistant(text)])
        if var p = studio.plan {
            p.finish(id, ok: ok, spent: c.spent, report: text)
            studio.plan = p
        }
        if failure is CancellationError || Task.isCancelled {
            Task { await studio.finish(actor, ok: false) }
            throw CancellationError()
        }
        Task { await studio.finish(actor, ok: ok) }

        var out = "\(actor.label) on “\(task.title)” (\(id))"
        if let failure { out += " stopped with an error: \(failure.localizedDescription)" } else if outcome == .stepLimit {
            out += " stopped after 30 steps"
        }
        out += ":\n" + (text.isEmpty ? "(no report)" : text)
        if c.spent > 0 { out += String(format: "\nIts paid runs cost ≈ $%.2f.", c.spent) }
        if let steps = studio.workerSteps[id], !steps.isEmpty {
            out += "\n\nSteps:\n" + steps.map { ($0.status == "done" ? "✓ " : $0.status == "running" ? "… " : "✗ ") + $0.title }.joined(separator: "\n")
        }
        if let p = studio.plan {
            let ready = p.tasks.filter { $0.status == .waiting && p.blockers($0.id).isEmpty }.map(\.id)
            if p.isFinished { out += "\n\nEvery task in the plan has ended." } else if !ready.isEmpty {
                out += "\n\nReady to delegate now: \(ready.joined(separator: ", "))."
            }
        }
        return out
    }

    /// The report's last line says the worker couldn't do its task (not that what it reviewed has problems).
    static func couldntFinish(_ report: String) -> Bool {
        let last = report.split(separator: "\n").last { !$0.trimmingCharacters(in: .whitespaces).isEmpty }?.lowercased() ?? ""
        return last.contains("result:") && (last.contains("couldn't finish") || last.contains("could not finish"))
    }

    /// Several tasks at once (one delegate call with `tasks`): each on its own worker, side by side; their reports in
    /// the order asked.
    func delegate(tasks ids: [String], callID: String) async throws -> String {
        if ids.count == 1 { return try await delegate(task: ids[0], feedback: nil, callID: callID) }
        var reports = [String](repeating: "", count: ids.count)
        try await withThrowingTaskGroup(of: (Int, String).self) { g in
            for (i, id) in ids.enumerated() {
                g.addTask { @MainActor in
                    do {
                        return (i, try await self.delegate(task: id, feedback: nil, callID: callID + "/" + id))
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        return (i, "\(id) didn't start: \(error.localizedDescription)")
                    }
                }
            }
            for try await (i, r) in g { reports[i] = r }
        }
        return reports.joined(separator: "\n\n———\n\n")
    }

    static func workerPrompt(_ actor: StudioActor, task: StudioPlan.Task, app: AppModel?, vision: Bool) -> String {
        let state = app.flatMap { try? StudioSession(app: $0, vision: vision).state() } ?? ""
        return """
        You are \(actor.label), one of the Studio agent's workers in AIStudio, a Mac app for making images, video, sound \
        and short films with AI models. The architect planned the job and gave you one task: “\(task.title)”. Do only \
        that task, with your tools, then report.

        - Your actions show on screen with your own pointer when their page is in front; when another worker is using \
        the screen, they happen off screen (that's fine: the tools still work).
        - Paid work shows the app's cost sheet: you can't press Run. The tool waits for the user and says what they chose.
        - You can't ask the user anything, approve stages, export or delete. Put questions in your report.
        - \(vision ? "Check images and clips you make or review with look_at." : "This model can't see images: say so in your report.")
        - End with a short report: what you did (shot and take numbers), what ran and cost, anything wrong you saw, and \
        a last line: `Result: done` when you did the task (also when what you checked has problems: those go in the \
        report), or `Result: couldn't finish — <why>` only when you couldn't do the task itself.

        The app now:
        \(state)
        """
    }
}

private final class WorkerReport: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""
    func set(_ s: String) { if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lock.withLock { text = s } } }
    var value: String { lock.withLock { text } }
}

extension StudioController {
    /// The plan card's buttons.
    func answerPlan(_ a: StudioPlanAnswer) {
        guard let p = pendingPlan else { return }
        pendingPlan = nil
        p.answer(a)
    }
}

extension StudioTools {
    /// The architect's planning tools.
    @MainActor
    static func architect(_ w: StudioWorkers) -> [any AgentTool] {
        [
            StudioTool("studio_plan",
                       "Shows the user your plan for a job with independent parts and waits for their answer (Approve, a change note, or Reject). Workers start only after Approve. Each task: id, title, brief (complete: the worker sees only this), kind (act or review), paid (the paid stages it runs), estimate (rough $ for them), depends_on (task ids that must be done first).",
                       .object([
                           "type": .string("object"),
                           "properties": .object([
                               "tasks": .object([
                                   "type": .string("array"),
                                   "items": .object([
                                       "type": .string("object"),
                                       "properties": .object([
                                           "id": .object(["type": .string("string")]),
                                           "title": .object(["type": .string("string")]),
                                           "brief": .object(["type": .string("string")]),
                                           "kind": .object(["type": .string("string"), "description": .string("act | review")]),
                                           "paid": .object(["type": .string("array"), "items": .object(["type": .string("string")])]),
                                           "estimate": .object(["type": .string("number")]),
                                           "depends_on": .object(["type": .string("array"), "items": .object(["type": .string("string")])]),
                                       ]),
                                       "required": .array([.string("id"), .string("title"), .string("brief")]),
                                   ]),
                               ]),
                           ]),
                           "required": .array([.string("tasks")]),
                       ]),
                       title: { a in "Planned \((a.raw["tasks"] as? [Any])?.count ?? 0) tasks" }) { a, id in
                ToolOutput(try await w.plan(a.raw["tasks"], callID: id))
            },
            StudioTool("delegate",
                       "Hands approved tasks to workers, each with its own pointer, and returns their reports. Put every task that's ready in tasks: they run at the same time (up to 3 workers). To have a worker redo its task, call with task and feedback (at most twice per task); after that do it yourself or ask the user.",
                       .object([
                           "type": .string("object"),
                           "properties": .object([
                               "tasks": .object(["type": .string("array"), "items": .object(["type": .string("string")]),
                                                 "description": .string("Ids of approved tasks to run now, side by side")]),
                               "task": .object(["type": .string("string"), "description": .string("One task's id (for a redo with feedback)")]),
                               "feedback": .object(["type": .string("string"), "description": .string("Only for a redo: what was wrong and what right looks like")]),
                           ]),
                       ]),
                       title: { a in
                           if let f = a.string("feedback"), !f.isEmpty { return "Sent \(a.string("task") ?? "a task") back" }
                           let ids = (a.raw["tasks"] as? [Any])?.compactMap { $0 as? String } ?? [a.string("task") ?? "a task"]
                           return "Handed out " + ids.joined(separator: ", ")
                       }) { a, id in
                let feedback = a.string("feedback").flatMap { $0.isEmpty ? nil : $0 }
                var ids = (a.raw["tasks"] as? [Any])?.compactMap { $0 as? String } ?? []
                if let one = a.string("task"), !one.isEmpty, !ids.contains(one) { ids.insert(one, at: 0) }
                guard !ids.isEmpty else { throw ToolError("say which tasks: tasks (ids from the approved plan)") }
                if let feedback {
                    guard ids.count == 1 else { throw ToolError("feedback goes with one task: call delegate with task and feedback") }
                    return ToolOutput(try await w.delegate(task: ids[0], feedback: feedback, callID: id))
                }
                return ToolOutput(try await w.delegate(tasks: ids, callID: id))
            },
        ]
    }
}
