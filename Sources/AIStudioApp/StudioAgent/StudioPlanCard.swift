import Agent
import SwiftUI

/// The architect's plan at the top of the panel: while it waits on you, its tasks with Approve, Change… and Reject;
/// once approved, a checklist whose rows show each task's worker, status, time and cost, and open to its steps.
struct StudioPlanCard: View {
    @Environment(AppModel.self) private var model
    let studio: StudioController
    @State private var note = ""
    @State private var changing = false
    @State private var folded = false

    var body: some View {
        if let pending = studio.pendingPlan {
            card(pending.plan, waiting: true)
        } else if let plan = studio.plan {
            card(plan, waiting: false)
        }
    }

    func card(_ plan: StudioPlan, waiting: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                StudioActorIcon(actor: .architect, state: waiting ? .waiting : .idle, size: 16,
                                motions: studio.motions(studio.cursor(.architect)))
                Text(waiting ? "Plan: waiting for you" : "Plan").font(.mgHeadline)
                Spacer(minLength: 4)
                if plan.estimate > 0 {
                    Text(String(format: "≈ $%.2f", plan.estimate)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                        .mgHelp("The architect's rough estimate for the plan's paid stages; each run still shows its cost sheet")
                }
                if !waiting {
                    Text("\(plan.tasks.filter { $0.status == .done }.count)/\(plan.tasks.count)")
                        .font(.mgFootnote.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                    Button { folded.toggle() } label: { Image(systemName: folded ? "chevron.down" : "chevron.up") }
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .mgHelp(folded ? "Show the plan's tasks" : "Hide the plan's tasks")
                }
            }
            if waiting || !folded {
                ForEach(plan.tasks) { t in StudioTaskRow(studio: studio, task: t, waiting: waiting) }
            }
            if waiting {
                if changing {
                    HStack(spacing: 6) {
                        TextField("What should change?", text: $note)
                            .textFieldStyle(.plain).font(.mgFootnote)
                            .padding(.horizontal, 8).padding(.vertical, 5)
                            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
                            .onSubmit(sendChange)
                        Button("Send", action: sendChange).buttonStyle(.mg(.prominent)).controlSize(.small)
                            .disabled(note.trimmingCharacters(in: .whitespaces).isEmpty)
                    }
                }
                HStack(spacing: 6) {
                    Spacer()
                    Button("Reject") { studio.answerPlan(.rejected) }.buttonStyle(.mg(.bordered)).controlSize(.small)
                        .mgTarget("studio-plan-reject")
                    Button("Change…") { changing.toggle() }.buttonStyle(.mg(.bordered)).controlSize(.small)
                    Button("Approve") { studio.answerPlan(.approved) }.buttonStyle(.mg(.prominent)).controlSize(.small)
                        .keyboardShortcut(.return, modifiers: [.command])
                        .mgTarget("studio-plan-approve")
                }
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous)
            .fill(waiting ? MG.warning.opacity(0.08) : MG.controlBackground.opacity(0.6)))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
        .padding(.horizontal, 10).padding(.vertical, 6)
        .animation(.easeOut(duration: 0.2), value: waiting)
    }

    func sendChange() {
        let n = note.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !n.isEmpty else { return }
        studio.answerPlan(.change(n))
        note = ""
        changing = false
    }
}

/// One task of the plan: its worker's icon (where the worker docks), title, status, time and cost; it opens to the
/// worker's steps, and a step's click goes back to where it happened.
struct StudioTaskRow: View {
    let studio: StudioController
    let task: StudioPlan.Task
    let waiting: Bool
    @State private var open = false
    @State private var started: Date?

    var body: some View {
        let actor = StudioActor.worker(task.worker)
        let steps = studio.workerSteps[task.id] ?? []
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 7) {
                StudioActorIcon(actor: actor, state: state, size: 18, motions: rowMotions(actor))
                    .mgTarget("studio-task-icon:\(task.id)")
                VStack(alignment: .leading, spacing: 1) {
                    Text(task.title).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(open ? nil : 1)
                    if waiting, !task.paid.isEmpty || !task.dependsOn.isEmpty {
                        Text(detail).font(.mg(size: 11)).foregroundStyle(MG.secondaryLabel).lineLimit(2)
                    }
                }
                Spacer(minLength: 4)
                if task.spent > 0 {
                    Text(String(format: "$%.2f", task.spent)).font(.mg(size: 11).monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                }
                if let started, task.status == .running {
                    Text(started, style: .timer).font(.mg(size: 11).monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                }
                statusLabel
                if !steps.isEmpty {
                    Button { open.toggle() } label: { Image(systemName: open ? "chevron.up" : "chevron.down") }
                        .buttonStyle(.mg(.icon)).controlSize(.mini)
                        .mgHelp(open ? "Hide \(actor.label)'s steps" : "Show \(actor.label)'s steps")
                }
            }
            if open {
                ForEach(steps) { s in
                    HStack(spacing: 6) {
                        stepIcon(s.status).frame(width: 12)
                        Text(s.title).font(.mg(size: 11)).foregroundStyle(MG.label).lineLimit(1)
                        Spacer(minLength: 0)
                    }
                    .padding(.leading, 24)
                    .contentShape(Rectangle())
                    .onTapGesture { studio.revisit(s.id) }
                    .mgHelp(studio.revisits[s.id] != nil ? "Show where this happened" : "")
                }
            }
        }
        .padding(.vertical, 2)
        .onChange(of: task.status) { _, s in if s == .running { started = Date() } }
    }

    var state: StudioActorState {
        switch task.status {
        case .waiting: return .idle
        case .running: return studio.cursor(.worker(task.worker)).waiting != nil ? .waiting : .working
        case .done: return .done
        case .failed: return .failed
        case .needsYou: return .waiting
        }
    }

    /// The row's icon moves while its task runs (or waits on you); finished rows stay still.
    func rowMotions(_ actor: StudioActor) -> StudioMotions {
        guard task.status == .running || task.status == .needsYou else { return [] }
        return StudioMotion.motions(state: studio.gate.isPaused ? .paused : state, pace: studio.pace, reduceMotion: studio.reduceMotion)
    }

    var detail: String {
        var parts: [String] = []
        if !task.paid.isEmpty {
            parts.append("Paid: " + task.paid.joined(separator: ", ") + (task.estimate.map { String(format: " (≈ $%.2f)", $0) } ?? ""))
        }
        if !task.dependsOn.isEmpty { parts.append("After " + task.dependsOn.joined(separator: ", ")) }
        if task.kind == .review { parts.insert("Review", at: 0) }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder var statusLabel: some View {
        switch task.status {
        case .waiting: if !waiting { Text("Waiting").font(.mg(size: 11)).foregroundStyle(MG.tertiaryLabel) }
        case .running: MGSpinner().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success).font(.mg(size: 11))
        case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.destructive).font(.mg(size: 11))
        case .needsYou: Image(systemName: "hourglass").foregroundStyle(MG.warning).font(.mg(size: 11))
        }
    }

    @ViewBuilder func stepIcon(_ status: String) -> some View {
        switch status {
        case "running": MGSpinner().controlSize(.mini)
        case "done": Image(systemName: "checkmark").foregroundStyle(MG.success).font(.system(size: 9, weight: .bold))
        case "denied": Image(systemName: "minus").foregroundStyle(MG.secondaryLabel).font(.system(size: 9, weight: .bold))
        default: Image(systemName: "xmark").foregroundStyle(MG.destructive).font(.system(size: 9, weight: .bold))
        }
    }
}
