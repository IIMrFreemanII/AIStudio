import Agent
import LLMClient
import SwiftUI

enum FlowFormat {
    /// "42s", "3m 05s", "1h 02m".
    static func duration(_ seconds: TimeInterval) -> String {
        let s = max(0, Int(seconds))
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m " + String(format: "%02d", s % 60) + "s" }
        return "\(s / 3600)h " + String(format: "%02d", (s % 3600) / 60) + "m"
    }

    /// "850", "12.3k", "1.2M".
    static func tokens(_ n: Int) -> String {
        if n < 1000 { return "\(n)" }
        if n < 1_000_000 { return String(format: "%.1fk", Double(n) / 1000) }
        return String(format: "%.1fM", Double(n) / 1_000_000)
    }

    static func cost(_ usd: Double) -> String { String(format: "≈ $%.2f GPU", usd) }
    static let costHelp = "Estimate: run time × the GPU's price. Parallel agents share the GPU."

    static func tint(_ status: FlowStatus) -> Color {
        switch status {
        case .running: return MG.accent
        case .done: return MG.success
        case .failed: return MG.destructive
        case .stopped: return MG.secondaryLabel
        }
    }

    static func tint(_ status: FlowAgentStatus) -> Color {
        switch status {
        case .running: return MG.accent
        case .done, .cached: return MG.success
        case .failed: return MG.destructive
        case .stopped: return MG.secondaryLabel
        }
    }

    /// Phase titles in order: the meta's, then any the agents named.
    static func phases(_ run: FlowRun) -> [String] {
        var out = run.meta.phases
        for a in run.agents { if let p = a.phase, !out.contains(p) { out.append(p) } }
        return out
    }
}

struct FlowStatusGlyph: View {
    let status: FlowStatus

    var body: some View {
        switch status {
        case .running: MGSpinner().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.destructive)
        case .stopped: Image(systemName: "stop.circle.fill").foregroundStyle(MG.secondaryLabel)
        }
    }
}

/// Elapsed time and estimated cost of a run, live while it runs.
struct FlowClock: View {
    let run: FlowRun
    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            let now = run.isFinished ? (run.endedAt ?? ctx.date) : ctx.date
            HStack(spacing: 8) {
                Text(FlowFormat.duration(now.timeIntervalSince(run.startedAt))).monospacedDigit()
                if let c = run.estimatedCost(now: now) {
                    Text(FlowFormat.cost(c)).monospacedDigit().mgHelp(FlowFormat.costHelp)
                }
            }
        }
    }
}

/// A finished (or running) `flow` call in the transcript: progress by phase, and what can be done with the run.
struct FlowRunCard: View {
    @Environment(AppModel.self) private var model
    let run: FlowController

    var doc: ChatDocument? { model.chats.first { $0.id == run.chatID } }

    var body: some View {
        let r = run.run
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                FlowStatusGlyph(status: r.status)
                Text(r.meta.name).font(.mgCallout.weight(.medium)).lineLimit(1)
                Text("\(r.doneAgents)/\(r.agents.count) agents").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer(minLength: 4)
                FlowClock(run: r).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            let phases = FlowFormat.phases(r)
            if !phases.isEmpty {
                FlowPhaseRow(run: r, phases: phases)
            }
            if let e = r.error, r.status == .failed {
                Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(3).textSelection(.enabled)
            }
            HStack(spacing: 8) {
                if let doc {
                    Button("Open") { model.openFlowPanel(r.id, in: doc) }.buttonStyle(.mg(.bordered))
                }
                if r.status == .running && !run.readOnly {
                    Button("Stop") { run.stop() }.buttonStyle(.mg(.bordered))
                }
                if r.isFinished, let doc {
                    Button("Resume") { model.resumeFlow(r, in: doc) }.buttonStyle(.mg(.bordered))
                        .mgHelp("Run it again, reusing the agents that finished")
                    Button("Save Flow…") { askWhereToSave(r, doc) }.buttonStyle(.mg(.bordered))
                    if !r.branches.isEmpty {
                        Button("Remove Worktrees") { model.removeFlowWorktrees(r, chat: doc) }.buttonStyle(.mg(.bordered))
                    }
                }
                Spacer(minLength: 0)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    func askWhereToSave(_ r: FlowRun, _ doc: ChatDocument) {
        var actions: [MGAlertAction] = []
        if doc.projectURL != nil {
            actions.append(MGAlertAction(label: "In the Project") { model.saveFlow(r, personal: false, chat: doc) })
        }
        actions.append(MGAlertAction(label: "Personal") { model.saveFlow(r, personal: true, chat: doc) })
        actions.append(.cancel())
        model.presenter.alert("Save “\(r.meta.name)” as a flow?",
                              message: "Project flows live in .aistudio/flows and go with the repository; personal ones are for all projects.",
                              actions: actions)
    }
}

/// One small capsule per phase: its title and done/total, the current one highlighted.
struct FlowPhaseRow: View {
    let run: FlowRun
    let phases: [String]

    var body: some View {
        HStack(spacing: 6) {
            ForEach(phases, id: \.self) { p in
                let mine = run.agents.filter { $0.phase == p }
                let done = mine.filter { $0.status == .done || $0.status == .cached }.count
                let current = run.status == .running && run.currentPhase == p
                let failed = mine.contains { $0.status == .failed }
                HStack(spacing: 4) {
                    Text(p).lineLimit(1)
                    if !mine.isEmpty { Text("\(done)/\(mine.count)").monospacedDigit().foregroundStyle(failed ? MG.destructive : MG.secondaryLabel) }
                }
                .font(.mgFootnote)
                .foregroundStyle(current ? MG.accent : MG.label)
                .padding(.horizontal, 8).frame(height: 20)
                .background(Capsule().fill(current ? MG.accent.opacity(0.14) : MG.fill))
                .overlay(Capsule().strokeBorder(current ? MG.accent : .clear, lineWidth: 1))
            }
            Spacer(minLength: 0)
        }
    }
}

/// A flow result in the chat that started it: what happened, the result, and the branches with changes.
struct FlowEventCard: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let event: FlowEvent
    @State private var expanded = false

    var verb: String { event.status == "failed" ? "failed" : event.status == "stopped" ? "stopped" : "finished" }
    var tint: Color { event.status == "failed" ? MG.destructive : event.status == "stopped" ? MG.secondaryLabel : MG.success }
    var icon: String { event.status == "failed" ? "xmark.octagon" : event.status == "stopped" ? "stop.circle" : "checkmark.circle" }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(tint)
                Text("Flow “\(event.name)” \(verb) · \(event.agents) agent\(event.agents == 1 ? "" : "s")")
                    .font(.mgCallout.weight(.medium)).lineLimit(1)
                if event.failed > 0 {
                    Text("\(event.failed) failed").font(.mgFootnote).foregroundStyle(MG.destructive)
                }
                Spacer(minLength: 4)
                Button("Open") { model.openFlowPanel(event.runID, in: doc) }.buttonStyle(.mg(.bordered))
            }
            if !event.summary.isEmpty {
                Text(event.summary)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .lineLimit(expanded ? nil : 3)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if event.summary.count > 180 || event.summary.split(separator: "\n").count > 3 {
                    Button(expanded ? "Less" : "More") { withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() } }
                        .buttonStyle(.plain).font(.mgFootnote).foregroundStyle(MG.accent)
                }
            }
            ForEach(event.branches, id: \.self) { b in
                HStack(spacing: 8) {
                    Label(b, systemImage: "arrow.triangle.branch")
                        .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
                        .padding(.horizontal, 6).frame(height: 20)
                        .background(Capsule().fill(MG.fill))
                    Spacer(minLength: 4)
                    Button("Merge…") { model.mergeFlowBranch(b, chat: doc) }.buttonStyle(.mg(.bordered))
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
    }
}

extension ChatDocument {
    /// The run a `flow` tool call started: remembered, else named in the call's result ("(run 1a2b3c4d)").
    @MainActor func flowRunID(for call: ToolCall, result: ChatTurn?) -> String? {
        if let id = flowRunByCall[call.id] { return id }
        guard let t = result?.text, let r = t.range(of: "(run ") else { return nil }
        let id = t[r.upperBound...].prefix { $0 != ")" }
        return id.isEmpty ? nil : String(id)
    }
}
