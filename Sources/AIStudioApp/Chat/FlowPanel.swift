import Agent
import LLMClient
import SwiftUI

/// The right-hand pane of a coding chat while a flow run is open: every agent by phase, live.
struct FlowPanel: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let run: FlowController

    var body: some View {
        let r = run.run
        VStack(spacing: 0) {
            header(r)
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    if r.agents.isEmpty {
                        Text(r.status == .running ? "Starting the first agents…" : "This run started no agents.")
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).padding(14)
                    }
                    ForEach(groups(r), id: \.title) { g in
                        Text(g.title).font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.secondaryLabel)
                            .padding(.horizontal, 12).padding(.top, 12).padding(.bottom, 4)
                        ForEach(g.agents) { a in
                            FlowAgentRow(run: run, agent: a, finished: r.isFinished)
                            MGDivider()
                        }
                    }
                }
            }
            .mgScrollIndicators()
            footer(r)
        }
        .background(MG.contentBackground)
    }

    func groups(_ r: FlowRun) -> [(title: String, agents: [FlowAgentRecord])] {
        var out: [(title: String, agents: [FlowAgentRecord])] = []
        for p in FlowFormat.phases(r) {
            let mine = r.agents.filter { $0.phase == p }
            if !mine.isEmpty { out.append((p, mine)) }
        }
        let rest = r.agents.filter { $0.phase == nil }
        if !rest.isEmpty { out.append((out.isEmpty ? "Agents" : "Other agents", rest)) }
        return out
    }

    @ViewBuilder private func header(_ r: FlowRun) -> some View {
        HStack(spacing: 8) {
            FlowStatusGlyph(status: r.status)
            VStack(alignment: .leading, spacing: 1) {
                Text(r.meta.name).font(.mgHeadline).lineLimit(1)
                HStack(spacing: 6) {
                    Text(r.status == .running ? "Running" : r.status.rawValue.capitalized)
                        .foregroundStyle(FlowFormat.tint(r.status))
                    FlowClock(run: r).foregroundStyle(MG.secondaryLabel)
                }
                .font(.mgFootnote)
            }
            Spacer(minLength: 0)
            if r.status == .running && !run.readOnly {
                Button("Stop") { run.stop() }.buttonStyle(.mg(.bordered))
            }
            Button { doc.flowPanelRunID = nil } label: { Image(systemName: "xmark") }
                .buttonStyle(.mg(.icon))
                .mgHelp("Close")
        }
        .padding(.horizontal, 12)
        .frame(height: 48)
    }

    @ViewBuilder private func footer(_ r: FlowRun) -> some View {
        let logs = Array(r.logs.suffix(5))
        if !logs.isEmpty || r.result != nil || r.error != nil {
            MGDivider()
            VStack(alignment: .leading, spacing: 6) {
                if !logs.isEmpty {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(logs.enumerated()), id: \.offset) { _, l in
                            Text(l).font(.mgMono).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.tail)
                        }
                    }
                }
                if let e = r.error {
                    Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).lineLimit(4).textSelection(.enabled)
                }
                if let result = r.result {
                    FlowResultBox(text: result)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct FlowResultBox: View {
    let text: String
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Result").font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.secondaryLabel)
                Spacer()
                if text.split(separator: "\n").count > 8 {
                    Button(expanded ? "Less" : "More") { expanded.toggle() }
                        .buttonStyle(.plain).font(.mgFootnote).foregroundStyle(MG.accent)
                }
            }
            ScrollView {
                Text(text).font(.mgMono).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: expanded ? 320 : 110)
            .mgScrollIndicators()
        }
    }
}

/// One agent: the summary line, and (clicked) every step, the final text or JSON, and the error.
struct FlowAgentRow: View {
    @Environment(AppModel.self) private var model
    let run: FlowController
    let agent: FlowAgentRecord
    let finished: Bool
    @State private var longText = false

    var open: Bool { run.expandedAgent == agent.id }

    @ViewBuilder var glyph: some View {
        switch agent.status {
        case .running: MGSpinner().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        case .cached: Image(systemName: "arrow.uturn.backward.circle.fill").foregroundStyle(MG.success).mgHelp("Reused from the earlier run")
        case .failed: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.destructive)
        case .stopped: Image(systemName: "stop.circle.fill").foregroundStyle(MG.secondaryLabel)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { run.expandedAgent = open ? nil : agent.id }
            } label: {
                summary.contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open {
                details.padding(.horizontal, 12).padding(.bottom, 10).padding(.leading, 22)
            }
        }
    }

    var summary: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                glyph.frame(width: 14)
                Text(agent.label).font(.mgCallout).lineLimit(1).truncationMode(.middle)
                if agent.mode == .edit {
                    Text("edit").font(.mgFootnote.weight(.medium)).foregroundStyle(MG.accent)
                        .padding(.horizontal, 6).frame(height: 18)
                        .background(Capsule().fill(MG.accent.opacity(0.14)))
                }
                Spacer(minLength: 4)
                let tokens = agent.usage.promptTokens + agent.usage.completionTokens
                if tokens > 0 {
                    Text(FlowFormat.tokens(tokens)).font(.mgFootnote).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                        .mgHelp("\(agent.usage.promptTokens) prompt + \(agent.usage.completionTokens) reply tokens")
                }
                TimelineView(.periodic(from: .now, by: 1)) { ctx in
                    let end = agent.endedAt ?? (finished ? agent.startedAt : ctx.date)
                    Text(FlowFormat.duration(end.timeIntervalSince(agent.startedAt))).font(.mgFootnote).monospacedDigit()
                        .foregroundStyle(MG.secondaryLabel)
                }
            }
            if let last = lastLine {
                Text(last).font(.mgFootnote).foregroundStyle(agent.status == .failed && !open ? MG.destructive : MG.secondaryLabel)
                    .lineLimit(1).truncationMode(.tail).padding(.leading, 22)
            }
            if let b = agent.branch {
                Label(b, systemImage: "arrow.triangle.branch")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
                    .padding(.horizontal, 6).frame(height: 18)
                    .background(Capsule().fill(MG.fill))
                    .padding(.leading, 22)
            }
        }
        .padding(.horizontal, 12).padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    var lastLine: String? {
        // Open, the details show the error in full: the summary keeps the last step.
        if !open, agent.status == .failed || agent.status == .stopped, let e = agent.error { return e }
        if agent.status == .running { return agent.steps.last }
        return agent.steps.last
    }

    @ViewBuilder var details: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !agent.steps.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(agent.steps.enumerated()), id: \.offset) { _, s in
                        Text(s).font(.system(size: 11, design: .monospaced)).foregroundStyle(MG.secondaryLabel)
                            .lineLimit(2).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            if let json = agent.json {
                Text(Self.pretty(json)).font(.system(size: 11, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if let text = agent.text, !text.isEmpty {
                Text(text).font(.mgFootnote).textSelection(.enabled).lineLimit(longText ? nil : 20)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if text.split(separator: "\n").count > 20 {
                    Button(longText ? "Less" : "More") { longText.toggle() }
                        .buttonStyle(.plain).font(.mgFootnote).foregroundStyle(MG.accent)
                }
            }
            if let e = agent.error {
                Text(e).font(.mgFootnote).foregroundStyle(MG.destructive).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    static func pretty(_ v: JSONValue) -> String {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return (try? enc.encode(v)).map { String(decoding: $0, as: UTF8.self) } ?? ""
    }
}
