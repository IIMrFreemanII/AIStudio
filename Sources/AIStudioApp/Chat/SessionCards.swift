import Agent
import LLMClient
import SwiftUI

/// A session's report in the chat that started it: what it did, and a button to open it.
struct SessionEventCard: View {
    @Environment(AppModel.self) private var model
    let event: SessionEvent
    @State private var expanded = false

    var verb: String { event.status == "failed" ? "failed" : event.status == "stopped" ? "stopped" : "finished" }
    var icon: String { event.status == "failed" ? "xmark.octagon" : event.status == "stopped" ? "stop.circle" : "checkmark.circle" }
    var tint: Color { event.status == "failed" ? MG.destructive : event.status == "stopped" ? MG.secondaryLabel : MG.success }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon).foregroundStyle(tint)
                Text("Session “\(event.title)” \(verb)").font(.mgCallout.weight(.medium)).lineLimit(1)
                if let b = event.branch {
                    Label(b, systemImage: "arrow.triangle.branch")
                        .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                        .padding(.horizontal, 6).frame(height: 20)
                        .background(Capsule().fill(MG.fill))
                }
                Spacer(minLength: 4)
                Button("Open") { model.openChat(event.childID) }.buttonStyle(.mg(.bordered))
            }
            if !event.report.isEmpty {
                Text(event.report)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .lineLimit(expanded ? nil : 3)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if event.report.count > 180 || event.report.split(separator: "\n").count > 3 {
                    Button(expanded ? "Less" : "More") { withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() } }
                        .buttonStyle(.plain).font(.mgFootnote).foregroundStyle(MG.accent)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
    }
}

/// The agent's suggestion of a session: the user starts it or dismisses it.
struct SessionProposalCard: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?
    @State private var starting = false
    @State private var expanded = false

    var spec: SessionSpec? {
        if let p = doc.proposals[call.id] { return p.spec }
        guard let a = try? ToolArgs(json: call.arguments), let title = a.string("title"), let prompt = a.string("prompt") else { return nil }
        let context = a.string("context").flatMap(SessionSpec.Context.init(rawValue:)) ?? .fresh
        return SessionSpec(title: title, prompt: prompt, context: context, worktree: a.raw["worktree"] == nil ? true : a.bool("worktree"))
    }

    var body: some View {
        if let spec {
            let decision = doc.proposalDecisions[call.id]  // nil: undecided; .some(nil): dismissed; .some(id): started
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "plus.rectangle.on.rectangle").foregroundStyle(MG.accent)
                    Text("New session: \(spec.title)").font(.mgCallout.weight(.medium)).lineLimit(1)
                    Spacer(minLength: 4)
                }
                Text(spec.prompt)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .lineLimit(expanded ? nil : 3)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onTapGesture { withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() } }
                if let decision {
                    if let id = decision {
                        HStack(spacing: 6) {
                            Text("Started —").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            Button("Open") { model.openChat(id) }.buttonStyle(.mg(.bordered))
                        }
                    } else {
                        Text("Dismissed").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    }
                } else if result != nil {
                    HStack(spacing: 8) {
                        Button("Start Session") {
                            starting = true
                            Task { @MainActor in
                                do {
                                    let info = try await model.startChild(spec, parent: doc)
                                    doc.proposalDecisions[call.id] = .some(info.id)
                                } catch {
                                    doc.attachNote = error.localizedDescription
                                }
                                starting = false
                            }
                        }
                        .buttonStyle(.mg(.prominent))
                        .disabled(starting)
                        Button("Dismiss") { doc.proposalDecisions[call.id] = .some(nil) }.buttonStyle(.mg(.bordered))
                        Spacer()
                    }
                }
            }
            .padding(10)
            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        } else {
            ToolCard(doc: doc, call: call, result: result)
        }
    }
}
