import AppKit
import LLMClient
import SwiftUI

/// The agent's plans in this chat, newest first, and the selected one (with Approve / Keep Planning while it waits).
struct PlansPane: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument

    var body: some View {
        let all = Array(model.plans(doc).reversed())
        VStack(spacing: 0) {
            SidePaneHeader(title: "Plans", count: all.count)
            list(all)
        }
    }

    @ViewBuilder private func list(_ all: [ChatPlan]) -> some View {
        if all.isEmpty {
            MGEmptyState(symbol: "list.bullet.clipboard", title: "No plans yet",
                         message: "In Plan mode the agent explores the project and proposes a plan; it shows here and is saved with the chat.",
                         action: doc.mode == .plan ? nil : ("Switch to Plan Mode", {
                             doc.mode = .plan
                             doc.composerFocusRequest += 1
                         }), actionAnchor: "plans-empty-plan-mode")
        } else {
            let selected = all.first { $0.callID == doc.selectedPlan } ?? all[0]
            VSplitView {
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(all) { p in
                            PlanRow(doc: doc, plan: p, selected: p.callID == selected.callID)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .mgScrollIndicators()
                .frame(minHeight: 70, idealHeight: CGFloat(min(all.count, 5)) * 40 + 8)
                PlanDetail(doc: doc, plan: selected)
                    .frame(minHeight: 160, idealHeight: 600)
            }
        }
    }
}

private struct PlanRow: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let plan: ChatPlan
    let selected: Bool

    var body: some View {
        HStack(spacing: 8) {
            PlanStatusIcon(doc: doc, plan: plan)
            VStack(alignment: .leading, spacing: 1) {
                Text(plan.title).font(.mgCallout.weight(.medium)).lineLimit(1)
                Text(plan.date.formatted(date: .abbreviated, time: .shortened) + " · " + PlanStatus.line(doc, plan))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
            }
            Spacer(minLength: 0)
        }
        .mgRow(selected: selected, height: 38)
        .onTapGesture { doc.selectedPlan = plan.callID }
        .mgTestAnchor("plan-row:\(plan.callID)")
        .mgContextMenu { PlanMenu.items(model, doc, plan) }
    }
}

private struct PlanDetail: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let plan: ChatPlan

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                PlanStatusIcon(doc: doc, plan: plan)
                Text(PlanStatus.line(doc, plan)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                Spacer(minLength: 4)
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(plan.plan, forType: .string)
                } label: { Image(systemName: "doc.on.doc") }
                    .mgHelp("Copy the plan")
                MGMenuButton(kind: .icon) { PlanMenu.items(model, doc, plan) } label: { Image(systemName: "ellipsis.circle") }
                    .mgHelp("More")
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.small)
            .padding(.horizontal, 10)
            .frame(height: 34)
            .background(MG.controlBackground)
            MGDivider()
            ScrollView {
                MarkdownView(text: plan.plan).padding(14)
            }
            .mgScrollIndicators()
            if doc.plan?.callID == plan.callID {
                MGDivider()
                PlanAnswerControls(doc: doc).padding(12)
            }
        }
    }
}

private struct PlanStatusIcon: View {
    let doc: ChatDocument
    let plan: ChatPlan

    var body: some View {
        if doc.plan?.callID == plan.callID {
            Image(systemName: "hand.raised.fill").foregroundStyle(MG.warning).mgHelp("Waiting for your answer")
        } else if plan.status == "running" {
            MGSpinner().controlSize(.mini)
        } else if plan.detail?.hasPrefix("Approved") == true {
            Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        } else {
            Image(systemName: "arrow.uturn.backward.circle").foregroundStyle(MG.secondaryLabel)
        }
    }
}

enum PlanStatus {
    @MainActor static func line(_ doc: ChatDocument, _ plan: ChatPlan) -> String {
        if doc.plan?.callID == plan.callID { return "Waiting for approval" }
        if plan.status == "running" { return "Not answered" }
        return plan.detail ?? "Not answered"
    }
}

enum PlanMenu {
    @MainActor @MGMenuBuilder
    static func items(_ model: AppModel, _ doc: ChatDocument, _ plan: ChatPlan) -> [MGMenuItem] {
        if let url = model.planFile(doc, plan.callID) {
            MGMenuItem.button("Open in Default App", icon: "arrow.up.forward.app") {
                model.savePlanFiles(doc)
                NSWorkspace.shared.open(url)
            }
            MGMenuItem.button("Reveal in Finder", icon: "folder") {
                model.savePlanFiles(doc)
                NSWorkspace.shared.activateFileViewerSelecting([url])
            }
        }
        MGMenuItem.button("Save As…", icon: "square.and.arrow.down") { model.savePlanAs(plan) }
        MGMenuItem.button("Mention in Message", icon: "text.badge.plus") {
            if let url = model.planFile(doc, plan.callID) {
                model.savePlanFiles(doc)
                model.mentionInDraft(doc, url.path)
            }
        }
    }
}

/// Approve (in a mode) or keep planning (with a note): the answer to a plan waiting for approval.
struct PlanAnswerControls: View {
    let doc: ChatDocument
    @State private var note = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button("Approve, Auto-accept Edits") { doc.answerPlan(.approve(.acceptEdits)) }
                    .buttonStyle(.mg(.prominent)).keyboardShortcut(.defaultAction)
                Button("Approve, Ask Each Time") { doc.answerPlan(.approve(.ask)) }.buttonStyle(.mg(.bordered))
                Spacer(minLength: 0)
            }
            HStack(spacing: 8) {
                TextField("Note for the agent (optional)", text: $note)
                    .textFieldStyle(.roundedBorder).font(.mgBody)
                    .onSubmit { keepPlanning() }
                Button("Keep Planning") { keepPlanning() }.buttonStyle(.mg(.bordered))
            }
        }
    }

    func keepPlanning() {
        let n = note.trimmingCharacters(in: .whitespacesAndNewlines)
        doc.answerPlan(.keepPlanning(n.isEmpty ? nil : n))
    }
}
