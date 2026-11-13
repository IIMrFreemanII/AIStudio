import Agent
import AppKit
import Foundation
import LLMClient
import UniformTypeIdentifiers

/// The agent's plans (its `exit_plan` calls), each also saved as a Markdown file in the chat's folder
/// (`<chat>/plans/2026-11-13 0840 Add a file explorer.md`).
extension AppModel {
    func plansFolder(_ doc: ChatDocument) -> URL {
        chatStore.filesFolder(doc.id).appendingPathComponent("plans", isDirectory: true)
    }

    /// The plans shown in the Plans tab, oldest first: the transcript's, and one waiting for approval that isn't in
    /// it yet.
    func plans(_ doc: ChatDocument) -> [ChatPlan] {
        var out = doc.transcript.plans
        if let p = doc.plan, !out.contains(where: { $0.callID == p.callID }) {
            out.append(ChatPlan(callID: p.callID, plan: p.plan, date: Date(), status: "running", detail: nil))
        }
        return out
    }

    /// The file a plan is saved in.
    func planFile(_ doc: ChatDocument, _ callID: String) -> URL? {
        let all = plans(doc)
        return PlanFiles.names(all)[callID].map { plansFolder(doc).appendingPathComponent($0) }
    }

    /// Writes the plans that have no file yet (a new one, or an older chat's).
    func savePlanFiles(_ doc: ChatDocument) {
        let all = plans(doc)
        guard !all.isEmpty else { return }
        let folder = plansFolder(doc)
        let names = PlanFiles.names(all)
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for p in all {
            guard let name = names[p.callID] else { continue }
            let url = folder.appendingPathComponent(name)
            if !FileManager.default.fileExists(atPath: url.path) {
                try? p.plan.write(to: url, atomically: true, encoding: .utf8)
            }
        }
    }

    /// A plan arrived for approval: save it and show it in the Plans tab (coding chats).
    func planProposed(_ doc: ChatDocument) {
        guard doc.isCoding, let p = doc.plan else { return }
        savePlanFiles(doc)
        doc.selectedPlan = p.callID
        if doc.flowPanelRunID == nil { showSidePanel(doc, .plans) }
    }

    /// A Plan-mode reply that wrote its plan as text instead of calling `exit_plan` (small models do): whether it can be
    /// taken as the plan.
    func canAdoptPlan(_ doc: ChatDocument, _ turn: ChatTurn) -> Bool {
        doc.isCoding && doc.mode == .plan && !doc.isStreaming && turn.role == .assistant && turn.error == nil
            && turn.toolCalls.isEmpty && !turn.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && doc.transcript.turns.last?.id == turn.id
    }

    /// Takes the reply as the plan and approves it, as if the model had called `exit_plan` and the user had approved:
    /// the call and its answer go into the transcript (the Plans tab lists it and saves its file), the mode switches,
    /// and the agent goes on to carry it out.
    func adoptPlan(_ doc: ChatDocument, _ turnID: UUID, mode: PermissionMode) {
        guard let turn = doc.transcript.turn(turnID), canAdoptPlan(doc, turn),
              let data = try? JSONSerialization.data(withJSONObject: ["plan": turn.text]),
              let args = String(data: data, encoding: .utf8) else { return }
        let callID = "plan-" + UUID().uuidString.prefix(8).lowercased()
        doc.transcript.update(turnID) { $0.toolCalls.append(ToolCall(id: callID, name: "exit_plan", arguments: args)) }
        var result = ChatTurn(role: .tool, text: "The user approved the plan. The mode is now \(mode.title): carry out the plan now.",
                              toolCallID: callID)
        result.toolTitle = "Plan ready for approval"
        result.toolDetail = "Approved — \(mode.title)"
        result.toolStatus = "done"
        doc.transcript.append(result)
        doc.mode = mode
        savePlanFiles(doc)
        doc.selectedPlan = callID
        reply(doc)
    }

    func showPlan(_ doc: ChatDocument, _ callID: String) {
        doc.selectedPlan = callID
        doc.flowPanelRunID = nil
        showSidePanel(doc, .plans)
    }

    /// "Save As…": a copy of the plan wherever the user picks.
    func savePlanAs(_ plan: ChatPlan) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = PlanFiles.name(date: plan.date, plan: plan.plan)
        panel.allowedContentTypes = [UTType(filenameExtension: "md") ?? .plainText]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? plan.plan.write(to: url, atomically: true, encoding: .utf8)
    }
}
