import Agent
import AppKit
import LLMClient
import SwiftUI

/// One tool call in a coding chat: what it does, how it went, its diff or command output, and — while it waits —
/// the permission question.
struct ToolCard: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let call: ToolCall
    /// The tool turn with its result (nil until the call starts).
    let result: ChatTurn?
    @State private var open: Bool?

    var status: String { result?.toolStatus ?? "running" }
    var asking: Bool { doc.permission?.call.id == call.id }
    var isCommand: Bool { call.name == "run_command" }
    var isDiff: Bool { call.name == "edit_file" || call.name == "write_file" }
    /// A graph helper change: its list of changes is shown.
    var isGraphEdit: Bool { call.name == "edit_graph" || call.name == "set_prompt" }

    var body: some View {
        let expanded = open ?? (asking || (isCommand && status == "running") || (isDiff && status != "denied")
                                || (isGraphEdit && status == "done"))
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { open = !expanded }
            } label: {
                HStack(spacing: 8) {
                    statusIcon
                    Text(result?.toolTitle ?? call.name).font(.mgCallout).lineLimit(1).truncationMode(.middle)
                    if isDiff, let d = result?.toolDetail {
                        let (a, r) = counts(d)
                        if a > 0 { Text("+\(a)").font(.mgFootnote).foregroundStyle(MG.success) }
                        if r > 0 { Text("−\(r)").font(.mgFootnote).foregroundStyle(MG.destructive) }
                    }
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .foregroundStyle(MG.tertiaryLabel).rotationEffect(.degrees(expanded ? 90 : 0))
                }
                .padding(.horizontal, 10)
                .frame(height: 32)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if expanded {
                MGDivider()
                details.padding(10)
            }
            if asking, let request = doc.permission {
                MGDivider()
                PermissionCard(doc: doc, request: request).padding(10)
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous),
                    color: asking ? MG.accent : MG.border)
    }

    @ViewBuilder
    var statusIcon: some View {
        if asking {
            Image(systemName: "hand.raised.fill").foregroundStyle(MG.warning).mgHelp("Waiting for your answer")
        } else {
            switch status {
            case "running": MGSpinner().controlSize(.mini)
            case "done": Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
            case "denied": Image(systemName: "nosign").foregroundStyle(MG.secondaryLabel).mgHelp("Not allowed")
            default: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.destructive)
            }
        }
    }

    @ViewBuilder
    var details: some View {
        if isDiff, let d = result?.toolDetail, !d.isEmpty {
            DiffView(diff: d)
        } else if isCommand {
            VStack(alignment: .leading, spacing: 6) {
                Text("$ " + (result?.toolDetail ?? "")).font(.mgMono).textSelection(.enabled)
                let live = doc.toolOutputs[call.id]
                let text = status == "running" ? (live ?? "") : (result?.text ?? "")
                if !text.isEmpty {
                    ScrollView {
                        Text(String(text.suffix(20_000))).font(.mgMono).foregroundStyle(MG.secondaryLabel)
                            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 220)
                    .mgScrollIndicators()
                }
            }
        } else if isGraphEdit, status == "done", let t = result?.text {
            VStack(alignment: .leading, spacing: 3) {
                ForEach(Array(t.split(separator: "\n").enumerated()), id: \.offset) { _, line in
                    let l = String(line)
                    let note = l.hasPrefix("Not runnable") || l.hasPrefix("The workflow is valid") || l.hasPrefix("ids:")
                    Label(l, systemImage: note ? (l.hasPrefix("Not") ? "exclamationmark.triangle" : "checkmark") : "circle.fill")
                        .labelStyle(GraphChangeLabelStyle(note: note, warn: l.hasPrefix("Not")))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else if call.name == "todo_write", let d = result?.toolDetail {
            Text(d).font(.mgFootnote).textSelection(.enabled)
        } else {
            Text(String((result?.text ?? call.arguments).prefix(4000)))
                .font(.mgMono).foregroundStyle(MG.secondaryLabel).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        if status == "failed" || status == "denied", let t = result?.text, !isCommand {
            Text(t).font(.mgFootnote).foregroundStyle(status == "denied" ? MG.secondaryLabel : MG.destructive)
                .padding(.top, 4)
        }
    }

    func counts(_ diff: String) -> (Int, Int) {
        var a = 0, r = 0
        for line in diff.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("+"), !line.hasPrefix("+++") { a += 1 }
            if line.hasPrefix("-"), !line.hasPrefix("---") { r += 1 }
        }
        return (a, r)
    }
}

/// A unified diff, added lines green and removed ones red.
struct DiffView: View {
    let diff: String

    var body: some View {
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(diff.split(separator: "\n", omittingEmptySubsequences: false).enumerated()), id: \.offset) { _, line in
                    let s = String(line)
                    let added = s.hasPrefix("+"), removed = s.hasPrefix("-"), hunk = s.hasPrefix("@@")
                    Text(hunk ? "⋯" : s.isEmpty ? " " : s)
                        .font(.mgMono)
                        .foregroundStyle(hunk ? MG.tertiaryLabel : added ? MG.success : removed ? MG.destructive : MG.secondaryLabel)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, 6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(added ? MG.success.opacity(0.10) : removed ? MG.destructive.opacity(0.10) : .clear)
                }
            }
            .textSelection(.enabled)
        }
        .frame(maxHeight: 320)
        .mgScrollIndicators()
    }
}

/// "Allow this edit?" with Allow, Always allow, Deny.
struct PermissionCard: View {
    let doc: ChatDocument
    let request: PermissionRequest

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(question).font(.mgCallout.weight(.medium))
            if !request.inProject {
                Label("This is outside the project folder.", systemImage: "exclamationmark.triangle")
                    .font(.mgFootnote).foregroundStyle(MG.warning)
            }
            HStack(spacing: 8) {
                Button("Allow") { doc.answer(.allow) }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                if let always = alwaysTitle {
                    Button(always) { doc.answer(.allowAlways) }.buttonStyle(.mg(.bordered))
                }
                Button("Deny") { doc.answer(.deny) }
                    .buttonStyle(.mg(.bordered))
                    .keyboardShortcut(.cancelAction)
                Spacer()
            }
        }
    }

    var question: String {
        switch request.effect {
        case .read: return "Allow reading outside the project?"
        case .edit: return "Allow this change to \(request.preview.paths.first.map(\.lastPathComponent) ?? "the file")?"
        case .execute: return "Run this command?"
        case .graphEdit: return "Change the workflow?"
        case .run: return "Run the workflow?"
        }
    }

    var alwaysTitle: String? {
        switch request.effect {
        case .edit where request.inProject: return "Allow All Edits in This Chat"
        case .execute:
            guard let cmd = request.preview.command else { return nil }
            let p = AllowRules.prefix(of: cmd)
            return p.isEmpty ? nil : "Always Allow “\(p)”"
        case .run: return "Allow Runs in This Chat"
        default: return nil
        }
    }
}

/// The agent's to-do list, above the message field while it has one.
struct TodoListView: View {
    let todos: [ToolContext.Todo]

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(todos.enumerated()), id: \.offset) { _, t in
                HStack(spacing: 6) {
                    Image(systemName: t.status == "completed" ? "checkmark.square.fill"
                          : t.status == "in_progress" ? "arrow.right.square" : "square")
                        .foregroundStyle(t.status == "completed" ? MG.success : t.status == "in_progress" ? MG.accent : MG.secondaryLabel)
                    Text(t.content).font(.mgFootnote)
                        .strikethrough(t.status == "completed")
                        .foregroundStyle(t.status == "completed" ? MG.secondaryLabel : MG.label)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
    }
}

/// A coding chat's project folder and permission mode, in the chat header.
struct CodingHeaderItems: View {
    let doc: ChatDocument

    var allowed: String {
        ((doc.rules.edits ? ["edits"] : []) + doc.rules.commandPrefixes.map { "“\($0)”" } + (doc.rules.runs ? ["runs"] : []))
            .joined(separator: ", ")
    }

    var body: some View {
        if let url = doc.projectURL {
            Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                Label(url.lastPathComponent, systemImage: "folder").lineLimit(1)
            }
            .buttonStyle(.mg(.bordered))
            .mgHelp(url.path + " — the agent works in this folder (click to show it in Finder)")
        }
        if doc.isAgent {
            MGMenuButton(kind: .bordered) {
                for m in PermissionMode.allCases {
                    MGMenuItem.button(m.title, checked: doc.mode == m) { doc.mode = m }
                }
                if doc.rules.edits || !doc.rules.commandPrefixes.isEmpty || doc.rules.runs {
                    MGMenuItem.divider
                    MGMenuItem.header("Always allowed: " + allowed)
                    MGMenuItem.button("Forget Always-Allow Answers") {
                        doc.transcript.allowEdits = nil
                        doc.transcript.allowCommands = nil
                        doc.transcript.allowRuns = nil
                    }
                }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: doc.mode == .bypass ? "exclamationmark.shield" : doc.mode == .plan ? "doc.text.magnifyingglass" : "lock.shield")
                        .foregroundStyle(doc.mode == .bypass ? MG.warning : MG.secondaryLabel)
                    Text(doc.mode.title)
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp(doc.isGraphHelper ? helperDetail : doc.mode.detail)
        }
    }

    var helperDetail: String {
        switch doc.mode {
        case .ask, .acceptEdits: return "Changes the workflow without asking (one undo step per reply); asks before each run."
        case .plan: return "Only looks at the workflow and explains; edits and runs are refused."
        case .bypass: return "Changes and runs the workflow without asking."
        }
    }
}

/// A line of a graph edit's result: a change (dot) or a note (valid / not runnable).
struct GraphChangeLabelStyle: LabelStyle {
    var note: Bool
    var warn: Bool

    func makeBody(configuration: Configuration) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            configuration.icon
                .font(.system(size: note ? 10 : 5))
                .foregroundStyle(warn ? MG.warning : note ? MG.success : MG.accent)
                .frame(width: 12)
            configuration.title
                .font(.mgFootnote)
                .foregroundStyle(note ? MG.secondaryLabel : MG.label)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// After a helper reply that changed the workflow: what changed, and Undo (the reply's one undo step).
struct HelperUndoBar: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        if !doc.isStreaming, let edit = doc.lastHelperEdit, let wf = model.helperWorkflow(doc), wf.graph == edit.after {
            HStack(spacing: 8) {
                Image(systemName: "wand.and.sparkles").foregroundStyle(MG.accent)
                Text(edit.changes.count == 1 ? "1 change to \(wf.name)" : "\(edit.changes.count) changes to \(wf.name)")
                    .font(.mgFootnote).lineLimit(1)
                    .mgHelp(edit.changes.joined(separator: "\n"))
                Spacer(minLength: 4)
                Button("Undo") { model.undoHelperEdit(doc) }
                    .buttonStyle(.mg(.bordered))
                    .controlSize(.small)
                    .mgHelp("Put the workflow back as it was before this reply (also Edit › Undo Graph Helper)")
            }
            .padding(.horizontal, 12).padding(.vertical, 6)
            .background(MG.accent.opacity(0.08))
        }
    }
}
