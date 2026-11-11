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
        case .run:
            let title = (try? ToolArgs(json: request.call.arguments))?.string("title")
            switch request.call.name {
            case "start_session": return "Start a new session\(title.map { " “\($0)”" } ?? "")?"
            case "hand_off": return "Hand this task off to a new session?"
            default: return "Run the workflow?"
            }
        case .network: return "Read \(request.preview.domain ?? "this web page")?"
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
        case .network: return request.preview.domain.map { "Always Allow \($0)" }
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
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var allowed: String {
        ((doc.rules.edits ? ["edits"] : []) + doc.rules.commandPrefixes.map { "“\($0)”" } + (doc.rules.runs ? ["runs"] : []) + doc.rules.domains)
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
        if let w = doc.transcript.worktree {
            MGMenuButton(kind: .bordered) {
                MGMenuItem.button("Merge into \(w.base)…", icon: "arrow.triangle.merge") { model.mergeWorktree(doc) }
                if doc.canCreatePR {
                    MGMenuItem.button("Create Pull Request…", icon: "arrow.up.right.square") { model.createPullRequest(doc) }
                }
                MGMenuItem.button("Show Worktree in Finder", icon: "folder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: w.path)])
                }
                MGMenuItem.divider
                MGMenuItem.button("Remove Worktree…", icon: "trash") { model.removeWorktree(doc) }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "arrow.triangle.branch").foregroundStyle(MG.secondaryLabel)
                    Text(w.branch).lineLimit(1).truncationMode(.middle)
                    if let s = doc.worktreeStatus {
                        if s.ahead > 0 { Text("+\(s.ahead)").foregroundStyle(MG.success) }
                        if s.dirty > 0 { Text("•\(s.dirty)").foregroundStyle(MG.warning) }
                    }
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp("Its own git worktree on \(w.branch), cut from \(w.base)")
        }
        if doc.isCoding, doc.isStreaming {
            Button { model.moveTaskToNewSession(doc) } label: {
                Label("New Session", systemImage: "arrow.up.forward.square").lineLimit(1)
            }
            .buttonStyle(.mg(.bordered))
            .mgHelp("Move Task to New Session: stop here and carry on in a separate session")
        }
        if !doc.runningJobs.isEmpty {
            MGMenuButton(kind: .bordered) {
                MGMenuItem.header("Running in the background")
                for id in doc.runningJobs {
                    let command = doc.jobs.job(id)?.command ?? ""
                    MGMenuItem.button("\(id) — \(command.prefix(40))", icon: "stop.circle") { doc.jobs.job(id)?.stop() }
                }
                MGMenuItem.divider
                MGMenuItem.button("Stop All") { doc.jobs.stopAll() }
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: "terminal").foregroundStyle(MG.secondaryLabel)
                    Text("\(doc.runningJobs.count) running")
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp("Commands the agent left running (click one to stop it)")
        }
        if doc.isAgent {
            MGMenuButton(kind: .bordered) {
                for m in PermissionMode.allCases {
                    MGMenuItem.button(m.title, checked: doc.mode == m) { doc.mode = m }
                }
                if doc.rules.edits || !doc.rules.commandPrefixes.isEmpty || doc.rules.runs || !doc.rules.domains.isEmpty {
                    MGMenuItem.divider
                    MGMenuItem.header("Always allowed: " + allowed)
                    MGMenuItem.button("Forget Always-Allow Answers") {
                        doc.transcript.allowEdits = nil
                        doc.transcript.allowCommands = nil
                        doc.transcript.allowRuns = nil
                        doc.transcript.allowDomains = nil
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

/// An `ask_user` call: while it waits, the questions with their options (radio buttons or checkboxes) and an
/// "Other" field; once answered, one line per question ("Database: SQLite"), which opens to the questions.
struct QuestionCard: View {
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?
    @State private var picked: [Int: Set<String>] = [:]
    @State private var other: [Int: String] = [:]
    @State private var open = false

    var questions: [UserQuestion] {
        (try? UserQuestion.parse(ToolArgs(json: call.arguments))) ?? []
    }
    var asking: Bool { doc.question?.callID == call.id }
    var status: String { result?.toolStatus ?? "running" }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if asking {
                form
            } else {
                summary
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: asking ? MG.accent : MG.border)
    }

    var form: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(Array(questions.enumerated()), id: \.offset) { i, q in
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        if let h = q.header, !h.isEmpty {
                            Text(h.uppercased()).font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.accent)
                        }
                        if q.multiSelect { Text("Pick any").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel) }
                    }
                    Text(q.question).font(.mgCallout.weight(.medium)).fixedSize(horizontal: false, vertical: true)
                    ForEach(q.options, id: \.self) { o in
                        option(o, question: i, multi: q.multiSelect)
                    }
                    HStack(spacing: 8) {
                        Image(systemName: "pencil").foregroundStyle(MG.tertiaryLabel).frame(width: 16)
                        TextField("Other…", text: Binding(get: { other[i] ?? "" }, set: { other[i] = $0 }))
                            .textFieldStyle(.roundedBorder).font(.mgBody)
                            .onSubmit { send() }
                    }
                }
            }
            HStack(spacing: 8) {
                Spacer()
                Button("Skip") { doc.answerQuestion(nil) }
                    .buttonStyle(.mg(.bordered)).keyboardShortcut(.cancelAction)
                Button("Send") { send() }
                    .buttonStyle(.mg(.prominent)).disabled(answers.allSatisfy(\.isEmpty))
            }
        }
        .padding(12)
    }

    func option(_ o: UserQuestion.Option, question i: Int, multi: Bool) -> some View {
        let on = picked[i]?.contains(o.label) == true
        return Button {
            var set = picked[i] ?? []
            if multi { if on { set.remove(o.label) } else { set.insert(o.label) } } else { set = on ? [] : [o.label] }
            picked[i] = set
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: multi ? (on ? "checkmark.square.fill" : "square") : (on ? "largecircle.fill.circle" : "circle"))
                    .foregroundStyle(on ? MG.accent : MG.secondaryLabel).frame(width: 16)
                VStack(alignment: .leading, spacing: 1) {
                    Text(o.label).font(.mgBody).foregroundStyle(MG.label)
                    if let d = o.description {
                        Text(d).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 5).padding(.horizontal, 8)
            .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                .fill(on ? MG.accent.opacity(0.10) : Color.clear))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    var answers: [UserAnswer] {
        questions.indices.map { i in
            let order = questions[i].options.map(\.label).filter { picked[i]?.contains($0) == true }
            return UserAnswer(picked: order, other: other[i])
        }
    }

    func send() {
        let a = answers
        guard !a.allSatisfy(\.isEmpty) else { return }
        doc.answerQuestion(a)
    }

    var summary: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { open.toggle() }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: 8) {
                    Group {
                        switch status {
                        case "running": MGSpinner().controlSize(.mini)
                        case "done": Image(systemName: "questionmark.bubble.fill").foregroundStyle(MG.accent)
                        default: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.secondaryLabel)
                        }
                    }
                    Text(line).font(.mgCallout).lineLimit(open ? nil : 1).truncationMode(.tail)
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .foregroundStyle(MG.tertiaryLabel).rotationEffect(.degrees(open ? 90 : 0))
                }
                .padding(.horizontal, 10).padding(.vertical, 8)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open {
                MGDivider()
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(questions.enumerated()), id: \.offset) { _, q in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(q.question).font(.mgFootnote.weight(.medium))
                            Text(q.options.map(\.label).joined(separator: " · ")).font(.mgFootnote)
                                .foregroundStyle(MG.secondaryLabel)
                        }
                    }
                    if let d = result?.toolDetail, status == "done" {
                        Text(d).font(.mgFootnote).foregroundStyle(MG.label).textSelection(.enabled)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    var line: String {
        switch status {
        case "running": return questions.first?.question ?? "Asking…"
        case "done":
            let d = result?.toolDetail ?? ""
            return d == "Skipped" ? "Skipped: \(questions.first?.question ?? "")" : d.replacingOccurrences(of: "\n", with: " · ")
        default: return result?.text ?? "Not answered"
        }
    }
}

/// An `exit_plan` call: the plan, and while it waits for approval, the buttons that start it (in a mode) or keep
/// planning (with a note).
struct PlanCard: View {
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?
    @State private var note = ""

    var plan: String { (try? ToolArgs(json: call.arguments).string("plan")) ?? "" }
    var asking: Bool { doc.plan?.callID == call.id }
    var status: String { result?.toolStatus ?? "running" }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Plan", systemImage: "doc.text.magnifyingglass")
                .font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.accent)
            MarkdownView(text: plan)
            MGDivider()
            if asking {
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
            } else {
                HStack(spacing: 6) {
                    if status == "running" { MGSpinner().controlSize(.mini) }
                    Text(statusLine).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: asking ? MG.accent : MG.border)
    }

    var statusLine: String {
        if status == "running" { return "Waiting…" }
        return result?.toolDetail ?? result?.text ?? "Not answered"
    }

    func keepPlanning() {
        let n = note.trimmingCharacters(in: .whitespacesAndNewlines)
        doc.answerPlan(.keepPlanning(n.isEmpty ? nil : n))
    }
}

/// "Rewind to before this message?": the files that would be restored, and what to put back.
struct RewindCard: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var message: String {
        let text = doc.rewindTarget.flatMap { doc.transcript.turn($0)?.text } ?? ""
        let one = text.replacingOccurrences(of: "\n", with: " ")
        return one.count > 60 ? String(one.prefix(60)) + "…" : one
    }

    var body: some View {
        let files = doc.rewindFiles
        VStack(alignment: .leading, spacing: 8) {
            Text("Rewind to before “\(message)”?").font(.mgCallout.weight(.medium)).fixedSize(horizontal: false, vertical: true)
            if let files {
                if files.isEmpty {
                    Text("No file changes to undo.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                } else {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("These files go back as they were:").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                        ForEach(files.prefix(12), id: \.path) { f in
                            Text(relative(f)).font(.mgMono).lineLimit(1).truncationMode(.middle)
                        }
                        if files.count > 12 {
                            Text("and \(files.count - 12) more").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                        }
                    }
                }
            } else {
                Text("Loading…").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            Text("Changes made by commands aren't undone.").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
            HStack(spacing: 8) {
                Button("Rewind Files and Chat") { model.rewind(doc, files: true, chat: true) }
                    .buttonStyle(.mg(.prominent)).keyboardShortcut(.defaultAction)
                Button("Chat Only") { model.rewind(doc, files: false, chat: true) }.buttonStyle(.mg(.bordered))
                Button("Files Only") { model.rewind(doc, files: true, chat: false) }
                    .buttonStyle(.mg(.bordered)).disabled(files?.isEmpty ?? true)
                Spacer(minLength: 0)
                Button("Cancel") { doc.rewindTarget = nil; doc.rewindFiles = nil }
                    .buttonStyle(.mg(.bordered)).keyboardShortcut(.cancelAction)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.accent)
    }

    func relative(_ url: URL) -> String {
        guard let root = doc.projectURL?.standardizedFileURL.path, url.path.hasPrefix(root + "/") else { return url.path }
        return String(url.path.dropFirst(root.count + 1))
    }
}
