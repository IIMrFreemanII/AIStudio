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
    /// In a run's folded steps: every tool is a one-line step until opened (its +/- and status on the line).
    var compact = false

    /// Opened or closed by hand (kept in the chat, so a rebuilt row remembers); nil: the default.
    var open: Bool? { doc.toolOpen[call.id] }
    var status: String { result?.toolStatus ?? "running" }
    var asking: Bool { doc.permission?.call.id == call.id }
    var isCommand: Bool { call.name == "run_command" }
    var isDiff: Bool { call.name == "edit_file" || call.name == "write_file" }
    /// A subagent: while it works, the card lists its steps.
    var isTask: Bool { call.name == "task" }
    /// A string argument of the call.
    func arg(_ key: String) -> String? {
        (try? ToolArgs(json: call.arguments))?.string(key).flatMap { $0.isEmpty ? nil : $0 }
    }
    /// A graph helper change: its list of changes is shown.
    var isGraphEdit: Bool { call.name == "edit_graph" || call.name == "set_prompt" }
    /// The file a call read or wrote, to show it in the Files tab (coding chats, files inside the project).
    var fileTarget: (path: String, line: Int?)? {
        guard doc.isCoding, ["read_file", "edit_file", "write_file"].contains(call.name),
              let args = try? ToolArgs(json: call.arguments), let path = args.string("path"),
              model.relativeProjectPath(doc, path) != nil else { return nil }
        return (path, call.name == "read_file" ? args.int("offset") : nil)
    }

    /// `open_page`'s address as the agent gave it.
    var pageArg: String? { (try? ToolArgs(json: call.arguments))?.string("url") }
    /// The page an `open_page` call showed (what it loaded, else its address; a project path as a file).
    var pageURL: URL? {
        if let d = result?.toolDetail, let u = URL(string: d), u.scheme != nil { return u }
        // The same rules as the tool: an address with a scheme, localhost:port, else a file in the project.
        guard let s = pageArg?.trimmingCharacters(in: .whitespaces), !s.isEmpty else { return nil }
        if s.contains("://") { return URL(string: s) }
        if s.lowercased().hasPrefix("localhost:") || s.hasPrefix("127.0.0.1:") { return URL(string: "http://" + s) }
        return doc.projectURL.map { s.hasPrefix("/") ? URL(fileURLWithPath: s) : $0.appendingPathComponent(s) }
    }

    /// Tools that only look (read, list, search, fetch): finished well, they're a plain line instead of a card, so the
    /// steps that need you (a question, a diff, a command, a failure) stand out.
    static let lookTools: Set<String> = ["read_file", "list_files", "search", "web_fetch", "web_search", "open_page",
                                         "use_skill", "code_graph", "command_output", "list_sessions", "read_terminal"]
    var quiet: Bool { status == "done" && !asking && Self.lookTools.contains(call.name) }

    var body: some View {
        let expanded = open ?? (compact ? asking : (asking || ((isCommand || isTask) && status == "running")
                                                     || (isDiff && status != "denied") || (isGraphEdit && status == "done")))
        let plain = (quiet || (compact && !asking)) && !expanded
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 0) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { doc.toolOpen[call.id] = !expanded }
            } label: {
                HStack(spacing: 8) {
                    statusIcon.font(plain ? .system(size: 10) : nil).opacity(plain ? 0.8 : 1)
                    Text(result?.toolTitle ?? call.name).font(.mgCallout).lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(plain ? MG.secondaryLabel : MG.label)
                        .layoutPriority(1)
                    if isDiff, let d = result?.toolDetail {
                        let (a, r) = counts(d)
                        if a > 0 { Text("+\(a)").font(.mgFootnote).foregroundStyle(MG.success) }
                        if r > 0 { Text("−\(r)").font(.mgFootnote).foregroundStyle(MG.destructive) }
                    }
                    if compact, status == "running", !expanded { runningNote }
                    Spacer(minLength: 4)
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .foregroundStyle(MG.tertiaryLabel).rotationEffect(.degrees(expanded ? 90 : 0))
                }
                .padding(.horizontal, plain ? 2 : 10)
                .frame(height: plain ? 24 : 32)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
                if call.name == "open_page", doc.isCoding, let url = pageURL {
                    Button { model.openWebPage(doc, url) } label: {
                        Image(systemName: "globe")
                    }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.small)
                    .padding(.trailing, 6)
                    .mgHelp("Show the page in the side panel")
                    .accessibilityLabel("Show Page")
                }
                if let target = fileTarget {
                    Button { model.showProjectFile(doc, path: target.path, line: target.line) } label: {
                        Image(systemName: "eye")
                    }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.small)
                    .padding(.trailing, 6)
                    .mgHelp("Show the file in the side panel")
                    .accessibilityLabel("Show File")
                }
            }
            if call.name == "flow", status != "running", let id = doc.flowRunID(for: call, result: result),
               let run = model.flowController(id) {
                MGDivider()
                FlowRunCard(run: run)
            } else if expanded {
                MGDivider()
                details.padding(10)
            }
            if asking, let request = doc.permission {
                MGDivider()
                PermissionCard(doc: doc, request: request).padding(10)
            }
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(plain ? .clear : MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous),
                    color: plain ? .clear : asking ? MG.accent : MG.border)
        .mgTestAnchor(plain ? "tool-line:\(call.id)" : "tool-card:\(call.id)")
    }

    /// On a running step's line: how long it has run and, for a subagent, its newest step.
    @ViewBuilder
    var runningNote: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            let since = result?.date ?? ctx.date
            let clock = RunSummary.clock(ctx.date.timeIntervalSince(since))
            if isTask, let step = (doc.toolOutputs[call.id] ?? "").split(separator: "\n").last(where: { $0.hasPrefix("•") }) {
                Text("\(clock) · \(step.dropFirst().trimmingCharacters(in: .whitespaces))")
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.tail)
                    .mgTestAnchor("task-steps:\(call.id)")
            } else {
                Text(isTask ? "\(clock) · Thinking…" : clock).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .monospacedDigit()
                    .mgTestAnchor(isTask ? "task-steps:\(call.id)" : "running:\(call.id)")
            }
        }
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
        } else if isTask {
            VStack(alignment: .leading, spacing: 6) {
                if let p = arg("prompt") {
                    Text(String(p.prefix(600))).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                let text = status == "running" ? (doc.toolOutputs[call.id] ?? "") : (result?.text ?? "")
                if !text.isEmpty {
                    ScrollView {
                        Text(String(text.suffix(20_000))).font(.mgMono).foregroundStyle(MG.secondaryLabel)
                            .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 220)
                    .mgScrollIndicators()
                } else if status == "running" {
                    Text("Thinking…").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgTestAnchor("task-steps:\(call.id)")
        } else if call.name == "flow", status == "running" {
            // Asking to run (or running): what will run.
            if let script = arg("script") {
                ScrollView {
                    Text(script).font(.mgMono).foregroundStyle(MG.secondaryLabel).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 260)
                .mgScrollIndicators()
                .mgTestAnchor("flow-script:\(call.id)")
            } else if let name = arg("name") {
                Text("The saved flow “\(name)”").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
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

    func counts(_ diff: String) -> (Int, Int) { Self.diffCounts(diff) }

    /// Lines added and removed in a unified diff.
    static func diffCounts(_ diff: String) -> (Int, Int) {
        var a = 0, r = 0
        for line in diff.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("+"), !line.hasPrefix("+++") { a += 1 }
            if line.hasPrefix("-"), !line.hasPrefix("---") { r += 1 }
        }
        return (a, r)
    }
}

/// A unified diff: line numbers, a +/− column, added lines green and removed ones red, ⋯ between hunks.
struct DiffView: View {
    let diff: String
    var maxHeight: CGFloat? = 320

    var body: some View {
        let rows = DiffNumbers.rows(diff)
        let digits = String(rows.compactMap(\.number).max() ?? 0).count
        let numbered = rows.contains { $0.number != nil }
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(rows) { row in
                    if row.gap {
                        Text("⋯").font(.mgMono).foregroundStyle(MG.tertiaryLabel)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.leading, numbered ? CGFloat(digits) * 8 + 14 : 6)
                    } else {
                        let tint = row.sign == "+" ? MG.success : row.sign == "-" ? MG.destructive : MG.secondaryLabel
                        HStack(alignment: .firstTextBaseline, spacing: 0) {
                            if numbered {
                                Text(row.number.map(String.init) ?? "")
                                    .foregroundStyle(MG.tertiaryLabel)
                                    .frame(width: CGFloat(digits) * 8, alignment: .trailing)
                                    .padding(.trailing, 6)
                            }
                            Text(row.sign.map(String.init) ?? " ").foregroundStyle(tint).frame(width: 12, alignment: .leading)
                            Text(row.text.isEmpty ? " " : row.text).foregroundStyle(tint)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .font(.mgMono)
                        .padding(.horizontal, 6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(row.sign == "+" ? MG.success.opacity(0.10) : row.sign == "-" ? MG.destructive.opacity(0.10) : .clear)
                    }
                }
            }
            .textSelection(.enabled)
        }
        .frame(maxHeight: maxHeight ?? .infinity)
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
            if let risk = request.risk {
                Label(risk, systemImage: "exclamationmark.triangle")
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
        case .orchestrate: return "Run this flow?"
        }
    }

    var alwaysTitle: String? {
        if request.risk != nil { return nil }
        switch request.effect {
        case .edit where request.inProject: return "Allow All Edits in This Chat"
        case .execute:
            guard let cmd = request.preview.command else { return nil }
            let ps = AllowRules.prefixes(of: cmd)
            return ps.isEmpty ? nil : "Always Allow " + ps.map { "“\($0)”" }.joined(separator: ", ")
        case .run: return "Allow Runs in This Chat"
        case .network: return request.preview.domain.map { "Always Allow \($0)" }
        default: return nil
        }
    }
}

/// One to-do row: a checkbox, the text, and (for the step in progress) a spinner and the elapsed time.
struct TodoRow: View {
    let todo: ToolContext.Todo
    var live = false
    var started: Date?

    var body: some View {
        HStack(spacing: 6) {
            switch todo.status {
            case "completed":
                Image(systemName: "checkmark.square.fill").foregroundStyle(MG.success)
            case "in_progress":
                if live { ProgressView().controlSize(.mini) }
                else { Image(systemName: "arrow.right.square").foregroundStyle(MG.accent) }
            default:
                Image(systemName: "square").foregroundStyle(MG.secondaryLabel)
            }
            if todo.status == "in_progress" {
                Text(todo.activeForm ?? todo.content).font(.mgFootnote.weight(.medium)).foregroundStyle(MG.label)
                if live, let started {
                    TimelineView(.periodic(from: .now, by: 1)) { ctx in
                        Text(" · \(max(0, Int(ctx.date.timeIntervalSince(started))))s")
                            .font(.mgFootnote).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                    }
                }
            } else {
                Text(todo.content).font(.mgFootnote)
                    .strikethrough(todo.status == "completed")
                    .foregroundStyle(todo.status == "completed" ? MG.secondaryLabel : MG.label)
            }
        }
    }
}

/// The agent's to-do list, above the message field while it has one.
struct TodoListView: View {
    @Bindable var doc: ChatDocument

    var body: some View {
        let todos = doc.todos
        let done = todos.filter { $0.status == "completed" }.count
        let allDone = !todos.isEmpty && done == todos.count
        let current = todos.first { $0.status == "in_progress" }
        VStack(alignment: .leading, spacing: 3) {
            Button { doc.todosCollapsed.toggle() } label: {
                HStack(spacing: 6) {
                    Image(systemName: doc.todosCollapsed ? "chevron.right" : "chevron.down").font(.system(size: 9, weight: .semibold))
                        .frame(width: 10)
                    if allDone {
                        Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
                        Text("All done").font(.mgFootnote.weight(.medium))
                    } else {
                        Text("Tasks").font(.mgFootnote.weight(.medium))
                    }
                    Text("\(done)/\(todos.count)").font(.mgFootnote).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                    if doc.todosCollapsed, let current {
                        Text(current.activeForm ?? current.content).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if !doc.todosCollapsed {
                ForEach(Array(todos.enumerated()), id: \.offset) { _, t in
                    TodoRow(todo: t, live: doc.isStreaming, started: doc.todoStepStarted?.content == t.content ? doc.todoStepStarted?.at : nil)
                }
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
    }
}

/// A `todo_write` call in the transcript: one line, click to see the list it wrote.
struct TodoUpdateRow: View {
    let call: ToolCall
    @State private var open = false

    var body: some View {
        let todos = TodoWriteTool.list(json: call.arguments)
        let done = todos.filter { $0.status == "completed" }.count
        VStack(alignment: .leading, spacing: 3) {
            Button { open.toggle() } label: {
                HStack(spacing: 4) {
                    Image(systemName: "checklist")
                    Text("Updated tasks")
                    if !todos.isEmpty { Text("· \(done)/\(todos.count)").monospacedDigit() }
                    Spacer(minLength: 0)
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open {
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(todos.enumerated()), id: \.offset) { _, t in TodoRow(todo: t) }
                }
                .padding(.leading, 4)
            }
        }
    }
}

/// A coding chat's project folder, branch, side panel and more, in the chat header (the mode is in the composer).
struct CodingHeaderItems: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        // The project is named in the sidebar, the field and the window title; the folder button stays only where
        // there's no branch chip (not a git repository). "Show Project in Finder" is in the ⋯ menu.
        if doc.git == nil, let url = doc.projectURL {
            Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                Label(url.lastPathComponent, systemImage: "folder").lineLimit(1)
            }
            .buttonStyle(.mg(.bordered))
            .mgHelp(url.path + " — the agent works in this folder (click to show it in Finder)")
        }
        if let git = doc.git {
            // The project in one group: branch / PR (the pane's footer shows the PR while Changes is open).
            HStack(spacing: 2) {
                GitBranchMenu(doc: doc, kind: .plain)
                if let pr = git.pr, !(model.sidePanelShown(doc) && doc.sidePanel == .changes) {
                    Text("/").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                    PRPill(pr: pr, kind: .plain)
                }
            }
            .padding(.horizontal, 4)
            .frame(height: 24)
            .background(Capsule().fill(MG.fill))
            .mgTestAnchor("header-project")
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
        if doc.isCoding {
            MGMenuButton(kind: .icon) {
                // Stops a running reply first; without one, the last task carries on in a new session.
                let task = doc.transcript.turns.contains { $0.role == .user && $0.sessionEvent == nil && $0.flowEvent == nil && $0.summary != true }
                MGMenuItem.button(doc.isStreaming ? "Move Task to New Session" : "Continue Task in New Session",
                                  icon: "arrow.up.forward.square", enabled: task) {
                    model.moveTaskToNewSession(doc)
                }
                MGMenuItem.menu("Run Flow", icon: "square.stack.3d.up") {
                    let saved = FlowStore.list(root: doc.projectURL)
                    if saved.isEmpty {
                        MGMenuItem.header("No saved flows (.aistudio/flows)")
                    }
                    for f in saved {
                        MGMenuItem.button(f.name) { model.runSavedFlow(f, in: doc) }
                    }
                }
                if let url = doc.projectURL {
                    MGMenuItem.button("Show Project in Finder", icon: "folder") {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .mgHelp("More: carry the task on in its own session, run a flow, show the project")
            .accessibilityLabel("More")
        }
    }
}

/// The permission mode (Ask / Auto-accept Edits / Plan / Bypass) under the composer's field, as in Claude Code;
/// ⇧⇥ in the field cycles the first three.
struct PermissionModeMenu: View {
    let doc: ChatDocument

    var allowed: String {
        ((doc.rules.edits ? ["edits"] : []) + doc.rules.commandPrefixes.map { "“\($0)”" } + (doc.rules.runs ? ["runs"] : []) + doc.rules.domains)
            .joined(separator: ", ")
    }

    var body: some View {
        if doc.isAgent {
            MGMenuButton(kind: .plain) {
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
                HStack(spacing: 4) {
                    Image(systemName: doc.mode == .bypass ? "exclamationmark.shield" : doc.mode == .plan ? "doc.text.magnifyingglass" : "lock.shield")
                    Text(doc.mode.title).lineLimit(1)
                }
                .font(.mgFootnote)
                .foregroundStyle(doc.mode == .bypass ? MG.warning : doc.mode == .ask ? MG.secondaryLabel : MG.accent)
            }
            .fixedSize()
            .mgHelp((doc.isGraphHelper ? helperDetail : doc.mode.detail) + " (⇧⇥ to switch)")
            .mgTestAnchor("permission-mode")
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

extension PermissionMode {
    /// The next mode for ⇧⇥: Ask → Auto-accept Edits → Plan → Ask (Bypass only from the menu; from it, back to Ask).
    var cycled: PermissionMode {
        switch self {
        case .ask: return .acceptEdits
        case .acceptEdits: return .plan
        case .plan, .bypass: return .ask
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
                        if q.multiSelect { Text("Pick any").font(.mgFootnote).foregroundStyle(MG.secondaryLabel) }
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
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?

    @State private var expanded = false

    var plan: String { (try? ToolArgs(json: call.arguments).string("plan")) ?? "" }
    var asking: Bool { doc.plan?.callID == call.id }
    var status: String { result?.toolStatus ?? "running" }

    /// A long plan's first lines, shown collapsed (nil: short enough to show whole). Only these are laid out, so a
    /// long plan doesn't slow the chat down.
    static func collapsed(_ plan: String) -> (text: String, hidden: Int)? {
        CollapsedText.of(plan, overLines: 24, overChars: 2400, showLines: 14, showChars: 1400).map { ($0.text, $0.hidden) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Plan", systemImage: "doc.text.magnifyingglass")
                    .font(.mgFootnote.weight(.semibold)).foregroundStyle(MG.accent)
                Spacer(minLength: 0)
                if doc.isCoding {
                    Button { model.showPlan(doc, call.id) } label: {
                        Label("Open in Plans", systemImage: "sidebar.right").font(.mgFootnote)
                    }
                    .buttonStyle(.mg(.borderless))
                    .mgHelp("Show the plan in the side panel (it's saved with the chat)")
                }
            }
            let short = Self.collapsed(plan)
            if let short, !expanded {
                MarkdownView(text: short.text)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(alignment: .bottom) {
                        LinearGradient(colors: [MG.controlBackground.opacity(0), MG.controlBackground],
                                       startPoint: .top, endPoint: .bottom)
                            .frame(height: 44)
                            .allowsHitTesting(false)
                    }
            } else {
                MarkdownView(text: plan)
            }
            if let short {
                Button {
                    withAnimation(.easeOut(duration: 0.15)) { expanded.toggle() }
                } label: {
                    Label(expanded ? "Show less" : "Show full plan (\(short.hidden) more lines)",
                          systemImage: expanded ? "chevron.up" : "chevron.down")
                        .font(.mgFootnote)
                }
                .buttonStyle(.mg(.borderless))
                .mgTestAnchor("plan-expand:\(call.id)")
            }
            MGDivider()
            if asking {
                PlanAnswerControls(doc: doc)
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
            Text("Changes made by commands aren't undone.").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
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
