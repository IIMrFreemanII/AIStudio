import Agent
import AppKit
import LLMClient
import SwiftUI
import UniformTypeIdentifiers

/// A chat tab: the model and its GPU at the top, the conversation, and the message field.
struct ChatView: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    /// Docked next to the graph (the graph helper): a narrower, two-row header.
    var compact = false
    /// False while kept built but hidden behind the chat in front (see `AppModel.keptChats`).
    var inFront = true
    @AppStorage("changesWidth") private var changesWidth = 440.0  // the side panel's width
    @AppStorage("flowPanelWidth") private var flowWidth = 460.0

    /// Coding chats: a suggested ```bash command's Run button runs it in the chat's terminal.
    private var runCommand: (@MainActor (String) -> Void)? {
        guard doc.isCoding else { return nil }
        let model = model, doc = doc
        return { command in model.runInTerminal(doc, command) }
    }

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                ChatHeader(doc: doc, compact: compact)
                MGDivider()
                if doc.isAgent && doc.mode == .bypass {
                    Label(doc.isGraphHelper ? "Bypass Permissions: the helper edits and runs the workflow without asking."
                          : "Bypass Permissions: the agent edits files and runs commands in \(doc.projectURL?.lastPathComponent ?? "the project") without asking.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.mgFootnote).foregroundStyle(MG.warning)
                        .padding(.horizontal, 14).padding(.vertical, 6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(MG.warning.opacity(0.12))
                }
                if doc.isGraphHelper { HelperUndoBar(doc: doc) }
                if doc.transcript.turns.isEmpty {
                    ChatEmptyState(doc: doc)
                } else {
                    ChatTranscriptView(doc: doc)
                }
                ChatComposer(doc: doc, inFront: inFront)
            }
            .frame(maxWidth: .infinity)
            if doc.isCoding && !compact, let runID = doc.flowPanelRunID, let run = model.flowController(runID) {
                MGDivider(vertical: true, gap: true)
                FlowPanel(doc: doc, run: run)
                    .frame(width: flowWidth)
                    .overlay(alignment: .leading) { MGResizeHandle(width: $flowWidth, range: 380...720).offset(x: -3) }
            } else if doc.isCoding && doc.sidePanel != nil && !compact {
                MGDivider(vertical: true, gap: true)
                CodingSidePanel(doc: doc)
                    .frame(width: changesWidth)
                    .overlay(alignment: .leading) { MGResizeHandle(width: $changesWidth, range: 360...900).offset(x: -3) }
            }
        }
        .background(MG.windowBackground)
        .environment(\.openURL, OpenURLAction { url in
            model.openChatLink(url, in: doc) ? .handled : .systemAction
        })
        .environment(\.chatFiles, ChatFilesContext(index: doc.isCoding ? doc.fileIndex : nil,
                                                   folder: model.chatStore.filesFolder(doc.id), root: doc.projectURL))
        .environment(\.chatRunCommand, runCommand)
        .onAppear { ChatQuoting.visibleChats += 1 }
        .onDisappear { ChatQuoting.visibleChats -= 1 }
        .onReceive(NotificationCenter.default.publisher(for: ChatQuoting.notification)) { note in
            guard let text = note.object as? String, inFront || compact else { return }
            // Several chats may be on screen (a tab and a docked helper): the one holding the passage takes it, else
            // the one in front.
            let holds = doc.transcript.turns.contains { $0.text.contains(text) || ($0.toolDetail ?? "").contains(text) }
            if holds || (model.activeChat === doc && !compact) { model.quoteSelection(doc, text) }
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in model.attach([url], to: doc) }
                }
            }
            return !providers.isEmpty
        }
        .task { await model.llm.refreshAll() }
        .task(id: doc.id) {
            model.refreshGitSoon(doc)
            model.refreshCodeGraph(doc)
            model.refreshProjectFiles(doc)  // also for links to files in replies
        }
        .onChange(of: inFront) { _, on in
            // Back in front (it was kept built): as a fresh view would, look at the repository again.
            if on {
                model.refreshGitSoon(doc)
                model.refreshCodeGraph(doc)
                model.refreshProjectFiles(doc)
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in
            guard inFront || compact else { return }
            model.refreshGitSoon(doc)
            model.refreshCodeGraph(doc)  // picks up edits made in other apps
            model.refreshProjectFiles(doc)
            doc.filesRevision += 1
        }
        .task(id: doc.git?.pr.map { "\($0.number)-\($0.pending)" }) {
            // Checks are running: look again every 30 seconds.
            while (doc.git?.pr?.pending ?? 0) > 0 {
                try? await Task.sleep(nanoseconds: 30_000_000_000)
                if Task.isCancelled { break }
                await model.refreshGit(doc)
            }
        }
        .mgSheet("create-pr-\(doc.id)", item: Binding(get: { doc.prSheet }, set: { doc.prSheet = $0 })) { _ in
            CreatePRSheet(doc: doc).environment(model)
        }
    }
}

// MARK: header

struct ChatHeader: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    var compact = false

    var body: some View {
        if compact {
            twoRows
        } else {
            // One row when everything fits at full width; two rows (model on top) rather than squeezing the chips to
            // "…" — e.g. with the Changes pane open.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    modelMenu
                    CodingHeaderItems(doc: doc)
                    Spacer(minLength: 8)
                    cost
                }
                .padding(.horizontal, 14)
                .frame(height: 44)
                twoRows
            }
        }
    }

    var twoRows: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                modelMenu
                Spacer(minLength: 4)
            }
            HStack(spacing: 8) {
                CodingHeaderItems(doc: doc)
                Spacer(minLength: 4)
                cost
            }
        }
        .padding(.horizontal, compact ? 10 : 14)
        .padding(.vertical, 8)
    }

    /// The model and its GPU in one control: name and engine state in the label; switching models, starting and
    /// shutting down in the menu.
    @ViewBuilder var modelMenu: some View {
        let llm = model.llm
        let key = doc.modelKey
        let m = key.flatMap { llm.model($0) }
        let chip = key.map { k -> EngineChip in
            let st = engineStatus(k)
            let since = doc.isStreaming ? doc.requestStarted : st?.bootSeconds.map { Date().addingTimeInterval(-$0) }
            return EngineChip(status: st, model: m, streaming: doc.isStreaming, plain: true, since: since)
        }
            MGMenuButton(kind: .bordered) {
                if llm.models.isEmpty {
                    MGMenuItem.header("No models yet")
                }
                for s in llm.servers {
                    let models = (llm.listings[s.id]?.models ?? []).filter { $0.installed != false }
                    if !models.isEmpty {
                        if llm.servers.count > 1 { MGMenuItem.header(s.name) }
                        for mm in models {
                            MGMenuItem.button("\(mm.name)  ·  \(mm.gpuLabel)",
                                              checked: key == LLMKey(serverID: s.id, modelID: mm.id)) {
                                model.setModel(LLMKey(serverID: s.id, modelID: mm.id), for: doc)
                            }
                        }
                    }
                }
                if let key {
                    MGMenuItem.divider
                    // The live status (as the chip shows it): the polled one can still say cold while a reply waits
                    // for a GPU, and then the menu offered only Start.
                    let live = engineStatus(key)
                    if let failure = llm.actionErrors[key] { MGMenuItem.header(failure) }
                    if let busy = llm.busyLabel(key), llm.pending[key] != .starting {
                        MGMenuItem.button(busy, icon: "power", enabled: false) {}
                    } else if live?.isStopping == true {
                        MGMenuItem.button("Shutting the GPU Down…", icon: "power", enabled: false) {}
                    } else if live?.isStarting == true {
                        MGMenuItem.button("Cancel the Start", icon: "xmark.circle") { Task { await llm.stop(key) } }
                    } else if live?.isActive == true {
                        MGMenuItem.button("Shut the GPU Down Now", icon: "power") { Task { await llm.stop(key) } }
                    } else {
                        MGMenuItem.button("Start the Model Now", icon: "play") { Task { await llm.warm(key) } }
                    }
                }
                MGMenuItem.divider
                MGMenuItem.button("Add a Model from a Link…", icon: "link") { model.llmAddRequest = LLMAddRequest() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "cpu").foregroundStyle(MG.secondaryLabel)
                    Text(m?.name ?? doc.transcript.modelID ?? "Choose a model").lineLimit(1)
                    if let chip {
                        Rectangle().fill(MG.separator).frame(width: 1, height: 14)
                        chip
                    }
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp(chip.map { "The model this chat talks to. " + $0.helpText } ?? "The model this chat talks to")
    }

    /// The engine's state: the reply stream's status lines while it streams, else the server's.
    func engineStatus(_ key: LLMKey) -> LLMStatus? {
        let llm = model.llm
        // Shutting down wins over the reply stream's last line ("ready"), which goes on until the stream ends.
        if let s = llm.status(key), s.isStopping { return s }
        return doc.engine.map { e in
            var st = LLMStatus(state: e.state, phase: e.phase, progress: e.progress)
            st.coldStart = llm.status(key)?.coldStart  // the stream's status lines don't carry it
            return st
        } ?? llm.status(key)
    }

    @ViewBuilder var cost: some View {
            if doc.transcript.costUSD > 0 {
                Text(String(format: "$%.2f", doc.transcript.costUSD))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).monospacedDigit()
                    .mgHelp(String(format: "This chat: $%.3f of GPU time for its replies. ", doc.transcript.costUSD)
                            + "The model's figure is the GPU's whole warm period, for every chat.")
            }
    }
}

/// The model's engine at a glance: cold, waiting for a GPU, loading (with progress), ready (with price), failed.
struct EngineChip: View {
    let status: LLMStatus?
    let model: LLMModel?
    var streaming = false
    /// Inside another control's label: no capsule of its own and no tooltip (the control shows `helpText`).
    var plain = false
    /// When the wait began (the reply's request, else the server's boot time): starting and loading show the time
    /// so far, ticking.
    var since: Date? = nil

    var helpText: String { help(status?.state ?? "cold") }

    var body: some View {
        if plain { content } else {
            content
                .padding(.horizontal, 8)
                .frame(height: 24)
                .background(Capsule().fill(MG.fill))
                .mgHelp(helpText)
        }
    }

    @ViewBuilder var content: some View {
        let s = status?.state ?? "cold"
        if (s == "starting" || s == "loading"), since != nil {
            TimelineView(.periodic(from: .now, by: 1)) { ctx in row(s, now: ctx.date) }
        } else {
            row(s, now: Date())
        }
    }

    func row(_ s: String, now: Date) -> some View {
        HStack(spacing: 6) {
            switch s {
            case "starting", "loading", "stopping":
                MGSpinner().controlSize(.mini)
            case "ready":
                Circle().fill(MG.success).frame(width: 7, height: 7)
            case "failed":
                Circle().fill(MG.destructive).frame(width: 7, height: 7)
            default:
                Circle().fill(MG.tertiaryLabel).frame(width: 7, height: 7)
            }
            Text(label(s, now: now)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).monospacedDigit()
            if s == "loading", let p = status?.progress {
                MGProgress(value: p).frame(width: 60, height: 4)
            }
        }
    }

    func label(_ s: String, now: Date = Date()) -> String {
        let gpu = status?.count.map { c in c > 1 ? "\(c)× \(status?.gpu ?? "")" : status?.gpu ?? "" } ?? model?.gpuLabel ?? ""
        // "1:20" so far while it starts.
        let waited = since.map { d -> String in
            let t = max(0, Int(now.timeIntervalSince(d)))
            return " · " + String(format: "%d:%02d", t / 60, t % 60)
        } ?? ""
        switch s {
        case "starting": return (status?.phase ?? "Waiting for a GPU") + waited + usually
        case "loading":
            if let p = status?.progress, p < 1 { return "\(status?.phase ?? "Loading") \(Int(p * 100))%" + waited + usually }
            return (status?.phase ?? "Loading") + waited + usually
        case "ready":
            var parts = [streaming ? "Generating" : "Ready", gpu]
            if let c = status?.costUSD, c > 0 { parts.append(String(format: "$%.2f so far", c)) }
            return parts.filter { !$0.isEmpty }.joined(separator: " · ")
        case "failed": return "Failed to start"
        case "stopping":
            var parts = ["Shutting down", gpu]
            if let c = status?.costUSD, c > 0 { parts.append(String(format: "$%.2f so far", c)) }
            return parts.filter { !$0.isEmpty }.joined(separator: " · ")
        default: return gpu.isEmpty ? "Cold" : "Cold · \(gpu)"
        }
    }

    /// " · usually ~3 min" while starting, once the model has started before.
    var usually: String {
        guard let t = status?.coldStart?.typicalSeconds else { return "" }
        return " · usually ~" + ColdStartRecord.duration(t >= 60 ? (t / 60).rounded() * 60 : t)
    }

    /// "Last start: 3 min 12 s (Waiting for a GPU 20 s · Loading weights 2 min …)".
    var lastStart: String? {
        guard let last = status?.coldStart?.last else { return nil }
        let how = last.how == "snapshot" ? ", from a snapshot" : ""
        return "Last start: \(ColdStartRecord.duration(last.seconds))\(how) (\(last.breakdown))."
    }

    func help(_ s: String) -> String {
        switch s {
        case "failed": return status?.error ?? "The model didn't start"
        case "ready":
            let rate = (status?.usdPerSecond ?? 0) * 3600
            let run = rate > 0 ? String(format: "Running: $%.2f an hour while warm. It shuts down a few minutes after the last message.", rate)
                : "Running."
            return [run, lastStart].compactMap { $0 }.joined(separator: "\n")
        case "stopping":
            return "Shutting the GPU down: it finishes the replies in progress, then stops (usually 15–25 s). Billing ends when it's gone."
        case "cold":
            return ["No GPU running: the first message starts one (big models take a few minutes).", lastStart]
                .compactMap { $0 }.joined(separator: "\n")
        default:
            return ["Starting the model" + (status?.bootSeconds.map { " (\(ColdStartRecord.duration($0)) so far)" } ?? "") + ".",
                    lastStart].compactMap { $0 }.joined(separator: "\n")
        }
    }
}

// MARK: transcript

struct ChatTranscriptView: View {
    let doc: ChatDocument
    /// Keeps the newest text in view while a reply streams in. Scrolling up stops it (to read back without being pulled
    /// down on every update); scrolling back to the bottom, sending, or "Jump to Latest" starts it again.
    @State private var following = true
    @State private var atBottom = true
    /// The scroll view's position, shared with the scroll indicators. When the user scrolls away it's set to where they
    /// are: SwiftUI otherwise re-applied the last `scrollTo("bottom")` on the next layout and pulled them back down.
    /// (`scrollTo(edge: .bottom)` on it isn't used to follow: with this lazy list it sometimes landed at the top.)
    @State private var position = ScrollPosition()

    /// Where the transcript is scrolled to, and whether that's (near) its end.
    struct Spot: Equatable {
        var offset: CGFloat
        var height: CGFloat
        var atBottom: Bool
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    // The path is walked once here (it's rebuilt from the message tree on each read), and each tool
                    // call's result is looked up in a table instead of by scanning every message per call.
                    let turns = doc.transcript.turns
                    let shown = turns.filter { $0.role != .tool }
                    let results = Dictionary(turns.compactMap { t in t.toolCallID.map { ($0, t) } },
                                             uniquingKeysWith: { _, last in last })
                    let runs = AgentRuns(shown, transcript: doc.transcript)
                    let folding = RunFolding(open: doc.openRuns, results: results, streaming: doc.isStreaming,
                                             asking: doc.permission?.call.id)
                    ForEach(TranscriptRow.rows(shown, live: doc.liveTurnID, files: doc.isCoding ? doc.fileIndex : nil,
                                               bare: runs.bare, folding: folding)) { row in
                        ChatTurnView(doc: doc, turn: row.turn, results: results, part: row.part,
                                     run: runs.steps[row.turn.id], latest: runs.latest.contains(row.turn.id))
                            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous)
                                .fill(MG.accent.opacity(doc.highlightTurnID == row.turn.id ? 0.10 : 0)).padding(-6))
                            .animation(.easeOut(duration: 0.3), value: doc.highlightTurnID)
                            .padding(.top, row.gap)
                    }
                    if let last = shown.last, last.role == .assistant, let items = last.followUps, !items.isEmpty,
                       !doc.isStreaming, doc.question == nil {
                        FollowUpChips(doc: doc, items: items).padding(.top, 18)
                    }
                    Color.clear.frame(height: 1).padding(.top, 18).id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 18)
                .frame(maxWidth: 860)
                .frame(maxWidth: .infinity)
            }
            .mgScrollIndicators(position: $position)
            .onScrollGeometryChange(for: Spot.self) { g in
                let bottom = g.contentOffset.y + g.containerSize.height
                return Spot(offset: g.contentOffset.y, height: g.contentSize.height,
                            atBottom: bottom >= g.contentSize.height - 80)
            } action: { old, new in
                atBottom = new.atBottom
                if Self.debug {
                    print("[scroll] \(Int(old.offset))→\(Int(new.offset)) h \(Int(old.height))→\(Int(new.height)) bottom \(new.atBottom) following \(following)")
                }
                // Moved up and away from the end: the user scrolled (text growing below, and our own scrolls to the
                // end, only ever move the offset down; content shrinking at the end leaves it at the end).
                if new.offset < old.offset - 2, !new.atBottom {
                    if following { position.scrollTo(y: new.offset) }  // replaces the pending "bottom" target
                    following = false
                } else if new.atBottom {
                    following = true
                }
            }
            .onChange(of: doc.transcript.nodes.last.map { $0.text.count + ($0.reasoning?.count ?? 0) + $0.toolCalls.count }) { _, _ in
                if Self.debug { print("[scroll] update, following \(following)") }
                if doc.isStreaming && following { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: doc.transcript.nodes.count) { _, _ in  // a tool card or the next reply started
                if doc.isStreaming && following { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            // The reply's last text can land in the same update that ends the stream (isStreaming already false
            // above), and scrolls asked for while the window was covered by another app's are dropped: the reply then
            // sat below the fold with nothing to say so. Catch up at the end of a reply and when the window shows again.
            .onChange(of: doc.isStreaming) { _, streaming in
                if !streaming, following { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onReceive(NotificationCenter.default.publisher(for: NSWindow.didChangeOcclusionStateNotification)) { n in
                if following, (n.object as? NSWindow)?.occlusionState.contains(.visible) == true {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
            .onChange(of: doc.transcript.activeLeaf) { _, _ in  // a new message, or another branch shown
                following = true
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            // Not `.defaultScrollAnchor(.bottom)`: with this lazy list it left the transcript blank in a real window.
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: doc.highlightTurnID) { _, id in
                if let id {
                    following = false
                    withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .center) }
                }
            }
            .onChange(of: doc.question?.callID) { _, id in
                if id != nil, following { withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
            .overlay(alignment: .bottom) {
                if !following && !atBottom {
                    Button {
                        following = true
                        withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
                    } label: {
                        Label(doc.isStreaming ? "Jump to Latest" : "Scroll to Bottom", systemImage: "arrow.down")
                            .font(.mgFootnote.weight(.medium))
                            .padding(.horizontal, 12).frame(height: 28)
                            // Opaque: the control fill is see-through in dark mode and the reply showed through.
                            .background(Capsule().fill(MG.windowBackground))
                            .mgHairline(Capsule(), color: MG.border)
                            .shadow(color: .black.opacity(0.25), radius: 8, y: 2)
                            .contentShape(Capsule())
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 10)
                    .transition(.opacity)
                    .mgTestAnchor("jump-to-latest")
                }
            }
            .animation(.easeOut(duration: 0.15), value: following || atBottom)
        }
    }

    /// `AISTUDIO_DEBUG_SCROLL=1`: prints the transcript's scroll positions and follow decisions.
    static let debug = ProcessInfo.processInfo.environment["AISTUDIO_DEBUG_SCROLL"] != nil
}

/// The model's suggested next messages under its last reply; a click sends one.
struct FollowUpChips: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let items: [String]

    var body: some View {
        MGFlowLayout(spacing: 6) {
            ForEach(items, id: \.self) { item in
                Button { model.sendText(doc, item) } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.turn.down.right").font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(MG.secondaryLabel)
                        Text(item).font(.mgFootnote).lineLimit(1)
                    }
                    .padding(.horizontal, 10).frame(height: 26)
                    .background(Capsule().fill(MG.fill))
                    .mgHairline(Capsule(), color: MG.border)
                    .contentShape(Capsule())
                }
                .buttonStyle(.plain)
                .mgHelp("Send “\(item)”")
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// What a message replies to, above its text (in the composer, with ✕; in a sent message, a click shows the original).
struct QuoteBlock: View {
    let quote: ChatQuote
    var onTap: (() -> Void)? = nil
    var onRemove: (() -> Void)? = nil

    /// The two lines shown: the start of a message, but the end of terminal output (where the prompt and the error are).
    private var shown: String {
        guard quote.source == "terminal output" else { return quote.text }
        let lines = quote.text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        let tail = Array(lines.reversed().drop { $0.isEmpty }.prefix(2).reversed())
        return (lines.count > tail.count ? "…" : "") + tail.joined(separator: "\n")
    }

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            RoundedRectangle(cornerRadius: 1).fill(MG.accent).frame(width: 2)
            VStack(alignment: .leading, spacing: 2) {
                Label("Replying to \(quote.source ?? (quote.role == .user ? "your message" : quote.role == .tool ? "a tool result" : "the reply"))",
                      systemImage: "arrowshape.turn.up.left")
                    .font(.mgFootnote.weight(.medium)).foregroundStyle(MG.secondaryLabel)
                Text(shown).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(2).truncationMode(.tail)
            }
            Spacer(minLength: 0)
            if let onRemove {
                Button(action: onRemove) { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Don't reply to it")
            }
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
        .fixedSize(horizontal: false, vertical: true)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .contentShape(Rectangle())
        .onTapGesture { onTap?() }
        .mgHelp(onTap != nil && quote.turnID != nil ? "Show the original" : "")
    }
}

/// Which piece of a message a transcript row shows.
enum TurnPart {
    /// A message that isn't a reply (yours, a session report, a flow result, a summary): one row.
    case message
    /// A reply's thinking, or the waiting line before anything arrives.
    case head
    /// A run of a reply's text (see `MarkdownSegments`).
    case segment(MarkdownSegment)
    /// A reply's tool calls, error and buttons.
    case tail
    /// A folded run's last reply: its error and buttons (its calls are in the fold or pinned).
    case foot
    /// A run's folded work as one row ("✓ Worked 2 min · 12 actions"); opened, the timeline follows.
    case runHeader(RunHeaderInfo)
    /// A live run: its header and a small scrolling box of its steps.
    case liveBox(LiveRunInfo)
    /// A card that stays outside the fold (a question, a plan, a widget, a permission request).
    case pinned(ToolCall)
    /// An opened run's steps, on its timeline rail: a reply's thinking, its text, its tool calls.
    case groupHead, groupProse, groupCalls
}

/// A row of the transcript's lazy list. A reply is several rows (its thinking, each run of its text, each code block
/// or table, its buttons), so only the pieces on screen are built: as one row, a long reply was built whole, in one
/// frame, as it scrolled into view, and opening a chat built the last few replies whole.
struct TranscriptRow: Identifiable {
    /// The first row of a message has the message's ID (what `scrollTo` and highlights use).
    let id: AnyHashable
    let turn: ChatTurn
    let part: TurnPart
    /// Space above: between messages, or between pieces of one reply.
    let gap: CGFloat

    /// `bare`: replies with nothing under them (mid-run, no tool calls, no error, no other versions): no tail row.
    /// `folding`: runs with tool calls fold their work into one row (see `RunFolding`); nil: every piece is a row.
    @MainActor static func rows(_ turns: [ChatTurn], live: UUID?, files: ProjectFileIndex? = nil,
                                bare: Set<UUID> = [], folding: RunFolding? = nil) -> [TranscriptRow] {
        var out: [TranscriptRow] = []
        // A message's first row has the message's ID (what `scrollTo` and highlights use); its other rows a stable
        // name (a position would change while a reply streams in and gains pieces, and the row would lose its state).
        func emit(_ t: ChatTurn, _ p: TurnPart, _ name: String?, gap: CGFloat) {
            let id = name.map { AnyHashable("\(t.id.uuidString)#\($0)") } ?? AnyHashable(t.id)
            out.append(TranscriptRow(id: id, turn: t, part: p, gap: out.isEmpty ? 0 : gap))
        }
        func segments(_ t: ChatTurn) -> [(TurnPart, String)] {
            MarkdownSegments.of(t.text, files: files).enumerated().map { (.segment($0.element), "s\($0.offset)") }
        }
        func reply(_ t: ChatTurn) {
            let isLive = live == t.id
            var parts: [(TurnPart, String)] = []
            if !(t.reasoning ?? "").isEmpty || (t.text.isEmpty && isLive) { parts.append((.head, "head")) }
            if !t.text.isEmpty { parts += segments(t) }
            if !t.toolCalls.isEmpty || t.error != nil || (!isLive && !bare.contains(t.id)) { parts.append((.tail, "tail")) }
            if parts.isEmpty { parts = [(.head, "head")] }
            for (k, (p, name)) in parts.enumerated() {
                emit(t, p, k == 0 ? nil : name, gap: k == 0 ? 18 : (p.isSegment && parts[k - 1].0.isSegment ? 10 : 8))
            }
        }
        // A folded run: its header (or live box), its opened steps on the rail, the pinned cards, then its answer.
        func folded(_ run: [ChatTurn], last: Bool, _ f: RunFolding) {
            let key = RunFolding.key(run)
            let isLive = f.isLive(run, last: last)
            let open = f.open.contains(key)
            let final = run[run.count - 1]
            // The answer shows outside: a finished run's last reply; while live, the last reply while it's only text.
            let finalOutside = !final.text.isEmpty && (!isLive || final.toolCalls.isEmpty)
            let summary = RunSummary.of(run, results: f.results, live: isLive)
            emit(run[0], isLive && !open
                 ? .liveBox(LiveRunInfo(key: key, summary: summary, steps: run, lastTextOutside: finalOutside))
                 : .runHeader(RunHeaderInfo(key: key, summary: summary, open: open)), nil, gap: 18)
            if open {
                for s in run {
                    let waiting = isLive && s.id == final.id && s.text.isEmpty && s.toolCalls.isEmpty
                    if !(s.reasoning ?? "").isEmpty || waiting { emit(s, .groupHead, "head", gap: 0) }
                    if !s.text.isEmpty, !(s.id == final.id && finalOutside) { emit(s, .groupProse, "prose", gap: 0) }
                    if s.toolCalls.contains(where: { !f.pinned($0) }) || (s.error != nil && s.id != final.id) {
                        emit(s, .groupCalls, "calls", gap: 0)
                    }
                }
            }
            for s in run {
                for call in s.toolCalls where f.pinned(call) { emit(s, .pinned(call), "pin:\(call.id)", gap: 10) }
            }
            if finalOutside {
                let segs = segments(final)
                for (k, (p, name)) in segs.enumerated() { emit(final, p, name, gap: k == 0 ? 12 : 10) }
            }
            if !isLive { emit(final, .foot, "tail", gap: 8) }
        }

        var i = 0
        while i < turns.count {
            guard turns[i].role == .assistant else {
                emit(turns[i], .message, nil, gap: 18)
                i += 1
                continue
            }
            var j = i
            while j < turns.count, turns[j].role == .assistant { j += 1 }
            let run = Array(turns[i..<j])
            if let folding, RunFolding.folds(run) {
                folded(run, last: j == turns.count, folding)
            } else {
                run.forEach(reply)
            }
            i = j
        }
        return out
    }
}

extension TurnPart {
    var isSegment: Bool { if case .segment = self { return true } else { return false } }
}

struct ChatTurnView: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let turn: ChatTurn
    /// Tool results by call ID, from the transcript view.
    var results: [String: ChatTurn] = [:]
    /// Which piece of a reply this row shows (a reply is several rows of the lazy list, see `TranscriptRow`).
    var part: TurnPart = .message
    /// The run's replies, on its last reply only (its buttons and the run's stats go there).
    var run: [ChatTurn]? = nil
    /// The newest message or run: its buttons stay shown; others' show while the pointer is over them.
    var latest = false
    @State private var editText = ""
    @State private var hovering = false

    var live: Bool { doc.liveTurnID == turn.id }

    var body: some View {
        switch turn.role {
        case .user where turn.flowEvent != nil:
            FlowEventCard(doc: doc, event: turn.flowEvent!)
        case .user where turn.sessionEvent != nil:
            SessionEventCard(event: turn.sessionEvent!)
        case .user where turn.summary == true:
            SummaryRow(text: turn.text)
        case .user:
            VStack(alignment: .trailing, spacing: 6) {
                if !turn.images.isEmpty || !(turn.files ?? []).isEmpty {
                    AttachmentStrip(images: turn.images, files: turn.files ?? [],
                                    folder: model.chatStore.filesFolder(doc.id))
                        .fixedSize(horizontal: true, vertical: false)
                }
                if doc.editingTurnID == turn.id {
                    VStack(alignment: .trailing, spacing: 8) {
                        TextEditor(text: $editText)
                            .font(.mgBody).scrollContentBackground(.hidden)
                            .frame(minHeight: 60, maxHeight: 220).fixedSize(horizontal: false, vertical: true)
                            .padding(8)
                            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.controlBackground))
                            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
                        HStack {
                            Button("Cancel") { doc.editingTurnID = nil }.buttonStyle(.mg(.bordered))
                            Button("Send") {
                                doc.editingTurnID = nil
                                model.resend(doc, editing: turn.id, text: editText)
                            }
                            .buttonStyle(.mg(.prominent))
                            .disabled(editText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                    .frame(maxWidth: 640)
                } else if !turn.text.isEmpty || turn.quote != nil {
                    if let q = turn.quote {
                        HStack {
                            Spacer(minLength: 80)
                            QuoteBlock(quote: q, onTap: { model.showQuoted(doc, q) }).frame(maxWidth: 520)
                        }
                    }
                    HStack {
                        Spacer(minLength: 80)
                        MessageBubbleText(doc: doc, turn: turn)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.selection))
                    }
                }
                HStack(spacing: 6) {
                    BranchSwitcher(doc: doc, turn: turn)
                    if doc.editingTurnID != turn.id, !doc.isStreaming {
                        // Shown while the pointer is over the message (they stay in the accessibility tree, and the
                        // context menu has them too).
                        HStack(spacing: 6) {
                            Button { model.quote(doc, turn.text, from: turn) } label: { Image(systemName: "arrowshape.turn.up.left") }.accessibilityLabel("Reply to This Message")
                                .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Reply to this message")
                            Button { editText = turn.text; doc.editingTurnID = turn.id } label: { Image(systemName: "pencil") }.accessibilityLabel("Edit Message")
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Edit and send again (the old version stays as a branch)")
                            if doc.isCoding {
                                Button { model.prepareRewind(doc, to: turn.id) } label: { Image(systemName: "clock.arrow.circlepath") }.accessibilityLabel("Rewind to Here")
                                    .buttonStyle(.mg(.icon)).controlSize(.small)
                                    .mgHelp("Rewind files and chat to before this message")
                            }
                        }
                        .opacity(hovering ? 1 : 0.001)  // not 0: SwiftUI drops fully transparent views from accessibility
                        .animation(.easeOut(duration: 0.12), value: hovering)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .contentShape(Rectangle())
            .onHover { hovering = $0 }
            .mgContextMenu {
                MGMenuItem.button("Copy") { copy(turn.text) }
                MGMenuItem.button("Reply") { model.quote(doc, turn.text, from: turn) }
                MGMenuItem.button("Edit…", enabled: !doc.isStreaming) { editText = turn.text; doc.editingTurnID = turn.id }
                if doc.isCoding {
                    MGMenuItem.button("Rewind to Here…", enabled: !doc.isStreaming) { model.prepareRewind(doc, to: turn.id) }
                }
                MGMenuItem.button("Fork from Here", enabled: !doc.isStreaming) { model.fork(doc, at: turn.id) }
            }
        case .assistant:
            switch part {
            case .runHeader(let info):
                RunHeaderView(doc: doc, info: info)
            case .liveBox(let info):
                LiveRunBox(doc: doc, info: info, results: results)
            case .pinned(let call):
                card(call)
            default:
                assistantPart
                    .mgContextMenu {
                        MGMenuItem.button("Copy") { copy(turn.text) }
                        MGMenuItem.button("Reply") { model.quote(doc, turn.text, from: turn) }
                        MGMenuItem.button("Regenerate", enabled: !doc.isStreaming) { model.regenerate(doc, turn.id) }
                        MGMenuItem.button("Fork from Here", enabled: !doc.isStreaming && !live) { model.fork(doc, at: turn.id) }
                    }
            }
        case .tool:
            EmptyView()  // never a row: a tool's result shows on its call's card in the reply
        }
    }

    /// One piece of a reply: its thinking (or the waiting line), a run of its text, or its tool calls and buttons.
    @ViewBuilder var assistantPart: some View {
        switch part {
        case .message:  // a reply as one piece (the transcript splits replies into rows; kept for other callers)
            VStack(alignment: .leading, spacing: 8) {
                assistantHead
                if !turn.text.isEmpty { MarkdownView(text: turn.text) }
                assistantTail(calls: turn.toolCalls)
            }
        case .head:
            assistantHead
        case .segment(let segment):
            Group {
                switch segment {
                case .prose(let a): Text(a).font(.mgBody).fixedSize(horizontal: false, vertical: true)
                case .block(let block): MarkdownBlockView(block: block)
                }
            }
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        case .tail:
            assistantTail(calls: turn.toolCalls)
        case .foot:
            assistantTail(calls: [])
        case .groupHead:
            Group {
                if let r = turn.reasoning, !r.isEmpty {
                    ThinkingDisclosure(doc: doc, turnID: turn.id, text: r, seconds: turn.thinkingSeconds,
                                       live: live && turn.text.isEmpty)
                } else {
                    WaitingLine(doc: doc)
                }
            }
            .runRail()
        case .groupProse:
            RunProse(doc: doc, turn: turn).runRail()
        case .groupCalls:
            VStack(alignment: .leading, spacing: 2) {
                ForEach(turn.toolCalls.filter { !RunFolding.pinnedTools.contains($0.name) && $0.id != doc.permission?.call.id },
                        id: \.id) { call in
                    stepCard(call)
                }
                if let e = turn.error {
                    Label(e, systemImage: "exclamationmark.triangle").font(.mgFootnote).foregroundStyle(MG.destructive)
                }
            }
            .runRail()
        case .runHeader, .liveBox, .pinned:
            EmptyView()  // drawn in `body`
        }
    }

    /// A call's card at full size (a reply's tail, a pinned card).
    @ViewBuilder func card(_ call: ToolCall) -> some View {
        let result = results[call.id]
        Group {
            switch call.name {
            case "ask_user": QuestionCard(doc: doc, call: call, result: result)
            case "exit_plan": PlanCard(doc: doc, call: call, result: result)
            case "show_widget": WidgetCard(doc: doc, call: call, result: result, live: live)
            case "todo_write": TodoUpdateRow(call: call)
            case "propose_session": SessionProposalCard(doc: doc, call: call, result: result)
            default: ToolCard(doc: doc, call: call, result: result)
            }
        }
        .mgContextMenu {
            MGMenuItem.button("Reply to This Result", enabled: result != nil) {
                if let r = result { model.quote(doc, r.text.isEmpty ? (r.toolDetail ?? "") : r.text, from: r) }
            }
        }
    }

    /// A call as a step of a folded run: one line until opened.
    @ViewBuilder func stepCard(_ call: ToolCall) -> some View {
        let result = results[call.id]
        Group {
            if call.name == "todo_write" { TodoUpdateRow(call: call) }
            else { ToolCard(doc: doc, call: call, result: result, compact: true) }
        }
        .mgContextMenu {
            MGMenuItem.button("Reply to This Result", enabled: result != nil) {
                if let r = result { model.quote(doc, r.text.isEmpty ? (r.toolDetail ?? "") : r.text, from: r) }
            }
        }
    }

    @ViewBuilder var assistantHead: some View {
        if let r = turn.reasoning, !r.isEmpty {
            ThinkingDisclosure(doc: doc, turnID: turn.id, text: r, seconds: turn.thinkingSeconds, live: live && turn.text.isEmpty)
        } else if turn.text.isEmpty, turn.toolCalls.isEmpty, live {
            // (A message that is only tool calls isn't waiting for the model: its cards say what it waits for.)
            WaitingLine(doc: doc)
        }
    }

    @ViewBuilder func assistantTail(calls: [ToolCall]) -> some View {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(calls, id: \.id) { call in card(call) }
                if let e = turn.error {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Label(e, systemImage: e == "Stopped" ? "stop.circle" : "exclamationmark.triangle")
                            .font(.mgFootnote).foregroundStyle(e == "Stopped" ? MG.secondaryLabel : MG.destructive)
                            .fixedSize(horizontal: false, vertical: true)
                        // The last reply failed (a network or model error): the same request once more.
                        if e != "Stopped", !live, !doc.isStreaming, doc.transcript.turns.last?.id == turn.id {
                            Button("Try Again") { model.regenerate(doc, turn.id) }
                                .buttonStyle(.mg(.bordered)).controlSize(.small)
                                .mgHelp("Send the same request again (this reply stays as a branch)")
                                .mgTestAnchor("try-again")
                        }
                    }
                }
                if !live, model.canAdoptPlan(doc, turn) { AdoptPlanBar(doc: doc, turn: turn) }
                let versions = doc.transcript.siblings(of: turn.id).count > 1
                if !live, run != nil || versions {
                    HStack(spacing: 6) {
                        BranchSwitcher(doc: doc, turn: turn)
                        if let run, !doc.isStreaming {
                            let text = run.map(\.text).filter { !$0.isEmpty }.joined(separator: "\n\n")
                            Button { model.regenerate(doc, turn.id) } label: { Image(systemName: "arrow.clockwise") }.accessibilityLabel("Regenerate")
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Another reply (this one stays as a branch)")
                            Button { copy(text) } label: { Image(systemName: "doc.on.doc") }.accessibilityLabel("Copy")
                                .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp(run.count > 1 ? "Copy the replies' text" : "Copy")
                            if !text.isEmpty {
                                Button { model.quote(doc, text, from: turn) } label: {
                                    Image(systemName: "arrowshape.turn.up.left")
                                }
                                .accessibilityLabel("Reply to This Message")
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Reply to this message (or select a passage, then right-click › Reply with Selection)")
                            }
                        }
                        if let run { RunStatsLine(steps: run, effort: turn.effort) }
                    }
                    // The newest run's buttons stay; older ones show while the pointer is over the reply's end.
                    .opacity(latest || hovering ? 1 : 0.001)  // not 0: SwiftUI drops fully transparent views from accessibility
                    .animation(.easeOut(duration: 0.12), value: hovering)
                    .mgTestAnchor(latest ? "run-buttons" : "run-buttons-old")
                }
            }
            .onHover { hovering = $0 }
    }

    func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// Where the conversation was summarized: a divider, which opens to the summary.
struct SummaryRow: View {
    let text: String
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { open.toggle() }
            } label: {
                HStack(spacing: 10) {
                    Rectangle().fill(MG.separator).frame(height: 1)
                    HStack(spacing: 6) {
                        Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                            .rotationEffect(.degrees(open ? 90 : 0))
                        Image(systemName: "arrow.down.right.and.arrow.up.left")
                        Text("Earlier conversation summarized")
                    }
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    Rectangle().fill(MG.separator).frame(height: 1)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .mgHelp("Older messages were replaced by a summary to save context (click to read it)")
            if open {
                MarkdownView(text: text)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// ‹ 2 / 3 › between a message's versions (edits of a question, retries of an answer).
struct BranchSwitcher: View {
    let doc: ChatDocument
    let turn: ChatTurn

    var body: some View {
        let siblings = doc.transcript.siblings(of: turn.id)
        if siblings.count > 1, let i = siblings.firstIndex(where: { $0.id == turn.id }) {
            HStack(spacing: 2) {
                Button { doc.transcript.select(siblings[i - 1].id) } label: { Image(systemName: "chevron.left") }.accessibilityLabel("Previous Version")
                    .disabled(i == 0 || doc.isStreaming)
                Text("\(i + 1) / \(siblings.count)").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).monospacedDigit()
                Button { doc.transcript.select(siblings[i + 1].id) } label: { Image(systemName: "chevron.right") }.accessibilityLabel("Next Version")
                    .disabled(i == siblings.count - 1 || doc.isStreaming)
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.small)
            .mgHelp("Versions of this message")
        }
    }
}

/// Attached images and videos (thumbnails) and files (chips); click one to preview it.
struct AttachmentStrip: View {
    @Environment(AppModel.self) private var model
    let images: [String]
    let files: [ChatFile]
    let folder: URL
    var onRemove: ((String) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(images.enumerated()), id: \.element) { i, name in
                    let video = ChatTurn.isVideo(name)
                    let file = folder.appendingPathComponent(name)
                    ZStack(alignment: .topTrailing) {
                        ZStack {
                            if let image = NSImage(contentsOf: video ? AppModel.posterURL(file) : file) {
                                Image(nsImage: image).resizable().aspectRatio(contentMode: .fill)
                            } else {
                                RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill)
                            }
                            if video {
                                Image(systemName: "play.circle.fill").font(.title2)
                                    .foregroundStyle(.white).shadow(radius: 2)
                            }
                        }
                        .frame(width: 64, height: 64)
                        .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
                        .contentShape(Rectangle())
                        .onTapGesture {
                            let titles = images.indices.map { j in
                                (ChatTurn.isVideo(images[j]) ? "Video" : "Image") + (images.count > 1 ? " \(j + 1)" : "")
                            }
                            model.chatViewer = ChatViewerItem(urls: images.map { folder.appendingPathComponent($0) }, index: i,
                                                              titles: titles)
                        }
                        .mgHelp(video ? "Video (click to play)" : "Image (click to view)")
                        .mgTestAnchor("attachment:\(name)")
                        remove(name)
                    }
                }
                ForEach(files, id: \.self) { f in
                    HStack(spacing: 4) {
                        Image(nsImage: FileIcons.icon(for: f.name)).resizable().frame(width: 14, height: 14)
                        Text(f.name).lineLimit(1)
                        remove(f.name)
                    }
                    .font(.mgFootnote)
                    .padding(.horizontal, 8).frame(height: 26)
                    .background(Capsule().fill(MG.fill))
                    .contentShape(Capsule())
                    .onTapGesture {
                        if let url = previewURL(f) { model.chatViewer = ChatViewerItem(urls: [url], index: 0, titles: [f.name]) }
                    }
                    .mgHelp("\(f.name) (click to preview)")
                    .mgTestAnchor("attachment:\(f.name)")
                }
            }
        }
        .frame(height: images.isEmpty ? 28 : 66)
    }

    /// The original when it was kept, else the text written out (once) for the preview.
    func previewURL(_ f: ChatFile) -> URL? {
        if let stored = f.stored { return folder.appendingPathComponent(stored) }
        let dir = folder.appendingPathComponent("previews", isDirectory: true)
        let url = dir.appendingPathComponent((f.name as NSString).lastPathComponent)
        if (try? String(contentsOf: url, encoding: .utf8)) != f.text {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            guard (try? f.text.write(to: url, atomically: true, encoding: .utf8)) != nil else { return nil }
        }
        return url
    }

    @ViewBuilder
    func remove(_ name: String) -> some View {
        if let onRemove {
            Button { onRemove(name) } label: { Image(systemName: "xmark.circle.fill") }.accessibilityLabel("Remove Attachment")
                .buttonStyle(.plain).foregroundStyle(MG.secondaryLabel).mgHelp("Remove")
        }
    }
}

/// "Thought for 12 s", collapsed once the answer starts; open while the model is still thinking.
struct ThinkingDisclosure: View {
    let doc: ChatDocument
    /// The reply it belongs to (its open state is kept in the chat as `think:<id>`).
    let turnID: UUID
    let text: String
    let seconds: Double?
    let live: Bool
    /// In a live run's box: while thinking, the label only (the box is too small for the text).
    var compact = false

    var key: String { "think:\(turnID.uuidString)" }

    var body: some View {
        let open = doc.openRuns.contains(key)
        let showsLive = live && !compact
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { doc.toggleOpen(key) }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .rotationEffect(.degrees(open || showsLive ? 90 : 0))
                    if live && compact { MGSpinner().controlSize(.mini) } else { Image(systemName: "brain") }
                    Text(live ? "Thinking…" : seconds.map { "Thought for \(Self.duration($0))" } ?? "Thoughts")
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .frame(height: 22)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            if open || showsLive {
                Text(live ? String(text.suffix(1200)) : text)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.leading, 10)
                    .overlay(alignment: .leading) { Rectangle().fill(MG.separator).frame(width: 2) }
            }
        }
    }

    static func duration(_ s: Double) -> String {
        s < 60 ? "\(max(1, Int(s.rounded()))) s" : "\(Int(s) / 60) min \(Int(s) % 60) s"
    }
}

/// While a reply hasn't started: what the request waits for (a GPU, the weights loading), and for how long.
struct WaitingLine: View {
    let doc: ChatDocument

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { ctx in
            HStack(spacing: 8) {
                MGSpinner().controlSize(.small)
                let elapsed = Int(ctx.date.timeIntervalSince(doc.requestStarted ?? ctx.date))
                let phase = doc.engine.map { e -> String in
                    if let p = e.progress, p < 1, e.state == "loading" { return "\(e.phase ?? "Loading") \(Int(p * 100))%" }
                    return e.phase ?? (e.state == "starting" ? "Waiting for a GPU" : "Starting the model")
                } ?? "Waiting for the model"
                Text(elapsed >= 3 ? "\(phase) · \(elapsed / 60):\(String(format: "%02d", elapsed % 60))" : phase)
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
        }
    }
}

/// The thinking-effort levels of a chat's model and the menu items for them (the header's menu and the composer's).
enum EffortLevels {
    static let help = "How hard the model thinks before answering. On/Off models switch thinking; gpt-oss has three levels."

    static func label(_ value: String?) -> String {
        guard let v = value else { return "Model's default" }
        switch v {
        case "off": return "Off"
        case "on": return "On"
        case "low": return "Low"
        case "medium": return "Medium"
        case "high": return "High"
        default: return v.prefix(1).uppercased() + v.dropFirst()
        }
    }

    /// The model's levels: its own, today's three for an older server, nil (hide the control) when it can't be controlled.
    @MainActor static func levels(_ model: AppModel, _ doc: ChatDocument) -> [String]? {
        let l = doc.modelKey.flatMap { model.llm.model($0) }?.efforts ?? ["low", "medium", "high"]
        return l.isEmpty ? nil : l
    }

    @MainActor @MGMenuBuilder static func items(_ levels: [String], doc: ChatDocument) -> [MGMenuItem] {
        MGMenuItem.header("Thinking effort")
        for value in [nil] + levels.map({ Optional($0) }) {
            MGMenuItem.button(label(value), checked: doc.transcript.reasoningEffort == value) {
                doc.transcript.reasoningEffort = value
            }
        }
        if let cur = doc.transcript.reasoningEffort, !levels.contains(cur) {
            MGMenuItem.button("\(label(cur)) (not supported by this model)", checked: true, enabled: false) {}
        }
    }
}

/// The composer's effort control: the brain and the level, the icon alone when the row is narrow.
struct ComposerEffortMenu: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        if let levels = EffortLevels.levels(model, doc) {
            let cur = doc.transcript.reasoningEffort
            let menu = { (text: Bool) in
                MGMenuButton(kind: .plain) {
                    EffortLevels.items(levels, doc: doc)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "brain")
                        if text { Text(cur == nil ? "Default" : EffortLevels.label(cur)).lineLimit(1) }
                    }
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                }
                .mgHelp(EffortLevels.help)
            }
            ViewThatFits(in: .horizontal) {
                menu(true).fixedSize()
                menu(false).fixedSize()
            }
        }
    }
}

struct StatsLine: View {
    let stats: RequestStats
    let model: String?
    var effort: String? = nil
    /// Model, speed, tokens and cost only (first token and waiting go in the tooltip).
    var compact = false

    var body: some View {
        var parts: [String] = []
        if let m = model { parts.append(m) }
        if let t = stats.tokPerSec { parts.append(String(format: "%.0f tok/s", t)) }
        if let c = stats.completionTokens { parts.append("\(c) tokens") }
        if !compact, let ttft = stats.ttftMs { parts.append(String(format: "first token %.1f s", ttft / 1000)) }
        if !compact, let w = stats.waitSeconds, w > 5 { parts.append(String(format: "waited %.0f s", w)) }
        if let c = stats.costUSD, c > 0 { parts.append(String(format: "$%.4f", c)) }
        if let e = effort { parts.append("effort \(EffortLevels.label(e).lowercased())") }
        return Text(parts.joined(separator: " · ")).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
    }

    /// One request's figures in full, for a tooltip.
    static func detail(_ s: RequestStats) -> String {
        var parts: [String] = []
        if let t = s.tokPerSec { parts.append(String(format: "%.0f tok/s", t)) }
        if let c = s.completionTokens { parts.append("\(c) tokens") }
        if let ttft = s.ttftMs { parts.append(String(format: "first token %.1f s", ttft / 1000)) }
        if let w = s.waitSeconds, w > 1 { parts.append(String(format: "waited %.0f s for the model", w)) }
        parts.append(RunStatsLine.duration(s.seconds))
        if let c = s.costUSD, c > 0 { parts.append(String(format: "$%.4f", c)) }
        return parts.joined(separator: " · ")
    }
}

/// The agent runs in the shown messages: the replies between two of your messages (session reports and flow results
/// count as messages too). The last reply of each run gets the run's buttons and stats.
struct AgentRuns {
    /// Each run's replies, by the id of its last reply.
    var steps: [UUID: [ChatTurn]] = [:]
    /// The newest run's last reply (its buttons stay shown).
    var latest: Set<UUID> = []
    /// Mid-run replies with nothing under them (no tool calls, no error, no other versions).
    var bare: Set<UUID> = []

    init(_ shown: [ChatTurn], transcript: ChatTranscript) {
        var current: [ChatTurn] = []
        func close() {
            guard let last = current.last else { return }
            steps[last.id] = current
            for t in current.dropLast() where t.toolCalls.isEmpty && t.error == nil && transcript.siblings(of: t.id).count <= 1 {
                bare.insert(t.id)
            }
            current = []
        }
        for t in shown {
            if t.role == .assistant { current.append(t) } else { close() }
        }
        close()
        if let last = shown.last(where: { $0.role == .assistant }) { latest.insert(last.id) }
    }
}

/// Under a run's last reply: the run's model, steps, tokens, time and cost; each step's figures in the tooltip.
struct RunStatsLine: View {
    let steps: [ChatTurn]
    var effort: String? = nil

    var body: some View {
        let stats = steps.compactMap(\.stats)
        if stats.count == 1, let s = stats.first {
            StatsLine(stats: s, model: steps.first(where: { $0.stats != nil })?.model, effort: effort, compact: true)
                .mgHelp(StatsLine.detail(s))
        } else if !stats.isEmpty {
            Text(summary(stats)).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                .mgHelp(stats.enumerated().map { "\($0.offset + 1). " + StatsLine.detail($0.element) }.joined(separator: "\n"))
                .mgTestAnchor("run-stats")
        }
    }

    func summary(_ stats: [RequestStats]) -> String {
        var parts: [String] = []
        if let m = steps.last(where: { $0.model != nil })?.model { parts.append(m) }
        parts.append("\(stats.count) steps")
        let tokens = stats.compactMap(\.completionTokens).reduce(0, +)
        if tokens > 0 { parts.append("\(tokens) tokens") }
        parts.append(Self.duration(stats.map(\.seconds).reduce(0, +)))
        let cost = stats.compactMap(\.costUSD).reduce(0, +)
        if cost > 0 { parts.append(String(format: "$%.3f", cost)) }
        if let e = effort { parts.append("effort \(EffortLevels.label(e).lowercased())") }
        return parts.joined(separator: " · ")
    }

    static func duration(_ s: Double) -> String {
        s >= 60 ? "\(Int(s) / 60) m \(Int(s) % 60) s" : String(format: "%.1f s", s)
    }
}

// MARK: empty state & composer

struct ChatEmptyState: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        VStack(spacing: 14) {
            Spacer()
            if model.llm.servers.isEmpty {
                MGEmptyState(symbol: "server.rack", title: "No GPU server",
                             message: "Chats run models on your GPU server (Modal, or a GPU box). Add one in Servers.")
                Button("Open Servers") { model.showServersPanel = true }.buttonStyle(.mg(.bordered))
            } else if model.llm.models.isEmpty {
                MGEmptyState(symbol: "link", title: "No models yet",
                             message: "Add any model from Hugging Face by its link: it's downloaded to your GPU server, which picks the GPUs it needs.")
                Button("Add a Model from a Link…") { model.llmAddRequest = LLMAddRequest() }.buttonStyle(.mg(.prominent))
            } else if doc.isGraphHelper {
                MGEmptyState(symbol: "wand.and.sparkles", title: "Graph Helper",
                             message: "It reads this workflow, changes it (one undo step per reply), runs it and looks at the results.")
                VStack(spacing: 6) {
                    ForEach(["Explain what this workflow does", "Why doesn't it run?", "Add an upscale before Save",
                             "Write a better prompt, run it and check the result"], id: \.self) { s in
                        Button(s) { doc.draft = s }.buttonStyle(.mg(.bordered))
                    }
                }
            } else {
                MGEmptyState(symbol: "bubble.left.and.text.bubble.right", title: doc.transcript.modelID ?? "New Chat",
                             message: "Ask anything. The GPU starts with your first message and stops a few minutes after the last one.")
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }
}

struct ChatComposer: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument
    var inFront = true
    @FocusState private var focused: Bool
    @State private var notice: String?
    /// The highlighted suggestion (↑/↓), and the draft the list was closed for (Esc).
    @State private var selected = 0
    @State private var dismissedFor: String?

    var body: some View {
        let images = doc.pending.compactMap { if case .image(let n) = $0 { return n }; return nil }
        let files = doc.pending.compactMap { if case .file(let f) = $0 { return f }; return nil }
        let blind = !images.isEmpty && doc.modelKey.flatMap { model.llm.model($0) }.map { $0.vision != true } == true
        VStack(alignment: .leading, spacing: 6) {
            if doc.isAgent, !doc.todos.isEmpty,
               doc.isStreaming || doc.todos.contains(where: { $0.status != "completed" }) || doc.transcript.todoListIsCurrent {
                TodoListView(doc: doc)
            }
            if !doc.pending.isEmpty {
                AttachmentStrip(images: images, files: files, folder: model.chatStore.filesFolder(doc.id)) { name in
                    doc.pending.removeAll { $0.name == name }
                }
            }
            if doc.preparingVideos > 0 {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(doc.preparingVideos == 1 ? "Preparing the video…" : "Preparing \(doc.preparingVideos) videos…")
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            if let note = blind ? "This model can't see images or videos: they're left out of what it gets."
                : (notice ?? doc.attachNote) {
                Label(note, systemImage: "exclamationmark.triangle").font(.mgFootnote).foregroundStyle(MG.warning)
            }
            if doc.rewindTarget != nil { RewindCard(doc: doc) }
            if let q = doc.replyTo {
                QuoteBlock(quote: q, onTap: { model.showQuoted(doc, q) }, onRemove: { doc.replyTo = nil })
            }
            if let q = doc.queued { QueuedBar(text: q, onEdit: { model.unqueue(doc) }, onRemove: { doc.queued = nil }) }
            let shown = suggestions
            if !shown.isEmpty {
                ComposerSuggestionList(items: shown, selected: min(selected, shown.count - 1)) { accept($0) }
            }
            HStack(alignment: .bottom, spacing: 10) {
                Button { choose() } label: { Image(systemName: "paperclip") }.accessibilityLabel("Attach Files")
                    .buttonStyle(.mg(.icon))
                    .controlSize(.large)
                    .mgHelp("Attach images, videos or text files (or drop them on the chat)")
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $doc.draft)
                        .font(.mgBody)
                        .scrollContentBackground(.hidden)
                        .focused($focused)
                        .frame(minHeight: 40, maxHeight: 180)
                        .fixedSize(horizontal: false, vertical: true)
                        .onKeyPress(phases: [.down, .repeat]) { handleKey($0) }
                        .onChange(of: doc.draft) { _, draft in
                            selected = 0
                            if dismissedFor != nil, dismissedFor != draft { dismissedFor = nil }
                            if doc.isCoding, doc.fileIndex == nil, ComposerSuggestions.mentionQuery(draft) != nil {
                                model.refreshProjectFiles(doc)
                            }
                        }
                    if doc.draft.isEmpty {
                        Text(doc.question != nil ? "Answer the question above, or type your own answer…"
                             : doc.isGraphHelper ? "Ask the helper to build, change or fix this workflow…"
                             : doc.isCoding ? "Ask the agent to change something in \(doc.transcript.worktree.map { ($0.repo as NSString).lastPathComponent } ?? doc.projectURL?.lastPathComponent ?? "the project")…  (/ for skills, @ for files)"
                             : "Message \(doc.transcript.modelID ?? "the model")…  (⇧↩ for a new line)")
                            .font(.mgBody).foregroundStyle(MG.placeholder).lineLimit(1).truncationMode(.tail)
                            .padding(.leading, 5).padding(.top, 1)
                            .allowsHitTesting(false)
                    }
                }
                .padding(8)
                .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.controlBackground))
                .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
                let typed = !doc.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                if doc.isStreaming && !(doc.question != nil && typed) {
                    Button { model.stopReply(doc) } label: { Label("Stop", systemImage: "stop.fill").labelStyle(.titleAndIcon) }
                        .buttonStyle(.mg(.bordered))
                        .controlSize(.large)
                        .mgHelp("Stop the reply (esc)")
                    if typed {
                        Button { model.send(doc) } label: { Image(systemName: "arrow.up") }
                            .buttonStyle(.mg(.prominent))
                            .controlSize(.large)
                            .mgHelp("Queue: sent when this reply ends (↩)")
                            .accessibilityLabel("Queue Message")
                    }
                } else {
                    Button { model.send(doc) } label: { Image(systemName: "arrow.up") }
                        .buttonStyle(.mg(.prominent))
                        .controlSize(.large)
                        .disabled((doc.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && doc.pending.isEmpty) || doc.preparingVideos > 0)
                        .mgHelp("Send (↩)")
                        .accessibilityLabel("Send")
                }
            }
            HStack(spacing: 8) {
                PermissionModeMenu(doc: doc)
                ComposerEffortMenu(doc: doc)
                ContextMeter(doc: doc)
                ColdModelHint(doc: doc)
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
        .onAppear { if inFront { focused = true } }
        .onChange(of: inFront) { _, on in if on { focused = true } }
        .onChange(of: doc.composerFocusRequest) { focused = true }
    }

    /// The skills for a "/" draft, or the project files for an "@word" being typed at the end (coding chats).
    var suggestions: [ComposerSuggestion] {
        guard doc.isCoding, doc.draft != dismissedFor, doc.question == nil else { return [] }
        if doc.draft.hasPrefix("/") { return ComposerSuggestions.slash(doc.draft, skills: model.loadSkills(root: doc.projectURL)) }
        if let q = ComposerSuggestions.mentionQuery(doc.draft), let index = doc.fileIndex {
            return index.matches(q).map { ComposerSuggestion(file: $0, draft: ComposerSuggestions.accept(mention: $0, in: doc.draft)) }
        }
        return []
    }

    func accept(_ item: ComposerSuggestion) {
        doc.draft = item.draft
        dismissedFor = nil
        focused = true
    }

    /// The composer's keys: ↩ sends (or takes the highlighted suggestion), ⇧↩/⌥↩ a new line, ↑/↓ and ⇥ for the list,
    /// ↑ in an empty field brings back the last message, esc closes the list or stops the reply, ⇧⇥ switches the mode.
    func handleKey(_ press: KeyPress) -> KeyPress.Result {
        let items = suggestions
        let down = press.phase == .down
        if (press.key == .tab && press.modifiers.contains(.shift)) || press.characters == "\u{19}" {
            guard doc.isAgent else { return .ignored }
            if down { doc.mode = doc.mode.cycled }
            return .handled
        }
        switch press.key {
        case .return:
            if press.modifiers.contains(.shift) || press.modifiers.contains(.option) { return .ignored }
            guard down else { return .handled }
            let item = items.isEmpty ? nil : items[min(selected, items.count - 1)]
            // A complete "/compact" is sent, not completed again.
            if let item, item.draft.trimmingCharacters(in: .whitespaces) != doc.draft {
                accept(item)
            } else {
                model.send(doc)
            }
            return .handled
        case .tab:
            guard !items.isEmpty, press.modifiers.isEmpty else { return .ignored }
            if down { accept(items[min(selected, items.count - 1)]) }
            return .handled
        case .upArrow:
            if !items.isEmpty {
                selected = (min(selected, items.count - 1) - 1 + items.count) % items.count
                return .handled
            }
            if down, doc.draft.isEmpty, let last = doc.transcript.turns.last(where: {
                $0.role == .user && $0.summary != true && $0.sessionEvent == nil && $0.flowEvent == nil && !$0.text.isEmpty
            }) {
                doc.draft = last.text
                return .handled
            }
            return .ignored
        case .downArrow:
            guard !items.isEmpty else { return .ignored }
            selected = (min(selected, items.count - 1) + 1) % items.count
            return .handled
        case .escape:
            if !items.isEmpty {
                dismissedFor = doc.draft
                return .handled
            }
            // A card waiting for an answer keeps esc (Deny, Skip, Cancel).
            if down, doc.isStreaming, doc.permission == nil, doc.question == nil, doc.plan == nil, doc.rewindTarget == nil {
                model.stopReply(doc)
                return .handled
            }
            return .ignored
        default:
            return .ignored
        }
    }

    func choose() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false
        panel.message = "Images, videos, or text and code files up to 256 KB"
        guard panel.runModal() == .OK else { return }
        doc.attachNote = nil
        notice = model.attach(panel.urls, to: doc)
    }
}

/// Under a Plan-mode reply that wrote its plan as text and never presented it for approval (small models do this):
/// approve the reply as the plan, as the plan card would.
struct AdoptPlanBar: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let turn: ChatTurn

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.text.magnifyingglass").foregroundStyle(MG.secondaryLabel)
            Text("No plan was presented for approval. Go ahead with this reply as the plan?")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 4)
            Button("Approve, Auto-accept Edits") { model.adoptPlan(doc, turn.id, mode: .acceptEdits) }
                .buttonStyle(.mg(.prominent)).controlSize(.small)
                .mgHelp("Use this reply as the plan: it's saved in Plans, and the agent carries it out, editing without asking")
                .mgTestAnchor("adopt-plan")
            Button("Ask Each Time") { model.adoptPlan(doc, turn.id, mode: .ask) }
                .buttonStyle(.mg(.bordered)).controlSize(.small)
                .mgHelp("Use this reply as the plan; the agent asks before each edit and command")
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
    }
}

/// Under the field while the chat's model has no GPU running: the next message starts one, which takes a while.
struct ColdModelHint: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        if !doc.isStreaming, let key = doc.modelKey, let st = model.llm.status(key), st.state == "cold" {
            let typical = st.coldStart?.typicalSeconds
            Label("GPU off: your message starts it" + (typical.map { " (usually ~\(ColdStartRecord.duration($0 >= 60 ? ($0 / 60).rounded() * 60 : $0)))" }
                                                       ?? " (a few minutes for big models)"),
                  systemImage: "moon.zzz")
                .font(.mgFootnote).foregroundStyle(MG.tertiaryLabel).lineLimit(1)
                .mgHelp("The model shuts its GPU down a few minutes after the last message, so nothing is billed while idle. "
                        + "Start it now from the model menu to have it ready.")
                .mgTestAnchor("cold-hint")
        }
    }
}

/// A line of the composer's suggestion list: a skill for "/", a project file for "@".
struct ComposerSuggestion: Identifiable, Equatable {
    var id: String { label }
    let label: String
    let detail: String
    /// A file's path (its icon).
    var file: String?
    /// The draft once it's chosen.
    let draft: String

    init(label: String, detail: String, draft: String) {
        self.label = label
        self.detail = detail
        self.draft = draft
    }

    init(file: String, draft: String) {
        label = (file as NSString).lastPathComponent
        let folder = (file as NSString).deletingLastPathComponent
        detail = folder
        self.file = file
        self.draft = draft
    }
}

enum ComposerSuggestions {
    /// The skills and /compact that match a "/" draft, until the command is followed by a space.
    static func slash(_ draft: String, skills: [Skill]) -> [ComposerSuggestion] {
        let typed = draft.dropFirst().prefix { !$0.isWhitespace }.lowercased()
        guard !draft.contains(where: \.isWhitespace) else { return [] }
        let all = [(name: "compact", detail: "Summarize the conversation to free up context")]
            + skills.filter(\.userInvocable).map { (name: $0.name, detail: $0.description.replacingOccurrences(of: "\n", with: " ")) }
        return all.filter { typed.isEmpty || $0.name.lowercased().hasPrefix(typed) }.prefix(6)
            .map { ComposerSuggestion(label: "/" + $0.name, detail: $0.detail, draft: "/\($0.name) ") }
    }

    /// The word after an "@" at the end of the draft (the "@" starts the draft or follows a space), nil when there's
    /// none: "fix @gre" → "gre"; "mail a@b" → nil.
    static func mentionQuery(_ draft: String) -> String? {
        guard let at = draft.lastIndex(of: "@") else { return nil }
        if at > draft.startIndex, !draft[draft.index(before: at)].isWhitespace { return nil }
        let word = draft[draft.index(after: at)...]
        guard !word.contains(where: { $0.isWhitespace || $0 == "`" }) else { return nil }
        return String(word)
    }

    /// The draft with its trailing "@word" replaced by the path in backticks (as Mention in Message writes it).
    static func accept(mention path: String, in draft: String) -> String {
        guard let at = draft.lastIndex(of: "@") else { return draft }
        return String(draft[..<at]) + "`\(path)` "
    }
}

/// The suggestions above the composer's field: ↑/↓ highlight, ↩ or ⇥ (or a click) choose.
struct ComposerSuggestionList: View {
    let items: [ComposerSuggestion]
    let selected: Int
    let choose: (ComposerSuggestion) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(items.enumerated()), id: \.element.id) { i, item in
                Button { choose(item) } label: {
                    HStack(spacing: 8) {
                        if let f = item.file {
                            Image(nsImage: FileIcons.icon(for: f)).resizable().frame(width: 14, height: 14)
                        }
                        Text(item.label).font(.mgCallout.weight(.medium)).lineLimit(1)
                        Text(item.detail).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.head)
                        Spacer(minLength: 0)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                        .fill(i == selected ? MG.accent.opacity(0.14) : .clear))
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
            Text("↑↓ to move · ↩ or ⇥ to choose · esc to close")
                .font(.mg(size: 10.5)).foregroundStyle(MG.tertiaryLabel)
                .padding(.horizontal, 10).padding(.vertical, 4)
        }
        .padding(3)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .mgTestAnchor("composer-suggestions")
    }
}

/// The message typed while a reply streams, above the field: sent when the reply ends.
struct QueuedBar: View {
    let text: String
    let onEdit: () -> Void
    let onRemove: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            RoundedRectangle(cornerRadius: 1).fill(MG.secondaryLabel).frame(width: 2)
            VStack(alignment: .leading, spacing: 2) {
                Label("Queued · sends when this reply ends", systemImage: "clock")
                    .font(.mgFootnote.weight(.medium)).foregroundStyle(MG.secondaryLabel)
                Text(text).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(3).truncationMode(.tail)
            }
            Spacer(minLength: 0)
            Button("Edit", action: onEdit).buttonStyle(.mg(.bordered)).controlSize(.small)
                .mgHelp("Back into the field, to change it")
            Button(action: onRemove) { Image(systemName: "xmark") }
                .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Don't send it")
        }
        .padding(.horizontal, 8).padding(.vertical, 6)
        .fixedSize(horizontal: false, vertical: true)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill))
        .mgTestAnchor("queued-bar")
    }
}

/// "Context: 45% of 262K tokens" under the field, and Compact.
struct ContextMeter: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        if doc.isAgent, let key = doc.modelKey, let limit = model.contextLimit(for: key), let used = doc.contextUsed {
            let share = Double(used) / Double(limit)
            HStack(spacing: 8) {
                Text("Context: \(Int((share * 100).rounded()))% of \(Self.tokens(limit)) tokens")
                    .font(.mgFootnote).foregroundStyle(share >= 0.75 ? MG.warning : MG.secondaryLabel)
                    .mgHelp("\(used) tokens used by the last request. Near the limit the agent summarizes older messages by itself.")
                Button {
                    model.compact(doc)
                } label: {
                    if doc.compacting {
                        HStack(spacing: 4) { ProgressView().controlSize(.mini); Text("Compacting…") }
                    } else {
                        Text("Compact")
                    }
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small)
                .disabled(doc.isStreaming || doc.compacting)
                .mgHelp("Replace the conversation so far with a summary, freeing up context")
                Spacer(minLength: 0)
            }
        }
    }

    static func tokens(_ n: Int) -> String {
        n >= 1_000_000 ? String(format: "%.1fM", Double(n) / 1e6) : n >= 1_000 ? "\(Int((Double(n) / 1000).rounded()))K" : "\(n)"
    }
}

/// Lays its views out in rows, wrapping to the next row when one is full (chips).
struct MGFlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let rows = arrange(width: proposal.width ?? .infinity, subviews)
        return CGSize(width: rows.map(\.width).max() ?? 0, height: rows.last.map { $0.y + $0.height } ?? 0)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for row in arrange(width: bounds.width, subviews) {
            var x = bounds.minX
            for i in row.items {
                let size = subviews[i].sizeThatFits(.unspecified)
                subviews[i].place(at: CGPoint(x: x, y: bounds.minY + row.y), proposal: ProposedViewSize(size))
                x += size.width + spacing
            }
        }
    }

    private struct Row { var items: [Int] = []; var width: CGFloat = 0; var height: CGFloat = 0; var y: CGFloat = 0 }

    private func arrange(width: CGFloat, _ subviews: Subviews) -> [Row] {
        var rows: [Row] = [Row()]
        for i in subviews.indices {
            let size = subviews[i].sizeThatFits(.unspecified)
            if !rows[rows.count - 1].items.isEmpty, rows[rows.count - 1].width + spacing + size.width > width {
                let last = rows[rows.count - 1]
                rows.append(Row(y: last.y + last.height + spacing))
            }
            var r = rows[rows.count - 1]
            r.width += (r.items.isEmpty ? 0 : spacing) + size.width
            r.height = max(r.height, size.height)
            r.items.append(i)
            rows[rows.count - 1] = r
        }
        return rows
    }
}
