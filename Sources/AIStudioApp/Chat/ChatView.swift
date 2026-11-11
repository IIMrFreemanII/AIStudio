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

    var body: some View {
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
            ChatComposer(doc: doc)
        }
        .background(MG.windowBackground)
        .onAppear { ChatQuoting.visibleChats += 1 }
        .onDisappear { ChatQuoting.visibleChats -= 1 }
        .onReceive(NotificationCenter.default.publisher(for: ChatQuoting.notification)) { note in
            guard let text = note.object as? String else { return }
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
    }
}

// MARK: header

struct ChatHeader: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    var compact = false

    var body: some View {
        if compact {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    modelMenu
                    Spacer(minLength: 4)
                    effortMenu
                }
                HStack(spacing: 8) {
                    engineItems
                    CodingHeaderItems(doc: doc)
                    Spacer(minLength: 4)
                    cost
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
        } else {
            HStack(spacing: 10) {
                modelMenu
                engineItems
                CodingHeaderItems(doc: doc)
                Spacer(minLength: 8)
                effortMenu
                cost
            }
            .padding(.horizontal, 14)
            .frame(height: 44)
        }
    }

    @ViewBuilder var modelMenu: some View {
        let llm = model.llm
        let key = doc.modelKey
        let m = key.flatMap { llm.model($0) }
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
                MGMenuItem.divider
                MGMenuItem.button("Add a Model from a Link…", icon: "link") { model.llmAddRequest = LLMAddRequest() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "cpu").foregroundStyle(MG.secondaryLabel)
                    Text(m?.name ?? doc.transcript.modelID ?? "Choose a model").lineLimit(1)
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
                }
            }
            .mgHelp("The model this chat talks to")
    }

    @ViewBuilder var engineItems: some View {
        let llm = model.llm
        let key = doc.modelKey
        let m = key.flatMap { llm.model($0) }
            if let key {
                EngineChip(status: doc.engine.map { e in
                               var st = LLMStatus(state: e.state, phase: e.phase, progress: e.progress)
                               st.coldStart = llm.status(key)?.coldStart  // the stream's status lines don't carry it
                               return st
                           } ?? llm.status(key), model: m, streaming: doc.isStreaming)
                let st = llm.status(key)
                if st?.isActive == true {
                    Button { Task { await llm.stop(key) } } label: { Image(systemName: "power") }
                        .buttonStyle(.mg(.icon))
                        .mgHelp("Shut the GPU down now (it also stops by itself a few minutes after the last message)")
                } else {
                    Button { Task { await llm.warm(key) } } label: { Image(systemName: "play.fill") }
                        .buttonStyle(.mg(.icon))
                        .mgHelp("Start the model now, so the first message doesn't wait for it")
                }
            }
    }

    @ViewBuilder var effortMenu: some View {
            MGMenuButton(kind: .icon) {
                MGMenuItem.header("Thinking effort")
                for (value, title) in [(nil, "Model's default"), ("low", "Low"), ("medium", "Medium"), ("high", "High")] as [(String?, String)] {
                    MGMenuItem.button(title, checked: doc.transcript.reasoningEffort == value) {
                        doc.transcript.reasoningEffort = value
                    }
                }
            } label: {
                Image(systemName: "brain")
            }
            .mgHelp("How hard thinking models think before answering")
    }

    @ViewBuilder var cost: some View {
            if doc.transcript.costUSD > 0 {
                Text(String(format: "$%.3f", doc.transcript.costUSD))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .mgHelp("GPU time this chat's replies used")
            }
    }
}

/// The model's engine at a glance: cold, waiting for a GPU, loading (with progress), ready (with price), failed.
struct EngineChip: View {
    let status: LLMStatus?
    let model: LLMModel?
    var streaming = false

    var body: some View {
        let s = status?.state ?? "cold"
        HStack(spacing: 6) {
            switch s {
            case "starting", "loading":
                MGSpinner().controlSize(.mini)
            case "ready":
                Circle().fill(MG.success).frame(width: 7, height: 7)
            case "failed":
                Circle().fill(MG.destructive).frame(width: 7, height: 7)
            default:
                Circle().fill(MG.tertiaryLabel).frame(width: 7, height: 7)
            }
            Text(label(s)).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
            if s == "loading", let p = status?.progress {
                MGProgress(value: p).frame(width: 60, height: 4)
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(Capsule().fill(MG.fill))
        .mgHelp(help(s))
    }

    func label(_ s: String) -> String {
        let gpu = status?.count.map { c in c > 1 ? "\(c)× \(status?.gpu ?? "")" : status?.gpu ?? "" } ?? model?.gpuLabel ?? ""
        switch s {
        case "starting": return (status?.phase ?? "Waiting for a GPU") + usually
        case "loading":
            if let p = status?.progress, p < 1 { return "\(status?.phase ?? "Loading") \(Int(p * 100))%" + usually }
            return (status?.phase ?? "Loading") + usually
        case "ready":
            var parts = [streaming ? "Generating" : "Ready", gpu]
            if let c = status?.costUSD, c > 0 { parts.append(String(format: "$%.2f", c)) }
            return parts.filter { !$0.isEmpty }.joined(separator: " · ")
        case "failed": return "Failed to start"
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

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    let shown = doc.transcript.turns.filter { $0.role != .tool }
                    ForEach(shown) { turn in
                        ChatTurnView(doc: doc, turn: turn)
                            .id(turn.id)
                            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous)
                                .fill(MG.accent.opacity(doc.highlightTurnID == turn.id ? 0.10 : 0)).padding(-6))
                            .animation(.easeOut(duration: 0.3), value: doc.highlightTurnID)
                    }
                    if let last = shown.last, last.role == .assistant, let items = last.followUps, !items.isEmpty,
                       !doc.isStreaming, doc.question == nil {
                        FollowUpChips(doc: doc, items: items)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 18)
                .frame(maxWidth: 860)
                .frame(maxWidth: .infinity)
            }
            .mgScrollIndicators()
            .onChange(of: doc.transcript.turns.last.map { $0.text.count + ($0.reasoning?.count ?? 0) }) { _, _ in
                if doc.isStreaming { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onChange(of: doc.transcript.turns.count) { _, _ in
                withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) }
            }
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: doc.highlightTurnID) { _, id in
                if let id { withAnimation(.easeOut(duration: 0.25)) { proxy.scrollTo(id, anchor: .center) } }
            }
            .onChange(of: doc.question?.callID) { _, id in
                if id != nil { withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo("bottom", anchor: .bottom) } }
            }
        }
    }
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

    var body: some View {
        HStack(alignment: .top, spacing: 8) {
            RoundedRectangle(cornerRadius: 1).fill(MG.accent).frame(width: 2)
            VStack(alignment: .leading, spacing: 2) {
                Label("Replying to \(quote.role == .user ? "your message" : quote.role == .tool ? "a tool result" : "the reply")",
                      systemImage: "arrowshape.turn.up.left")
                    .font(.mgFootnote.weight(.medium)).foregroundStyle(MG.secondaryLabel)
                Text(quote.text).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(2).truncationMode(.tail)
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

struct ChatTurnView: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let turn: ChatTurn
    @State private var editText = ""

    var live: Bool { doc.liveTurnID == turn.id }

    var body: some View {
        switch turn.role {
        case .user where turn.summary == true:
            SummaryRow(text: turn.text)
        case .user:
            VStack(alignment: .trailing, spacing: 6) {
                if !turn.images.isEmpty || !(turn.files ?? []).isEmpty {
                    AttachmentStrip(images: turn.images, files: (turn.files ?? []).map(\.name),
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
                        Text(turn.text)
                            .font(.mgBody)
                            .textSelection(.enabled)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.selection))
                    }
                }
                HStack(spacing: 6) {
                    BranchSwitcher(doc: doc, turn: turn)
                    if doc.editingTurnID != turn.id, !doc.isStreaming {
                        Button { model.quote(doc, turn.text, from: turn) } label: { Image(systemName: "arrowshape.turn.up.left") }
                            .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Reply to this message")
                        Button { editText = turn.text; doc.editingTurnID = turn.id } label: { Image(systemName: "pencil") }
                            .buttonStyle(.mg(.icon)).controlSize(.small)
                            .mgHelp("Edit and send again (the old version stays as a branch)")
                        if doc.isCoding {
                            Button { model.prepareRewind(doc, to: turn.id) } label: { Image(systemName: "clock.arrow.circlepath") }
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Rewind files and chat to before this message")
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .mgContextMenu {
                MGMenuItem.button("Copy") { copy(turn.text) }
                MGMenuItem.button("Reply") { model.quote(doc, turn.text, from: turn) }
                MGMenuItem.button("Edit…", enabled: !doc.isStreaming) { editText = turn.text; doc.editingTurnID = turn.id }
                if doc.isCoding {
                    MGMenuItem.button("Rewind to Here…", enabled: !doc.isStreaming) { model.prepareRewind(doc, to: turn.id) }
                }
            }
        case .assistant:
            VStack(alignment: .leading, spacing: 8) {
                if let r = turn.reasoning, !r.isEmpty {
                    ThinkingDisclosure(text: r, seconds: turn.thinkingSeconds, live: live && turn.text.isEmpty)
                }
                if !turn.text.isEmpty {
                    MarkdownView(text: turn.text)
                } else if live, turn.reasoning == nil {
                    WaitingLine(doc: doc)
                }
                ForEach(turn.toolCalls, id: \.id) { call in
                    let result = doc.transcript.turns.last { $0.toolCallID == call.id }
                    Group {
                        switch call.name {
                        case "ask_user": QuestionCard(doc: doc, call: call, result: result)
                        case "exit_plan": PlanCard(doc: doc, call: call, result: result)
                        case "show_widget": WidgetCard(doc: doc, call: call, result: result, live: live)
                        default: ToolCard(doc: doc, call: call, result: result)
                        }
                    }
                    .mgContextMenu {
                        MGMenuItem.button("Reply to This Result", enabled: result != nil) {
                            if let r = result { model.quote(doc, r.text.isEmpty ? (r.toolDetail ?? "") : r.text, from: r) }
                        }
                    }
                }
                if let e = turn.error {
                    Label(e, systemImage: e == "Stopped" ? "stop.circle" : "exclamationmark.triangle")
                        .font(.mgFootnote).foregroundStyle(e == "Stopped" ? MG.secondaryLabel : MG.destructive)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !live {
                    HStack(spacing: 6) {
                        BranchSwitcher(doc: doc, turn: turn)
                        if !doc.isStreaming {
                            Button { model.regenerate(doc, turn.id) } label: { Image(systemName: "arrow.clockwise") }
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Another reply (this one stays as a branch)")
                            Button { copy(turn.text) } label: { Image(systemName: "doc.on.doc") }
                                .buttonStyle(.mg(.icon)).controlSize(.small).mgHelp("Copy")
                            if !turn.text.isEmpty {
                                Button { model.quote(doc, turn.text, from: turn) } label: {
                                    Image(systemName: "arrowshape.turn.up.left")
                                }
                                .buttonStyle(.mg(.icon)).controlSize(.small)
                                .mgHelp("Reply to this message (or select a passage, then right-click › Reply with Selection)")
                            }
                        }
                        if let s = turn.stats { StatsLine(stats: s, model: turn.model) }
                    }
                }
            }
            .mgContextMenu {
                MGMenuItem.button("Copy") { copy(turn.text) }
                MGMenuItem.button("Reply") { model.quote(doc, turn.text, from: turn) }
                MGMenuItem.button("Regenerate", enabled: !doc.isStreaming) { model.regenerate(doc, turn.id) }
            }
        case .tool:
            Text(turn.text).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(6)
        }
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
                Button { doc.transcript.select(siblings[i - 1].id) } label: { Image(systemName: "chevron.left") }
                    .disabled(i == 0 || doc.isStreaming)
                Text("\(i + 1) / \(siblings.count)").font(.mgFootnote).foregroundStyle(MG.secondaryLabel).monospacedDigit()
                Button { doc.transcript.select(siblings[i + 1].id) } label: { Image(systemName: "chevron.right") }
                    .disabled(i == siblings.count - 1 || doc.isStreaming)
            }
            .buttonStyle(.mg(.icon))
            .controlSize(.small)
            .mgHelp("Versions of this message")
        }
    }
}

/// Attached images and videos (thumbnails) and file names.
struct AttachmentStrip: View {
    let images: [String]
    let files: [String]
    let folder: URL
    var onRemove: ((String) -> Void)? = nil

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(images, id: \.self) { name in
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
                        .onTapGesture(count: 2) { if video { NSWorkspace.shared.open(file) } }
                        .mgHelp(video ? "Video (double-click to play)" : "Image")
                        remove(name)
                    }
                }
                ForEach(files, id: \.self) { name in
                    HStack(spacing: 4) {
                        Image(systemName: "doc.text")
                        Text(name).lineLimit(1)
                        remove(name)
                    }
                    .font(.mgFootnote)
                    .padding(.horizontal, 8).frame(height: 26)
                    .background(Capsule().fill(MG.fill))
                }
            }
        }
        .frame(height: images.isEmpty ? 28 : 66)
    }

    @ViewBuilder
    func remove(_ name: String) -> some View {
        if let onRemove {
            Button { onRemove(name) } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain).foregroundStyle(MG.secondaryLabel).mgHelp("Remove")
        }
    }
}

/// "Thought for 12 s", collapsed once the answer starts; open while the model is still thinking.
struct ThinkingDisclosure: View {
    let text: String
    let seconds: Double?
    let live: Bool
    @State private var open = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.15)) { open.toggle() }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .bold))
                        .rotationEffect(.degrees(open || live ? 90 : 0))
                    Image(systemName: "brain")
                    Text(live ? "Thinking…" : seconds.map { "Thought for \(Self.duration($0))" } ?? "Thoughts")
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            .buttonStyle(.plain)
            if open || live {
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

struct StatsLine: View {
    let stats: RequestStats
    let model: String?

    var body: some View {
        var parts: [String] = []
        if let m = model { parts.append(m) }
        if let t = stats.tokPerSec { parts.append(String(format: "%.0f tok/s", t)) }
        if let c = stats.completionTokens { parts.append("\(c) tokens") }
        if let ttft = stats.ttftMs { parts.append(String(format: "first token %.1f s", ttft / 1000)) }
        if let w = stats.waitSeconds, w > 5 { parts.append(String(format: "waited %.0f s", w)) }
        if let c = stats.costUSD, c > 0 { parts.append(String(format: "$%.4f", c)) }
        return Text(parts.joined(separator: " · ")).font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
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
    @FocusState private var focused: Bool
    @State private var notice: String?

    var body: some View {
        let images = doc.pending.compactMap { if case .image(let n) = $0 { return n }; return nil }
        let files = doc.pending.compactMap { if case .file(let f) = $0 { return f.name }; return nil }
        let blind = !images.isEmpty && doc.modelKey.flatMap { model.llm.model($0) }.map { $0.vision != true } == true
        VStack(alignment: .leading, spacing: 6) {
            if doc.isCoding, !doc.todos.isEmpty, doc.todos.contains(where: { $0.status != "completed" }) || doc.isStreaming {
                TodoListView(todos: doc.todos)
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
            if doc.isCoding, doc.draft.hasPrefix("/") { SlashSuggestions(doc: doc) }
            HStack(alignment: .bottom, spacing: 10) {
                Button { choose() } label: { Image(systemName: "paperclip") }
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
                        .onKeyPress(.return, phases: .down) { press in
                            if press.modifiers.contains(.shift) || press.modifiers.contains(.option) { return .ignored }
                            model.send(doc)
                            return .handled
                        }
                    if doc.draft.isEmpty {
                        Text(doc.question != nil ? "Answer the question above, or type your own answer…"
                             : doc.isGraphHelper ? "Ask the helper to build, change or fix this workflow…"
                             : doc.isCoding ? "Ask the agent to change something in \(doc.projectURL?.lastPathComponent ?? "the project")…  (/ for skills, ⇧↩ for a new line)"
                             : "Message \(doc.transcript.modelID ?? "the model")…  (⇧↩ for a new line)")
                            .font(.mgBody).foregroundStyle(MG.placeholder)
                            .padding(.leading, 5).padding(.top, 1)
                            .allowsHitTesting(false)
                    }
                }
                .padding(8)
                .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.controlBackground))
                .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
                if doc.isStreaming && !(doc.question != nil && !doc.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) {
                    Button { model.stopReply(doc) } label: { Image(systemName: "stop.fill") }
                        .buttonStyle(.mg(.bordered))
                        .controlSize(.large)
                        .mgHelp("Stop the reply")
                } else {
                    Button { model.send(doc) } label: { Image(systemName: "arrow.up") }
                        .buttonStyle(.mg(.prominent))
                        .controlSize(.large)
                        .disabled((doc.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && doc.pending.isEmpty) || doc.preparingVideos > 0)
                        .mgHelp("Send (↩)")
                }
            }
            ContextMeter(doc: doc)
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
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

/// "/" in a coding chat's field: the skills and /compact that match what's typed.
struct SlashSuggestions: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var items: [(name: String, detail: String)] {
        let typed = doc.draft.dropFirst().prefix { !$0.isWhitespace }.lowercased()
        // Once a command is complete and followed by text, the list has done its job.
        guard !doc.draft.contains(where: \.isWhitespace) else { return [] }
        let skills = model.loadSkills(root: doc.projectURL).filter(\.userInvocable)
            .map { (name: $0.name, detail: $0.description.replacingOccurrences(of: "\n", with: " ")) }
        let all = [(name: "compact", detail: "Summarize the conversation to free up context")] + skills
        return Array(all.filter { typed.isEmpty || $0.name.lowercased().hasPrefix(typed) }.prefix(6))
    }

    var body: some View {
        let shown = items
        if !shown.isEmpty {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(shown, id: \.name) { item in
                    Button { doc.draft = "/\(item.name) " } label: {
                        HStack(spacing: 8) {
                            Text("/" + item.name).font(.mgCallout.weight(.medium))
                            Text(item.detail).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                            Spacer(minLength: 0)
                        }
                        .padding(.horizontal, 10).padding(.vertical, 5)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        }
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
