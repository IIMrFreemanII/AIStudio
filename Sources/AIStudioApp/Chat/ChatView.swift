import AppKit
import LLMClient
import SwiftUI

/// A chat tab: the model and its GPU at the top, the conversation, and the message field.
struct ChatView: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        VStack(spacing: 0) {
            ChatHeader(doc: doc)
            MGDivider()
            if doc.transcript.turns.isEmpty {
                ChatEmptyState(doc: doc)
            } else {
                ChatTranscriptView(doc: doc)
            }
            ChatComposer(doc: doc)
        }
        .background(MG.windowBackground)
        .task { await model.llm.refreshAll() }
    }
}

// MARK: header

struct ChatHeader: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        let llm = model.llm
        let key = doc.modelKey
        let m = key.flatMap { llm.model($0) }
        HStack(spacing: 10) {
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
            if let key {
                EngineChip(status: doc.engine.map { LLMStatus(state: $0.state, phase: $0.phase, progress: $0.progress) }
                           ?? llm.status(key), model: m, streaming: doc.isStreaming)
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
            Spacer(minLength: 8)
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
            if doc.transcript.costUSD > 0 {
                Text(String(format: "$%.3f", doc.transcript.costUSD))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .mgHelp("GPU time this chat's replies used")
            }
        }
        .padding(.horizontal, 14)
        .frame(height: 44)
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
        case "starting": return status?.phase ?? "Waiting for a GPU"
        case "loading":
            if let p = status?.progress, p < 1 { return "\(status?.phase ?? "Loading") \(Int(p * 100))%" }
            return status?.phase ?? "Loading"
        case "ready":
            var parts = [streaming ? "Generating" : "Ready", gpu]
            if let c = status?.costUSD, c > 0 { parts.append(String(format: "$%.2f", c)) }
            return parts.filter { !$0.isEmpty }.joined(separator: " · ")
        case "failed": return "Failed to start"
        default: return gpu.isEmpty ? "Cold" : "Cold · \(gpu)"
        }
    }

    func help(_ s: String) -> String {
        switch s {
        case "failed": return status?.error ?? "The model didn't start"
        case "ready":
            let rate = (status?.usdPerSecond ?? 0) * 3600
            return rate > 0 ? String(format: "Running: $%.2f an hour while warm. It shuts down a few minutes after the last message.", rate)
                : "Running"
        case "cold": return "No GPU running: the first message starts one (big models take a few minutes)."
        default: return "Starting the model"
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
                    ForEach(doc.transcript.turns) { turn in
                        ChatTurnView(doc: doc, turn: turn)
                            .id(turn.id)
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
        }
    }
}

struct ChatTurnView: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let turn: ChatTurn

    var live: Bool { doc.liveTurnID == turn.id }

    var body: some View {
        switch turn.role {
        case .user:
            HStack {
                Spacer(minLength: 80)
                Text(turn.text)
                    .font(.mgBody)
                    .textSelection(.enabled)
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.selection))
                    .mgContextMenu {
                        MGMenuItem.button("Copy") {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(turn.text, forType: .string)
                        }
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
                    Label("\(call.name)(\(call.arguments))", systemImage: "wrench.and.screwdriver")
                        .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2)
                }
                if let e = turn.error {
                    HStack(spacing: 8) {
                        Label(e, systemImage: e == "Stopped" ? "stop.circle" : "exclamationmark.triangle")
                            .font(.mgFootnote).foregroundStyle(e == "Stopped" ? MG.secondaryLabel : MG.destructive)
                            .fixedSize(horizontal: false, vertical: true)
                        if turn.id == doc.transcript.turns.last?.id, !doc.isStreaming {
                            Button("Retry") { retry() }.buttonStyle(.mg(.borderless)).font(.mgFootnote)
                        }
                    }
                }
                if !live, let s = turn.stats {
                    StatsLine(stats: s, model: turn.model)
                }
            }
            .mgContextMenu {
                MGMenuItem.button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(turn.text, forType: .string)
                }
                MGMenuItem.button("Regenerate", enabled: turn.id == doc.transcript.turns.last?.id && !doc.isStreaming) {
                    retry()
                }
            }
        case .tool:
            Text(turn.text).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(6)
        }
    }

    /// Replaces the last reply with a new one.
    func retry() {
        if doc.transcript.turns.last?.role == .assistant { doc.transcript.turns.removeLast() }
        model.reply(doc)
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

    var body: some View {
        HStack(alignment: .bottom, spacing: 10) {
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
                    Text("Message \(doc.transcript.modelID ?? "the model")…  (⇧↩ for a new line)")
                        .font(.mgBody).foregroundStyle(MG.placeholder)
                        .padding(.leading, 5).padding(.top, 1)
                        .allowsHitTesting(false)
                }
            }
            .padding(8)
            .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.controlBackground))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
            if doc.isStreaming {
                Button { model.stopReply(doc) } label: { Image(systemName: "stop.fill") }
                    .buttonStyle(.mg(.bordered))
                    .controlSize(.large)
                    .mgHelp("Stop the reply")
            } else {
                Button { model.send(doc) } label: { Image(systemName: "arrow.up") }
                    .buttonStyle(.mg(.prominent))
                    .controlSize(.large)
                    .disabled(doc.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .mgHelp("Send (↩)")
            }
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 12)
        .frame(maxWidth: 860)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
    }
}
