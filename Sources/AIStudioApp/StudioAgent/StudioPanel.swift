import Agent
import AppKit
import LLMClient
import SwiftUI

/// The Studio agent's floating panel: over AIStudio (and only while it's active), movable and resizable anywhere,
/// remembered where you left it.
final class StudioPanelWindow: NSPanel, NSWindowDelegate {
    static let pillSize = NSSize(width: 340, height: 46)
    private weak var controller: StudioController?
    /// The frame before folding, put back when unfolded.
    private var unfolded: NSRect?

    @MainActor
    init(controller: StudioController) {
        self.controller = controller
        super.init(contentRect: NSRect(x: 0, y: 0, width: 380, height: 580),
                   styleMask: [.titled, .closable, .resizable, .fullSizeContentView, .nonactivatingPanel],
                   backing: .buffered, defer: true)
        title = "Studio Agent"
        titleVisibility = .hidden
        titlebarAppearsTransparent = true
        isFloatingPanel = true
        level = .floating
        hidesOnDeactivate = true
        becomesKeyOnlyIfNeeded = true
        isMovableByWindowBackground = true
        isReleasedWhenClosed = false
        isOpaque = false
        backgroundColor = .clear
        minSize = NSSize(width: 320, height: 320)
        collectionBehavior = [.fullScreenAuxiliary]
        standardWindowButton(.miniaturizeButton)?.isHidden = true
        standardWindowButton(.zoomButton)?.isHidden = true
        delegate = self
    }

    override var canBecomeKey: Bool { true }

    /// Esc doesn't close the panel (a panel's default): closing it stops the agent's reply. Its menus still close on Esc.
    override func cancelOperation(_ sender: Any?) {}

    @MainActor
    func show(_ model: AppModel) {
        if contentView == nil || !(contentView is NSHostingView<AnyView>) {
            let root = StudioPanelView().environment(model).environment(\.mgGlassBlending, .behindWindow).mgRoot()
            let host = NSHostingView(rootView: AnyView(root))
            host.sizingOptions = []  // the panel's size is the user's, not the content's
            contentView = host
            if !setFrameUsingName("StudioAgentPanel"), let main = controller?.mainWindow {
                // First time: along the main window's right edge.
                let f = main.frame
                setFrame(NSRect(x: f.maxX - 400, y: f.maxY - 640, width: 380, height: 580), display: false)
            }
            setFrameAutosaveName("StudioAgentPanel")
            // Never taller than the screen it's on (the message field would be out of reach).
            if let screen = (self.screen ?? NSScreen.main)?.visibleFrame, !screen.contains(frame) {
                let h = min(frame.height, screen.height - 20)
                setFrame(NSRect(x: min(max(frame.minX, screen.minX), screen.maxX - frame.width),
                                y: max(screen.minY + 10, min(frame.minY, screen.maxY - h)), width: frame.width, height: h), display: false)
            }
        }
        makeKeyAndOrderFront(nil)
    }

    /// Folds to a pill (the current step, and buttons to unfold or stop) and back.
    @MainActor
    func setFolded(_ folded: Bool) {
        if folded {
            unfolded = frame
            minSize = Self.pillSize
            let top = frame.maxY
            setFrame(NSRect(x: frame.minX, y: top - Self.pillSize.height, width: Self.pillSize.width, height: Self.pillSize.height),
                     display: true, animate: true)
            styleMask.remove(.resizable)
        } else {
            styleMask.insert(.resizable)
            minSize = NSSize(width: 320, height: 320)
            if let f = unfolded {
                setFrame(NSRect(x: frame.minX, y: frame.maxY - f.height, width: f.width, height: f.height), display: true, animate: true)
            }
        }
    }

    func windowWillClose(_ notification: Notification) {
        MainActor.assumeIsolated { controller?.panelClosed() }
    }
}

/// The panel: the model and pace on top, the step list, then the pause or Next bar and the message field.
struct StudioPanelView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let studio = model.studio
        Group {
            if let chat = studio.chat {
                if studio.folded {
                    StudioPill(studio: studio, chat: chat)
                } else {
                    VStack(spacing: 0) {
                        StudioPanelHeader(studio: studio, chat: chat)
                        ChatHeader(doc: chat, compact: true)
                        if let note = modelNote(chat) {
                            Label(note, systemImage: "info.circle")
                                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                                .fixedSize(horizontal: false, vertical: true)
                                .padding(.horizontal, 12).padding(.bottom, 6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        MGDivider()
                        StudioPlanCard(studio: studio)
                        if chat.transcript.turns.isEmpty {
                            StudioEmptyState(chat: chat)
                        } else {
                            ChatTranscriptView(doc: chat).id(chat.id)
                        }
                        StudioGateBar(studio: studio, chat: chat)
                        ChatComposer(doc: chat)
                    }
                    .environment(\.chatFiles, ChatFilesContext(index: nil, folder: model.chatStore.filesFolder(chat.id), root: nil))
                }
            } else {
                Color.clear
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(MG.windowBackground.opacity(0.55))
        .mgGlass(.floatingPanel, radius: 0)
        .ignoresSafeArea()
        .task {
            await model.llm.refreshAll()
            // A new conversation made before the model list came in: the model chats use.
            if let chat = studio.chat, chat.modelKey == nil, chat.transcript.turns.isEmpty, let k = model.defaultChatModel {
                chat.transcript.serverID = k.serverID
                chat.transcript.modelID = k.modelID
            }
        }
    }
}

extension StudioPanelView {
    /// What the chosen model can't do here: call tools (then it can only talk) or see images (then you check results).
    func modelNote(_ chat: ChatDocument) -> String? {
        guard let key = chat.modelKey, let m = model.llm.model(key) else { return nil }
        if m.toolParser == nil { return "This model can't call tools, so it can only talk: choose one that can." }
        if m.vision != true { return "This model can't see images: it will ask you to check what it makes." }
        return nil
    }
}

/// "Studio Agent", the pace menu, and New, Fold and Close.
struct StudioPanelHeader: View {
    @Environment(AppModel.self) private var model
    let studio: StudioController
    let chat: ChatDocument

    var body: some View {
        HStack(spacing: 6) {
            let architect = studio.cursor(.architect)
            StudioActorIcon(actor: .architect, state: chat.isStreaming ? (architect.waiting != nil ? .waiting : .working) : .idle,
                            size: 18, motions: chat.isStreaming ? studio.motions(architect) : [], moment: architect.moment)
                .mgTarget("studio-header-icon")
            Text("Studio Agent").font(.mgHeadline).lineLimit(1).fixedSize()
            Spacer(minLength: 4)
            MGMenuButton(kind: .plain) {
                for p in StudioPace.allCases {
                    MGMenuItem.button(p.title, checked: studio.pace == p) { studio.pace = p }
                }
                MGMenuItem.divider
                MGMenuItem.header(studio.pace.detail)
                MGMenuItem.divider
                MGMenuItem.button("Wander when idle", checked: studio.wanders) { studio.wanders.toggle() }
            } label: {
                Label(studio.pace.title, systemImage: studio.pace == .step ? "forward.frame" : studio.pace == .fast ? "hare" : "tortoise")
                    .font(.mgFootnote)
            }
            .mgHelp("How fast the agent's actions play out in the window")
            .mgTarget("studio-pace")
            Button { model.newStudioConversation() } label: { Image(systemName: "square.and.pencil") }
                .buttonStyle(.mg(.icon))
                .disabled(chat.isStreaming || chat.transcript.turns.isEmpty)
                .mgHelp("Start a new conversation (this one stays in Sessions)")
            Button { studio.setFolded(true) } label: { Image(systemName: "rectangle.compress.vertical") }
                .buttonStyle(.mg(.icon))
                .mgHelp("Fold the panel to a small bar")
            Button { model.closeStudioAgent() } label: { Image(systemName: "xmark") }
                .buttonStyle(.mg(.icon))
                .mgHelp("Close the panel (⌥⌘A opens it again; a reply in progress stops)")
        }
        .padding(.leading, 30)  // the title bar's close button
        .padding(.trailing, 10)
        .frame(height: 36)
    }
}

/// What the agent can do, with a few things to ask for.
struct StudioEmptyState: View {
    let chat: ChatDocument

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Spacer()
            StudioActorIcon(actor: .architect, size: 30)
            Text("I can use AIStudio for you").font(.mgHeadline)
            Text("Make a movie from an idea, write prompts, make images, video and sound, look at the results and fix what's wrong. You'll see where I'm working in the window; any click or key there pauses me. Paid runs wait for you to press Run on the cost sheet.")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(["Open my latest movie and tell me what's left to do",
                     "Look at the keyframes and retake any that are wrong",
                     "Make an image of a lighthouse at dawn on the cloud GPU"], id: \.self) { s in
                Button { chat.draft = s } label: {
                    Text(s).font(.mgFootnote).multilineTextAlignment(.leading)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.mg(.bordered))
            }
            Spacer()
        }
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }
}

/// Above the message field: paused (you used the window), Step pace's Next, or waiting for your Run.
struct StudioGateBar: View {
    @Environment(AppModel.self) private var model
    let studio: StudioController
    let chat: ChatDocument

    var body: some View {
        Group {
            if let waiting = studio.waitingOnUser {
                bar("hourglass", MG.warning, waiting) { EmptyView() }
            } else {
                switch studio.gate.state {
                case .running:
                    EmptyView()
                case .pausedByUser:
                    bar("pause.circle.fill", MG.accent, "Paused because you used the window") {
                        Button("Resume") { studio.gate.resume() }.buttonStyle(.mg(.prominent)).controlSize(.small)
                            .mgTarget("studio-resume")
                        Button("Stop") { model.stopReply(chat) }.buttonStyle(.mg(.bordered)).controlSize(.small)
                    }
                case .awaitingNext(let doing):
                    bar("forward.frame", MG.accent, "Next: " + doing + (studio.gate.queue.count > 1 ? " (+\(studio.gate.queue.count - 1) waiting)" : "")) {
                        Button("Next") { studio.gate.next() }.buttonStyle(.mg(.prominent)).controlSize(.small)
                            .keyboardShortcut(.return, modifiers: [.command])
                            .mgTarget("studio-next")
                        Button("Skip") { studio.gate.skip() }.buttonStyle(.mg(.bordered)).controlSize(.small)
                    }
                }
            }
        }
        .animation(.easeOut(duration: 0.15), value: studio.gate.state)
    }

    func bar<B: View>(_ icon: String, _ tint: Color, _ text: String, @ViewBuilder buttons: () -> B) -> some View {
        HStack(spacing: 8) {
            Image(systemName: icon).foregroundStyle(tint)
            Text(text).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(2)
            Spacer(minLength: 4)
            buttons()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(tint.opacity(0.10))
        .overlay(alignment: .top) { MGDivider() }
    }
}

/// The folded panel: what the agent is doing, Unfold, and Stop.
struct StudioPill: View {
    @Environment(AppModel.self) private var model
    let studio: StudioController
    let chat: ChatDocument

    var body: some View {
        HStack(spacing: 8) {
            MGStateDot(kind: chat.isStreaming ? (studio.gate.state == .running ? .busy : .idle) : .ready)
            Text(line).font(.mgFootnote).foregroundStyle(MG.label).lineLimit(1)
            Spacer(minLength: 4)
            if studio.gate.state == .pausedByUser {
                Button("Resume") { studio.gate.resume() }.buttonStyle(.mg(.bordered)).controlSize(.small)
            } else if case .awaitingNext = studio.gate.state {
                Button("Next") { studio.gate.next() }.buttonStyle(.mg(.bordered)).controlSize(.small)
            }
            if chat.isStreaming {
                Button { model.stopReply(chat) } label: { Image(systemName: "stop.fill") }
                    .buttonStyle(.mg(.icon, destructive: true)).mgHelp("Stop the agent")
            }
            Button { studio.setFolded(false) } label: { Image(systemName: "rectangle.expand.vertical") }
                .buttonStyle(.mg(.icon)).mgHelp("Show the whole panel")
        }
        .padding(.leading, 30).padding(.trailing, 10)
        .frame(maxHeight: .infinity)
    }

    var line: String {
        if let w = studio.waitingOnUser { return w }
        switch studio.gate.state {
        case .pausedByUser: return "Paused"
        case .awaitingNext(let d): return "Next: " + d
        case .running:
            if let c = studio.caption, chat.isStreaming { return c }
            return chat.isStreaming ? "Working…" : "Studio Agent"
        }
    }
}

/// One step in the panel's list: what the agent did (a ✓, spinner or ✗ and its title). A click goes back to where it
/// happened; free text edits can be put back; it opens to the details and the images the agent looked at.
struct StudioStepRow: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?
    @State private var open = false
    @State private var hovering = false

    var body: some View {
        let studio = model.studio
        let status = doc.toolStatus(result)
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 7) {
                icon(status).frame(width: 14)
                Text(result?.toolTitle ?? call.name).font(.mgFootnote).foregroundStyle(status == "denied" ? MG.secondaryLabel : MG.label)
                    .lineLimit(open ? nil : 1)
                    .strikethrough(studio.reverted.contains(call.id))
                Spacer(minLength: 4)
                if studio.reverts[call.id] != nil, !studio.reverted.contains(call.id), !doc.isStreaming {
                    Button { studio.revert(call.id) } label: { Image(systemName: "arrow.uturn.backward") }
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .mgHelp("Put back what was there before this change")
                        .opacity(hovering ? 1 : 0.001)
                }
                Button { open.toggle() } label: { Image(systemName: open ? "chevron.up" : "chevron.down") }
                    .buttonStyle(.mg(.icon)).controlSize(.small)
                    .mgHelp(open ? "Hide the details" : "Show the details")
                    .opacity(hovering || open ? 1 : 0.001)
            }
            .contentShape(Rectangle())
            .onTapGesture { studio.revisit(call.id) }
            .mgHelp(studio.revisits[call.id] != nil ? "Show where this happened" : "")
            if let request = doc.permission, request.call.id == call.id {
                PermissionCard(doc: doc, request: request)
                    .padding(10)
                    .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.warning.opacity(0.08)))
            }
            if open, let result {
                if !result.text.isEmpty {
                    Text(result.text).font(.mgMono).foregroundStyle(MG.secondaryLabel)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if !result.images.isEmpty {
                    AttachmentStrip(images: result.images, files: [], folder: model.chatStore.filesFolder(doc.id))
                }
            }
        }
        .padding(.vertical, 3).padding(.horizontal, 6)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(hovering ? MG.controlBackground : .clear))
        .onHover { hovering = $0 }
    }

    @ViewBuilder func icon(_ status: String) -> some View {
        switch status {
        case "running":
            if doc.permission?.call.id == call.id { Image(systemName: "questionmark.circle.fill").foregroundStyle(MG.warning) }
            else if model.studio.waitingOnUser != nil { Image(systemName: "hourglass").foregroundStyle(MG.warning) }
            else { MGSpinner().controlSize(.mini) }
        case "done": Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
        case "denied": Image(systemName: "minus.circle").foregroundStyle(MG.secondaryLabel)
        default: Image(systemName: "xmark.circle.fill").foregroundStyle(MG.destructive)
        }
    }
}
