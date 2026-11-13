import SwiftUI
import UniformTypeIdentifiers

/// The main window: icon rail and panel on the left, the graph (or a chat) and the results pane in the middle,
/// the inspector on the right, and the full-window viewer on top when it's open.
struct ContentView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("showInspector") private var showInspector = true
    @AppStorage("inspectorWidth") private var inspectorWidth = 330.0
    @AppStorage("helperWidth") private var helperWidth = 380.0
    @State private var dropTargeted = false
    @State private var showPlot = false
    @AppStorage("sidebarPanel") private var panelRaw = SidebarPanel.workflows.rawValue
    @AppStorage("sidebarOpen") private var panelOpen = true
    @AppStorage("sidebarWidth") private var sidebarWidth = 270.0

    var graphEditor: some View {
        GraphEditorView(doc: model.active)
            .id(model.activeTabID)
            .frame(minHeight: 240)
            .overlay(alignment: .bottom) { ErrorBanner() }
    }

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            MGTitleBar(bar: .unified, title: model.activeChat?.title ?? model.active.name,
                       subtitle: model.activeChat == nil && model.active.isDirty ? "Edited" : nil) {
                Button { withAnimation(.easeOut(duration: 0.18)) { panelOpen.toggle() } } label: {
                    Image(systemName: "sidebar.left")
                }
                .buttonStyle(.mg(.icon))
                .mgHelp(panelOpen ? "Hide the sidebar (⌃⌘S)" : "Show the sidebar (⌃⌘S)")
            } trailing: {
                if model.activeChat == nil { toolbarContent }
            }
            HStack(spacing: 0) {
                let panel = Binding(get: { SidebarPanel(rawValue: panelRaw) ?? .workflows }, set: { panelRaw = $0.rawValue })
                SidebarRail(selection: panel, open: $panelOpen)
                if panelOpen {
                    MGDivider(vertical: true)
                    AppSidebar(selection: panel.wrappedValue)
                        .frame(width: sidebarWidth)
                        .mgChrome(MG.sidebarTint)
                        .overlay(alignment: .trailing) {
                            MGResizeHandle(width: $sidebarWidth, range: 240...420, trailingColumn: false).offset(x: 3)
                        }
                }
                MGDivider(vertical: true, gap: true)
                VStack(spacing: 0) {
                    if model.activeChat != nil {
                        // The chat in front and the last few used stay built, hidden: going back to one shows it
                        // at once instead of building its transcript again. Hidden ones take no clicks, keys
                        // (a permission card's Return) or focus.
                        ZStack {
                            ForEach(model.keptChats) { chat in
                                let on = chat.id == model.activeChatID
                                ChatView(doc: chat, inFront: on)
                                    .opacity(on ? 1 : 0)
                                    .allowsHitTesting(on)
                                    .disabled(!on)
                                    .accessibilityHidden(!on)
                                    .zIndex(on ? 1 : 0)
                            }
                        }
                    } else if model.showResultsPane && ResultsPane.hasContent(model.active, model: model) {
                        VSplitView {
                            graphEditor
                            ResultsPane()
                                .frame(minHeight: 150, idealHeight: 280, maxHeight: .infinity)
                        }
                    } else {
                        // Nothing to show yet: the results pane is just its title bar, so the graph gets the room.
                        graphEditor
                        if model.showResultsPane { ResultsPane(collapsed: true) }
                    }
                }
                .background(MG.windowBackground)
                .overlay {
                    if dropTargeted {
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(MG.accent, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                            .padding(8)
                            .allowsHitTesting(false)
                    }
                }
                if model.showHelper && model.activeChat == nil {
                    MGDivider(vertical: true, gap: true)
                    HelperPanel(doc: model.active)
                        .id(model.activeTabID)
                        .frame(width: helperWidth)
                        .frame(maxHeight: .infinity)
                        .overlay(alignment: .leading) {
                            MGResizeHandle(width: $helperWidth, range: 320...600).offset(x: -3)
                        }
                }
                if showInspector && model.activeChat == nil {
                    MGDivider(vertical: true, gap: true)
                    InspectorView()
                        .frame(width: inspectorWidth)
                        .frame(maxHeight: .infinity)
                        .background(MG.groupedBackground)
                        .overlay(alignment: .leading) {
                            MGResizeHandle(width: $inspectorWidth, range: 280...460).offset(x: -3)
                        }
                }
            }
        }
        .ignoresSafeArea(.container, edges: .top)
        .overlay {
            if model.viewerResult != nil { ResultViewer().transition(.opacity) }
            if model.chatViewer != nil { ChatMediaViewer().transition(.opacity) }
        }
        .onChange(of: model.showServersPanel) { _, show in
            guard show else { return }
            panelRaw = SidebarPanel.servers.rawValue
            panelOpen = true
            model.showServersPanel = false
        }
        .navigationTitle(model.active.name + (model.active.isDirty ? " — Edited" : ""))
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in model.open(url) }
                }
            }
            return !providers.isEmpty
        }
        .task { if model.autoloadModel { model.preloadModel() } }
        .mgSheet("plot", isPresented: $showPlot) { PlotSheet() }
        .mgSheet("preset-gallery", isPresented: $model.showPresetGallery) { PresetGallery() }
        .mgSheet("mask-editor", item: $model.maskEditorTarget) { target in MaskEditorSheet(target: target) }
        .mgSheet("library-add", item: $model.libraryRequest) { r in
            if let lib = model.store?.library { LibraryAddSheet(request: r, library: lib) }
        }
        .mgSheet("save-preset", isPresented: $model.showSavePresetSheet) { SavePresetSheet() }
        .mgSheet("llm-add", item: $model.llmAddRequest) { r in LLMAddSheet(request: r) }
        .mgSheet("llm-settings", item: $model.llmSettingsRequest) { r in LLMSettingsSheet(request: r) }
        .task { model.llm.start() }
        .mgPresentationHost(model.presenter)
    }

    /// The tools in the title row, trailing.
    @ViewBuilder
    var toolbarContent: some View {
        HStack(spacing: 6) {
            MemoryGauge()
                .padding(.trailing, 6)
            if model.isRunning {
                Button { model.cancelCurrent() } label: { Image(systemName: "stop.fill") }
                    .buttonStyle(.mg(.icon, destructive: true))
                    .mgHelp("Cancel the running job (⌘.)")
                    .accessibilityLabel("Cancel")
            }
            Button { showPlot = true } label: { Image(systemName: "square.grid.3x3") }
                .buttonStyle(.mg(.icon))
                .mgHelp("XY Plot: queue a grid of results varying one or two inputs")
                .accessibilityLabel("XY Plot")
            RunTargetPicker()
            Button { model.enqueue(undoManager: model.activeUndoManager) } label: {
                let verb = model.isRunning ? "Queue" : "Generate"
                Label(model.batchCount > 1 ? "\(verb) \(model.batchCount)" : verb, systemImage: "sparkles")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.mg(.prominent))
            .mgHelp("Queue the current workflow (⌘↩)")
            Button { withAnimation { model.showResultsPane.toggle() } } label: {
                Image(systemName: "rectangle.bottomhalf.inset.filled")
            }
            .buttonStyle(.mg(.icon))
            .mgHelp("Show or hide the results pane (⌥⌘R)")
            .accessibilityLabel("Results")
            .mgTestAnchor("results-button")
            Button { withAnimation(.easeOut(duration: 0.18)) { model.showHelper.toggle() } } label: {
                Image(systemName: "wand.and.sparkles")
            }
            .buttonStyle(.mg(.icon))
            .mgHelp("Graph Helper: a model that builds, changes, runs and fixes this workflow (⇧⌘H)")
            .accessibilityLabel("Graph Helper")
            Button { withAnimation(.easeOut(duration: 0.18)) { showInspector.toggle() } } label: {
                Image(systemName: "sidebar.right")
            }
            .buttonStyle(.mg(.icon))
            .mgHelp("Show or hide the inspector")
            .accessibilityLabel("Inspector")
        }
    }
}

struct MemoryGauge: View {
    @Environment(AppModel.self) private var model
    @State private var showModels = false

    var body: some View {
        gauge
            .contentShape(Rectangle())
            .onTapGesture { showModels.toggle() }
            .mgPopover(isPresented: $showModels, size: CGSize(width: 320, height: LoadedModelsView.height(model))) {
                LoadedModelsView().environment(model)
            }
    }

    @ViewBuilder var gauge: some View {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        let used = Double(model.activeMemory)
        HStack(spacing: 6) {
            MGProgress(value: min(used / total, 1), tint: used / total > 0.8 ? MG.destructive : MG.accent)
                .frame(width: 70)
            // Binary GB, the way Macs are sold ("48 GB"), not 10⁹ bytes.
            Text(String(format: "%.1f of %.0f GB", used / 1_073_741_824, total / 1_073_741_824))
                .font(.mgSubheadline.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .mgHelp("\(model.modelStatus) — MLX active memory \(String(format: "%.1f", used / 1_073_741_824)) of "
              + "\(String(format: "%.0f", total / 1_073_741_824)) GB · click for loaded models")
    }
}

struct ErrorBanner: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let err = model.lastError {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(MG.warning)
                Text(err).font(.mgCallout).textSelection(.enabled).lineLimit(4)
                Spacer()
                Button { model.lastError = nil } label: { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon))
                    .controlSize(.small)
            }
            .padding(10)
            .mgGlass(.popover, radius: MGRadius.lg)
            .padding(12)
            .frame(maxWidth: 640)
        }
    }
}

/// Models in memory, with their estimated size and an Unload button.
struct LoadedModelsView: View {
    @Environment(AppModel.self) private var model

    /// Its height in the popover (it's sized up front, as floating panels are).
    static func height(_ model: AppModel) -> CGFloat {
        14 * 2 + 20 + (model.loadedModels.isEmpty ? 44 : CGFloat(model.loadedModels.count) * 44) + 11 + 62
    }

    var body: some View {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        VStack(alignment: .leading, spacing: 10) {
            Text("Loaded Models").font(.mgHeadline)
            if model.loadedModels.isEmpty {
                Text("None. Models load on first use and stay until memory is needed.")
                    .font(.mgSubheadline).foregroundStyle(.secondary)
            }
            ForEach(model.loadedModels) { m in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(m.name)
                        Text(String(format: "%.1f GB%@", Double(m.bytes) / 1e9, m.pinned ? " · in use" : ""))
                            .font(.mgSubheadline).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Unload") { model.unloadModel(m.key) }.disabled(m.pinned)
                }
            }
            MGDivider()
            Text(String(format: "Budget %.0f GB (%.0f%% of RAM). The least recently used model is unloaded when a new one doesn't fit; change it in Settings.",
                        total * model.memoryBudget / 1e9, model.memoryBudget * 100))
                .font(.mgSubheadline).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(width: 320)
    }
}

/// Floating "Sampler · step 3/8" status over the graph while a job runs (in any tab).
struct RunStatusPill: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.isRunning {
            let doc = model.document(id: model.runningDocID)
            let running = doc?.nodeStates.first { if case .running = $0.value { return true }; return false }
            HStack(spacing: 10) {
                MGSpinner().controlSize(.small)
                if let doc, doc.id != model.activeTabID {
                    Text(doc.name).font(.mgCallout.weight(.medium))
                }
                if let (id, state) = running, case .running(let step, let total) = state {
                    let name = model.displayName(flatID: id, in: doc)
                    Text(total > 0 ? "\(name) · step \(step) of \(total)" : "\(name)…")
                        .font(.mgCallout.monospacedDigit())
                } else {
                    Text("Starting…").font(.mgCallout)
                }
                if model.pendingCount > 0 {
                    Text("+\(model.pendingCount) queued").font(.mgSubheadline).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .mgGlass(.tooltip, in: Capsule())
            .onTapGesture { if let doc { model.activate(doc.id) } }
        }
    }
}
