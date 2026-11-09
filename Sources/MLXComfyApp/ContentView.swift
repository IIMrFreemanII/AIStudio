import SwiftUI
import UniformTypeIdentifiers

/// The main window: icon rail and panel on the left, workflow tabs over the graph and the results pane in the middle,
/// the inspector on the right, and the full-window viewer on top when it's open.
struct ContentView: View {
    @Environment(AppModel.self) private var model
    @State private var showInspector = true
    @State private var dropTargeted = false
    @State private var showPlot = false
    @AppStorage("sidebarPanel") private var panelRaw = SidebarPanel.workflows.rawValue
    @AppStorage("sidebarOpen") private var panelOpen = true

    var body: some View {
        @Bindable var model = model
        HStack(spacing: 0) {
            SidebarRail(selection: Binding(get: { SidebarPanel(rawValue: panelRaw) ?? .workflows },
                                           set: { panelRaw = $0.rawValue }),
                        open: $panelOpen)
            Divider()
            if panelOpen {
                SidebarPanelView(panel: SidebarPanel(rawValue: panelRaw) ?? .workflows)
                    .frame(width: 290)
                Divider()
            }
            VStack(spacing: 0) {
                TabStrip()
                Divider()
                VSplitView {
                    GraphEditorView(doc: model.active)
                        .id(model.activeTabID)
                        .frame(minHeight: 240)
                        .overlay(alignment: .bottom) { ErrorBanner() }
                    if model.showResultsPane {
                        ResultsPane()
                            .frame(minHeight: 150, idealHeight: 280, maxHeight: .infinity)
                    }
                }
            }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                        .padding(8)
                        .allowsHitTesting(false)
                }
            }
        }
        .inspector(isPresented: $showInspector) {
            InspectorView()
                .inspectorColumnWidth(min: 280, ideal: 330, max: 460)
        }
        .overlay {
            if model.viewerResult != nil { ResultViewer().transition(.opacity) }
        }
        .navigationTitle(model.active.name + (model.active.isDirty ? " — Edited" : ""))
        .toolbar { toolbarContent }
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
        .sheet(isPresented: $showPlot) { PlotSheet() }
        .sheet(isPresented: $model.showPresetGallery) { PresetGallery() }
        .sheet(item: $model.maskEditorTarget) { target in MaskEditorSheet(target: target) }
        .sheet(isPresented: $model.showSavePresetSheet) { SavePresetSheet() }
    }

    @ToolbarContentBuilder
    var toolbarContent: some ToolbarContent {
        @Bindable var model = model
        ToolbarItemGroup(placement: .primaryAction) {
            MemoryGauge()
            if model.isRunning {
                Button { model.cancelCurrent() } label: { Label("Cancel", systemImage: "stop.fill") }
                    .help("Cancel the running job (⌘.)")
            }
            Button { showPlot = true } label: { Label("XY Plot", systemImage: "square.grid.3x3") }
                .help("Queue a grid of results varying one or two inputs")
            Button { model.enqueue(undoManager: model.activeUndoManager) } label: {
                Label(model.isRunning ? "Queue" : "Generate", systemImage: "sparkles")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderedProminent)
            .help("Queue the current workflow (⌘↩)")
            Button { withAnimation { model.showResultsPane.toggle() } } label: {
                Label("Results", systemImage: "rectangle.bottomhalf.inset.filled")
            }
            .help("Show or hide the results pane (⌥⌘R)")
            Button { showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.right") }
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
            .popover(isPresented: $showModels, arrowEdge: .bottom) { LoadedModelsView().environment(model) }
    }

    @ViewBuilder var gauge: some View {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        let used = Double(model.activeMemory)
        HStack(spacing: 6) {
            Gauge(value: min(used / total, 1)) { EmptyView() }
                .gaugeStyle(.accessoryLinearCapacity)
                .tint(used / total > 0.8 ? .red : .accentColor)
                .frame(width: 70)
            Text(String(format: "%.1f GB", used / 1e9))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .help("\(model.modelStatus) — MLX active memory \(String(format: "%.1f", used / 1e9)) of "
              + "\(String(format: "%.0f", total / 1e9)) GB · click for loaded models")
    }
}

struct ErrorBanner: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let err = model.lastError {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.yellow)
                Text(err).font(.callout).textSelection(.enabled).lineLimit(4)
                Spacer()
                Button { model.lastError = nil } label: { Image(systemName: "xmark") }
                    .buttonStyle(.plain)
            }
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
            .padding(12)
            .frame(maxWidth: 640)
        }
    }
}

/// Models in memory, with their estimated size and an Unload button.
struct LoadedModelsView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        let total = Double(ProcessInfo.processInfo.physicalMemory)
        VStack(alignment: .leading, spacing: 10) {
            Text("Loaded Models").font(.headline)
            if model.loadedModels.isEmpty {
                Text("None. Models load on first use and stay until memory is needed.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            ForEach(model.loadedModels) { m in
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(m.name)
                        Text(String(format: "%.1f GB%@", Double(m.bytes) / 1e9, m.pinned ? " · in use" : ""))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Unload") { model.unloadModel(m.key) }.disabled(m.pinned)
                }
            }
            Divider()
            Text(String(format: "Budget %.0f GB (%.0f%% of RAM). The least recently used model is unloaded when a new one doesn't fit; change it in Settings.",
                        total * model.memoryBudget / 1e9, model.memoryBudget * 100))
                .font(.caption).foregroundStyle(.secondary)
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
                ProgressView().controlSize(.small)
                if let doc, doc.id != model.activeTabID {
                    Text(doc.name).font(.callout.weight(.medium))
                }
                if let (id, state) = running, case .running(let step, let total) = state {
                    let name = model.displayName(flatID: id, in: doc)
                    Text(total > 0 ? "\(name) · step \(step) of \(total)" : "\(name)…")
                        .font(.callout.monospacedDigit())
                } else {
                    Text("Starting…").font(.callout)
                }
                if model.pendingCount > 0 {
                    Text("+\(model.pendingCount) queued").font(.caption).foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.regularMaterial, in: Capsule())
            .onTapGesture { if let doc { model.activate(doc.id) } }
        }
    }
}
