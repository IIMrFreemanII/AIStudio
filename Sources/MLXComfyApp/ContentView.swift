import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    @State private var showInspector = true
    @State private var dropTargeted = false
    @State private var showPlot = false

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 220, ideal: 270, max: 380)
        } detail: {
            ZStack {
                switch model.workspace {
                case .image: CanvasView()
                case .graph: GraphEditorView()
                }
            }
            .overlay(alignment: .bottom) { ErrorBanner() }
            .overlay {
                if dropTargeted {
                    RoundedRectangle(cornerRadius: 12)
                        .strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [8, 6]))
                        .padding(8)
                        .allowsHitTesting(false)
                }
            }
            .inspector(isPresented: $showInspector) {
                InspectorView()
                    .inspectorColumnWidth(min: 280, ideal: 330, max: 460)
            }
        }
        .navigationTitle(model.documentURL?.deletingPathExtension().lastPathComponent ?? "MLXComfyUI")
        .toolbar { toolbarContent }
        .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
            guard let provider = providers.first else { return false }
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                guard let url else { return }
                Task { @MainActor in
                    if model.open(url, undoManager: undoManager) { model.workspace = .graph }
                }
            }
            return true
        }
        .task { if model.autoloadModel { model.preloadModel() } }
        .sheet(isPresented: $showPlot) { PlotSheet() }
    }

    @ToolbarContentBuilder
    var toolbarContent: some ToolbarContent {
        @Bindable var model = model
        ToolbarItem(placement: .principal) {
            Picker("Workspace", selection: $model.workspace) {
                Label("Image", systemImage: "photo").tag(Workspace.image)
                Label("Graph", systemImage: "point.3.connected.trianglepath.dotted").tag(Workspace.graph)
            }
            .pickerStyle(.segmented)
            .labelStyle(.titleAndIcon)
            .help("Switch between the image and the node graph (⌘1 / ⌘2)")
        }
        ToolbarItemGroup(placement: .primaryAction) {
            MemoryGauge()
            if model.isRunning {
                Button { model.cancelCurrent() } label: { Label("Cancel", systemImage: "stop.fill") }
                    .help("Cancel the running job (⌘.)")
            }
            Button { showPlot = true } label: { Label("XY Plot", systemImage: "square.grid.3x3") }
                .help("Queue a grid of images varying one or two inputs")
            Button { model.enqueue(undoManager: undoManager) } label: {
                Label(model.isRunning ? "Queue" : "Generate", systemImage: "sparkles")
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderedProminent)
            .help("Queue the current graph (⌘↩)")
            Button { showInspector.toggle() } label: { Label("Inspector", systemImage: "sidebar.right") }
        }
    }
}

struct MemoryGauge: View {
    @Environment(AppModel.self) private var model

    var body: some View {
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
              + "\(String(format: "%.0f", total / 1e9)) GB")
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
