import AppKit
import NodesStandard
import AIStudioLibrary
import SwiftUI
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the App so launch-argument hooks can drive the same model the window uses.
    @MainActor static weak var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running as a bare SwiftPM executable: become a regular foreground app with a Dock icon and menu bar.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Text fields' edit menu, drawn by the design system (event monitors need the running app).
        MainActor.assumeIsolated { MGTextMenu.install() }
        // Preview files only live for one session.
        try? FileManager.default.removeItem(at: previewDirectory)
        // LoRA links: the keys from the Keychain (Models window), read only when a site asks for one.
        LoRASources.tokenProvider = { host in
            let env = ProcessInfo.processInfo.environment
            switch host {
            case .civitai: return KeychainToken.civitai.read() ?? env["CIVITAI_TOKEN"]
            case .huggingface: return KeychainToken.huggingFace.read() ?? env["HF_TOKEN"]
            case .direct: return nil
            }
        }
        if let url = SmokeTest.argument("--smoke-gpu") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runGPU(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-graph-helper") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runGraphHelper(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-agent") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runAgent(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-sessions") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runSessions(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-code-graph") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runCodeGraph(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-git") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runGit(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-chat") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runChat(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if let url = SmokeTest.argument("--smoke-remote") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runRemote(AppDelegate.model!, url: url)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-sound") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runSound(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-speech") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runSpeech(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-music") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runMusic(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-video") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runVideo(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-wan") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runWan(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if let link = SmokeTest.argument("--smoke-wan-lora") {
            Task { @MainActor in
                while AppDelegate.model == nil || AppDelegate.model?.store == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runWanLoRA(AppDelegate.model!, link: link)
                exit(ok ? 0 : 1)
            }
        }
        if let link = SmokeTest.argument("--smoke-library-model") {
            Task { @MainActor in
                while AppDelegate.model == nil || AppDelegate.model?.store == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runLibraryModel(AppDelegate.model!, link: link)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-flux") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runFlux(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-media") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runMedia(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if let prompt = SmokeTest.requestedPrompt {
            Task { @MainActor in
                // Wait for SwiftUI to create the model.
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.run(AppDelegate.model!, prompt: prompt)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-xy") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runPlot(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if let idea = SmokeTest.argument("--smoke-assist") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runAssistant(AppDelegate.model!, idea: idea)
                exit(ok ? 0 : 1)
            }
        }
        // `--open <graph.json|png|mp4…>` opens a graph at launch; `--generate` also queues it, like pressing Generate.
        if let path = SmokeTest.argument("--open") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let model = AppDelegate.model!
                let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath,
                              relativeTo: URL(fileURLWithPath: FileManager.default.currentDirectoryPath))
                guard model.open(url.standardizedFileURL, undoManager: nil) else { return }
                if CommandLine.arguments.contains("--generate") {
                    try? await Task.sleep(nanoseconds: 300_000_000)
                    model.enqueue(undoManager: nil)
                }
            }
        }
        if CommandLine.arguments.contains("--smoke-workflows") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.runWorkflows(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if CommandLine.arguments.contains("--smoke-chrome") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeChrome.run(AppDelegate.model!)
                exit(ok ? 0 : 1)
            }
        }
        if let dir = SmokeTest.argument("--export-presets") {
            Task { @MainActor in
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let only = SmokeTest.argument("--only").map { Set($0.split(separator: ",").map(String.init)) }
                let ok = PresetExport.run(AppDelegate.model!, to: URL(fileURLWithPath: dir), only: only)
                exit(ok ? 0 : 1)
            }
        }
        if let dir = Snapshot.requestedDirectory {
            Task { @MainActor in
                await Snapshot.renderAll(to: dir)
                NSApp.terminate(nil)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Open tabs and their unsaved changes come back on the next launch.
    func applicationWillTerminate(_ notification: Notification) {
        MainActor.assumeIsolated {
            if let model = AppDelegate.model {
                // A reply streaming in the background is stopped and saved (the tab's text stays).
                for doc in model.chats + model.helperChats {
                    if doc.isStreaming || doc.permission != nil || doc.question != nil || doc.plan != nil {
                        model.stopReply(doc)
                        model.saveChat(doc)
                    }
                }
                model.chats.forEach { $0.jobs.stopAll() }
            }
            AppDelegate.model?.saveSession()
        }
    }
}

@main
struct AIStudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @State private var model: AppModel
    @State private var store: ModelStore
    @State private var modelsPresenter = MGPresenter()
    @State private var settingsPresenter = MGPresenter()

    init() {
        // An install from before the rename keeps its folders and settings; this runs before either is read.
        LegacyMigration.run()
        // JetBrains Mono, the design system's typeface, before any view draws.
        MGFonts.register()
        _model = State(initialValue: AppModel())
        _store = State(initialValue: ModelStore())
    }

    var body: some Scene {
        Window("AIStudio", id: "main") {
            ContentView()
                .environment(model)
                .environment(store)
                .mgRoot()
                .frame(minWidth: 1000, minHeight: 640)
                .onAppear {
                    AppDelegate.model = model
                    model.store = store
                    store.adopt(model.modelPath)
                }
        }
        .defaultSize(width: 1440, height: 900)
        .windowStyle(.hiddenTitleBar)
        .commands { AppCommands(model: model) }

        Window("Models", id: "models") {
            ModelManagerView()
                .mgWindow("Models")
                .mgPresentationHost(modelsPresenter)
                .environment(model)
                .environment(store)
                .mgRoot()
        }
        .defaultSize(width: 680, height: 720)
        .windowStyle(.hiddenTitleBar)

        Settings {
            SettingsView()
                .mgWindow("Settings")
                .mgPresentationHost(settingsPresenter)
                .environment(model)
                .environment(store)
                .mgRoot()
        }
    }
}

struct AppCommands: Commands {
    let model: AppModel
    @Environment(\.openWindow) private var openWindow
    @AppStorage("wireStyle") private var wireStyle = WireStyle.rightAngle.rawValue
    @AppStorage("showModelWires") private var showModelWires = false
    @AppStorage("showMinimap") private var showMinimap = true
    @AppStorage("snapToGrid") private var snapToGrid = true
    @AppStorage("appearance") private var appearance = MGAppearance.system.rawValue

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Workflow from Preset…") { model.showPresetGallery = true }
                .keyboardShortcut("n")
            Button("New Tab") { model.newTab() }
                .keyboardShortcut("t")
            Button("New Chat") { model.newChat() }
                .keyboardShortcut("n", modifiers: [.command, .option])
            Button("New Coding Chat…") { model.chooseProjectAndChat() }
                .keyboardShortcut("n", modifiers: [.command, .option, .shift])
            Button("Open…") { Self.openPanel(model: model) }
                .keyboardShortcut("o")
            Divider()
            Button("Close Tab") {
                if let chat = model.activeChat { model.closeChat(chat.id) } else { TabCommands.close(model.active, model: model) }
            }
            .keyboardShortcut("w")
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save") { TabCommands.save(model.active, model: model) }
                .keyboardShortcut("s")
            Button("Save As…") { TabCommands.saveAs(model.active, model: model) }
                .keyboardShortcut("s", modifiers: [.command, .shift])
            Button("Save as Preset…") { model.showSavePresetSheet = true }
        }
        // Each tab has its own undo history.
        CommandGroup(replacing: .undoRedo) {
            let _ = (model.undoRevision, model.activeTabID, model.activeChatID)
            if model.activeChat != nil {
                // A chat in front: its text field's own undo.
                Button("Undo") { NSApp.sendAction(Selector(("undo:")), to: nil, from: nil) }
                    .keyboardShortcut("z")
                Button("Redo") { NSApp.sendAction(Selector(("redo:")), to: nil, from: nil) }
                    .keyboardShortcut("z", modifiers: [.command, .shift])
            } else {
                Button(model.activeUndoManager.canUndo ? "Undo \(model.activeUndoManager.undoActionName)" : "Undo") {
                    model.activeUndoManager.undo()
                }
                .keyboardShortcut("z")
                .disabled(!model.activeUndoManager.canUndo)
                Button(model.activeUndoManager.canRedo ? "Redo \(model.activeUndoManager.redoActionName)" : "Redo") {
                    model.activeUndoManager.redo()
                }
                .keyboardShortcut("z", modifiers: [.command, .shift])
                .disabled(!model.activeUndoManager.canRedo)
            }
        }
        CommandMenu("Generate") {
            Button("Queue Workflow") { model.enqueue(undoManager: model.activeUndoManager) }
                .keyboardShortcut(.return, modifiers: .command)
            Button("Cancel Current") { model.cancelCurrent() }
                .keyboardShortcut(".", modifiers: .command)
                .disabled(!model.isRunning)
            Button("Cancel All") { model.cancelAll() }
                .keyboardShortcut(".", modifiers: [.command, .shift])
            Divider()
            Button("Models…") { openWindow(id: "models") }
                .keyboardShortcut("m", modifiers: [.command, .shift])
            Button("Load Model Now") { model.preloadModel() }
            Button("Unload Models") { model.unloadModels() }
        }
        CommandMenu("Graph") {
            let undo = model.activeUndoManager
            Button("Add Node…") {
                model.pendingWire = nil
                model.paletteLocation = nil
                model.showPalette = true
            }
            .keyboardShortcut("k")
            Menu("Add to Workflow") { MGNativeMenuItems(model.addToWorkflowItems()) }
            Divider()
            Button(model.selection.count > 1 ? "Arrange Selection" : "Arrange Graph") {
                model.arrange(selectionOnly: model.selection.count > 1, undoManager: undo)
            }
            .keyboardShortcut("l", modifiers: [.command, .shift])
            Menu("Align") {
                MGNativeMenuItems(MGMenuItem.alignItems(count: model.selection.count) { model.alignSelection($0, undoManager: undo) })
            }
            Button(model.selectionAllCollapsed ? "Expand Node" : "Collapse Node") {
                model.toggleCollapsed(model.selection, undoManager: undo)
            }
            .keyboardShortcut("c", modifiers: .option)
            .disabled(model.selection.isEmpty)
            Toggle("Snap to Grid", isOn: $snapToGrid)
            Divider()
            Button("Copy Nodes") { model.copySelection() }
                .disabled(model.selection.isEmpty)
            Button("Cut Nodes") { model.cutSelection(undoManager: undo) }
                .disabled(model.selection.isEmpty)
            Button("Paste Nodes") { model.paste(undoManager: undo) }
            Button("Paste Nodes with Connections") { model.paste(keepExternalLinks: true, undoManager: undo) }
                .keyboardShortcut("v", modifiers: [.command, .shift])
            Button("Duplicate") { model.duplicateSelection(undoManager: undo) }
                .keyboardShortcut("d")
                .disabled(model.selection.isEmpty)
            Button("Group Selection") { model.groupSelection(undoManager: undo) }
                .keyboardShortcut("g")
                .disabled(model.selection.isEmpty)
            Button("Collapse to Subgraph") { model.collapseSelection(undoManager: undo) }
                .keyboardShortcut("g", modifiers: [.command, .option])
                .disabled(model.selection.isEmpty)
            Button("Expand Subgraph") {
                if let id = model.selection.first { model.expandSubgraph(id, undoManager: undo) }
            }
            .keyboardShortcut("g", modifiers: [.command, .shift])
            .disabled(model.selection.count != 1
                      || model.editorGraph[model.selection.first!]?.type != "Subgraph")
            Button("Exit Subgraph") { model.exitSubgraph() }
                .keyboardShortcut(.upArrow, modifiers: .command)
                .disabled(model.editingPath.isEmpty)
        }
        CommandGroup(after: .toolbar) {
            Button(model.showResultsPane ? "Hide Results Pane" : "Show Results Pane") { model.showResultsPane.toggle() }
                .keyboardShortcut("r", modifiers: [.command, .option])
            Button(model.showHelper ? "Hide Graph Helper" : "Show Graph Helper") { model.showHelper.toggle() }
                .keyboardShortcut("h", modifiers: [.command, .shift])
            Button("Open Result in Viewer") { model.openViewer() }
                .keyboardShortcut("y")
                .disabled(model.displayedResult == nil)
            Button(model.activeChat?.changesOpen == true ? "Hide Changes" : "Show Changes") {
                if let chat = model.activeChat { model.toggleChanges(chat) }
            }
            .keyboardShortcut("d", modifiers: [.command, .shift])
            .disabled(model.activeChat?.isCoding != true)
            Divider()
            Picker("Appearance", selection: $appearance) {
                ForEach(MGAppearance.allCases) { Text($0.title).tag($0.rawValue) }
            }
            Picker("Wire Style", selection: $wireStyle) {
                ForEach(WireStyle.allCases) { Text($0.title).tag($0.rawValue) }
            }
            Toggle("Show Model Wires", isOn: $showModelWires)
                .keyboardShortcut("m", modifiers: [.command, .option])
            Toggle("Show Minimap", isOn: $showMinimap)
            Divider()
            Button("Show Next Tab") { model.selectTab(offset: 1) }
                .keyboardShortcut("]", modifiers: [.command, .shift])
            Button("Show Previous Tab") { model.selectTab(offset: -1) }
                .keyboardShortcut("[", modifiers: [.command, .shift])
            Divider()
        }
    }

    static let openableTypes: [UTType] = [.json, .png, .mpeg4Movie, .quickTimeMovie, .wav, .mpeg4Audio]

    /// File › Open: the workflow library in the app's picker (Finder for anywhere else).
    @MainActor static func openPanel(model: AppModel) {
        let message = "Open a workflow, or a PNG, video or sound made by AIStudio to restore its graph."
        model.presenter.chooseFile(
            title: "Open", message: message,
            roots: [MGFileRoot(title: "Workflows", url: model.library.workflows),
                    MGFileRoot(title: "Output", url: URL(fileURLWithPath: model.outputDirectory))],
            mode: .open(types: openableTypes, multiple: true),
            finder: {
                let panel = NSOpenPanel()
                panel.allowedContentTypes = openableTypes
                panel.allowsMultipleSelection = true
                panel.directoryURL = model.library.workflows
                panel.message = message
                return panel.runModal() == .OK ? panel.urls : nil
            },
            onChoose: { urls in for url in urls { model.open(url) } })
    }
}
