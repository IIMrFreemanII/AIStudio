import AppKit
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the App so launch-argument hooks can drive the same model the window uses.
    @MainActor static weak var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running as a bare SwiftPM executable: become a regular foreground app with a Dock icon and menu bar.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        // Preview files only live for one session.
        try? FileManager.default.removeItem(at: previewDirectory)
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
        if let dir = Snapshot.requestedDirectory {
            Task { @MainActor in
                await Snapshot.renderAll(to: dir)
                NSApp.terminate(nil)
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

@main
struct MLXComfyApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @State private var model = AppModel()
    @State private var store = ModelStore()

    var body: some Scene {
        Window("MLXComfyUI", id: "main") {
            ContentView()
                .environment(model)
                .environment(store)
                .frame(minWidth: 1000, minHeight: 640)
                .onAppear {
                    AppDelegate.model = model
                    store.adopt(model.modelPath)
                }
        }
        .defaultSize(width: 1440, height: 900)
        .commands { AppCommands(model: model) }

        Window("Models", id: "models") {
            ModelManagerView()
                .environment(model)
                .environment(store)
        }
        .defaultSize(width: 680, height: 720)

        Settings {
            SettingsView()
                .environment(model)
                .environment(store)
        }
    }
}

struct AppCommands: Commands {
    let model: AppModel
    @Environment(\.undoManager) private var undoManager
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Graph") { model.newGraph(undoManager: undoManager) }
                .keyboardShortcut("n")
            Button("Open…") { openPanel() }
                .keyboardShortcut("o")
        }
        CommandGroup(replacing: .saveItem) {
            Button("Save Graph…") { savePanel() }
                .keyboardShortcut("s")
        }
        CommandMenu("Generate") {
            Button("Queue Prompt") { model.enqueue(undoManager: undoManager) }
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
            Button("Add Node…") {
                model.workspace = .graph
                model.paletteLocation = nil
                model.showPalette = true
            }
            .keyboardShortcut("k")
            Button("Duplicate") { model.duplicateSelection(undoManager: undoManager) }
                .keyboardShortcut("d")
                .disabled(model.selection.isEmpty)
            Button("Group Selection") { model.groupSelection(undoManager: undoManager) }
                .keyboardShortcut("g")
                .disabled(model.selection.isEmpty)
            Button("Collapse to Subgraph") { model.collapseSelection(undoManager: undoManager) }
                .keyboardShortcut("g", modifiers: [.command, .option])
                .disabled(model.selection.isEmpty)
            Button("Expand Subgraph") {
                if let id = model.selection.first { model.expandSubgraph(id, undoManager: undoManager) }
            }
            .keyboardShortcut("g", modifiers: [.command, .shift])
            .disabled(model.selection.count != 1
                      || model.editorGraph[model.selection.first!]?.type != "Subgraph")
            Button("Exit Subgraph") { model.exitSubgraph() }
                .keyboardShortcut(.upArrow, modifiers: .command)
                .disabled(model.editingPath.isEmpty)
            Divider()
            Button("Show Image") { model.workspace = .image }
                .keyboardShortcut("1")
            Button("Show Graph") { model.workspace = .graph }
                .keyboardShortcut("2")
        }
    }

    @MainActor func openPanel() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json, .png]
        panel.message = "Open a graph JSON file, or a PNG saved by MLXComfyUI to restore its graph."
        if panel.runModal() == .OK, let url = panel.url { model.open(url, undoManager: undoManager) }
    }

    @MainActor func savePanel() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = model.documentURL?.lastPathComponent ?? "graph.json"
        if panel.runModal() == .OK, let url = panel.url { model.save(to: url) }
    }
}
