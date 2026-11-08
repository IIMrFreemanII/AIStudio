import AppKit
import SwiftUI
import UniformTypeIdentifiers

final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Set by the App so launch-argument hooks can drive the same model the window uses.
    @MainActor static weak var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Running as a bare SwiftPM executable: become a regular foreground app with a Dock icon and menu bar.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let prompt = SmokeTest.requestedPrompt {
            Task { @MainActor in
                // Wait for SwiftUI to create the model.
                while AppDelegate.model == nil { try? await Task.sleep(nanoseconds: 50_000_000) }
                let ok = await SmokeTest.run(AppDelegate.model!, prompt: prompt)
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
}

@main
struct MLXComfyApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @State private var model = AppModel()

    var body: some Scene {
        Window("MLXComfyUI", id: "main") {
            ContentView()
                .environment(model)
                .frame(minWidth: 1000, minHeight: 640)
                .onAppear { AppDelegate.model = model }
        }
        .defaultSize(width: 1440, height: 900)
        .commands { AppCommands(model: model) }

        Settings {
            SettingsView()
                .environment(model)
        }
    }
}

struct AppCommands: Commands {
    let model: AppModel
    @Environment(\.undoManager) private var undoManager

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
