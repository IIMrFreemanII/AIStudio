import AppKit
import GraphRuntime
import ImageIO
import MLXDiffusionCore
import SwiftUI

/// `MLXComfyApp --snapshot <dir>` renders the main window offscreen in a few representative states and exits.
/// Used to check layouts without driving the GUI by hand.
@MainActor
enum Snapshot {
    static let store = ModelStore()

    static var requestedDirectory: URL? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count else { return nil }
        return URL(fileURLWithPath: args[i + 1])
    }

    static func renderAll(to dir: URL) async {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let projectOutputs = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("outputs")

        let model = AppModel(persistSettings: false, autoloadModel: false)
        model.outputDirectory = projectOutputs.path
        model.modelStatus = "Qwen-Image 2.1 (qwen-turbo-mlx)"
        if let first = model.history.first { model.show(first) }

        // 1. Image workspace, idle with a result.
        model.workspace = .image
        await render(model, to: dir.appendingPathComponent("1-image.png"))

        // 2. Image workspace mid-run: live preview, queue, node states.
        let g = model.graph
        model.queue = [
            QueueItem(graph: g, title: "A red fox standing in fresh snow at sunrise", seed: 1,
                      status: .done(seconds: 52.6)),
            QueueItem(graph: g, title: "A cozy coffee shop storefront with a wooden sign", seed: 7,
                      status: .running(step: 3, total: 8)),
            QueueItem(graph: g, title: "A lighthouse at dusk, long exposure", seed: 42),
        ]
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .cached,
                            "latent": .cached, "sampler": .running(step: 3, total: 8)]
        await render(model, to: dir.appendingPathComponent("2-running.png"))

        // 3. Graph workspace with a selected node.
        model.workspace = .graph
        model.selection = ["sampler"]
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .done(seconds: 3.3, peakMemory: 22_600_000_000),
                            "latent": .cached, "sampler": .running(step: 5, total: 8)]
        await render(model, to: dir.appendingPathComponent("3-graph.png"))

        // 4. Graph workspace, nothing selected (Simple panel in the inspector), idle.
        model.queue = []
        model.nodeStates = [:]
        model.selection = []
        await render(model, to: dir.appendingPathComponent("4-graph-idle.png"))

        // 5. Inpainting set up from the Simple panel: reference image + mask, graph rewired.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let ref = cwd.appendingPathComponent("refs/girl.png"), mask = cwd.appendingPathComponent("examples/inpaint-mask.png")
        if FileManager.default.fileExists(atPath: ref.path) {
            model.setReferenceImage(ref, undoManager: nil)
            model.setMask(mask, undoManager: nil)
            model.workspace = .graph
            await render(model, to: dir.appendingPathComponent("5-inpaint-graph.png"))
            model.workspace = .image
            await render(model, to: dir.appendingPathComponent("6-inpaint-panel.png"))
            // 7. The mask editor on its own.
            await render(MaskEditorView(imageURL: ref, existingMask: mask, saveDirectory: dir) { _ in },
                         to: dir.appendingPathComponent("7-mask-editor.png"), size: CGSize(width: 900, height: 760))
            // 8. Inpainting with the hi-res fix on: feather and hi-res controls in the panel.
            model.setHiRes(true, undoManager: nil)
            await render(model, to: dir.appendingPathComponent("8-inpaint-hires-panel.png"))
            // 9. Outpainting instead of inpainting, in the graph and the panel.
            model.setOutpaint(left: 256, top: 0, right: 256, bottom: 128, undoManager: nil)
            model.workspace = .graph
            await render(model, to: dir.appendingPathComponent("9-outpaint-graph.png"))
            model.workspace = .image
            await render(model, to: dir.appendingPathComponent("10-outpaint-panel.png"))
            print("graph: " + model.graph.jsonString())
            // 11. The prompt assistant mid-run (progress + Stop in place of Enhance / Describe).
            model.clearOutpaint(undoManager: nil)
            model.setHiRes(false, undoManager: nil)
            model.assistantMode = .describe
            await render(model, to: dir.appendingPathComponent("11-assistant-running.png"))
            model.assistantMode = nil
            await render(model, to: dir.appendingPathComponent("12-assistant-idle.png"))
            // 13. Settings with the assistant's instructions.
            await render(SettingsView().environment(model), to: dir.appendingPathComponent("13-settings.png"),
                         size: CGSize(width: 600, height: 900))
        }

        // 14. Style picker in the panel, the XY plot sheet.
        model.clearReferenceImage(undoManager: nil)
        model.selectedStyleName = "Cinematic"
        await render(model, to: dir.appendingPathComponent("14-style.png"))
        await render(PlotSheet().environment(model), to: dir.appendingPathComponent("15-plot-sheet.png"),
                     size: CGSize(width: 520, height: 620))
        model.selectedStyleName = nil

        // 16. History as a grid with a favorite; 17–19. A/B compare in each mode.
        let defaults = UserDefaults.standard
        let hadGrid = defaults.object(forKey: "historyGrid")
        defaults.set(true, forKey: "historyGrid")
        if model.history.count >= 2 {
            model.history[1].isFavorite = true
            model.show(model.history[0])
            model.compare(with: model.history[1].url)
            for (i, mode) in CompareMode.allCases.enumerated() {
                model.compareMode = mode
                await render(model, to: dir.appendingPathComponent("\(16 + i)-compare-\(mode.rawValue.lowercased().replacingOccurrences(of: " ", with: "-")).png"))
            }
            model.endCompare()
        }
        if let hadGrid { defaults.set(hadGrid, forKey: "historyGrid") } else { defaults.removeObject(forKey: "historyGrid") }
        // 19–21. Graph structure on a fresh default graph, built with the same AppModel calls as the editor:
        // a Reroute on the model wire, a Note, a group, then prompt/latent/sampler collapsed into a subgraph.
        let s = AppModel(persistSettings: false, autoloadModel: false)
        s.outputDirectory = projectOutputs.path
        s.workspace = .graph
        s.insertReroute(into: "sampler", input: "model", at: GraphPoint(x: 470, y: 230), undoManager: nil)
        s.addNode(type: "Note", at: GraphPoint(x: 40, y: 360), undoManager: nil)
        if let note = s.selection.first {
            s.setEditorLiteral(.string("Prompt, size and sampler live in the subgraph. Double-click it to edit."),
                               node: note, input: "text", undoManager: nil)
        }
        s.selection = ["model"]
        s.groupSelection(undoManager: nil)
        s.updateGroup(s.selectedGroup ?? "", actionName: "", undoManager: nil) { $0.title = "Model" }
        s.selectedGroup = nil
        s.selection = ["prompt", "latent", "sampler"]
        s.collapseSelection(undoManager: nil)
        s.selection = []
        await render(s, to: dir.appendingPathComponent("19-structure-graph.png"))
        if let inst = s.graph.nodes.first(where: { $0.type == Graph.subgraphType })?.id {
            s.enterSubgraph(inst)
            await render(s, to: dir.appendingPathComponent("20-inside-subgraph.png"))
            s.exitSubgraph()
            s.selection = [inst]
            await render(s, to: dir.appendingPathComponent("21-subgraph-inspector.png"))
        }
        print("structured: " + s.graph.jsonString())
        print("structured-valid: \(s.validationError ?? "ok")")

        // 22–26. Media: the Video mode panel, an audio and a video result, the Models window, loaded models.
        let m = AppModel(persistSettings: false, autoloadModel: false)
        m.outputDirectory = projectOutputs.path
        m.setSimpleMode(.video, videoModelPath: nil, undoManager: nil)
        await render(m, to: dir.appendingPathComponent("22-video-mode.png"))
        m.setSimpleMode(.image, videoModelPath: nil, undoManager: nil)
        let tone = MediaOps.mix(MediaOps.tone(frequency: 220, seconds: 3, sampleRate: 48000),
                                MediaOps.tone(frequency: 660, seconds: 1, sampleRate: 48000, amplitude: 0.6), offset: 1)
        let audioURL = dir.appendingPathComponent("snapshot-audio.wav")
        try? await MediaIO.writeAudio(tone, to: audioURL)
        m.displayedMedia = (audioURL, .audio)
        await render(m, to: dir.appendingPathComponent("23-audio-result.png"))
        let videoURL = dir.appendingPathComponent("snapshot-video.mp4")
        try? await MediaIO.writeVideo(MediaOps.testPattern(width: 640, height: 360, frames: 48, fps: 24), audio: tone, to: videoURL)
        m.displayedMedia = (videoURL, .video)
        await render(m, to: dir.appendingPathComponent("24-video-result.png"))
        let store = ModelStore()
        store.adopt(m.modelPath)
        await render(ModelManagerView().environment(m).environment(store),
                     to: dir.appendingPathComponent("25-models-window.png"), size: CGSize(width: 680, height: 900))
        m.loadedModels = [LoadedModel(key: "model:/Users/me/models/qwen-turbo-mlx", bytes: 24_100_000_000, pinned: false)]
        await render(LoadedModelsView().environment(m), to: dir.appendingPathComponent("26-loaded-models.png"),
                     size: CGSize(width: 340, height: 220))
    }

    static func render<V: View>(_ view: V, to url: URL, size: CGSize) async {
        let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: view.frame(width: size.width, height: size.height))
        window.contentView = host
        window.orderFrontRegardless()
        try? await Task.sleep(nanoseconds: 800_000_000)
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return }
        host.cacheDisplay(in: host.bounds, to: rep)
        if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: url) }
        window.close()
        print("wrote \(url.path)")
    }

    static func render(_ model: AppModel, to url: URL, size: CGSize = CGSize(width: 1440, height: 900)) async {
        let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                              styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: ContentView().environment(model).environment(store)
            .frame(width: size.width, height: size.height))
        window.contentView = host
        window.orderFrontRegardless()
        try? await Task.sleep(nanoseconds: 1_200_000_000)  // let layout, tasks and thumbnails settle
        host.layoutSubtreeIfNeeded()
        let view = window.contentView!.superview ?? host
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: url) }
        window.close()
        print("wrote \(url.path)")
    }
}
