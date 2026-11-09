import AppKit
import GraphRuntime
import ImageIO
import MLXComfyLibrary
import MLXDiffusionCore
import SwiftUI

/// `MLXComfyApp --snapshot <dir>` renders the main window offscreen in a few representative states and exits.
/// Used to check layouts without driving the GUI by hand. Runs against a throwaway library.
@MainActor
enum Snapshot {
    static let store = ModelStore()

    static var requestedDirectory: URL? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count else { return nil }
        return URL(fileURLWithPath: args[i + 1])
    }

    static func makeModel(_ outputs: URL) -> AppModel {
        let m = AppModel(persistSettings: false, autoloadModel: false)
        m.store = store
        m.outputDirectory = outputs.path
        m.modelStatus = "Qwen-Image 2.1 (qwen-turbo-mlx)"
        return m
    }

    /// Shows one sidebar panel in the next renders.
    static func panel(_ p: SidebarPanel?) {
        let d = UserDefaults.standard
        if let p { d.set(p.rawValue, forKey: "sidebarPanel"); d.set(true, forKey: "sidebarOpen") }
        else { d.set(false, forKey: "sidebarOpen") }
    }

    static func renderAll(to dir: URL) async {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let projectOutputs = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("outputs")
        let defaults = UserDefaults.standard
        let saved = ["sidebarPanel", "sidebarOpen", "historyGrid"].map { ($0, defaults.object(forKey: $0)) }
        defer { for (k, v) in saved { if let v { defaults.set(v, forKey: k) } else { defaults.removeObject(forKey: k) } } }

        let model = makeModel(projectOutputs)
        // Start from the newest image (smoke tests leave audio and video items on top of the history).
        let newestImage = model.history.first { $0.kind == .image }
        if let newestImage { model.show(newestImage) }
        // A couple of saved workflows for the library panel.
        _ = model.saveAs(model.active, name: "Fox at Sunrise")
        let v = model.openPreset(model.preset(id: "ltx-t2v")!)
        _ = model.saveAs(v, name: "Clips/Snow Walk")
        model.newWorkflowFolder(in: nil)
        model.activate(model.tabs[0].id)

        // 1. A workflow tab with its result in the results pane, the library open.
        panel(.workflows)
        await render(model, to: dir.appendingPathComponent("1-workflow.png"))

        // 2. Mid-run: queue, node states, the live preview under the sampler, a second (unsaved) tab.
        let g = model.graph
        model.queue = [
            QueueItem(graph: g, title: "A red fox standing in fresh snow at sunrise", seed: 1,
                      status: .done(seconds: 52.6), workflow: "Fox at Sunrise"),
            QueueItem(graph: g, title: "A cozy coffee shop storefront with a wooden sign", seed: 7,
                      status: .running(step: 3, total: 8), docID: model.activeTabID, workflow: "Fox at Sunrise"),
            QueueItem(graph: g, title: "A lighthouse at dusk, long exposure", seed: 42, workflow: "Untitled 2"),
        ]
        model.runningDocID = model.activeTabID
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .cached,
                            "latent": .cached, "sampler": .running(step: 3, total: 8)]
        model.active.previewImage = model.displayedImage
        model.active.previewNode = "sampler"
        model.setLiteral(.string("A cozy coffee shop storefront with a wooden sign"), node: "prompt", input: "prompt", undoManager: nil)
        panel(.queue)
        await render(model, to: dir.appendingPathComponent("2-running.png"))
        model.queue = []
        model.runningDocID = nil
        model.active.previewImage = nil
        model.active.previewNode = nil

        // 3. A selected sampler; 4. the Text Encode inspector (assistant, style); 5. the Load Model menu.
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .done(seconds: 3.3, peakMemory: 22_600_000_000),
                            "latent": .cached, "sampler": .done(seconds: 41.2, peakMemory: 24_000_000_000)]
        if let newestImage { model.active.record([EngineOutput(node: "save", kind: .image, image: model.displayedImage, url: newestImage.url)]) }
        model.selection = ["sampler"]
        panel(nil)
        await render(model, to: dir.appendingPathComponent("3-sampler-selected.png"))
        model.setStyle("Cinematic", node: "prompt", undoManager: nil)
        model.selection = ["prompt"]
        await render(model, to: dir.appendingPathComponent("4-text-encode.png"))
        model.selection = ["model"]
        await render(model, to: dir.appendingPathComponent("5-load-model.png"))
        model.setStyle(nil, node: "prompt", undoManager: nil)
        model.selection = []

        // 6–10. Add to Workflow chains on the inpaint preset with a generated picture; the mask editor on its own.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let ref = newestImage?.url ?? cwd.appendingPathComponent("examples/inpaint-mask.png")
        let mask = cwd.appendingPathComponent("examples/inpaint-mask.png")
        let inpaint = model.openPreset(model.preset(id: "qwen-inpaint")!)
        if let src = model.referenceImageNode?.id {
            model.setLiteral(.string(ref.path), node: src, input: "path", undoManager: nil)
            if let m = model.graph[AppModel.inpaintIDs.mask]?.id { model.setLiteral(.string(mask.path), node: m, input: "path", undoManager: nil) }
            panel(.presets)
            await render(model, to: dir.appendingPathComponent("6-inpaint-preset.png"))
            model.selection = [src]
            await render(model, to: dir.appendingPathComponent("7-load-image-inspector.png"))
            model.selection = []
            await render(MaskEditorView(imageURL: ref, existingMask: mask, saveDirectory: dir) { _ in },
                         to: dir.appendingPathComponent("8-mask-editor.png"), size: CGSize(width: 900, height: 760))
            model.add(.outpaint)
            model.setHiRes(true, undoManager: nil)
            model.groupAddedChain(.hires)
            await render(model, to: dir.appendingPathComponent("9-outpaint-hires.png"))
            panel(nil)
            await render(model, to: dir.appendingPathComponent("10-workflow-inspector.png"))
            print("graph: " + model.graph.jsonString())
        }
        model.closeTab(inpaint.id)

        // 11. Settings; 12. the XY plot sheet; 13. the preset gallery; 14. save as preset.
        await render(SettingsView().environment(model).environment(store), to: dir.appendingPathComponent("11-settings.png"),
                     size: CGSize(width: 600, height: 900))
        await render(PlotSheet().environment(model), to: dir.appendingPathComponent("12-plot-sheet.png"),
                     size: CGSize(width: 520, height: 620))
        await render(PresetGallery().environment(model).environment(store), to: dir.appendingPathComponent("13-preset-gallery.png"),
                     size: CGSize(width: 820, height: 640))
        await render(SavePresetSheet().environment(model), to: dir.appendingPathComponent("14-save-preset.png"),
                     size: CGSize(width: 440, height: 360))

        // 15–17. A/B compare in the results pane; 18. the viewer; 19. history as a grid in the Queue panel.
        let images = model.history.indices.filter { model.history[$0].kind == .image }
        if images.count >= 2 {
            model.history[images[1]].isFavorite = true
            model.show(model.history[images[0]])
            model.compare(with: model.history[images[1]].url)
            panel(nil)
            for (i, mode) in CompareMode.allCases.enumerated() {
                model.compareMode = mode
                await render(model, to: dir.appendingPathComponent("\(15 + i)-compare-\(mode.rawValue.lowercased().replacingOccurrences(of: " ", with: "-")).png"))
            }
            model.endCompare()
            model.openViewer()
            await render(model, to: dir.appendingPathComponent("18-viewer.png"))
            model.viewerResult = nil
        }
        defaults.set(true, forKey: "historyGrid")
        panel(.queue)
        await render(model, to: dir.appendingPathComponent("19-history-grid.png"))

        // 20–22. Graph structure on a fresh default graph, built with the same AppModel calls as the editor:
        // a Reroute on the model wire, a Note, a group, then prompt/latent/sampler collapsed into a subgraph.
        let s = makeModel(projectOutputs)
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
        panel(.nodes)
        await render(s, to: dir.appendingPathComponent("20-structure-graph.png"))
        if let inst = s.graph.nodes.first(where: { $0.type == Graph.subgraphType })?.id {
            s.enterSubgraph(inst)
            await render(s, to: dir.appendingPathComponent("21-inside-subgraph.png"))
            s.exitSubgraph()
            s.selection = [inst]
            await render(s, to: dir.appendingPathComponent("22-subgraph-inspector.png"))
        }
        print("structured: " + s.graph.jsonString())
        print("structured-valid: \(s.validationError ?? "ok")")

        // 23–28. Media presets with results: video, sound, music and speech tabs; the Models window; loaded models.
        let m = makeModel(projectOutputs)
        let tone = MediaOps.mix(MediaOps.tone(frequency: 220, seconds: 3, sampleRate: 48000),
                                MediaOps.tone(frequency: 660, seconds: 1, sampleRate: 48000, amplitude: 0.6), offset: 1)
        let audioURL = dir.appendingPathComponent("snapshot-audio.wav")
        try? await MediaIO.writeAudio(tone, to: audioURL)
        let videoURL = dir.appendingPathComponent("snapshot-video.mp4")
        try? await MediaIO.writeVideo(MediaOps.testPattern(width: 640, height: 360, frames: 48, fps: 24), audio: tone, to: videoURL)
        let poster = await MediaIO.posterFrame(videoURL)
        let vt = m.openPreset(m.preset(id: "ltx-t2v")!)
        vt.record([EngineOutput(node: "save", kind: .video, image: poster, url: videoURL)])
        panel(.presets)
        await render(m, to: dir.appendingPathComponent("23-video-result.png"))
        m.openViewer()
        await render(m, to: dir.appendingPathComponent("24-video-viewer.png"))
        m.viewerResult = nil
        for (i, id) in ["sound", "music", "speech"].enumerated() {
            let t = m.openPreset(m.preset(id: id)!)
            t.record([EngineOutput(node: "save", kind: .audio, image: nil, url: audioURL)])
            if id == "speech" { m.selection = ["speak"] }
            await render(m, to: dir.appendingPathComponent("\(25 + i)-\(id)-result.png"))
        }
        let st = ModelStore()
        st.adopt(m.modelPath)
        await render(ModelManagerView().environment(m).environment(st),
                     to: dir.appendingPathComponent("28-models-window.png"), size: CGSize(width: 680, height: 900))
        m.loadedModels = [LoadedModel(key: "model:/Users/me/models/qwen-turbo-mlx", bytes: 24_100_000_000, pinned: false)]
        await render(LoadedModelsView().environment(m), to: dir.appendingPathComponent("29-loaded-models.png"),
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
