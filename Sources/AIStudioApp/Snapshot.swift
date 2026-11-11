import AppKit
import GraphRuntime
import LLMClient
import ImageIO
import MLX
import AIStudioLibrary
import MLXDiffusionCore
import NodesRemote
import NodesStandard
import SwiftUI

/// `AIStudioApp --snapshot <dir>` renders the main window offscreen in a few representative states and exits.
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
        let saved = ["sidebarPanel", "sidebarOpen", "historyGrid", "assetsTab", "wireStyle", "showModelWires", "showMinimap",
                     "snapToGrid", "appearance"].map { ($0, defaults.object(forKey: $0)) }
        defer { for (k, v) in saved { if let v { defaults.set(v, forKey: k) } else { defaults.removeObject(forKey: k) } } }
        // Every render is written twice, light and dark (`-light.png`, `-dark.png`), following the window's appearance.
        defaults.set(MGAppearance.system.rawValue, forKey: "appearance")
        print("font: \(MGFonts.description)")

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
            await render(sheet(MaskEditorView(imageURL: ref, existingMask: mask, saveDirectory: dir) { _ in }),
                         to: dir.appendingPathComponent("8-mask-editor.png"), size: CGSize(width: 900, height: 760))
            model.add(.outpaint)
            model.setHiRes(true, undoManager: nil)
            model.groupAddedChain(.hires)
            await render(model, to: dir.appendingPathComponent("9-outpaint-hires.png"))
            panel(nil)
            await render(model, to: dir.appendingPathComponent("10-workflow-inspector.png"))
            print("graph: " + model.graph.jsonString())
            await renderEditor(model, inpaint, to: dir)
        }
        model.closeTab(inpaint.id)

        // 11. Settings; 12–14. the XY plot sheet, the preset gallery and save as preset, on the window.
        await render(SettingsView().mgWindow("Settings").environment(model).environment(store),
                     to: dir.appendingPathComponent("11-settings.png"), size: CGSize(width: 600, height: 900))
        panel(.workflows)
        for (n, id, view) in [(12, "plot", AnyView(PlotSheet())), (13, "gallery", AnyView(PresetGallery())),
                              (14, "save-preset", AnyView(SavePresetSheet()))] {
            model.presenter.present(id: id) { view }
            await render(model, to: dir.appendingPathComponent("\(n)-\(["plot-sheet", "preset-gallery", "save-preset"][n - 12]).png"))
            model.presenter.dismiss(id)
        }

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
        defaults.set("output", forKey: "assetsTab")
        panel(.assets)
        await render(model, to: dir.appendingPathComponent("19-assets-output.png"))

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
        // 30. Assets › Input with an image, a mask, a video and a sound; a Load Image node using one.
        let cwdURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        m.importAssets([audioURL, videoURL] + (newestImage.map { [$0.url] } ?? []))
        m.importAssets([cwdURL.appendingPathComponent("examples/inpaint-mask.png")], asMask: true)
        let edit = m.openPreset(m.preset(id: "qwen-edit")!)
        if let ref = m.referenceImageNode?.id, let img = m.assets.first(where: { $0.kind == .image }) {
            m.setLiteral(.string(img.relativePath), node: ref, input: "path", undoManager: nil)
            m.selection = [ref]
        }
        defaults.set("input", forKey: "assetsTab")
        panel(.assets)
        await render(m, to: dir.appendingPathComponent("30-assets-input.png"))
        m.selection = []
        m.closeTab(edit.id)
        let st = ModelStore()
        st.adopt(m.modelPath)
        await render(ModelManagerView().mgWindow("Models").environment(m).environment(st),
                     to: dir.appendingPathComponent("28-models-window.png"), size: CGSize(width: 680, height: 900))
        m.loadedModels = [LoadedModel(key: "model:/Users/me/models/qwen-turbo-mlx", bytes: 24_100_000_000, pinned: false)]
        await render(onCanvas(MGPopoverSurface { LoadedModelsView().environment(m) }
                        .frame(width: 320, height: LoadedModelsView.height(m))),
                     to: dir.appendingPathComponent("29-loaded-models.png"), size: CGSize(width: 380, height: LoadedModelsView.height(m) + 60))

        await renderChat(to: dir)
        await renderChrome(m, to: dir)
    }

    /// 50: a chat tab mid-conversation (thinking, Markdown, code, stats), the Chats panel open.
    static func renderChat(to dir: URL) async {
        let m = makeModel(URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("outputs"))
        let key = LLMKey(serverID: "snapshot", modelID: "qwen3.8-flash-next")
        var t = ChatTranscript(title: "Add a --json flag to the CLI", serverID: key.serverID, modelID: key.modelID)
        t.turns = [
            ChatTurn(role: .user, text: "How do I add a --json flag to the CLI's list command?"),
            ChatTurn(role: .assistant, text: """
            Add a flag to the command and branch on it when printing:

            ```swift
            @Flag(help: "Print JSON instead of a table.")
            var json = false
            ```

            Then, in `run()`:

            1. Build the rows as before.
            2. If `json` is set, encode them with `JSONEncoder` and print the string.
            3. Otherwise print the **table**.

            | Flag | Output |
            |---|---|
            | none | table |
            | `--json` | JSON |
            """, reasoning: "The user wants a flag. ArgumentParser's @Flag is the way; keep the table as the default.",
                     model: key.modelID),
        ]
        t.turns[1].thinkingSeconds = 4
        t.turns[1].stats = RequestStats(seconds: 6.2, ttftMs: 410, tokPerSec: 142, completionTokens: 512, costUSD: 0.0156)
        t.turns.append(ChatTurn(role: .user, text: "And for the remote command?"))
        let doc = ChatDocument(transcript: t)
        doc.transcript.turns.append(ChatTurn(role: .assistant, text: "", reasoning: "Same pattern: the remote command", model: key.modelID))
        doc.liveTurnID = doc.transcript.turns.last?.id
        m.chats = [doc]
        m.activeChatID = doc.id
        var ready = LLMStatus(state: "ready", phase: "Ready", progress: 1)
        ready.gpu = "H200"
        ready.count = 2
        ready.costUSD = 0.42
        ready.usdPerSecond = 0.002522
        m.llm.statuses[key] = ready
        panel(.chats)
        await render(m, to: dir.appendingPathComponent("50-chat.png"))
    }

    /// 41–49: the design system's own menus, picker, tooltips, alert, prompt, file pickers and controls.
    static func renderChrome(_ model: AppModel, to dir: URL) async {
        let session = MGMenuSession(items: [], window: nil)
        func menu(_ items: [MGMenuItem], highlighted: Int? = nil, searchable: Bool = false, query: String = "",
                  width: CGFloat = 0) -> some View {
            let level = MGMenuLevel(items: items, searchable: searchable)
            level.query = query
            level.highlighted = highlighted
            level.minWidth = width
            let size = level.size(maxHeight: 420)
            level.frame = CGRect(origin: .zero, size: size)
            return MGMenuView(level: level, session: session).frame(width: size.width, height: size.height)
        }
        let node = nodeMenuItems(NodeMenu(toggleCollapse: {}, copy: {}, duplicate: {}, arrange: {}, align: { _ in },
                                          alignCount: 3, group: {}, collapse: {}, delete: {}))
        let align = MGMenuItem.alignItems(count: 3) { _ in }
        await render(onCanvas(HStack(alignment: .top, spacing: -6) {
            menu(node, highlighted: 5)
            menu(align, highlighted: 1).padding(.top, 5 + 22 * 5)
        }), to: dir.appendingPathComponent("41-context-menu.png"), size: CGSize(width: 620, height: 380))
        let models = ["Qwen-Image 2.1", "Qwen-Image Edit", "FLUX.1 [dev]", "FLUX.1 [schnell]", "LTX-Video 2", "Wan 2.2 T2V",
                      "Wan 2.2 I2V", "Stable Audio Open", "ACE-Step", "Kokoro", "SDXL Turbo", "SD 3.5 Medium", "Qwen3 4B"]
        await render(onCanvas(VStack(alignment: .leading, spacing: 4) {
            MGPicker(selection: .constant("Qwen-Image 2.1"), options: models.map { ($0, $0) })
            menu(models.filter { $0.localizedCaseInsensitiveContains("qw") }.map { MGMenuItem.button($0, checked: $0 == "Qwen-Image 2.1") {} },
                 highlighted: 0, searchable: true, query: "qw", width: 220)
        }), to: dir.appendingPathComponent("42-picker-search.png"), size: CGSize(width: 360, height: 260))
        await render(onCanvas(VStack(alignment: .leading, spacing: 16) {
            let short = "Queue the current workflow (⌘↩)"
            MGTooltipView(text: short, card: false).frame(width: MGTooltipController.size(of: short).width,
                                                          height: MGTooltipController.size(of: short).height)
            let long = "Qwen-Image 2.1 (qwen-turbo-mlx) — MLX active memory 24.1 of 128 GB · click for loaded models"
            MGTooltipView(text: long, card: true).frame(width: MGTooltipController.size(of: long).width,
                                                        height: MGTooltipController.size(of: long).height)
        }), to: dir.appendingPathComponent("43-tooltips.png"), size: CGSize(width: 480, height: 160))

        // The alert, the rename prompt and both file pickers, on the window.
        UserDefaults.standard.set(SidebarPanel.workflows.rawValue, forKey: "sidebarPanel")
        _ = model.newTab()
        if let te = model.firstNode(ofType: "TextEncode") {
            model.setLiteral(.string("A lighthouse at dusk"), node: te.id, input: "prompt", undoManager: model.activeUndoManager)
        }
        TabCommands.close(model.active, model: model)
        await render(model, to: dir.appendingPathComponent("44-alert.png"))
        if let top = model.presenter.top { model.presenter.dismiss(top.id) }
        model.presenter.present(id: "rename") {
            MGPromptSheet(title: "Rename “Fox at Sunrise”", actionTitle: "Rename", text: "Fox at Sunrise") { _ in }
        }
        await render(model, to: dir.appendingPathComponent("45-rename.png"))
        model.presenter.dismiss("rename")
        model.chooseAsset([.image], title: "Image", message: "Images from elsewhere are copied into Assets › Input.") { _ in }
        await render(model, to: dir.appendingPathComponent("46-file-open.png"))
        model.presenter.dismiss("file-picker")
        TabCommands.saveAs(model.active, model: model)
        await render(model, to: dir.appendingPathComponent("47-file-save.png"))
        model.presenter.dismiss("file-picker")

        // Controls, and a list with its scroll thumb showing.
        MGScrollDebug.alwaysVisible = true
        await render(onCanvas(ControlGallery()), to: dir.appendingPathComponent("48-controls.png"), size: CGSize(width: 560, height: 360))
        MGScrollDebug.alwaysVisible = false
    }

    /// Content on the window background, centred (for surfaces rendered on their own).
    static func onCanvas<V: View>(_ v: V) -> some View {
        v.padding(24).frame(maxWidth: .infinity, maxHeight: .infinity).background(MG.windowBackground)
    }

    /// A sheet's content with the presenter's chrome, on the dimmed window background.
    static func sheet<V: View>(_ v: V) -> some View {
        v.mgSheetChrome().padding(24).frame(maxWidth: .infinity, maxHeight: .infinity).background(MG.windowBackground)
            .overlay(MG.scrim.allowsHitTesting(false).opacity(0.0))
    }

    struct ControlGallery: View {
        @State private var slider = 0.4
        @State private var steps = 4
        @State private var on = true
        @State private var mode = "Brush"

        var body: some View {
            HStack(alignment: .top, spacing: 24) {
                VStack(alignment: .leading, spacing: 14) {
                    MGFormRow("Brush") { MGSlider(value: $slider, in: 0...1).frame(width: 180) }
                    MGFormRow("Batch") { HStack { Text("\(steps) runs").foregroundStyle(MG.secondaryLabel); MGStepper(value: $steps, in: 1...16) } }
                    MGFormRow("Style") { MGPicker(selection: $mode, options: [("Brush", "Brush"), ("Eraser", "Eraser")]) }
                    MGFormRow("Add") { MGMenuButton("Add to Workflow") { MGMenuItem.button("Hi-res Fix") {} } }
                    MGProgress(value: 0.62).frame(width: 220)
                    MGProgress().frame(width: 220)
                    HStack(spacing: 14) {
                        MGSpinner().controlSize(.mini)
                        MGSpinner().controlSize(.small)
                        MGSpinner()
                    }
                }
                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        ForEach(1..<40) { i in Text("Workflow \(i)").mgRow(selected: i == 3) }
                    }
                }
                .mgScrollIndicators()
                .frame(width: 200, height: 300)
                .background(RoundedRectangle(cornerRadius: MGRadius.md).fill(MG.contentBackground))
            }
        }
    }

    /// 31–40: the graph editor on the inpaint preset with Outpaint and Hi-res added: before and after Arrange, each
    /// wire style, hidden and shown model wires, collapsed nodes, zoomed out with the minimap, the wire-drop palette.
    static func renderEditor(_ model: AppModel, _ doc: WorkflowDocument, to dir: URL) async {
        let d = UserDefaults.standard
        d.set(WireStyle.rightAngle.rawValue, forKey: "wireStyle")
        d.set(false, forKey: "showModelWires")
        d.set(true, forKey: "showMinimap")
        panel(nil)
        model.showResultsPane = false
        model.selection = []
        doc.viewport = nil
        await render(model, to: dir.appendingPathComponent("31-editor-spliced.png"))
        model.arrange(selectionOnly: false, undoManager: nil)
        doc.viewport = nil
        await render(model, to: dir.appendingPathComponent("32-editor-arranged.png"))
        for style in [WireStyle.curved, .straight] {
            d.set(style.rawValue, forKey: "wireStyle")
            await render(model, to: dir.appendingPathComponent("\(style == .curved ? 33 : 34)-wires-\(style.rawValue).png"))
        }
        d.set(WireStyle.rightAngle.rawValue, forKey: "wireStyle")
        if let loader = model.firstNode(ofType: "LoadModel") { model.selection = [loader.id] }
        await render(model, to: dir.appendingPathComponent("35-model-selected.png"))
        if let sampler = model.firstNode(ofType: "Sampler") { model.selection = [sampler.id] }
        d.set(true, forKey: "showModelWires")
        await render(model, to: dir.appendingPathComponent("36-model-wires-shown-path.png"))
        d.set(false, forKey: "showModelWires")
        model.selection = []
        let collapse = model.graph.nodes.filter { ["LoadModel", "EmptyLatent", "TextEncode", "FeatherMask"].contains($0.type) }.map(\.id)
        model.toggleCollapsed(Set(collapse), undoManager: nil)
        await render(model, to: dir.appendingPathComponent("37-collapsed.png"))
        model.toggleCollapsed(Set(collapse), undoManager: nil)
        // Zoomed in on the left of the graph so the rest is off screen (minimap shows), then far out.
        for (n, z) in [(38, 0.5), (39, 0.3)] as [(Int, CGFloat)] {
            doc.viewport = (z, CGSize(width: 40, height: 60))
            await render(model, to: dir.appendingPathComponent("\(n)-zoom-\(Int(z * 100)).png"))
        }
        doc.viewport = nil
        model.showResultsPane = true
        if let sampler = model.firstNode(ofType: "Sampler") {
            let wire = AppModel.PendingWire(node: sampler.id, socket: "latent", type: .latent, fromOutput: true,
                                            point: GraphPoint(x: 0, y: 0))
            await render(sheet(NodePalette(wire: wire) { _ in }.environment(model)), to: dir.appendingPathComponent("40-wire-palette.png"),
                         size: CGSize(width: 480, height: 420))
        }

        // 49. The Servers panel with this tab sent to a server (nothing listens there, so it shows offline; the public
        // plain-http one is never contacted and shows its warning); 50. the editor.
        let studio = ServerConnection(name: "Studio Mac", baseURL: "http://127.0.0.1:9")
        let rented = ServerConnection(name: "Rented Mac", baseURL: "http://203.0.113.7:8188")
        model.connections.save(studio, token: "snapshot")
        model.connections.save(rented, token: "snapshot")
        model.active.backendID = studio.id
        panel(.servers)
        await model.connections.test(studio.id)
        await render(model, to: dir.appendingPathComponent("49-servers.png"))
        await render(sheet(ServerEditor(connection: rented).environment(model)), to: dir.appendingPathComponent("50-server-editor.png"),
                     size: CGSize(width: 500, height: 360))
        model.active.backendID = BackendID.local
        for c in [studio, rented] { model.connections.remove(c.id) }

        // 51. A GPU server in the Servers panel (a Modal deployment with Wan installed, FLUX not yet); 52. adding one;
        // 53. a Cloud GPU preset with its node selected (the inspector's server picker), after a run.
        let gpu = ServerConnection(name: "Modal", kind: .gpu, baseURL: "https://you--aistudio-gpu-api.modal.run")
        model.connections.save(gpu, token: "snapshot")
        model.connections.save(studio, token: "snapshot")
        model.connections.checksOnAppear = false
        let info = try? JSONDecoder().decode(GPUServerInfo.self, from: Data("""
            {"server":"aistudio-gpu","version":"0.1.0","api":1,"runner":"modal","idleSeconds":120,"tasks":[
             {"id":"wan22-t2v","title":"Wan 2.2 Text to Video","kind":"video","gpu":"H100","pricePerSecond":0.001097,"installed":true,"inputs":[]},
             {"id":"wan22-i2v","title":"Wan 2.2 Image to Video","kind":"video","gpu":"H100","pricePerSecond":0.001097,"installed":true,"inputs":["image"]},
             {"id":"flux1-dev","title":"FLUX.1 dev","kind":"image","gpu":"L40S","pricePerSecond":0.000542,"installed":false,"inputs":[]},
             {"id":"ace-step-xl","title":"ACE-Step 1.5 XL music","kind":"audio","gpu":"L40S","pricePerSecond":0.000542,"installed":true,"inputs":[]},
             {"id":"stable-audio-3","title":"Stable Audio 3 sound","kind":"audio","gpu":"L40S","pricePerSecond":0.000542,"installed":true,"inputs":[]},
             {"id":"qwen3-tts","title":"Qwen3-TTS speech","kind":"audio","gpu":"L40S","pricePerSecond":0.000542,"installed":true,"inputs":[],"optionalInputs":["reference"]},
             {"id":"test-video","title":"Test pattern video","kind":"video","gpu":"CPU","pricePerSecond":null,"installed":true,"inputs":[]}]}
            """.utf8))
        if let info { model.connections.status[gpu.id] = .gpuOnline(info) }
        await model.connections.test(studio.id)
        panel(.servers)
        await render(model, to: dir.appendingPathComponent("51-gpu-server.png"))
        await render(sheet(ServerEditor(connection: ServerConnection(name: "", kind: .gpu, baseURL: "https://")).environment(model)),
                     to: dir.appendingPathComponent("52-gpu-server-editor.png"), size: CGSize(width: 500, height: 420))
        if let preset = model.preset(id: "cloud-wan-t2v") {
            let doc = model.openPreset(preset)
            doc.nodeStates = ["cloud": .done(seconds: 61.4, peakMemory: nil), "save": .done(seconds: 0.2, peakMemory: nil)]
            doc.nodeStatus = ["cloud": "GPU 23.5 s + 48 s loading · ≈ $0.078"]
            model.selection = ["cloud"]
            panel(nil)
            await render(model, to: dir.appendingPathComponent("53-cloud-node.png"))
            model.selection = []
            model.closeTab(doc.id)
        }
        // 54. The Cloud GPU speech node: voice picker and the clone / design / built-in rule.
        if let preset = model.preset(id: "cloud-speech") {
            let doc = model.openPreset(preset)
            doc.nodeStates = ["cloud": .done(seconds: 4.1, peakMemory: nil), "save": .done(seconds: 0.1, peakMemory: nil)]
            doc.nodeStatus = ["cloud": "GPU 3.2 s · ≈ $0.0017"]
            model.selection = ["cloud"]
            panel(nil)
            await render(model, to: dir.appendingPathComponent("54-cloud-speech.png"))
            model.selection = []
            model.closeTab(doc.id)
        }
        // 55. A LoRA from the library, picked in the Cloud GPU LoRA node (a throwaway library, no network).
        if model.store == nil { model.store = Self.store }
        if let preset = model.preset(id: "cloud-flux-lora"), let lib = model.store?.library {
            let savedModels = ModelPaths.directory
            ModelPaths.directory = FileManager.default.temporaryDirectory.appendingPathComponent("snapshot-library-\(UUID().uuidString)")
            let thumbDir = ModelPaths.directory.appendingPathComponent("library/thumbs")
            try? FileManager.default.createDirectory(at: thumbDir, withIntermediateDirectories: true)
            try? ImageArray.savePNG(MLXArray.ones([64, 64, 4]) * MLXArray([0.85, 0.55, 0.35, 1.0] as [Float]),
                                    to: thumbDir.appendingPathComponent("hands.png"))
            try? Library.save([LibraryEntry(
                id: "hands", kind: .lora, family: "flux.1", title: "Hands XL · Hand F1D v2.0",
                link: "https://civitai.com/models/200255?modelVersionId=804967", source: nil, fileName: "Hand v2.safetensors",
                sha256: nil, bytes: 343_805_376, baseModel: "Flux.1 D", triggerWords: ["Detailed hand", "Hand", "Perfect hand"],
                thumbnail: "library/thumbs/hands.png", localFile: nil)])
            lib.reload()
            let doc = model.openPreset(preset)
            model.setLiteral(.string("hands"), node: "lora1", input: "lora", undoManager: nil)
            model.selection = ["lora1"]
            panel(nil)
            try? await Task.sleep(nanoseconds: 300_000_000)
            await render(model, to: dir.appendingPathComponent("55-lora-library.png"))
            model.selection = []
            model.closeTab(doc.id)
            // 56. The Models window with that LoRA in its LoRAs section.
            if let st = model.store {
                await render(ModelManagerView().mgWindow("Models").environment(model).environment(st),
                             to: dir.appendingPathComponent("56-models-loras.png"), size: CGSize(width: 760, height: 1400))
            }
            try? FileManager.default.removeItem(at: ModelPaths.directory)
            ModelPaths.directory = savedModels
            lib.reload()
        }
        model.connections.checksOnAppear = true
        for c in [gpu, studio] { model.connections.remove(c.id) }
    }

    static let appearances: [(String, NSAppearance.Name)] = [("light", .aqua), ("dark", .darkAqua)]

    /// `foo.png` → `foo-light.png` / `foo-dark.png`.
    static func variant(_ url: URL, _ name: String) -> URL {
        url.deletingLastPathComponent().appendingPathComponent(url.deletingPathExtension().lastPathComponent + "-\(name).png")
    }

    static func render<V: View>(_ view: V, to url: URL, size: CGSize) async {
        for (name, appearance) in appearances {
            let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                                  styleMask: [.titled], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.appearance = NSAppearance(named: appearance)
            let host = NSHostingView(rootView: view.mgRoot().frame(width: size.width, height: size.height))
            window.contentView = host
            window.orderFrontRegardless()
            try? await Task.sleep(nanoseconds: 800_000_000)
            guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { continue }
            host.cacheDisplay(in: host.bounds, to: rep)
            let out = variant(url, name)
            if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: out) }
            window.close()
            print("wrote \(out.path)")
        }
    }

    static func render(_ model: AppModel, to url: URL, size: CGSize = CGSize(width: 1440, height: 900)) async {
        for (name, appearance) in appearances {
            let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                                  styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
                                  backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.appearance = NSAppearance(named: appearance)
            let host = NSHostingView(rootView: ContentView().environment(model).environment(store).mgRoot()
                .frame(width: size.width, height: size.height))
            window.contentView = host
            window.orderFrontRegardless()
            try? await Task.sleep(nanoseconds: 1_200_000_000)  // let layout, tasks and thumbnails settle
            host.layoutSubtreeIfNeeded()
            let view = window.contentView!.superview ?? host
            guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { continue }
            view.cacheDisplay(in: view.bounds, to: rep)
            let out = variant(url, name)
            if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: out) }
            window.close()
            print("wrote \(out.path)")
        }
    }
}
