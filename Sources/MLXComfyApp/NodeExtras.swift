import AppKit
import GraphRuntime
import MLXDiffusionCore
import NodesRemote
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

/// Extra controls in the inspector for nodes that have a friendlier way to set them: a model menu, size presets,
/// the prompt's style and assistant, the mask editor, a voice preview.
struct NodeExtras: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    let node: GraphNode
    private var undoManager: UndoManager? { model.activeUndoManager }

    var body: some View {
        // Only at the top level: the helpers rewire the top-level graph.
        if model.editingPath.isEmpty {
            switch node.type {
            case "LoadModel": modelSection
            case "EmptyLatent": sizeSection(GraphTemplates.sizePresets, video: false)
            case "EmptyVideoLatent":
                sizeSection(VideoFamily.of(path: modelPathOfGraph) == .wan ? GraphTemplates.wanSizePresets
                            : GraphTemplates.videoSizePresets, video: true)
            case "TextEncode": promptSection
            case "LoadImage", "LoadMask": imageSection
            case "LoadVideo", "LoadAudio": mediaSection
            case "TextToSpeech": voiceSection
            case let t where RemoteNodes.types.contains(t): gpuServerSection
            case "RemoteLoRA":
                loraLinkPreview
                Text("A LoRA for the Cloud GPU image node (FLUX.1 dev or Qwen-Image), picked from the library (Models window). One "
                     + "added from a link is downloaded by the server itself; a file is uploaded with the first job. "
                     + "Chain up to four; strength 0 skips one.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            case "LoRA":
                loraLinkPreview
                Text("A LoRA for this workflow's model, picked from the library (Models window): add one from a "
                     + "Civitai or Hugging Face link, or a file. One that isn't on this Mac yet is downloaded on the "
                     + "first run. Trigger words from Civitai show on the node.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            default: EmptyView()
            }
        }
    }

    var b: LiteralBindings { LiteralBindings(model: model, undoManager: undoManager, editor: true) }

    /// The picked library LoRA's card.
    @ViewBuilder var loraLinkPreview: some View {
        if case .value(.string(let id))? = node.inputs["lora"], !id.isEmpty, let item = model.store?.library.item(id) {
            LibraryItemCard(node: node, item: item)
        }
    }

    // MARK: GPU server (Cloud GPU nodes)

    @ViewBuilder var gpuServerSection: some View {
        let store = model.connections
        let servers = store.gpuServers
        let selection = b.string(node.id, "server")
        MGSection("GPU Server") {
            if servers.isEmpty {
                Text("These nodes run on a GPU server: Modal (billed per second) or your own GPU box.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
                Button("Add a GPU Server…") { model.showServersPanel = true }
            } else {
                MGFormRow("Server") {
                    MGPicker(selection: selection,
                             options: [("", "First (\(servers[0].name))")] + servers.map { ($0.id, $0.name) }
                                + (selection.wrappedValue.isEmpty || servers.contains { $0.id == selection.wrappedValue }
                                   ? [] : [(selection.wrappedValue, "Removed server")]),
                             maxWidth: 220)
                }
                let id = selection.wrappedValue.isEmpty ? servers[0].id : selection.wrappedValue
                if case .gpuOnline(let info)? = store.status[id], let taskID = RemoteNodes.taskID(forNodeType: node.type),
                   let task = info.task(taskID) {
                    Text(task.installed
                         ? "\(task.gpu)" + (task.pricePerSecond.map { String(format: ", $%.2f per GPU-hour, billed per second", $0 * 3600) } ?? "")
                         : "\(task.title) isn't downloaded on this server yet: run its download step.")
                        .font(.mgSubheadline).foregroundStyle(task.installed ? MG.secondaryLabel : MG.warning)
                        .fixedSize(horizontal: false, vertical: true)
                    // A server with other built-in voices than this app knows: offer its own list.
                    if let voices = task.params?["voice"]?.choices, voices != RemoteNodes.speechVoices {
                        MGFormRow("Server voice") {
                            MGPicker(selection: b.string(node.id, "voice"), options: voices.map { ($0, $0) }, maxWidth: 220)
                        }
                    }
                }
            }
            if node.type == "RemoteWanTextToVideo" || node.type == "RemoteWanImageToVideo" {
                Text("The negative prompt works with fast on too: it steers inside the model (Normalized Attention "
                     + "Guidance, a few percent slower). With fast off, or guidance above 1, it uses classifier-free "
                     + "guidance, which runs each step twice.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            }
            if node.type == "RemoteTextToSpeech" {
                Text("Connect a reference clip to clone its voice (a few seconds; its transcript helps). Otherwise it "
                     + "speaks with the chosen voice, or with \"designed\", a new voice made from the description. "
                     + "With a built-in voice, the description sets the delivery (\"excited, whispering\").")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            }
        }
        .task { for c in servers where store.status[c.id] == nil || store.status[c.id] == .unknown { await store.test(c.id) } }
    }

    var modelPathOfGraph: String {
        guard let m = model.firstNode(ofType: "LoadModel"), case .value(.string(let p))? = m.inputs["path"] else { return "" }
        return p
    }

    // MARK: model

    @ViewBuilder var modelSection: some View {
        let path = b.string(node.id, "path")
        let catalog = ModelCatalog.entries.compactMap { e in store.installedURL(e.id).map { ($0.standardizedFileURL.path, e.name) } }
        // Fine-tunes installed from links (Models window › Models from Links).
        let linked = store.library.macModels().map { ($0.url.standardizedFileURL.path, "\($0.title) (fine-tune)") }
        let installed = catalog + linked
        MGSection("Model") {
            MGFormRow("Model") {
                let current = URL(fileURLWithPath: path.wrappedValue).standardizedFileURL.path
                MGPicker(selection: Binding(get: { current }, set: { p in setModel(p) }),
                         options: installed
                            + (installed.contains { $0.0 == current }
                               ? [] : [(current, (path.wrappedValue as NSString).lastPathComponent)]),
                         maxWidth: 220)
            }
            HStack {
                Button("Choose Folder…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    panel.canChooseFiles = false
                    panel.message = "Choose a model folder (with model_index.json or the model's config)."
                    if panel.runModal() == .OK, let url = panel.url { setModel(url.path) }
                }
                Spacer()
                Text(model.modelStatus).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
            }
        }
    }

    /// Points this loader at another model, rewiring image-to-image for models that need it (Flux).
    func setModel(_ path: String) {
        model.mutate("Change Model", undoManager: undoManager) { g in
            try? g.setValue(.string(path), node: node.id, input: "path")
            g = AppModel.syncingImg2Img(g)
        }
    }

    // MARK: size

    @ViewBuilder func sizeSection(_ presets: [(name: String, width: Int, height: Int)], video: Bool) -> some View {
        let w = b.int(node.id, "width"), h = b.int(node.id, "height")
        MGSection("Size") {
            MGFormRow("Preset") {
            MGPicker(selection: Binding(
                get: {
                    if w.wrappedValue == 0 && h.wrappedValue == 0 { return -2 }
                    return presets.firstIndex { $0.width == w.wrappedValue && $0.height == h.wrappedValue } ?? -1
                },
                set: { i in
                    guard i >= 0 || i == -2 else { return }
                    let (pw, ph) = i == -2 ? (0, 0) : (presets[i].width, presets[i].height)
                    model.mutate("Change Size", undoManager: undoManager) { g in
                        try? g.setValue(.int(pw), node: node.id, input: "width")
                        try? g.setValue(.int(ph), node: node.id, input: "height")
                    }
                }),
                options: (video ? [] : [(-2, "Match reference image")]) + presets.indices.map { ($0, presets[$0].name) }
                    + [(-1, "Custom")])
            }
        }
    }

    // MARK: prompt

    @ViewBuilder var promptSection: some View {
        MGSection("Prompt Tools") {
            if model.assistantMode != nil {
                HStack(spacing: 8) {
                    MGSpinner().controlSize(.small)
                    Text(model.assistantMode == .describe ? "Describing the image…" : "Rewriting the prompt…")
                        .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                    Spacer()
                    Button("Stop") { model.stopAssistant() }
                }
            } else {
                HStack {
                    Button { model.runAssistant(.enhance, node: node.id, undoManager: undoManager) } label: {
                        Label("Enhance", systemImage: "wand.and.stars")
                    }
                    .disabled(!model.canEnhance(node.id))
                    .mgHelp("Rewrite the idea as a detailed prompt (or the edit instruction as a clear one) with Qwen")
                    Button { model.runAssistant(.describe, node: node.id, undoManager: undoManager) } label: {
                        Label("Describe Image", systemImage: "text.viewfinder")
                    }
                    .disabled(!model.canDescribe(node.id))
                    .mgHelp("Write a prompt that describes the image wired to this node")
                }
                .controlSize(.small)
            }
            MGFormRow("Style") {
                MGPicker(selection: Binding(
                    get: { model.graph.metadata?[AppModel.styleKey(node.id)] },
                    set: { model.setStyle($0, node: node.id, undoManager: undoManager) }),
                         options: [(String?.none, "None")] + model.styles.map { (String?.some($0.name), $0.name) })
            }
            if let style = model.style(of: node.id) {
                Text(style.template).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(2)
            }
            // verbatim: as a Markdown string, __name__ would render as bold "name".
            Text(verbatim: "{a|b|c} picks one option per run; __name__ picks a line from wildcards/name.txt.")
                .font(.mgFootnote).foregroundStyle(.tertiary)
        }
    }

    // MARK: image

    @ViewBuilder var imageSection: some View {
        let path = model.resolvedPath(b.string(node.id, "path").wrappedValue)
        let isReference = model.referenceImageNode?.id == node.id
        MGSection(node.type == "LoadMask" ? "Mask" : "Image") {
            if !path.isEmpty {
                HStack(spacing: 10) {
                    Thumbnail(path: path, maskPath: isReference ? model.maskPath : nil).frame(width: 72, height: 72)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                    VStack(alignment: .leading, spacing: 3) {
                        Text((path as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        if let size = Thumbnail.pixelSize(path) {
                            Text(verbatim: "\(size.width) × \(size.height)").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        }
                        if isReference {
                            Text(model.maskPath != nil ? "Inpainting: only the painted area changes."
                                 : model.isOutpainting ? "Outpainting: the new border is generated."
                                 : model.isImg2Img ? "Image to image: redrawn from the prompt."
                                 : "Editing: describe the change in the prompt.")
                                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        }
                    }
                }
            }
            HStack {
                Button("Choose…") {
                    let binding = b.string(node.id, "path")
                    model.chooseAsset([.image], title: node.type == "LoadMask" ? "Mask" : "Image",
                                      message: "Images from elsewhere are copied into Assets › Input.",
                                      asMask: node.type == "LoadMask") { url in
                        binding.wrappedValue = model.storedPath(url)
                    }
                }
                if isReference, !path.isEmpty, model.hasImageSampler {
                    Button(model.maskPath == nil ? "Paint Mask…" : "Edit Mask…") {
                        model.maskEditorTarget = MaskEditorTarget(imageNode: node.id, imagePath: path)
                    }
                    if model.maskPath != nil { Button("Clear Mask") { model.remove(.inpaint) } }
                }
            }
            .controlSize(.small)
        }
    }

    // MARK: video & audio

    @ViewBuilder var mediaSection: some View {
        let stored = b.string(node.id, "path").wrappedValue
        let isVideo = node.type == "LoadVideo"
        MGSection(isVideo ? "Video" : "Sound") {
            if !stored.isEmpty {
                Text((stored as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
            }
            Button("Choose…") {
                let binding = b.string(node.id, "path")
                model.chooseAsset(isVideo ? [.movie] : [.audio], title: isVideo ? "Video" : "Sound",
                                  message: "Files from elsewhere are copied into Assets › Input.") { url in
                    binding.wrappedValue = model.storedPath(url)
                }
            }
            .controlSize(.small)
        }
    }

    // MARK: voice

    @ViewBuilder var voiceSection: some View {
        MGSection("Voice") {
            Button {
                model.previewVoice(b.string(node.id, "voice").wrappedValue, speed: b.double(node.id, "speed").wrappedValue,
                                   modelPath: modelPathOfGraph)
            } label: { Label("Preview Voice", systemImage: "play.circle") }
                .disabled(model.isRunning)
            Text("Grades (A to F) are from the model's authors: how well each voice was trained.")
                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
        }
    }
}
