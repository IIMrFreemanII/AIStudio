import AppKit
import GraphRuntime
import ImageIO
import SwiftUI

/// Bindings onto literal node inputs, with undo.
@MainActor
struct LiteralBindings {
    let model: AppModel
    let undoManager: UndoManager?

    func string(_ node: String, _ input: String) -> Binding<String> {
        Binding(
            get: { model.literal(node, input)?.description ?? "" },
            set: { model.setLiteral(.string($0), node: node, input: input, undoManager: undoManager) })
    }

    func int(_ node: String, _ input: String) -> Binding<Int> {
        Binding(
            get: {
                switch model.literal(node, input) {
                case .int(let v)?: return v
                case .double(let d)?: return Int(d)
                default: return 0
                }
            },
            set: { model.setLiteral(.int($0), node: node, input: input, undoManager: undoManager) })
    }

    func double(_ node: String, _ input: String) -> Binding<Double> {
        Binding(
            get: {
                switch model.literal(node, input) {
                case .int(let v)?: return Double(v)
                case .double(let d)?: return d
                default: return 0
                }
            },
            set: { model.setLiteral(.double($0), node: node, input: input, undoManager: undoManager) })
    }

    func bool(_ node: String, _ input: String) -> Binding<Bool> {
        Binding(
            get: { if case .bool(let b)? = model.literal(node, input) { return b } else { return false } },
            set: { model.setLiteral(.bool($0), node: node, input: input, undoManager: undoManager) })
    }
}

struct InspectorView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.workspace == .graph, model.selection.count == 1, let id = model.selection.first,
           let node = model.graph[id] {
            NodeInspector(node: node)
        } else if model.workspace == .graph, model.selection.count > 1 {
            ContentUnavailableView("\(model.selection.count) nodes selected", systemImage: "square.on.square",
                                   description: Text("Select a single node to edit its values."))
        } else {
            SimplePanel()
        }
    }
}

/// Simple mode: edits the first TextEncode / EmptyLatent / Sampler nodes of whatever graph is loaded.
struct SimplePanel: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    @State private var showMaskEditor = false
    @State private var dropTargeted = false

    var body: some View {
        @Bindable var model = model
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let prompt = model.firstNode(ofType: "TextEncode")
        let latent = model.firstNode(ofType: "EmptyLatent")
        let sampler = model.firstNode(ofType: "Sampler")

        Form {
            Section("Prompt") {
                if let prompt {
                    TextEditor(text: b.string(prompt.id, "prompt"))
                        .font(.body)
                        .frame(minHeight: 120)
                        .scrollContentBackground(.hidden)
                } else {
                    missing("TextEncode")
                }
            }

            if prompt != nil { referenceSection }

            Section("Size") {
                if let latent {
                    let w = b.int(latent.id, "width"), h = b.int(latent.id, "height")
                    Picker("Preset", selection: Binding(
                        get: {
                            if w.wrappedValue == 0 && h.wrappedValue == 0 { return -2 }
                            return GraphTemplates.sizePresets.firstIndex { $0.width == w.wrappedValue && $0.height == h.wrappedValue } ?? -1
                        },
                        set: { i in
                            guard i >= 0 || i == -2 else { return }
                            let (pw, ph) = i == -2 ? (0, 0) : (GraphTemplates.sizePresets[i].width, GraphTemplates.sizePresets[i].height)
                            model.mutate("Change Size", undoManager: undoManager) { g in
                                try? g.setValue(.int(pw), node: latent.id, input: "width")
                                try? g.setValue(.int(ph), node: latent.id, input: "height")
                            }
                        })) {
                        if model.referenceImagePath != nil { Text("Match reference image").tag(-2) }
                        ForEach(GraphTemplates.sizePresets.indices, id: \.self) { i in
                            Text(GraphTemplates.sizePresets[i].name).tag(i)
                        }
                        Text("Custom").tag(-1)
                    }
                    HStack {
                        TextField("Width", value: w, format: .number.grouping(.never))
                        Text("×").foregroundStyle(.secondary)
                        TextField("Height", value: h, format: .number.grouping(.never))
                    }
                    .multilineTextAlignment(.trailing)
                } else {
                    missing("EmptyLatent")
                }
            }

            Section("Sampling") {
                if let sampler {
                    HStack {
                        TextField("Seed", value: b.int(sampler.id, "seed"), format: .number.grouping(.never))
                        Button { b.int(sampler.id, "seed").wrappedValue = Int.random(in: 0...Int(Int32.max)) } label: {
                            Image(systemName: "dice")
                        }
                        .buttonStyle(.borderless)
                        .help("New random seed")
                    }
                    Toggle("Randomize seed each run", isOn: $model.randomizeSeed)
                    Stepper(value: b.int(sampler.id, "steps"), in: 0...100) {
                        let s = b.int(sampler.id, "steps").wrappedValue
                        LabeledContent("Steps", value: s == 0 ? "Model default" : "\(s)")
                    }
                } else {
                    missing("Sampler")
                }
                if sampler != nil { hiResControls(b) }
                Stepper(value: $model.batchCount, in: 1...16) {
                    LabeledContent("Batch", value: "\(model.batchCount) image\(model.batchCount == 1 ? "" : "s")")
                }
            }

            Section("Model") {
                LabeledContent("Status") {
                    HStack(spacing: 6) {
                        if model.modelLoading { ProgressView().controlSize(.mini) }
                        Text(model.modelStatus).lineLimit(1).truncationMode(.middle)
                    }
                }
                LabeledContent("Folder") {
                    Text((model.modelPath as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                }
                Button("Choose Model Folder…") { chooseModel() }
            }
        }
        .formStyle(.grouped)
        .safeAreaInset(edge: .bottom) {
            Button { model.enqueue(undoManager: undoManager) } label: {
                Label(model.batchCount > 1 ? "Generate \(model.batchCount)" : "Generate", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .keyboardShortcut(.return, modifiers: .command)
            .padding(12)
            .background(.bar)
        }
    }

    @ViewBuilder var referenceSection: some View {
        let b = LiteralBindings(model: model, undoManager: undoManager)
        Section {
            if let path = model.referenceImagePath {
                HStack(spacing: 10) {
                    Thumbnail(path: path, maskPath: model.maskPath).frame(width: 64, height: 64)
                    VStack(alignment: .leading, spacing: 3) {
                        Text((path as NSString).lastPathComponent).lineLimit(1).truncationMode(.middle)
                        Text(model.maskPath != nil ? "Inpainting: only the painted area changes."
                             : model.isOutpainting ? "Outpainting: the new border is generated."
                             : "Editing: describe the change in the prompt.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                HStack {
                    Button(model.maskPath == nil ? "Paint Mask…" : "Edit Mask…") { showMaskEditor = true }
                    if model.maskPath != nil { Button("Clear Mask") { model.clearMask(undoManager: undoManager) } }
                    if model.maskPath == nil && !model.isOutpainting {
                        Button("Extend…") {
                            model.setOutpaint(left: 256, top: 0, right: 256, bottom: 0, undoManager: undoManager)
                        }
                        .help("Outpaint: grow the canvas and fill the new border")
                    }
                    Spacer()
                    Button("Remove", role: .destructive) { model.clearReferenceImage(undoManager: undoManager) }
                }
                if model.maskPath != nil, model.graph[AppModel.inpaintIDs.feather] != nil {
                    featherSlider(AppModel.inpaintIDs.feather, b)
                }
                if model.isOutpainting { outpaintControls(b) }
            } else {
                VStack(spacing: 6) {
                    Image(systemName: "photo.badge.plus").font(.title2).foregroundStyle(.secondary)
                    Text("Drop an image to edit it or use it as a reference").font(.caption)
                        .foregroundStyle(.secondary).multilineTextAlignment(.center)
                    Button("Choose…") { chooseReference() }.controlSize(.small)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .background(RoundedRectangle(cornerRadius: 8).strokeBorder(
                    dropTargeted ? Color.accentColor : Color.secondary.opacity(0.4),
                    style: StrokeStyle(lineWidth: dropTargeted ? 2 : 1, dash: [5, 4])))
                .onDrop(of: [.fileURL], isTargeted: $dropTargeted) { providers in
                    _ = providers.first?.loadObject(ofClass: URL.self) { url, _ in
                        guard let url else { return }
                        Task { @MainActor in model.setReferenceImage(url, undoManager: undoManager) }
                    }
                    return true
                }
            }
        } header: {
            Text("Reference Image")
        }
        .sheet(isPresented: $showMaskEditor) {
            if let path = model.referenceImagePath {
                MaskEditorView(imageURL: URL(fileURLWithPath: path),
                               existingMask: model.maskPath.map { URL(fileURLWithPath: $0) },
                               saveDirectory: model.maskDirectory) { url in
                    model.setMask(url, undoManager: undoManager)
                }
            }
        }
    }

    func featherSlider(_ node: String, _ b: LiteralBindings) -> some View {
        let f = b.int(node, "feather")
        return LabeledContent("Feather") {
            HStack {
                Slider(value: Binding(get: { Double(f.wrappedValue) }, set: { f.wrappedValue = Int($0.rounded()) }),
                       in: 0...96)
                Text("\(f.wrappedValue) px").monospacedDigit().frame(width: 48, alignment: .trailing)
            }
        }
        .help("Softens the edge where the generated area meets the original")
    }

    @ViewBuilder func outpaintControls(_ b: LiteralBindings) -> some View {
        let pad = AppModel.outpaintIDs.pad
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 4) {
            GridRow {
                sideStepper("Left", b.int(pad, "left"))
                sideStepper("Right", b.int(pad, "right"))
            }
            GridRow {
                sideStepper("Top", b.int(pad, "top"))
                sideStepper("Bottom", b.int(pad, "bottom"))
            }
        }
        featherSlider(AppModel.outpaintIDs.feather, b)
        HStack {
            Text(outpaintSizeText(b)).font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Stop Extending") { model.clearOutpaint(undoManager: undoManager) }
        }
    }

    func sideStepper(_ label: String, _ v: Binding<Int>) -> some View {
        Stepper(value: v, in: 0...1024, step: 64) {
            LabeledContent(label, value: "\(v.wrappedValue)")
        }
    }

    func outpaintSizeText(_ b: LiteralBindings) -> String {
        let pad = AppModel.outpaintIDs.pad
        let dx = b.int(pad, "left").wrappedValue + b.int(pad, "right").wrappedValue
        let dy = b.int(pad, "top").wrappedValue + b.int(pad, "bottom").wrappedValue
        guard let path = model.referenceImagePath, let size = Thumbnail.pixelSize(path) else {
            return "+\(dx) × +\(dy) px"
        }
        let w = size.width + dx, h = size.height + dy
        return "Output \(w) × \(h) (" + String(format: "%.1f", Double(w * h) / 1_048_576) + " MP)"
    }

    @ViewBuilder func hiResControls(_ b: LiteralBindings) -> some View {
        Toggle("Hi-res fix", isOn: Binding(get: { model.isHiResEnabled },
                                           set: { model.setHiRes($0, undoManager: undoManager) }))
            .help("Upscale the result and refine it with a second, low-denoise pass")
        if model.isHiResEnabled {
            let ids = AppModel.hiresIDs
            let scale = b.double(ids.scale, "factor"), denoise = b.double(ids.sampler, "denoise")
            LabeledContent("Upscale") {
                HStack {
                    Slider(value: scale, in: 1.25...2, step: 0.25)
                    Text(String(format: "%.2g×", scale.wrappedValue)).monospacedDigit().frame(width: 48, alignment: .trailing)
                }
            }
            LabeledContent("Denoise") {
                HStack {
                    Slider(value: denoise, in: 0.2...0.7, step: 0.05)
                    Text(String(format: "%.2f", denoise.wrappedValue)).monospacedDigit().frame(width: 48, alignment: .trailing)
                }
            }
        }
    }

    func chooseReference() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { model.setReferenceImage(url, undoManager: undoManager) }
    }

    func missing(_ type: String) -> some View {
        Text("This graph has no \(type) node.").foregroundStyle(.secondary)
    }

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.message = "Choose a diffusers-layout model folder (with model_index.json)."
        if panel.runModal() == .OK, let url = panel.url {
            model.applyModelPath(url.path, undoManager: undoManager)
            model.preloadModel()
        }
    }
}

/// Graph mode: every literal input of the selected node.
struct NodeInspector: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    let node: GraphNode

    var body: some View {
        let b = LiteralBindings(model: model, undoManager: undoManager)
        let schema = model.schema(for: node)
        Form {
            Section {
                LabeledContent("Type", value: schema?.displayName ?? node.type)
                LabeledContent("ID", value: node.id)
                if let state = model.nodeStates[node.id] { LabeledContent("Last run", value: describe(state)) }
            }
            if let schema {
                Section("Inputs") {
                    ForEach(schema.inputs, id: \.name) { spec in
                        if case .link(let src, let out)? = node.inputs[spec.name] {
                            LabeledContent(spec.name, value: "← \(src).\(out)")
                        } else if spec.type.isScalar {
                            editor(spec, b)
                        } else {
                            LabeledContent(spec.name) {
                                Text("not connected").foregroundStyle(spec.optional ? Color.secondary : Color.red)
                            }
                        }
                    }
                }
                if !schema.outputs.isEmpty {
                    Section("Outputs") {
                        ForEach(schema.outputs, id: \.name) { o in
                            LabeledContent(o.name, value: o.type.rawValue)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
    }

    @ViewBuilder
    func editor(_ spec: InputSpec, _ b: LiteralBindings) -> some View {
        switch (spec.type, spec.widget) {
        case (.string, .text(multiline: true)?):
            VStack(alignment: .leading) {
                Text(spec.name).font(.caption).foregroundStyle(.secondary)
                TextEditor(text: b.string(node.id, spec.name)).frame(minHeight: 120)
                    .scrollContentBackground(.hidden)
            }
        case (.string, .filePath?):
            HStack {
                TextField(spec.name, text: b.string(node.id, spec.name))
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = true
                    if panel.runModal() == .OK, let url = panel.url { b.string(node.id, spec.name).wrappedValue = url.path }
                }
            }
        case (.string, .choice(let options)?):
            Picker(spec.name, selection: b.string(node.id, spec.name)) {
                ForEach(options, id: \.self) { Text($0).tag($0) }
            }
        case (.string, _):
            TextField(spec.name, text: b.string(node.id, spec.name))
        case (.int, .seed?):
            HStack {
                TextField(spec.name, value: b.int(node.id, spec.name), format: .number.grouping(.never))
                Button { b.int(node.id, spec.name).wrappedValue = Int.random(in: 0...Int(Int32.max)) } label: {
                    Image(systemName: "dice")
                }.buttonStyle(.borderless)
            }
        case (.int, _):
            TextField(spec.name, value: b.int(node.id, spec.name), format: .number.grouping(.never))
        case (.float, _):
            TextField(spec.name, value: b.double(node.id, spec.name), format: .number)
        case (.bool, _):
            Toggle(spec.name, isOn: b.bool(node.id, spec.name))
        default:
            LabeledContent(spec.name, value: spec.type.rawValue)
        }
    }

    func describe(_ s: NodeRunState) -> String {
        switch s {
        case .pending: return "waiting"
        case .running(let step, let total): return total > 0 ? "running \(step)/\(total)" : "running"
        case .cached: return "cached"
        case .done(let secs, let peak):
            return String(format: "%.2fs", secs) + (peak.map { String(format: ", peak %.1f GB", Double($0) / 1e9) } ?? "")
        case .failed(let m): return "failed: \(m)"
        }
    }
}

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager

    var body: some View {
        @Bindable var model = model
        Form {
            LabeledContent("Model folder") {
                HStack {
                    Text(model.modelPath).lineLimit(1).truncationMode(.middle)
                    Button("Choose…") {
                        let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false
                        if p.runModal() == .OK, let u = p.url { model.applyModelPath(u.path, undoManager: undoManager) }
                    }
                }
            }
            LabeledContent("Output folder") {
                HStack {
                    Text(model.outputDirectory).lineLimit(1).truncationMode(.middle)
                    Button("Choose…") {
                        let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false
                        p.canCreateDirectories = true
                        if p.runModal() == .OK, let u = p.url { model.outputDirectory = u.path }
                    }
                }
            }
            Text("New graphs save into the output folder; existing SaveImage nodes keep their own directory.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .frame(width: 520)
        .padding()
    }
}


/// Small image preview with the inpainting mask tinted red on top.
struct Thumbnail: View {
    let path: String
    let maskPath: String?
    @State private var image: CGImage?
    @State private var mask: CGImage?

    var body: some View {
        ZStack {
            if let image {
                Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
            } else {
                Rectangle().fill(.quaternary)
            }
            if let mask {
                Image(decorative: mask, scale: 1).resizable().aspectRatio(contentMode: .fill).opacity(0.55)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task(id: path + (maskPath ?? "")) {
            image = Self.thumb(path)
            mask = maskPath.flatMap(Self.thumb).flatMap { MaskEditorView.redOverlay($0) }
        }
    }

    static func thumb(_ path: String) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateThumbnailAtIndex(src, 0, [
            kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: 192,
        ] as CFDictionary)
    }
}

extension Thumbnail {
    /// Pixel size from the image header, without decoding it.
    static func pixelSize(_ path: String) -> (width: Int, height: Int)? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int, let h = props[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return (w, h)
    }
}
