import UniformTypeIdentifiers
import AppKit
import GraphRuntime
import ImageIO
import MLXDiffusionCore
import MLXComfyLibrary
import NodesStandard
import SwiftUI

/// Bindings onto literal node inputs, with undo.
@MainActor
struct LiteralBindings {
    let model: AppModel
    let undoManager: UndoManager?
    /// Bind to nodes at the editor's level (inside a subgraph) instead of the top-level graph.
    var editor = false

    func get(_ node: String, _ input: String) -> JSONValue? {
        editor ? model.editorLiteral(node, input) : model.literal(node, input)
    }

    func set(_ v: JSONValue, _ node: String, _ input: String) {
        if editor { model.setEditorLiteral(v, node: node, input: input, undoManager: undoManager) }
        else { model.setLiteral(v, node: node, input: input, undoManager: undoManager) }
    }

    func string(_ node: String, _ input: String) -> Binding<String> {
        Binding(
            get: { get(node, input)?.description ?? "" },
            set: { set(.string($0), node, input) })
    }

    func int(_ node: String, _ input: String) -> Binding<Int> {
        Binding(
            get: {
                switch get(node, input) {
                case .int(let v)?: return v
                case .double(let d)?: return Int(d)
                default: return 0
                }
            },
            set: { set(.int($0), node, input) })
    }

    func double(_ node: String, _ input: String) -> Binding<Double> {
        Binding(
            get: {
                switch get(node, input) {
                case .int(let v)?: return Double(v)
                case .double(let d)?: return d
                default: return 0
                }
            },
            set: { set(.double($0), node, input) })
    }

    func bool(_ node: String, _ input: String) -> Binding<Bool> {
        Binding(
            get: { if case .bool(let b)? = get(node, input) { return b } else { return false } },
            set: { set(.bool($0), node, input) })
    }
}

struct InspectorView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.selection.count == 1, let id = model.selection.first, let node = model.editorGraph[id] {
            NodeInspector(node: node)
        } else if model.selection.isEmpty, let gid = model.selectedGroup,
                  let group = model.editorGraph.groups?.first(where: { $0.id == gid }) {
            GroupInspector(group: group)
        } else if model.selection.count > 1 {
            ContentUnavailableView("\(model.selection.count) nodes selected", systemImage: "square.on.square",
                                   description: Text("Select a single node to edit its values."))
        } else {
            WorkflowInspector()
        }
    }
}

/// Nothing selected: the workflow itself — name and file, how it runs, and what can be added to it.
struct WorkflowInspector: View {
    @Environment(AppModel.self) private var model
    @State private var name = ""

    var body: some View {
        @Bindable var model = model
        let doc = model.active
        Form {
            Section("Workflow") {
                TextField("Name", text: $name)
                    .onSubmit { rename(doc) }
                LabeledContent("File") {
                    Text(doc.fileURL.map { $0.path.replacingOccurrences(of: model.library.workflows.path + "/", with: "") }
                         ?? "Not saved")
                        .lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                }
                HStack {
                    Button(doc.fileURL == nil ? "Save…" : "Save") { TabCommands.save(doc, model: model) }
                        .disabled(doc.fileURL != nil && !doc.isDirty)
                    Button("Save As…") { TabCommands.saveAs(doc, model: model) }
                    Button("Save as Preset…") { model.showSavePresetSheet = true }
                }
                .controlSize(.small)
            }
            Section("Run") {
                Stepper(value: $model.batchCount, in: 1...16) {
                    LabeledContent("Batch", value: "\(model.batchCount) run\(model.batchCount == 1 ? "" : "s")")
                }
                Toggle("Randomize seed each run", isOn: $model.randomizeSeed)
                if let err = model.validationError {
                    Label(err, systemImage: "exclamationmark.triangle").font(.caption).foregroundStyle(.orange)
                }
            }
            Section {
                ForEach(AddOn.allCases.filter { model.canAdd($0) || model.isActive($0) }, id: \.self) { a in
                    HStack {
                        Text(a.title.replacingOccurrences(of: "…", with: ""))
                        Spacer()
                        if model.isActive(a) {
                            Image(systemName: "checkmark").foregroundStyle(.green)
                            if a == .inpaint || a == .reference || a == .lora || a == .firstFrame {
                                Button("Change") { model.add(a) }.controlSize(.small)
                            }
                            Button("Remove") { model.remove(a) }.controlSize(.small)
                        } else {
                            Button("Add") { model.add(a) }.controlSize(.small)
                        }
                    }
                }
            } header: {
                Text("Add to Workflow")
            } footer: {
                Text("Splices a ready-made chain into the graph, wired to its sampler. Also in the Graph menu and the canvas's context menu.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Model") {
                LabeledContent("Status") {
                    HStack(spacing: 6) {
                        if model.modelLoading { ProgressView().controlSize(.mini) }
                        Text(model.modelStatus).lineLimit(1).truncationMode(.middle)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .safeAreaInset(edge: .bottom) {
            Button { model.enqueue(undoManager: model.activeUndoManager) } label: {
                Label(model.batchCount > 1 ? "Generate \(model.batchCount)" : "Generate", systemImage: "sparkles")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .padding(12)
            .background(.bar)
        }
        .onAppear { name = doc.name }
        .onChange(of: doc.id) { _, _ in name = model.active.name }
        .onChange(of: doc.name) { _, n in name = n }
    }

    func rename(_ doc: WorkflowDocument) {
        let n = WorkflowStore.sanitize(name)
        guard n != doc.name else { return }
        if let url = doc.fileURL { model.renameWorkflow(url, to: n) } else { doc.name = model.uniqueTabName(n) }
        name = doc.name
    }
}

/// Graph mode: every literal input of the selected node.
struct NodeInspector: View {
    @Environment(AppModel.self) private var model
    private var undoManager: UndoManager? { model.activeUndoManager }
    let node: GraphNode

    var body: some View {
        let b = LiteralBindings(model: model, undoManager: undoManager, editor: true)
        let schema = model.schema(for: node)
        let bound = model.boundInputs[node.id] ?? [:]
        Form {
            Section {
                TextField("Title", text: Binding(get: { node.title ?? "" },
                                                 set: { model.setTitle($0, node: node.id, undoManager: undoManager) }),
                          prompt: Text(schema?.displayName ?? node.type))
                LabeledContent("Type", value: node.type == Graph.subgraphType ? "Subgraph" : (schema?.displayName ?? node.type))
                LabeledContent("ID", value: node.id)
                if let state = model.nodeStates[model.nodeStatePrefix + node.id] {
                    LabeledContent("Last run", value: describe(state))
                }
            }
            if node.type == Graph.subgraphType {
                Section {
                    HStack {
                        Button("Open Subgraph") { model.enterSubgraph(node.id) }
                        Button("Expand") { model.expandSubgraph(node.id, undoManager: undoManager) }
                    }
                    if let def = model.graph.subgraphDef(for: node)?.def {
                        Text("\(def.graph.nodes.count) nodes inside; \(def.inputs.count) input\(def.inputs.count == 1 ? "" : "s"), "
                             + "\(def.outputs.count) output\(def.outputs.count == 1 ? "" : "s").")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            NodeExtras(node: node)
            if let schema {
                Section("Inputs") {
                    ForEach(schema.inputs, id: \.name) { spec in
                        if case .link(let src, let out)? = node.inputs[spec.name] {
                            LabeledContent(spec.name, value: "← \(src).\(out)")
                        } else if let port = bound[spec.name] {
                            LabeledContent(spec.name, value: "← subgraph input \(port)")
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
    private var undoManager: UndoManager? { model.activeUndoManager }

    var body: some View {
        @Bindable var model = model
        Form {
            Section("Library") {
                LabeledContent("Workflows, presets") {
                    HStack {
                        Text(model.library.root.path).lineLimit(1).truncationMode(.middle)
                        Button("Show") { NSWorkspace.shared.open(model.library.root) }
                        Button("Change…") {
                            let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false
                            p.canCreateDirectories = true
                            p.message = "Choose the folder for your workflows and presets (used from the next launch)."
                            if p.runModal() == .OK, let u = p.url { UserDefaults.standard.set(u.path, forKey: "libraryRoot") }
                        }
                    }
                }
                Text("Saved workflows live in its workflows folder, your presets in presets. Open tabs and their unsaved changes are kept separately and come back on the next launch.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            LabeledContent("Qwen-Image folder") {
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
            Text("Each workflow saves into its own subfolder of the output folder; Save nodes pointed elsewhere keep their own directory.")
                .font(.caption).foregroundStyle(.secondary)
            Section("Memory") {
                @Bindable var model = model
                let total = Double(ProcessInfo.processInfo.physicalMemory)
                LabeledContent("Models may use") {
                    HStack {
                        Slider(value: $model.memoryBudget, in: 0.5...0.95, step: 0.05)
                        Text(String(format: "%.0f GB", total * model.memoryBudget / 1e9)).monospacedDigit()
                            .frame(width: 56, alignment: .trailing)
                    }
                }
                Text("When a model doesn't fit, the least recently used one is unloaded first. Models used by the running job are never unloaded.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Prompt Styles") {
                StylesEditor()
            }
            Section("Prompt Assistant") {
                AssistantPromptEditor(title: "Enhance", key: "enhance", fallback: PromptAssistant.enhanceSystem)
                AssistantPromptEditor(title: "Enhance with a reference image", key: "edit",
                                      fallback: PromptAssistant.editSystem)
                AssistantPromptEditor(title: "Describe Image", key: "describe", fallback: PromptAssistant.describeSystem)
            }
        }
        .formStyle(.grouped)
        .frame(width: 560)
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

/// One editable assistant system prompt; empty means the built-in default.
struct AssistantPromptEditor: View {
    let title: String
    let key: String
    let fallback: String
    @AppStorage private var text: String

    init(title: String, key: String, fallback: String) {
        self.title = title
        self.key = key
        self.fallback = fallback
        _text = AppStorage(wrappedValue: "", AppModel.assistantDefaultsKey(key))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).font(.callout)
                Spacer()
                if !text.isEmpty { Button("Use Default") { text = "" }.controlSize(.small) }
                else { Button("Customize") { text = fallback }.controlSize(.small) }
            }
            if text.isEmpty {
                Text(fallback).font(.caption).foregroundStyle(.secondary).lineLimit(3)
            } else {
                TextEditor(text: $text).font(.callout).frame(minHeight: 80).scrollContentBackground(.hidden)
            }
        }
    }
}

/// Add, rename, edit and remove prompt styles; `{prompt}` marks where the prompt goes.
struct StylesEditor: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        ForEach($model.styles) { $style in
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    TextField("Name", text: $style.name).font(.callout.weight(.medium))
                    TextField("Template with {prompt}", text: $style.template, axis: .vertical)
                        .font(.caption).lineLimit(1...3)
                }
                Button(role: .destructive) { model.styles.removeAll { $0.id == style.id } } label: {
                    Image(systemName: "minus.circle")
                }
                .buttonStyle(.borderless)
            }
        }
        HStack {
            Button("Add Style") {
                var name = "New Style"
                var n = 2
                while model.styles.contains(where: { $0.name == name }) { name = "New Style \(n)"; n += 1 }
                model.styles.append(PromptStyle(name: name, template: "{prompt}, "))
            }
            Spacer()
            Button("Restore Built-in Styles") { model.styles = PromptStyle.builtIn }
            Button("Open Wildcards Folder") {
                try? FileManager.default.createDirectory(at: AppModel.wildcardDirectory, withIntermediateDirectories: true)
                NSWorkspace.shared.open(AppModel.wildcardDirectory)
            }
        }
        .controlSize(.small)
    }
}

/// The selected group: title, color, delete.
struct GroupInspector: View {
    @Environment(AppModel.self) private var model
    private var undoManager: UndoManager? { model.activeUndoManager }
    let group: GraphGroup

    var body: some View {
        Form {
            Section("Group") {
                TextField("Title", text: Binding(get: { group.title }, set: { t in
                    model.updateGroup(group.id, actionName: "Rename Group", undoManager: undoManager) { $0.title = t }
                }))
                Picker("Color", selection: Binding(get: { group.color }, set: { c in
                    model.updateGroup(group.id, actionName: "Group Color", undoManager: undoManager) { $0.color = c }
                })) {
                    ForEach(AppModel.groupColors, id: \.self) { Text($0.capitalized).tag($0) }
                }
                LabeledContent("Nodes inside", value: "\(model.nodes(in: group.id).count)")
            }
            Section {
                Button("Delete Group") { model.deleteGroup(group.id, withNodes: false, undoManager: undoManager) }
                Button("Delete Group and Its Nodes", role: .destructive) {
                    model.deleteGroup(group.id, withNodes: true, undoManager: undoManager)
                }
            }
        }
        .formStyle(.grouped)
    }
}
