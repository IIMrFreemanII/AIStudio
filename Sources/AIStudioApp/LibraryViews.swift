import AppKit
import GraphRuntime
import MLXDiffusionCore
import NodesRemote
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

/// What the Add sheet adds: a link, or a file from disk. `node`: a LoRA node to point at the new item.
struct LibraryAddRequest: Identifiable {
    enum Source { case link, file(URL) }
    let id = UUID()
    var source: Source
    var node: String?
    /// The node input the new item goes into (`lora`, or `model` for a cloud model node).
    var input = "lora"
}

// MARK: - the Models window's LoRAs section

/// A library section of the Models window: models from links, or LoRAs.
struct LibrarySection: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    let kind: LoRARemote.Kind
    @Binding var adding: LibraryAddRequest?

    var body: some View {
        let lib = store.library
        let items = lib.items.filter { $0.kind == kind }.sorted { $0.added > $1.added }
        MGSection {
            if items.isEmpty {
                Text(kind == .lora
                     ? "Add LoRAs from a Civitai model page or a Hugging Face link, or from a file. Each can live on this Mac, on your GPU servers, or both; the LoRA nodes pick from this list."
                     : "Add a FLUX.1 dev fine-tune from a Civitai checkpoint page or a Hugging Face link. It installs on this Mac and/or your GPU server, next to FLUX.1 dev (reusing its text encoders and VAE); the Load Model menu and the cloud FLUX node's model menu pick it.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).fixedSize(horizontal: false, vertical: true)
            }
            ForEach(items) { item in LibraryRow(item: item) }
        } header: {
            HStack {
                Text(kind == .lora ? "LoRAs" : "Models from Links")
                Spacer()
                if kind == .lora {
                    Button("Add File…") { if let url = chooseLoRAFile() { adding = LibraryAddRequest(source: .file(url)) } }
                        .buttonStyle(.mg(.borderless)).font(.mgSubheadline)
                }
                Button("Add from Link…") { adding = LibraryAddRequest(source: .link, input: kind == .lora ? "lora" : "model") }
                    .buttonStyle(.mg(.borderless)).font(.mgSubheadline)
            }
        } footer: {
            Text(kind == .lora
                 ? "On this Mac they're kept in \(LoRADownloads.folder(models: store.modelsDirectory).path). Most Civitai files need your API key (below); a GPU server uses its own (the civitai Modal secret)."
                 : "On this Mac a fine-tune is converted to 8-bit (about 12 GB, in \(store.modelsDirectory.path)); FLUX.1 dev must be installed. A GPU server downloads and converts it itself (about 24 GB on its models Volume, a few minutes of CPU time).")
                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
        }
        .task(id: model.connections.gpuServers.map(\.id)) { lib.checkServers(model.connections.gpuServers.map(\.id)) }
    }
}

@MainActor
func chooseLoRAFile() -> URL? {
    let p = NSOpenPanel()
    p.allowedContentTypes = [UTType(filenameExtension: "safetensors") ?? .data]
    p.message = "Choose a LoRA (.safetensors) to add to the library."
    return p.runModal() == .OK ? p.url : nil
}

struct LibraryRow: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    let item: LibraryEntry
    @State private var renaming = false
    @State private var newTitle = ""

    var body: some View {
        let lib = store.library
        HStack(alignment: .top, spacing: 12) {
            LibraryThumb(url: lib.thumbnail(item), kind: item.kind)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(item.title).fontWeight(.medium).lineLimit(1)
                    FamilyBadge(family: item.family)
                    Spacer(minLength: 4)
                    if let b = item.bytes {
                        Text(ByteCountFormatter.string(fromByteCount: b, countStyle: .file))
                            .font(.mgFootnote.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                    }
                }
                HStack(spacing: 6) {
                    StateChip(title: "Mac", state: lib.macState(item))
                    ForEach(model.connections.gpuServers) { s in
                        StateChip(title: s.name, state: lib.serverState(item, s.id))
                    }
                }
                if !item.triggerWords.isEmpty {
                    Text("Trigger: " + item.triggerWords.joined(separator: ", "))
                        .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.tail)
                }
            }
            MGMenuButton(kind: .icon, items: { menu(lib) }) { Image(systemName: "ellipsis") }
        }
        .padding(.vertical, 4)
        .mgSheet("rename-\(item.id)", isPresented: $renaming) {
            VStack(alignment: .leading, spacing: 12) {
                Text("Rename").font(.mgHeadline)
                TextField("Name", text: $newTitle).frame(width: 320)
                HStack {
                    Spacer()
                    Button("Cancel") { renaming = false }.keyboardShortcut(.cancelAction)
                    Button("Rename") { lib.rename(item.id, newTitle); renaming = false }.keyboardShortcut(.defaultAction)
                }
            }
            .padding(20)
        }
    }

    @MGMenuBuilder func menu(_ lib: LinkLibrary) -> [MGMenuItem] {
        let mac = lib.macState(item)
        MGMenuItem.button("Rename…") { newTitle = item.title; renaming = true }
        MGMenuItem.menu("Family") {
            for f in [nil, "flux.1", "qwen-image", "qwen-image-2.1", "wan-2.2-t2v", "wan-2.2-i2v"] as [String?] {
                MGMenuItem.button(LoRASources.familyName(f), checked: item.family == f) { lib.setFamily(item.id, f) }
            }
        }
        MGMenuItem.divider
        if mac == .installed, item.kind == .model, let url = Library.localURL(item, models: store.modelsDirectory) {
            MGMenuItem.button("Use in This Workflow") { model.applyModelPath(url.path, undoManager: nil); model.preloadModel() }
        }
        if mac == .installed {
            MGMenuItem.button("Show in Finder") {
                if let url = Library.localURL(item, models: store.modelsDirectory) { NSWorkspace.shared.activateFileViewerSelecting([url]) }
            }
            MGMenuItem.button("Remove from This Mac", enabled: item.source != nil) { lib.removeFromMac(item.id) }
        } else if item.kind == .model, let why = LinkLibrary.macModelRefusal(item.family) {
            MGMenuItem.button("Install on This Mac (\(why))", enabled: false) {}
        } else {
            MGMenuItem.button("Install on This Mac", enabled: item.source != nil) { lib.installOnMac(item.id) }
        }
        for s in model.connections.gpuServers {
            if lib.serverState(item, s.id) == .installed {
                MGMenuItem.button("Remove from \(s.name)") { lib.removeFromServer(item.id, s.id) }
            } else if item.kind == .model, let why = LinkLibrary.serverModelRefusal(item.family) {
                MGMenuItem.button("Install on \(s.name) (\(why))", enabled: false) {}
            } else {
                MGMenuItem.button("Install on \(s.name)") { lib.installOnServer(item.id, s.id) }
            }
        }
        MGMenuItem.divider
        if let link = item.link {
            MGMenuItem.button("Copy Link") { copy(link) }
            if let url = URL(string: link) { MGMenuItem.button("Open Link") { NSWorkspace.shared.open(url) } }
        }
        if !item.triggerWords.isEmpty {
            MGMenuItem.button("Copy Trigger Words") { copy(item.triggerWords.joined(separator: ", ")) }
        }
        MGMenuItem.divider
        MGMenuItem.button("Remove from Library", destructive: true) { lib.remove(item.id) }
    }
}

@MainActor func copy(_ s: String) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(s, forType: .string)
}

struct LibraryThumb: View {
    let url: URL?
    let kind: LoRARemote.Kind
    var size: CGFloat = 48

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill)
            if let url, let image = NSImage(contentsOf: url) {
                Image(nsImage: image).resizable().aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: kind == .lora ? "slider.horizontal.3" : "cube").foregroundStyle(MG.tertiaryLabel)
            }
        }
        .frame(width: size, height: size)
        .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
    }
}

struct FamilyBadge: View {
    let family: String?
    var body: some View {
        Text(LoRASources.familyName(family))
            .font(.mg(size: 10, weight: .medium))
            .padding(.horizontal, 6).padding(.vertical, 1)
            .background(Capsule().fill(MG.fill))
            .foregroundStyle(MG.secondaryLabel)
    }
}

/// "Mac ✓", "GPU ↓ 41 % …", "Mac —": where an item is installed.
struct StateChip: View {
    let title: String
    let state: LinkLibrary.State

    var body: some View {
        HStack(spacing: 4) {
            switch state {
            case .installed: Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.success)
            case .working: MGSpinner().controlSize(.mini)
            case .failed: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(MG.warning)
            case .checking: MGSpinner().controlSize(.mini)
            case .absent, .unknown: Image(systemName: "minus.circle").foregroundStyle(MG.tertiaryLabel)
            }
            Text(text).lineLimit(1).truncationMode(.middle)
        }
        .font(.mg(size: 10).monospacedDigit())
        .padding(.horizontal, 7).padding(.vertical, 2)
        .background(Capsule().fill(MG.fill))
        .foregroundStyle(MG.secondaryLabel)
        .mgHelp(help)
    }

    var text: String {
        switch state {
        case .working(let line): return "\(title) · \(line)"
        case .failed: return "\(title) · failed"
        default: return title
        }
    }

    var help: String {
        switch state {
        case .installed: return "Installed on \(title)"
        case .absent: return "Not on \(title)"
        case .unknown: return "\(title): not checked"
        case .checking: return "Checking \(title)…"
        case .working(let line): return line
        case .failed(let why): return why
        }
    }
}

// MARK: - the Add sheet

struct LibraryAddSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    let request: LibraryAddRequest
    let library: LinkLibrary

    enum Lookup: Equatable {
        case idle, looking
        case failed(String)
        case found(LoRARemote)
    }

    @State private var link = ""
    @State private var lookup: Lookup = .idle
    @State private var title = ""
    @State private var family: String?
    @State private var onMac = true
    @State private var servers: Set<String> = []
    @State private var macAvailability: LoRAAvailability?
    @State private var serverAvailability: [String: LoRAAvailability] = [:]

    var fileURL: URL? { if case .file(let u) = request.source { return u }; return nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(fileURL == nil ? "Add from Link" : "Add LoRA File").font(.mgHeadline)
            if let file = fileURL {
                LabeledContent("File", value: file.lastPathComponent)
                LabeledContent("Size", value: ByteCountFormatter.string(
                    fromByteCount: Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0), countStyle: .file))
            } else {
                TextField("Civitai model page or Hugging Face link", text: $link).frame(width: 460)
                    .onSubmit { Task { await resolve() } }
                switch lookup {
                case .idle: EmptyView()
                case .looking:
                    HStack(spacing: 6) { MGSpinner().controlSize(.mini); Text("Looking up the link…").foregroundStyle(MG.secondaryLabel) }
                        .font(.mgSubheadline)
                case .failed(let m):
                    Label(m, systemImage: "exclamationmark.triangle").font(.mgSubheadline).foregroundStyle(MG.warning)
                        .fixedSize(horizontal: false, vertical: true)
                case .found(let r):
                    LinkCard(remote: r, thumbnail: r.previewImage)
                }
            }
            if canChooseDetails {
                MGFormRow("Name") { TextField("Name", text: $title).frame(width: 300) }
                MGFormRow("Family") {
                    MGPicker(selection: $family, options: ([nil, "flux.1", "qwen-image", "qwen-image-2.1", "wan-2.2-t2v",
                                                            "wan-2.2-i2v"] as [String?]).map { ($0, LoRASources.familyName($0)) })
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Install on").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                    if isModel, let why = macRefusal {
                        Toggle(isOn: .constant(false)) {
                            HStack(spacing: 6) { Text("This Mac"); Text(why).foregroundStyle(MG.secondaryLabel) }
                        }
                        .disabled(true)
                    } else {
                        Toggle(isOn: fileURL == nil ? $onMac : .constant(true)) {
                            HStack(spacing: 6) { Text("This Mac"); availabilityText(fileURL == nil ? macAvailability : .stored, mac: true) }
                        }
                        .disabled(fileURL != nil)
                    }
                    if isModel, onMac, macRefusal == nil, case .found = lookup,
                       !ModelStore.isComplete(ModelPaths.directory.appendingPathComponent(LinkLibrary.macModelBases[family ?? ""] ?? "-")) {
                        Text("This Mac needs \(LoRASources.familyName(family)) installed first (Available Models): fine-tunes reuse its text encoders and VAE.")
                            .foregroundStyle(MG.warning).fixedSize(horizontal: false, vertical: true)
                    }
                    if isModel, family?.hasPrefix("wan-2.2") == true, case .found(let r) = lookup, r.expert == nil,
                       LinkLibrary.huggingFaceRepo(r) == nil {
                        Text("This file doesn't say which Wan 2.2 expert it is (high or low noise), so it can't be installed. Link a version whose file name says HIGH or LOW.")
                            .foregroundStyle(MG.warning).fixedSize(horizontal: false, vertical: true)
                    }
                    if isModel && family == nil {
                        Text("Pick the family: FLUX.1, Qwen-Image (cloud), Qwen-Image 2.1 (this Mac) or Wan 2.2.")
                            .foregroundStyle(MG.warning)
                    }
                    ForEach(model.connections.gpuServers) { s in
                        if isModel, let why = serverRefusal {
                            Toggle(isOn: .constant(false)) {
                                HStack(spacing: 6) { Text(s.name); Text(why).foregroundStyle(MG.secondaryLabel) }
                            }
                            .disabled(true)
                        } else {
                            Toggle(isOn: Binding(get: { servers.contains(s.id) },
                                                 set: { if $0 { servers.insert(s.id) } else { servers.remove(s.id) } })) {
                                HStack(spacing: 6) {
                                    Text(s.name)
                                    if fileURL == nil { availabilityText(serverAvailability[s.id], mac: false) }
                                    else { Text("uploaded from this Mac").foregroundStyle(MG.secondaryLabel) }
                                }
                            }
                        }
                    }
                }
                .font(.mgSubheadline)
            }
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                if fileURL == nil, case .found = lookup {} else if fileURL == nil {
                    Button("Look Up") { Task { await resolve() } }.keyboardShortcut(.defaultAction)
                        .disabled(link.trimmingCharacters(in: .whitespaces).isEmpty || lookup == .looking)
                }
                if canChooseDetails {
                    Button("Add") { add() }.keyboardShortcut(.defaultAction)
                        .disabled(title.trimmingCharacters(in: .whitespaces).isEmpty || !modelFamilyOK
                                  || (!installsOnMac && chosenServers.isEmpty))
                }
            }
        }
        .padding(20)
        .frame(width: 520, alignment: .leading)
        .onAppear {
            if let file = fileURL { title = file.deletingPathExtension().lastPathComponent }
            servers = Set(model.connections.gpuServers.prefix(fileURL == nil ? 1 : 0).map(\.id))
        }
        .onChange(of: link) { _, _ in if lookup != .looking { lookup = .idle } }
    }

    /// The details and targets show once there's something addable: a file, or a link to a LoRA or a model.
    var canChooseDetails: Bool {
        if fileURL != nil { return true }
        if case .found = lookup { return true }
        return false
    }

    var isModel: Bool { if case .found(let r) = lookup { return r.kind == .model }; return false }

    /// A fine-tune needs a family some target runs.
    var modelFamilyOK: Bool { !isModel || macRefusal == nil || serverRefusal == nil }
    var macRefusal: String? { isModel ? LinkLibrary.macModelRefusal(family) : nil }
    var serverRefusal: String? { isModel ? LinkLibrary.serverModelRefusal(family) : nil }
    var installsOnMac: Bool { (onMac || fileURL != nil) && macRefusal == nil }
    var chosenServers: [String] { serverRefusal == nil ? Array(servers) : [] }

    @ViewBuilder func availabilityText(_ a: LoRAAvailability?, mac: Bool) -> some View {
        switch a {
        case nil: if fileURL == nil, case .found = lookup { MGSpinner().controlSize(.mini) }
        case .stored?: Text(mac ? "already downloaded" : "already there").foregroundStyle(MG.success)
        case .available?: Text("can download").foregroundStyle(MG.success)
        case .needsKey(let h)?:
            Text(h == .civitai ? (mac ? "needs your Civitai API key (Models window)" : "needs the civitai Modal secret")
                 : "gated: needs your Hugging Face token").foregroundStyle(MG.warning)
        case .unavailable(let why)?: Text(why).foregroundStyle(MG.warning).lineLimit(2)
        }
    }

    func resolve() async {
        let text = link.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        lookup = .looking
        macAvailability = nil
        serverAvailability = [:]
        do {
            // A Civitai page with several versions: the newest one for a model the app runs.
            let r = try await LoRASources.resolve(text, fits: { LoRASources.family(forBaseModel: $0) != nil })
            lookup = .found(r)
            title = (r.title ?? r.fileName).components(separatedBy: " · ").first ?? r.fileName
            family = r.baseModel.flatMap(LoRASources.family(forBaseModel:))
            if r.baseModel != nil, family == nil {
                lookup = .failed("This \(r.kind == .lora ? "LoRA" : "model") is for \(r.baseModel!), which the app doesn't run (FLUX.1, Qwen-Image and Wan 2.2 do).")
                return
            }
            if r.kind == .model { onMac = LinkLibrary.macModelRefusal(family) == nil }
            async let mac = LoRADownloads.probe(r, models: ModelPaths.directory)
            for s in model.connections.gpuServers {
                let id = s.id
                Task { serverAvailability[id] = await checkLinkOnServer(r, serverID: id) }
            }
            macAvailability = await mac
        } catch {
            lookup = .failed(String(describing: error))
        }
    }

    func add() {
        let name = title.trimmingCharacters(in: .whitespacesAndNewlines)
        var id: String?
        if let file = fileURL {
            id = try? library.addFile(file, title: name, family: family, servers: Array(servers))
        } else if case .found(let r) = lookup {
            id = library.add(r, link: link.trimmingCharacters(in: .whitespacesAndNewlines), title: name, family: family,
                             mac: installsOnMac, servers: chosenServers)
        }
        if let id, let node = request.node {
            model.setLiteral(.string(id), node: node, input: request.input, undoManager: model.activeUndoManager)
        }
        dismiss()
    }
}

/// What a link points at: sample image, title, file and size, base model, trigger words.
struct LinkCard: View {
    let remote: LoRARemote
    let thumbnail: URL?

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if let thumbnail {
                AsyncImage(url: thumbnail) { $0.resizable().aspectRatio(contentMode: .fill) } placeholder: {
                    RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill)
                }
                .frame(width: 64, height: 64)
                .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(remote.title ?? remote.fileName).font(.mg(size: 12, weight: .semibold)).lineLimit(2)
                Text([remote.kind == .lora ? "LoRA" : "Model", remote.fileName,
                      remote.companion != nil ? "high + low noise pair"
                        : remote.expert.map { "\($0)-noise expert only" },
                      remote.totalBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) },
                      remote.host == .civitai ? "Civitai" : remote.host == .huggingface ? "Hugging Face" : remote.downloadURL.host]
                        .compactMap { $0 }.joined(separator: " · "))
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(2).truncationMode(.middle)
                if let base = remote.baseModel {
                    Text("For \(base)").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                }
                if !remote.triggerWords.isEmpty {
                    Text("Trigger: " + remote.triggerWords.joined(separator: ", ")).font(.mgSubheadline)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }
}

// MARK: - the LoRA nodes: picker and inspector card

extension AppModel {
    /// The family a LoRA node's LoRAs must fit: the cloud image node's for a cloud LoRA, else the graph's model
    /// folder's.
    func loraFamily(for node: GraphNode) -> String? {
        if node.type == "RemoteLoRA" { return cloudImageNode.flatMap { RemoteNodes.libraryFamilies[$0.type] } ?? "flux.1" }
        guard let path = firstNode(ofType: "LoadModel").flatMap({ n -> String? in
            if case .value(.string(let p))? = n.inputs["path"] { return p }
            return nil
        }), !path.isEmpty else { return nil }
        let cls = ModelRegistry.className(at: ModelPaths.resolve(path)) ?? ""
        if cls.contains("Flux") { return "flux.1" }
        if cls.contains("Wan") { return "wan-2.2-t2v" }
        if cls.contains("Qwen") { return "qwen-image-2.1" }
        return nil
    }
}

/// The `lora` input's menu: the library's LoRAs that fit, then Add from Link… and Add File….
struct LibraryPicker: View {
    @Environment(AppModel.self) private var model
    let node: GraphNode
    let input: String
    var compact = false

    var body: some View {
        let lib = model.store?.library
        let current: String = { if case .value(.string(let s))? = node.inputs[input] { return s }; return "" }()
        let isModel = input == "model"
        let family = RemoteNodes.libraryFamilies[node.type] ?? "flux.1"
        let base = RemoteNodes.baseTitle(family: family)
        let title = current.isEmpty ? (isModel ? base : "Pick a LoRA")
            : (lib?.item(current)?.title ?? "Missing: \(current)")
        MGMenuButton(kind: compact ? .borderless : .bordered, items: {
            if isModel {
                MGMenuItem.button(base, checked: current.isEmpty) { set("") }
                if let lib {
                    let fits = lib.models(for: family)
                    if !fits.isEmpty { MGMenuItem.divider }
                    for item in fits { MGMenuItem.button(item.title, checked: item.id == current) { set(item.id) } }
                }
                MGMenuItem.divider
                MGMenuItem.button("Add from Link…") {
                    model.libraryRequest = LibraryAddRequest(source: .link, node: node.id, input: input)
                }
            } else {
                if let lib {
                    let fits = lib.loras(for: model.loraFamily(for: node))
                    if fits.isEmpty { MGMenuItem.header("No LoRAs for this model yet") }
                    for item in fits {
                        MGMenuItem.button(item.title, checked: item.id == current) { set(item.id) }
                    }
                }
                MGMenuItem.divider
                MGMenuItem.button("Add from Link…") { model.libraryRequest = LibraryAddRequest(source: .link, node: node.id) }
                MGMenuItem.button("Add File…") {
                    if let url = chooseLoRAFile() { model.libraryRequest = LibraryAddRequest(source: .file(url), node: node.id) }
                }
                if !current.isEmpty { MGMenuItem.button("None") { set("") } }
            }
        }) {
            HStack(spacing: 5) {
                Text(title).lineLimit(1).truncationMode(.middle)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
            }
            .font(compact ? .mg(size: 11) : .mgBody)
        }
    }

    func set(_ id: String) { model.setLiteral(.string(id), node: node.id, input: input, undoManager: model.activeUndoManager) }
}

/// The picked LoRA in the inspector: image, base model, trigger words (Copy, Add to Prompt), where it's installed.
struct LibraryItemCard: View {
    @Environment(AppModel.self) private var model
    let node: GraphNode
    let item: LibraryEntry

    var body: some View {
        let lib = model.store?.library
        MGSection("LoRA") {
            HStack(alignment: .top, spacing: 10) {
                LibraryThumb(url: lib?.thumbnail(item), kind: item.kind, size: 64)
                VStack(alignment: .leading, spacing: 3) {
                    Text(item.title).font(.mg(size: 12, weight: .semibold)).lineLimit(2)
                    HStack(spacing: 6) {
                        FamilyBadge(family: item.family)
                        if let b = item.bytes {
                            Text(ByteCountFormatter.string(fromByteCount: b, countStyle: .file))
                                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        }
                    }
                    if let base = item.baseModel, let fam = model.loraFamily(for: node),
                       let fits = LoRASources.civitaiBaseModels(for: fam), !fits(base) {
                        Label("Made for \(base)", systemImage: "xmark.octagon").font(.mgSubheadline).foregroundStyle(MG.destructive)
                    }
                    if let lib {
                        HStack(spacing: 6) {
                            StateChip(title: "Mac", state: lib.macState(item))
                            ForEach(model.connections.gpuServers) { s in StateChip(title: s.name, state: lib.serverState(item, s.id)) }
                        }
                    }
                }
            }
            if !item.triggerWords.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Trigger words").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                    Text(item.triggerWords.joined(separator: ", ")).font(.mg(size: 11)).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack {
                        Button("Copy") { copy(item.triggerWords.joined(separator: ", ")) }
                        if let target = promptTarget {
                            Button("Add to Prompt") { addToPrompt(item.triggerWords, target) }
                                .disabled(promptContains(item.triggerWords, target))
                        }
                    }
                    .controlSize(.small)
                }
            }
        }
    }

    /// The prompt the trigger words belong in: the cloud FLUX node's, or the first Text Encode's.
    var promptTarget: (node: String, input: String)? {
        if node.type == "RemoteLoRA" { return model.cloudImageNode.map { ($0.id, "prompt") } }
        return model.firstNode(ofType: "TextEncode").map { ($0.id, "prompt") }
    }

    func prompt(_ t: (node: String, input: String)) -> String {
        if case .value(.string(let s))? = model.graph[t.node]?.inputs[t.input] { return s }
        return ""
    }

    func promptContains(_ words: [String], _ t: (node: String, input: String)) -> Bool {
        let p = prompt(t).lowercased()
        return words.allSatisfy { p.contains($0.lowercased()) }
    }

    func addToPrompt(_ words: [String], _ t: (node: String, input: String)) {
        let current = prompt(t).trimmingCharacters(in: .whitespacesAndNewlines)
        let missing = words.filter { !current.lowercased().contains($0.lowercased()) }
        guard !missing.isEmpty else { return }
        let joined = missing.joined(separator: ", ")
        model.setLiteral(.string(current.isEmpty ? joined : current + ", " + joined), node: t.node, input: t.input,
                         undoManager: model.activeUndoManager)
    }
}
