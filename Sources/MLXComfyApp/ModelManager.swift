import AppKit
import Foundation
import MLXDiffusionCore
import Observation
import Security
import SwiftUI

/// The Hugging Face token for gated repos, kept in the login Keychain. The user enters it; the app never fills it.
///
/// Reading the secret makes macOS ask for permission whenever the app's signature changed (every ad-hoc rebuild), so
/// it is read only when a gated download needs it, at most once per launch. Showing whether a token is saved only
/// looks at the item's attributes, which never asks.
@MainActor
enum HFToken {
    static let service = "MLXComfyUI.huggingface"
    /// The token read (or saved) this launch.
    private static var cached: String?

    static func read() -> String? {
        if let cached { return cached }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        cached = String(data: d, encoding: .utf8)
        return cached
    }

    /// Whether a token is saved, without reading it (no Keychain prompt).
    static var isSaved: Bool {
        if cached != nil { return true }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecReturnAttributes as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        return SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess
    }

    static func save(_ token: String) {
        delete()
        guard !token.isEmpty else { return }
        cached = token
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecValueData as String: Data(token.utf8)]
        SecItemAdd(q as CFDictionary, nil)
    }

    static func delete() {
        cached = nil
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service] as CFDictionary)
    }
}

/// Installed models: catalog downloads in the models folder, plus folders the user added.
@MainActor @Observable
final class ModelStore {
    enum Status: Equatable {
        case notInstalled
        case checking
        case downloading(ModelDownloader.Progress)
        case installed(URL)
        case failed(String)

        static func == (a: Status, b: Status) -> Bool {
            switch (a, b) {
            case (.notInstalled, .notInstalled), (.checking, .checking): return true
            case (.downloading(let x), .downloading(let y)): return x.done == y.done && x.file == y.file
            case (.installed(let x), .installed(let y)): return x == y
            case (.failed(let x), .failed(let y)): return x == y
            default: return false
            }
        }
    }

    /// A folder added by hand, with what it was recognized as.
    struct ManualFolder: Codable, Hashable, Identifiable {
        var path: String
        var className: String?
        var id: String { path }
        var url: URL { URL(fileURLWithPath: path) }
    }

    var modelsDirectory: URL {
        didSet {
            UserDefaults.standard.set(modelsDirectory.path, forKey: "modelsDirectory")
            ModelPaths.directory = modelsDirectory
            refresh()
        }
    }
    var manualFolders: [ManualFolder] {
        didSet { if let d = try? JSONEncoder().encode(manualFolders) { UserDefaults.standard.set(d, forKey: "modelFolders") } }
    }
    private(set) var status: [String: Status] = [:]
    /// File lists fetched for a pending download, keyed by catalog id.
    /// Files to fetch per part of an entry (aligned with `entry.parts`), once its file lists are known.
    private(set) var plans: [String: [[RemoteFile]]] = [:]
    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]

    init() {
        let d = UserDefaults.standard
        modelsDirectory = d.string(forKey: "modelsDirectory").map { URL(fileURLWithPath: $0) }
            ?? ModelPaths.defaultDirectory
        manualFolders = d.data(forKey: "modelFolders").flatMap { try? JSONDecoder().decode([ManualFolder].self, from: $0) } ?? []
        ModelPaths.directory = modelsDirectory
        refresh()
    }

    func folder(for entry: CatalogEntry) -> URL { modelsDirectory.appendingPathComponent(entry.folder) }

    /// A folder counts as installed when it has weights and no unfinished `.part` files.
    static func isComplete(_ url: URL) -> Bool {
        guard ModelRegistry.weightBytes(at: url) > 0,
              let e = FileManager.default.enumerator(at: url, includingPropertiesForKeys: nil) else { return false }
        for case let f as URL in e where f.pathExtension == "part" { return false }
        return true
    }

    /// Folders found outside the models directory (the app's current image model), checked by class name.
    @ObservationIgnored var knownFolders: [URL] = []

    /// Remembers a folder in use elsewhere so a matching catalog entry shows as installed.
    func adopt(_ path: String) {
        let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL
        if !knownFolders.contains(url) { knownFolders.append(url) }
        refresh()
    }

    func refresh() {
        let elsewhere = knownFolders + manualFolders.map(\.url)
        for entry in ModelCatalog.entries {
            if case .downloading? = status[entry.id] { continue }
            let url = folder(for: entry)
            if Self.isComplete(url) {
                status[entry.id] = .installed(url)
            } else if let cls = entry.className,
                      let found = elsewhere.first(where: { ModelRegistry.className(at: $0) == cls && Self.isComplete($0) }) {
                status[entry.id] = .installed(found)
            } else {
                status[entry.id] = .notInstalled
            }
        }
    }

    /// The installed folder for a kind (catalog download, or a matching manual folder), if any.
    func installed(_ kind: CatalogEntry.Kind) -> URL? {
        for entry in ModelCatalog.entries where entry.kind == kind && entry.supported {
            if case .installed(let url)? = status[entry.id] { return url }
        }
        return nil
    }

    /// The installed folder of one catalog entry.
    func installedURL(_ id: String) -> URL? {
        if case .installed(let u)? = status[id] { return u }
        return nil
    }

    /// The image models the Image panel can switch between.
    var imageChoices: (qwen: URL?, flux: URL?) { (installedURL("qwen-image-2.1-turbo"), installedURL("flux-1-dev")) }

    /// The video models the Video panel can switch between.
    var videoChoices: (ltx: URL?, wanT2V: URL?, wanI2V: URL?) {
        (installedURL("ltx-2.3"), installedURL("wan-2.2-t2v"), installedURL("wan-2.2-i2v"))
    }

    /// The video model a new Video graph starts with: the family last chosen in the Video panel, if installed.
    func preferredVideoPath() -> String? {
        let c = videoChoices
        if UserDefaults.standard.string(forKey: "videoFamily") == VideoFamily.wan.rawValue, let u = c.wanT2V ?? c.wanI2V {
            return u.path
        }
        return (c.ltx ?? c.wanT2V ?? c.wanI2V ?? installed(.video))?.path
    }

    // MARK: downloads

    /// Fetches the file list so the user sees the real size before agreeing to download.
    func prepare(_ entry: CatalogEntry) async {
        status[entry.id] = .checking
        do {
            var lists: [[RemoteFile]] = []
            for part in entry.parts {
                let source = HTTPByteSource(repo: part.repo, revision: part.revision, token: part.gated ? HFToken.read() : nil)
                lists.append(HFAPI.select(try await source.list(), include: part.include, exclude: part.exclude))
            }
            plans[entry.id] = lists
            status[entry.id] = .notInstalled
        } catch {
            status[entry.id] = .failed(String(describing: error))
        }
    }

    /// Unfinished `.part` files from an earlier, interrupted download.
    func hasPartialDownload(_ entry: CatalogEntry) -> Bool {
        guard let e = FileManager.default.enumerator(at: folder(for: entry), includingPropertiesForKeys: nil) else { return false }
        for case let f as URL in e where f.pathExtension == "part" { return true }
        return false
    }

    func remainingBytes(_ entry: CatalogEntry) -> Int64? {
        plans[entry.id].map { lists in
            zip(entry.parts, lists).reduce(Int64(0)) { sum, pl in
                sum + ModelDownloader.remainingBytes(pl.1, in: partFolder(entry, pl.0))
            }
        }
    }

    func partFolder(_ entry: CatalogEntry, _ part: CatalogPart) -> URL {
        part.subfolder.isEmpty ? folder(for: entry) : folder(for: entry).appendingPathComponent(part.subfolder)
    }

    func start(_ entry: CatalogEntry) {
        guard let lists = plans[entry.id], tasks[entry.id] == nil else { return }
        let dest = folder(for: entry)
        let total = lists.joined().reduce(Int64(0)) { $0 + $1.size }
        status[entry.id] = .downloading(.init(done: 0, total: total, file: ""))
        tasks[entry.id] = Task {
            var lastUpdate = Date.distantPast
            do {
                var before: Int64 = 0  // bytes of the parts already fetched, so progress spans the whole entry
                for (part, planned) in zip(entry.parts, lists) {
                    let source = HTTPByteSource(repo: part.repo, revision: part.revision,
                                                token: part.gated ? HFToken.read() : nil)
                    // A gated repo listed before access was granted hides its checksums; list it again now.
                    let files = part.gated
                        ? HFAPI.select(try await source.list(), include: part.include, exclude: part.exclude) : planned
                    let offset = before
                    try await ModelDownloader.download(files, to: partFolder(entry, part), from: source) { p in
                        // Progress arrives per chunk; publish a few times a second.
                        guard Date().timeIntervalSince(lastUpdate) > 0.2 || p.done == p.total else { return }
                        lastUpdate = Date()
                        let q = ModelDownloader.Progress(done: offset + p.done, total: total, file: p.file)
                        Task { @MainActor in if case .downloading? = self.status[entry.id] { self.status[entry.id] = .downloading(q) } }
                    }
                    before += files.reduce(0) { $0 + $1.size }
                }
                if let manifest = entry.manifest, ModelRegistry.className(at: dest) == nil,
                   let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]) {
                    try? data.write(to: dest.appendingPathComponent("mlxcomfy.json"))
                }
                status[entry.id] = .installed(dest)
            } catch is CancellationError {
                status[entry.id] = .notInstalled
            } catch {
                status[entry.id] = .failed(String(describing: error))
            }
            tasks[entry.id] = nil
        }
    }

    /// Stops a download; the partial files stay so it resumes next time.
    func cancel(_ entry: CatalogEntry) {
        tasks[entry.id]?.cancel()
    }

    /// Moves the installed copy (wherever it is) to the Trash; the view confirms first.
    func moveToTrash(_ entry: CatalogEntry) {
        let url: URL = { if case .installed(let u)? = status[entry.id] { return u }; return folder(for: entry) }()
        try? FileManager.default.trashItem(at: url, resultingItemURL: nil)
        knownFolders.removeAll { $0 == url }
        plans[entry.id] = nil
        refresh()
    }

    // MARK: manual folders

    func addFolder(_ url: URL) {
        let folder = ManualFolder(path: url.standardizedFileURL.path, className: ModelRegistry.className(at: url))
        manualFolders.removeAll { $0.path == folder.path }
        manualFolders.append(folder)
        refresh()
    }

    func removeFolder(_ folder: ManualFolder) { manualFolders.removeAll { $0 == folder } }
}

/// The Models window: catalog downloads, folders added by hand, and the Hugging Face token.
struct ModelManagerView: View {
    @Environment(AppModel.self) private var model
    @Environment(ModelStore.self) private var store
    @State private var confirming: CatalogEntry?
    @State private var trashing: CatalogEntry?
    @State private var token = ""
    @State private var tokenSaved = HFToken.isSaved

    var body: some View {
        MGForm(maxWidth: 760) {
            MGSection {
                ForEach(ModelCatalog.entries) { entry in row(entry) }
            } header: {
                HStack {
                    Text("Available Models")
                    Spacer()
                    Button("Open Models Folder") {
                        try? FileManager.default.createDirectory(at: store.modelsDirectory, withIntermediateDirectories: true)
                        NSWorkspace.shared.open(store.modelsDirectory)
                    }
                    .buttonStyle(.mg(.borderless)).font(.mgSubheadline)
                }
            } footer: {
                Text("Downloads go to \(store.modelsDirectory.path) and resume if interrupted. Each model has its own license; you accept it before downloading.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
            }

            MGSection("Model Folders Added by Hand") {
                if store.manualFolders.isEmpty {
                    Text("Use a model you downloaded yourself: add its folder (diffusers layout with model_index.json, or an mlxcomfy.json manifest).")
                        .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                }
                ForEach(store.manualFolders) { f in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(f.url.lastPathComponent)
                            Text(f.className.map { ModelRegistry.shared.isRegistered($0) ? $0 : "\($0) (not supported)" }
                                 ?? "Not recognized: no model_index.json")
                                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                        }
                        Spacer()
                        if let c = f.className, ModelRegistry.shared.isRegistered(c) {
                            Button("Use for Images") { model.applyModelPath(f.path, undoManager: nil); model.preloadModel() }
                        }
                        Button(role: .destructive) { store.removeFolder(f) } label: { Image(systemName: "minus.circle") }
                            .buttonStyle(.mg(.icon))
                    }
                }
                Button("Add Model Folder…") {
                    let p = NSOpenPanel()
                    p.canChooseDirectories = true
                    p.canChooseFiles = false
                    if p.runModal() == .OK, let u = p.url { store.addFolder(u) }
                }
            }

            MGSection {
                MGFormRow("Token") {
                SecureField("Token", text: $token, prompt: Text(tokenSaved ? "Saved in the Keychain — type to replace" : "hf_…"))
                    .frame(maxWidth: 320)
                }
                HStack {
                    Button("Save to Keychain") { HFToken.save(token); token = ""; tokenSaved = true }
                        .disabled(token.isEmpty)
                    Button("Remove") { HFToken.delete(); token = ""; tokenSaved = false }
                        .disabled(!tokenSaved && token.isEmpty)
                    if tokenSaved { Label("Saved", systemImage: "checkmark.circle.fill").font(.mgSubheadline).foregroundStyle(MG.success) }
                }
            } header: {
                Text("Hugging Face Token")
            } footer: {
                Text("Only needed for gated models (marked with a lock). Create a read token on huggingface.co and accept the model's terms on its page first.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
            }
        }
        .frame(minWidth: 620, minHeight: 560)
        .onAppear { store.adopt(model.modelPath) }
        .mgSheet("download", item: $confirming) { entry in DownloadConfirmation(entry: entry).environment(store) }
        .mgConfirmationDialog("Move \(trashing?.name ?? "") to the Trash?", isPresented: Binding(
            get: { trashing != nil }, set: { if !$0 { trashing = nil } }), message: trashMessage) {
            [.destructive("Move to Trash") { if let e = trashing { store.moveToTrash(e) }; trashing = nil },
             .cancel { trashing = nil }]
        }
    }

    var trashMessage: String? {
        guard let e = trashing, case .installed(let url)? = store.status[e.id] else { return nil }
        return String(format: "%@ (%.1f GB) goes to the Trash. You can restore it from there.", url.path,
                      Double(ModelRegistry.weightBytes(at: url)) / 1e9)
    }

    @ViewBuilder
    func row(_ entry: CatalogEntry) -> some View {
        let status = store.status[entry.id] ?? .notInstalled
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: entry.kind.symbol)
                .font(.mgTitle3).frame(width: 24).foregroundStyle(MG.secondaryLabel)
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(entry.name).fontWeight(.medium)
                    if entry.gated { Image(systemName: "lock.fill").font(.caption2).foregroundStyle(MG.secondaryLabel) }
                }
                Text(entry.notes).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                HStack(spacing: 8) {
                    Text(String(format: "about %.1f GB", entry.approxGB))
                    Link(entry.license, destination: URL(string: entry.licenseURL)!)
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                switch status {
                case .downloading(let p):
                    MGProgress(value: p.fraction)
                    Text(String(format: "%.1f of %.1f GB · %@", Double(p.done) / 1e9, Double(p.total) / 1e9,
                                (p.file as NSString).lastPathComponent))
                        .font(.mgFootnote.monospacedDigit()).foregroundStyle(MG.secondaryLabel)
                case .failed(let msg):
                    Text(msg).font(.mgSubheadline).foregroundStyle(MG.destructive).lineLimit(3)
                default:
                    EmptyView()
                }
            }
            Spacer()
            actions(entry, status)
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    func actions(_ entry: CatalogEntry, _ status: ModelStore.Status) -> some View {
        switch status {
        case .installed(let url):
            VStack(alignment: .trailing, spacing: 4) {
                Label("Installed", systemImage: "checkmark.circle.fill").foregroundStyle(MG.success).font(.mgSubheadline)
                HStack {
                    if entry.kind == .image {
                        Button("Use") { model.applyModelPath(url.path, undoManager: nil); model.preloadModel() }
                    }
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: { Image(systemName: "folder") }
                    Button(role: .destructive) { trashing = entry } label: { Image(systemName: "trash") }
                }
                .controlSize(.small)
            }
        case .downloading:
            Button("Stop") { store.cancel(entry) }
        case .checking:
            MGSpinner().controlSize(.small)
        case .notInstalled, .failed:
            if entry.supported {
                Button(store.hasPartialDownload(entry) ? "Resume…" : "Download…") {
                    confirming = entry
                }
            } else {
                Text("Coming later").font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
            }
        }
    }
}

/// Shows the real size, free space and license, and starts the download only when the user agrees.
struct DownloadConfirmation: View {
    @Environment(ModelStore.self) private var store
    @Environment(\.mgDismiss) private var dismiss
    let entry: CatalogEntry
    @State private var accepted = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Download \(entry.name)?").font(.mgHeadline)
            if let lists = store.plans[entry.id], let remaining = store.remainingBytes(entry) {
                let free = ModelDownloader.freeSpace(at: store.modelsDirectory)
                ForEach(Array(zip(entry.parts, lists).enumerated()), id: \.offset) { _, pl in
                    LabeledContent(pl.0.subfolder.isEmpty ? "Files" : "Plus", value: "\(pl.1.count) from huggingface.co/\(pl.0.repo)")
                }
                LabeledContent("To download", value: String(format: "%.2f GB", Double(remaining) / 1e9))
                LabeledContent("Free space", value: String(format: "%.1f GB", Double(free) / 1e9))
                if remaining > free { Text("Not enough disk space.").foregroundStyle(MG.destructive) }
                Toggle(isOn: $accepted) {
                    VStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(Set(entry.parts.map(\.license)).sorted().enumerated()), id: \.offset) { i, lic in
                            HStack(spacing: 4) {
                                Text(i == 0 ? "I accept the" : "and the")
                                if let part = entry.parts.first(where: { $0.license == lic }), let url = URL(string: part.licenseURL) {
                                    Link(lic, destination: url)
                                }
                            }
                        }
                    }
                }
                HStack {
                    Spacer()
                    Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                    Button("Download") { store.start(entry); dismiss() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(!accepted || remaining > free)
                }
            } else if case .failed(let msg)? = store.status[entry.id] {
                Text(msg).foregroundStyle(MG.destructive)
                HStack { Spacer(); Button("Close") { dismiss() } }
            } else {
                HStack { MGSpinner().controlSize(.small); Text("Checking the file list…") }
                HStack { Spacer(); Button("Cancel") { dismiss() } }
            }
        }
        .padding(20)
        .frame(width: 460)
        .task { if store.plans[entry.id] == nil { await store.prepare(entry) } }
    }
}
