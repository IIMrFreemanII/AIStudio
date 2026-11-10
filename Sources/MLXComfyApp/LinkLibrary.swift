import AppKit
import Foundation
import MLXDiffusionCore
import ModelImport
import NodesRemote
import NodesStandard
import Observation

/// The app's side of the library (`<models>/library.json`, see `Library` in NodesStandard): LoRAs (and, later,
/// models) added from Civitai / Hugging Face links or files, where each is installed — this Mac, each GPU server —
/// and the jobs that install or remove them. The Models window draws it; the LoRA nodes pick from it.
@MainActor @Observable
final class LinkLibrary {
    enum State: Equatable {
        case unknown
        case checking
        case installed
        case absent
        case working(String)
        case failed(String)
    }

    private(set) var items: [LibraryEntry] = []
    /// This Mac's state per item, while something's happening (otherwise read from the item's `localFile`).
    private(set) var macJobs: [String: State] = [:]
    /// Each GPU server's state per item: [item id: [server id: state]].
    private(set) var server: [String: [String: State]] = [:]
    @ObservationIgnored private var tasks: [String: Task<Void, Never>] = [:]
    @ObservationIgnored let models: () -> URL
    @ObservationIgnored private var observer: NSObjectProtocol?

    init(models: @escaping () -> URL) {
        self.models = models
        reload()
        observer = NotificationCenter.default.addObserver(forName: Library.changed, object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.reload() }
        }
    }

    func reload() { items = Library.load(models: models()) }

    func item(_ id: String) -> LibraryEntry? { items.first { $0.id == id } }

    /// Fine-tunes of a family (the cloud FLUX node's `model` menu).
    func models(for family: String) -> [LibraryEntry] {
        items.filter { $0.kind == .model && $0.family == family }
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    /// LoRAs that fit a model family (nil family: all). Items of unknown family fit everything.
    func loras(for family: String?) -> [LibraryEntry] {
        items.filter { $0.kind == .lora && (family == nil || $0.family == nil || Self.fits($0.family!, family!)) }
            .sorted { $0.title.localizedCaseInsensitiveCompare($1.title) == .orderedAscending }
    }

    /// Whether an item's family suits a target family (Qwen-Image 2.1 runs only its own LoRAs, FLUX.1 the FLUX ones).
    nonisolated static func fits(_ item: String, _ target: String) -> Bool {
        item == target || (item.hasPrefix("wan-2.2") && target.hasPrefix("wan-2.2"))
    }

    /// Whole models from links install on this Mac too (converted by ModelImport, next to the base model).
    nonisolated static let macModelsReady = true
    /// The families a fine-tune can be installed for on this Mac, and the catalog folder of the base it reuses.
    nonisolated static let macModelBases = ["flux.1": "flux-1-dev", "qwen-image-2.1": "qwen-turbo-mlx",
                                            "wan-2.2-t2v": "wan-2.2-t2v", "wan-2.2-i2v": "wan-2.2-i2v"]
    /// The families a GPU server runs fine-tunes of (server/gpu variants.py).
    nonisolated static let serverModelFamilies: Set<String> = ["flux.1", "qwen-image", "wan-2.2-t2v", "wan-2.2-i2v"]

    /// Why a fine-tune of `family` can't be installed on this Mac (nil: it can).
    nonisolated static func macModelRefusal(_ family: String?) -> String? {
        guard let family else { return "pick its family" }
        if macModelBases[family] != nil { return nil }
        if family == "qwen-image" { return "Qwen-Image (20B) runs on a GPU server; this Mac runs Qwen-Image 2.1" }
        return "\(LoRASources.familyName(family)) fine-tunes come in a later update"
    }

    /// Why a fine-tune of `family` can't be installed on a GPU server (nil: it can).
    nonisolated static func serverModelRefusal(_ family: String?) -> String? {
        guard let family else { return "pick its family" }
        if serverModelFamilies.contains(family) { return nil }
        if family == "qwen-image-2.1" { return "GPU servers run the original Qwen-Image, not 2.1" }
        return "\(LoRASources.familyName(family)) fine-tunes come in a later update"
    }

    func macState(_ item: LibraryEntry) -> State {
        if let s = macJobs[item.id] { return s }
        if item.kind == .model && !Self.macModelsReady { return .absent }
        return Library.localURL(item, models: models()) != nil ? .installed : .absent
    }

    func serverState(_ item: LibraryEntry, _ serverID: String) -> State { server[item.id]?[serverID] ?? .unknown }

    func thumbnail(_ item: LibraryEntry) -> URL? { item.thumbnail.map { models().appendingPathComponent($0) } }

    // MARK: adding

    /// Adds a resolved link and starts installing it on the chosen targets.
    @discardableResult
    func add(_ remote: LoRARemote, link: String, title: String, family: String?, mac: Bool, servers: [String]) -> String {
        let id = Library.newID(for: title, existing: items)
        let item = LibraryEntry(id: id, kind: remote.kind, family: family, title: title, link: link, source: remote,
                               fileName: remote.fileName, sha256: remote.sha256, bytes: remote.totalBytes,
                               baseModel: remote.baseModel, triggerWords: remote.triggerWords)
        save(items + [item])
        fetchThumbnail(id, remote.previewImage)
        if mac { installOnMac(id) }
        for s in servers { installOnServer(id, s) }
        return id
    }

    /// Adds a LoRA file from disk: copied into `<models>/loras`, then uploaded to the chosen servers.
    @discardableResult
    func addFile(_ url: URL, title: String, family: String?, servers: [String]) throws -> String {
        let id = Library.newID(for: title, existing: items)
        let folder = LoRADownloads.folder(models: models())
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var dest = folder.appendingPathComponent(url.lastPathComponent)
        if FileManager.default.fileExists(atPath: dest.path), dest.standardizedFileURL != url.standardizedFileURL {
            dest = folder.appendingPathComponent("\(id)-\(url.lastPathComponent)")
        }
        if dest.standardizedFileURL != url.standardizedFileURL { try FileManager.default.copyItem(at: url, to: dest) }
        let size = Int64((try? dest.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        let item = LibraryEntry(id: id, kind: .lora, family: family, title: title, link: nil, source: nil,
                               fileName: url.lastPathComponent, sha256: nil, bytes: size, baseModel: nil, triggerWords: [],
                               localFile: "loras/" + dest.lastPathComponent)
        save(items + [item])
        // The SHA-256 names it on GPU servers; hash once, in the background.
        let models = models()
        macJobs[id] = .working("Checking \(url.lastPathComponent)")
        tasks["hash:\(id)"] = Task {
            let sha = try? await Task.detached { try ModelDownloader.sha256(of: dest) }.value
            if let sha {
                try? Library.update(id, models: models) { $0.sha256 = sha }
                try? sha.write(to: dest.appendingPathExtension("sha256"), atomically: true, encoding: .utf8)
            }
            macJobs[id] = nil
            reload()
            for s in servers { installOnServer(id, s) }
        }
        return id
    }

    // MARK: installing

    func installOnMac(_ id: String) {
        guard tasks["mac:\(id)"] == nil else { return }
        if let item = item(id), item.kind == .model { return installModelOnMac(item) }
        let models = models()
        macJobs[id] = .working("Starting")
        tasks["mac:\(id)"] = Task {
            do {
                try await Library.installLocally(id, models: models) { line in
                    Task { @MainActor in self.macJobs[id] = .working(line) }
                }
                macJobs[id] = nil
            } catch is CancellationError {
                macJobs[id] = nil
            } catch {
                macJobs[id] = .failed(String(describing: error))
            }
            tasks["mac:\(id)"] = nil
            reload()
        }
    }

    func installOnServer(_ id: String, _ serverID: String) {
        guard tasks["srv:\(id):\(serverID)"] == nil, let item = item(id) else { return }
        if item.kind == .model { return importModelOnServer(item, serverID) }
        guard let endpoint = RemoteEndpoints.resolve(serverID) else {
            setServer(id, serverID, .failed("the GPU server was removed")); return
        }
        let client = GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
        let local = Library.localURL(item, models: models())
        setServer(id, serverID, .working("Starting"))
        tasks["srv:\(id):\(serverID)"] = Task {
            let status: @Sendable (String) -> Void = { line in
                Task { @MainActor in self.setServer(id, serverID, .working(line)) }
            }
            do {
                if let source = item.source {
                    _ = try await serverFetch(source, client: client, status: status)
                    if let low = source.companionRemote { _ = try await serverFetch(low, client: client, status: status) }
                } else if let local {
                    _ = try await client.uploadFile(local, ext: "lora", progress: uploadReporter(item.fileName, status: status))
                } else {
                    throw LibraryError.notOnThisMac(item.title)
                }
                setServer(id, serverID, .installed)
            } catch is CancellationError {
                setServer(id, serverID, .unknown)
            } catch {
                setServer(id, serverID, .failed(String(describing: error)))
            }
            tasks["srv:\(id):\(serverID)"] = nil
        }
    }

    /// A whole model (a fine-tune): the server downloads and converts it next to its base model, on CPUs; this
    /// follows its progress.
    private func importModelOnServer(_ item: LibraryEntry, _ serverID: String) {
        guard let endpoint = RemoteEndpoints.resolve(serverID) else {
            setServer(item.id, serverID, .failed("the GPU server was removed")); return
        }
        guard let source = item.source, let family = item.family else {
            setServer(item.id, serverID, .failed("no link or family to install from")); return
        }
        if let why = Self.serverModelRefusal(family) { setServer(item.id, serverID, .failed(why)); return }
        let files: [GPUJobClient.ImportFile]
        do { files = try Self.importFiles(source, family: family) } catch {
            setServer(item.id, serverID, .failed(String(describing: error))); return
        }
        let client = GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
        let id = item.id
        setServer(id, serverID, .working("Starting"))
        tasks["srv:\(id):\(serverID)"] = Task {
            do {
                let repo = Self.huggingFaceRepo(source)
                _ = try await client.importModel(family: family, id: id, name: item.title,
                                                 files: repo == nil ? files : [], repo: repo)
                let began = Date()
                while true {
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                    let listing = try await client.models()
                    if listing.families[family]?.contains(where: { $0.id == id }) == true { break }
                    guard let state = listing.imports["\(family)/\(id)"] else { continue }
                    if state.state == "failed" { throw RemoteNodeError.fetchFailed(item.title, state.error ?? "failed") }
                    setServer(id, serverID, .working(Self.importLine(state, since: began, name: item.fileName)))
                }
                setServer(id, serverID, .installed)
            } catch is CancellationError {
                setServer(id, serverID, .unknown)
            } catch {
                setServer(id, serverID, .failed(String(describing: error)))
            }
            tasks["srv:\(id):\(serverID)"] = nil
        }
    }

    /// The files a server imports a fine-tune from: one, or a Wan 2.2 expert each (`high` / `low`).
    nonisolated static func importFiles(_ source: LoRARemote, family: String) throws -> [GPUJobClient.ImportFile] {
        var files = [GPUJobClient.ImportFile(url: source.downloadURL, sha256: source.sha256)]
        guard family.hasPrefix("wan-2.2") else { return files }
        guard let expert = source.expert else { throw LibraryError.wanExpertUnknown(source.fileName) }
        files[0].role = expert
        if let c = source.companion {
            files.append(GPUJobClient.ImportFile(url: c.downloadURL, sha256: c.sha256, role: expert == "high" ? "low" : "high"))
        }
        return files
    }

    /// "Server downloading 3.1 / 11.6 GB · 80 MB/s · 2 min left", "Converting".
    nonisolated static func importLine(_ s: GPUModelListing.Import, since: Date, name: String) -> String {
        switch s.state {
        case "converting": return "Converting"
        case "downloading":
            let verb = (s.files ?? 1) > 1 ? "Server downloading \(s.file ?? 1)/\(s.files!)" : "Server downloading"
            return TransferStatus.line(verb: verb, sent: s.bytes ?? 0, total: s.total,
                                       seconds: Date().timeIntervalSince(since))
        default: return "Starting"
        }
    }

    /// `owner/name@revision` for a Hugging Face diffusers repo link (the resolver's tree URL); nil for a file.
    nonisolated static func huggingFaceRepo(_ r: LoRARemote) -> String? {
        guard r.host == .huggingface, r.kind == .model else { return nil }
        let parts = r.downloadURL.path.split(separator: "/").map(String.init)
        guard parts.count >= 4, parts[2] == "tree" else { return nil }
        return "\(parts[0])/\(parts[1])@\(parts[3...].joined(separator: "/"))"
    }

    /// Asks each GPU server which items it has (a HEAD per LoRA by SHA-256, one listing for models): cheap, and
    /// never starts a GPU.
    func checkServers(_ serverIDs: [String]) {
        for serverID in serverIDs {
            guard let endpoint = RemoteEndpoints.resolve(serverID) else { continue }
            let client = GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
            let models = items.filter { $0.kind == .model && tasks["srv:\($0.id):\(serverID)"] == nil }
            if !models.isEmpty {
                Task {
                    let listing = try? await client.models()
                    for m in models where tasks["srv:\(m.id):\(serverID)"] == nil {
                        let has = listing?.families[m.family ?? ""]?.contains { $0.id == m.id }
                        setServer(m.id, serverID, has.map { $0 ? .installed : .absent } ?? .unknown)
                    }
                }
            }
            for item in items where item.kind == .lora && tasks["srv:\(item.id):\(serverID)"] == nil {
                guard let name = item.serverName else { setServer(item.id, serverID, .absent); continue }
                if serverState(item, serverID) == .unknown { setServer(item.id, serverID, .checking) }
                Task {
                    let has = try? await client.has(name)
                    if tasks["srv:\(item.id):\(serverID)"] == nil {
                        setServer(item.id, serverID, has.map { $0 ? .installed : .absent } ?? .unknown)
                    }
                }
            }
        }
    }

    /// A fine-tune on this Mac: download the checkpoint (or a Hugging Face repo's `transformer/`), convert it to the
    /// app's layout (an 8-bit transformer, FLUX.1 dev or Qwen-Image 2.1) and link the base model's text encoders and
    /// VAE: `<models>/<id>`.
    private func installModelOnMac(_ item: LibraryEntry) {
        let id = item.id
        let models = models()
        guard let family = item.family, let baseName = Self.macModelBases[family] else {
            macJobs[id] = .failed(Self.macModelRefusal(item.family) ?? "not for this Mac")
            return
        }
        let base = models.appendingPathComponent(baseName).resolvingSymlinksInPath()
        guard ModelStore.isComplete(base), ModelRegistry.className(at: base) != nil else {
            macJobs[id] = .failed("Install \(LoRASources.familyName(family)) from Available Models first: fine-tunes reuse its text encoders and VAE")
            return
        }
        guard let source = item.source else { macJobs[id] = .failed("no link to download from"); return }
        let dest = models.appendingPathComponent(id)
        let downloads = models.appendingPathComponent(".downloads")
        macJobs[id] = .working("Starting")
        tasks["mac:\(id)"] = Task {
            var scratch: URL?
            do {
                let checkpoint: URL
                if let repo = Self.huggingFaceRepo(source) {
                    let parts = repo.split(separator: "@", maxSplits: 1).map(String.init)
                    let src = HTTPByteSource(repo: parts[0], revision: parts.count > 1 ? parts[1] : "main",
                                             token: KeychainToken.huggingFace.read())
                    let wan = family.hasPrefix("wan-2.2")
                    let files = HFAPI.select(try await src.list(), include: wan ? ["transformer/*", "transformer_2/*"]
                                             : ["transformer/*"], exclude: [])
                    let dir = downloads.appendingPathComponent(id)
                    scratch = dir
                    let began = Date()
                    try await ModelDownloader.download(files, to: dir, from: src) { p in
                        let line = TransferStatus.line(verb: "Downloading", sent: p.done, total: p.total,
                                                       seconds: Date().timeIntervalSince(began))
                        Task { @MainActor in self.macJobs[id] = .working(line) }
                    }
                    // Wan: the folder with both experts (transformer = high noise, transformer_2 = low).
                    checkpoint = wan ? dir : dir.appendingPathComponent("transformer")
                } else {
                    checkpoint = try await LoRADownloads.fetch(source, models: models, into: downloads) { line in
                        Task { @MainActor in self.macJobs[id] = .working(line) }
                    }
                    scratch = checkpoint
                }
                macJobs[id] = .working("Converting")
                let link = item.link ?? ""
                // Wan 2.2: each expert's file (a pair, or one; the other expert stays the base's).
                var experts: [String: URL] = [:]
                if family.hasPrefix("wan-2.2"), Self.huggingFaceRepo(source) == nil {
                    guard let e = source.expert else { throw LibraryError.wanExpertUnknown(source.fileName) }
                    experts[e] = checkpoint
                    if let c = source.companion {
                        experts[e == "high" ? "low" : "high"] = checkpoint.deletingLastPathComponent()
                            .appendingPathComponent(c.cacheName)
                    }
                }
                let wanFiles = experts
                _ = try await Task.detached(priority: .utility) {
                    let report: (ImportProgress) -> Void = { p in
                        let line = "Converting \(Int(p.fraction * 100)) %"
                        Task { @MainActor in self.macJobs[id] = .working(line) }
                    }
                    if !wanFiles.isEmpty {
                        return try ModelImporter.importWan(high: wanFiles["high"], low: wanFiles["low"], base: base,
                                                           into: dest, sourceInfo: ["link": link], progress: report)
                    }
                    return try ModelImporter.importTransformer(from: checkpoint, family: family, base: base, into: dest,
                                                               sourceInfo: ["link": link], progress: report)
                }.value
                for (_, url) in experts where url != checkpoint {
                    try? FileManager.default.removeItem(at: url)
                    try? FileManager.default.removeItem(at: url.appendingPathExtension("sha256"))
                }
                if let scratch { try? FileManager.default.removeItem(at: scratch) }
                try? FileManager.default.removeItem(at: (scratch ?? checkpoint).appendingPathExtension("sha256"))
                try Library.update(id, models: models) { $0.localFile = id }
                macJobs[id] = nil
            } catch is CancellationError {
                macJobs[id] = nil
            } catch {
                macJobs[id] = .failed(String(describing: error))
            }
            tasks["mac:\(id)"] = nil
            reload()
        }
    }

    /// Fine-tunes installed on this Mac: (folder, title), for the model menus.
    func macModels() -> [(url: URL, title: String)] {
        items.filter { $0.kind == .model }.compactMap { item in
            Library.localURL(item, models: models()).map { ($0, item.title) }
        }
    }

    // MARK: removing, renaming

    func removeFromMac(_ id: String) {
        tasks["mac:\(id)"]?.cancel()
        guard let item = item(id), let url = Library.localURL(item, models: models()) else { return }
        if let other = Library.companionURL(item, models: models()) {
            try? FileManager.default.removeItem(at: other)
            try? FileManager.default.removeItem(at: other.appendingPathExtension("sha256"))
        }
        try? FileManager.default.removeItem(at: url)
        try? FileManager.default.removeItem(at: url.appendingPathExtension("sha256"))
        try? Library.update(id, models: models()) { $0.localFile = nil }
    }

    func removeFromServer(_ id: String, _ serverID: String) {
        tasks["srv:\(id):\(serverID)"]?.cancel()
        guard let item = item(id), let endpoint = RemoteEndpoints.resolve(serverID) else { return }
        let client = GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
        setServer(id, serverID, .working("Removing"))
        Task {
            do {
                if item.kind == .model {
                    try await client.deleteModel(family: item.family ?? "", id: item.id)
                } else if let name = item.serverName {
                    try await client.deleteFile(name)
                }
                setServer(id, serverID, .absent)
            } catch {
                setServer(id, serverID, .failed(String(describing: error)))
            }
        }
    }

    /// Takes it out of the library, and off this Mac (servers keep their copy until removed there, or pruned).
    func remove(_ id: String) {
        for (k, t) in tasks where k.hasSuffix(":\(id)") || k.contains(":\(id):") { t.cancel() }
        removeFromMac(id)
        if let thumb = item(id).flatMap(thumbnail) { try? FileManager.default.removeItem(at: thumb) }
        save(items.filter { $0.id != id })
        server[id] = nil
        macJobs[id] = nil
    }

    func rename(_ id: String, _ title: String) {
        let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        try? Library.update(id, models: models()) { $0.title = t }
    }

    func setFamily(_ id: String, _ family: String?) {
        try? Library.update(id, models: models()) { $0.family = family }
    }

    // MARK: helpers

    private func save(_ new: [LibraryEntry]) {
        do { try Library.save(new, models: models()) } catch { NSSound.beep() }
        items = new
    }

    private func setServer(_ id: String, _ serverID: String, _ state: State) {
        server[id, default: [:]][serverID] = state
    }

    private func fetchThumbnail(_ id: String, _ url: URL?) {
        guard let url else { return }
        let models = models()
        Task {
            guard let (data, response) = try? await URLSession.shared.data(from: url),
                  (response as? HTTPURLResponse)?.statusCode == 200, NSImage(data: data) != nil else { return }
            let dir = models.appendingPathComponent("library/thumbs")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let ext = url.pathExtension.isEmpty ? "jpg" : url.pathExtension.lowercased()
            let file = dir.appendingPathComponent("\(id).\(ext)")
            guard (try? data.write(to: file)) != nil else { return }
            try? Library.update(id, models: models) { $0.thumbnail = "library/thumbs/\(id).\(ext)" }
        }
    }
}
