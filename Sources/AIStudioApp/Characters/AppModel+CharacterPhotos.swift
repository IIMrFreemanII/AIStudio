import CharacterCore
import Foundation
import LLMClient
import MovieCore
import NodesRemote

/// The pictures in a character chat: the photos a reply's tags ask for (SDXL or Chroma on the chat's Cloud GPU server,
/// keeping the face of the character's portrait), and the portraits themselves.
extension AppModel {
    /// What one photo may cost without asking (UserDefaults; 5 cents when unset).
    /// Automated runs keep it in memory: they must not change the user's setting.
    var characterAutoPhotoUSD: Double {
        get {
            if Self.isAutomatedRun { return Self.automatedAutoPhotoUSD }
            return UserDefaults.standard.object(forKey: "characterAutoPhotoUSD") as? Double ?? 0.05
        }
        set {
            if Self.isAutomatedRun { Self.automatedAutoPhotoUSD = newValue }
            else { UserDefaults.standard.set(newValue, forKey: "characterAutoPhotoUSD") }
        }
    }

    private static var automatedAutoPhotoUSD = 0.05

    /// The server a chat's photos are drawn on: the chat's own when it is a GPU server, else the first one.
    func photoServer(for doc: ChatDocument) -> String? {
        if let id = doc.transcript.serverID, connections.connection(id)?.kind == .gpu { return id }
        return connections.gpuServers.first?.id
    }

    /// What one more photo of `family` would cost on `server` (its job, plus loading the model when it's cold); nil
    /// when the price isn't known.
    func photoEstimate(server: String, family: ArtModel.Family = .sdxl) -> Double? {
        let task = CharacterArt.task(family)
        guard case .gpuOnline(let info)? = connections.status[server],
              let price = info.task(task)?.pricePerSecond else { return nil }
        let timings = GPUTimings.load(gpuTimingsURL)
        let warm = timings.isWarm(task: task, idleSeconds: Double(info.idleSeconds ?? 120))
        let mp = Double(CharacterArt.photoSize.width * CharacterArt.photoSize.height) / 1e6
        let seconds = timings.seconds(task: task, megapixels: mp) + (warm ? 0 : timings.loadSeconds(task: task))
        return seconds * price
    }

    // MARK: photos

    /// Makes a `ChatMedia` for each tag of the turn's text, and starts the cheap photos (the rest wait for the user).
    func startCharacterMedia(_ doc: ChatDocument, turnID: UUID, character c: Companion) {
        guard let turn = doc.transcript.turn(turnID), turn.error == nil, turn.speaker == c.id, turn.media == nil else { return }
        let requests = MediaTags.parse(turn.text).requests
        guard !requests.isEmpty else { return }
        let server = photoServer(for: doc)
        let estimate = server.flatMap { photoEstimate(server: $0, family: c.artModel.family) }
        let media = requests.map { r -> ChatMedia in
            guard r.kind == .photo else {
                return ChatMedia(kind: r.kind.rawValue, prompt: r.text, status: .skipped, error: "Only photos for now")
            }
            guard server != nil else {
                return ChatMedia(kind: r.kind.rawValue, prompt: r.text, status: .failed, error: "No GPU server for photos")
            }
            if let estimate, estimate > characterAutoPhotoUSD {
                return ChatMedia(kind: r.kind.rawValue, prompt: r.text, status: .waiting, estimateUSD: estimate)
            }
            return ChatMedia(kind: r.kind.rawValue, prompt: r.text, status: .running, estimateUSD: estimate)
        }
        doc.transcript.update(turnID) { $0.media = media }
        saveChat(doc)
        let running = media.filter { $0.status == .running }.map(\.id)
        guard !running.isEmpty else { return }
        Task { @MainActor [weak self, weak doc] in
            for id in running {
                guard let self, let doc else { return }
                await self.takePhoto(doc, turnID: turnID, mediaID: id, character: c)
            }
        }
    }

    /// The user said yes to a waiting photo.
    func approveMedia(_ doc: ChatDocument, turnID: UUID, mediaID: UUID) {
        guard let c = doc.transcript.turn(turnID)?.speaker.flatMap({ character($0) }) ?? characters(of: doc).first,
              doc.transcript.turn(turnID)?.media?.first(where: { $0.id == mediaID })?.status == .waiting else { return }
        setMedia(doc, turnID, mediaID) { $0.status = .running; $0.error = nil }
        Task { @MainActor [weak self, weak doc] in
            guard let self, let doc else { return }
            await self.takePhoto(doc, turnID: turnID, mediaID: mediaID, character: c)
        }
    }

    /// The user said no to a waiting photo.
    func skipMedia(_ doc: ChatDocument, turnID: UUID, mediaID: UUID) {
        setMedia(doc, turnID, mediaID) { $0.status = .skipped }
    }

    private func setMedia(_ doc: ChatDocument, _ turnID: UUID, _ mediaID: UUID, _ change: (inout ChatMedia) -> Void) {
        guard doc.transcript.turn(turnID)?.media?.contains(where: { $0.id == mediaID }) == true else { return }
        doc.transcript.update(turnID) { t in
            guard let i = t.media?.firstIndex(where: { $0.id == mediaID }) else { return }
            change(&t.media![i])
        }
        saveChat(doc)
    }

    private func failMedia(_ doc: ChatDocument, _ turnID: UUID, _ mediaID: UUID, _ error: Error) {
        let text = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        setMedia(doc, turnID, mediaID) { $0.status = .failed; $0.error = text }
    }

    private func takePhoto(_ doc: ChatDocument, turnID: UUID, mediaID: UUID, character c: Companion) async {
        guard let media = doc.transcript.turn(turnID)?.media?.first(where: { $0.id == mediaID }) else { return }
        do {
            try CharacterArt.check(media.prompt)
        } catch {
            failMedia(doc, turnID, mediaID, error)
            return
        }
        guard let server = photoServer(for: doc) else {
            setMedia(doc, turnID, mediaID) { $0.status = .failed; $0.error = "No GPU server for photos" }
            return
        }
        var face: URL?
        if let name = c.selectedPortrait {
            let url = characterStore.portraitsFolder(c.id).appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { face = url }
        }
        let scene = scene(of: doc)
        let render = CharacterArt.render(c, prompt: CharacterArt.photoPrompt(c, request: media.prompt, scene: scene),
                                         seed: Int.random(in: 0..<Int(Int32.max)), server: server, face: face)
        do {
            let (outputs, jobs) = try await runHeadlessGraph(template: CharacterArt.template(c.artModel.family, loras: c.loras.count),
                                                             render: render, title: "\(c.name)'s photo")
            // The jobs cost money even if the picture then can't be saved.
            let cost = recordGPUJobs(jobs)
            guard doc.transcript.turn(turnID) != nil else { return }
            guard let image = outputs.first(where: { $0.kind == .image })?.image else { throw MovieArtError.noImage }
            let name = "photo-\(mediaID.uuidString.prefix(8).lowercased()).png"
            let fm = FileManager.default
            let folder = chatStore.filesFolder(doc.id)
            try fm.createDirectory(at: folder, withIntermediateDirectories: true)
            let file = folder.appendingPathComponent(name)
            try MovieComposer.writePNG(image, to: file)
            let album = characterStore.albumFolder(c.id)
            try fm.createDirectory(at: album, withIntermediateDirectories: true)
            let copy = album.appendingPathComponent(name)
            try? fm.removeItem(at: copy)
            try fm.copyItem(at: file, to: copy)
            var m = characterStore.memory(c.id)
            m.album.append(AlbumItem(file: name, kind: .photo, prompt: media.prompt, scene: scene?.title, chatID: doc.id))
            m.spentUSD += cost ?? 0
            try? characterStore.saveMemory(m, for: c.id)
            setMedia(doc, turnID, mediaID) { $0.status = .done; $0.file = name; $0.costUSD = cost; $0.error = nil }
        } catch {
            failMedia(doc, turnID, mediaID, error)
        }
    }

    // MARK: portraits

    /// Draws `count` portrait candidates (no face reference) on the first GPU server; each is added to the
    /// character as it finishes (the first becomes the chosen one). Returns the new file names.
    func generatePortraits(for id: UUID, count: Int = 4, progress: @escaping @MainActor (String) -> Void = { _ in })
        async throws -> [String]
    {
        guard let c = character(id) else { throw PortraitError.noCharacter }
        guard let server = connections.gpuServers.first?.id else { throw PortraitError.noServer }
        let folder = characterStore.portraitsFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var names: [String] = []
        for i in 0..<count {
            try Task.checkCancellation()
            let render = CharacterArt.render(c, prompt: CharacterArt.portraitPrompt(c), seed: Int.random(in: 0..<Int(Int32.max)),
                                             server: server, face: nil)
            let (outputs, jobs) = try await runHeadlessGraph(
                template: CharacterArt.template(c.artModel.family, loras: c.loras.count), render: render,
                title: "\(c.name)'s portrait \(i + 1) of \(count)", progress: progress)
            let cost = recordGPUJobs(jobs)
            var m = characterStore.memory(id)
            m.spentUSD += cost ?? 0
            try? characterStore.saveMemory(m, for: id)
            guard let image = outputs.first(where: { $0.kind == .image })?.image else { throw MovieArtError.noImage }
            let name = "portrait-\(UUID().uuidString.prefix(8).lowercased()).png"
            try MovieComposer.writePNG(image, to: folder.appendingPathComponent(name))
            addPortrait(name, to: id)
            names.append(name)
        }
        return names
    }

    /// Copies a picture file into the character's portraits; returns its name.
    func importPortrait(_ url: URL, for id: UUID) throws -> String {
        let ext = url.pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "heic", "webp"].contains(ext) else { throw PortraitError.notAPicture }
        guard character(id) != nil else { throw PortraitError.noCharacter }
        let folder = characterStore.portraitsFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = "portrait-\(UUID().uuidString.prefix(8).lowercased()).\(ext)"
        try FileManager.default.copyItem(at: url, to: folder.appendingPathComponent(name))
        addPortrait(name, to: id)
        return name
    }

    private func addPortrait(_ name: String, to id: UUID) {
        guard var c = characterStore.load(id) else { return }
        c.portraits.append(name)
        if c.selectedPortrait == nil { c.selectedPortrait = name }
        saveCharacter(c)
    }

    func selectPortrait(_ name: String, for id: UUID) {
        guard var c = characterStore.load(id), c.portraits.contains(name), c.selectedPortrait != name else { return }
        c.selectedPortrait = name
        saveCharacter(c)
    }

    /// Takes the portrait out of the character and its file to the Trash.
    func deletePortrait(_ name: String, for id: UUID) {
        guard var c = characterStore.load(id), c.portraits.contains(name) else { return }
        c.portraits.removeAll { $0 == name }
        if c.selectedPortrait == name { c.selectedPortrait = c.portraits.last }
        saveCharacter(c)
        let url = characterStore.portraitsFolder(id).appendingPathComponent(name)
        if Self.isAutomatedRun { try? FileManager.default.removeItem(at: url) }
        else { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
    }
}

enum PortraitError: Error, CustomStringConvertible {
    case noCharacter, noServer, notAPicture
    var description: String {
        switch self {
        case .noCharacter: "This character doesn't exist."
        case .noServer: "Add a GPU server in Servers first."
        case .notAPicture: "Choose a PNG, JPEG, HEIC or WebP picture."
        }
    }
}
