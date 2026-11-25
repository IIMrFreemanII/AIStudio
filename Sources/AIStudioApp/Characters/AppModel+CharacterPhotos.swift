import CharacterCore
import Foundation
import LLMClient
import MovieCore
import NodesRemote

/// A character's portraits: drawn on the first GPU server (no face reference) or imported, and which one is the face
/// in their photos and videos.
extension AppModel {
    // MARK: portraits

    /// Draws `count` portrait candidates of the stored character `id`; see the other `generatePortraits`.
    func generatePortraits(for id: UUID, count: Int = 4, progress: @escaping @MainActor (String) -> Void = { _ in })
        async throws -> [String]
    {
        guard let c = character(id) else { throw PortraitError.noCharacter }
        return try await generatePortraits(for: c, count: count, progress: progress)
    }

    /// Draws `count` portrait candidates (no face reference) of `c` as given (saved or not yet: the files go to its
    /// portraits folder either way) on the first GPU server. Each is passed to `added` as it finishes, and added to
    /// the stored character when there is one (the first becomes the chosen one). Returns the new file names.
    func generatePortraits(for c: Companion, count: Int = 4, progress: @escaping @MainActor (String) -> Void = { _ in },
                           added: @escaping @MainActor (String) -> Void = { _ in }) async throws -> [String]
    {
        let id = c.id
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
            added(name)
        }
        return names
    }

    /// Copies a picture file into the character's portraits folder (saved or not yet), and into the stored
    /// character when there is one; returns its name.
    func importPortrait(_ url: URL, for id: UUID) throws -> String {
        let ext = url.pathExtension.lowercased()
        guard ["png", "jpg", "jpeg", "heic", "webp"].contains(ext) else { throw PortraitError.notAPicture }
        let folder = characterStore.portraitsFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let name = "portrait-\(UUID().uuidString.prefix(8).lowercased()).\(ext)"
        try FileManager.default.copyItem(at: url, to: folder.appendingPathComponent(name))
        addPortrait(name, to: id)
        return name
    }

    /// A character that was never saved (Cancel in the creator or a new character's editor): its folder holds only
    /// what was made for it, so it goes. A saved character's folder stays.
    func discardDraft(_ id: UUID) {
        let folder = characterStore.folder(id)
        guard !FileManager.default.fileExists(atPath: folder.appendingPathComponent("character.json").path) else { return }
        try? FileManager.default.removeItem(at: folder)
    }

    /// The selected portrait's file, when it exists.
    func portraitURL(_ c: Companion) -> URL? {
        guard let name = c.selectedPortrait else { return nil }
        let url = characterStore.portraitsFolder(c.id).appendingPathComponent(name)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
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
    /// Also for a character not saved yet (only the file goes).
    func deletePortrait(_ name: String, for id: UUID) {
        if var c = characterStore.load(id), c.portraits.contains(name) {
            c.portraits.removeAll { $0 == name }
            if c.selectedPortrait == name { c.selectedPortrait = c.portraits.last }
            saveCharacter(c)
        }
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
