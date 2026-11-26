import CharacterCore
import Foundation
import MovieCore
import NodesRemote

/// One Prompt Builder preview: the picture and everything that made it.
struct PromptPreviewShot: Codable, Identifiable, Equatable {
    var id = UUID()
    /// The PNG's name in the previews folder.
    var file: String
    var prompt: String
    var negative: String
    var model: PromptModel
    var seed: Int
    var date = Date()
    var costUSD: Double?
    /// The choices, to restore them ("Use These Choices") and to say what changed between two previews.
    var draft: PromptBuilderDraft
}

/// The Prompt Builder's preview: draws the written prompt on the first GPU server, on the model it was written for.
/// With a linked character on their own image family it draws through `CharacterArt` (their face, LoRAs and
/// fine-tune); otherwise through `PromptPreview`'s one-node graph.
extension AppModel {
    static let maxPromptPreviews = 60

    var promptPreviewFolder: URL { library.root.appendingPathComponent("prompt-previews") }
    private var promptPreviewIndex: URL { promptPreviewFolder.appendingPathComponent("index.json") }

    /// The saved previews, newest first.
    func promptPreviews() -> [PromptPreviewShot] {
        guard let data = try? Data(contentsOf: promptPreviewIndex),
              let shots = try? JSONDecoder().decode([PromptPreviewShot].self, from: data) else { return [] }
        return shots
    }

    func promptPreviewURL(_ shot: PromptPreviewShot) -> URL { promptPreviewFolder.appendingPathComponent(shot.file) }

    func savePromptPreviews(_ shots: [PromptPreviewShot]) throws {
        try FileManager.default.createDirectory(at: promptPreviewFolder, withIntermediateDirectories: true)
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try e.encode(shots).write(to: promptPreviewIndex, options: .atomic)
    }

    func deletePromptPreview(_ shot: PromptPreviewShot) {
        try? FileManager.default.removeItem(at: promptPreviewURL(shot))
        try? savePromptPreviews(promptPreviews().filter { $0.id != shot.id })
    }

    /// The linked character, when the preview draws through their own art model (the draft's model is one their
    /// family writes for).
    func previewCharacter(_ draft: PromptBuilderDraft) -> Companion? {
        guard let id = draft.characterID, let c = character(id) else { return nil }
        let own: Set<PromptModel> = switch c.artModel.family {
        case .sdxl: [.sdxl, .sdxlPony, .sdxlIllustrious]
        case .chroma: [.chroma]
        case .minimax: [.minimaxImage]
        }
        return own.contains(draft.model) ? c : nil
    }

    /// The server task a preview runs (for its price); nil for video models.
    func previewTask(_ draft: PromptBuilderDraft) -> String? {
        if let c = previewCharacter(draft) { return CharacterArt.task(c.artModel.family) }
        return PromptPreview.task(draft.model)
    }

    /// What one preview would cost on the first GPU server, and whether its model is loaded; nil when the server is
    /// offline or the price isn't known.
    func promptPreviewEstimate(_ draft: PromptBuilderDraft) -> (usd: Double, warm: Bool)? {
        guard let server = connections.gpuServers.first?.id, let task = previewTask(draft),
              case .gpuOnline(let info)? = connections.status[server], let price = info.task(task)?.pricePerSecond
        else { return nil }
        let timings = GPUTimings.load(gpuTimingsURL)
        let (w, h) = PromptPreview.size(draft.model, aspect: draft.aspect)
        let warm = timings.isWarm(task: task, idleSeconds: Double(info.idleSeconds ?? 120))
        var seconds = timings.seconds(task: task, megapixels: Double(w * h) / 1e6)
        if !warm { seconds += timings.loadSeconds(task: task) }
        return (seconds * price, warm)
    }

    /// Draws one preview of the draft (with its own seed) and keeps it with the newest `maxPromptPreviews`.
    func previewPrompt(_ draft: PromptBuilderDraft, progress: @escaping @MainActor (String) -> Void = { _ in })
        async throws -> PromptPreviewShot
    {
        if let problem = draft.problem { throw PromptPreviewError.refused(problem) }
        guard PromptPreview.nodeType(draft.model) != nil else { throw PromptPreviewError.video }
        guard let server = connections.gpuServers.first?.id else { throw PortraitError.noServer }
        let (w, h) = PromptPreview.size(draft.model, aspect: draft.aspect)
        let template: MovieTemplate
        var render: MovieArt.Render
        let written: WrittenPrompt
        if let c = previewCharacter(draft) {
            // CharacterArt.render adds MiniMax's face lead itself, so the prompt is written without one.
            var d = draft
            d.faceReference = false
            guard let wp = d.written else { throw PromptPreviewError.empty }
            written = wp
            try CharacterArt.check(wp.prompt)
            let context = artContext(c)
            var who = c
            who.artModel.variant = draft.variant.isEmpty ? nil : draft.variant
            render = CharacterArt.render(who, prompt: wp.prompt, seed: draft.seed, server: server,
                                         face: await faceReference(c), negative: wp.negative, context: context)
            render.values["width"] = .int(w)
            render.values["height"] = .int(h)
            template = CharacterArt.template(c.artModel.family, loras: c.loras.count)
        } else {
            guard let wp = draft.written else { throw PromptPreviewError.empty }
            written = wp
            try CharacterArt.check(wp.prompt)
            guard let t = PromptPreview.template(draft.model, negative: !wp.negative.isEmpty) else {
                throw PromptPreviewError.video
            }
            template = t
            render = MovieArt.Render(values: PromptPreview.values(wp, model: draft.model, seed: draft.seed,
                                                                 aspect: draft.aspect, server: server,
                                                                 fineTune: draft.variant),
                                     unusedPictures: [])
        }
        guard !written.prompt.isEmpty else { throw PromptPreviewError.empty }
        let (outputs, jobs) = try await runHeadlessGraph(template: template, render: render, title: "Prompt preview",
                                                         progress: progress)
        let cost = recordGPUJobs(jobs)
        guard let image = outputs.first(where: { $0.kind == .image })?.image else { throw MovieArtError.noImage }
        try FileManager.default.createDirectory(at: promptPreviewFolder, withIntermediateDirectories: true)
        let shot = PromptPreviewShot(file: "preview-\(UUID().uuidString.prefix(8).lowercased()).png",
                                     prompt: written.prompt, negative: draft.model.takesNegative ? written.negative : "",
                                     model: draft.model, seed: draft.seed, costUSD: cost, draft: draft)
        try MovieComposer.writePNG(image, to: promptPreviewURL(shot))
        var shots = [shot] + promptPreviews()
        for old in shots.dropFirst(Self.maxPromptPreviews) { try? FileManager.default.removeItem(at: promptPreviewURL(old)) }
        shots = Array(shots.prefix(Self.maxPromptPreviews))
        try savePromptPreviews(shots)
        return shot
    }
}

enum PromptPreviewError: Error, CustomStringConvertible {
    case refused(String), video, empty
    var description: String {
        switch self {
        case .refused(let why): why
        case .video: "The preview draws stills: pick an image model."
        case .empty: "Describe something first."
        }
    }
}
