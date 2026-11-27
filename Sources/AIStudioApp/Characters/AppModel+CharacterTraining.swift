import CharacterCore
import Foundation
import MovieCore
import NodesRemote
import NodesStandard

/// Character LoRAs (CharacterCore.CharacterTraining): training pictures drawn with the character's face on their own
/// image model, picked by the user, then a LoRA trained on the first GPU server and added to the Models library and to
/// the character. State lives in `<character>/training/training.json`.
extension AppModel {
    enum TrainingError: Error, CustomStringConvertible {
        case noServer, noTask(String), noPortrait, pick(Int), noEndpoint
        var description: String {
            switch self {
            case .noServer: "Add a GPU server in Servers first"
            case .noTask(let model): "LoRA training isn't available for \(model) yet: use SDXL or MiniMax"
            case .noPortrait: "Pick a profile photo first: training pictures use its face"
            case .pick(let n): "Pick \(CharacterTraining.minPicked)–\(CharacterTraining.maxPicked) pictures (\(n) picked)"
            case .noEndpoint: "The GPU server isn't reachable"
            }
        }
    }

    /// Rough GPU seconds of a training run, until the server's timings are learned.
    static func trainingSeconds(_ family: ArtModel.Family, pictures: Int) -> Double {
        switch family {
        case .sdxl: return Double(min(max(40 * pictures, 600), 1200)) * 1.0 + 120  // ≈ 1 s a step on an L40S
        case .minimax: return Double(min(max(50 * pictures, 800), 1600)) * 2.2 + Double(pictures) * 12 + 240
        case .chroma: return 0
        }
    }

    /// (drawing `pictures` training pictures, training on `picked`) in USD; nil when a price isn't known.
    func trainingEstimate(_ c: Companion, pictures: Int, picked: Int) -> (draw: Double?, train: Double?) {
        guard let server = connections.gpuServers.first?.id else { return (nil, nil) }
        let draw = photoEstimate(server: server, family: c.artModel.family).map { $0 * Double(pictures) }
        var train: Double?
        if let task = CharacterTraining.task(c.artModel.family), case .gpuOnline(let info)? = connections.status[server],
           let price = info.task(task)?.pricePerSecond {
            train = price * Self.trainingSeconds(c.artModel.family, pictures: picked)
        }
        return (draw, train)
    }

    /// The cost sheet shown before drawing `count` training pictures, or drawing one again (`redraw`).
    func trainingPicturesCost(_ c: Companion, count: Int, redraw: Bool = false) -> CostEstimate {
        let family = c.artModel.family
        let task = trainingServerInfo()?.task(CharacterArt.task(family))
        let usd = trainingEstimate(c, pictures: count, picked: CharacterTraining.minPicked).draw
        var notes = [redraw ? "The picture is drawn again with their profile photo's face and replaces this one."
                     : "Each picture is drawn with their profile photo's face; you pick the ones to train on afterwards."]
        if usd == nil { notes.append(Self.unknownPrice) }
        return CostEstimate(title: redraw ? "Redraw a Training Picture" : "Make \(count) Training Pictures",
                            lines: [.init(label: "Pictures (\(family.title))", count: count,
                                          gpu: task?.gpu ?? (family == .minimax ? "B200" : "L40S"),
                                          seconds: Self.seconds(usd, price: task?.pricePerSecond), usd: usd ?? 0)],
                            notes: notes)
    }

    /// The cost sheet shown before training a LoRA on `picked` pictures.
    func trainingLoRACost(_ c: Companion, picked: Int) -> CostEstimate {
        let family = c.artModel.family
        let task = CharacterTraining.task(family).flatMap { trainingServerInfo()?.task($0) }
        let seconds = Self.trainingSeconds(family, pictures: picked)
        // The server's GPUs for the training tasks (GPU_SD, GPU_H3_IMAGE), for Modal's list price until it says its own.
        let gpu = task?.gpu ?? (family == .minimax ? "B200" : "L40S")
        let usd = trainingEstimate(c, pictures: 0, picked: picked).train
            ?? MovieCost.price(gpu: gpu).map { $0 * seconds }
        var notes = ["Training takes about \(Int((seconds / 60).rounded())) min "
                     + "and can be cancelled; the LoRA then joins their photos and your Models library."]
        if usd == nil { notes.append(Self.unknownPrice) }
        return CostEstimate(title: "Train a LoRA",
                            lines: [.init(label: "LoRA training (\(family.title))", count: picked, gpu: gpu,
                                          seconds: seconds, usd: usd ?? 0)],
                            notes: notes)
    }

    private static let unknownPrice = "The GPU server hasn't said its price yet, so the cost isn't known."

    private static func seconds(_ usd: Double?, price: Double?) -> Double {
        guard let usd, let price, price > 0 else { return 0 }
        return usd / price
    }

    private func trainingServerInfo() -> GPUServerInfo? {
        guard let server = connections.gpuServers.first?.id, case .gpuOnline(let info)? = connections.status[server]
        else { return nil }
        return info
    }

    func trainingSet(_ id: UUID) -> TrainingSet { characterStore.training(id) }

    private func saveTraining(_ set: TrainingSet, _ id: UUID) {
        try? characterStore.saveTraining(set, for: id)
        trainingRevision += 1
    }

    func setTrainingPicked(_ id: UUID, file: String, picked: Bool) {
        var set = trainingSet(id)
        guard let i = set.pictures.firstIndex(where: { $0.file == file }) else { return }
        set.pictures[i].picked = picked
        saveTraining(set, id)
    }

    /// Draws `count` training pictures (added to any already there), with the character's face, on their model.
    func makeTrainingPictures(_ c: Companion, count: Int = CharacterTraining.pictureCount,
                              progress: @escaping @MainActor (String) -> Void) async throws {
        guard let server = connections.gpuServers.first?.id else { throw TrainingError.noServer }
        guard CharacterTraining.task(c.artModel.family) != nil else { throw TrainingError.noTask(c.artModel.family.title) }
        guard portraitURL(c) != nil else { throw TrainingError.noPortrait }
        var set = trainingSet(c.id)
        if set.family != c.artModel.family { set.pictures = [] }  // pictures of another model don't train this one
        set.family = c.artModel.family
        set.trigger = CharacterTraining.trigger(for: c)
        set.status = .drawing
        set.error = nil
        saveTraining(set, c.id)
        let folder = characterStore.trainingFolder(c.id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let start = set.pictures.count
        let requests = CharacterTraining.requests(count: start + count, gender: c.gender).suffix(count)
        defer {
            var s = trainingSet(c.id)
            if s.status == .drawing { s.status = s.pictures.isEmpty ? .empty : .ready }
            saveTraining(s, c.id)
        }
        for (k, request) in requests.enumerated() {
            try Task.checkCancellation()
            progress("Drawing training picture \(k + 1) of \(count)")
            let file = String(format: "train-%02d-", start + k + 1) + UUID().uuidString.prefix(6).lowercased() + ".png"
            let prompt = CharacterArt.photoPrompt(c, request: request, scene: nil, context: artContext(c))
            let cost = try await drawPhoto(c, prompt: prompt, server: server, to: folder.appendingPathComponent(file),
                                           title: "Training picture", faceStrength: CharacterTraining.faceStrength,
                                           progress: { _ in })
            var s = trainingSet(c.id)
            s.pictures.append(.init(file: file, request: request, picked: s.picked.count < CharacterTraining.maxPicked))
            s.spentUSD += cost ?? 0
            saveTraining(s, c.id)
        }
    }

    /// Draws one training picture again (same request, new seed).
    func redrawTrainingPicture(_ c: Companion, file: String, progress: @escaping @MainActor (String) -> Void) async throws {
        guard let server = connections.gpuServers.first?.id else { throw TrainingError.noServer }
        var set = trainingSet(c.id)
        guard let p = set.pictures.first(where: { $0.file == file }) else { return }
        let prompt = CharacterArt.photoPrompt(c, request: p.request, scene: nil, context: artContext(c))
        let cost = try await drawPhoto(c, prompt: prompt, server: server,
                                       to: characterStore.trainingFolder(c.id).appendingPathComponent(file),
                                       title: "Training picture", faceStrength: CharacterTraining.faceStrength,
                                       progress: progress)
        set = trainingSet(c.id)
        set.spentUSD += cost ?? 0
        saveTraining(set, c.id)
    }

    /// Trains a LoRA on the picked pictures; adds it to the Models library (trigger word, base model) and to the
    /// character (replacing their previous trained one). Returns its library id. `steps` 0: the server's default.
    @discardableResult
    func trainLoRA(_ c: Companion, steps: Int = 0, progress: @escaping @MainActor (String) -> Void) async throws -> String {
        guard let server = connections.gpuServers.first?.id else { throw TrainingError.noServer }
        var set = trainingSet(c.id)
        let family = set.family ?? c.artModel.family
        guard let task = CharacterTraining.task(family) else { throw TrainingError.noTask(family.title) }
        let picked = set.picked
        guard steps > 0 ? picked.count >= 4 : set.canTrain else { throw TrainingError.pick(picked.count) }
        guard let endpoint = RemoteEndpoints.resolve(server) else { throw TrainingError.noEndpoint }
        let trigger = set.trigger ?? CharacterTraining.trigger(for: c)
        let folder = characterStore.trainingFolder(c.id)
        var entries: [(name: String, data: Data)] = []
        for (i, p) in picked.enumerated() {
            entries.append((String(format: "%02d.png", i + 1), try Data(contentsOf: folder.appendingPathComponent(p.file))))
            let caption = CharacterTraining.caption(request: p.request, trigger: trigger, gender: c.gender)
            entries.append((String(format: "%02d.txt", i + 1), Data(caption.utf8)))
        }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("training-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let zip = tmp.appendingPathComponent("dataset.zip")
        try ZipArchive.write(entries, to: zip)

        set.status = .training
        set.error = nil
        saveTraining(set, c.id)
        do {
            let client = GPUJobClient(baseURL: endpoint.baseURL, token: endpoint.token)
            progress("Uploading \(picked.count) pictures")
            let name = try await client.uploadFile(zip, ext: "zip")
            var params: [String: Any] = ["seed": Int.random(in: 0..<Int(Int32.max)), "steps": steps]
            if family == .sdxl { params["model"] = c.artModel.variant ?? "base" }
            progress("Starting the training")
            let job = try await client.submit(task: task, params: params, inputs: ["dataset": name])
            let result = try await withTaskCancellationHandler {
                try await client.follow(job) { e in
                    let line: String? = switch e.kind {
                    case .progress where e.total > 0: "Training: step \(e.step) of \(e.total)"
                    case .nodeStatus where !e.message.isEmpty: e.message
                    default: nil
                    }
                    if let line { Task { @MainActor in progress(line) } }
                }
            } onCancel: {
                Task { await client.cancel(job.id) }
            }
            progress("Downloading the LoRA")
            let file = tmp.appendingPathComponent("\(trigger).safetensors")
            try await client.downloadResult(job.id, to: file)
            guard let library = store?.library else { throw TrainingError.noEndpoint }
            let id = try library.addFile(file, title: "\(c.name) (trained \(family.title))",
                                         family: family.libraryFamily, servers: [server])
            try? Library.update(id) {
                $0.triggerWords = [trigger]
                $0.baseModel = family == .sdxl ? "SDXL 1.0" : "MiniMax H3 Image"
            }
            library.reload()
            if var saved = character(c.id) {
                saved.loras.removeAll { $0.lora == set.loraID }
                if saved.loras.count < CharacterArt.maxLoRAs { saved.loras.append(CharacterLoRA(lora: id, strength: CharacterTraining.loraStrength)) }
                _ = saveCharacter(saved)
            }
            set = trainingSet(c.id)
            set.loraID = id
            set.status = .done
            set.spentUSD += result.costUSD ?? 0
            saveTraining(set, c.id)
            return id
        } catch {
            set = trainingSet(c.id)
            set.status = error is CancellationError ? .ready : .failed
            set.error = error is CancellationError ? nil : String(describing: error)
            saveTraining(set, c.id)
            throw error
        }
    }

    /// Runs `work` as the character's training task (one at a time), keeping its status line.
    func runTraining(_ id: UUID, _ work: @escaping @MainActor (@escaping @MainActor (String) -> Void) async throws -> Void) {
        guard trainingTasks[id] == nil else { return }
        trainingStatus[id] = "Starting…"
        trainingTasks[id] = Task { @MainActor in
            do {
                try await work { self.trainingStatus[id] = $0 }
                self.trainingStatus[id] = nil
            } catch is CancellationError {
                self.trainingStatus[id] = nil
            } catch {
                self.trainingStatus[id] = "Failed: \(error)"
            }
            self.trainingTasks[id] = nil
            self.trainingRevision += 1
        }
    }
}
