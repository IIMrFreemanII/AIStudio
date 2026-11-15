import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO
import MovieCore
import NodesRemote

/// The Cast and Keyframes stages: reference sheets (Qwen-Image) and keyframes (Qwen-Image-Edit from the sheets),
/// rendered through the movie's templates on a GPU server, each batch behind its cost gate.
extension AppModel {
    var gpuTimingsURL: URL { library.root.appendingPathComponent("gpu-timings.json") }

    /// The GPU server a movie renders on: the one its writing model runs on, else the first GPU server.
    func movieGPUServer(_ doc: MovieDocument) -> RemoteEndpoints.Endpoint? {
        if let id = doc.movie.llm?.serverID, connections.connections.contains(where: { $0.id == id && $0.kind == .gpu }),
           let e = RemoteEndpoints.resolve(id) {
            return e
        }
        return RemoteEndpoints.resolve("")
    }

    /// One kind of GPU job in a batch, for the cost sheet.
    struct ImageWork {
        var label: String
        var task: String
        var jobs: Int
        /// A job's size: megapixels for a picture, megapixels × frames for a clip (`MovieShots.work`).
        var megapixels: Double
    }

    /// The download step that installs a task's models on the server.
    private static func downloadGroup(_ task: String) -> String {
        switch task {
        case "qwen-image-edit": "qwen-edit"
        case "qwen-image": "qwen"
        case "wan22-i2v", "wan22-t2v": "wan22"
        default: task
        }
    }

    /// Shows the cost sheet for a batch of image jobs, after asking the server for its tasks (installed? which GPU,
    /// what price?) and its idle window.
    func gateImages(_ doc: MovieDocument, title: String, work: [ImageWork], extraNotes: [String] = [],
                    run: @escaping @MainActor () -> Void) {
        doc.error = nil
        let work = work.filter { $0.jobs > 0 }
        guard doc.job == nil, !work.isEmpty else { return }
        guard let server = movieGPUServer(doc) else {
            doc.error = "No GPU server: add one in the Servers panel (+ › GPU server)."
            return
        }
        doc.job = MovieJob(title: "Checking the cost…", detail: "Asking \(server.name) about its models")
        Task { @MainActor [weak self] in
            defer { if doc.job?.title == "Checking the cost…" { doc.job = nil } }
            guard let self else { return }
            await self.connections.test(server.id)
            guard case .gpuOnline(let info)? = self.connections.status[server.id] else {
                doc.error = "\(server.name) isn't reachable right now."
                return
            }
            let timings = GPUTimings.load(self.gpuTimingsURL)
            let idle = Double(info.idleSeconds ?? 120)
            var lines: [CostEstimate.Line] = []
            for w in work {
                guard let task = info.task(w.task), task.installed else {
                    doc.error = "\(server.name) can't run \(w.task) yet: run its download step (see docs/remote-server.md: "
                        + "`--models \(Self.downloadGroup(w.task))`)."
                    return
                }
                let price = task.pricePerSecond ?? MovieCost.price(gpu: task.gpu) ?? 0
                lines.append(MovieCost.images(label: w.label, task: w.task, jobs: w.jobs, megapixels: w.megapixels,
                                              gpu: task.gpu, usdPerSecond: price, timings: timings,
                                              warm: timings.isWarm(task: w.task, idleSeconds: idle),
                                              idleSeconds: idle))
            }
            let learned = work.allSatisfy { w in timings.jobs.keys.contains { $0.hasPrefix(w.task + "@") } }
            var notes = [
                "Each kind of job runs on its own GPU worker, which loads its model first when it's cold (counted above) "
                    + "and stays up \(Int(idle)) s after the last job (counted too).",
                learned ? "Job times are what these jobs took before; billing is Modal's."
                    : "Job times are a first guess until a job has run; billing is Modal's.",
            ]
            if work.contains(where: { $0.task == "qwen-image-edit" }) {
                notes.append("Keyframes are made from the characters' chosen sheets, so faces and clothes stay the same.")
            }
            notes += extraNotes
            doc.pendingGate = CostGate(estimate: CostEstimate(title: title, lines: lines, notes: notes), run: run)
        }
    }

    // MARK: cast

    /// Cast › Make Sheets: one take per character (`characters` nil: those without a sheet yet, else all of them).
    func requestSheets(_ doc: MovieDocument, characters ids: [String]? = nil, takes: Int = 1) {
        let missing = doc.movie.characters.filter { $0.sheets.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? doc.movie.characters.map(\.id) : missing)
        guard !ids.isEmpty else { return }
        let size = MovieArt.sheetSize
        let jobs = ids.count * takes
        gateImages(doc, title: jobs == 1 ? "Make a Reference Sheet" : "Make Reference Sheets",
                   work: [ImageWork(label: jobs == 1 ? "1 reference sheet" : "\(jobs) reference sheets",
                                    task: "qwen-image", jobs: jobs,
                                    megapixels: Double(size.width * size.height) / 1e6)]) { [weak self] in
            self?.makeSheets(doc, characters: ids, takes: takes)
        }
    }

    private func makeSheets(_ doc: MovieDocument, characters ids: [String], takes: Int) {
        let items = ids.flatMap { id in Array(repeating: id, count: takes) }
        runImageJob(doc, title: "Making reference sheets", count: items.count) { [weak self] doc, i, server in
            guard let self, let c = doc.movie.character(items[i]) else { return }
            doc.job?.detail = "\(c.name) (\(i + 1) of \(items.count))"
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let render = MovieArt.sheetRender(c, movie: doc.movie, seed: seed, server: server)
            let template = self.movieTemplate(MovieArt.TemplateName.sheet, for: doc) {
                MovieArt.textToImageTemplate(name: MovieArt.TemplateName.sheet)
            }
            let file = "cast/\(c.id)-\(UUID().uuidString.prefix(8).lowercased()).png"
            let cost = try await self.renderStill(doc, template: template, render: render, to: file,
                                                  title: "\(doc.movie.title): \(c.name)")
            guard let j = doc.movie.characters.firstIndex(where: { $0.id == c.id }) else { return }
            let take = Take(file: file, seed: seed, costUSD: cost)
            doc.movie.characters[j].sheets.append(take)
            if doc.movie.characters[j].selectedSheet == nil { doc.movie.characters[j].selectedSheet = take.id }
        }
    }

    // MARK: keyframes

    /// Keyframes › Make Keyframes: one take per shot (`shots` nil: those without a keyframe yet, else all). Shots
    /// with characters (or continuing the one before) are edits of the sheets; the rest plain pictures.
    func requestKeyframes(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let missing = doc.movie.shots.filter { $0.keyframes.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? doc.movie.shots.map(\.id) : missing)
        let shots = ids.compactMap { id in doc.movie.shots.first { $0.id == id } }
        guard !shots.isEmpty else { return }
        let lacking = Set(shots.flatMap { MovieArt.missingSheets(for: $0, in: doc.movie) })
        guard lacking.isEmpty else {
            doc.error = "Choose a reference sheet for \(lacking.sorted().joined(separator: ", ")) first (Cast)."
            return
        }
        let edits = shots.filter { usesPictures($0, in: doc.movie) }.count
        let size = doc.movie.aspect.keyframeSize
        let mp = Double(size.width * size.height) / 1e6
        gateImages(doc, title: shots.count == 1 ? "Make a Keyframe" : "Make Keyframes", work: [
            ImageWork(label: "Keyframes with the cast", task: "qwen-image-edit", jobs: edits, megapixels: mp),
            ImageWork(label: "Keyframes without people", task: "qwen-image", jobs: shots.count - edits, megapixels: mp),
        ]) { [weak self] in
            self?.makeKeyframes(doc, shots: shots.map(\.id))
        }
    }

    /// Whether a shot's keyframe is an edit of pictures: it has characters, or continues the shot before in the
    /// same scene (whose keyframe it starts from).
    private func usesPictures(_ shot: Shot, in movie: MovieProject) -> Bool {
        !shot.characters.compactMap(movie.character).isEmpty
            || (shot.continuesPrevious && movie.shot(before: shot)?.scene == shot.scene)
    }

    private func makeKeyframes(_ doc: MovieDocument, shots ids: [UUID]) {
        runImageJob(doc, title: "Making keyframes", count: ids.count) { [weak self] doc, i, server in
            // In film order, so a shot that continues the one before sees that shot's new keyframe.
            guard let self, let shot = doc.movie.shots.first(where: { $0.id == ids[i] }) else { return }
            let number = (doc.movie.shots.firstIndex { $0.id == shot.id } ?? 0) + 1
            doc.job?.detail = "Shot \(number) (\(i + 1) of \(ids.count))"
            let seed = Int.random(in: 0..<Int(UInt32.max))
            let folder = self.movieStore.folder(doc.id)
            let (render, refs) = MovieArt.keyframeRender(shot, movie: doc.movie, folder: folder, seed: seed,
                                                         server: server)
            let template = refs.isEmpty
                ? self.movieTemplate(MovieArt.TemplateName.keyframe, for: doc) {
                    MovieArt.textToImageTemplate(name: MovieArt.TemplateName.keyframe)
                }
                : self.movieTemplate(MovieArt.TemplateName.keyframeEdit, for: doc) { MovieArt.editTemplate() }
            let file = "keyframes/shot-\(shot.id.uuidString.prefix(8).lowercased())-\(UUID().uuidString.prefix(6).lowercased()).png"
            let cost = try await self.renderStill(doc, template: template, render: render, to: file,
                                                  title: "\(doc.movie.title): shot \(number)")
            guard let j = doc.movie.shots.firstIndex(where: { $0.id == shot.id }) else { return }
            let take = Take(file: file, seed: seed, costUSD: cost)
            doc.movie.shots[j].keyframes.append(take)
            if doc.movie.shots[j].selectedKeyframe == nil { doc.movie.shots[j].selectedKeyframe = take.id }
        }
    }

    // MARK: running

    /// Runs `count` renders one after another as the movie's job; each saves the movie, so a stop or a failure
    /// keeps what's done.
    func runImageJob(_ doc: MovieDocument, title: String, count: Int,
                             _ step: @escaping @MainActor (MovieDocument, Int, String) async throws -> Void) {
        guard doc.job == nil, count > 0 else { return }
        guard let server = movieGPUServer(doc) else {
            doc.error = "No GPU server: add one in the Servers panel (+ › GPU server)."
            return
        }
        doc.job = MovieJob(title: title, detail: "Starting…", progress: 0)
        let task = Task { @MainActor [weak self] in
            do {
                for i in 0..<count {
                    try Task.checkCancellation()
                    doc.job?.progress = Double(i) / Double(count)
                    try await step(doc, i, server.id)
                    self?.saveMovie(doc)
                }
            } catch is CancellationError {
            } catch {
                if !Task.isCancelled { doc.error = "\(title) failed: \(error)" }
            }
            doc.job = nil
            self?.saveMovie(doc)
        }
        doc.job?.task = task
    }

    /// Renders one still through a template and writes it into the movie's folder; returns what its GPU job cost
    /// (also added to the movie's spend, and the job's time to the learned timings).
    private func renderStill(_ doc: MovieDocument, template: MovieTemplate, render: MovieArt.Render, to file: String,
                             title: String) async throws -> Double? {
        // What it's making ("Mara (1 of 2)"), then the node's step or status.
        let (outputs, jobs) = try await runMovieGraph(doc, template: template, render: render, title: title)
        guard let image = outputs.first(where: { $0.kind == .image })?.image else {
            throw MovieArtError.noImage
        }
        try MovieComposer.writePNG(image, to: movieStore.url(doc.id, relative: file))
        return recordJobs(jobs, doc: doc)
    }

    /// Runs a template, reporting what it's doing in the job bar, and returns its outputs and the Cloud GPU jobs
    /// it ran.
    func runMovieGraph(_ doc: MovieDocument, template: MovieTemplate, render: MovieArt.Render, title: String)
        async throws -> (outputs: [EngineOutput], jobs: [RemoteJobRecord])
    {
        let graph = try template.bind(render.values, optional: Set(render.values.keys), remove: render.unusedPictures)
        let jobs = JobRecords()
        let token = RemoteJobLog.observe { jobs.append($0) }
        defer { RemoteJobLog.remove(token) }
        // (Read first: assigning through `doc.job?` while reading it in the same statement is an exclusive-access
        // conflict.)
        let what = doc.job?.detail ?? ""
        let outputs = try await movieRenderer.run(graph, on: localBackend, title: title) { event in
            if case .progress(_, let step, let total, _) = event, total > 0 {
                doc.job?.detail = what + " · step \(step)/\(total)"
            } else if case .status(_, let text?) = event, !text.isEmpty, !text.hasPrefix("GPU ") {
                doc.job?.detail = what + " · \(text)"
            }
        }
        return (outputs, jobs.all)
    }

    /// Adds the jobs' cost to the movie's spend and their times to the learned timings; returns the cost. `work`
    /// is a job's size for the timings when it isn't its picture's megapixels (a clip's megapixels × frames).
    func recordJobs(_ records: [RemoteJobRecord], doc: MovieDocument, work: Double? = nil) -> Double? {
        var cost: Double?
        var timings = GPUTimings.load(gpuTimingsURL)
        for r in records {
            if let c = r.costUSD { cost = (cost ?? 0) + c }
            let mp = work ?? Double((r.width ?? 0) * (r.height ?? 0)) / 1e6
            timings.record(task: r.task, megapixels: mp > 0 ? mp : 1, gpuSeconds: r.gpuSeconds, loadSeconds: r.loadSeconds)
        }
        try? timings.save(gpuTimingsURL)
        if let cost { doc.movie.costSpent += cost }
        return cost
    }

    // MARK: choosing takes

    func selectSheet(_ doc: MovieDocument, character id: String, take: UUID) {
        guard let j = doc.movie.characters.firstIndex(where: { $0.id == id }),
              doc.movie.characters[j].selectedSheet != take else { return }
        doc.movie.characters[j].selectedSheet = take
        doc.movie.reopen(.cast)  // keyframes made from the old sheet no longer match
        saveMovie(doc)
    }

    func selectKeyframe(_ doc: MovieDocument, shot id: UUID, take: UUID) {
        guard let j = doc.movie.shots.firstIndex(where: { $0.id == id }), doc.movie.shots[j].selectedKeyframe != take
        else { return }
        doc.movie.shots[j].selectedKeyframe = take
        doc.movie.reopen(.keyframes)
        saveMovie(doc)
    }

    /// Takes the take out of the movie and its file to the Trash.
    func deleteTake(_ doc: MovieDocument, character id: String? = nil, shot: UUID? = nil, take: UUID) {
        var file: String?
        if let id, let j = doc.movie.characters.firstIndex(where: { $0.id == id }) {
            file = doc.movie.characters[j].sheets.first { $0.id == take }?.file
            doc.movie.characters[j].sheets.removeAll { $0.id == take }
            if doc.movie.characters[j].selectedSheet == take {
                doc.movie.characters[j].selectedSheet = doc.movie.characters[j].sheets.last?.id
                doc.movie.reopen(.cast)
            }
        }
        var poster: String?
        var lastFrame: String?
        if let shot, let j = doc.movie.shots.firstIndex(where: { $0.id == shot }) {
            if let k = doc.movie.shots[j].keyframes.first(where: { $0.id == take }) {
                file = k.file
                doc.movie.shots[j].keyframes.removeAll { $0.id == take }
                if doc.movie.shots[j].selectedKeyframe == take {
                    doc.movie.shots[j].selectedKeyframe = doc.movie.shots[j].keyframes.last?.id
                    doc.movie.reopen(.keyframes)
                }
            } else if let v = doc.movie.shots[j].videos.first(where: { $0.id == take }) {
                (file, poster) = (v.file, v.poster)
                lastFrame = MovieShots.lastFrameFile(of: v.file)
                doc.movie.shots[j].videos.removeAll { $0.id == take }
                if doc.movie.shots[j].selectedVideo == take {
                    doc.movie.shots[j].selectedVideo = doc.movie.shots[j].videos.last?.id
                    doc.movie.reopen(.video)
                }
            }
        }
        for f in [file, poster, lastFrame].compactMap({ $0 }) {
            let url = movieStore.url(doc.id, relative: f)
            if Self.isAutomatedRun { try? FileManager.default.removeItem(at: url) }
            else { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        }
        saveMovie(doc)
    }
}

enum MovieArtError: Error, CustomStringConvertible {
    case noImage
    var description: String { "the template gave no picture (it needs a Preview Image or Save Image at the end)" }
}

/// Finished Cloud GPU jobs heard while one render runs.
final class JobRecords: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [RemoteJobRecord] = []
    func append(_ r: RemoteJobRecord) { lock.withLock { records.append(r) } }
    var all: [RemoteJobRecord] { lock.withLock { records } }
}
