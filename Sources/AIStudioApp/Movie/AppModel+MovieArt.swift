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
        case "qwen3-tts": "speech"
        case "stable-audio-3": "sound"
        case "ace-step-xl": "music"
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
            var waiting: [String] = []
            // Two ways to run it: one GPU per kind of job, or as many at once as are free (each extra one loads the
            // model and idles too). Lines and wall time for each.
            var single: [CostEstimate.Line] = [], wide: [CostEstimate.Line] = []
            var singleWall = 0.0, wideWall = 0.0, widest = 1
            for w in work {
                guard let task = info.task(w.task), task.installed else {
                    doc.error = "\(server.name) can't run \(w.task) yet: run its download step (see docs/remote-server.md: "
                        + "`--models \(Self.downloadGroup(w.task))`)."
                    return
                }
                let price = task.pricePerSecond ?? MovieCost.price(gpu: task.gpu) ?? 0
                // Other work may hold some of this model's GPUs: only the free ones start (and load) for these jobs.
                let gpus = MovieCost.gpus(jobs: w.jobs, parallel: info.parallelJobs ?? 1, busy: task.activeJobs ?? 0,
                                          limit: MovieRenderer.maxParallel)
                if gpus.waits { waiting.append(w.label.lowercased()) }
                let warm = gpus.waits || timings.isWarm(task: w.task, idleSeconds: idle)
                let perJob = timings.seconds(task: w.task, megapixels: w.megapixels)
                let load = timings.loadSeconds(task: w.task)
                for (k, into) in [(1, 0), (gpus.count, 1)] {
                    let line = MovieCost.images(label: w.label, task: w.task, jobs: w.jobs, megapixels: w.megapixels,
                                                gpu: task.gpu, usdPerSecond: price, timings: timings, warm: warm,
                                                idleSeconds: idle, containers: k)
                    let wall = MovieCost.wallSeconds(jobs: w.jobs, containers: k, perJob: perJob, load: load, warm: warm)
                    if into == 0 { single.append(line); singleWall = max(singleWall, wall) }
                    else { wide.append(line); wideWall = max(wideWall, wall) }
                }
                widest = max(widest, gpus.count)
            }
            let learned = work.allSatisfy { w in timings.jobs.keys.contains { $0.hasPrefix(w.task + "@") } }
            var notes = [
                "Each kind of job runs on its own GPU worker, which loads its model first when it's cold (counted above) "
                    + "and stays up \(Int(idle)) s after the last job (counted too).",
                learned ? "Job times are what these jobs took before; billing is Modal's."
                    : "Job times are a first guess until a job has run; billing is Modal's.",
            ]
            if !waiting.isEmpty {
                notes.append("Every GPU for the \(waiting.joined(separator: " and ")) is busy with other work right "
                    + "now: they'll wait their turn on one whose model is loaded.")
            }
            let plans = widest > 1
                ? [CostEstimate.Plan(parallel: 1, lines: single, wallSeconds: singleWall),
                   CostEstimate.Plan(parallel: widest, lines: wide, wallSeconds: wideWall)]
                : []
            if !plans.isEmpty {
                notes.append("Several GPUs at once finish sooner, but each extra one loads the model and stays up "
                    + "\(Int(idle)) s too (counted in that choice).")
            }
            if work.contains(where: { $0.task == "qwen-image-edit" }) {
                notes.append("Keyframes are made from the characters' chosen sheets, so faces and clothes stay the same.")
            }
            notes += extraNotes
            let estimate = plans.isEmpty
                ? CostEstimate(title: title, lines: single, notes: notes)
                : CostEstimate(title: title, plans: plans,
                               chosen: MovieCost.prefersParallel(single: plans[0], parallel: plans[1]) ? widest : 1,
                               notes: notes)
            doc.pendingGate = CostGate(estimate: estimate, run: run, choose: { doc.parallelLimit = $0 })
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
        runImageJob(doc, title: "Making reference sheets", count: items.count, task: { _ in "qwen-image" }) {
            [weak self] doc, i, server in
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
        // A shot that continues the one before waits for that shot's new keyframe; the rest run at once.
        let tasks = ids.map { id in
            doc.movie.shots.first { $0.id == id }.map { usesPictures($0, in: doc.movie) } == true ? "qwen-image-edit" : "qwen-image"
        }
        runImageJob(doc, title: "Making keyframes", count: ids.count, chains: MovieShots.chains(ids, in: doc.movie),
                    task: { tasks[$0] }) { [weak self] doc, i, server in
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

    /// How many renders a movie sends to `server` at once: what the server runs in parallel (on Modal, GPU containers
    /// per model), at most `MovieRenderer.maxParallel`; one when it hasn't said.
    func movieParallelJobs(_ server: String) -> Int {
        guard case .gpuOnline(let info)? = connections.status[server] else { return 1 }
        return min(max(info.parallelJobs ?? 1, 1), MovieRenderer.maxParallel)
    }

    /// Runs `count` renders as the movie's job, as many at once as the server runs in parallel. `chains` groups the
    /// renders that must run in order (a shot that continues the one before starts from its new keyframe or clip);
    /// nil: each on its own. Each saves the movie, so a stop or a failure keeps what's done; after a failure no more
    /// start, and the ones running finish.
    ///
    /// `task(i)` names the server task render `i` runs: before starting another render while some of this job's are
    /// running, it asks the server whether that task has a free GPU (other work may hold them), and else waits for
    /// one of its own to finish. One always runs, so a busy server only slows the job down.
    func runImageJob(_ doc: MovieDocument, title: String, count: Int, chains: [[Int]]? = nil,
                     task: ((Int) -> String)? = nil,
                     _ step: @escaping @MainActor (MovieDocument, Int, String) async throws -> Void) {
        guard doc.job == nil, count > 0 else { return }
        guard let server = movieGPUServer(doc) else {
            doc.error = "No GPU server: add one in the Servers panel (+ › GPU server)."
            return
        }
        let chains = chains ?? (0..<count).map { [$0] }
        // What the cost sheet's choice allows (one GPU, or several at once), at most what the server runs at once.
        let limit = min(doc.parallelLimit ?? .max, movieParallelJobs(server.id))
        doc.parallelLimit = nil
        doc.job = MovieJob(title: title, detail: "Starting…", progress: 0)
        let task = Task { @MainActor [weak self] in
            let done = DoneCount()
            var failure: Error?
            await withTaskGroup(of: Error?.self) { group in
                var next = 0, running = 0
                while true {
                    while running < limit, next < chains.count, failure == nil, !Task.isCancelled {
                        if running > 0, let first = chains[next].first, let name = task?(first),
                           await !(self?.serverHasRoom(server.id, task: name) ?? true) {
                            break  // its GPUs are all busy: wait for one of ours to finish, then ask again
                        }
                        let chain = chains[next], lane = next
                        next += 1
                        running += 1
                        group.addTask { @MainActor in
                            await MovieJob.$lane.withValue(lane) {
                                defer { doc.job?.endLane(lane) }
                                do {
                                    for i in chain {
                                        try Task.checkCancellation()
                                        try await step(doc, i, server.id)
                                        done.value += 1
                                        doc.job?.progress = Double(done.value) / Double(count)
                                        self?.saveMovie(doc)
                                    }
                                    return nil
                                } catch {
                                    return error
                                }
                            }
                        }
                    }
                    guard let result = await group.next() else { break }
                    running -= 1
                    if let error = result, !(error is CancellationError), failure == nil { failure = error }
                }
            }
            if let failure, !Task.isCancelled { doc.error = "\(title) failed: \(failure)" }
            doc.job = nil
            self?.saveMovie(doc)
        }
        doc.job?.task = task
    }

    /// Whether the server would start one more `task` job now: it runs fewer of that model's jobs than it can at
    /// once (counting everyone's). Unknown: yes (the job just waits its turn if not).
    func serverHasRoom(_ server: String, task: String) async -> Bool {
        guard let e = RemoteEndpoints.resolve(server),
              let info = try? await GPUJobClient(baseURL: e.baseURL, token: e.token).info() else { return true }
        return info.freeSlots(task) > 0
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
        // Only this graph's GPU jobs (others may run at the same time), and its events to its own lane's line.
        let jobs = JobRecords()
        let scope = UUID()
        let token = RemoteJobLog.observe(scope: scope) { jobs.append($0) }
        defer { RemoteJobLog.remove(token) }
        let lane = MovieJob.lane
        // (Read first: assigning through `doc.job?` while reading it in the same statement is an exclusive-access
        // conflict.)
        let what = doc.job?.detail ?? ""
        let outputs = try await RemoteJobLog.$scope.withValue(scope) {
            try await movieRenderer.run(graph, on: localBackend, title: title) { event in
                MovieJob.$lane.withValue(lane) {
                    if case .progress(_, let step, let total, _) = event, total > 0 {
                        doc.job?.detail = what + " · step \(step)/\(total)"
                    } else if case .status(_, let text?) = event, !text.isEmpty, !text.hasPrefix("GPU ") {
                        doc.job?.detail = what + " · \(text)"
                    }
                }
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

/// Renders finished so far in a job whose renders run at once.
@MainActor private final class DoneCount {
    var value = 0
}

/// Finished Cloud GPU jobs heard while one render runs.
final class JobRecords: @unchecked Sendable {
    private let lock = NSLock()
    private var records: [RemoteJobRecord] = []
    func append(_ r: RemoteJobRecord) { lock.withLock { records.append(r) } }
    var all: [RemoteJobRecord] { lock.withLock { records } }
}
