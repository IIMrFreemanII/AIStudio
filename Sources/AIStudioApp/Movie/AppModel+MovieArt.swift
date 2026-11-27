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
        /// The task's fast mode (the movie's Fast switch for Qwen pictures), which is timed on its own.
        var fast = false
        /// What the learned timings call these jobs (`GPUTimings.timingTask`).
        var timing: String { GPUTimings.timingTask(task, fast: fast) }
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
        let title = doc.stage.isPerSequence ? title + (doc.scopeTitle.map { " · " + $0 } ?? "") : title
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
            var fallbacks: [String: [Double]] = [:], fallbackFrom: [String: String] = [:]  // extra USD: one GPU, several
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
                let perJob = timings.seconds(task: w.timing, megapixels: w.megapixels)
                let load = timings.loadSeconds(task: w.timing)
                for (k, into) in [(1, 0), (gpus.count, 1)] {
                    let line = MovieCost.images(label: w.label, task: w.timing, jobs: w.jobs, megapixels: w.megapixels,
                                                gpu: task.gpu, usdPerSecond: price, timings: timings, warm: warm,
                                                idleSeconds: idle, containers: k)
                    let wall = MovieCost.wallSeconds(jobs: w.jobs, containers: k, perJob: perJob, load: load, warm: warm)
                    if into == 0 { single.append(line); singleWall = max(singleWall, wall) }
                    else { wide.append(line); wideWall = max(wideWall, wall) }
                    // The same seconds on the dearer GPU Modal may give instead when none of this kind is free.
                    if let fb = task.fallbackPricePerSecond, let gpu = task.fallbackGpu, price > 0, fb > price {
                        fallbacks[gpu, default: [0, 0]][into] += line.usd * (fb / price - 1)
                        fallbackFrom[gpu] = task.gpu
                    }
                }
                widest = max(widest, gpus.count)
            }
            let learned = work.allSatisfy { w in timings.jobs.keys.contains { $0.hasPrefix(w.timing + "@") } }
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
            for (gpu, extra) in fallbacks.sorted(by: { $0.key < $1.key }) {
                let base = single.reduce(0) { $0 + $1.usd }
                notes.append("When no \(fallbackFrom[gpu] ?? "GPU") is free, Modal gives a \(gpu) instead, which costs more: "
                    + "up to \(CostEstimate.format(base + extra[0])) for one GPU each.")
            }
            notes.append("Prices count the GPU and its container's memory and CPU, as Modal bills them.")
            if work.contains(where: { $0.task == "qwen-image-edit" }) {
                notes.append("Keyframes are made from the characters' chosen sheets, so faces and clothes stay the same.")
            }
            if work.contains(where: { $0.fast && GPUTimings.modeTimed.contains($0.task) }) {
                notes.append(Self.fastPicturesNote)
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

    /// The cost sheet's note while a movie's Fast switch is on.
    static let fastPicturesNote = "Fast mode is on: Qwen's Lightning LoRAs, 8 steps without guidance (≈ 6× cheaper for "
        + "sheets and plates, ≈ 10× for keyframes with the cast). It ignores the negative prompt, so text or extra "
        + "people can appear on sheets and people on plates; turn Fast off for the full models."

    // MARK: cast

    /// What the Cast page makes: a character's reference sheet or a place's plate.
    enum CastItem: Hashable {
        case character(String)
        case location(String)
    }

    /// Cast › Make Sheets: one take per character seen on screen and one plate per place (both nil: those without
    /// one yet, else all of them; one given: only those).
    func requestSheets(_ doc: MovieDocument, characters ids: [String]? = nil, locations places: [String]? = nil,
                       takes: Int = 1) {
        let seen = doc.movie.characters.filter { !$0.isNarrator }
        var chars = ids ?? []
        var locs = places ?? []
        if ids == nil && places == nil {
            let missingChars = seen.filter { $0.sheets.isEmpty }.map(\.id)
            let missingPlates = doc.movie.locations.filter { $0.plates.isEmpty }.map(\.id)
            if missingChars.isEmpty && missingPlates.isEmpty {
                chars = seen.map(\.id)
                locs = doc.movie.locations.map(\.id)
            } else {
                chars = missingChars
                locs = missingPlates
            }
        }
        let items = (chars.map(CastItem.character) + locs.map(CastItem.location)).flatMap { Array(repeating: $0, count: takes) }
        guard !items.isEmpty else { return }
        let size = MovieArt.sheetSize, plate = doc.movie.aspect.keyframeSize
        let sheets = chars.count * takes, plates = locs.count * takes
        gateImages(doc, title: plates == 0 ? (sheets == 1 ? "Make a Reference Sheet" : "Make Reference Sheets")
                       : sheets == 0 ? (plates == 1 ? "Make a Place's Plate" : "Make Plates of the Places")
                       : "Make Reference Sheets and Plates",
                   work: [ImageWork(label: sheets == 1 ? "1 reference sheet" : "\(sheets) reference sheets",
                                    task: "qwen-image", jobs: sheets,
                                    megapixels: Double(size.width * size.height) / 1e6, fast: doc.movie.fastPictures),
                          ImageWork(label: plates == 1 ? "1 place's plate" : "\(plates) plates of places",
                                    task: "qwen-image", jobs: plates,
                                    megapixels: Double(plate.width * plate.height) / 1e6, fast: doc.movie.fastPictures)],
                   extraNotes: plates > 0 ? ["A plate is the place itself, empty: every keyframe there is made from it, "
                                             + "so the place stays the same from shot to shot."] : []) { [weak self] in
            self?.makeSheets(doc, items: items)
        }
    }

    private func makeSheets(_ doc: MovieDocument, items: [CastItem]) {
        runImageJob(doc, title: "Making the cast and places", count: items.count, task: { _ in "qwen-image" }) {
            [weak self] doc, i, server in
            guard let self else { return }
            let seed = Int.random(in: 0..<Int(UInt32.max))
            switch items[i] {
            case .character(let id):
                guard let c = doc.movie.character(id) else { return }
                doc.job?.detail = "\(c.name) (\(i + 1) of \(items.count))"
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
            case .location(let id):
                guard let l = doc.movie.locations.first(where: { $0.id == id }) else { return }
                doc.job?.detail = "\(l.name) (\(i + 1) of \(items.count))"
                let render = MovieArt.plateRender(l, movie: doc.movie, seed: seed, server: server)
                let template = self.movieTemplate(MovieArt.TemplateName.plate, for: doc) {
                    MovieArt.textToImageTemplate(name: MovieArt.TemplateName.plate)
                }
                let file = "places/\(l.id)-\(UUID().uuidString.prefix(8).lowercased()).png"
                let cost = try await self.renderStill(doc, template: template, render: render, to: file,
                                                      title: "\(doc.movie.title): \(l.name)")
                guard let j = doc.movie.locations.firstIndex(where: { $0.id == l.id }) else { return }
                let take = Take(file: file, seed: seed, costUSD: cost)
                doc.movie.locations[j].plates.append(take)
                if doc.movie.locations[j].selectedPlate == nil { doc.movie.locations[j].selectedPlate = take.id }
            }
        }
    }

    // MARK: keyframes

    /// Keyframes › Make Keyframes: one take per shot (`shots` nil: those without a keyframe yet, else all). Shots
    /// with characters (or continuing the one before) are edits of the sheets; the rest plain pictures.
    func requestKeyframes(_ doc: MovieDocument, shots ids: [UUID]? = nil) {
        let scoped = doc.scopeShots
        let missing = scoped.filter { $0.keyframes.isEmpty }.map(\.id)
        let ids = ids ?? (missing.isEmpty ? scoped.map(\.id) : missing)
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
            ImageWork(label: "Keyframes with the cast", task: "qwen-image-edit", jobs: edits, megapixels: mp,
                      fast: doc.movie.fastPictures),
            ImageWork(label: "Keyframes without people", task: "qwen-image", jobs: shots.count - edits, megapixels: mp,
                      fast: doc.movie.fastPictures),
        ]) { [weak self] in
            self?.makeKeyframes(doc, shots: shots.map(\.id))
        }
    }

    /// Whether a shot's keyframe is an edit of pictures: it has characters, or continues the shot before in the
    /// same scene (whose keyframe it starts from).
    private func usesPictures(_ shot: Shot, in movie: MovieProject) -> Bool {
        !shot.characters.compactMap(movie.character).filter { !$0.isNarrator }.isEmpty
            || (shot.continuesPrevious && movie.shot(before: shot)?.scene == shot.scene)
            || MovieArt.location(of: shot, in: movie)?.selectedPlateFile != nil
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
        // What it's making ("Mara (1 of 2)"), then the node's step or status. The movie's Fast switch sets the Qwen
        // nodes' mode, in its edited templates too.
        let template = MovieArt.fastMode(doc.movie.fastPictures, template)
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
        let lane = MovieJob.lane
        // (Read first: assigning through `doc.job?` while reading it in the same statement is an exclusive-access
        // conflict.)
        let what = doc.job?.detail ?? ""
        return try await runHeadlessGraph(template: template, render: render, title: title) { text in
            MovieJob.$lane.withValue(lane) { doc.job?.detail = what + " · " + text }
        }
    }

    /// Adds the jobs' cost to the movie's spend and their times to the learned timings; returns the cost. `work`
    /// is a job's size for the timings when it isn't its picture's megapixels (a clip's megapixels × frames).
    func recordJobs(_ records: [RemoteJobRecord], doc: MovieDocument, work: Double? = nil) -> Double? {
        let cost = recordGPUJobs(records, work: work)
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

    func selectPlate(_ doc: MovieDocument, location id: String, take: UUID) {
        guard let j = doc.movie.locations.firstIndex(where: { $0.id == id }),
              doc.movie.locations[j].selectedPlate != take else { return }
        doc.movie.locations[j].selectedPlate = take
        doc.movie.reopen(.cast)  // keyframes made from the old plate no longer match
        saveMovie(doc)
    }

    /// Takes a plate take out of the movie and its file to the Trash.
    func deletePlate(_ doc: MovieDocument, location id: String, take: UUID) {
        guard let j = doc.movie.locations.firstIndex(where: { $0.id == id }),
              let file = doc.movie.locations[j].plates.first(where: { $0.id == take })?.file else { return }
        doc.movie.locations[j].plates.removeAll { $0.id == take }
        if doc.movie.locations[j].selectedPlate == take {
            doc.movie.locations[j].selectedPlate = doc.movie.locations[j].plates.last?.id
            doc.movie.reopen(.cast)
        }
        let url = movieStore.url(doc.id, relative: file)
        if Self.isAutomatedRun { try? FileManager.default.removeItem(at: url) }
        else { try? FileManager.default.trashItem(at: url, resultingItemURL: nil) }
        saveMovie(doc)
    }

    func selectKeyframe(_ doc: MovieDocument, shot id: UUID, take: UUID) {
        guard let j = doc.movie.shots.firstIndex(where: { $0.id == id }), doc.movie.shots[j].selectedKeyframe != take
        else { return }
        doc.movie.shots[j].selectedKeyframe = take
        doc.reopen(.keyframes, shot: id)
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
                    doc.reopen(.keyframes, shot: shot)
                }
            } else if let v = doc.movie.shots[j].videos.first(where: { $0.id == take }) {
                (file, poster) = (v.file, v.poster)
                lastFrame = MovieShots.lastFrameFile(of: v.file)
                doc.movie.shots[j].videos.removeAll { $0.id == take }
                if doc.movie.shots[j].selectedVideo == take {
                    doc.movie.shots[j].selectedVideo = doc.movie.shots[j].videos.last?.id
                    doc.reopen(.video, shot: shot)
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
