import Agent
import Foundation
import LLMClient
import MovieCore

/// Movies: opening, closing and saving them, and the LLM stages (script, shot list) behind their cost gates.
extension AppModel {
    var movieStore: MovieStore { MovieStore(root: library.movies) }

    func reloadMovies() { movieSummaries = movieStore.list() }

    // MARK: lifecycle

    @discardableResult
    func newMovie() -> MovieDocument {
        var project = MovieProject()
        project.llm = defaultMovieLLM
        let doc = MovieDocument(movie: project)
        movies.append(doc)
        activeMovieID = doc.id
        saveMovie(doc)
        return doc
    }

    /// The model a new movie writes with: the last one used for a movie, else the first chat model there is.
    var defaultMovieLLM: MovieLLM? {
        if let s = UserDefaults.standard.string(forKey: "movieLLM"), let data = s.data(using: .utf8),
           let key = try? JSONDecoder().decode(MovieLLM.self, from: data),
           llm.model(LLMKey(serverID: key.serverID, modelID: key.modelID)) != nil {
            return key
        }
        return llm.models.first.map { MovieLLM(serverID: $0.server.id, modelID: $0.model.id) }
    }

    func rememberMovieLLM(_ key: MovieLLM?) {
        guard let key, let data = try? JSONEncoder().encode(key) else { return }
        UserDefaults.standard.set(String(data: data, encoding: .utf8), forKey: "movieLLM")
    }

    func openMovie(_ id: UUID) {
        if movies.contains(where: { $0.id == id }) { activeMovieID = id; return }
        do {
            let project = try movieStore.load(id)
            movies.append(MovieDocument(movie: project, saved: project))
            activeMovieID = id
        } catch {
            reloadMovies()
            lastError = "Couldn't open the movie: \(error.localizedDescription)"
        }
    }

    /// ⌘W on a movie: saved and closed (its job, if any, is stopped).
    func closeMovie(_ id: UUID) {
        guard let doc = movies.first(where: { $0.id == id }) else { return }
        doc.job?.task?.cancel()
        saveMovie(doc)
        leaveFront(.movie(id))
        movies.removeAll { $0.id == id }
        if activeMovieID == id { activeMovieID = nil }
    }

    /// Moves a movie and everything generated for it to the Trash.
    func deleteMovie(_ id: UUID) {
        if movies.contains(where: { $0.id == id }) {
            movies.first { $0.id == id }?.job?.task?.cancel()
            leaveFront(.movie(id))
            movies.removeAll { $0.id == id }
            if activeMovieID == id { activeMovieID = nil }
        }
        try? movieStore.delete(id, useTrash: !Self.isAutomatedRun)
        reloadMovies()
    }

    func saveMovie(_ doc: MovieDocument) {
        guard doc.isDirty else { return }
        do {
            doc.movie.modified = Date()
            try movieStore.save(doc.movie)
            doc.saved = doc.movie
            // Just this movie's row: reading every movie back from disk after each render is slow with long films.
            let m = doc.movie
            let row = MovieSummary(id: m.id, title: m.title, stage: m.currentStage, shots: m.shots.count, modified: m.modified)
            if let i = movieSummaries.firstIndex(where: { $0.id == m.id }) {
                movieSummaries[i] = row
                movieSummaries.sort { $0.modified > $1.modified }
            } else {
                reloadMovies()
            }
        } catch {
            doc.error = "Couldn't save: \(error.localizedDescription)"
        }
    }

    /// The session timer: changed movies are written, and which are open (and in front) is kept for the next launch.
    func saveMovies() {
        for doc in movies { saveMovie(doc) }
        guard persistSettings, !Self.isAutomatedRun else { return }
        UserDefaults.standard.set(movies.map(\.id.uuidString), forKey: "openMovies")
        UserDefaults.standard.set(activeMovieID?.uuidString, forKey: "activeMovie")
    }

    func restoreOpenMovies() {
        reloadMovies()
        guard persistSettings, !Self.isAutomatedRun else { return }
        for s in UserDefaults.standard.stringArray(forKey: "openMovies") ?? [] {
            if let id = UUID(uuidString: s), let p = try? movieStore.load(id) {
                movies.append(MovieDocument(movie: p, saved: p))
            }
        }
        if let s = UserDefaults.standard.string(forKey: "activeMovie"), let id = UUID(uuidString: s),
           movies.contains(where: { $0.id == id }), activeChatID == nil {
            activeMovieID = id
        }
    }

    // MARK: LLM stages

    /// The movie's model and a client for it, or why not.
    private func movieLLM(_ doc: MovieDocument) -> (LLMKey, any ChatStreaming)? {
        guard let m = doc.movie.llm else {
            doc.error = "Choose a model to write with (Idea › Model)."
            return nil
        }
        let key = LLMKey(serverID: m.serverID, modelID: m.modelID)
        guard let client = llm.client(m.serverID) else {
            doc.error = "The movie's model is on a server that isn't connected any more: choose another one."
            return nil
        }
        return (key, client.chat)
    }

    /// The cost of `requests` requests to the movie's model, as the gate shows it.
    func llmEstimate(_ doc: MovieDocument, label: String, requests: Int) -> CostEstimate.Line? {
        guard let m = doc.movie.llm else { return nil }
        let key = LLMKey(serverID: m.serverID, modelID: m.modelID)
        let model = llm.model(key)
        let status = llm.status(key)
        let gpu = model?.gpuLabel ?? status?.gpu ?? "GPU"
        let price = status?.usdPerSecond ?? model.flatMap { MovieCost.price(gpu: $0.gpu, count: $0.count) } ?? 0
        let warm = status?.state == "ready"
        return MovieCost.llm(label: label, requests: requests, gpu: gpu, usdPerSecond: price, warm: warm,
                             coldStartSeconds: status?.coldStart?.typicalSeconds,
                             idleSeconds: status?.idleSeconds ?? 180)
    }

    private func llmNotes(_ doc: MovieDocument) -> [String] {
        guard let m = doc.movie.llm else { return [] }
        let key = LLMKey(serverID: m.serverID, modelID: m.modelID)
        let warm = llm.status(key)?.state == "ready"
        let name = llm.model(key)?.name ?? m.modelID
        return [
            warm ? "\(name) is running: no start-up." : "\(name) is cold: it starts first (counted above).",
            "Includes the server's idle time before the GPU shuts down. Billing is Modal's; this is an estimate.",
        ]
    }

    /// Shows the cost sheet for `requests` requests to the movie's model, after making sure the app knows the model
    /// (its GPU and price: right after launch the server's list may not be in yet) and its current state.
    private func gateLLM(_ doc: MovieDocument, title: String, label: String, requests: Int,
                         run: @escaping @MainActor () -> Void) {
        doc.error = nil
        guard doc.job == nil, movieLLM(doc) != nil, let m = doc.movie.llm else { return }
        let key = LLMKey(serverID: m.serverID, modelID: m.modelID)
        doc.job = MovieJob(title: "Checking the cost…", detail: "Asking the server about \(m.modelID)")
        Task { @MainActor [weak self] in
            defer { if doc.job?.title == "Checking the cost…" { doc.job = nil } }
            guard let self else { return }
            if self.llm.model(key) == nil { await self.llm.refresh(m.serverID) }
            guard self.llm.model(key) != nil else {
                doc.error = "\(m.modelID) isn't on its server any more: choose another model (Idea › Writes with)."
                return
            }
            await self.llm.refreshStatus(key)
            guard let line = self.llmEstimate(doc, label: label, requests: requests) else { return }
            doc.pendingGate = CostGate(estimate: CostEstimate(title: title, lines: [line], notes: self.llmNotes(doc)),
                                       run: run)
        }
    }

    /// Idea › Write Story: asks to confirm the cost, then writes the bible and the outline (two requests).
    func requestStory(_ doc: MovieDocument) {
        gateLLM(doc, title: "Write the Story", label: "Write the bible and outline", requests: 2) { [weak self] in
            self?.writeStory(doc)
        }
    }

    /// Script › Write: each scene in order (`scenes` nil: all of them), then the critique of each sequence they're in
    /// (when the movie has it on).
    func requestScript(_ doc: MovieDocument, scenes: [String]? = nil) {
        let ids = scenes ?? doc.movie.scenes.map(\.id)
        guard !ids.isEmpty else { return }
        let sequences = critiqued(doc, ids)
        let requests = ids.count + sequences.count
        gateLLM(doc, title: ids.count == 1 ? "Write the Scene" : "Write the Script",
                label: (ids.count == 1 ? "Write 1 scene" : "Write \(ids.count) scenes")
                    + (sequences.isEmpty ? "" : " + critique \(sequences.count) sequence\(sequences.count == 1 ? "" : "s")"),
                requests: requests) { [weak self] in
            self?.writeScript(doc, scenes: ids)
        }
    }

    /// The sequences the scenes are in, in order, when the critique pass is on.
    private func critiqued(_ doc: MovieDocument, _ scenes: [String]) -> [String] {
        guard doc.movie.critique else { return [] }
        let ids = Set(scenes.map { doc.movie.sequence(ofScene: $0) })
        return doc.movie.sequences.map(\.id).filter { ids.contains($0) }
    }

    /// Shot List › Break Down: one request per scene (`scenes` nil: all of them).
    func requestShots(_ doc: MovieDocument, scenes: [String]? = nil) {
        let ids = scenes ?? doc.movie.scenes.map(\.id)
        guard !ids.isEmpty else { return }
        gateLLM(doc, title: "Write the Shot List",
                label: ids.count == 1 ? "Break down 1 scene" : "Break down \(ids.count) scenes",
                requests: ids.count) { [weak self] in
            self?.writeShots(doc, scenes: ids)
        }
    }

    /// Runs `work` as the movie's job (one at a time). Each request's cost (the server's own figure) goes onto the
    /// movie as it finishes.
    private func runLLMJob(_ doc: MovieDocument, title: String,
                           _ work: @escaping @MainActor (MovieDocument, LLMKey, any ChatStreaming) async throws -> Void) {
        guard doc.job == nil, let (key, client) = movieLLM(doc) else { return }
        rememberMovieLLM(doc.movie.llm)
        llm.lastActivity = Date()
        doc.job = MovieJob(title: title, detail: "Asking \(llm.model(key)?.name ?? key.modelID)…")
        let task = Task { @MainActor [weak self] in
            do {
                try await work(doc, key, client)
            } catch is CancellationError {
            } catch {
                doc.error = "\(title) failed: \(error)"
            }
            doc.job = nil
            self?.saveMovie(doc)
        }
        doc.job?.task = task
    }

    private func progress(_ doc: MovieDocument, _ prefix: String) -> @Sendable (MoviePlanner.Progress) -> Void {
        let started = Date()
        return { p in
            Task { @MainActor in
                if p.characters > 0 {
                    let n = p.characters < 1000 ? "\(p.characters)" : String(format: "%.1fk", Double(p.characters) / 1000)
                    doc.job?.detail = "\(prefix) \(n) characters"
                } else {
                    let t = Int(Date().timeIntervalSince(started))
                    doc.job?.detail = "\(prefix) \(p.phase ?? "waiting for the model") · \(t / 60):\(String(format: "%02d", t % 60))"
                }
            }
        }
    }

    private func spend(_ doc: MovieDocument) -> @Sendable (Double) -> Void {
        { usd in Task { @MainActor in doc.movie.costSpent += usd } }
    }

    private func writeStory(_ doc: MovieDocument) {
        runLLMJob(doc, title: "Writing the story") { [weak self] doc, key, client in
            guard let self else { return }
            doc.job?.progress = 0
            var (system, user) = MovieWriters.biblePrompt(for: doc.movie)
            let bible = try await MoviePlanner.ask(client: client, model: key.modelID, system: system, user: user,
                                                   maxTokens: 8000,
                                                   onProgress: self.progress(doc, "The bible:"),
                                                   onCost: self.spend(doc), parse: MovieWriters.parseBible)
            MovieWriters.apply(bible, to: &doc.movie)
            doc.job?.progress = 0.5
            (system, user) = MovieWriters.outlinePrompt(for: doc.movie)
            let cast = doc.movie.characters, places = doc.movie.locations, target = doc.movie.targetSeconds
            let outline = try await MoviePlanner.ask(client: client, model: key.modelID, system: system, user: user,
                                                     maxTokens: 12000,
                                                     onProgress: self.progress(doc, "The outline:"),
                                                     onCost: self.spend(doc)) {
                try MovieWriters.parseOutline($0, cast: cast, locations: places, targetSeconds: target)
            }
            MovieWriters.apply(outline, to: &doc.movie)
            doc.shotNotes = [:]
            // Breeze says each line as its delivery directs, in English and Chinese: the voice for those scripts,
            // until a voice is made (then it's the user's choice on the Voices page).
            if !doc.movie.characters.contains(where: { !$0.voiceSamples.isEmpty }) {
                let text = [doc.movie.idea, doc.movie.premise, doc.movie.logline].joined(separator: " ")
                doc.movie.voiceEngine = MovieVoices.suitsBreeze(text) ? .breeze : .qwen3
            }
            doc.movie.approve(.idea)
            doc.sequence = doc.movie.sequences.count > 1 ? doc.movie.sequences.first?.id : nil
            doc.stage = .story
        }
    }

    private func writeScript(_ doc: MovieDocument, scenes: [String]) {
        runLLMJob(doc, title: "Writing the script") { [weak self] doc, key, client in
            guard let self else { return }
            doc.movie.reopen(.script)
            let sequences = self.critiqued(doc, scenes)
            let total = Double(scenes.count + sequences.count)
            for (i, id) in scenes.enumerated() {
                try Task.checkCancellation()
                guard let scene = doc.movie.scenes.first(where: { $0.id == id }) else { continue }
                let prefix = "Scene \(i + 1) of \(scenes.count):"
                doc.job?.detail = "\(prefix) asking…"
                doc.job?.progress = Double(i) / total
                let (system, user) = MovieWriters.scenePrompt(for: doc.movie, scene: scene)
                let cast = doc.movie.characters
                let draft = try await MoviePlanner.ask(client: client, model: key.modelID, system: system, user: user,
                                                       maxTokens: 6000,
                                                       onProgress: self.progress(doc, prefix),
                                                       onCost: self.spend(doc)) {
                    try MovieWriters.parseScene($0, cast: cast)
                }
                MovieWriters.apply(draft, scene: id, to: &doc.movie)
                self.saveMovie(doc)
            }
            for (k, q) in sequences.enumerated() {
                try Task.checkCancellation()
                let prefix = "Critique \(k + 1) of \(sequences.count):"
                doc.job?.detail = "\(prefix) asking…"
                doc.job?.progress = Double(scenes.count + k) / total
                let (system, user) = MovieWriters.critiquePrompt(for: doc.movie, sequence: q)
                let cast = doc.movie.characters
                let critique = try await MoviePlanner.ask(client: client, model: key.modelID, system: system, user: user,
                                                          maxTokens: 16000,
                                                          onProgress: self.progress(doc, prefix),
                                                          onCost: self.spend(doc)) {
                    try MovieWriters.parseCritique($0, cast: cast)
                }
                MovieWriters.apply(critique, sequence: q, to: &doc.movie)
            }
            doc.movie.reopen(.script)
            doc.stage = .script
        }
    }

    private func writeShots(_ doc: MovieDocument, scenes: [String]) {
        runLLMJob(doc, title: "Writing the shot list") { [weak self] doc, key, client in
            guard let self else { return }
            for (i, id) in scenes.enumerated() {
                try Task.checkCancellation()
                guard let scene = doc.movie.scenes.first(where: { $0.id == id }) else { continue }
                let prefix = "Scene \(i + 1) of \(scenes.count):"
                doc.job?.detail = "\(prefix) asking…"
                doc.job?.progress = Double(i) / Double(scenes.count)
                let (system, user) = MoviePlanner.shotPrompt(for: doc.movie, scene: scene)
                let cast = doc.movie.characters
                let cap = MoviePlanner.talkingCap(for: doc.movie)
                let seconds = MoviePlanner.sceneSeconds(for: doc.movie)[id] ?? 15
                var notes: [String] = []
                let list = try await MoviePlanner.ask(client: client, model: key.modelID, system: system, user: user,
                                                      maxTokens: 8000,
                                                      onProgress: self.progress(doc, prefix),
                                                      onCost: self.spend(doc),
                                                      check: { MovieWriters.shotProblems($0.shots, scene: scene, seconds: seconds) },
                                                      onProblems: { notes = $0 }) {
                    try MoviePlanner.parseShotList($0, scene: id, cast: cast, talkingCap: cap)
                }
                MoviePlanner.replaceShots(list.shots, scene: id, in: &doc.movie)
                if let j = doc.movie.scenes.firstIndex(where: { $0.id == id }), !list.blocking.isEmpty {
                    doc.movie.scenes[j].blocking = list.blocking
                }
                doc.shotNotes[id] = notes.isEmpty ? nil : notes.joined(separator: "; ")
                doc.movie.reopen(.shots, in: doc.movie.sequence(ofScene: id))
            }
            doc.stage = .shots
        }
    }

    func stopMovieJob(_ doc: MovieDocument) {
        doc.job?.task?.cancel()
    }
}
