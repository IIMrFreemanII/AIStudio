import AIStudioLibrary
import AVFoundation
import Agent
import AppKit
import Foundation
import ImageIO
import LLMClient
import MovieCore
import NodesRemote

/// A Studio agent tool: closures into its session, with the call id (for the step row's revisit and revert), skips
/// turned into a note, and a reminder to look again when you used the window meanwhile.
struct StudioTool: AgentTool {
    let name: String
    let effect: ToolEffect
    let description: String
    let parameters: LLMClient.JSONValue
    let makePreview: @Sendable (ToolArgs) -> ToolPreview
    let body: @Sendable (ToolArgs, String) async throws -> ToolOutput

    init(_ name: String, effect: ToolEffect = .studio, _ description: String, _ parameters: LLMClient.JSONValue,
         title: @escaping @Sendable (ToolArgs) -> String,
         run: @escaping @Sendable (ToolArgs, String) async throws -> ToolOutput) {
        self.name = name
        self.effect = effect
        self.description = description
        self.parameters = parameters
        makePreview = { ToolPreview(kind: effect == .read ? .read : .run, title: title($0)) }
        body = run
    }

    func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview { makePreview(args) }

    func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        // A worker's calls are "<its delegate call>/<call>", so step rows of different workers never share an id.
        let id = StudioActorScope.prefix + ctx.callID
        do {
            var out = try await body(args, id)
            if await MainActor.run(body: { () -> Bool in
                guard let s = AppDelegate.model?.studio, s.reread else { return false }
                s.reread = false
                return true
            }) {
                out.text = StudioGate.rereadNote + "\n\n" + out.text
            }
            return out
        } catch is StudioSkipped {
            return ToolOutput(StudioGate.skippedNote)
        }
    }
}

/// One Studio agent reply's side: the app it drives, the controller showing its focus, and the graph helper sessions of
/// the workflows it edits (each workflow's edits are one undo step at the end).
@MainActor
final class StudioSession {
    weak var app: AppModel?
    let studio: StudioController
    let vision: Bool
    private var helpers: [UUID: GraphHelperSession] = [:]

    init(app: AppModel, vision: Bool) {
        self.app = app
        studio = app.studio
        self.vision = vision
    }

    func appModel() throws -> AppModel {
        guard let app else { throw ToolError("AIStudio is closing") }
        return app
    }

    /// The reply ended: each workflow's edits become one undo step.
    func finish() {
        for h in helpers.values { _ = h.commit() }
        helpers.removeAll()
    }

    // MARK: finding things

    /// The movie a call means: by title or id, else the one in front, else the only open one.
    func movie(_ name: String?) throws -> MovieDocument {
        let app = try appModel()
        if let name, !name.isEmpty {
            let q = name.lowercased()
            if let d = app.movies.first(where: { $0.id.uuidString.lowercased() == q || $0.movie.title.lowercased() == q }) { return d }
            if let m = app.movieSummaries.first(where: { $0.id.uuidString.lowercased() == q || $0.title.lowercased() == q })
                ?? app.movieSummaries.first(where: { $0.title.lowercased().contains(q) }) {
                app.openMovie(m.id)
                if let d = app.movies.first(where: { $0.id == m.id }) { return d }
            }
            throw ToolError("no movie called “\(name)”. Movies: \(app.movieSummaries.map(\.title).joined(separator: ", "))")
        }
        if let d = app.activeMovie { return d }
        if app.movies.count == 1 { return app.movies[0] }
        throw ToolError("say which movie (movie: its title); open: \(app.movies.map(\.movie.title).joined(separator: ", "))")
    }

    /// Shot `n` (1-based) of a movie.
    func shot(_ doc: MovieDocument, _ n: Int?) throws -> (Shot, Int) {
        guard let n else { throw ToolError("shot (its number, from 1) is missing") }
        guard n >= 1, n <= doc.movie.shots.count else { throw ToolError("there is no shot \(n): the movie has \(doc.movie.shots.count)") }
        let s = doc.movie.shots[n - 1]
        // Its card is on its sequence's pages: show that one when another is shown.
        if let q = doc.sequence, doc.movie.sequence(of: s) != q { doc.show(sequence: doc.movie.sequence(of: s)) }
        return (s, n)
    }

    func character(_ doc: MovieDocument, _ id: String?) throws -> MovieCharacter {
        guard let id, !id.isEmpty else { throw ToolError("character (its id or name) is missing") }
        let q = id.lowercased()
        guard let c = doc.movie.characters.first(where: { $0.id.lowercased() == q || $0.name.lowercased() == q }) else {
            throw ToolError("no character “\(id)”: \(doc.movie.characters.map { "\($0.id) (\($0.name))" }.joined(separator: ", "))")
        }
        return c
    }

    func cue(_ doc: MovieDocument, _ scene: String?) throws -> MusicCue {
        guard let scene, !scene.isEmpty else { throw ToolError("scene (its id) is missing") }
        guard let c = doc.movie.music.first(where: { $0.id == scene }) else {
            throw ToolError("no music cue for scene “\(scene)”: \(doc.movie.music.map(\.id).joined(separator: ", "))")
        }
        return c
    }

    /// Take `n` (1-based) of a list, else the chosen one.
    func take(_ takes: [Take], _ n: Int?, chosen: UUID?) throws -> Take {
        if let n {
            guard n >= 1, n <= takes.count else { throw ToolError("there is no take \(n): there are \(takes.count)") }
            return takes[n - 1]
        }
        if let t = takes.first(where: { $0.id == chosen }) ?? takes.last { return t }
        throw ToolError("there are no takes yet")
    }

    static func stage(_ s: String?) throws -> MovieStage {
        guard let s else { throw ToolError("stage is missing") }
        let k = s.lowercased()
        if k == "clips" { return .video }
        if k == "cast" || k == "sheets" { return .cast }
        guard let st = MovieStage(rawValue: k) else {
            throw ToolError("no stage “\(s)”: \(MovieStage.allCases.map(\.rawValue).joined(separator: ", "))")
        }
        return st
    }

    // MARK: where the app is

    func state() throws -> String {
        let app = try appModel()
        var out: [String] = []
        let panel = SidebarPanel(rawValue: UserDefaults.standard.string(forKey: "sidebarPanel") ?? "") ?? .workflows
        let open = UserDefaults.standard.object(forKey: "sidebarOpen") as? Bool ?? true
        switch app.frontItem {
        case .movie:
            if let d = app.activeMovie { out.append("In front: the movie “\(d.movie.title)” on its \(d.stage.title) page.") }
        case .workflow:
            out.append("In front: the workflow “\(app.active.name)”.")
        case .chat:
            out.append("In front: the chat “\(app.activeChat?.title ?? "")”.")
        }
        out.append("Sidebar: \(open ? panel.title + " panel" : "hidden").")
        out.append("Open workflows: " + app.tabs.map { "“\($0.name)”" }.joined(separator: ", ") + ".")
        if !app.movies.isEmpty { out.append("Open movies: " + app.movies.map { "“\($0.movie.title)”" }.joined(separator: ", ") + ".") }
        for d in app.movies {
            if let j = d.job {
                out.append("Movie “\(d.movie.title)” is working: \(j.title)\(j.progress.map { String(format: " (%.0f%%)", $0 * 100) } ?? "").")
            }
            if let g = d.pendingGate {
                out.append("Movie “\(d.movie.title)” shows a cost sheet: \(g.estimate.title), ≈ $\(String(format: "%.2f", g.estimate.lines.reduce(0) { $0 + $1.usd })) — the user decides.")
            }
            if let e = d.error { out.append("Movie “\(d.movie.title)” error: \(e)") }
        }
        let pending = app.queue.filter { !$0.isFinished }.count
        if pending > 0 { out.append("Workflow queue: \(pending) job(s) waiting or running.") }
        if let v = app.chatViewer { out.append("The viewer is open on \(v.titles?[safe: v.index] ?? "an item") (\(v.urls.count) items).") }
        if let d = app.activeMovie {
            out.append("\nThe movie:\n" + movieOverview(d))
        } else if app.workflowInFront {
            out.append("\nThe workflow:\n" + ((try? helper().getGraph()) ?? ""))
        }
        return out.joined(separator: "\n")
    }

    /// Shows the sequence `a` names (its number or id), when it names one.
    func selectSequence(_ doc: MovieDocument, _ a: ToolArgs) throws {
        guard let raw = a.string("sequence")?.trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return }
        let qs = doc.movie.sequences
        if ["all", "film", "whole film"].contains(raw.lowercased()) { doc.show(sequence: nil); return }
        let q = Int(raw).flatMap { qs.indices.contains($0 - 1) ? qs[$0 - 1] : nil }
            ?? qs.first { $0.id == raw.lowercased() || $0.title.lowercased() == raw.lowercased() }
        guard let q else { throw ToolError("no sequence \(raw): this film has \(qs.count) (1…\(qs.count))") }
        doc.show(sequence: q.id)
    }

    func movieOverview(_ d: MovieDocument) -> String {
        let m = d.movie
        var out = ["“\(m.title)” (\(m.aspect.title), about \(m.targetSeconds) s; spent $\(String(format: "%.2f", m.costSpent))"
                   + (m.budget.map { " of a $\(String(format: "%.2f", $0)) budget" } ?? "") + ")"]
        out.append("Stages: " + MovieStage.allCases.map { s in
            "\(s.rawValue) " + (d.isApproved(s) ? "approved" : d.isUnlocked(s) ? "open" : "locked")
        }.joined(separator: ", ") + (d.scopeTitle.map { " (per-sequence stages: \($0))" } ?? ""))
        if m.sequences.count > 1 {
            out.append("Sequences: " + m.sequences.enumerated().map { i, q in
                "\(i + 1). \(q.title) (act \(q.act), \(m.shots(inSequence: q.id).count) shots, at \(m.currentStage(in: q.id).rawValue))"
            }.joined(separator: "; "))
        }
        out.append("\(m.characters.count) characters, \(m.scenes.count) scenes, \(m.shots.count) shots; "
                   + "\(m.shots.filter { !$0.keyframes.isEmpty }.count) with keyframes, \(m.shots.filter { !$0.videos.isEmpty }.count) with clips.")
        return out.joined(separator: "\n")
    }

    // MARK: navigation

    func open(_ a: ToolArgs, _ id: String) async throws -> String {
        let app = try appModel()
        let what = (a.string("what") ?? "").lowercased()
        let name = a.string("name")
        if let panel = SidebarPanel.allCases.first(where: { $0.rawValue == what || $0.title.lowercased() == what }) {
            try await studio.act(StudioStep(doing: "Opening the \(panel.title) panel", done: "Opened \(panel.title)",
                                            target: "sidebar-tab:\(panel.rawValue)"), at: .none, callID: id) {
                await studio.go(.panel(panel))
            }
            return "The \(panel.title) panel is open in the sidebar."
        }
        switch what {
        case "movie":
            let doc = try movie(name)
            try await studio.act(StudioStep(doing: "Opening “\(doc.movie.title)”", done: "Opened “\(doc.movie.title)”",
                                            target: "movie-row:\(doc.movie.title)"), at: .panel(.movies), callID: id) {
                app.openMovie(doc.id)
            }
            try selectSequence(doc, a)
            if let s = a.string("stage") {
                let stage = try Self.stage(s)
                guard doc.isUnlocked(stage) else { throw ToolError("the \(stage.title) page is locked until \(stage.previous?.title ?? "the stage before") is approved") }
                try await studio.act(StudioStep(doing: "Going to \(stage.title)", done: "Went to \(stage.title)",
                                                target: "movie-stage-\(stage.rawValue)"), at: .movie(doc.id, nil), callID: id) {
                    doc.stage = stage
                    await studio.settle("movie-view")
                }
            }
            return "“\(doc.movie.title)” is in front on its \(doc.stage.title) page.\n" + movieOverview(doc)
        case "workflow":
            guard let name, !name.isEmpty else { throw ToolError("name (the workflow's) is missing") }
            let q = name.lowercased()
            if let tab = app.tabs.first(where: { $0.name.lowercased() == q }) ?? app.tabs.first(where: { $0.name.lowercased().contains(q) }) {
                try await studio.act(StudioStep(doing: "Switching to “\(tab.name)”", done: "Switched to “\(tab.name)”",
                                                target: "graph-canvas"), at: .workflow(tab.id), callID: id) {}
                return "The workflow “\(tab.name)” is in front.\n" + ((try? helper().getGraph()) ?? "")
            }
            let all = app.workflowTree.allWorkflows
            guard let w = all.first(where: { $0.name.lowercased() == q }) ?? all.first(where: { $0.name.lowercased().contains(q) }) else {
                throw ToolError("no workflow called “\(name)”. Saved: \(all.map(\.name).joined(separator: ", ")); open: \(app.tabs.map(\.name).joined(separator: ", "))")
            }
            try await studio.act(StudioStep(doing: "Opening “\(w.name)”", done: "Opened “\(w.name)”",
                                            target: "workflow-row:\(w.name)"), at: .panel(.workflows), callID: id) {
                _ = app.open(w.url)
                await studio.settle("graph-canvas")
            }
            return "The workflow “\(w.name)” is open.\n" + ((try? helper().getGraph()) ?? "")
        case "preset":
            guard let name, !name.isEmpty else { throw ToolError("name (the preset's name or id) is missing") }
            let q = name.lowercased()
            guard let p = app.presets.first(where: { $0.id.lowercased() == q || $0.name.lowercased() == q })
                ?? app.presets.first(where: { $0.name.lowercased().contains(q) }) else {
                throw ToolError("no preset “\(name)”: studio_list with what: presets lists them")
            }
            try await studio.act(StudioStep(doing: "Opening the preset “\(p.name)”", done: "Opened the preset “\(p.name)”",
                                            target: "sidebar-tab:presets"), at: .panel(.presets), callID: id) {
                _ = app.openPreset(p)
                await studio.settle("graph-canvas")
            }
            return "The preset “\(p.name)” is open as the workflow “\(app.active.name)”"
                + (p.needsServer == "gpu" ? " (it runs on the cloud GPU server)." : " (it runs on this Mac).")
                + "\n" + ((try? helper().getGraph()) ?? "")
        case "viewer":
            throw ToolError("open the viewer with look_at, which shows what it looks at")
        case "close_viewer":
            app.chatViewer = nil
            app.viewerResult = nil
            return "The viewer is closed."
        default:
            throw ToolError("what: one of \((SidebarPanel.allCases.map(\.rawValue) + ["movie", "workflow", "preset", "close_viewer"]).joined(separator: ", "))")
        }
    }

    func list(_ what: String) throws -> String {
        let app = try appModel()
        switch what.lowercased() {
        case "movies":
            app.reloadMovies()
            return app.movieSummaries.isEmpty ? "No movies yet." : app.movieSummaries.map {
                "- “\($0.title)”: \($0.shots) shots, at the \($0.stage.title) stage"
            }.joined(separator: "\n")
        case "workflows":
            let saved = app.workflowTree.allWorkflows.map { "- “\($0.name)” (saved)" }
            let open = app.tabs.map { "- “\($0.name)” (open\($0.id == app.activeTabID ? ", in front" : ""))" }
            return (open + saved).joined(separator: "\n")
        case "presets":
            return app.presets.sorted { ($0.category.rawValue, $0.order) < ($1.category.rawValue, $1.order) }.map {
                "- \($0.id): “\($0.name)” — \($0.category.title), \($0.modelName)\($0.needsServer == "gpu" ? ", cloud GPU" : ", this Mac"): \($0.summary)"
            }.joined(separator: "\n")
        case "servers", "gpus", "models":
            var out: [String] = []
            for c in app.connections.gpuServers {
                guard case .gpuOnline(let info)? = app.connections.status[c.id] else { out.append("- \(c.name): offline"); continue }
                out.append("- \(c.name) (GPU server, \(info.parallelJobs ?? 1) jobs at once per model):")
                for t in info.tasks {
                    let price = t.pricePerSecond.map { String(format: "$%.5f/s", $0) } ?? "price unknown"
                    out.append("  - \(t.id): \(t.title) (\(t.kind)) on \(t.gpu), \(price)\(t.installed ? "" : ", not installed")")
                }
            }
            if out.isEmpty { out.append("No GPU servers: add one in the Servers panel.") }
            return out.joined(separator: "\n")
        default:
            throw ToolError("what: movies, workflows, presets or servers")
        }
    }

    // MARK: movies

    func movieCreate(_ a: ToolArgs, _ id: String) async throws -> String {
        let app = try appModel()
        let idea = try a.required("idea")
        let doc = try await studio.act(StudioStep(doing: "Making a new movie", done: "Made a new movie", target: "movies-new"),
                                       at: .panel(.movies), callID: id) { app.newMovie() }
        if let t = a.string("title"), !t.isEmpty { doc.movie.title = t }
        if let s = a.string("style"), !s.isEmpty { doc.movie.style = s }
        if let asp = a.string("aspect"), let v = MovieAspect.allCases.first(where: { $0.rawValue == asp || $0.title.lowercased() == asp.lowercased() }) {
            doc.movie.aspect = v
        }
        if let secs = a.int("seconds"), secs > 0 { doc.movie.targetSeconds = min(secs, MovieProject.maxTargetSeconds) }
        doc.stage = .idea
        try await studio.act(StudioStep(doing: "Typing the idea", done: "Typed the idea", target: "movie-idea"),
                             at: .movie(doc.id, .idea), callID: id) {
            await studio.type(idea) { doc.movie.idea = $0 }
        }
        app.saveMovie(doc)
        return "Made the movie “\(doc.movie.title)” (id \(doc.id.uuidString)) with the idea filled in. Next: movie_run with stage script writes the script (after the user approves its cost)."
    }

    func movieGet(_ a: ToolArgs) throws -> String {
        let doc = try movie(a.string("movie"))
        try selectSequence(doc, a)
        let m = doc.movie
        func takes(_ ts: [Take], _ chosen: UUID?) -> String {
            ts.isEmpty ? "no takes" : "\(ts.count) take(s)" + (ts.firstIndex { $0.id == chosen }.map { ", take \($0 + 1) chosen" } ?? ", none chosen")
        }
        switch (a.string("part") ?? "overview").lowercased() {
        case "overview":
            return movieOverview(doc) + "\nIdea: \(m.idea)\nStyle: \(m.style)"
        case "story", "outline", "bible":
            var out = ["Title: \(m.title)", "Logline: \(m.logline)", "Premise: \(m.premise)", "Tone: \(m.tone)",
                       "Main theme: \(m.theme)", "Characters:"]
            out += m.characters.map {
                "- \($0.id): \($0.name)\($0.isNarrator ? " (narrator)" : "") — role \($0.role); wants: \($0.goal); flaw: \($0.flaw); "
                    + "arc: \($0.arc); relationships: \($0.relationships); look: \($0.look); voice: \($0.voice)"
            }
            out.append("Places:")
            out += m.locations.map { "- \($0.id): \($0.name) — \($0.look)" }
            let secs = MoviePlanner.sceneSeconds(for: m)
            for (i, q) in m.sequences.enumerated() {
                out.append("Sequence \(i + 1) (act \(q.act)): \(q.title) — \(q.summary)")
                out += m.scenes(inSequence: q.id).map {
                    "  scene \($0.id): \($0.heading) at \($0.location), \(Int(secs[$0.id] ?? 0)) s, with \($0.characters.joined(separator: ", ")) — beat: \($0.beat)"
                }
            }
            return out.joined(separator: "\n")
        case "script":
            var out = ["Title: \(m.title)", "Logline: \(m.logline)", "Characters:"]
            out += m.characters.map { "- \($0.id): \($0.name) — \($0.look)" }
            for s in m.scenes {
                if let q = m.sequences.first(where: { $0.id == s.sequence }), m.scenes.first(where: { $0.sequence == q.id })?.id == s.id,
                   let notes = m.critiqueNotes[q.id], !notes.isEmpty {
                    out.append("Editor's notes on \(q.title): \(notes)")
                }
                out.append("Scene \(s.id): \(s.heading) — \(s.summary) (mood: \(s.mood))\(s.isWritten ? "" : " — not written yet")")
                if !s.action.isEmpty { out.append("  Action: \(s.action)") }
                out += s.dialogue.map { "  \($0.character): “\($0.text)” \($0.delivery.isEmpty ? "" : "(\($0.delivery))")" }
            }
            return out.joined(separator: "\n")
        case "shots", "keyframes", "clips", "video", "sound", "voices", "lines":
            return m.shots.enumerated().map { i, s in
                var l = ["Shot \(i + 1) (scene \(s.scene), \(String(format: "%.1f", s.duration)) s, \(s.framing), camera \(s.camera); characters: \(s.characters.joined(separator: ", ")))"]
                l.append("  keyframe prompt: \(s.keyframePrompt)")
                l.append("  motion prompt: \(s.motionPrompt)")
                if !s.sfx.isEmpty { l.append("  sound effects: \(s.sfx)") }
                for (k, d) in s.dialogue.enumerated() {
                    l.append("  line \(k + 1): \(d.character): “\(d.text)” — \(takes(d.takes, d.selectedTake))")
                }
                l.append("  keyframes: \(takes(s.keyframes, s.selectedKeyframe)); clips: \(takes(s.videos, s.selectedVideo)); effects: \(takes(s.sounds, s.selectedSound))")
                return l.joined(separator: "\n")
            }.joined(separator: "\n")
        case "cast":
            return m.characters.map {
                "- \($0.id): \($0.name). Look: \($0.look). Voice: \($0.voice). Sheets: \(takes($0.sheets, $0.selectedSheet)); voice samples: \(takes($0.voiceSamples, $0.selectedVoiceSample))"
            }.joined(separator: "\n")
        case "music":
            return m.music.map { "- scene \($0.id): \($0.prompt) — \(takes($0.takes, $0.selected))" }.joined(separator: "\n")
        default:
            throw ToolError("part: overview, story, script, shots, cast or music")
        }
    }

    /// The page and control a movie field is on, the stage its change un-approves, and how to read and write it.
    struct Field {
        var page: MovieStage
        var target: String
        var reopens: MovieStage?
        var label: String
        var get: @MainActor () -> String
        var set: @MainActor (String) -> Void
    }

    func field(_ doc: MovieDocument, _ a: ToolArgs) throws -> Field {
        let name = (a.string("field") ?? "").lowercased()
        func shotField(_ kp: WritableKeyPath<Shot, String>, page: MovieStage, target: String, reopens: MovieStage, what: String) throws -> Field {
            let (s, n) = try shot(doc, a.int("shot"))
            let sid = s.id
            func index() -> Int? { doc.movie.shots.firstIndex { $0.id == sid } }
            return Field(page: page, target: "\(target):\(sid.uuidString)", reopens: reopens, label: "shot \(n)'s \(what)",
                         get: { index().map { doc.movie.shots[$0][keyPath: kp] } ?? "" },
                         set: { v in if let i = index() { doc.movie.shots[i][keyPath: kp] = v } })
        }
        func characterField(_ kp: WritableKeyPath<MovieCharacter, String>, page: MovieStage, target: String, reopens: MovieStage, what: String) throws -> Field {
            let c = try character(doc, a.string("character"))
            let cid = c.id
            func index() -> Int? { doc.movie.characters.firstIndex { $0.id == cid } }
            return Field(page: page, target: "\(target):\(cid)", reopens: reopens, label: "\(c.name)'s \(what)",
                         get: { index().map { doc.movie.characters[$0][keyPath: kp] } ?? "" },
                         set: { v in if let i = index() { doc.movie.characters[i][keyPath: kp] = v } })
        }
        func movieField(_ kp: WritableKeyPath<MovieProject, String>, page: MovieStage, target: String, reopens: MovieStage?, what: String) -> Field {
            Field(page: page, target: target, reopens: reopens, label: "the \(what)",
                  get: { doc.movie[keyPath: kp] }, set: { doc.movie[keyPath: kp] = $0 })
        }
        switch name {
        case "title": return movieField(\.title, page: .idea, target: "movie-idea", reopens: nil, what: "title")
        case "idea": return movieField(\.idea, page: .idea, target: "movie-idea", reopens: .story, what: "idea")
        case "style": return movieField(\.style, page: .idea, target: "movie-idea", reopens: .story, what: "style")
        case "logline": return movieField(\.logline, page: .script, target: "movie-view", reopens: .script, what: "logline")
        case "premise": return movieField(\.premise, page: .story, target: "movie-premise", reopens: .story, what: "premise")
        case "tone": return movieField(\.tone, page: .story, target: "movie-view", reopens: .story, what: "tone")
        case "theme": return movieField(\.theme, page: .story, target: "movie-view", reopens: .story, what: "main theme")
        case "goal": return try characterField(\.goal, page: .story, target: "movie-character-bible", reopens: .story, what: "goal")
        case "flaw": return try characterField(\.flaw, page: .story, target: "movie-character-bible", reopens: .story, what: "flaw")
        case "arc": return try characterField(\.arc, page: .story, target: "movie-character-bible", reopens: .story, what: "arc")
        case "beat":
            guard let sid = a.string("scene"), let scene = doc.movie.scenes.first(where: { $0.id == sid }) else {
                throw ToolError("scene (its id, as movie_get story lists them) is missing or unknown")
            }
            let id = scene.id
            func index() -> Int? { doc.movie.scenes.firstIndex { $0.id == id } }
            return Field(page: .story, target: "movie-outline", reopens: .story, label: "scene \(id)'s beat",
                         get: { index().map { doc.movie.scenes[$0].beat } ?? "" },
                         set: { v in if let i = index() { doc.movie.scenes[i].beat = v } })
        case "location_look", "place_look":
            let ref = (a.string("location") ?? "").lowercased()
            guard let l = doc.movie.locations.first(where: { $0.id == ref || $0.name.lowercased() == ref }) else {
                throw ToolError("location (a place's id or name) is missing or unknown")
            }
            let id = l.id
            func index() -> Int? { doc.movie.locations.firstIndex { $0.id == id } }
            return Field(page: .story, target: "movie-view", reopens: .story, label: "\(l.name)'s look",
                         get: { index().map { doc.movie.locations[$0].look } ?? "" },
                         set: { v in if let i = index() { doc.movie.locations[i].look = v } })
        case "keyframe_prompt": return try shotField(\.keyframePrompt, page: .keyframes, target: "movie-keyframe", reopens: .keyframes, what: "keyframe prompt")
        case "motion_prompt": return try shotField(\.motionPrompt, page: .video, target: "movie-clip", reopens: .video, what: "motion prompt")
        case "sfx", "effects": return try shotField(\.sfx, page: .sound, target: "movie-effects", reopens: .sound, what: "sound effects")
        case "look": return try characterField(\.look, page: .cast, target: "movie-cast", reopens: .cast, what: "look")
        case "voice": return try characterField(\.voice, page: .voices, target: "movie-voice", reopens: .voices, what: "voice")
        case "music_prompt":
            let c = try cue(doc, a.string("scene"))
            let cid = c.id
            func index() -> Int? { doc.movie.music.firstIndex { $0.id == cid } }
            return Field(page: .sound, target: "movie-music:\(cid)", reopens: .sound, label: "scene \(cid)'s music prompt",
                         get: { index().map { doc.movie.music[$0].prompt } ?? "" },
                         set: { v in if let i = index() { doc.movie.music[i].prompt = v } })
        case "line":
            let (s, n) = try shot(doc, a.int("shot"))
            let k = (a.int("line") ?? 1) - 1
            guard k >= 0, k < s.dialogue.count else { throw ToolError("shot \(n) has \(s.dialogue.count) line(s)") }
            let sid = s.id
            func index() -> Int? { doc.movie.shots.firstIndex { $0.id == sid } }
            return Field(page: .voices, target: "movie-lines:\(sid.uuidString)", reopens: .voices, label: "shot \(n)'s line \(k + 1)",
                         get: { index().flatMap { i in doc.movie.shots[i].dialogue[safe: k]?.text } ?? "" },
                         set: { v in if let i = index(), k < doc.movie.shots[i].dialogue.count { doc.movie.shots[i].dialogue[k].text = v } })
        default:
            throw ToolError("field: title, idea, style, logline, premise, tone, theme, goal, flaw, arc, beat, location_look, keyframe_prompt, motion_prompt, sfx, look, voice, music_prompt or line")
        }
    }

    func movieEdit(_ a: ToolArgs, _ id: String) async throws -> String {
        let doc = try movie(a.string("movie"))
        let value = try a.required("value")
        let f = try field(doc, a)
        guard doc.isUnlocked(f.page) || f.page == .idea else {
            throw ToolError("the \(f.page.title) page is locked until the stage before it is approved")
        }
        guard !doc.isBusy else { throw ToolError("the movie is busy (\(doc.job?.title ?? "working")): wait for it to finish") }
        let old = f.get()
        let approvedBefore = doc.movie.approvalKeys
        try await studio.act(StudioStep(doing: "Typing \(f.label)", done: "Typed \(f.label)", target: f.target),
                             at: .movie(doc.id, f.page), callID: id) {
            await studio.type(value) { f.set($0) }
        }
        if let r = f.reopens, doc.isApproved(r) { doc.reopen(r) }
        let after = doc.movie.approvalKeys
        let undone = MovieStage.allCases.filter { s in
            approvedBefore.contains { $0 == s.rawValue || $0.hasPrefix(s.rawValue + "@") }
                && !after.contains { $0 == s.rawValue || $0.hasPrefix(s.rawValue + "@") }
        }
        let set = f.set, reopen = f.reopens
        studio.reverts[id] = StudioRevert(title: "Put back \(f.label)") { [weak doc] in
            set(old)
            if let doc, let reopen, doc.isApproved(reopen) { doc.reopen(reopen) }
        }
        try appModel().saveMovie(doc)
        return "Set \(f.label)." + (undone.isEmpty ? "" : " That took back the approval of: \(undone.map(\.title).joined(separator: ", ")) (approve them again when done).")
    }

    /// What `movie_run` starts: the request function, the page, its button, and how to count what it made.
    func work(_ doc: MovieDocument, _ a: ToolArgs) throws -> (label: String, page: MovieStage, target: String, start: @MainActor () -> Void, count: @MainActor () -> Int) {
        let app = try appModel()
        let what = (a.string("work") ?? a.string("stage") ?? "").lowercased()
        let shots = try (a.raw["shots"] as? [Any])?.compactMap { ($0 as? Int) ?? Int("\($0)") }.map { try shot(doc, $0).0.id }
        let chars = try (a.raw["characters"] as? [Any])?.map { try character(doc, "\($0)").id }
        let scenes = (a.raw["scenes"] as? [Any])?.map { "\($0)" }
        let m = { doc.movie }
        switch what {
        case "story", "outline":
            return ("the story", .idea, "movie-write-story", { app.requestStory(doc) }, { m().scenes.count })
        case "script", "scenes":
            return ("the script", .script, "movie-write-script", { app.requestScript(doc, scenes: scenes) },
                    { m().scenes.filter(\.isWritten).count })
        case "shots":
            return ("the shot list", .shots, "movie-write-shots", { app.requestShots(doc, scenes: scenes) }, { m().shots.count })
        case "cast", "sheets":
            return ("character sheets and plates of places", .cast, "movie-make-sheets",
                    { app.requestSheets(doc, characters: chars) },
                    { m().characters.reduce(0) { $0 + $1.sheets.count } + m().locations.reduce(0) { $0 + $1.plates.count } })
        case "keyframes":
            return ("keyframes", .keyframes, "movie-make-keyframes", { app.requestKeyframes(doc, shots: shots) },
                    { m().shots.reduce(0) { $0 + $1.keyframes.count } })
        case "voices", "voice_samples":
            return ("voice samples", .voices, "movie-make-voices", { app.requestVoiceSamples(doc, characters: chars) },
                    { m().characters.reduce(0) { $0 + $1.voiceSamples.count } })
        case "lines":
            return ("spoken lines", .voices, "movie-speak-lines", { app.requestLines(doc, shots: shots) },
                    { m().shots.reduce(0) { $0 + $1.dialogue.reduce(0) { $0 + $1.takes.count } } })
        case "clips", "video":
            return ("clips", .video, "movie-make-clips", { app.requestClips(doc, shots: shots) },
                    { m().shots.reduce(0) { $0 + $1.videos.count } })
        case "effects", "sfx":
            return ("sound effects", .sound, "movie-make-effects", { app.requestEffects(doc, shots: shots) },
                    { m().shots.reduce(0) { $0 + $1.sounds.count } })
        case "music":
            return ("music", .sound, "movie-make-music", { app.requestMusic(doc, scenes: scenes) },
                    { m().music.reduce(0) { $0 + $1.takes.count } })
        default:
            throw ToolError("work: story, script, shots, cast, keyframes, voices, lines, clips, effects or music")
        }
    }

    /// Starts a stage's work, waits for the user's answer on the cost sheet, then for the work to finish.
    func movieRun(_ a: ToolArgs, _ id: String) async throws -> String {
        let doc = try movie(a.string("movie"))
        try selectSequence(doc, a)
        let w = try work(doc, a)
        guard doc.isUnlocked(w.page) || w.page == .idea else {
            throw ToolError("the \(w.page.title) page is locked until the stage before it is approved")
        }
        // One job per movie: another actor's (or the user's) run first.
        let turn = "movie:\(doc.id)"
        try await studio.lock(turn, caption: studio.actorPrefix + "Waiting for “\(doc.movie.title)” to be free")
        defer { studio.unlock(turn) }
        while doc.isBusy || doc.pendingGate != nil {
            let c = studio.actorPrefix + "Waiting for \(doc.job?.title.lowercased() ?? "the cost sheet") to finish"
            if studio.caption != c { studio.caption = c }
            try await Task.sleep(for: .milliseconds(300))
        }
        // One cost sheet on screen at a time, and its page stays in front until the user answers.
        try await studio.lock("sheet", caption: studio.actorPrefix + "Waiting for another cost sheet")
        var sheetHeld = true
        defer { if sheetHeld { studio.unlock("sheet"); studio.releaseScreen() } }
        try await studio.holdScreen(.movie(doc.id, w.page))
        let before = w.count()
        let spent = doc.movie.costSpent
        let started = Date()
        // Whether the cost sheet came up (it may come and go while the pointer still holds on the button).
        final class Seen { var gate = false; var estimate: CostGate? }
        let seen = Seen()
        let watcher = Task { @MainActor in
            while !Task.isCancelled {
                if let g = doc.pendingGate { seen.gate = true; seen.estimate = g }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        defer { watcher.cancel() }
        try await studio.act(StudioStep(doing: "Starting \(w.label)", done: "Made \(w.label)", target: w.target),
                             at: .movie(doc.id, w.page), callID: id) {
            doc.error = nil
            w.start()
            if let g = doc.pendingGate { seen.gate = true; seen.estimate = g }
        }
        do {
            // The cost check, then the sheet.
            let deadline = Date().addingTimeInterval(90)
            while doc.pendingGate == nil, doc.job != nil, doc.job?.title == "Checking the cost…", Date() < deadline {
                try await Task.sleep(for: .milliseconds(200))
            }
            if doc.job == nil, doc.pendingGate == nil {
                try await Task.sleep(for: .milliseconds(300))
            }
            if doc.pendingGate != nil || seen.gate {
                let usd = (doc.pendingGate ?? seen.estimate)?.estimate.lines.reduce(0) { $0 + $1.usd } ?? 0
                studio.waitingOnUser = studio.actorPrefix + String(format: "\(w.label.prefix(1).uppercased() + w.label.dropFirst()) ≈ $%.2f: waiting for you to press Run", usd)
                await studio.point(at: "movie-gate-run", fallback: "movie-cost-gate", caption: studio.waitingOnUser ?? "")
                defer { studio.waitingOnUser = nil }
                while doc.pendingGate != nil { try await Task.sleep(for: .milliseconds(200)) }
                try await Task.sleep(for: .milliseconds(400))
                if doc.job == nil, w.count() == before {
                    return String(format: "The user cancelled \(w.label) on the cost sheet (it estimated ≈ $%.2f). Nothing was made; ask them how to go on.", usd)
                }
            } else if doc.job == nil, w.count() == before {
                return doc.error.map { "It didn't start: \($0)" } ?? "Nothing new to make for \(w.label): it's all there already (pass shots, characters or scenes to retake some)."
            }
            studio.unlock("sheet")
            studio.releaseScreen()
            sheetHeld = false
            // The work.
            while let job = doc.job {
                let caption = job.title + (job.progress.map { String(format: " · %.0f%%", $0 * 100) } ?? "")
                if studio.caption != caption { studio.caption = caption }
                try await Task.sleep(for: .milliseconds(400))
            }
        } catch {
            if doc.pendingGate != nil { doc.pendingGate = nil }
            if doc.job != nil { try appModel().stopMovieJob(doc) }
            throw error
        }
        let made = w.count() - before
        var out = "Finished \(w.label) in \(Int(Date().timeIntervalSince(started))) s: \(made) new."
        let cost = doc.movie.costSpent - spent
        if cost > 0 {
            out += String(format: " It cost ≈ $%.2f.", cost)
            studio.current.spent += cost
        }
        if let e = doc.error { out += " Error: \(e)" }
        out += " Check what was made with look_at (images and clips) before going on."
        return out
    }

    func chooseTake(_ a: ToolArgs, _ id: String) async throws -> String {
        let app = try appModel()
        let doc = try movie(a.string("movie"))
        let n = a.int("take")
        guard let n else { throw ToolError("take (its number, from 1) is missing") }
        switch (a.string("kind") ?? "").lowercased() {
        case "keyframe":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.keyframes, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing shot \(k)'s keyframe \(n)", done: "Chose shot \(k)'s keyframe \(n)",
                                            target: "movie-keyframe:\(s.id.uuidString)"), at: .movie(doc.id, .keyframes), callID: id) {
                app.selectKeyframe(doc, shot: s.id, take: t.id)
            }
        case "clip":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.videos, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing shot \(k)'s clip \(n)", done: "Chose shot \(k)'s clip \(n)",
                                            target: "movie-clip:\(s.id.uuidString)"), at: .movie(doc.id, .video), callID: id) {
                app.selectClip(doc, shot: s.id, take: t.id)
            }
        case "effects":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.sounds, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing shot \(k)'s effects take \(n)", done: "Chose shot \(k)'s effects take \(n)",
                                            target: "movie-effects:\(s.id.uuidString)"), at: .movie(doc.id, .sound), callID: id) {
                app.selectSound(doc, shot: s.id, take: t.id)
            }
        case "line":
            let (s, k) = try shot(doc, a.int("shot"))
            let li = (a.int("line") ?? 1) - 1
            guard let line = s.dialogue[safe: li] else { throw ToolError("shot \(k) has \(s.dialogue.count) line(s)") }
            let t = try take(line.takes, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing shot \(k)'s line \(li + 1) take \(n)", done: "Chose shot \(k)'s line \(li + 1) take \(n)",
                                            target: "movie-lines:\(s.id.uuidString)"), at: .movie(doc.id, .voices), callID: id) {
                app.selectLine(doc, shot: s.id, line: line.id, take: t.id)
            }
        case "sheet":
            let c = try character(doc, a.string("character"))
            let t = try take(c.sheets, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing \(c.name)'s sheet \(n)", done: "Chose \(c.name)'s sheet \(n)",
                                            target: "movie-cast:\(c.id)"), at: .movie(doc.id, .cast), callID: id) {
                app.selectSheet(doc, character: c.id, take: t.id)
            }
        case "voice":
            let c = try character(doc, a.string("character"))
            let t = try take(c.voiceSamples, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing \(c.name)'s voice sample \(n)", done: "Chose \(c.name)'s voice sample \(n)",
                                            target: "movie-voice:\(c.id)"), at: .movie(doc.id, .voices), callID: id) {
                app.selectVoiceSample(doc, character: c.id, take: t.id)
            }
        case "music":
            let c = try cue(doc, a.string("scene"))
            let t = try take(c.takes, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Choosing scene \(c.id)'s music take \(n)", done: "Chose scene \(c.id)'s music take \(n)",
                                            target: "movie-music:\(c.id)"), at: .movie(doc.id, .sound), callID: id) {
                app.selectMusic(doc, scene: c.id, take: t.id)
            }
        default:
            throw ToolError("kind: keyframe, clip, effects, line, sheet, voice or music")
        }
        app.saveMovie(doc)
        return "Chosen. (Choosing a take can take back the approval of later stages; movie_get overview shows them.)"
    }

    static let approveTargets: [MovieStage: String] = [
        .story: "movie-approve-story", .script: "movie-approve-script", .shots: "movie-approve-shots", .cast: "movie-approve-cast",
        .keyframes: "movie-approve-keyframes", .voices: "movie-approve-voices", .video: "movie-approve-clips",
        .sound: "movie-approve-sound", .edit: "movie-export",
    ]

    func approve(_ a: ToolArgs, _ id: String) async throws -> String {
        let doc = try movie(a.string("movie"))
        let stage = try Self.stage(a.string("stage"))
        try selectSequence(doc, a)
        guard doc.isUnlocked(stage) else { throw ToolError("\(stage.title) is locked: approve \(stage.requires.map(\.title).joined(separator: " and ")) first") }
        guard !doc.isBusy else { throw ToolError("the movie is busy: wait for it to finish") }
        if stage == .edit && !doc.movie.editIsApprovable {
            throw ToolError("the edit can be approved once every sequence's sound is")
        }
        let scope = doc.scope(stage)
        try await studio.act(StudioStep(doing: "Approving \(stage.title)", done: "Approved \(stage.title)", target: Self.approveTargets[stage]),
                             at: .movie(doc.id, stage), callID: id) {
            doc.approve(stage)
        }
        studio.reverts[id] = StudioRevert(title: "Take back the approval") { [weak doc] in doc?.movie.unapprove(stage, in: scope) }
        try appModel().saveMovie(doc)
        return "\(stage.title) is approved" + (doc.scopeTitle.map { " for \($0)" } ?? "") + "; the next page is unlocked."
    }

    func deleteTake(_ a: ToolArgs, _ id: String) async throws -> String {
        let app = try appModel()
        let doc = try movie(a.string("movie"))
        let n = a.int("take")
        guard n != nil else { throw ToolError("take (its number) is missing") }
        switch (a.string("kind") ?? "").lowercased() {
        case "keyframe":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.keyframes, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Deleting shot \(k)'s keyframe \(n!)", done: "Deleted shot \(k)'s keyframe \(n!)",
                                            target: "movie-keyframe:\(s.id.uuidString)"), at: .movie(doc.id, .keyframes), callID: id) {
                app.deleteTake(doc, shot: s.id, take: t.id)
            }
        case "clip":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.videos, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Deleting shot \(k)'s clip \(n!)", done: "Deleted shot \(k)'s clip \(n!)",
                                            target: "movie-clip:\(s.id.uuidString)"), at: .movie(doc.id, .video), callID: id) {
                app.deleteTake(doc, shot: s.id, take: t.id)
            }
        case "sheet":
            let c = try character(doc, a.string("character"))
            let t = try take(c.sheets, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Deleting \(c.name)'s sheet \(n!)", done: "Deleted \(c.name)'s sheet \(n!)",
                                            target: "movie-cast:\(c.id)"), at: .movie(doc.id, .cast), callID: id) {
                app.deleteTake(doc, character: c.id, take: t.id)
            }
        case "effects":
            let (s, k) = try shot(doc, a.int("shot"))
            let t = try take(s.sounds, n, chosen: nil)
            try await studio.act(StudioStep(doing: "Deleting shot \(k)'s effects take \(n!)", done: "Deleted shot \(k)'s effects take \(n!)",
                                            target: "movie-effects:\(s.id.uuidString)"), at: .movie(doc.id, .sound), callID: id) {
                app.deleteSoundTake(doc, take: t.id)
            }
        default:
            throw ToolError("kind: keyframe, clip, sheet or effects")
        }
        app.saveMovie(doc)
        return "Deleted (its file went to the Trash)."
    }

    func export(_ a: ToolArgs, _ id: String) async throws -> String {
        let app = try appModel()
        let doc = try movie(a.string("movie"))
        guard !doc.isBusy else { throw ToolError("the movie is busy: wait for it to finish") }
        let folder = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Movies")
        let name = a.string("name").flatMap { $0.isEmpty ? nil : $0 } ?? doc.movie.title
        let url = StudioFiles.unused(name, ext: "mp4", in: folder) { FileManager.default.fileExists(atPath: $0.path) }
        let made = try await studio.act(StudioStep(doing: "Exporting the film", done: "Exported the film", target: "movie-export"),
                                        at: .movie(doc.id, .edit), callID: id) {
            await app.exportMovie(doc, to: url)
        }
        guard let made else { return "The export failed: \(doc.error ?? "unknown error")" }
        return "Exported to \(made.path) (a new file; nothing was overwritten)."
    }

    // MARK: looking

    /// Images (or a clip's frames) of a take for the model, opened in the viewer while it looks.
    func look(_ a: ToolArgs, _ id: String) async throws -> ToolOutput {
        let app = try appModel()
        let kind = (a.string("kind") ?? "").lowercased()
        if kind == "outputs" {
            let h = try helper()
            let out = try h.lastOutputs(count: a.int("count") ?? 1, node: nil)
            return vision ? out : ToolOutput(Self.blind + "\n" + out.text)
        }
        let doc = try movie(a.string("movie"))
        let folder = app.movieStore.folder(doc.id)
        var takes: [Take] = []
        var name = ""
        var target: String?
        var page = doc.stage
        var openViewer: () -> Void = {}
        switch kind {
        case "keyframe":
            let (s, n) = try shot(doc, a.int("shot"))
            takes = s.keyframes; name = "Shot \(n) keyframe"; target = "movie-keyframe:\(s.id.uuidString)"; page = .keyframes
            let chosen = try take(s.keyframes, a.int("take"), chosen: s.selectedKeyframe)
            openViewer = { app.viewKeyframes(doc, shot: s.id, take: chosen.id) }
            takes = [chosen]
        case "sheet":
            let c = try character(doc, a.string("character"))
            name = "\(c.name) sheet"; target = "movie-cast:\(c.id)"; page = .cast
            let chosen = try take(c.sheets, a.int("take"), chosen: c.selectedSheet)
            openViewer = { app.viewSheets(doc, character: c.id, take: chosen.id) }
            takes = [chosen]
        case "clip":
            let (s, n) = try shot(doc, a.int("shot"))
            name = "Shot \(n)"; target = "movie-clip:\(s.id.uuidString)"; page = .video
            let chosen = try take(s.videos, a.int("take"), chosen: s.selectedVideo)
            openViewer = { app.viewShotClips(doc, shot: s.id, take: chosen.id) }
            takes = [chosen]
        default:
            throw ToolError("kind: keyframe, sheet, clip or outputs (the workflow's latest results)")
        }
        let t = takes[0]
        let url = folder.appendingPathComponent(t.file)
        let images = try await studio.act(StudioStep(doing: "Looking at \(name)", done: "Looked at \(name)", target: target),
                                          at: .movie(doc.id, page), callID: id) { () -> [String] in
            // The viewer opens only on screen, one actor at a time; off screen it just reads the file.
            let shows = !studio.isOffscreen
            if shows {
                try await studio.lock("viewer")
                openViewer()
                try? await Task.sleep(for: .seconds(studio.pace == .fast ? 0.3 : 1.2))
            }
            let frames = MediaFrames.isVideo(url) ? await MediaFrames.video(url, at: [0.1, 0.35, 0.65, 0.9]) : MediaFrames.image(url).map { [$0] } ?? []
            if shows {
                app.chatViewer = nil
                studio.unlock("viewer")
            }
            return frames.compactMap { GraphHelper.dataURL($0, maxSide: 1024) }
        }
        guard !images.isEmpty else { return ToolOutput("Couldn't read \(t.file).", isError: true) }
        let what = MediaFrames.isVideo(url) ? "4 frames (at 10, 35, 65 and 90 %) of \(name)\(t.seconds.map { String(format: ", %.1f s long", $0) } ?? "")" : name
        guard vision else { return ToolOutput(Self.blind + " (\(what) is \(t.file).)") }
        return ToolOutput("Here is \(what). Check it against its prompt: the right people (how many, who), clothes and faces matching the cast, the setting, and anything broken (extra limbs, doubled people, garbled text).", images: images)
    }

    static let blind = "This model can't see images, so it can't check results itself: ask the user to look, or switch the panel to a model that sees images."

    // MARK: workflows

    /// The graph helper session for the workflow in front.
    func helper() throws -> GraphHelperSession {
        let app = try appModel()
        guard app.workflowInFront else { throw ToolError("no workflow is in front: open one with studio_open (what: workflow or preset)") }
        let wf = app.active
        if let h = helpers[wf.id] { return h }
        let h = GraphHelperSession(app: app, workflow: wf)
        helpers[wf.id] = h
        return h
    }

    func editGraph(_ ops: [Any], _ id: String) async throws -> String {
        let h = try helper()
        let parsed: [GraphOps.Op]
        do { parsed = try GraphOps.parse(ops) } catch { throw ToolError("\(error) — nothing was changed") }
        let wf = try appModel().active
        let node: String? = parsed.lazy.compactMap { op -> String? in
            switch op {
            case .set(let n, _, _), .disconnect(let n, _), .remove(let n), .rename(let n, _): return n
            case .connect(_, _, let to, _): return to
            case .add: return nil
            }
        }.first
        let before = Set(wf.graph.nodes.map(\.id))
        let result = try await studio.act(StudioStep(doing: parsed.count == 1 ? "Changing the workflow" : "Making \(parsed.count) changes to the workflow",
                                                     done: parsed.count == 1 ? "Changed the workflow" : "Made \(parsed.count) changes to the workflow",
                                                     target: node.map { "graph-node:\($0)" } ?? "graph-canvas"),
                                          at: .workflow(wf.id), callID: id) {
            try h.edit(parsed)
        }
        // Only new nodes: the ring moves to the first one once it's on the canvas (and a revisit goes there).
        if node == nil, let added = wf.graph.nodes.first(where: { !before.contains($0.id) }) {
            let target = "graph-node:\(added.id)"
            await studio.settle(target)
            await studio.point(at: target, fallback: "graph-canvas", caption: "Added \(added.title ?? added.type)")
            studio.revisits[id] = (.workflow(wf.id), target)
            await studio.hold()
        }
        return result
    }

    func setPrompt(_ text: String, node: String?, _ id: String) async throws -> String {
        let h = try helper()
        let wf = try appModel().active
        let target = node ?? wf.graph.nodes.first { AppModel.promptFields[$0.type] != nil }?.id
        return try await studio.act(StudioStep(doing: "Writing the prompt", done: "Wrote the prompt", target: target.map { "graph-node:\($0)" }),
                                    at: .workflow(wf.id), callID: id) {
            try h.setPrompt(text, node: node)
        }
    }

    /// Runs the workflow in front: Cloud GPU nodes first show a cost sheet and wait for the user's Run.
    func runWorkflow(_ id: String) async throws -> String {
        let app = try appModel()
        let h = try helper()
        let wf = app.active
        if let err = app.validationError(of: wf.graph) { throw ToolError("the workflow can't run: \(err)") }
        let remote = wf.graph.nodes.compactMap { RemoteNodes.taskID(forNodeType: $0.type) }
        try await studio.act(StudioStep(doing: "Running the workflow", done: "Ran the workflow", target: "generate-button"),
                             at: .workflow(wf.id), callID: id) {}
        if !remote.isEmpty {
            try await studio.lock("sheet", caption: studio.actorPrefix + "Waiting for another cost sheet")
            defer { studio.unlock("sheet") }
            let estimate = await app.workflowRunEstimate(tasks: remote, workflow: wf.name)
            let usd = estimate.lines.reduce(0) { $0 + $1.usd }
            final class Pressed { var run = false }
            let pressed = Pressed()
            app.studioRunGate = CostGate(estimate: estimate, run: { pressed.run = true })
            studio.waitingOnUser = studio.actorPrefix + String(format: "Cost sheet ≈ $%.2f: waiting for you to press Run", usd)
            await studio.point(at: "movie-gate-run", fallback: "movie-cost-gate", caption: studio.waitingOnUser ?? "")
            defer { studio.waitingOnUser = nil }
            do {
                while app.studioRunGate != nil { try await Task.sleep(for: .milliseconds(200)) }
            } catch {
                app.studioRunGate = nil
                throw error
            }
            guard pressed.run else {
                return String(format: "The user cancelled the run on the cost sheet (it estimated ≈ $%.2f). Nothing ran; ask them how to go on.", usd)
            }
        }
        studio.caption = "Running “\(wf.name)”…"
        return try await h.run()
    }
}

/// Pictures of a file for the model: an image, or frames of a clip.
enum MediaFrames {
    static func isVideo(_ url: URL) -> Bool { ["mp4", "mov", "m4v"].contains(url.pathExtension.lowercased()) }

    static func image(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    static func video(_ url: URL, at fractions: [Double]) async -> [CGImage] {
        let asset = AVURLAsset(url: url)
        guard let duration = try? await asset.load(.duration), duration.seconds > 0 else { return [] }
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: 1024, height: 1024)
        var out: [CGImage] = []
        for f in fractions {
            let t = CMTime(seconds: duration.seconds * f, preferredTimescale: 600)
            if let img = try? await gen.image(at: t).image { out.append(img) }
        }
        return out
    }
}

extension Array {
    subscript(safe i: Int) -> Element? { indices.contains(i) ? self[i] : nil }
}
