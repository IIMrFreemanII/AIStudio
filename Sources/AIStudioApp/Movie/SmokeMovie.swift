import Foundation
import LLMClient
import MovieCore

extension SmokeTest {
    /// `--smoke-movie <gpu server url>`: a movie from an idea to an approved shot list against the GPU server's
    /// fake LLM (`AISTUDIO_FAKE_LLM=1`, which answers the planner's prompts with a canned script and shots): the
    /// cost gate comes first, the script and shots land in the movie, it saves, reopens and goes to the Trash.
    static func runMovie(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-movie] \(s)"); fflush(stdout) }
        var ok = true
        func check(_ cond: Bool, _ what: String) {
            say((cond ? "ok   " : "FAIL ") + what)
            if !cond { ok = false }
        }

        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: "smoke")
        await model.connections.test(conn.id)
        guard case .gpuOnline? = model.connections.status[conn.id] else { say("server not reachable"); return false }
        let llm = model.llm
        guard let client = llm.client(conn.id) else { say("no client"); return false }
        await llm.refresh(conn.id)
        var mid = llm.listings[conn.id]?.models.first { $0.repo == "fake/tiny" || $0.id == "fake/tiny" }?.id
        if mid == nil {
            guard let st = try? await client.install(LLMInstallRequest(url: "fake/tiny")) else { say("install failed"); return false }
            mid = st.id
            for _ in 0..<150 {
                await llm.refresh(conn.id)
                if llm.listings[conn.id]?.models.contains(where: { $0.id == mid && $0.installed == true }) == true { break }
                try? await Task.sleep(for: .milliseconds(200))
            }
        }
        guard let mid, llm.model(LLMKey(serverID: conn.id, modelID: mid))?.installed == true else {
            say("model didn't install"); return false
        }

        func waitForGate(_ doc: MovieDocument) async {
            for _ in 0..<100 where doc.pendingGate == nil && doc.error == nil { try? await Task.sleep(for: .milliseconds(100)) }
        }
        func waitForJob(_ doc: MovieDocument) async {
            for _ in 0..<600 where doc.job != nil { try? await Task.sleep(for: .milliseconds(100)) }
        }

        // Idea → the gate → the script.
        let doc = model.newMovie()
        check(model.activeMovieID == doc.id && model.frontItem == .movie(doc.id), "the new movie is in front")
        check(model.openItems.contains(.movie(doc.id)), "it's in the Window menu's list")
        doc.movie.llm = MovieLLM(serverID: conn.id, modelID: mid)
        doc.movie.idea = "A girl keeps a lighthouse lit through a storm."
        doc.movie.targetSeconds = 30
        model.requestScript(doc)
        await waitForGate(doc)
        guard let gate = doc.pendingGate else { say("FAIL no cost gate for the script: \(doc.error ?? "")"); return false }
        check(gate.estimate.lines.count == 1 && gate.estimate.lines[0].count == 1, "the gate lists one request")
        check(doc.job == nil && doc.movie.scenes.isEmpty, "nothing runs before Run")
        doc.pendingGate = nil
        gate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the script was written\(doc.error.map { ": \($0)" } ?? "")")
        check(doc.movie.title == "The Last Lantern" && doc.movie.scenes.count == 2 && doc.movie.characters.count == 2,
              "title, 2 scenes and 2 characters came in")
        check(doc.movie.approved == [.idea] && doc.stage == .script, "the idea is approved and the script is shown")

        // Script approved → the shot list, one request per scene.
        doc.movie.approve(.script)
        model.requestShots(doc)
        await waitForGate(doc)
        guard let shotGate = doc.pendingGate else { say("FAIL no cost gate for the shots"); return false }
        check(shotGate.estimate.lines.first?.count == 2, "the gate counts one request per scene")
        doc.pendingGate = nil
        shotGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the shot list was written\(doc.error.map { ": \($0)" } ?? "")")
        check(doc.movie.shots.count == 4 && doc.movie.shots.map(\.scene) == ["s1", "s1", "s2", "s2"],
              "two shots per scene, in scene order")
        guard doc.movie.shots.count == 4 else { say("FAILED"); return false }
        check(doc.movie.shots[1].talking && doc.movie.shots[1].dialogue.first?.character == "mara",
              "the talking shot has Mara's line")
        check(doc.movie.shots[1].continuesPrevious && !doc.movie.shots[2].continuesPrevious,
              "continues within a scene, never across into the next one")
        doc.movie.approve(.shots)

        // Saved, closed, reopened as it was, then trashed.
        model.saveMovies()
        let id = doc.id
        let snapshot = doc.movie
        model.closeMovie(id)
        check(model.activeMovieID == nil && !model.openItems.contains(.movie(id)), "⌘W closed it")
        check(model.movieSummaries.first?.id == id && model.movieSummaries.first?.stage == .cast, "the list shows it at Cast")
        model.openMovie(id)
        check(model.activeMovie?.movie == snapshot, "it reopens exactly as saved")
        model.deleteMovie(id)
        check(!model.movieSummaries.contains { $0.id == id } && model.activeMovieID == nil, "deleted")

        model.connections.remove(conn.id)
        say(ok ? "PASS" : "FAILED")
        return ok
    }
}
