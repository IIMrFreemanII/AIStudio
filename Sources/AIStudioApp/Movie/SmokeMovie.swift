import CoreGraphics
import Foundation
import ImageIO
import LLMClient
import MovieCore

extension SmokeTest {
    /// `--smoke-movie <gpu server url>`: a movie from an idea to an approved shot list against the GPU server's
    /// fake LLM (`AISTUDIO_FAKE_LLM=1`, which answers the planner's prompts with a canned script and shots): the
    /// cost gate comes first, the script and shots land in the movie; with `AISTUDIO_FAKE_MODELS=1` too, the cast's
    /// sheets and the keyframes (the fake edit marks how many pictures it got); it saves, reopens and goes to the Trash.
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

        // Cast: one reference sheet per character (Qwen-Image), behind its gate.
        let spentOnWriting = doc.movie.costSpent
        model.requestSheets(doc)
        await waitForGate(doc)
        guard let sheetGate = doc.pendingGate else { say("FAIL no cost gate for the sheets: \(doc.error ?? "")"); return false }
        check(sheetGate.estimate.lines.map(\.count) == [2] && sheetGate.estimate.usd > 0,
              "the sheet gate counts one job per character, with a price")
        check(doc.job == nil && doc.movie.characters.allSatisfy { $0.sheets.isEmpty }, "no sheet before Run")
        doc.pendingGate = nil
        sheetGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the sheets were made\(doc.error.map { ": \($0)" } ?? "")")
        let store = model.movieStore
        let sheetFiles = doc.movie.characters.compactMap(\.selectedSheetFile).map { store.url(doc.id, relative: $0) }
        check(doc.movie.castIsReady && sheetFiles.count == 2 && sheetFiles.allSatisfy { pixelSize($0) == "1664x928" },
              "each character has a chosen 1664×928 sheet on disk")
        check(doc.movie.costSpent > spentOnWriting && doc.movie.characters.allSatisfy { ($0.sheets.first?.costUSD ?? 0) > 0 },
              "the sheets' cost is on the takes and the movie")
        doc.movie.approve(.cast)

        // Keyframes: edits of the sheets (and of the shot before, for a continuing shot), in film order.
        model.requestKeyframes(doc)
        await waitForGate(doc)
        guard let keyGate = doc.pendingGate else { say("FAIL no cost gate for the keyframes: \(doc.error ?? "")"); return false }
        check(keyGate.estimate.lines.map(\.label) == ["Keyframes with the cast"] && keyGate.estimate.lines[0].count == 4,
              "the keyframe gate counts four edits")
        doc.pendingGate = nil
        keyGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the keyframes were made\(doc.error.map { ": \($0)" } ?? "")")
        let keys = doc.movie.shots.map { $0.selectedKeyframeFile.map { store.url(doc.id, relative: $0) } }
        check(doc.movie.keyframesAreReady && keys.allSatisfy { $0.flatMap(pixelSize) == "1344x768" },
              "every shot has a chosen 1344×768 keyframe")
        // The fake edit's bottom row: the number of pictures / 4 (Mara's sheet; plus the shot before for shot 2).
        let rows = keys.map { $0.flatMap { bottomRow($0) } ?? -1 }
        check(abs(rows[0] - 0.25) < 0.02 && abs(rows[1] - 0.5) < 0.02 && abs(rows[2] - 0.25) < 0.02,
              "shot 1 from one picture, shot 2 from two (the sheet and shot 1); a new scene starts afresh: \(rows)")
        let timings = GPUTimings.load(model.gpuTimingsURL)
        check(timings.jobs.keys.contains { $0.hasPrefix("qwen-image-edit@") } && timings.lastJob["qwen-image"] != nil,
              "the jobs' times were learned")
        // Another take of one shot: chosen only when picked.
        let first = doc.movie.shots[0]
        model.requestKeyframes(doc, shots: [first.id])
        await waitForGate(doc)
        if let g = doc.pendingGate { doc.pendingGate = nil; g.run(); await waitForJob(doc) }
        check(doc.movie.shots[0].keyframes.count == 2 && doc.movie.shots[0].selectedKeyframe == first.selectedKeyframe,
              "a retake adds a take and keeps the chosen one")
        doc.movie.approve(.keyframes)
        model.selectKeyframe(doc, shot: first.id, take: doc.movie.shots[0].keyframes[1].id)
        check(!doc.movie.approved.contains(.keyframes) && doc.movie.approved.contains(.cast),
              "choosing another take reopens Keyframes only")
        doc.movie.approve(.keyframes)

        // Saved, closed, reopened as it was, then trashed.
        model.saveMovies()
        let id = doc.id
        let snapshot = doc.movie
        model.closeMovie(id)
        check(model.activeMovieID == nil && !model.openItems.contains(.movie(id)), "⌘W closed it")
        check(model.movieSummaries.first?.id == id && model.movieSummaries.first?.stage == .voices, "the list shows it at Voices")
        model.openMovie(id)
        check(model.activeMovie?.movie == snapshot, "it reopens exactly as saved")
        model.deleteMovie(id)
        check(!model.movieSummaries.contains { $0.id == id } && model.activeMovieID == nil, "deleted")

        model.connections.remove(conn.id)
        say(ok ? "PASS" : "FAILED")
        return ok
    }

    /// "1344x768".
    private static func pixelSize(_ url: URL) -> String? {
        Thumbnail.pixelSize(url.path).map { "\($0.width)x\($0.height)" }
    }

    /// The bottom-left pixel's red channel, 0…1.
    private static func bottomRow(_ url: URL) -> Double? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        var px = [UInt8](repeating: 0, count: 4)
        guard let ctx = CGContext(data: &px, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        // The image's bottom-left pixel lands on the 1×1 context (CG's origin is bottom left).
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return Double(px[0]) / 255
    }
}
