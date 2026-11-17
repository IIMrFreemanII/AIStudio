import AppKit
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import LLMClient
import MovieCore
import NodesRemote

extension SmokeTest {
    /// `--smoke-movie <gpu server url>`: a movie from an idea to an approved shot list against the GPU server's
    /// fake LLM (`AISTUDIO_FAKE_LLM=1`, which answers the planner's prompts with a canned script and shots): the
    /// cost gate comes first, the script and shots land in the movie; with `AISTUDIO_FAKE_MODELS=1` too, the cast's
    /// sheets and the keyframes (the fake edit marks how many pictures it got), the clips (the fake Wan starts on its
    /// start picture and ends on its end picture), the edit and its MP4; it saves, reopens and goes to the Trash.
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

        // Estimates from the default job times: the fake server's jobs take no time, and what earlier runs learned
        // from them would make every task look as cheap as the next.
        try? FileManager.default.removeItem(at: model.gpuTimingsURL)

        /// With AISTUDIO_SMOKE_SNAPSHOTS=<folder>: the main window as it is, in dark and in light.
        func snapshots(_ name: String) async {
            guard let dir = ProcessInfo.processInfo.environment["AISTUDIO_SMOKE_SNAPSHOTS"],
                  let w = NSApp.windows.first(where: { $0.isVisible && $0.contentView != nil && $0.frame.width > 600 }) else { return }
            let old = w.appearance
            for (scheme, look) in [("dark", NSAppearance.Name.darkAqua), ("light", .aqua)] {
                w.appearance = NSAppearance(named: look)
                try? await Task.sleep(for: .milliseconds(700))
                guard let v = w.contentView, let rep = v.bitmapImageRepForCachingDisplay(in: v.bounds) else { continue }
                v.cacheDisplay(in: v.bounds, to: rep)
                try? rep.representation(using: .png, properties: [:])?
                    .write(to: URL(fileURLWithPath: dir).appendingPathComponent("movie-\(name)-\(scheme).png"))
            }
            w.appearance = old
        }

        func waitForGate(_ doc: MovieDocument) async {
            for _ in 0..<100 where doc.pendingGate == nil && doc.error == nil { try? await Task.sleep(for: .milliseconds(100)) }
        }
        /// Returns the most renders seen running at once (the job bar's lines).
        @discardableResult
        func waitForJob(_ doc: MovieDocument, seconds: Int = 60) async -> Int {
            var most = 0
            for _ in 0..<(seconds * 10) where doc.job != nil {
                most = max(most, doc.job?.lines.count ?? 0)
                try? await Task.sleep(for: .milliseconds(100))
            }
            return most
        }

        // Idea → the gate → the script.
        let doc = model.newMovie()
        check(model.activeMovieID == doc.id && model.frontItem == .movie(doc.id), "the new movie is in front")
        check(model.openItems.contains(.movie(doc.id)), "it's in the Window menu's list")
        doc.movie.llm = MovieLLM(serverID: conn.id, modelID: mid)
        doc.movie.idea = "A girl keeps a lighthouse lit through a storm."
        doc.movie.targetSeconds = 30
        model.requestStory(doc)
        await waitForGate(doc)
        guard let gate = doc.pendingGate else { say("FAIL no cost gate for the story: \(doc.error ?? "")"); return false }
        check(gate.estimate.lines.count == 1 && gate.estimate.lines[0].count == 2, "the gate lists two requests (bible, outline)")
        check(doc.job == nil && doc.movie.scenes.isEmpty, "nothing runs before Run")
        doc.pendingGate = nil
        gate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the story was written\(doc.error.map { ": \($0)" } ?? "")")
        check(doc.movie.title == "The Last Lantern" && doc.movie.characters.count == 2 && doc.movie.locations.count == 2
              && doc.movie.characters[0].goal == "keep the light burning" && !doc.movie.premise.isEmpty,
              "the bible came in: title, premise, 2 characters with goals, 2 places")
        check(doc.movie.sequences.map(\.title) == ["The Lamp Dies", "The Run"] && doc.movie.scenes.map(\.id) == ["s1", "s2"]
              && doc.movie.scenes.map(\.location) == ["lighthouse", "cliff"] && !doc.movie.scenes.contains(where: \.isWritten),
              "the outline: 2 sequences of one scene each, at their places, not written yet")
        check(MoviePlanner.sceneSeconds(for: doc.movie) == ["s1": 15, "s2": 15], "the scenes' seconds add up to the length")
        check(doc.movie.approved == [.idea] && doc.stage == .story, "the idea is approved and the story is shown")
        check(doc.sequence == "q1", "a film in sequences opens its per-sequence pages on the first")
        await snapshots("story")
        doc.show(sequence: nil)  // this pass makes the whole film at once; the sequences get their own pass below

        // Story approved → the script, a scene at a time, then the critique of each sequence.
        doc.movie.approve(.story)
        model.requestScript(doc)
        await waitForGate(doc)
        guard let scriptGate = doc.pendingGate else { say("FAIL no cost gate for the script: \(doc.error ?? "")"); return false }
        check(scriptGate.estimate.lines.first?.count == 4, "2 scenes + 2 critiques: \(scriptGate.estimate.lines.map(\.count))")
        doc.pendingGate = nil
        scriptGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the script was written\(doc.error.map { ": \($0)" } ?? "")")
        check(doc.movie.scenes.allSatisfy(\.isWritten) && doc.movie.scenes[1].dialogue.first?.text == "Grandpa, the light!",
              "every scene is written")
        check(doc.movie.scenes[0].action.contains("gutters") && doc.movie.critiqueNotes["q1"]?.contains("slower") == true,
              "the critique revised scene 1 and left its notes: \(doc.movie.critiqueNotes)")
        check(doc.stage == .script && !doc.movie.isApproved(.script), "the script waits for approval")
        await snapshots("script")

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
        // The fake's shots run 7 s against the scenes' 15: asked again, then noted on the page.
        check(doc.shotNotes.count == 2 && doc.shotNotes["s1"]?.contains("run 7 s") == true,
              "a shot list far off the scene's length is noted: \(doc.shotNotes)")
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
        check(!doc.movie.isApproved(.keyframes) && doc.movie.isApproved(.cast),
              "choosing another take reopens Keyframes only")
        doc.movie.approve(.keyframes)
        check(doc.movie.currentStage == .voices && !doc.movie.isUnlocked(.video), "Voices come next, before the shots")

        // Voices: one designed voice per speaking character, then every line cloned from it (Qwen3-TTS).
        doc.stage = .voices  // on screen as a user would have it (the Shot List page would see the shots stretch)
        check(model.speakingCharacters(doc.movie).map(\.id) == ["mara"], "Mara is the one who speaks")
        model.requestLines(doc)
        check(doc.error?.contains("Mara") == true && doc.pendingGate == nil, "lines wait for a voice")
        doc.error = nil
        model.requestVoiceSamples(doc)
        await waitForGate(doc)
        guard let voiceGate = doc.pendingGate else { say("FAIL no cost gate for the voices: \(doc.error ?? "")"); return false }
        check(voiceGate.estimate.lines.map(\.count) == [1] && voiceGate.estimate.lines[0].gpu == "L40S",
              "the voice gate counts one Qwen3-TTS job on an L40S")
        doc.pendingGate = nil
        voiceGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the voice was made\(doc.error.map { ": \($0)" } ?? "")")
        let mara = doc.movie.character("mara")
        check(mara?.selectedVoiceSampleFile.map { FileManager.default.fileExists(atPath: store.url(doc.id, relative: $0).path) } ?? false
              && (mara?.voiceSamples.first?.seconds ?? 0) > 1, "Mara has a chosen voice on disk, measured")
        model.requestLines(doc)
        await waitForGate(doc)
        guard let lineGate = doc.pendingGate else { say("FAIL no cost gate for the lines: \(doc.error ?? "")"); return false }
        check(lineGate.estimate.lines.map(\.count) == [2], "the line gate counts both lines")
        doc.pendingGate = nil
        lineGate.run()
        await waitForJob(doc)
        check(doc.error == nil, "the lines were spoken\(doc.error.map { ": \($0)" } ?? "")")
        check(doc.movie.silentLines == 0 && doc.movie.shots.flatMap(\.dialogue).allSatisfy {
            ($0.takes.first?.seconds ?? 0) > 0.5 && $0.takes.first?.gainDB != nil }, "every line has a measured take")
        check(MovieVoices.layout(doc.movie.shots[1]).first?.start == MovieVoices.leadIn, "the line starts after a lead-in")

        // Breeze TTS 2: another voice take for Mara, then a new take of each line, cloned from her chosen (Qwen3-TTS)
        // voice and directed by its delivery. The chosen takes stay; it goes back to Qwen3-TTS afterwards.
        doc.movie.voiceEngine = .breeze
        let qwenLineTakes = doc.movie.shots.flatMap(\.dialogue).map(\.selectedTake)
        model.requestVoiceSamples(doc)
        await waitForGate(doc)
        if let gate = doc.pendingGate {
            check(gate.estimate.lines.map(\.count) == [1] && gate.estimate.notes.contains { $0.contains("Breeze TTS 2") }
                  && gate.estimate.notes.contains { $0.contains("non-commercial") },
                  "the Breeze voice gate names it and its non-commercial licence")
            doc.pendingGate = nil
            gate.run()
            await waitForJob(doc)
            check(doc.error == nil && doc.movie.character("mara")?.voiceSamples.count == 2
                  && doc.movie.character("mara")?.selectedVoiceSample == mara?.selectedVoiceSample,
                  "Breeze made a second voice take; the chosen one stays\(doc.error.map { ": \($0)" } ?? "")")
        } else { check(false, "no cost gate for a Breeze voice: \(doc.error ?? "")") }
        model.requestLines(doc)
        await waitForGate(doc)
        if let gate = doc.pendingGate {
            check(gate.estimate.lines.map(\.count) == [2] && gate.estimate.notes.contains { $0.contains("delivery") },
                  "the Breeze line gate counts both lines and says it follows their delivery")
            doc.pendingGate = nil
            gate.run()
            await waitForJob(doc)
            let dialogue = doc.movie.shots.flatMap(\.dialogue)
            check(doc.error == nil && dialogue.allSatisfy { $0.takes.count == 2 && ($0.takes.last?.seconds ?? 0) > 0.5 }
                  && dialogue.map(\.selectedTake) == qwenLineTakes,
                  "Breeze spoke a measured new take of each line; the chosen ones stay\(doc.error.map { ": \($0)" } ?? "")")
        } else { check(false, "no cost gate for Breeze lines: \(doc.error ?? "")") }
        doc.movie.voiceEngine = .qwen3
        doc.movie.approve(.voices)
        check(doc.movie.currentStage == .video && doc.movie.isUnlocked(.video),
              "then the shots: \(doc.movie.currentStage) \(doc.movie.approved.map(\.rawValue).sorted())")
        // The fake voice plays the reference first, so its lines are long: the shots stretched to fit them.
        check(doc.movie.shots[1].duration >= (MovieVoices.neededDuration(doc.movie.shots[1]) ?? 99) - 0.001
              && doc.movie.shots[1].duration > 3, "a talking shot grew to fit its line: \(doc.movie.shots[1].duration) s")

        doc.stage = .video
        // Shots: a clip per shot (Wan 2.2 image to video), the continuing one from the clip before's last frame.
        model.requestClips(doc)
        await waitForGate(doc)
        guard let clipGate = doc.pendingGate else { say("FAIL no cost gate for the clips: \(doc.error ?? "")"); return false }
        // A server that runs jobs in parallel (AISTUDIO_PARALLEL_JOBS on the fake one) gets each kind's two jobs at once.
        let parallel = model.movieParallelJobs(conn.id)
        let gpus = parallel > 1 ? "2× H100" : "H100"
        check(clipGate.estimate.lines.map(\.count) == [2, 2] && clipGate.estimate.lines.allSatisfy { $0.gpu == gpus }
              && clipGate.estimate.usd > 0,
              "the clip gate counts two Wan I2V and two lip-synced S2V jobs on \(gpus), with a price: "
              + "\(clipGate.estimate.lines.map(\.gpu))")
        check(clipGate.estimate.lines[1].usd > clipGate.estimate.lines[0].usd * 2,
              "a talking clip is priced far above a plain one: \(clipGate.estimate.lines.map(\.usd))")
        if parallel > 1 {
            check(clipGate.estimate.notes.contains { $0.contains("at once") }, "it says jobs run at once")
        }
        check(clipGate.estimate.notes.contains { $0.contains("continue") } && clipGate.estimate.notes.contains { $0.contains("S2V") },
              "it says a shot continues the one before, and that talking shots are lip-synced")
        check(doc.job == nil && doc.movie.shots.allSatisfy { $0.videos.isEmpty }, "no clip before Run")
        doc.pendingGate = nil
        clipGate.run()
        let together = await waitForJob(doc)
        check(doc.error == nil, "the clips were made\(doc.error.map { ": \($0)" } ?? "")")
        // Shot 2 continues shot 1, so those two render in order; shots 3 and 4 alongside.
        if parallel > 1 { check(together > 1, "the clips rendered \(together) at a time") }
        check(doc.movie.videosAreReady, "every shot has a chosen clip")
        guard doc.movie.videosAreReady else { say("FAILED"); return false }
        let folder = store.folder(doc.id)
        let clipURLs = doc.movie.shots.map { folder.appendingPathComponent($0.selectedVideoFile!) }
        var lengths: [Double] = []
        for u in clipURLs { lengths.append(await MovieComposer.duration(of: u) ?? 0) }
        let talkLength = (doc.movie.shots[1].duration * 16).rounded(.up) / 16
        check(zip(lengths, [4.0625, talkLength, 4.0625, talkLength]).allSatisfy { abs($0 - $1) < 0.07 },
              "the clips are 65 frames (4 s shots) and, lip-synced, as long as their voice tracks: \(lengths)")
        let talkAudio = try? await AVURLAsset(url: clipURLs[1]).loadTracks(withMediaType: .audio)
        check(talkAudio?.isEmpty == false, "the lip-synced clip carries its voice")
        check(FileManager.default.fileExists(atPath: folder.appendingPathComponent(MovieShots.voiceTrackFile(
            for: doc.movie.shots[1])).path), "its voice track was written for the server")
        check(doc.movie.shots.allSatisfy { $0.selectedVideoTake?.poster.map { FileManager.default.fileExists(
            atPath: folder.appendingPathComponent($0).path) } ?? false }, "each clip has a poster")
        // The fake server: frame 0 is the start picture, the last frame the end picture.
        let end1 = try? await MovieComposer.frame(of: clipURLs[0], at: .last)
        let start2 = try? await MovieComposer.frame(of: clipURLs[1], at: .first)
        let end2 = try? await MovieComposer.frame(of: clipURLs[1], at: .last)
        let key2 = doc.movie.shots[1].selectedKeyframeFile.flatMap { loadImage(folder.appendingPathComponent($0)) }
        check(close(end1, start2), "shot 2 starts on shot 1's last frame")
        _ = (end2, key2)  // a lip-synced shot has no end frame: it follows its voice
        let start3 = try? await MovieComposer.frame(of: clipURLs[2], at: .first)
        let key3 = doc.movie.shots[2].selectedKeyframeFile.flatMap { loadImage(folder.appendingPathComponent($0)) }
        check(close(start3, key3), "a new scene's shot starts on its keyframe")
        let learned = GPUTimings.load(model.gpuTimingsURL).jobs.keys
        check(learned.contains { $0.hasPrefix("wan22-i2v@") } && learned.contains { $0.hasPrefix("wan22-s2v@") },
              "clip times were learned")
        // The other lip-sync models, on the talking shots only: a new take each, at their 25 fps, with the voice in.
        let talkingIDs = doc.movie.shots.filter { MovieShots.usesLipSync($0, in: doc.movie) }.map(\.id)
        for engine in [LipSyncEngine.longcat, .infiniteTalk] {
            doc.movie.lipSyncEngine = engine
            // The Shots page's picker re-fits the shots on a change: with InfiniteTalk the fake voices' lines are too
            // long for its talking shots, which it says (expected; a race with the sheet otherwise).
            try? await Task.sleep(for: .milliseconds(300))
            if engine == .infiniteTalk, doc.error?.contains("longer than the shot can be") == true { doc.error = nil }
            let before = doc.movie.shots.map(\.videos.count)
            model.requestClips(doc, shots: talkingIDs)
            await waitForGate(doc)
            guard let gate = doc.pendingGate else { say("FAIL no cost gate for \(engine.title): \(doc.error ?? "")"); return false }
            let counts = gate.estimate.lines.map(\.count)
            // InfiniteTalk dubs each shot's ordinary clip, so those are counted too.
            check(engine == .infiniteTalk ? counts == [2, 2] : counts.last == 2,
                  "the \(engine.title) gate counts the talking shots: \(counts)")
            check(gate.estimate.notes.contains { $0.contains(engine.title) }, "it names \(engine.title)")
            if engine == .infiniteTalk {
                // The fake voices' 9.3 s shots are past a clip's 7.5 s: the sheet says the last frame holds.
                check(gate.estimate.notes.contains { $0.contains("7.5 s") }, "it says the long shots hold their last frame")
            }
            doc.pendingGate = nil
            gate.run()
            await waitForJob(doc)
            // Switching to InfiniteTalk (the Shots page's picker) re-fits the shots and warns that these lines are
            // longer than its talking shots can be; any other error is a failure.
            if engine == .infiniteTalk, doc.error?.contains("longer than the shot can be") == true { doc.error = nil }
            check(doc.error == nil, "\(engine.title) made the talking clips\(doc.error.map { ": \($0)" } ?? "")")
            for id in talkingIDs {
                guard let i = doc.movie.shots.firstIndex(where: { $0.id == id }) else { continue }
                let shot = doc.movie.shots[i]
                check(shot.videos.count == before[i] + 1, "shot \(i + 1) has a new \(engine.title) take")
                guard let file = shot.videos.last?.file else { continue }
                let url = folder.appendingPathComponent(file)
                let seconds = await MovieComposer.duration(of: url) ?? 0
                let tracks = try? await AVURLAsset(url: url).loadTracks(withMediaType: .video)
                let fps = try? await tracks?.first?.load(.nominalFrameRate)
                check(abs(seconds - shot.duration) < 0.08 && abs(Double(fps ?? 0) - 25) < 0.5,
                      "its take is as long as the shot at 25 fps: \(seconds) s, \(fps ?? 0) fps")
                let audio = try? await AVURLAsset(url: url).loadTracks(withMediaType: .audio)
                check(audio?.isEmpty == false, "it carries the voice")
            }
            let learned = GPUTimings.load(model.gpuTimingsURL).jobs.keys
            check(learned.contains { $0.hasPrefix("\(engine.task)@") }, "\(engine.title) times were learned")
        }
        doc.movie.lipSyncEngine = .wanS2V

        // MiniMax H3: every shot again, each clip with its own sound; the talking shots from references (the
        // keyframe, Mara's sheet and her voice). The chosen takes stay; it goes back to Wan 2.2 afterwards.
        doc.movie.videoEngine = .minimaxH3
        let beforeH3 = doc.movie.shots.map(\.videos.count), chosen = doc.movie.shots.map(\.selectedVideo)
        model.requestClips(doc, shots: doc.movie.shots.map(\.id))
        await waitForGate(doc)
        if let gate = doc.pendingGate {
            let counts = gate.estimate.lines.map(\.count)
            check(counts == [2, 2] && gate.estimate.lines.allSatisfy { $0.gpu == "H200" },
                  "the H3 gate counts two clips and two talking clips on an H200: \(counts)")
            check(gate.estimate.notes.contains { $0.contains("MiniMax H3") && $0.contains("US, EU, UK and Korea") },
                  "it names H3 and its licence")
            doc.pendingGate = nil
            gate.run()
            await waitForJob(doc, seconds: 600)  // the fake encodes four full-size 24 fps clips
            check(doc.job == nil, "the H3 clips finished")
            check(doc.error == nil, "H3 made the clips\(doc.error.map { ": \($0)" } ?? "")")
            for (i, shot) in doc.movie.shots.enumerated() {
                check(shot.videos.count == beforeH3[i] + 1 && shot.selectedVideo == chosen[i],
                      "shot \(i + 1) has a new H3 take; the chosen one stays")
                guard let file = shot.videos.last?.file else { continue }
                let asset = AVURLAsset(url: folder.appendingPathComponent(file))
                let seconds = await MovieComposer.duration(of: folder.appendingPathComponent(file)) ?? 0
                let track = try? await asset.loadTracks(withMediaType: .video).first
                let fps = try? await track?.load(.nominalFrameRate)
                let size = try? await track?.load(.naturalSize)
                let audio = try? await asset.loadTracks(withMediaType: .audio)
                check(seconds >= max(5, shot.duration) - 0.05 && abs(Double(fps ?? 0) - 24) < 0.5
                      && size == CGSize(width: 1344, height: 768) && audio?.isEmpty == false,
                      "its take is 1344×768 at 24 fps, at least 5 s and as long as the shot, with sound: \(seconds) s")
            }
            // The edit with these takes: each clip's own sound, for its shot's length.
            var h3 = doc.movie
            for i in h3.shots.indices { h3.shots[i].selectedVideo = h3.shots[i].videos.last?.id }
            let files = h3.shots.compactMap(\.selectedVideoFile)
            let t = MovieTimeline(movie: h3, durations: await MovieComposer.durations(of: files, in: folder))
            check(t.sounds.filter { $0.kind == .clip }.count == 4 && !t.sounds.contains { $0.kind == .dialogue }
                  && zip(t.clips, h3.shots).allSatisfy { abs($0.duration - $1.duration) < 0.01 }
                  && t.subtitles.count == 2,
                  "its edit plays each clip's own sound for the shot's length, with the lines as subtitles")
            let learned = GPUTimings.load(model.gpuTimingsURL).jobs.keys
            check(learned.contains { $0.hasPrefix("h3-video@") } && learned.contains { $0.hasPrefix("h3-references@") },
                  "H3 times were learned")
        } else { check(false, "no cost gate for H3 clips: \(doc.error ?? "")") }
        doc.movie.videoEngine = .wan22
        // The Shots page's engine picker reopens the later stages when the engine changes: let it, then approve.
        try? await Task.sleep(for: .milliseconds(300))
        doc.movie.approve(.video)

        doc.stage = .sound
        // Sound: effects per shot (Stable Audio 3), music per scene (ACE-Step), both behind their gates.
        check(doc.movie.currentStage == .sound, "Sound comes next: \(doc.movie.currentStage)")
        doc.movie.music = MovieSound.cues(for: doc.movie)
        model.requestEffects(doc)
        await waitForGate(doc)
        guard let fxGate = doc.pendingGate else { say("FAIL no cost gate for the effects: \(doc.error ?? "")"); return false }
        check(fxGate.estimate.lines.map(\.count) == [4], "the effects gate counts four shots")
        doc.pendingGate = nil
        fxGate.run()
        await waitForJob(doc)
        check(doc.error == nil && doc.movie.shots.allSatisfy { $0.selectedSound != nil },
              "every shot has its effects\(doc.error.map { ": \($0)" } ?? "")")
        model.requestMusic(doc)
        await waitForGate(doc)
        guard let musicGate = doc.pendingGate else { say("FAIL no cost gate for the music: \(doc.error ?? "")"); return false }
        check(musicGate.estimate.lines.map(\.count) == [2], "the music gate counts two scenes")
        doc.pendingGate = nil
        musicGate.run()
        await waitForJob(doc)
        check(doc.error == nil && doc.movie.music.count == 2 && doc.movie.music.allSatisfy { ($0.selectedTake?.seconds ?? 0) >= 10 },
              "each scene has its music, at least 10 s\(doc.error.map { ": \($0)" } ?? "")")
        // A take's bar: stopped, a click halfway only moves the playhead; play starts there; a click while it plays
        // plays on from the new place; pause keeps the place.
        if let fx = doc.movie.shots[0].sounds.first(where: { $0.id == doc.movie.shots[0].selectedSound })?.file {
            let url = store.folder(doc.id).appendingPathComponent(fx)
            let player = MovieSoundPlayer.shared
            player.seek(url, fraction: 0.5)
            check(player.current == url && !player.isPlaying && abs(player.position - player.duration / 2) < 0.01,
                  "stopped, a click halfway moves the playhead without playing: \(player.position) of \(player.duration) s")
            player.toggle(url)
            check(player.isPlaying && abs(player.position - player.duration / 2) < 0.2, "play starts from there")
            player.seek(url, fraction: 0.25)
            check(player.isPlaying && abs(player.position - player.duration / 4) < 0.2, "a click while playing plays on from there")
            player.toggle(url)
            let paused = player.position
            check(!player.isPlaying && paused >= player.duration / 4 - 0.2, "pause keeps the place")
            player.restart(url)
            check(player.isPlaying && player.position < 0.2, "replay starts from the top")
            // The viewer: a shot's takes, titled, at the one asked for; opening it stops the bar's sound.
            model.openMovieViewer(doc, takes: doc.movie.shots[0].sounds, name: "Shot 1 effects",
                                  take: doc.movie.shots[0].selectedSound)
            check(model.chatViewer?.urls.first == url && model.chatViewer?.titles?.first == "Shot 1 effects · Take 1"
                  && player.current == nil, "a sound take opens in the viewer and the bar stops")
            model.chatViewer = nil
        }
        model.viewShotClips(doc, shot: doc.movie.shots[0].id)
        let clips = doc.movie.shots[0].videos
        check(model.chatViewer?.urls.count == clips.count && model.chatViewer?.titles?.last == "Shot 1 · Take \(clips.count)"
              && model.chatViewer.map { $0.urls[$0.index].lastPathComponent }
                == doc.movie.shots[0].selectedVideoFile.map { ($0 as NSString).lastPathComponent },
              "shot 1's clips open in the viewer at the chosen take: \(String(describing: model.chatViewer?.titles))")
        model.viewKeyframes(doc, shot: doc.movie.shots[0].id)
        check(model.chatViewer?.titles?.first == "Shot 1 keyframe · Take 1", "and its keyframes")
        if let c = doc.movie.characters.first {
            model.viewSheets(doc, character: c.id)
            check(model.chatViewer?.titles?.first == "\(c.name) sheet · Take 1", "and a character's sheets")
        }
        model.chatViewer = nil
        doc.movie.approve(.sound)
        check(doc.movie.currentStage == .edit, "then the edit")

        doc.stage = .edit
        // Edit: the timeline, a reorder, and the MP4 (free: on this Mac).
        let timeline = await model.movieTimeline(doc)
        check(timeline.clips.count == 4 && abs(timeline.duration - lengths.reduce(0, +)) < 0.01
              && timeline.clips[0].fadeIn == MovieTimeline.fade, "the timeline has the four clips, the first fading in")
        let ids = doc.movie.shots.map(\.id)
        model.moveShot(doc, ids[3], before: ids[2])
        check(doc.movie.shots.map(\.id) == [ids[0], ids[1], ids[3], ids[2]], "a clip dragged before another moves there")
        model.moveShot(doc, ids[2], before: ids[3])
        check(doc.movie.shots.map(\.id) == ids, "and back")
        doc.movie.shots[1].trimEnd = 1
        let trimmed = await model.movieTimeline(doc)
        check(abs(trimmed.duration - (timeline.duration - 1)) < 0.01, "a trim shortens the film")
        check(trimmed.sounds.filter { $0.kind == .dialogue }.count == 2 && trimmed.sounds.filter { $0.kind == .effects }.count == 4
              && trimmed.sounds.filter { $0.kind == .music }.count == 2 && trimmed.subtitles.count == 2,
              "the timeline has both lines, four effects, two pieces of music and two subtitles")
        if let music = trimmed.sounds.first(where: { $0.kind == .music }),
           let line = trimmed.sounds.first(where: { $0.kind == .dialogue }) {
            let points = trimmed.volume(for: music)
            let under = points.filter { $0.time >= line.start && $0.time <= line.end }.map(\.gain)
            check(!under.isEmpty && under.allSatisfy { $0 < music.gain * 0.5 }, "the music dips under the line")
        }
        let film = await model.exportMovie(doc)
        check(doc.error == nil && film != nil, "the film was exported\(doc.error.map { ": \($0)" } ?? "")")
        if let film {
            let audio = (try? await AVURLAsset(url: film).loadTracks(withMediaType: .audio)) ?? []
            check(!audio.isEmpty, "with sound")
            let srt = (try? String(contentsOf: film.deletingPathExtension().appendingPathExtension("srt"), encoding: .utf8)) ?? ""
            check(srt.hasPrefix("1\n00:00:") && srt.contains("Grandpa, the light!"), "and an .srt beside it")
        }
        doc.movie.mix.burnSubtitles = true
        let burned = await model.exportMovie(doc, to: store.folder(doc.id, .renders).appendingPathComponent("burned.mp4"))
        check(doc.error == nil && burned != nil, "an export with burned-in subtitles works\(doc.error.map { ": \($0)" } ?? "")")
        doc.movie.mix.burnSubtitles = false
        if let film {
            let asset = AVURLAsset(url: film)
            let track = try? await asset.loadTracks(withMediaType: .video).first
            let info = try? await track?.load(.naturalSize, .nominalFrameRate, .timeRange)
            check(info.map { $0.0 == CGSize(width: 832, height: 480) && abs($0.1 - 24) < 0.5
                && abs($0.2.duration.seconds - trimmed.duration) < 0.1 } ?? false,
                  "an 832×480 MP4 at 24 fps, as long as the timeline: \(String(describing: info))")
            check(film.path.hasPrefix(store.folder(doc.id, .renders).path), "in the movie's renders folder")
        }
        // A busy server: other work holds every Wan I2V GPU, so two retakes that could run at once go one at a time
        // (one always runs; the next waits until a GPU is free), and both still finish.
        if parallel > 1, let png = try? Data(contentsOf: folder.appendingPathComponent(doc.movie.shots[0].selectedKeyframeFile!)) {
            let gpu = GPUJobClient(baseURL: URL(string: url)!, token: "smoke")
            var blockers: [String] = []
            if let image = try? await gpu.uploadPNG(png) {
                for _ in 0..<parallel {
                    if let j = try? await gpu.submit(task: "wan22-i2v", params: ["steps": 60, "fast": false, "length": 5,
                                                                                 "width": 256, "height": 256],
                                                     inputs: ["image": image]) { blockers.append(j.id) }
                }
            }
            let held = (try? await gpu.info())?.freeSlots("wan22-i2v")
            check(blockers.count == parallel && held == 0, "other work holds every I2V GPU: \(blockers.count) jobs, \(String(describing: held)) free")
            let retakes = [doc.movie.shots[0].id, doc.movie.shots[2].id]
            let before = retakes.map { id in doc.movie.shots.first { $0.id == id }!.videos.count }
            model.requestClips(doc, shots: retakes)
            await waitForGate(doc)
            if let g = doc.pendingGate {
                doc.pendingGate = nil
                g.run()
                let together = await waitForJob(doc)
                check(doc.error == nil && together == 1, "on a busy server the retakes ran one at a time: \(together) at once\(doc.error.map { ", \($0)" } ?? "")")
                let after = retakes.map { id in doc.movie.shots.first { $0.id == id }!.videos.count }
                check(after == before.map { $0 + 1 }, "and both were made: \(before) → \(after)")
            } else {
                check(false, "no cost gate for the retakes: \(doc.error ?? "")")
            }
        }
        // A long film is made a sequence at a time: split this one in two, and each goes on by itself.
        if doc.movie.scenes.count >= 2 {
            let last = doc.movie.scenes[doc.movie.scenes.count - 1].id
            doc.movie.sequences = [MovieSequence(id: "q1", title: "Opening"), MovieSequence(id: "q2", title: "Ending", act: 3)]
            doc.movie.scenes[doc.movie.scenes.count - 1].sequence = "q2"
            for s in MovieStage.allCases where s.isPerSequence { doc.movie.approve(s, in: "q2") }
            check(doc.movie.isApproved(.sound) && doc.movie.editIsApprovable, "split in two sequences, both approved")
            doc.show(sequence: "q2")
            let q2 = doc.movie.shots(in: last).map(\.id)
            check(doc.scopeShots.map(\.id) == q2 && !q2.isEmpty, "sequence 2 shows its own \(q2.count) shots")
            doc.reopen(.keyframes)
            check(doc.movie.isApproved(.keyframes, in: "q1") && !doc.movie.isApproved(.keyframes, in: "q2")
                  && doc.movie.isApproved(.cast) && !doc.movie.isApproved(.edit) && !doc.movie.editIsApprovable
                  && doc.movie.currentStage(in: "q2") == .keyframes,
                  "reopening sequence 2's keyframes leaves sequence 1 and the cast: \(doc.movie.sequenceApproved)")
            doc.stage = .keyframes
            await snapshots("sequences")
            model.requestKeyframes(doc)
            await waitForGate(doc)
            if let g = doc.pendingGate {
                check(g.estimate.title.contains("Sequence 2") && g.estimate.lines.reduce(0) { $0 + $1.count } == q2.count,
                      "its cost sheet covers only sequence 2: \(g.estimate.title), \(g.estimate.lines.map(\.count))")
                doc.pendingGate = nil
            } else {
                check(false, "no cost gate for sequence 2's keyframes: \(doc.error ?? "")")
            }
            doc.job = nil
            for s in [MovieStage.keyframes, .voices, .video, .sound] { doc.approve(s) }
            check(doc.movie.editIsApprovable && doc.movie.isUnlocked(.edit), "approved again, the edit can be")
            doc.movie.budget = 0.5
            doc.stage = .idea
            await snapshots("budget")
            doc.movie.budget = nil
            doc.show(sequence: nil)
            doc.stage = .edit
        }
        doc.movie.approve(.edit)

        // Saved, closed, reopened as it was, then trashed.
        model.saveMovies()
        let id = doc.id
        let snapshot = doc.movie
        model.closeMovie(id)
        check(model.activeMovieID == nil && !model.openItems.contains(.movie(id)), "⌘W closed it")
        check(model.movieSummaries.first?.id == id && model.movieSummaries.first?.stage == .edit, "the list shows it at Edit")
        model.openMovie(id)
        check(model.activeMovie?.movie == snapshot, "it reopens exactly as saved")
        model.deleteMovie(id)
        check(!model.movieSummaries.contains { $0.id == id } && model.activeMovieID == nil, "deleted")

        model.connections.remove(conn.id)
        say(ok ? "PASS" : "FAILED")
        return ok
    }

    /// Whether two pictures look alike (8×8 averages within a few percent; H.264 and resizing blur them a little).
    private static func close(_ a: CGImage?, _ b: CGImage?) -> Bool {
        guard let a, let b, let x = tiny(a), let y = tiny(b) else { return false }
        let diff = zip(x, y).map { abs($0 - $1) }.reduce(0, +) / Double(x.count)
        if diff >= 0.08 { print("[smoke-movie]      pictures differ by \(diff)") }
        return diff < 0.08  // H.264 noise on a first frame reaches about 0.065; another picture is 0.25 or more
    }

    private static func tiny(_ image: CGImage) -> [Double]? {
        var px = [UInt8](repeating: 0, count: 8 * 8 * 4)
        guard let ctx = CGContext(data: &px, width: 8, height: 8, bitsPerComponent: 8, bytesPerRow: 32,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .medium
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: 8, height: 8))
        return px.enumerated().filter { $0.offset % 4 != 3 }.map { Double($0.element) / 255 }
    }

    private static func loadImage(_ url: URL) -> CGImage? {
        CGImageSourceCreateWithURL(url as CFURL, nil).flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) }
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
