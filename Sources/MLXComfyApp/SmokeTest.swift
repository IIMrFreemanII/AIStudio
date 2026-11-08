import Foundation
import GraphRuntime
import MLXDiffusionCore
import NodesStandard

/// `MLXComfyApp --smoke-test "<prompt>"` queues one generation through the same path as the Generate button
/// (AppModel → Engine → events), reports what the UI received, and exits.
@MainActor
enum SmokeTest {
    static var requestedPrompt: String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--smoke-test"), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func argument(_ name: String) -> String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    /// `--smoke-assist "<idea>"` (+ `--smoke-image`): Enhance the idea, then Describe the image, through the same
    /// AppModel path as the buttons, checking streaming, the final prompt, undo, and memory before first use.
    static func runAssistant(_ model: AppModel, idea: String) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        // Explicit undo groups, as a window's event loop would provide: registering outside a group throws an
        // Objective-C exception, which must never unwind through a Swift task.
        let undo = UndoManager()
        undo.groupsByEvent = false
        // Load the model first so the memory comparison isolates the assistant.
        model.preloadModel()
        while !(await model.engine.isLoaded(modelPath: model.modelPath)) {
            if model.lastError != nil { say("load failed: \(model.lastError!)"); return false }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        let before = Engine.activeMemory()
        say(String(format: "model loaded (weights map in lazily), memory %.2f GB", Double(before) / 1e9))
        guard let te = model.firstNode(ofType: "TextEncode") else { say("no TextEncode"); return false }
        model.setLiteral(.string(idea), node: te.id, input: "prompt", undoManager: nil)

        func run(_ mode: PromptAssistant.Mode) async -> String? {
            var updates = 0
            var last = ""
            let started = Date()
            let result: String? = await withCheckedContinuation { cont in
                undo.beginUndoGrouping()
                model.runAssistant(mode, undoManager: undo) { r in
                    undo.endUndoGrouping()
                    cont.resume(returning: r)
                }
                Task { @MainActor in
                    while model.assistantMode != nil {
                        let now = model.literal(te.id, "prompt")?.description ?? ""
                        if now != last { updates += 1; last = now }
                        try? await Task.sleep(nanoseconds: 50_000_000)
                    }
                }
            }
            say("\(mode.rawValue): \(String(format: "%.1fs", Date().timeIntervalSince(started))), "
                + "\(updates) live updates seen")
            say("\(mode.rawValue) result: \(result ?? "nil (error: \(model.lastError ?? "none"))")")
            return result
        }

        guard let enhanced = await run(.enhance), !enhanced.isEmpty else { return false }
        let after = Engine.activeMemory()
        say(String(format: "memory after first use %.2f GB (+%.2f GB)", Double(after) / 1e9,
                   Double(after - before) / 1e9))
        let shown = model.literal(te.id, "prompt")?.description ?? ""
        say("prompt field matches result: \(shown == enhanced)")
        undo.undo()
        let restored = model.literal(te.id, "prompt")?.description ?? ""
        say("undo restores the idea: \(restored == idea)")
        guard shown == enhanced, restored == idea else { return false }

        if let image = argument("--smoke-image") {
            model.setReferenceImage(URL(fileURLWithPath: image), undoManager: nil)
            guard let described = await run(.describe), !described.isEmpty else { return false }
            // An edit instruction with the image attached.
            model.setLiteral(.string("make it winter"), node: te.id, input: "prompt", undoManager: nil)
            guard let edit = await run(.enhance), !edit.isEmpty else { return false }
        }
        // Stop mid-way: the partial text stays and one undo restores the previous prompt.
        model.setLiteral(.string(idea), node: te.id, input: "prompt", undoManager: nil)
        undo.beginUndoGrouping()
        model.runAssistant(.enhance, undoManager: undo)
        try? await Task.sleep(nanoseconds: 3_500_000_000)
        model.stopAssistant()
        while model.assistantMode != nil { try? await Task.sleep(nanoseconds: 50_000_000) }
        undo.endUndoGrouping()
        let partial = model.literal(te.id, "prompt")?.description ?? ""
        say("stopped: \(partial.count) chars kept")
        undo.undo()
        say("undo after stop restores the idea: \((model.literal(te.id, "prompt")?.description ?? "") == idea)")
        return true
    }

    static func run(_ model: AppModel, prompt: String) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        if let image = argument("--smoke-image") {
            model.setReferenceImage(URL(fileURLWithPath: image), undoManager: nil)
            say("reference image \(image)")
        }
        if let mask = argument("--smoke-mask") {
            model.setMask(URL(fileURLWithPath: mask), undoManager: nil)
            say("mask \(mask); graph nodes: \(model.graph.nodes.map(\.id).joined(separator: ", "))")
        }
        if let pad = argument("--smoke-outpaint") {
            // l,t,r,b in pixels
            let v = pad.split(separator: ",").compactMap { Int($0) } + [0, 0, 0, 0]
            model.setOutpaint(left: v[0], top: v[1], right: v[2], bottom: v[3], undoManager: nil)
            say("outpaint \(pad)")
        }
        if CommandLine.arguments.contains("--smoke-hires") {
            model.setHiRes(true, undoManager: nil)
            say("hi-res fix on")
        }
        if model.lastError != nil { say("setup error: \(model.lastError!)") }
        say("model folder \(model.modelPath)")
        say("output folder \(model.outputDirectory)")
        guard let p = model.firstNode(ofType: "TextEncode") else { say("no TextEncode node"); return false }
        model.setLiteral(.string(prompt), node: p.id, input: "prompt", undoManager: nil)
        model.randomizeSeed = false
        let historyBefore = model.history.count
        let start = Date()
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return false }

        var lastStep = -1
        while true {
            try? await Task.sleep(nanoseconds: 200_000_000)
            guard let cur = model.queue.first(where: { $0.id == item.id }) else { return false }
            if case .running(let step, let total) = cur.status, step != lastStep, total > 0 {
                lastStep = step
                say("step \(step)/\(total), previews so far \(model.previewCount), "
                    + String(format: "memory %.1f GB", Double(model.activeMemory) / 1e9))
            }
            if cur.isFinished {
                try? await Task.sleep(nanoseconds: 300_000_000)  // let finish() publish results
                let states = model.nodeStates.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: ", ")
                say("status \(cur.status) in " + String(format: "%.1fs", Date().timeIntervalSince(start)))
                say("node states: \(states)")
                say("previews received: \(model.previewCount)")
                say("displayed image: \(model.displayedImage.map { "\($0.width)x\($0.height)" } ?? "none")")
                say("saved: \(model.displayedURL?.path ?? "none")")
                say("history \(historyBefore) → \(model.history.count)")
                if case .done = cur.status, model.displayedImage != nil, model.previewCount > 0 { return true }
                say("error: \(model.lastError ?? "none")")
                return false
            }
        }
    }
}

extension SmokeTest {
    /// `--smoke-xy`: a 2×2 plot (seed × steps) at 512² with a choice template and a style, through enqueuePlot.
    /// Checks the cells' expanded prompts, the collapsed progress, and the saved grid PNG with its metadata.
    static func runPlot(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        guard let te = model.firstNode(ofType: "TextEncode"), let sampler = model.firstNode(ofType: "Sampler"),
              let latent = model.firstNode(ofType: "EmptyLatent") else { say("graph is not the standard one"); return false }
        model.setLiteral(.string("a {red|arctic} fox sitting in fresh snow"), node: te.id, input: "prompt", undoManager: nil)
        model.setLiteral(.int(512), node: latent.id, input: "width", undoManager: nil)
        model.setLiteral(.int(512), node: latent.id, input: "height", undoManager: nil)
        let previousStyle = model.selectedStyleName
        model.selectedStyleName = "Watercolor"
        defer { model.selectedStyleName = previousStyle }  // the real app persists this setting
        do {
            let spec = PlotSpec(x: try PlotAxis.parse("1, 2", type: .int, node: sampler.id, input: "seed"),
                                y: try PlotAxis.parse("2, 4", type: .int, node: sampler.id, input: "steps"))
            let historyBefore = model.history.count
            let start = Date()
            model.enqueuePlot(spec)
            let cells = model.queue.filter { $0.groupID != nil }
            say("queued \(cells.count) cells: " + cells.map(\.title).joined(separator: " | "))
            for c in cells {
                let p = c.graph[te.id].flatMap { n -> String? in
                    if case .value(.string(let s))? = n.inputs["prompt"] { return s }; return nil
                } ?? "?"
                say("cell \(c.cell!) prompt: \(p)")
                guard !p.contains("{"), p.contains("watercolor"),
                      c.graph.metadata?["prompt_template.\(te.id)"] == "a {red|arctic} fox sitting in fresh snow",
                      !c.graph.nodes.contains(where: { $0.type == "SaveImage" }) else {
                    say("cell graph not prepared as expected"); return false
                }
            }
            guard let gid = cells.first?.groupID else { return false }
            var lastDone = -1
            while model.plots[gid] != nil {
                let p = model.plotProgress(gid)
                if p.done != lastDone { lastDone = p.done; say("progress \(p.done)/\(p.total)") }
                try? await Task.sleep(nanoseconds: 300_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            say(String(format: "plot finished in %.1fs", Date().timeIntervalSince(start)))
            guard let url = model.displayedURL, url.lastPathComponent.hasPrefix("plot_"),
                  let img = model.displayedImage else { say("no grid shown: \(model.lastError ?? "none")"); return false }
            say("grid \(img.width)x\(img.height) saved to \(url.path)")
            let text = ImageArray.readPNGText(url)
            let restored = AppModel.embeddedGraph(in: url)
            say("grid metadata: graph \(restored != nil), plot \(text["mlxcomfy.plot"]?.isEmpty == false), "
                + "history \(historyBefore) → \(model.history.count)")
            return restored?.metadata?["xy_plot"] != nil && model.history.count == historyBefore + 1
        } catch {
            say("axis error: \(error)")
            return false
        }
    }
}

extension SmokeTest {
    /// `--smoke-media`: runs examples/media-test.json through the queue (TestPattern + tones → mix → video with
    /// soundtrack, plus an M4A), checks the files with AVFoundation, restores the graph from the MP4, and checks that
    /// a big new resource evicts the image model and that it loads again.
    static func runMedia(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        guard var g = try? Graph.load(cwd.appendingPathComponent("examples/media-test.json")) else {
            say("examples/media-test.json not found (run from the repo)"); return false
        }
        for id in ["save_video", "save_audio"] { try? g.setValue(.string(model.outputDirectory), node: id, input: "directory") }
        model.setGraph(g, actionName: "Media Test", undoManager: nil)
        let historyBefore = model.history.count
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return false }
        while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        guard case .done(let s)? = model.queue.first(where: { $0.id == item.id })?.status else {
            say("job failed: \(model.lastError ?? "?")"); return false
        }
        say(String(format: "job done in %.2fs; history %d → %d", s, historyBefore, model.history.count))
        let videos = model.history.filter { $0.kind == .video && $0.url.lastPathComponent.hasPrefix("media-test") }
        let audios = model.history.filter { $0.kind == .audio && $0.url.lastPathComponent.hasPrefix("media-test") }
        guard let video = videos.first?.url, let audio = audios.first?.url else { say("outputs missing from history"); return false }
        say("displayed: \(model.displayedMedia.map { "\($0.kind.rawValue) \($0.url.lastPathComponent)" } ?? "none")")

        guard let clip = try? await MediaIO.readVideo(video) else { say("can't read \(video.path)"); return false }
        say(String(format: "video %dx%d, %d frames at %.1f fps, audio %.2fs", clip.width, clip.height, clip.frameCount,
                   clip.fps, clip.audio?.duration ?? 0))
        let sound = try? await MediaIO.readAudio(audio)
        say(String(format: "m4a %.2fs at %d Hz, %d channels", sound?.duration ?? 0, sound?.sampleRate ?? 0, sound?.channels ?? 0))
        let meta = await MediaIO.readMetadata(video)
        guard clip.frameCount == 48, abs((clip.audio?.duration ?? 0) - 2) < 0.1, abs((sound?.duration ?? 0) - 3) < 0.1,
              meta[StandardNodes.graphMetadataKey] != nil else { say("media check failed"); return false }

        // Restore the graph from the MP4.
        model.newGraph(undoManager: nil)
        _ = model.open(video, undoManager: nil)
        for _ in 0..<50 where model.graph["save_video"] == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
        say("graph restored from MP4: \(model.graph["save_video"] != nil && model.graph["attach"] != nil)")
        guard model.graph["attach"] != nil else { return false }

        // Eviction: the image model's estimate plus 30 GB exceeds the budget, so it must be unloaded.
        model.newGraph(undoManager: nil)
        let path = model.modelPath
        _ = try? await model.engine.preload(modelPath: path)
        let before = model.engine.loadedModelsSnapshot().map(\.name)
        try? await model.engine.debugReserve("fake:30GB", bytes: 30_000_000_000)
        let during = model.engine.loadedModelsSnapshot().map(\.name)
        await model.engine.unload("fake:30GB")
        _ = try? await model.engine.preload(modelPath: path)
        let after = model.engine.loadedModelsSnapshot().map(\.name)
        say("loaded before: \(before), after reserving 30 GB: \(during), after reloading: \(after)")
        return !before.isEmpty && during.isEmpty && after == before
    }

    /// `--smoke-video`: the Video mode end to end in the app. Preloads the image model, switches to Video (graph
    /// swap), generates a short clip (the image model must be evicted to fit), checks previews, history and the
    /// file, then switches back to Image and expects the image graph again.
    static func runVideo(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let ltx = ModelStore().modelsDirectory.appendingPathComponent("ltx-2.3")
        guard ModelRegistry.className(at: ltx) == "LTX2Pipeline" else { say("no LTX-2.3 at \(ltx.path)"); return false }
        _ = try? await model.engine.preload(modelPath: model.modelPath)
        let before = model.engine.loadedModelsSnapshot().map(\.name)
        model.setSimpleMode(.video, videoModelPath: ltx.path, undoManager: nil)
        guard model.simpleMode == .video, let latent = model.firstNode(ofType: "EmptyVideoLatent") else {
            say("video graph not set up"); return false
        }
        model.mutate("Small", undoManager: nil) { g in
            try? g.setValue(.int(512), node: latent.id, input: "width")
            try? g.setValue(.int(320), node: latent.id, input: "height")
            try? g.setValue(.double(2), node: latent.id, input: "seconds")
        }
        model.previewCount = 0
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return false }
        while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        guard case .done(let secs)? = model.queue.first(where: { $0.id == item.id })?.status else {
            say("job failed: \(model.lastError ?? "?")"); return false
        }
        let after = model.engine.loadedModelsSnapshot().map(\.name)
        say(String(format: "job done in %.1fs, previews %d", secs, model.previewCount))
        say("loaded before: \(before), after: \(after)")
        guard let video = model.history.first(where: { $0.kind == .video && $0.url.lastPathComponent.hasPrefix("video") })?.url,
              let clip = try? await MediaIO.readVideo(video) else { say("no video in history"); return false }
        say(String(format: "%@: %dx%d, %d frames, audio %.2fs; displayed %@", video.lastPathComponent, clip.width,
                   clip.height, clip.frameCount, clip.audio?.duration ?? 0,
                   model.displayedMedia.map { $0.kind.rawValue } ?? "none"))
        model.setSimpleMode(.image, videoModelPath: ltx.path, undoManager: nil)
        let back = model.simpleMode == .image && model.firstNode(ofType: "Sampler") != nil
        say("back to image graph: \(back)")
        return model.previewCount > 0 && clip.frameCount == 49 && (clip.audio?.duration ?? 0) > 1.9
            && after.contains { $0.contains("ltx-2.3") } && !after.contains { $0.contains("qwen") } && back
            && model.displayedMedia?.kind == .video
    }

    /// `--smoke-sound`: the Sound mode end to end: switch modes (graph swap), generate a short clip, check history
    /// and the file, then switch back to Image.
    static func runSound(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("stable-audio-open-small")
        guard ModelRegistry.className(at: dir) == "StableAudioPipeline" else { say("no Stable Audio at \(dir.path)"); return false }
        model.setSimpleMode(.sound, videoModelPath: nil, soundModelPath: dir.path, undoManager: nil)
        guard model.simpleMode == .sound, let gen = model.firstNode(ofType: "TextToAudio") else {
            say("sound graph not set up"); return false
        }
        model.mutate("Short", undoManager: nil) { g in try? g.setValue(.double(4), node: gen.id, input: "seconds") }
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return false }
        while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        guard case .done(let secs)? = model.queue.first(where: { $0.id == item.id })?.status else {
            say("job failed: \(model.lastError ?? "?")"); return false
        }
        guard let url = model.history.first(where: { $0.kind == .audio && $0.url.lastPathComponent.hasPrefix("sound") })?.url,
              let clip = try? await MediaIO.readAudio(url) else { say("no sound in history"); return false }
        let meta = await MediaIO.readMetadata(url)
        say(String(format: "job done in %.1fs: %@ %.2fs at %d Hz, %d ch, graph embedded %@; displayed %@", secs,
                   url.lastPathComponent, clip.duration, clip.sampleRate, clip.channels,
                   meta[StandardNodes.graphMetadataKey] != nil ? "yes" : "no",
                   model.displayedMedia.map { $0.kind.rawValue } ?? "none"))
        model.setSimpleMode(.image, videoModelPath: nil, undoManager: nil)
        let back = model.simpleMode == .image && model.firstNode(ofType: "Sampler") != nil
        say("back to image graph: \(back)")
        return abs(clip.duration - 4) < 0.05 && clip.sampleRate == 44100 && clip.channels == 2
            && meta[StandardNodes.graphMetadataKey] != nil && model.displayedMedia?.kind == .audio && back
    }

    /// `--smoke-music`: the Music mode end to end: switch modes, generate a short instrumental (planner on), check the
    /// file and history, then switch back to Image.
    static func runMusic(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("ace-step-1.5")
        guard ModelRegistry.className(at: dir) == "ACEStepPipeline" else { say("no ACE-Step at \(dir.path)"); return false }
        model.setSimpleMode(.music, videoModelPath: nil, musicModelPath: dir.path, undoManager: nil)
        guard model.simpleMode == .music, let gen = model.firstNode(ofType: "TextToMusic") else {
            say("music graph not set up"); return false
        }
        model.mutate("Short", undoManager: nil) { g in try? g.setValue(.double(10), node: gen.id, input: "seconds") }
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return false }
        var maxTotal = 0
        while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
            if case .running(_, let total)? = model.queue.first(where: { $0.id == item.id })?.status { maxTotal = max(maxTotal, total) }
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        try? await Task.sleep(nanoseconds: 500_000_000)
        guard case .done(let secs)? = model.queue.first(where: { $0.id == item.id })?.status else {
            say("job failed: \(model.lastError ?? "?")"); return false
        }
        guard let url = model.history.first(where: { $0.kind == .audio && $0.url.lastPathComponent.hasPrefix("music") })?.url,
              let clip = try? await MediaIO.readAudio(url) else { say("no music in history"); return false }
        let meta = await MediaIO.readMetadata(url)
        let title = model.history.first { $0.url == url }?.prompt ?? "?"
        say(String(format: "job done in %.1fs (progress total %d): %@ %.2fs at %d Hz, %d ch, graph embedded %@; displayed %@; history prompt %@",
                   secs, maxTotal, url.lastPathComponent, clip.duration, clip.sampleRate, clip.channels,
                   meta[StandardNodes.graphMetadataKey] != nil ? "yes" : "no",
                   model.displayedMedia.map { $0.kind.rawValue } ?? "none", title))
        model.setSimpleMode(.image, videoModelPath: nil, undoManager: nil)
        let back = model.simpleMode == .image && model.firstNode(ofType: "Sampler") != nil
        say("back to image graph: \(back)")
        return abs(clip.duration - 10) < 0.05 && clip.sampleRate == 48000 && clip.channels == 2 && maxTotal > 8
            && meta[StandardNodes.graphMetadataKey] != nil && model.displayedMedia?.kind == .audio && back
    }

    /// `--smoke-speech`: the Speech mode end to end: a voice preview (shown, not saved), then a generated clip in
    /// history with the graph embedded, then back to Image.
    static func runSpeech(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("kokoro-82m")
        guard ModelRegistry.className(at: dir) == "KokoroPipeline" else { say("no Kokoro at \(dir.path)"); return false }
        model.setSimpleMode(.speech, videoModelPath: nil, speechModelPath: dir.path, undoManager: nil)
        guard model.simpleMode == .speech, model.firstNode(ofType: "TextToSpeech") != nil else {
            say("speech graph not set up"); return false
        }
        func wait(_ id: UUID) async -> QueueItem.Status? {
            while model.queue.first(where: { $0.id == id })?.isFinished == false {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            return model.queue.first { $0.id == id }?.status
        }
        let historyBefore = model.history.count
        model.previewVoice("bm_george", speed: 1, modelPath: dir.path)
        guard let preview = model.queue.last, case .done? = await wait(preview.id) else {
            say("preview failed: \(model.lastError ?? "?")"); return false
        }
        let previewShown = model.displayedMedia?.kind == .audio
            && model.displayedMedia?.url.path.hasPrefix(previewDirectory.path) == true
        say("preview shown \(previewShown), history unchanged \(model.history.count == historyBefore)")
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last, case .done(let secs)? = await wait(item.id) else {
            say("job failed: \(model.lastError ?? "?")"); return false
        }
        guard let url = model.history.first(where: { $0.kind == .audio && $0.url.lastPathComponent.hasPrefix("speech") })?.url,
              let clip = try? await MediaIO.readAudio(url) else { say("no speech in history"); return false }
        let meta = await MediaIO.readMetadata(url)
        say(String(format: "job done in %.1fs: %@ %.2fs at %d Hz, %d ch, graph embedded %@; displayed %@", secs,
                   url.lastPathComponent, clip.duration, clip.sampleRate, clip.channels,
                   meta[StandardNodes.graphMetadataKey] != nil ? "yes" : "no",
                   model.displayedMedia.map { $0.kind.rawValue } ?? "none"))
        model.setSimpleMode(.image, videoModelPath: nil, undoManager: nil)
        let back = model.simpleMode == .image && model.firstNode(ofType: "Sampler") != nil
        say("back to image graph: \(back)")
        return previewShown && model.history.count > historyBefore && clip.duration > 4 && clip.sampleRate == 24000
            && meta[StandardNodes.graphMetadataKey] != nil && model.displayedMedia?.kind == .audio && back
    }
}
