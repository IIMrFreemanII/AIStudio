import AppKit
import Foundation
import GraphRuntime
import ImageIO
import MLX
import AIStudioLibrary
import MLXDiffusionCore
import NodesStandard

/// `AIStudioApp --smoke-test "<prompt>"` queues one generation through the same path as the Generate button
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
        model.setStyle("Watercolor", node: te.id, undoManager: nil)
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
            say("grid metadata: graph \(restored != nil), plot \(text["aistudio.plot"]?.isEmpty == false), "
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

        // Restore the graph from the MP4, in a new tab.
        let tabsBefore = model.tabs.count
        model.newTab()
        _ = model.open(video)
        for _ in 0..<50 where model.graph["save_video"] == nil { try? await Task.sleep(nanoseconds: 100_000_000) }
        say("graph restored from MP4 in a new tab: \(model.graph["save_video"] != nil && model.graph["attach"] != nil), "
            + "tabs \(tabsBefore) → \(model.tabs.count)")
        guard model.graph["attach"] != nil, model.tabs.count == tabsBefore + 2 else { return false }

        // Eviction: the image model's estimate plus 30 GB exceeds the budget, so it must be unloaded.
        model.newTab()
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
        let imageTab = model.activeTabID
        guard let tab = openPreset(model, "ltx-t2v", expecting: ltx),
              let latent = model.firstNode(ofType: "EmptyVideoLatent") else {
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
        let onNode = tab.nodeResults["save"]?.kind == .video
        say("result on the Save Video node: \(onNode); saved in \(video.deletingLastPathComponent().lastPathComponent)/ for tab “\(tab.name)”")
        let shown = model.displayedMedia?.kind == .video
        model.activate(imageTab)
        let back = model.firstNode(ofType: "Sampler") != nil
        say("back to the image tab: \(back)")
        return model.previewCount > 0 && clip.frameCount == 49 && (clip.audio?.duration ?? 0) > 1.9
            && after.contains { $0.contains("ltx-2.3") } && !after.contains { $0.contains("qwen") } && back
            && shown && onNode
    }

    /// `--smoke-wan`: the Video mode with Wan 2.2 T2V (fast, 512×288, 2 s): switch modes and model, generate, check
    /// previews, the frame count, that the clip is silent, that Wan replaced the other models, then switch back.
    static func runWan(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let wan = ModelStore().modelsDirectory.appendingPathComponent("wan-2.2-t2v")
        guard ModelRegistry.className(at: wan) == "WanPipeline" else { say("no Wan 2.2 T2V at \(wan.path)"); return false }
        _ = try? await model.engine.preload(modelPath: model.modelPath)
        let before = model.engine.loadedModelsSnapshot().map(\.name)
        let imageTab = model.activeTabID
        guard let tab = openPreset(model, "wan-t2v", expecting: wan),
              let latent = model.firstNode(ofType: "EmptyVideoLatent"),
              VideoFamily.of(path: wan.path) == .wan else { say("video graph not set up"); return false }
        model.mutate("Small", undoManager: nil) { g in
            try? g.setValue(.int(512), node: latent.id, input: "width")
            try? g.setValue(.int(288), node: latent.id, input: "height")
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
        say(String(format: "%@: %dx%d, %d frames at %.0f fps, audio %.2fs; displayed %@", video.lastPathComponent,
                   clip.width, clip.height, clip.frameCount, clip.fps, clip.audio?.duration ?? 0,
                   model.displayedMedia.map { $0.kind.rawValue } ?? "none"))
        let onNode = tab.nodeResults["save"]?.kind == .video
        let shown = model.displayedMedia?.kind == .video
        model.activate(imageTab)
        let back = model.firstNode(ofType: "Sampler") != nil
        say("result on the node: \(onNode); back to the image tab: \(back)")
        return model.previewCount > 0 && clip.frameCount == 33 && clip.width == 512 && clip.height == 288
            && (clip.audio?.duration ?? 0) < 0.1 && after.contains { $0.contains("wan-2.2-t2v") }
            && !after.contains { $0.contains("qwen") } && back && shown && onNode
    }

    /// `--smoke-flux`: the Image mode with FLUX.1 [dev] (512², 8 steps): switch model (Qwen evicted), apply a synthetic
    /// no-op LoRA in kohya's fused layout through the LoRA section, generate, then redraw the result image-to-image
    /// (at about 1 MP), and switch back to Qwen (image-to-image chain removed, reference editing again).
    /// `--smoke-library-model <link> [--family qwen-image-2.1] [--expect-same]`: a fine-tune from a link (FLUX.1 dev by
    /// default), through the library: Add (This Mac) → download → convert → a picture with it and with the base
    /// model, same seed. Uses a throwaway models folder that links your installed base model (never modified); the
    /// pictures are kept in it for a look, the rest removed. `--expect-same`: the link is the base model's own
    /// transformer, so the pictures should match (a check of the conversion itself).
    static func runLibraryModel(_ model: AppModel, link: String) async -> Bool {
        func say(_ s: String) { print("[smoke-library] \(s)"); fflush(stdout) }
        guard let lib = model.store?.library else { say("no library"); return false }
        let family = argument("--family") ?? "flux.1"
        let expectSame = CommandLine.arguments.contains("--expect-same")
        guard let baseName = LinkLibrary.macModelBases[family] else { say("no Mac base for \(family)"); return false }
        let real = ModelPaths.directory
        let base = real.appendingPathComponent(baseName)
        guard ModelStore.isComplete(base), ModelRegistry.className(at: base) != nil else {
            say("no \(LoRASources.familyName(family)) at \(base.path)"); return false
        }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-library-model")
        try? FileManager.default.removeItem(at: tmp)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        try? FileManager.default.createSymbolicLink(at: tmp.appendingPathComponent(baseName), withDestinationURL: base)
        ModelPaths.directory = tmp
        lib.reload()
        defer { ModelPaths.directory = real; lib.reload() }

        guard let remote = try? await LoRASources.resolve(link, fits: LoRASources.civitaiBaseModels(for: family)) else {
            say("couldn't resolve \(link)"); return false
        }
        say("resolved: \(remote.kind) \(remote.fileName) \(remote.bytes.map { String(format: "%.1f GB", Double($0) / 1e9) } ?? "?")")
        let began = Date()
        let id = lib.add(remote, link: link, title: "Smoke fine-tune", family: family, mac: true, servers: [])
        var last = ""
        while true {
            guard let item = lib.item(id) else { say("entry vanished"); return false }
            let state = lib.macState(item)
            let line = "\(state)"
            if line != last && (!line.contains("Downloading") || Int(Date().timeIntervalSince(began)) % 30 == 0) {
                say(String(format: "[%4.0fs] %@", Date().timeIntervalSince(began), line)); last = line
            }
            if state == .installed { break }
            if case .failed(let why) = state { say("install failed: \(why)"); return false }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        guard let folder = lib.item(id).flatMap({ Library.localURL($0, models: tmp) }) else { say("no folder"); return false }
        say(String(format: "installed in %.0fs: %@ (%@)", Date().timeIntervalSince(began), folder.path,
                   ModelRegistry.className(at: folder) ?? "?"))
        let inMenu = lib.macModels().contains { $0.url.standardizedFileURL == folder.standardizedFileURL }

        func render(_ path: String) async -> URL? {
            model.setImageModel(path: path, undoManager: nil)
            guard let latent = model.firstNode(ofType: "EmptyLatent"), let sampler = model.firstNode(ofType: "Sampler"),
                  let te = model.firstNode(ofType: "TextEncode") else { say("image graph not set up"); return nil }
            model.mutate("Smoke", undoManager: nil) { g in
                try? g.setValue(.int(768), node: latent.id, input: "width")
                try? g.setValue(.int(768), node: latent.id, input: "height")
                if family == "flux.1" { try? g.setValue(.int(20), node: sampler.id, input: "steps") }
                try? g.setValue(.int(42), node: sampler.id, input: "seed")
                try? g.setValue(.string("portrait photo of a woman with freckles by a window, soft daylight"), node: te.id, input: "prompt")
            }
            model.randomizeSeed = false
            let t0 = Date()
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return nil }
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard case .done? = model.queue.first(where: { $0.id == item.id })?.status,
                  let url = model.history.first(where: { $0.kind == .image })?.url else {
                say("render failed: \(model.lastError ?? "?")"); return nil
            }
            let keep = tmp.appendingPathComponent((path as NSString).lastPathComponent + ".png")
            try? FileManager.default.removeItem(at: keep)
            try? FileManager.default.copyItem(at: url, to: keep)
            say(String(format: "%@: %.0fs → %@", (path as NSString).lastPathComponent, Date().timeIntervalSince(t0), keep.path))
            return keep
        }
        guard let tuned = await render(folder.path), let plain = await render(base.path) else { return false }
        let a = (try? ImageArray.load(tuned)).map { ImageArray.array(from: $0) }, b = (try? ImageArray.load(plain)).map { ImageArray.array(from: $0) }
        let diff = (a != nil && b != nil) ? abs(a! - b!).mean().item(Float.self) : 0
        say(String(format: "mean difference fine-tune vs base: %.4f (%@); in the Load Model menu: %@", diff,
                   expectSame ? "should be tiny" : "should differ", inMenu ? "yes" : "no"))
        lib.removeFromMac(id)
        let removed = !FileManager.default.fileExists(atPath: folder.path)
        lib.remove(id)
        say("removed from this Mac: \(removed); base untouched: \(ModelStore.isComplete(base))")
        return (expectSame ? diff < 0.02 : diff > 0.01) && inMenu && removed && ModelStore.isComplete(base)
    }

    /// `--smoke-wan-lora <link>`: a Wan 2.2 LoRA (a high/low pair or one file) from a link, through the library onto
    /// this Mac, then the Wan text-to-video preset at a small size with and without it (same seed), added with Add to
    /// Workflow › LoRA. A throwaway models folder links your installed Wan 2.2 T2V; the clips are kept in it.
    static func runWanLoRA(_ model: AppModel, link: String) async -> Bool {
        func say(_ s: String) { print("[smoke-wan-lora] \(s)"); fflush(stdout) }
        guard let lib = model.store?.library, let preset = model.preset(id: "wan-t2v") else { say("no library or preset"); return false }
        let real = ModelPaths.directory
        let base = real.appendingPathComponent("wan-2.2-t2v")
        guard ModelRegistry.className(at: base) == "WanPipeline" else { say("no Wan 2.2 T2V at \(base.path)"); return false }
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-wan-lora")
        try? FileManager.default.removeItem(at: tmp)
        try? FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        try? FileManager.default.createSymbolicLink(at: tmp.appendingPathComponent("wan-2.2-t2v"), withDestinationURL: base)
        ModelPaths.directory = tmp
        lib.reload()
        defer { ModelPaths.directory = real; lib.reload() }

        guard let remote = try? await LoRASources.resolve(link, fits: LoRASources.civitaiBaseModels(for: "wan-2.2-t2v")) else {
            say("couldn't resolve \(link)"); return false
        }
        say("resolved: \(remote.fileName) + \(remote.companion?.fileName ?? "-") (expert \(remote.expert ?? "both"))")
        let id = lib.add(remote, link: link, title: "Smoke Wan LoRA", family: "wan-2.2-t2v", mac: true, servers: [])
        let began = Date()
        while true {
            guard let item = lib.item(id) else { return false }
            let state = lib.macState(item)
            if state == .installed { break }
            if case .failed(let why) = state { say("install failed: \(why)"); return false }
            try? await Task.sleep(nanoseconds: 500_000_000)
        }
        let pairOnDisk = lib.item(id).flatMap { Library.companionURL($0, models: tmp) } != nil
        say(String(format: "installed in %.0fs; the pair's second file on disk: %@", Date().timeIntervalSince(began),
                   pairOnDisk ? "yes" : "no"))

        let doc = model.openPreset(preset)
        model.mutate("Smoke", undoManager: nil) { g in
            try? g.setValue(.int(480), node: "latent", input: "width")
            try? g.setValue(.int(272), node: "latent", input: "height")
            try? g.setValue(.double(1), node: "latent", input: "seconds")
            try? g.setValue(.int(7), node: "sampler", input: "seed")
        }
        model.randomizeSeed = false
        func render(_ name: String) async -> URL? {
            let t0 = Date()
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { return nil }
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard case .done? = model.queue.first(where: { $0.id == item.id })?.status, let url = doc.nodeResults["save"]?.url else {
                say("\(name) failed: \(model.lastError ?? "?")"); return nil
            }
            let keep = tmp.appendingPathComponent("\(name).mp4")
            try? FileManager.default.copyItem(at: url, to: keep)
            say(String(format: "%@: %.0fs → %@", name, Date().timeIntervalSince(t0), keep.path))
            return keep
        }
        guard let plain = await render("without-lora") else { return false }
        guard model.canAdd(.lora) else { say("Add to Workflow › LoRA unavailable on the Wan graph"); return false }
        model.add(.lora)
        let node = model.selection.first ?? ""
        model.setLiteral(.string(id), node: node, input: "lora", undoManager: nil)
        guard let tuned = await render("with-lora") else { return false }
        let report = doc.nodeStatus[node] ?? ""
        let a = try? await MediaIO.readVideo(plain), b = try? await MediaIO.readVideo(tuned)
        let diff = (a != nil && b != nil) ? abs(a!.frames - b!.frames).mean().item(Float.self) : 0
        say(String(format: "mean difference with vs without: %.3f; LoRA node: %@", diff, report))
        lib.remove(id)
        model.closeTab(doc.id)
        return pairOnDisk == (remote.companion != nil) && diff > 0.01
    }

    static func runFlux(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let flux = ModelStore().modelsDirectory.appendingPathComponent("flux-1-dev")
        guard ModelRegistry.className(at: flux) == "FluxPipeline" else { say("no FLUX.1 [dev] at \(flux.path)"); return false }
        let qwen = model.modelPath
        _ = try? await model.engine.preload(modelPath: qwen)
        let before = model.engine.loadedModelsSnapshot().map(\.name)

        // A LoRA with zero up matrices (like a freshly initialised one) in the kohya/BFL fused layout.
        let lora = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-flux-lora.safetensors")
        var arrays: [String: MLXArray] = [:]
        for (name, out) in [("lora_unet_double_blocks_0_img_attn_qkv", 9216), ("lora_unet_single_blocks_0_linear1", 21504)] {
            arrays[name + ".lora_down.weight"] = MLXRandom.normal([4, 3072], key: MLXRandom.key(1)).asType(.bfloat16)
            arrays[name + ".lora_up.weight"] = MLXArray.zeros([out, 4], dtype: .bfloat16)
        }
        try? save(arrays: arrays, url: lora)

        model.setImageModel(path: flux.path, undoManager: nil)
        guard let latent = model.firstNode(ofType: "EmptyLatent"), let sampler = model.firstNode(ofType: "Sampler"),
              let te = model.firstNode(ofType: "TextEncode") else { say("image graph not set up"); return false }
        model.addLoRA(lora, undoManager: nil)
        model.mutate("Small", undoManager: nil) { g in
            try? g.setValue(.int(512), node: latent.id, input: "width")
            try? g.setValue(.int(512), node: latent.id, input: "height")
            try? g.setValue(.int(8), node: sampler.id, input: "steps")
            try? g.setValue(.string("A lighthouse on a rocky coast at dusk, oil painting"), node: te.id, input: "prompt")
        }
        say("LoRAs in the panel: \(model.simpleLoRAs.map(\.id))")

        func run() async -> (Double, URL)? {
            model.previewCount = 0
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { say("nothing queued: \(model.lastError ?? "?")"); return nil }
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            guard case .done(let secs)? = model.queue.first(where: { $0.id == item.id })?.status,
                  let url = model.history.first(where: { $0.kind == .image })?.url else {
                say("job failed: \(model.lastError ?? "?")"); return nil
            }
            return (secs, url)
        }
        guard let (secs, url) = await run() else { return false }
        let after = model.engine.loadedModelsSnapshot().map(\.name)
        let cg = try? ImageArray.load(url)
        let meta = ImageArray.readPNGText(url)
        say(String(format: "txt2img done in %.1fs, previews %d: %@ %dx%d, graph embedded %@", secs, model.previewCount,
                   url.lastPathComponent, cg?.width ?? 0, cg?.height ?? 0, meta[StandardNodes.graphMetadataKey] != nil ? "yes" : "no"))
        say("loaded before: \(before), after: \(after)")
        let ok1 = cg?.width == 512 && cg?.height == 512 && meta[StandardNodes.graphMetadataKey] != nil
            && after.contains { $0.contains("flux-1-dev") } && !after.contains { $0.contains("qwen") }

        model.setReferenceImage(url, undoManager: nil)
        let i2i = model.isImg2Img
        say("image-to-image chain: \(i2i)")
        guard i2i, let (secs2, url2) = await run() else { return false }
        let cg2 = try? ImageArray.load(url2)
        say(String(format: "img2img done in %.1fs: %@ %dx%d", secs2, url2.lastPathComponent, cg2?.width ?? 0, cg2?.height ?? 0))

        model.setImageModel(path: qwen, undoManager: nil)
        let back = !model.isImg2Img && model.referenceImagePath != nil
            && ImageFamily.of(path: model.firstNode(ofType: "LoadModel").flatMap { n -> String? in
                if case .value(.string(let p))? = n.inputs["path"] { return p }; return nil
            } ?? "") == .qwen
        model.clearReferenceImage(undoManager: nil)
        for l in model.simpleLoRAs { model.removeLoRA(l.id, undoManager: nil) }
        say("back to Qwen (reference editing, no image-to-image chain): \(back); LoRAs left \(model.simpleLoRAs.count)")
        // Image-to-image scales the source to about 1 MP, as inpainting does: 512² → 1024².
        return ok1 && cg2?.width == 1024 && cg2?.height == 1024 && back && model.simpleLoRAs.isEmpty
    }

    /// `--smoke-sound`: the Sound mode end to end: switch modes (graph swap), generate a short clip, check history
    /// and the file, then switch back to Image.
    static func runSound(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("stable-audio-open-small")
        guard ModelRegistry.className(at: dir) == "StableAudioPipeline" else { say("no Stable Audio at \(dir.path)"); return false }
        let imageTab = model.activeTabID
        guard let tab = openPreset(model, "sound", expecting: dir), let gen = model.firstNode(ofType: "TextToAudio") else {
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
        let shown = model.displayedMedia?.kind == .audio && tab.nodeResults["save"]?.kind == .audio
        model.activate(imageTab)
        let back = model.firstNode(ofType: "Sampler") != nil
        say("shown on the node and in the pane: \(shown); back to the image tab: \(back)")
        return abs(clip.duration - 4) < 0.05 && clip.sampleRate == 44100 && clip.channels == 2
            && meta[StandardNodes.graphMetadataKey] != nil && shown && back
    }

    /// `--smoke-music`: the Music mode end to end: switch modes, generate a short instrumental (planner on), check the
    /// file and history, then switch back to Image.
    static func runMusic(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("ace-step-1.5")
        guard ModelRegistry.className(at: dir) == "ACEStepPipeline" else { say("no ACE-Step at \(dir.path)"); return false }
        let imageTab = model.activeTabID
        guard let tab = openPreset(model, "music", expecting: dir), let gen = model.firstNode(ofType: "TextToMusic") else {
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
        let shown = model.displayedMedia?.kind == .audio && tab.nodeResults["save"]?.kind == .audio
        model.activate(imageTab)
        let back = model.firstNode(ofType: "Sampler") != nil
        say("shown on the node and in the pane: \(shown); back to the image tab: \(back)")
        return abs(clip.duration - 10) < 0.05 && clip.sampleRate == 48000 && clip.channels == 2 && maxTotal > 8
            && meta[StandardNodes.graphMetadataKey] != nil && shown && back
    }

    /// `--smoke-speech`: the Speech mode end to end: a voice preview (shown, not saved), then a generated clip in
    /// history with the graph embedded, then back to Image.
    static func runSpeech(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        let dir = ModelStore().modelsDirectory.appendingPathComponent("kokoro-82m")
        guard ModelRegistry.className(at: dir) == "KokoroPipeline" else { say("no Kokoro at \(dir.path)"); return false }
        let imageTab = model.activeTabID
        guard let tab = openPreset(model, "speech", expecting: dir), model.firstNode(ofType: "TextToSpeech") != nil else {
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
        let shown = model.displayedMedia?.kind == .audio && tab.nodeResults["save"]?.kind == .audio
        model.activate(imageTab)
        let back = model.firstNode(ofType: "Sampler") != nil
        say("shown on the node and in the pane: \(shown); back to the image tab: \(back)")
        return previewShown && model.history.count > historyBefore && clip.duration > 4 && clip.sampleRate == 24000
            && meta[StandardNodes.graphMetadataKey] != nil && shown && back
    }
}

extension SmokeTest {
    /// Opens a built-in preset in a new tab and checks it points at `folder`.
    static func openPreset(_ model: AppModel, _ id: String, expecting folder: URL) -> WorkflowDocument? {
        guard let p = model.preset(id: id) else { print("[smoke] no preset \(id)"); return nil }
        let doc = model.openPreset(p)
        let path: String? = model.firstNode(ofType: "LoadModel").flatMap { n in
            if case .value(.string(let s))? = n.inputs["path"] { return s }; return nil
        }
        let ok = path.map { URL(fileURLWithPath: $0).standardizedFileURL == folder.standardizedFileURL } ?? false
        print("[smoke] opened preset \(id) in tab “\(doc.name)”, model \(path ?? "none") \(ok ? "✓" : "≠ \(folder.path)")")
        fflush(stdout)
        return ok ? doc : nil
    }
}

extension SmokeTest {
    /// `--smoke-workflows` (no model needed; runs against a throwaway library): tabs with their own undo, saving,
    /// renaming, moving and duplicating workflows, the restored session, presets, every Add to Workflow chain, the
    /// per-workflow output folder with results on the nodes, and the menu shortcuts.
    static func runWorkflows(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        var failures: [String] = []
        func check(_ ok: Bool, _ what: String) {
            say((ok ? "✓ " : "✗ ") + what)
            if !ok { failures.append(what) }
        }
        let settle = { try? await Task.sleep(nanoseconds: 150_000_000) }  // lets undo groups close
        say("library \(model.library.root.path)")
        check(model.tabs.count == 1 && model.active.name == "Untitled" && !model.active.isDirty, "starts with one clean tab")

        // Tabs and undo.
        let first = model.active
        guard let te = model.firstNode(ofType: "TextEncode") else { say("no TextEncode"); return false }
        model.setLiteral(.string("a fox in tab one"), node: te.id, input: "prompt", undoManager: model.activeUndoManager)
        await settle()
        let second = model.newTab()
        check(model.tabs.count == 2 && model.activeTabID == second.id && second.name == "Untitled 2", "⌘T opens “Untitled 2”")
        model.setLiteral(.string("a fox in tab two"), node: te.id, input: "prompt", undoManager: model.activeUndoManager)
        await settle()
        check(first.isDirty && second.isDirty, "edits mark both tabs dirty")
        model.activeUndoManager.undo()
        await settle()
        check(!second.isDirty && first.literalPrompt(te.id) == "a fox in tab one", "undo in tab two leaves tab one alone")
        model.activeUndoManager.redo()
        model.activate(first.id)
        check(model.graph == first.graph && model.literal(te.id, "prompt")?.description == "a fox in tab one",
              "switching tabs switches the graph")

        // Saving and the library.
        check(model.saveActive() == false, "an untitled tab needs a name to save")
        check(model.saveAs(first, name: "Smoke Fox"), "save as “Smoke Fox”")
        let url = model.library.workflows.appendingPathComponent("Smoke Fox.json")
        check(FileManager.default.fileExists(atPath: url.path) && !first.isDirty && first.name == "Smoke Fox",
              "file written, tab clean and named after it")
        check(model.workflowTree.workflows.map(\.name) == ["Smoke Fox"], "library lists it")
        model.setLiteral(.int(512), node: "latent", input: "width", undoManager: nil)
        check(first.isDirty && model.saveActive() && !first.isDirty && (try? Graph.load(url)) == first.graph, "⌘S saves in place")
        model.renameWorkflow(url, to: "Arctic Fox")
        check(first.name == "Arctic Fox" && first.fileURL?.lastPathComponent == "Arctic Fox.json", "rename follows the open tab")
        model.newWorkflowFolder(in: nil)
        let folder = model.library.workflows.appendingPathComponent("New Folder")
        model.moveWorkflow(first.fileURL!, to: folder)
        check(first.fileURL?.deletingLastPathComponent().lastPathComponent == "New Folder", "move into a folder follows the tab")
        model.duplicateWorkflow(first.fileURL!)
        check(model.workflowTree.allWorkflows.map(\.name).sorted() == ["Arctic Fox", "Arctic Fox 2"], "duplicate")
        let tabCount = model.tabs.count
        model.open(first.fileURL!)
        check(model.tabs.count == tabCount && model.activeTabID == first.id, "opening an open workflow switches to its tab")

        // Presets.
        check(model.presets.filter(\.isBuiltIn).count >= 15, "\(model.presets.count) presets")
        let installed = model.installedModelIDs
        say("models installed: \(installed.sorted().joined(separator: ", "))")
        for p in model.presets {
            let doc = model.openPreset(p)
            let err = model.validationError
            check(err == nil && doc.presetID == p.id && !doc.isDirty
                  && !doc.graph.nodes.contains { n in n.inputs.values.contains(.value(.string(Preset.outputPlaceholder))) },
                  "preset \(p.id) opens as “\(doc.name)”\(p.isAvailable(installed: installed) ? "" : " (model not installed)")"
                  + (err.map { ": \($0)" } ?? ""))
            model.closeTab(doc.id)
        }
        let fake = Preset(id: "x", name: "x", category: .image, models: ["no-such-model"], modelName: "", summary: "",
                          symbol: "photo", graph: Graph())
        check(!fake.isAvailable(installed: installed), "a preset with a missing model is unavailable")

        // Add to Workflow on the text-to-image preset.
        guard let t2i = model.preset(id: "qwen-txt2img") else { return false }
        model.openPreset(t2i)
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let picture = cwd.appendingPathComponent("examples/inpaint-mask.png")
        func groups() -> [String] { (model.graph.groups ?? []).map(\.title).sorted() }
        model.setHiRes(true, undoManager: nil); model.groupAddedChain(.hires); model.tidyChainGroups()
        check(model.isHiResEnabled && groups() == ["Hi-res Fix"], "hi-res fix arrives as a group")
        model.remove(.hires)
        check(!model.isHiResEnabled && groups().isEmpty, "removing it drops the group")
        model.setReferenceImage(picture, undoManager: nil)
        check(model.isActive(.reference) && model.canAdd(.inpaint) && model.canAdd(.outpaint), "reference image")
        model.setMask(picture, undoManager: nil); model.groupAddedChain(.inpaint); model.tidyChainGroups()
        check(model.maskPath != nil && groups() == ["Inpainting"], "inpaint mask arrives as a group")
        model.add(.outpaint)
        check(model.isOutpainting && model.maskPath == nil && groups() == ["Outpainting"], "outpaint replaces inpainting")
        model.addLoRA(picture, undoManager: nil)
        check(model.simpleLoRAs.count == 1, "LoRA spliced before the sampler")
        model.remove(.lora); model.remove(.outpaint); model.remove(.reference)
        check(model.validationError == nil && groups().isEmpty && model.referenceImageNode == nil, "everything removed again")
        if let ltx = model.preset(id: "ltx-t2v") {
            model.openPreset(ltx)
            check(model.canAdd(.firstFrame) && !model.canAdd(.hires), "video presets offer a first frame, not hi-res")
            model.setFirstFrame(picture, undoManager: nil)
            check(model.isActive(.firstFrame), "first frame")
        }

        // Styles are per Text Encode node and applied at queue time.
        model.activate(first.id)
        model.setStyle("Watercolor", node: te.id, undoManager: nil)
        let prepared = model.prepared(model.graph, seed: 1, workflow: first.name)
        let p0: String = { if case .value(.string(let s))? = prepared[te.id]?.inputs["prompt"] { return s }; return "" }()
        let dir: String = { if case .value(.string(let s))? = prepared["save"]?.inputs["directory"] { return s }; return "" }()
        check(p0.lowercased().contains("watercolor") && prepared.metadata?["workflow"] == "Arctic Fox"
              && dir == model.outputFolder(forWorkflow: "Arctic Fox"),
              "queued graph: style applied, workflow recorded, saved into “<output>/Arctic Fox”")
        model.setStyle(nil, node: te.id, undoManager: nil)

        // A real (model-free) run in its own tab: results on the nodes and in the workflow's folder.
        if var g = try? Graph.load(cwd.appendingPathComponent("examples/media-test.json")) {
            for id in ["save_video", "save_audio"] { try? g.setValue(.string(model.outputDirectory), node: id, input: "directory") }
            let doc = model.openTab(g, name: "Smoke Media")
            model.enqueue(undoManager: nil)
            if let item = model.queue.last {
                model.activate(first.id)  // the result must land in its own tab, not the active one
                while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
            let onNodes = doc.nodeResults["save_video"]?.kind == .video && doc.nodeResults["save_audio"]?.kind == .audio
            let files = model.history.filter { $0.workflow == "Smoke Media" }
            check(onNodes && first.nodeResults.isEmpty, "results shown on the Save nodes of the tab that ran")
            check(files.count >= 2 && files.allSatisfy { $0.url.deletingLastPathComponent().lastPathComponent == "Smoke Media" },
                  "outputs saved in “<output>/Smoke Media” and listed with their workflow")
            check(doc.outputs.count >= 2 && model.viewerItems.isEmpty == false, "the tab keeps its results for the viewer")
            for f in files { try? FileManager.default.removeItem(at: f.url) }
            try? FileManager.default.removeItem(at: URL(fileURLWithPath: model.outputFolder(forWorkflow: "Smoke Media")))
            model.reloadHistory()
        }

        // Assets: imports, Load-node paths relative to the input folder, drops on the graph.
        let maskFile = cwd.appendingPathComponent("examples/inpaint-mask.png")
        let imported = model.importAssets([maskFile])
        check(imported.first?.lastPathComponent == "inpaint-mask.png" && model.importAssets([maskFile]) == imported
              && model.assets.map(\.relativePath) == ["inpaint-mask.png"], "import copies into the input folder once")
        let assetsTab = model.openTab(Graph(), name: "Assets")
        model.dropFiles([maskFile], at: GraphPoint(x: 100, y: 100), onto: nil)
        let load = model.graph.nodes.first { $0.type == "LoadImage" }
        check(load?.inputs["path"] == .value(.string("inpaint-mask.png")), "a dropped picture becomes a Load Image with its input name")
        let other = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-drop-\(UUID().uuidString.prefix(6)).png")
        try? FileManager.default.copyItem(at: maskFile, to: other)
        model.dropFiles([other], at: GraphPoint(x: 0, y: 0), onto: load?.id)
        check(model.graph.nodes.count == 1 && model.graph[load?.id ?? ""]?.inputs["path"] == .value(.string(other.lastPathComponent)),
              "dropping onto a Load node replaces its file")
        model.mutate("Preview", undoManager: nil) { g in
            g.nodes.append(GraphNode(id: "mask", type: "LoadMask", inputs: ["path": .value(.string(""))], position: GraphPoint(x: 0, y: 300)))
            g.nodes.append(GraphNode(id: "preview", type: "PreviewImage", inputs: ["image": .link(node: load?.id ?? "", output: "image")],
                                     position: GraphPoint(x: 300, y: 0)))
        }
        model.dropFiles([maskFile], at: GraphPoint(x: 0, y: 0), onto: "mask")
        check(model.graph["mask"]?.inputs["path"] == .value(.string("masks/inpaint-mask.png")), "an image dropped on Load Mask goes to masks/")
        model.renameAsset(model.library.input.appendingPathComponent(other.lastPathComponent), to: "renamed")
        check(model.graph[load?.id ?? ""]?.inputs["path"] == .value(.string("renamed.png")), "renaming an asset updates its Load nodes")
        model.mutate("No mask", undoManager: nil) { g in g.nodes.removeAll { $0.id == "mask" } }
        model.enqueue(undoManager: nil)
        if let item = model.queue.last {
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false { try? await Task.sleep(nanoseconds: 50_000_000) }
            try? await Task.sleep(nanoseconds: 300_000_000)
        }
        check(assetsTab.nodeResults["preview"]?.image?.width == 960, "the graph runs with the asset found in the input folder")
        let tabsNow = model.tabs.count
        model.dropFiles([first.fileURL!], at: GraphPoint(x: 0, y: 0), onto: nil)
        check(model.activeTabID == first.id && model.tabs.count == tabsNow, "a dropped workflow opens (its open tab)")
        if let output = model.history.first(where: { $0.kind == .image && AppModel.embeddedGraph(in: $0.url) != nil })?.url {
            model.activate(assetsTab.id)
            model.dropFiles([output], at: GraphPoint(x: 0, y: 0), onto: nil)
            let opened = model.activeTabID != assetsTab.id
            model.activate(assetsTab.id)
            let before = model.graph.nodes.count
            model.dropFiles([output], at: GraphPoint(x: 0, y: 500), onto: nil, forceAsInput: true)
            check(opened && model.graph.nodes.count == before + 1, "an output with a graph opens it; with ⌥ it becomes an input")
        }
        try? FileManager.default.removeItem(at: other)

        // Graph editor: tidy splices, Arrange, align, copy/paste, wires, collapse.
        /// Pairs of nodes whose cards overlap, and nodes lying on a group they don't belong to.
        func clashes() -> [String] {
            let level = model.editorGraph
            let frames = level.nodes.compactMap { n in model.frame(of: n).map { (n.id, $0) } }
            var out: [String] = []
            for i in frames.indices { for j in frames.indices where j > i && frames[i].1.intersects(frames[j].1) {
                out.append("\(frames[i].0)×\(frames[j].0)")
            } }
            for g in level.groups ?? [] {
                let r = CGRect(x: g.x, y: g.y, width: g.width, height: g.height)
                let members = model.nodes(in: g.id)
                for (id, f) in frames where !members.contains(id) && f.intersects(r) { out.append("\(id)×\(g.title)") }
            }
            return out
        }
        /// Data links point right, except into and out of groups (a chain like Outpainting both feeds the sampler
        /// and takes the decoded image, so one of its links must loop back).
        func leftToRight() -> Bool {
            let level = model.editorGraph
            let grouped = Set((level.groups ?? []).flatMap { model.nodes(in: $0.id) })
            return model.layoutEdges(level).filter { !$0.weak && !grouped.contains($0.source) && !grouped.contains($0.target) }
                .allSatisfy { e in
                guard let a = level[e.source].flatMap(model.frame(of:)), let b = level[e.target].flatMap(model.frame(of:)) else { return true }
                return a.maxX < b.minX
            }
        }
        var spliceClashes: [String] = []
        for p in model.presets where p.category == .image && p.isBuiltIn {
            model.openPreset(p)
            if model.canAdd(.hires) { model.setHiRes(true, undoManager: nil); model.groupAddedChain(.hires) }
            if model.referenceImageNode == nil, model.canAdd(.reference) { model.setReferenceImage(picture, undoManager: nil) }
            if model.canAdd(.outpaint) { model.add(.outpaint) }
            spliceClashes += clashes().map { "\(p.id): \($0)" }
        }
        check(spliceClashes.isEmpty, "Add to Workflow places chains in free space" + (spliceClashes.isEmpty ? "" : " — \(spliceClashes)"))
        guard let inpaint = model.preset(id: "qwen-inpaint") else { return false }
        let editorTab = model.openPreset(inpaint)
        if model.canAdd(.outpaint) { model.add(.outpaint) }
        if model.canAdd(.hires) { model.setHiRes(true, undoManager: nil); model.groupAddedChain(.hires) }
        await settle()
        // In this headless run undo groups don't close between steps; start Arrange on a clean history.
        model.activeUndoManager.removeAllActions()
        let beforeArrange = model.graph
        model.arrange(selectionOnly: false, undoManager: model.activeUndoManager)
        await settle()
        let backwards = model.layoutEdges(model.editorGraph).filter { e in
            guard !e.weak, let a = model.editorGraph[e.source].flatMap(model.frame(of:)),
                  let b = model.editorGraph[e.target].flatMap(model.frame(of:)) else { return false }
            return a.maxX >= b.minX
        }.map { "\($0.source)→\($0.target)" }
        check(model.graph != beforeArrange && clashes().isEmpty && leftToRight(),
              "Arrange Graph: no overlaps, every data link points right"
                + " (changed \(model.graph != beforeArrange), clashes \(clashes()), backwards \(backwards))")
        let undoName = model.activeUndoManager.undoActionName
        model.activeUndoManager.undo()
        await settle()
        check(model.graph == beforeArrange, "Arrange is one undo step (undid “\(undoName)”, nodes moved back: "
              + "\(model.graph.nodes.filter { beforeArrange[$0.id]?.position == $0.position }.count)/\(model.graph.nodes.count), "
              + "groups equal \(model.graph.groups == beforeArrange.groups))")
        model.activeUndoManager.redo(); await settle()

        let sampler = model.graph.nodes.first { $0.type == "Sampler" }!.id
        let decode = model.graph.nodes.first { $0.type == "VAEDecode" && $0.inputs["latent"] == .link(node: sampler, output: "latent") }!.id
        let latentNode = model.graph.nodes.first { $0.type == "EmptyLatent" || $0.type == "VAEEncode" }!.id
        model.selection = [sampler, decode, latentNode]
        model.alignSelection(.top, undoManager: model.activeUndoManager)
        let tops = Set([sampler, decode, latentNode].compactMap { model.graph[$0]?.position?.y })
        check(tops.count == 1, "Align Top Edges")

        // Copy two linked nodes into another tab: new tab keeps their ids, internal link intact, external links dropped.
        model.selection = [sampler, decode]
        let clip = model.clipboardForSelection()!
        let pasteTab = model.openTab(Graph(), name: "Paste")
        model.insert(clip, at: GraphPoint(x: 0, y: 0), keepExternalLinks: false, actionName: "Paste", undoManager: nil)
        check(model.graph.nodes.count == 2 && model.graph[decode]?.inputs["latent"] == .link(node: sampler, output: "latent")
              && model.graph[sampler]?.inputs["model"] == nil && model.selection == [sampler, decode],
              "copy and paste into another tab keeps the link between the copies")
        model.insert(clip, at: GraphPoint(x: 0, y: 400), keepExternalLinks: false, actionName: "Paste", undoManager: nil)
        check(model.graph.nodes.count == 4 && model.graph["\(decode)_2"]?.inputs["latent"] == .link(node: "\(sampler)_2", output: "latent"),
              "pasting again renames the copies")
        model.closeTab(pasteTab.id)

        // Wires.
        model.activate(editorTab.id)
        model.disconnect(node: decode, input: "latent", undoManager: model.activeUndoManager)
        await settle()
        check(model.graph[decode]?.inputs["latent"] == nil, "deleting a selected wire disconnects it")
        model.activeUndoManager.undo(); await settle()
        check(model.graph[decode]?.inputs["latent"] == .link(node: sampler, output: "latent"), "…and undo brings it back")
        let count = model.graph.nodes.count
        model.addNode(type: "VAEDecode", connecting: AppModel.PendingWire(node: sampler, socket: "latent", type: .latent,
                                                                         fromOutput: true, point: GraphPoint(x: 2000, y: 0)),
                      undoManager: model.activeUndoManager)
        let added = model.selection.first ?? ""
        check(model.graph.nodes.count == count + 1 && model.graph[added]?.inputs["latent"] == .link(node: sampler, output: "latent")
              && model.graph[added]?.position?.x == 2000, "a wire dropped on the canvas adds a connected node there")
        let save = model.graph.nodes.first { $0.type == "SaveImage" }!.id
        model.addNode(type: "LoadImage", connecting: AppModel.PendingWire(node: save, socket: "image", type: .image,
                                                                         fromOutput: false, point: GraphPoint(x: 2000, y: 600)),
                      undoManager: model.activeUndoManager)
        let feeder = model.selection.first ?? ""
        check(model.graph[feeder]?.type == "LoadImage" && model.graph[save]?.inputs["image"] == .link(node: feeder, output: "image"),
              "a wire dragged back from an input adds a node that feeds it")
        await settle()
        model.activeUndoManager.undo(); await settle()
        check(model.graph[feeder] == nil && model.graph[save]?.inputs["image"] != .link(node: feeder, output: "image"),
              "adding and connecting is one undo step")

        // Collapse survives saving.
        model.toggleCollapsed([sampler], undoManager: nil)
        check(model.graph[sampler]?.isCollapsed == true && model.frame(of: model.graph[sampler]!)?.height == NodeLayout.header,
              "a collapsed node is just its title bar")
        check(model.saveAs(editorTab, name: "Smoke Editor"), "save the arranged workflow")
        let reloaded = editorTab.fileURL.flatMap { try? Graph.load($0) }
        check(reloaded?[sampler]?.isCollapsed == true && reloaded?[decode]?.collapsed == nil, "collapse is saved with the workflow")
        model.toggleCollapsed([sampler], undoManager: nil)
        check(model.graph[sampler]?.collapsed == nil, "expanding removes the flag")

        // Session: drafts and the tab list come back in a new model.
        let dirty = model.newTab()
        model.setLiteral(.string("never saved"), node: te.id, input: "prompt", undoManager: nil)
        model.saveSession()
        let names = model.tabs.map(\.name)
        let restored = AppModel(persistSettings: false, autoloadModel: false, library: model.library, restoreSession: true)
        check(restored.tabs.map(\.name) == names && restored.activeTabID == dirty.id, "tabs restored in order with the active one")
        let rDirty = restored.tabs.first { $0.id == dirty.id }
        let rFirst = restored.tabs.first { $0.id == first.id }
        check(rDirty?.isDirty == true && rDirty?.literalPrompt(te.id) == "never saved" && rDirty?.fileURL == nil,
              "an unsaved tab comes back from its draft, still marked unsaved")
        check(rFirst?.fileURL == first.fileURL && rFirst?.isDirty == first.isDirty && rFirst?.graph == first.graph,
              "a saved tab comes back with its file")

        // No tab strip: ⌃Tab goes back to what was used before, ⌘W on a chat brings it back, and only the most
        // recent chats stay loaded.
        let wa = model.newTab(), wb = model.newTab()
        model.switchToPrevious()
        check(model.activeTabID == wa.id && model.activeChat == nil, "⌃Tab goes back to the workflow used before")
        let firstChat = model.newChat()
        model.switchToPrevious()
        check(model.activeChat == nil && model.activeTabID == wa.id, "⌃Tab from a chat goes back to the workflow")
        model.switchToPrevious()
        check(model.activeChatID == firstChat.id, "⌃Tab again brings the chat back")
        model.closeChat(firstChat.id)
        check(model.activeChat == nil && model.activeTabID == wa.id && !model.chats.contains { $0.id == firstChat.id },
              "closing a chat shows what was used before it and unloads it")
        model.activate(wb.id)
        model.closeTab(wb.id)
        check(model.activeTabID == wa.id, "closing a workflow shows the workflow used before it")
        let many = (0..<(AppModel.recentChatLimit + 2)).map { _ in model.newChat() }
        await settle()
        check(model.chats.filter(\.isRecent).count == AppModel.recentChatLimit && model.activeChatID == many.last?.id
              && !model.chats.contains { $0.id == many[0].id },
              "only the \(AppModel.recentChatLimit) most recent chats stay loaded (\(model.chats.filter(\.isRecent).count))")
        check(model.openItems.count == model.tabs.count + AppModel.recentChatLimit, "the Window menu lists open workflows and recent chats")
        for d in model.chats { model.closeChat(d.id) }
        model.activate(wa.id)

        // Closing the last tab leaves a fresh one.
        for d in model.tabs { model.closeTab(d.id) }
        check(model.tabs.count == 1 && model.active.name == "Untitled", "closing every tab leaves a new one")

        // The File menu's ⌘W closes the workflow in front, not the window (exactly one ⌘W item).
        let cmdW = Self.menuItems(NSApp.mainMenu).filter { $0.keyEquivalent == "w" && $0.keyEquivalentModifierMask == .command }
        check(cmdW.map(\.title) == ["Close Workflow"], "⌘W is Close Workflow (found: \(cmdW.map(\.title)))")

        say(failures.isEmpty ? "all workflow checks passed" : "\(failures.count) failed: \(failures.joined(separator: "; "))")
        return failures.isEmpty
    }

    static func menuItems(_ menu: NSMenu?) -> [NSMenuItem] {
        guard let menu else { return [] }
        return menu.items.flatMap { [$0] + menuItems($0.submenu) }
    }
}

extension WorkflowDocument {
    func literalPrompt(_ node: String) -> String? {
        if case .value(.string(let s))? = graph[node]?.inputs["prompt"] { return s }
        return nil
    }
}

extension SmokeTest {
    /// `--smoke-remote <url> --smoke-token-file <path>`: the same graph on this Mac and on a running
    /// `aistudio-cli serve`, through the queue like the Generate button. Checks the remote result matches the local
    /// one pixel for pixel, carries the same embedded graph, lands in history, streamed previews, and that cancelling
    /// a remote run stops it.
    static func runRemote(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        guard let tokenPath = argument("--smoke-token-file"),
              let token = try? String(contentsOfFile: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        else { say("needs --smoke-token-file"); return false }
        let conn = ServerConnection(name: "Smoke server", baseURL: url)
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard let info = model.connections.info(conn.id) else {
            say("server not reachable: \(model.connections.status[conn.id].map { "\($0)" } ?? "?")"); return false
        }
        say("server \(info.serverVersion) on \(info.hostName): \(info.models.count) models")
        guard let te = model.firstNode(ofType: "TextEncode"), let sampler = model.firstNode(ofType: "Sampler"),
              let latent = model.firstNode(ofType: "EmptyLatent") else { say("graph not set up"); return false }
        model.setLiteral(.string("a red lighthouse on a cliff at sunset, oil painting"), node: te.id, input: "prompt", undoManager: nil)
        model.setLiteral(.int(512), node: latent.id, input: "width", undoManager: nil)
        model.setLiteral(.int(512), node: latent.id, input: "height", undoManager: nil)
        model.setLiteral(.int(1234), node: sampler.id, input: "seed", undoManager: nil)
        model.randomizeSeed = false

        func runOnce(on backend: String) async -> (QueueItem.Status?, Double, Int, URL?) {
            model.active.backendID = backend
            let previews = model.previewCount
            let start = Date()
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { return (nil, 0, 0, nil) }
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
            try? await Task.sleep(nanoseconds: 500_000_000)
            return (model.queue.first { $0.id == item.id }?.status, Date().timeIntervalSince(start),
                    model.previewCount - previews, model.displayedURL)
        }

        let (localStatus, localSecs, _, localURL) = await runOnce(on: BackendID.local)
        guard case .done? = localStatus, let localURL else { say("local run failed: \(model.lastError ?? "?")"); return false }
        say(String(format: "local run %.1fs → %@", localSecs, localURL.lastPathComponent))
        // One copy of the model at a time: this Mac's engine and the server share the same GPU and memory.
        await model.engine.unloadModels()

        let (remoteStatus, remoteSecs, previews, remoteURL) = await runOnce(on: conn.id)
        guard case .done? = remoteStatus, let remoteURL, remoteURL != localURL else {
            say("remote run failed: \(remoteStatus.map { "\($0)" } ?? "?") \(model.lastError ?? "")"); return false
        }
        say(String(format: "remote run %.1fs → %@ (%@), %d previews", remoteSecs, remoteURL.lastPathComponent,
                   remoteURL.deletingLastPathComponent().path, previews))
        let diff = pixelDifference(localURL, remoteURL)
        let localGraph = AppModel.embeddedGraph(in: localURL), remoteGraph = AppModel.embeddedGraph(in: remoteURL)
        let sameGraph = localGraph != nil && localGraph == remoteGraph
        let inHistory = model.history.contains { $0.url == remoteURL }
        let sameFolder = localURL.deletingLastPathComponent() == remoteURL.deletingLastPathComponent()
        say("max pixel difference \(diff.map(String.init) ?? "n/a"); same embedded graph \(sameGraph); in history \(inHistory); same folder \(sameFolder)")

        // Cancel mid-run.
        model.active.backendID = conn.id
        model.setLiteral(.int(4321), node: sampler.id, input: "seed", undoManager: nil)
        model.enqueue(undoManager: nil)
        guard let item = model.queue.last else { return false }
        while true {
            try? await Task.sleep(nanoseconds: 100_000_000)
            guard let cur = model.queue.first(where: { $0.id == item.id }) else { return false }
            if case .running(let step, _) = cur.status, step >= 2 { break }
            if cur.isFinished { say("finished before it could be cancelled: \(cur.status)"); return false }
        }
        let cancelAt = Date()
        model.cancelCurrent()
        while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let cancelled = model.queue.first { $0.id == item.id }?.status == .cancelled
        say(String(format: "cancelled %@ after %.2fs", cancelled ? "yes" : "no", Date().timeIntervalSince(cancelAt)))
        model.active.backendID = BackendID.local
        model.connections.remove(conn.id)
        return diff == 0 && sameGraph && inHistory && sameFolder && previews > 0 && cancelled
    }

    /// Largest per-channel difference between two images of the same size (nil when they can't be compared).
    static func pixelDifference(_ a: URL, _ b: URL) -> Int? {
        func pixels(_ u: URL) -> (Int, Int, [UInt8])? {
            guard let src = CGImageSourceCreateWithURL(u as CFURL, nil),
                  let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
            var buf = [UInt8](repeating: 0, count: img.width * img.height * 4)
            guard let ctx = CGContext(data: &buf, width: img.width, height: img.height, bitsPerComponent: 8,
                                      bytesPerRow: img.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
            ctx.draw(img, in: CGRect(x: 0, y: 0, width: img.width, height: img.height))
            return (img.width, img.height, buf)
        }
        guard let (w1, h1, p1) = pixels(a), let (w2, h2, p2) = pixels(b), w1 == w2, h1 == h2 else { return nil }
        var m = 0
        for i in 0..<p1.count { m = max(m, abs(Int(p1[i]) - Int(p2[i]))) }
        return m
    }
}

// MARK: Cloud GPU (M22)

extension Optional {
    func asyncMap<T>(_ f: (Wrapped) async -> T?) async -> T? {
        guard let v = self else { return nil }
        return await f(v)
    }
}

extension SmokeTest {
    /// The fake GPU server's marks on a clip's middle frame: the top two 8-pixel bands' brightness, the right band's.
    static func frameMarks(_ url: URL) async -> (top: Float, second: Float, right: Float)? {
        guard let clip = try? await MediaIO.readVideo(url) else { return nil }
        let f = clip.frames[clip.frames.dim(0) / 2]  // [H, W, C] in 0…1
        let w = f.dim(1)
        return (f[2..<6, 16..<(w - 16), ..<3].mean().item(Float.self), f[10..<14, 16..<(w - 16), ..<3].mean().item(Float.self),
                f[24..., (w - 6)..<(w - 2), ..<3].mean().item(Float.self))
    }

    /// `--smoke-gpu <url> --smoke-token-file <file>`: adds a GPU server, then runs the three Cloud GPU presets through
    /// the queue (small sizes) and checks status lines, steps, previews, the cost line and the saved files.
    /// Against `python -m aistudio_gpu serve` with `AISTUDIO_FAKE_MODELS=1` it needs no GPU.
    static func runGPU(_ model: AppModel, url: String) async -> Bool {
        func say(_ s: String) { print("[smoke-gpu] \(s)"); fflush(stdout) }
        guard let tokenPath = argument("--smoke-token-file"),
              let token = try? String(contentsOfFile: tokenPath, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        else { say("needs --smoke-token-file"); return false }
        guard let flux = model.preset(id: "cloud-flux") else { say("no cloud presets"); return false }
        if model.presetAvailable(flux, installed: model.installedModelIDs) { say("preset available without a server"); return false }
        let conn = ServerConnection(name: "Smoke GPU", kind: .gpu, baseURL: url)
        model.connections.save(conn, token: token)
        await model.connections.test(conn.id)
        guard case .gpuOnline(let info)? = model.connections.status[conn.id] else {
            say("server not reachable: \(model.connections.status[conn.id].map { "\($0)" } ?? "?")"); return false
        }
        say("GPU server \(info.version) (\(info.runner)): \(info.tasks.filter(\.installed).map(\.id).joined(separator: ", "))")
        guard model.presetAvailable(flux, installed: model.installedModelIDs) else { say("preset still greyed"); return false }

        // A start image for image-to-video.
        let start = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-gpu-start.png")
        try? ImageArray.savePNG(MLXArray.ones([64, 96, 4]) * MLXArray([0.2, 0.5, 0.9, 1.0] as [Float]), to: start)
        var ok = true
        // Two LoRA files for the LoRA preset (the fake server marks the image's top row with their count / 4).
        let loras = (1...2).map { FileManager.default.temporaryDirectory.appendingPathComponent("smoke-gpu-lora\($0).safetensors") }
        for (i, url) in loras.enumerated() { try? Data(repeating: UInt8(i + 1), count: 1 << 20).write(to: url) }
        for (id, video) in [("cloud-flux", false), ("cloud-flux-lora", false), ("cloud-qwen-image", false),
                            ("cloud-wan-t2v", true), ("cloud-wan-i2v", true)] {
            guard let p = model.preset(id: id) else { say("\(id): missing"); ok = false; continue }
            let doc = model.openPreset(p)
            model.setLiteral(.int(256), node: "cloud", input: "width", undoManager: nil)
            model.setLiteral(.int(256), node: "cloud", input: "height", undoManager: nil)
            if video { model.setLiteral(.int(9), node: "cloud", input: "length", undoManager: nil) }
            if id == "cloud-wan-i2v" { model.setLiteral(.string(start.path), node: "first_frame", input: "path", undoManager: nil) }
            if id == "cloud-flux-lora" {
                model.setLiteral(.string(loras[0].path), node: "lora1", input: "path", undoManager: nil)
                guard model.canAdd(.lora) else { say("\(id): Add to Workflow › LoRA unavailable"); ok = false; continue }
                model.addCloudLoRA(loras[1], strength: 0.7, undoManager: nil)
            }
            if id == "cloud-qwen-image" {  // a LoRA chain on the Qwen-Image node, from Add to Workflow
                guard model.canAdd(.lora) else { say("\(id): Add to Workflow › LoRA unavailable"); ok = false; continue }
                model.addCloudLoRA(loras[0], undoManager: nil)
            }
            let previews = model.previewCount
            let began = Date()
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { ok = false; continue }
            var statuses: Set<String> = []
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                if let s = doc.nodeStatus["cloud"] { statuses.insert(s) }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            try? await Task.sleep(nanoseconds: 300_000_000)
            let finished = model.queue.first { $0.id == item.id }
            guard case .done? = finished?.status, let result = doc.nodeResults["save"]?.url else {
                say("\(id): \(finished.map { "\($0.status)" } ?? "?") \(model.lastError ?? "")"); ok = false; continue
            }
            let cost = doc.nodeStatus["cloud"] ?? ""
            let embedded = video ? await MediaIO.readMetadata(result)["aistudio.graph"] : AppModel.embeddedGraph(in: result).map { _ in "png" }
            var loraOK = true
            if id == "cloud-flux-lora" || id == "cloud-qwen-image" {
                let count: Float = id == "cloud-flux-lora" ? 2 : 1
                let image = (try? ImageArray.load(result)).map { ImageArray.array(from: $0) }
                let top = image.map { abs($0[0, 0..., ..<3] - count / 4).max().item(Float.self) } ?? 1
                loraOK = model.cloudLoRAs.map(\.id) == (count == 2 ? ["lora1", "lora2"] : ["lora1"]) && top < 0.01
                say("\(id): LoRAs \(model.cloudLoRAs.map(\.id)), top row off by \(top)")
            }
            let good = cost.hasPrefix("GPU ") && model.previewCount > previews && statuses.contains("Sampling")
                && embedded != nil && model.history.contains { $0.url == result } && loraOK
            say(String(format: "%@: %.1fs → %@; statuses %@; final \"%@\"; %d previews; %@", id, Date().timeIntervalSince(began),
                       result.lastPathComponent, statuses.sorted().joined(separator: " | "), cost,
                       model.previewCount - previews, good ? "ok" : "MISSING SOMETHING"))
            ok = ok && good
            model.closeTab(doc.id)
        }

        // `--smoke-lora-link <url>`: a LoRA added to the library from a link (a file on a local server the GPU
        // server may fetch from), picked in a node added with Add to Workflow › LoRA, next to an uploaded file: the
        // server downloads it itself. The library lives in a throwaway models folder for the test.
        if let link = argument("--smoke-lora-link"), let host = URL(string: link)?.host,
           let p = model.preset(id: "cloud-flux-lora"), let lib = model.store?.library,
           let remote = await { () async -> LoRARemote? in
               LoRASources.directHosts = [host]
               return try? await LoRASources.resolve(link)
           }() {
            let savedModels = ModelPaths.directory
            ModelPaths.directory = FileManager.default.temporaryDirectory.appendingPathComponent("smoke-library-\(UUID().uuidString)")
            lib.reload()
            let libID = lib.add(remote, link: link, title: "Smoke link LoRA", family: "flux.1", mac: false, servers: [])
            let doc = model.openPreset(p)
            model.setLiteral(.int(256), node: "cloud", input: "width", undoManager: nil)
            model.setLiteral(.int(256), node: "cloud", input: "height", undoManager: nil)
            model.setLiteral(.string(loras[0].path), node: "lora1", input: "path", undoManager: nil)
            model.add(.lora)
            let added = model.selection.first ?? ""
            model.setLiteral(.string(libID), node: added, input: "lora", undoManager: nil)
            model.enqueue(undoManager: nil)
            var statuses: [String] = []
            if let item = model.queue.last {
                while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                    if let s = doc.nodeStatus["cloud"], statuses.last != s { statuses.append(s) }
                    try? await Task.sleep(nanoseconds: 10_000_000)
                }
            }
            try? await Task.sleep(nanoseconds: 300_000_000)
            let result = doc.nodeResults["save"]?.url
            let top = result.flatMap { try? ImageArray.load($0) }.map { abs(ImageArray.array(from: $0)[0, 0..., ..<3] - 0.5).max().item(Float.self) } ?? 1
            let good = model.cloudLoRAs.map(\.id) == ["lora1", added] && added == "lora2" && top < 0.01
            let fits = lib.loras(for: "flux.1").map { $0.id } == [libID]
            say("LoRA from the library (a link): node \(added), top row off by \(top); in the FLUX menu: \(fits); statuses \(statuses.joined(separator: " | ")); \(good && fits ? "ok" : "MISSING SOMETHING")")
            ok = ok && good && fits
            model.closeTab(doc.id)
            lib.remove(libID)

            // A Wan 2.2 LoRA pair (high- and low-noise files) from the library on the cloud Wan node: the server fetches
            // both (the fake server marks the top row with the slot count, the second with the pairs).
            if let wanPreset = model.preset(id: "cloud-wan-t2v") {
                var pair = remote
                pair.baseModel = "Wan Video 2.2 T2V-A14B"
                pair.expert = "high"
                let lowURL = remote.downloadURL.deletingLastPathComponent().appendingPathComponent("link-lora-low.safetensors")
                pair.companion = LoRAFileRef(downloadURL: lowURL, fileName: "link-lora-low.safetensors", bytes: nil,
                                             sha256: nil, cacheName: "link-lora-low.safetensors")
                let pairID = lib.add(pair, link: link, title: "Smoke Wan pair", family: "wan-2.2-t2v", mac: false, servers: [])
                let wdoc = model.openPreset(wanPreset)
                model.setLiteral(.int(256), node: "cloud", input: "width", undoManager: nil)
                model.setLiteral(.int(256), node: "cloud", input: "height", undoManager: nil)
                model.setLiteral(.int(9), node: "cloud", input: "length", undoManager: nil)
                let canAdd = model.canAdd(.lora)
                model.add(.lora)
                let added = model.selection.first ?? ""
                model.setLiteral(.string(pairID), node: added, input: "lora", undoManager: nil)
                let inMenu = lib.loras(for: model.loraFamily(for: model.graph[added]!)).contains { $0.id == pairID }
                model.enqueue(undoManager: nil)
                if let item = model.queue.last {
                    while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                        try? await Task.sleep(nanoseconds: 20_000_000)
                    }
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
                let marks = await wdoc.nodeResults["save"].flatMap { r in r.url }.asyncMap(frameMarks)
                let good = canAdd && inMenu && marks.map { abs($0.top - 0.25) < 0.08 && abs($0.second - 0.25) < 0.08 } == true
                say("Wan LoRA pair from the library: rows \(marks.map { String(format: "%.2f / %.2f", $0.top, $0.second) } ?? "?") (want 0.25 / 0.25); in the menu \(inMenu); \(good ? "ok" : "MISSING SOMETHING \(model.lastError ?? "")")")
                ok = ok && good
                model.closeTab(wdoc.id)
                lib.remove(pairID)
            }

            // A fine-tune from the same kind of link, installed on the server through the library (the fake server
            // keeps the file), then picked in the cloud node's model menu: FLUX.1 dev and Qwen-Image.
            var modelRemote = remote
            modelRemote.kind = .model
            for (family, presetID) in [("flux.1", "cloud-flux"), ("qwen-image", "cloud-qwen-image"),
                                       ("wan-2.2-t2v", "cloud-wan-t2v")] {
                guard let preset = model.preset(id: presetID) else { ok = false; continue }
                var fineTune = modelRemote
                if family.hasPrefix("wan") { fineTune.expert = "low" }  // one expert's file; the other stays the base's
                let modelID = lib.add(fineTune, link: link, title: "Smoke fine-tune \(family)", family: family,
                                      mac: false, servers: [conn.id])
                var waited = 0
                while lib.item(modelID).map({ lib.serverState($0, conn.id) }) != .installed && waited < 300 {
                    try? await Task.sleep(nanoseconds: 100_000_000)
                    waited += 1
                }
                let installedOnServer = lib.item(modelID).map { lib.serverState($0, conn.id) } == .installed
                let fdoc = model.openPreset(preset)
                model.setLiteral(.int(256), node: "cloud", input: "width", undoManager: nil)
                model.setLiteral(.int(256), node: "cloud", input: "height", undoManager: nil)
                model.setLiteral(.string(modelID), node: "cloud", input: "model", undoManager: nil)
                model.enqueue(undoManager: nil)
                if let item = model.queue.last {
                    while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                        try? await Task.sleep(nanoseconds: 20_000_000)
                    }
                }
                try? await Task.sleep(nanoseconds: 300_000_000)
                var right: Float = 1
                if let url = fdoc.nodeResults["save"]?.url {
                    if family.hasPrefix("wan") {
                        right = await frameMarks(url).map { 1 - $0.right } ?? 1
                    } else if let img = try? ImageArray.load(url) {
                        right = abs(ImageArray.array(from: img)[0..., 255, ..<3] - 1).max().item(Float.self)
                    }
                }
                let modelOK = installedOnServer && right < (family.hasPrefix("wan") ? 0.08 : 0.01)
                    && lib.models(for: family).map { $0.id } == [modelID]
                say("\(family) fine-tune from the library: installed on the server \(installedOnServer), right column off by \(right); \(modelOK ? "ok" : "MISSING SOMETHING")")
                ok = ok && modelOK
                model.closeTab(fdoc.id)
                lib.removeFromServer(modelID, conn.id)
                try? await Task.sleep(nanoseconds: 500_000_000)
                lib.remove(modelID)
            }
            try? FileManager.default.removeItem(at: ModelPaths.directory)
            ModelPaths.directory = savedModels
            lib.reload()
            LoRASources.directHosts = []
        }

        // Music, sound and speech: each model's own rate and channels, the length asked for, the graph embedded.
        let audio: [(String, [String: JSONValue], Int, Int, Double?)] = [
            ("cloud-sound", ["seconds": .double(3)], 44100, 2, 3),
            ("cloud-music", ["seconds": .double(10), "lyrics": .string("[Instrumental]")], 48000, 2, 10),
            ("cloud-speech", ["text": .string("Hello from the cloud.")], 24000, 1, nil),
        ]
        for (id, literals, rate, channels, seconds) in audio {
            guard let p = model.preset(id: id) else { say("\(id): missing"); ok = false; continue }
            let doc = model.openPreset(p)
            for (k, v) in literals { model.setLiteral(v, node: "cloud", input: k, undoManager: nil) }
            let began = Date()
            model.enqueue(undoManager: nil)
            guard let item = model.queue.last else { ok = false; continue }
            var statuses: Set<String> = []
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false {
                if let s = doc.nodeStatus["cloud"] { statuses.insert(s) }
                try? await Task.sleep(nanoseconds: 20_000_000)
            }
            try? await Task.sleep(nanoseconds: 300_000_000)
            let finished = model.queue.first { $0.id == item.id }
            guard case .done? = finished?.status, let result = doc.nodeResults["save"]?.url else {
                say("\(id): \(finished.map { "\($0.status)" } ?? "?") \(model.lastError ?? "")"); ok = false; continue
            }
            let clip = try? await MediaIO.readAudio(result)
            let cost = doc.nodeStatus["cloud"] ?? ""
            let embedded = await MediaIO.readMetadata(result)["aistudio.graph"]
            let lengthOK = clip.map { c in seconds.map { abs(c.duration - $0) < 0.1 } ?? (c.duration > 0.5) } ?? false
            let good = clip?.sampleRate == rate && clip?.channels == channels && lengthOK && cost.hasPrefix("GPU ")
                && embedded != nil && model.history.contains { $0.url == result }
            say(String(format: "%@: %.1fs → %@ (%d Hz, %d ch, %.2f s); statuses %@; final \"%@\"; %@", id,
                       Date().timeIntervalSince(began), result.lastPathComponent, clip?.sampleRate ?? 0,
                       clip?.channels ?? 0, clip?.duration ?? 0, statuses.sorted().joined(separator: " | "), cost,
                       good ? "ok" : "MISSING SOMETHING"))
            ok = ok && good
            model.closeTab(doc.id)
        }

        // A Mac-server target refuses Cloud GPU nodes with a clear message.
        let mac = ServerConnection(name: "Smoke Mac", baseURL: "http://127.0.0.1:9")
        model.connections.save(mac, token: "x")
        let doc = model.openPreset(flux)
        doc.backendID = mac.id
        model.enqueue(undoManager: nil)
        if let item = model.queue.last {
            while model.queue.first(where: { $0.id == item.id })?.isFinished == false { try? await Task.sleep(nanoseconds: 50_000_000) }
            let refused: Bool = {
                if case .failed(let m)? = model.queue.first(where: { $0.id == item.id })?.status { return m.contains("This Mac") }
                return false
            }()
            say("Mac server target refused: \(refused)")
            ok = ok && refused
        }
        model.closeTab(doc.id)
        for c in [conn, mac] { model.connections.remove(c.id) }
        return ok
    }
}
