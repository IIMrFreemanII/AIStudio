import Foundation
import MLXDiffusionCore

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
