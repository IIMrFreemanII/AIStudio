import Foundation

/// `MLXComfyApp --smoke-test "<prompt>"` queues one generation through the same path as the Generate button
/// (AppModel → Engine → events), reports what the UI received, and exits.
@MainActor
enum SmokeTest {
    static var requestedPrompt: String? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--smoke-test"), i + 1 < args.count else { return nil }
        return args[i + 1]
    }

    static func run(_ model: AppModel, prompt: String) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
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
