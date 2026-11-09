import Foundation
import GraphRuntime
import MLXDiffusionCore

/// Enhance / Describe on a Text Encode node: runs the model's language model directly (not through the queue) and
/// streams the answer into the first Text Encode's prompt. The whole replacement is one undo step.
extension AppModel {
    /// User-edited system prompts, keyed like ``PromptAssistant/messages(mode:prompt:hasImage:instruction:overrides:)``.
    static let assistantOverrideKeys = ["enhance", "edit", "describe"]

    static func assistantDefaultsKey(_ key: String) -> String { "assistant.system.\(key)" }

    var assistantOverrides: [String: String] {
        guard persistSettings else { return [:] }
        var out: [String: String] = [:]
        for k in Self.assistantOverrideKeys {
            if let s = UserDefaults.standard.string(forKey: Self.assistantDefaultsKey(k)) { out[k] = s }
        }
        return out
    }

    /// The Text Encode node the assistant writes into (`node`, else the first one).
    func assistantTarget(_ node: String?) -> GraphNode? {
        if let node { return graph[node].flatMap { $0.type == "TextEncode" ? $0 : nil } }
        return firstNode(ofType: "TextEncode")
    }

    /// The image linked to a Text Encode's `image` input (the reference it edits), if it is a Load Image.
    func assistantImage(for te: GraphNode) -> String? {
        guard case .link(let src, _)? = te.inputs["image"], let n = graph[src], n.type == "LoadImage",
              case .value(.string(let p))? = n.inputs["path"], !p.isEmpty else { return nil }
        return p
    }

    func canEnhance(_ node: String? = nil) -> Bool {
        guard assistantMode == nil, let te = assistantTarget(node) else { return false }
        if case .link? = te.inputs["prompt"] { return false }  // driven by another node
        return !(literal(te.id, "prompt")?.description ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func canDescribe(_ node: String? = nil) -> Bool {
        guard assistantMode == nil, let te = assistantTarget(node) else { return false }
        if case .link? = te.inputs["prompt"] { return false }
        return assistantImage(for: te) != nil
    }

    /// Starts Enhance or Describe on a Text Encode (`node`, else the first). `completion` (for the smoke test) gets
    /// the final prompt, or nil on failure.
    func runAssistant(_ mode: PromptAssistant.Mode, node: String? = nil, undoManager: UndoManager?,
                      completion: ((String?) -> Void)? = nil) {
        guard assistantMode == nil, let te = assistantTarget(node) else { completion?(nil); return }
        let current = literal(te.id, "prompt")?.description ?? ""
        let image = assistantImage(for: te)
        let doc = active
        if mode == .describe && image == nil { completion?(nil); return }
        // Describe starts from the image alone; the old prompt usually belongs to something else.
        let (system, user) = PromptAssistant.messages(mode: mode, prompt: mode == .describe ? "" : current,
                                                      hasImage: image != nil, overrides: assistantOverrides)
        let options = TextGenerationOptions(maxTokens: 320, temperature: mode == .describe ? 0.3 : 0.7,
                                            seed: UInt64.random(in: 0...UInt64(UInt32.max)))
        let start = graph
        let run = UUID()
        assistantRun = run
        assistantMode = mode
        lastError = nil
        let path = assistantModelPath
        let nodeID = te.id
        assistantNode = nodeID
        assistantTask = Task { [weak self] in
            guard let self else { return }
            var result: String?
            do {
                let text = try await engine.generateText(
                    modelPath: path, system: system, user: user, imagePaths: image.map { [$0] } ?? [],
                    options: options) { partial in
                    Task { @MainActor [weak self] in
                        guard let self, self.assistantRun == run else { return }
                        self.setPromptLive(PromptAssistant.clean(partial), node: nodeID, in: doc)
                    }
                }
                result = PromptAssistant.clean(text)
            } catch is CancellationError {
            } catch {
                lastError = "Prompt assistant: \(error)"
            }
            assistantRun = nil
            if let result, !result.isEmpty { setPromptLive(result, node: nodeID, in: doc) }
            if doc.graph != start {
                commitLiveEdit(from: start, in: doc, actionName: mode == .describe ? "Describe Image" : "Enhance Prompt",
                               undoManager: undoManager)
            }
            assistantMode = nil
            assistantNode = nil
            assistantTask = nil
            completion?(result)
        }
    }

    /// Stops the assistant; the text written so far stays (one undo step restores the previous prompt).
    func stopAssistant() { assistantTask?.cancel() }

    private func setPromptLive(_ text: String, node: String, in doc: WorkflowDocument) {
        guard var n = doc.graph[node] else { return }
        n.inputs["prompt"] = .value(.string(text))
        doc.graph[node] = n
    }
}

extension AppModel {
    /// The model that writes prompts: the image model when it has a language model (Qwen-Image), else the Qwen-Image
    /// folder (Flux can't generate text).
    var assistantModelPath: String {
        ImageFamily.of(path: modelPath) == .flux ? AppModel.guessModelPath() : modelPath
    }
}
