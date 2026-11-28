import Agent
import CharacterCore
import Foundation
import LLMClient
import MovieCore

/// One prompt-engine run, headless: the agent loop with the prompt tools on the engine's vision LLM, from an idea to
/// a checked draft. Each draft waits on a cost sheet (`app.promptEngineGate`, or `gate` for a host that shows its own).
@MainActor @Observable
final class PromptEngineSession: PromptToolHost {
    weak var app: AppModel?
    let record = PromptEngineRecord()
    let idea: String
    let video: Bool
    let fixedModel: PromptModel?
    let characterID: UUID?
    let llm: LLMKey
    /// Presses Run on every cost sheet (the harness, after the user approved the run).
    var autoRun = false
    /// A scripted model instead of the LLM (the fake-server smoke test).
    var clientOverride: (any ChatStreaming)?
    /// The cost sheet waiting for the user, when the host shows it itself (`showsOwnGate`: the Prompt Builder's
    /// Engine pane, in its own window or sheet); otherwise it goes to `app.promptEngineGate` (the main window).
    var gate: CostGate?
    var showsOwnGate = false
    /// The step titles, newest last ("Reading the rules for sdxl").
    private(set) var steps: [String] = []
    /// The model's last words (its one-line summary after `finish`).
    private(set) var reply = ""
    private(set) var running = false
    private(set) var error: String?
    private var task: Task<Void, Never>?

    init(app: AppModel, idea: String, video: Bool = false, fixedModel: PromptModel? = nil, characterID: UUID? = nil,
         llm: LLMKey) {
        self.app = app
        self.idea = idea
        self.video = video
        self.fixedModel = fixedModel
        self.characterID = characterID
        self.llm = llm
    }

    var character: String? { characterID.flatMap { app?.character($0)?.name } }

    func start() {
        guard task == nil else { return }
        task = Task { await run() }
    }

    func stop() {
        task?.cancel()
        gate = nil
        if app?.promptEngineGate != nil { app?.promptEngineGate = nil }
    }

    /// Waits for the run to end (the harness).
    func wait() async { await task?.value }

    func approve(_ estimate: CostEstimate) async throws -> Bool {
        if autoRun { return true }
        guard let app else { return false }
        final class Pressed { var run = false }
        let pressed = Pressed()
        let g = CostGate(estimate: estimate, run: { pressed.run = true })
        let own = showsOwnGate
        if own { gate = g } else { app.promptEngineGate = g }
        func waiting() -> Bool { own ? gate != nil : app.promptEngineGate != nil }
        do {
            while waiting() { try await Task.sleep(for: .milliseconds(200)) }
        } catch {
            if own { gate = nil } else { app.promptEngineGate = nil }
            throw error
        }
        return pressed.run
    }

    private func run() async {
        running = true
        defer { running = false; task = nil }
        guard let app else { return }
        var config = AgentConfig(model: llm.modelID, tools: PromptTools.tools(self), mode: .acceptEdits)
        config.maxSteps = 40
        let chat: any ChatStreaming
        if let scripted = clientOverride {
            chat = scripted
            config.vision = true
        } else {
            await app.llm.refresh(llm.serverID)
            guard let client = app.llm.client(llm.serverID), let m = app.llm.model(llm) else {
                error = "The prompt engine's language model isn't installed."
                return
            }
            chat = client.chat
            config.vision = m.vision == true
            config.inlineThinking = m.reasoningParser == nil
            config.contextLimit = m.context
            if (m.efforts ?? []).contains("on") { config.reasoningEffort = "on" }
        }
        var user = "The idea: \(idea)"
        if let fixedModel { user += "\nThe model is fixed: \(fixedModel.rawValue)." }
        if let character { user += "\nDraw \(character) (their face reference and LoRAs come with every draft)." }
        let messages: [ChatMessage] = [
            .system(PromptEngine.systemPrompt(video: video, fixed: fixedModel, character: character)), .user(user),
        ]
        record.status = "Starting the language model…"
        let ctx = ToolContext(root: app.library.root)
        let record = self.record
        do {
            let outcome = try await AgentLoop.run(client: chat, messages: messages, config: config, context: ctx,
                                                  ask: { _ in .deny }, events: { [weak self] e in
                await MainActor.run {
                    guard let self else { return }
                    switch e {
                    case .toolStarted(_, let preview):
                        self.steps.append(preview.title)
                        if !preview.title.hasPrefix("Drafting") { record.status = preview.title + "…" }
                    case .assistantFinished(let content, _, _):
                        let t = content.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !t.isEmpty { self.reply = t }
                    case .delta(.stats(let s)):
                        record.llmUSD += s.costUSD ?? 0
                    case .delta(.status(let st)):
                        record.status = "Language model: \(st.phase ?? st.state)"
                    default: break
                    }
                }
            })
            if record.result == nil {
                record.finish(best: nil, notes: outcome == .stepLimit ? "Stopped at the step limit." : reply)
            }
            record.status = "Done"
        } catch is CancellationError {
            if record.result == nil { record.finish(best: nil, notes: "Stopped.") }
            record.status = "Stopped"
        } catch {
            self.error = error.localizedDescription
            if record.result == nil { record.finish(best: nil, notes: "Failed: \(error.localizedDescription)") }
            record.status = "Failed"
        }
    }
}

extension AppModel {
    static let promptEngineLLMKey = "promptEngineLLM"
    static let promptEngineLLMUpgradedKey = "promptEngineLLMUpgraded"  // a saved bf16 pick was moved to NVFP4 once

    /// The engine's LLM: the one picked last (if still installed and with vision; a pick of the bf16 Qwen3.8-27B
    /// moves once to its NVFP4 copy), else that copy, else the bf16 model, else the first installed vision model
    /// (PromptEngineLLM).
    var defaultPromptEngineLLM: LLMKey? {
        let vision = llm.listings.flatMap { server, listing in
            listing.models.filter { $0.installed == true && $0.vision == true }.map { LLMKey(serverID: server, modelID: $0.id) }
        }
        let keys = vision.map { "\($0.serverID)/\($0.modelID)" }
        let d = UserDefaults.standard
        let (pick, save) = PromptEngineLLM.pick(installed: keys, saved: d.string(forKey: Self.promptEngineLLMKey),
                                               upgraded: d.bool(forKey: Self.promptEngineLLMUpgradedKey))
        if let save, persistSettings {
            d.set(save, forKey: Self.promptEngineLLMKey)
            d.set(true, forKey: Self.promptEngineLLMUpgradedKey)
        }
        return pick.flatMap { p in vision.first { "\($0.serverID)/\($0.modelID)" == p } }
    }

    func setPromptEngineLLM(_ key: LLMKey) {
        guard persistSettings else { return }
        UserDefaults.standard.set("\(key.serverID)/\(key.modelID)", forKey: Self.promptEngineLLMKey)
    }
}
