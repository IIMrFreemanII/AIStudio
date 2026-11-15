import Foundation
import LLMClient

/// GPU fallback: when a model's start waits more than 30 s for its GPU, the chat moves it to the next preset in the
/// model's list (Settings › GPUs, Context and Start-up › "If the GPU doesn't come"), and once the reply begins the
/// model's own GPU is put back for its next start. Every chat stream, the agents and Movie Studio's writer use it.
extension LLMCatalog {
    static let gpuWaitLimit: Double = 30

    static func fallbackKey(_ key: LLMKey) -> String { key.serverID + "|" + key.modelID }

    func fallbacks(_ key: LLMKey) -> [GPUPreset] { gpuFallbacks[Self.fallbackKey(key)] ?? [] }

    func setFallbacks(_ list: [GPUPreset], for key: LLMKey) {
        gpuFallbacks[Self.fallbackKey(key)] = list.isEmpty ? nil : list
    }

    func gpuWaitPolicy(_ serverID: String) -> GPUWaitPolicy {
        GPUWaitPolicy(limit: Self.gpuWaitLimit,
                      switchGPU: { [weak self] model in await self?.switchGPU(LLMKey(serverID: serverID, modelID: model)) },
                      settle: { [weak self] model in await self?.settleGPU(LLMKey(serverID: serverID, modelID: model)) },
                      switchedRecently: { [weak self] model in
                          await self?.switchedRecently(LLMKey(serverID: serverID, modelID: model)) ?? false
                      })
    }

    func switchedRecently(_ key: LLMKey) -> Bool {
        switchTasks[key] != nil || lastSwitch[key].map { Date().timeIntervalSince($0) < 60 } == true
    }

    /// Moves a model that's still waiting for its GPU to the next preset in its list; the new preset's label, or nil
    /// when there's none left (the start keeps waiting where it is). Chats waiting on the same model share one move.
    func switchGPU(_ key: LLMKey) async -> String? {
        if let t = switchTasks[key] { return await t.value }
        // Another chat moved it a moment ago: this one just asks again.
        if let at = lastSwitch[key], Date().timeIntervalSince(at) < 20, let m = model(key) { return GPUPreset(m).label }
        let t = Task { @MainActor in await self.moveToNextGPU(key) }
        switchTasks[key] = t
        let label = await t.value
        switchTasks[key] = nil
        return label
    }

    private func moveToNextGPU(_ key: LLMKey) async -> String? {
        let list = fallbacks(key)
        guard !list.isEmpty, let m = model(key), let client = client(key.serverID) else { return nil }
        var run = fallbackRuns[key] ?? (original: m, tried: [])
        let current = GPUPreset(m)
        run.tried.insert(current)
        guard let next = GPUFallback.next(in: list, current: current, tried: run.tried) else {
            gpuNotes[key] = "No other GPU in its list: waiting for \(current.label)"
            return nil
        }
        // The waiting start is cancelled first (nothing was billed: no GPU yet), then the model's GPU changes.
        await stop(key)
        for _ in 0..<20 where status(key)?.isActive == true {
            try? await Task.sleep(for: .milliseconds(500))
            await refreshStatus(key)
        }
        do {
            _ = try await client.update(key.modelID, GPUFallback.changes(to: next, from: run.original))
        } catch {
            gpuNotes[key] = "Couldn't switch to \(next.label): \(error.localizedDescription)"
            return nil
        }
        run.tried.insert(next)
        fallbackRuns[key] = run
        lastSwitch[key] = Date()
        gpuNotes[key] = "\(current.label) didn't come within \(Int(Self.gpuWaitLimit)) s: this start uses \(next.label)"
        await refresh(key.serverID)
        return next.label
    }

    /// The reply began (or the chat ended): the model's own GPU and start-up settings go back for its next start (the
    /// running container keeps going on the preset it got).
    func settleGPU(_ key: LLMKey) async {
        if let t = switchTasks[key] { _ = await t.value }
        guard let run = fallbackRuns.removeValue(forKey: key), let client = client(key.serverID) else { return }
        _ = try? await client.update(key.modelID, GPUFallback.restore(run.original))
        await refresh(key.serverID)
        if let note = gpuNotes[key], note.contains("this start uses") {
            gpuNotes[key] = note.replacingOccurrences(of: "this start uses", with: "running on")
                + ". Next start tries \(GPUPreset(run.original).label) first."
        }
    }
}
