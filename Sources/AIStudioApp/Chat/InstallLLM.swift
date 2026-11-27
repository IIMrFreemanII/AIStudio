import Foundation
import LLMClient

/// `--install-llm <link> [--install-server <server id or name>] [--install-id <id>] [--install-gpu <gpu>]
/// [--inspect-only]`: inspects an LLM link on a GPU server (facts, parsers, the GPU plan), then installs it and waits
/// until the download is done. No chat is sent, so no GPU starts; only the download container runs.
enum InstallLLM {
    @MainActor
    static func run(_ model: AppModel, link: String) async -> Bool {
        func say(_ s: String) { print("[install-llm] \(s)"); fflush(stdout) }
        let servers = model.connections.gpuServers
        let wanted = SmokeTest.argument("--install-server")
        guard let server = wanted.flatMap({ w in servers.first { $0.id == w || $0.name == w } })
                ?? (wanted == nil && servers.count == 1 ? servers[0] : nil) else {
            say(servers.isEmpty ? "no GPU server" : "pass --install-server <id or name> (\(servers.count) servers)")
            return false
        }
        await model.connections.test(server.id)
        let llm = model.llm
        guard let client = llm.client(server.id) else { say("no client for \(server.name)"); return false }
        do {
            let found = try await client.inspect(url: link)
            let f = found.facts
            say("\(f.repo)@\(f.revision ?? "main"): \(f.architecture) (\(f.modelType)), \(f.weightBytes / 1_000_000_000) GB, "
                + "vision \(f.vision), context \(f.nativeContext)")
            say("parsers: tools \(found.profile.toolParser ?? "none"), reasoning \(found.profile.reasoningParser ?? "none")"
                + (found.profile.engineArgs.map { ", args \($0.joined(separator: " "))" } ?? ""))
            for n in f.notes ?? [] { say("note: \(n)") }
            say("recommended \(found.plan.recommended?.label ?? "none"), context \(found.plan.context)")
            for o in found.plan.options {
                say("  \(o.label): fits \(o.fits), $\(String(format: "%.2f", o.usdPerHour))/h, max context \(o.maxContext)"
                    + (o.note.map { " (\($0))" } ?? ""))
            }
            if CommandLine.arguments.contains("--inspect-only") { return true }
            var req = LLMInstallRequest(url: link, id: SmokeTest.argument("--install-id"))
            if let gpu = SmokeTest.argument("--install-gpu") { req.gpu = gpu; req.count = 1 }
            let st = try await client.install(req)
            let id = st.id ?? SmokeTest.argument("--install-id") ?? found.suggestedId
            say("installing as \(id)")
            var lastBytes = -1
            while true {
                await llm.refresh(server.id)
                if llm.listings[server.id]?.models.contains(where: { $0.id == id && $0.installed == true }) == true { break }
                if let s = llm.listings[server.id]?.installs[id] {
                    if s.state == "failed" { say("install failed: \(s.error ?? "?")"); return false }
                    if let b = s.bytes, b / 1_000_000_000 != lastBytes / 1_000_000_000 {
                        lastBytes = b
                        say("\(s.state) \(b / 1_000_000_000) / \((s.total ?? 0) / 1_000_000_000) GB")
                    }
                }
                try? await Task.sleep(for: .seconds(15))
            }
            let m = llm.model(LLMKey(serverID: server.id, modelID: id))
            say("installed \(id) on \(m?.gpuLabel ?? "?"), context \(m?.context ?? 0), vision \(m?.vision ?? false)")
            return true
        } catch {
            say("failed: \(error.localizedDescription)"); return false
        }
    }
}
