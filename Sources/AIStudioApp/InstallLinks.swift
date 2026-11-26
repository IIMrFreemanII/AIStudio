import Foundation
import NodesStandard

/// `--install-links <file> [--install-server <server id or name>]`: adds each link in the file to the real library
/// and installs it on one GPU server, then waits until every item is installed or has failed. One item per line:
/// `<link>` or `<link>` TAB `<title>`; blank lines and lines starting with `#` are skipped.
enum InstallLinks {
    /// The links of a list file, with the title written after a tab (nil: the app picks one).
    nonisolated static func parse(_ text: String) -> [(link: String, title: String?)] {
        var items: [(link: String, title: String?)] = []
        for raw in text.components(separatedBy: .newlines) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.isEmpty || line.hasPrefix("#") { continue }
            let parts = line.split(separator: "\t", maxSplits: 1, omittingEmptySubsequences: false)
            let link = parts[0].trimmingCharacters(in: .whitespaces)
            if link.isEmpty { continue }
            let title = parts.count > 1 ? parts[1].trimmingCharacters(in: .whitespaces) : ""
            items.append((link, title.isEmpty ? nil : title))
        }
        return items
    }

    /// The title an item gets when the file doesn't give one: the page's title, its version in parentheses.
    nonisolated static func defaultTitle(_ r: LoRARemote) -> String {
        guard let t = r.title, !t.isEmpty else { return r.fileName }
        guard let range = t.range(of: " · ") else { return t }
        return t[..<range.lowerBound] + " (" + t[range.upperBound...] + ")"
    }

    @MainActor
    static func run(_ model: AppModel, file: String, server wanted: String?) async -> Bool {
        func say(_ s: String) { print("[install-links] \(s)"); fflush(stdout) }
        guard let lib = model.store?.library else { say("no library"); return false }
        guard let text = try? String(contentsOfFile: file, encoding: .utf8) else { say("can't read \(file)"); return false }
        let servers = model.connections.gpuServers
        let server: ServerConnection
        if let wanted {
            guard let s = servers.first(where: { $0.id == wanted || $0.name == wanted }) else {
                say("no GPU server with id or name \(wanted)"); return false
            }
            server = s
        } else if servers.count == 1 {
            server = servers[0]
        } else {
            say(servers.isEmpty ? "no GPU server" : "\(servers.count) GPU servers: pass --install-server <id or name>")
            return false
        }
        say("server \(server.name) (\(server.id))")

        var skipped = 0
        var added: [String] = []
        for (link, given) in parse(text) {
            do {
                let r = try await LoRASources.resolve(link, fits: { LoRASources.family(forBaseModel: $0) != nil })
                let family = LoRASources.family(for: r)
                var why: String?
                if family == nil { why = "no family the app runs (base model \(r.baseModel ?? "unknown"))" }
                else if r.kind == .model, let refusal = LinkLibrary.serverModelRefusal(family) { why = refusal }
                else if family?.hasPrefix("wan-2.2") == true, r.kind == .model, r.expert == nil {
                    why = "a Wan 2.2 model whose file doesn't say high or low noise"
                }
                if let why { say("skip \(link): \(why)"); skipped += 1; continue }
                let title = given ?? defaultTitle(r)
                let id = lib.add(r, link: link, title: title, family: family, mac: false, servers: [server.id])
                if !added.contains(id) { added.append(id) }
                say("add \(id) [\(family ?? "?"), \(r.kind.rawValue)] \(title)")
            } catch {
                say("skip \(link): \(error)"); skipped += 1
            }
        }

        var installed = 0, failed = 0
        var pending = Set(added)
        var last: [String: LinkLibrary.State] = [:]
        var lastWorking: [String: Date] = [:]
        let started = Date()
        while !pending.isEmpty {
            for id in pending.sorted() {
                guard let item = lib.item(id) else {
                    say("failed \(id): the entry vanished"); failed += 1; pending.remove(id); continue
                }
                let state = lib.serverState(item, server.id)
                switch state {
                case .installed:
                    say("ok \(id)"); installed += 1; pending.remove(id)
                case .failed(let msg):
                    say("failed \(id): \(msg)"); failed += 1; pending.remove(id)
                case .working(let line):
                    if state != last[id], Date().timeIntervalSince(lastWorking[id] ?? .distantPast) >= 30 {
                        say("\(id): \(line)"); lastWorking[id] = Date()
                    }
                default: break
                }
                last[id] = state
                if pending.contains(id), Date().timeIntervalSince(started) > 90 * 60 {
                    say("failed \(id): timed out after 90 minutes"); failed += 1; pending.remove(id)
                }
            }
            if !pending.isEmpty { try? await Task.sleep(nanoseconds: 2_000_000_000) }
        }
        say("\(installed) installed, \(skipped) skipped, \(failed) failed")
        return failed == 0
    }
}
