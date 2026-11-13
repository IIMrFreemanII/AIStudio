import Agent
import Foundation

/// The project map (graphify) for coding chats: the code_graph tool, keeping the graph current, and installing it.
extension AppModel {
    /// Coding chats get code_graph when the setting is on (the default) and graphify is installed.
    var codeGraphEnabled: Bool {
        (UserDefaults.standard.object(forKey: "codeGraph") as? Bool ?? true) && ProjectGraphs.findBinary() != nil
    }

    /// Brings the project's map up to date in the background (incremental: only changed files are parsed again), so
    /// the agent's first query doesn't wait and edits made outside the agent are picked up.
    func refreshCodeGraph(_ doc: ChatDocument) {
        guard doc.isCoding, let root = doc.projectURL, codeGraphEnabled else { return }
        Task.detached(priority: .utility) {
            await ProjectGraphs.shared.markStale(root: root)
            await ProjectGraphs.shared.warm(root: root)
        }
    }

    /// After a tool that can change files, the map is rebuilt before its next query.
    func codeGraphToolFinished(_ doc: ChatDocument, callID: String) {
        guard doc.isCoding, let root = doc.projectURL else { return }
        let name = doc.transcript.turns.lazy.flatMap(\.toolCalls).first { $0.id == callID }?.name
        guard let name, ["edit_file", "write_file", "run_command"].contains(name) else { return }
        Task { await ProjectGraphs.shared.markStale(root: root) }
        projectFilesChanged(doc)
    }
}

enum CodeGraphInstall {
    /// `uv`, which installs graphify as a tool (`uv tool install graphifyy`), if it's on the login PATH.
    static var uvPath: String? {
        ShellRunner.loginPath.split(separator: ":").lazy
            .map { URL(fileURLWithPath: String($0)).appendingPathComponent("uv").path }
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    /// Installs graphify with uv; returns nil on success, else the error output's last lines.
    static func install() async -> String? {
        guard let uv = uvPath else { return "uv isn't installed" }
        let r = await ShellRunner().run("\(shellQuoteArg(uv)) tool install graphifyy", cwd: URL(fileURLWithPath: NSHomeDirectory()),
                                        timeout: 600)
        await ProjectGraphs.shared.resetBinary()
        if r.status == 0, ProjectGraphs.findBinary() != nil { return nil }
        let tail = r.output.split(separator: "\n").suffix(4).joined(separator: "\n")
        return tail.isEmpty ? "graphify was installed but isn't on your PATH" : tail
    }

    private static func shellQuoteArg(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }
}
