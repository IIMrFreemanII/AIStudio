import Foundation

/// A fenced code block of shell commands (```bash, ```console …): what a chat's Run button types into the terminal.
public enum ShellBlock {
    /// Languages whose blocks are commands (the info string's first word).
    static let shells: Set<String> = ["bash", "sh", "zsh", "shell", "console", "terminal", "shell-session"]
    /// Transcripts: commands after a `$ ` or `% ` prompt, the other lines are their output.
    static let transcripts: Set<String> = ["console", "terminal", "shell-session"]

    /// The commands to run, one per line, or nil when the block isn't shell commands (another language, or nothing
    /// to run). `language` is the fence's whole info string ("bash title=x" counts as bash).
    public static func commands(language: String, code: String) -> String? {
        let lang = language.split(whereSeparator: \.isWhitespace).first.map { $0.lowercased() } ?? ""
        guard shells.contains(lang) else { return nil }
        let lines = code.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var commands: [String]
        if transcripts.contains(lang) {
            commands = lines.compactMap { line in
                let l = line.drop { $0 == " " }
                guard l.hasPrefix("$ ") || l.hasPrefix("% ") else { return nil }
                return String(l.dropFirst(2))
            }
        } else {
            commands = lines.map { line in
                let l = line.drop { $0 == " " }
                return l.hasPrefix("$ ") ? String(l.dropFirst(2)) : line
            }
        }
        while let f = commands.first, f.trimmingCharacters(in: .whitespaces).isEmpty { commands.removeFirst() }
        while let l = commands.last, l.trimmingCharacters(in: .whitespaces).isEmpty { commands.removeLast() }
        return commands.isEmpty ? nil : commands.joined(separator: "\n")
    }
}
