import Foundation

/// What a tool does, for the permission rules.
public enum ToolEffect: String, Codable, Sendable {
    case read, edit, execute
}

/// How much the agent may do without asking, as in Claude Code.
public enum PermissionMode: String, Codable, CaseIterable, Sendable {
    /// Reads in the project are free; edits and commands ask.
    case ask
    /// Edits in the project are free too; commands still ask.
    case acceptEdits
    /// Read only: edits and commands are refused (the agent plans instead).
    case plan
    /// Everything is allowed, without asking.
    case bypass

    public var title: String {
        switch self {
        case .ask: return "Ask"
        case .acceptEdits: return "Auto-accept Edits"
        case .plan: return "Plan (read only)"
        case .bypass: return "Bypass Permissions"
        }
    }

    public var detail: String {
        switch self {
        case .ask: return "Reads files freely; asks before each edit and command."
        case .acceptEdits: return "Edits files in the project without asking; asks before commands."
        case .plan: return "Only reads and searches; edits and commands are refused."
        case .bypass: return "Edits and runs anything without asking. Only for projects you can afford to lose."
        }
    }
}

public enum PermissionDecision: Equatable, Sendable {
    case allow
    case ask
    case deny(String)
}

/// "Always allow" answers given in a chat: all edits, or commands starting with a prefix.
public struct AllowRules: Codable, Equatable, Sendable {
    public var edits = false
    public var commandPrefixes: [String] = []

    public init(edits: Bool = false, commandPrefixes: [String] = []) {
        self.edits = edits
        self.commandPrefixes = commandPrefixes
    }

    /// The prefix "Always allow" remembers for a command: its first word, or two for tools with subcommands
    /// (`git status`, `swift build`, `npm run`…).
    public static func prefix(of command: String) -> String {
        let words = command.trimmingCharacters(in: .whitespaces).split(separator: " ", omittingEmptySubsequences: true)
            .map(String.init)
        guard let first = words.first else { return "" }
        let twoWord: Set<String> = ["git", "swift", "npm", "pnpm", "yarn", "cargo", "go", "make", "xcodebuild", "python",
                                    "python3", "uv", "pip", "docker", "kubectl", "brew", "gh"]
        if twoWord.contains(first), words.count > 1, !words[1].hasPrefix("-") { return "\(first) \(words[1])" }
        return first
    }

    func allows(command: String) -> Bool {
        // A command chained with ;, &&, | or $(…) is only allowed when every part is.
        let parts = command.components(separatedBy: CharacterSet(charactersIn: ";|&\n"))
            .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        // Substitutions and redirections can do anything: they always ask.
        guard !parts.isEmpty, !command.contains("$("), !command.contains("`"), !command.contains(">") else { return false }
        return parts.allSatisfy { part in
            commandPrefixes.contains { part == $0 || part.hasPrefix($0 + " ") }
        }
    }
}

/// Decides whether a tool call may run, must ask, or is refused.
public enum PermissionEngine {
    /// - `inProject`: every path the call touches is inside the project folder.
    public static func decide(effect: ToolEffect, inProject: Bool, command: String? = nil, mode: PermissionMode,
                              rules: AllowRules) -> PermissionDecision {
        if mode == .bypass { return .allow }
        switch effect {
        case .read:
            return inProject ? .allow : .ask
        case .edit:
            if mode == .plan { return .deny("Plan mode is read only: describe the change instead of making it.") }
            if !inProject { return .ask }
            return mode == .acceptEdits || rules.edits ? .allow : .ask
        case .execute:
            if mode == .plan { return .deny("Plan mode is read only: commands don't run.") }
            if let command, rules.allows(command: command) { return .allow }
            return .ask
        }
    }
}
