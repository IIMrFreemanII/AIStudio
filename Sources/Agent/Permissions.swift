import Foundation

/// What a tool does, for the permission rules.
public enum ToolEffect: String, Codable, Sendable {
    case read, edit, execute
    /// Changes the open workflow (live, one undo step per reply): free except in plan mode.
    case graphEdit
    /// Queues a workflow run (spends time, maybe GPU money): asks like a command.
    case run
    /// Reads from the internet (`web_fetch`, `web_search`): asks in Ask mode unless the domain is always-allowed;
    /// free in the other modes (plan mode included: it only reads).
    case network
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

/// "Always allow" answers given in a chat: all edits, commands starting with a prefix, workflow runs, web domains.
public struct AllowRules: Codable, Equatable, Sendable {
    public var edits = false
    public var commandPrefixes: [String] = []
    public var runs = false
    /// Hosts `web_fetch` may read without asking (`example.com` also allows `docs.example.com`).
    public var domains: [String] = []

    public init(edits: Bool = false, commandPrefixes: [String] = [], runs: Bool = false, domains: [String] = []) {
        self.edits = edits
        self.commandPrefixes = commandPrefixes
        self.runs = runs
        self.domains = domains
    }

    func allows(domain: String) -> Bool {
        let d = domain.lowercased()
        return domains.contains { d == $0 || d.hasSuffix("." + $0) }
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
    /// - `domain`: the host a `.network` call reads from.
    public static func decide(effect: ToolEffect, inProject: Bool, command: String? = nil, domain: String? = nil,
                              mode: PermissionMode, rules: AllowRules) -> PermissionDecision {
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
            if let command, GitSafety.risk(command) != nil { return .ask }
            if let command, rules.allows(command: command) { return .allow }
            return .ask
        case .graphEdit:
            if mode == .plan { return .deny("Plan mode is read only: describe the change to the workflow instead of making it.") }
            return .allow
        case .run:
            if mode == .plan { return .deny("Plan mode is read only: workflows aren't run.") }
            return rules.runs ? .allow : .ask
        case .network:
            if mode != .ask { return .allow }
            if let domain, rules.allows(domain: domain) { return .allow }
            return .ask
        }
    }
}

/// Git and GitHub commands that ask every time, whatever the "always allow" rules say.
public enum GitSafety {
    /// Why a command needs the user's OK every time (shown on the permission card), nil when it's an ordinary command.
    public static func risk(_ command: String) -> String? {
        let parts = command.components(separatedBy: CharacterSet(charactersIn: ";|&\n"))
        for part in parts {
            let tokens = part.split(whereSeparator: { $0 == " " || $0 == "\t" }).map {
                String($0).trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            }
            if let r = risk(tokens) { return r }
        }
        return nil
    }

    private static func risk(_ tokens: [String]) -> String? {
        guard let first = tokens.first else { return nil }
        if first == "gh" {
            guard tokens.count >= 3, tokens[1] == "pr" else { return nil }
            switch tokens[2] {
            case "create": return "It opens a pull request on GitHub."
            case "merge": return "It merges a pull request on GitHub."
            case "close": return "It closes a pull request on GitHub."
            default: return nil
            }
        }
        guard first == "git" else { return nil }
        var i = 1
        while i < tokens.count, tokens[i].hasPrefix("-") { i += (tokens[i] == "-C" || tokens[i] == "-c") ? 2 : 1 }
        guard i < tokens.count else { return nil }
        let sub = tokens[i]
        let args = Array(tokens[(i + 1)...])
        func has(_ flags: String...) -> Bool { args.contains { flags.contains($0) } }
        // A short-flag cluster such as -fd or -xdf.
        func cluster(_ letter: Character) -> Bool {
            args.contains { $0.hasPrefix("-") && !$0.hasPrefix("--") && $0.dropFirst().contains(letter) }
        }

        if sub == "push", has("-f", "--force", "--force-if-includes") || cluster("f")
            || args.contains(where: { $0.hasPrefix("--force-with-lease") || $0.hasPrefix("+") }) {
            return "Force-pushing rewrites the remote branch's history."
        }
        if ["commit", "push", "merge", "rebase"].contains(sub), has("--no-verify") { return "It skips the repository's hooks." }
        if sub == "push" { return "It sends commits to the remote." }
        if sub == "reset", has("--hard") { return "It discards uncommitted changes." }
        if sub == "clean", has("--force") || cluster("f") { return "It deletes untracked files." }
        if sub == "checkout" {
            let dashes = args.firstIndex(of: "--")
            if (dashes.map { $0 + 1 < args.count } ?? false) || has(".", "-f", "--force") { return "It discards changes to files." }
        }
        if sub == "restore", !(has("--staged", "-S") && !has("--worktree", "-W")) { return "It discards changes to files." }
        if sub == "branch", has("-D") || (has("--delete", "-d") && has("--force", "-f")) {
            return "It deletes a branch even if it isn't merged."
        }
        if sub == "stash", let a = args.first, a == "drop" || a == "clear" { return "It deletes stashed changes." }
        if ["rebase", "filter-branch", "filter-repo"].contains(sub) { return "It rewrites commit history." }
        if sub == "commit", has("--amend") { return "It rewrites the last commit." }
        if sub == "config" {
            let writes = has("--unset", "--unset-all", "--add", "--replace-all", "--edit", "-e", "--remove-section",
                             "--rename-section")
            let reads = has("--get", "--get-all", "--list", "-l") || args.filter { !$0.hasPrefix("-") }.count == 1
            if writes || !reads { return "It changes git settings." }
        }
        return nil
    }
}
