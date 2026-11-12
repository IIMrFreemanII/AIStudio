import Foundation

/// A text file attached to a message (its contents go to the model inside the message).
public struct ChatFile: Codable, Equatable, Sendable, Hashable {
    public var name: String
    public var text: String
    public init(name: String, text: String) {
        self.name = name
        self.text = text
    }
}

/// What a message replies to: a passage of an earlier message (its role, and its id when it's known).
public struct ChatQuote: Codable, Equatable, Sendable, Hashable {
    public static let maxLength = 2000
    public var turnID: UUID?
    public var role: ChatTurn.Role
    public var text: String
    public init(turnID: UUID?, role: ChatTurn.Role, text: String) {
        self.turnID = turnID
        self.role = role
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        self.text = t.count > Self.maxLength ? String(t.prefix(Self.maxLength - 1)) + "…" : t
    }

    /// As Markdown, for the model: each line behind `> `.
    public var markdown: String {
        text.split(separator: "\n", omittingEmptySubsequences: false).map { "> " + $0 }.joined(separator: "\n")
    }
}

/// Where a session works when it has its own git worktree.
public struct SessionWorktree: Codable, Equatable, Sendable, Hashable {
    public var repo: String
    public var path: String
    public var branch: String
    public var base: String
    public init(repo: String, path: String, branch: String, base: String) {
        self.repo = repo
        self.path = path
        self.branch = branch
        self.base = base
    }
}

/// A child session's report, posted into the session that started it.
public struct SessionEvent: Codable, Equatable, Sendable, Hashable {
    public var childID: UUID
    public var title: String
    public var status: String
    public var branch: String?
    public var report: String
    public init(childID: UUID, title: String, status: String, branch: String? = nil, report: String) {
        self.childID = childID
        self.title = title
        self.status = status
        self.branch = branch
        self.report = report
    }

    /// What the model gets for the report.
    public var textForModel: String {
        "[Session \u{201C}\(title)\u{201D} \(status == "done" ? "finished" : status)\(branch.map { " on branch \($0)" } ?? ""): \(report)]"
    }
}

/// One message of a saved chat, with what the app shows around it (thinking, stats, errors).
public struct ChatTurn: Codable, Identifiable, Equatable, Sendable {
    public enum Role: String, Codable, Sendable { case user, assistant, tool }

    public var id: UUID
    /// The turn it answers or follows (nil: the first). Edits and retries are siblings with the same parent.
    public var parent: UUID?
    public var role: Role
    public var text: String
    /// The model's thinking (assistant turns of thinking models).
    public var reasoning: String?
    /// How long it thought, in seconds.
    public var thinkingSeconds: Double?
    /// Attached images and videos (`.mp4`, see `isVideo`): file names in the chat's files folder.
    public var images: [String]
    /// Attached text files.
    public var files: [ChatFile]?
    public var toolCalls: [ToolCall]
    public var toolCallID: String?
    /// Tool results: what the call did ("Edit README.md"), how it ended (done, failed, denied), and its diff or
    /// command for the card.
    public var toolTitle: String?
    public var toolStatus: String?
    public var toolDetail: String?
    /// The model that wrote it (assistant turns).
    public var model: String?
    public var stats: RequestStats?
    /// Why the reply stopped early (a failed request, a cancel). The turn isn't sent back to the model.
    public var error: String?
    public var date: Date
    /// Assistant turns: the follow-ups the model suggested (taken out of its text, see `FollowUps`).
    public var followUps: [String]?
    /// User turns: the passage it replies to.
    public var quote: ChatQuote?
    /// User turns: what the model gets besides the text, not shown (the chat's widgets' state).
    public var context: String?
    /// A summary of everything before it (the context was filling up, or the user asked to compact). A user turn; the
    /// model gets only it and what follows. Shown as a divider.
    public var summary: Bool?
    /// A user turn: a child session's report (the model gets it as a bracketed note).
    public var sessionEvent: SessionEvent?

    public init(id: UUID = UUID(), role: Role, text: String, reasoning: String? = nil, images: [String] = [],
                files: [ChatFile]? = nil, toolCalls: [ToolCall] = [], toolCallID: String? = nil, model: String? = nil,
                date: Date = Date()) {
        self.id = id
        self.role = role
        self.text = text
        self.reasoning = reasoning
        self.images = images
        self.files = files
        self.toolCalls = toolCalls
        self.toolCallID = toolCallID
        self.model = model
        self.date = date
    }

    /// What a summary turn's text follows, for the model.
    public static let summaryPrefix = "Summary of the earlier conversation (older messages were compacted to save context):\n\n"

    /// Whether an attachment is a video (the app saves them as `<sha256>.mp4`).
    public static func isVideo(_ name: String) -> Bool { name.lowercased().hasSuffix(".mp4") }

    /// The text the model gets for a user message: the passage it replies to, the message and its files, then the
    /// hidden context.
    public var textForModel: String {
        var parts: [String] = []
        if let quote { parts.append("Replying to this part of \(quote.role == .user ? "my earlier message" : quote.role == .tool ? "a tool result" : "your reply"):\n" + quote.markdown) }
        parts.append(textWithFiles)
        if let context, !context.isEmpty { parts.append(context) }
        return parts.filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// The text the model gets: the message, then each attached file in a fenced block.
    public var textWithFiles: String {
        guard let files, !files.isEmpty else { return text }
        let blocks = files.map { f -> String in
            let ext = (f.name as NSString).pathExtension
            return "`\(f.name)`:\n```\(ext)\n\(f.text)\n```"
        }
        return ([text] + blocks).filter { !$0.isEmpty }.joined(separator: "\n\n")
    }
}

/// A saved conversation: which server and model it talks to, and its turns as a tree. Editing a message or asking
/// for another reply adds a sibling; `activeLeaf` picks the branch shown and sent (`turns`).
public struct ChatTranscript: Codable, Identifiable, Equatable, Sendable {
    public var version = 2
    public var id: UUID
    public var title: String
    public var created: Date
    public var updated: Date
    /// The GPU server connection and the model id on it.
    public var serverID: String?
    public var modelID: String?
    public var systemPrompt: String?
    /// "low", "medium" or "high" (nil: the model's default).
    public var reasoningEffort: String?
    /// A coding chat's project folder (nil: a plain chat, no tools).
    public var projectPath: String?
    /// A coding chat's permission mode ("ask", "acceptEdits", "plan", "bypass") and its "Always allow" answers.
    public var permissionMode: String?
    public var allowEdits: Bool?
    public var allowCommands: [String]?
    /// A graph helper chat: its tools edit and run a workflow tab (the one it's docked on, saved by that tab).
    public var graphHelper: Bool?
    /// "Always allow" for a graph helper's workflow runs.
    public var allowRuns: Bool?
    /// "Always allow" for web domains (`web_fetch`).
    public var allowDomains: [String]?
    /// Sessions: pinned and archived in the list, when it was last looked at.
    public var pinned: Bool?
    public var archived: Bool?
    public var lastViewed: Date?
    /// The session that started this one (or that it was forked from).
    public var parentID: UUID?
    /// Its own git worktree, if it has one.
    public var worktree: SessionWorktree?
    /// The Changes pane is open (coding chats).
    public var changesOpen: Bool?
    /// The task it was started with.
    public var brief: String?
    /// 0 or nil for a top-level session, the parent's depth + 1 for children.
    public var depth: Int?
    /// Every turn, in the order they were made.
    public private(set) var nodes: [ChatTurn]
    public var activeLeaf: UUID?
    /// The version last shown after each turn (parent → child; `root` for the first message), so going back to a
    /// version brings back the conversation that followed it.
    public private(set) var choices: [UUID: UUID] = [:]
    static let root = UUID(uuid: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    public init(id: UUID = UUID(), title: String = "New Chat", serverID: String? = nil, modelID: String? = nil,
                turns: [ChatTurn] = [], date: Date = Date()) {
        self.id = id
        self.title = title
        self.created = date
        self.updated = date
        self.serverID = serverID
        self.modelID = modelID
        nodes = []
        for t in turns { append(t) }
    }

    // MARK: the tree

    /// The branch shown: from the first turn to the active leaf.
    public var turns: [ChatTurn] {
        guard let leaf = activeLeaf else { return [] }
        let byID = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        var path: [ChatTurn] = []
        var cur: UUID? = leaf
        while let id = cur, let t = byID[id] {
            path.append(t)
            cur = t.parent
        }
        return path.reversed()
    }

    public func turn(_ id: UUID) -> ChatTurn? { nodes.first { $0.id == id } }

    /// Turns with the same parent as `id` (itself included), oldest first.
    public func siblings(of id: UUID) -> [ChatTurn] {
        guard let t = turn(id) else { return [] }
        return nodes.filter { $0.parent == t.parent && $0.role == t.role }
    }

    /// Adds a turn after the active leaf and makes it the leaf.
    public mutating func append(_ turn: ChatTurn) {
        var t = turn
        t.parent = activeLeaf
        nodes.append(t)
        show(t.id)
    }

    /// Makes `id` the leaf and remembers the path to it.
    mutating func show(_ id: UUID) {
        activeLeaf = id
        let byID = Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
        var cur: UUID? = id
        while let c = cur, let t = byID[c] {
            choices[t.parent ?? Self.root] = c
            cur = t.parent
        }
    }

    /// Adds a turn next to `sibling` (same parent): an edited message, another reply. It becomes the leaf.
    public mutating func branch(from sibling: UUID, with turn: ChatTurn) {
        var t = turn
        t.parent = self.turn(sibling)?.parent
        nodes.append(t)
        show(t.id)
    }

    /// Shows the branch through `id`, continuing with the versions last shown after it (else the newest).
    public mutating func select(_ id: UUID) {
        var cur = id
        while true {
            let children = nodes.filter { $0.parent == cur }
            guard !children.isEmpty else { break }
            cur = children.first { $0.id == choices[cur] }?.id ?? children[children.count - 1].id
        }
        show(cur)
    }

    public mutating func update(_ id: UUID, _ change: (inout ChatTurn) -> Void) {
        guard let i = nodes.firstIndex(where: { $0.id == id }) else { return }
        change(&nodes[i])
    }

    /// Removes a turn and everything after it (a reply that never started, say).
    public mutating func remove(_ id: UUID) {
        guard let t = turn(id) else { return }
        var gone: Set<UUID> = [id]
        var changed = true
        while changed {
            changed = false
            for n in nodes where !gone.contains(n.id) && n.parent.map(gone.contains) == true {
                gone.insert(n.id)
                changed = true
            }
        }
        nodes.removeAll { gone.contains($0.id) }
        if let leaf = activeLeaf, gone.contains(leaf) {
            if let p = t.parent { select(p) } else { activeLeaf = nodes.last(where: { $0.parent == nil }).map(\.id) }
            if let leaf = activeLeaf, turn(leaf) == nil { activeLeaf = nil }
        }
    }

    // MARK: sending

    /// What the chat has cost so far (every reply's GPU time, other branches included).
    public var costUSD: Double { nodes.compactMap { $0.stats?.costUSD }.reduce(0, +) }

    /// The shown branch as OpenAI messages, without failed replies. `image(name)` gives an attached image's data URL,
    /// or a video's URL (`upload:<name>`); nil leaves it out.
    /// `system` false leaves the chat's system prompt out (the caller sends its own, with it inside).
    public func messages(system: Bool = true, image: (String) -> String? = { _ in nil }) -> [ChatMessage] {
        var out: [ChatMessage] = []
        if system, let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out.append(.system(systemPrompt))
        }
        // Everything before the latest summary is in it.
        let shown = turns
        let from = shown.lastIndex { $0.summary == true } ?? shown.startIndex
        for t in shown[from...] where t.error == nil || !t.text.isEmpty || t.sessionEvent != nil {
            if let e = t.sessionEvent {
                out.append(ChatMessage(role: .user, parts: [.text(e.textForModel)]))
                continue
            }
            if t.summary == true {
                out.append(ChatMessage(role: .user, parts: [.text(ChatTurn.summaryPrefix + t.text)]))
                continue
            }
            switch t.role {
            case .user:
                var parts: [ContentPart] = t.images.compactMap { name in
                    image(name).map { ChatTurn.isVideo(name) ? .video(url: $0) : .image(url: $0) }
                }
                parts.append(.text(t.textForModel))
                out.append(ChatMessage(role: .user, parts: parts))
            case .assistant:
                out.append(ChatMessage(role: .assistant, parts: [.text(t.text)], toolCalls: t.toolCalls))
            case .tool:
                out.append(ChatMessage(role: .tool, parts: [.text(t.text)], toolCallID: t.toolCallID))
            }
        }
        return out
    }

    /// A copy of the shown branch up to and including `turnID`, as a new chat: fresh turn ids, the same settings.
    /// Nil when `turnID` isn't on the shown branch.
    public func fork(upTo turnID: UUID, date: Date = Date()) -> ChatTranscript? {
        let shown = turns
        guard let end = shown.firstIndex(where: { $0.id == turnID }) else { return nil }
        let range = Array(shown[...end])
        var copy = ChatTranscript(title: "\(title) (fork)", serverID: serverID, modelID: modelID, date: date)
        copy.systemPrompt = systemPrompt
        copy.reasoningEffort = reasoningEffort
        copy.projectPath = projectPath
        copy.permissionMode = permissionMode
        copy.allowEdits = allowEdits
        copy.allowCommands = allowCommands
        copy.allowDomains = allowDomains
        copy.graphHelper = graphHelper
        copy.allowRuns = allowRuns
        copy.brief = brief
        copy.parentID = id
        copy.depth = depth
        let newIDs = Dictionary(uniqueKeysWithValues: range.map { ($0.id, UUID()) })
        for old in range {
            var t = old
            t.id = newIDs[old.id]!
            if var q = t.quote {
                q.turnID = q.turnID.flatMap { newIDs[$0] }
                t.quote = q
            }
            copy.append(t)
        }
        return copy
    }

    /// The images and videos attached to the shown branch's messages.
    public var attachments: [String] { turns.flatMap(\.images) }

    /// A title from the first user message: its first non-blank line, at most 48 characters.
    public static func title(from text: String) -> String {
        let line = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty } ?? ""
        guard !line.isEmpty else { return "New Chat" }
        return line.count > 48 ? String(line.prefix(47)) + "…" : line
    }

    // MARK: Codable (version 1 files had a flat `turns` list)

    enum CodingKeys: String, CodingKey {
        case version, id, title, created, updated, serverID, modelID, systemPrompt, reasoningEffort, nodes, activeLeaf
        case choices, turns, projectPath, permissionMode, allowEdits, allowCommands, graphHelper, allowRuns, allowDomains
        case pinned, archived, lastViewed, parentID, worktree, brief, depth, changesOpen
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        created = try c.decode(Date.self, forKey: .created)
        updated = try c.decode(Date.self, forKey: .updated)
        serverID = try c.decodeIfPresent(String.self, forKey: .serverID)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        reasoningEffort = try c.decodeIfPresent(String.self, forKey: .reasoningEffort)
        projectPath = try c.decodeIfPresent(String.self, forKey: .projectPath)
        permissionMode = try c.decodeIfPresent(String.self, forKey: .permissionMode)
        allowEdits = try c.decodeIfPresent(Bool.self, forKey: .allowEdits)
        allowCommands = try c.decodeIfPresent([String].self, forKey: .allowCommands)
        graphHelper = try c.decodeIfPresent(Bool.self, forKey: .graphHelper)
        allowRuns = try c.decodeIfPresent(Bool.self, forKey: .allowRuns)
        allowDomains = try c.decodeIfPresent([String].self, forKey: .allowDomains)
        pinned = try c.decodeIfPresent(Bool.self, forKey: .pinned)
        archived = try c.decodeIfPresent(Bool.self, forKey: .archived)
        lastViewed = try c.decodeIfPresent(Date.self, forKey: .lastViewed)
        parentID = try c.decodeIfPresent(UUID.self, forKey: .parentID)
        worktree = try c.decodeIfPresent(SessionWorktree.self, forKey: .worktree)
        changesOpen = try c.decodeIfPresent(Bool.self, forKey: .changesOpen)
        brief = try c.decodeIfPresent(String.self, forKey: .brief)
        depth = try c.decodeIfPresent(Int.self, forKey: .depth)
        if let nodes = try c.decodeIfPresent([ChatTurn].self, forKey: .nodes) {
            self.nodes = nodes
            activeLeaf = try c.decodeIfPresent(UUID.self, forKey: .activeLeaf)
            choices = try c.decodeIfPresent([UUID: UUID].self, forKey: .choices) ?? [:]
        } else {
            nodes = []
            for t in try c.decodeIfPresent([ChatTurn].self, forKey: .turns) ?? [] { append(t) }
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(2, forKey: .version)
        try c.encode(id, forKey: .id)
        try c.encode(title, forKey: .title)
        try c.encode(created, forKey: .created)
        try c.encode(updated, forKey: .updated)
        try c.encodeIfPresent(serverID, forKey: .serverID)
        try c.encodeIfPresent(modelID, forKey: .modelID)
        try c.encodeIfPresent(systemPrompt, forKey: .systemPrompt)
        try c.encodeIfPresent(reasoningEffort, forKey: .reasoningEffort)
        try c.encodeIfPresent(projectPath, forKey: .projectPath)
        try c.encodeIfPresent(permissionMode, forKey: .permissionMode)
        try c.encodeIfPresent(allowEdits, forKey: .allowEdits)
        try c.encodeIfPresent(allowCommands, forKey: .allowCommands)
        try c.encodeIfPresent(graphHelper, forKey: .graphHelper)
        try c.encodeIfPresent(allowRuns, forKey: .allowRuns)
        try c.encodeIfPresent(allowDomains, forKey: .allowDomains)
        try c.encodeIfPresent(pinned, forKey: .pinned)
        try c.encodeIfPresent(archived, forKey: .archived)
        try c.encodeIfPresent(lastViewed, forKey: .lastViewed)
        try c.encodeIfPresent(parentID, forKey: .parentID)
        try c.encodeIfPresent(worktree, forKey: .worktree)
        try c.encodeIfPresent(changesOpen, forKey: .changesOpen)
        try c.encodeIfPresent(brief, forKey: .brief)
        try c.encodeIfPresent(depth, forKey: .depth)
        try c.encode(nodes, forKey: .nodes)
        try c.encodeIfPresent(activeLeaf, forKey: .activeLeaf)
        try c.encode(choices, forKey: .choices)
    }
}
