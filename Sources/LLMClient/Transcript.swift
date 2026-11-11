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
    /// Attached images: file names in the chat's files folder.
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

    /// The shown branch as OpenAI messages, without failed replies. `image(name)` gives an attached image's data URL
    /// (nil leaves it out).
    public func messages(image: (String) -> String? = { _ in nil }) -> [ChatMessage] {
        var out: [ChatMessage] = []
        if let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            out.append(.system(systemPrompt))
        }
        for t in turns where t.error == nil || !t.text.isEmpty {
            switch t.role {
            case .user:
                var parts: [ContentPart] = t.images.compactMap { image($0) }.map { .image(url: $0) }
                parts.append(.text(t.textWithFiles))
                out.append(ChatMessage(role: .user, parts: parts))
            case .assistant:
                out.append(ChatMessage(role: .assistant, parts: [.text(t.text)], toolCalls: t.toolCalls))
            case .tool:
                out.append(ChatMessage(role: .tool, parts: [.text(t.text)], toolCallID: t.toolCallID))
            }
        }
        return out
    }

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
        case choices, turns, projectPath, permissionMode, allowEdits, allowCommands, graphHelper, allowRuns
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
        try c.encode(nodes, forKey: .nodes)
        try c.encodeIfPresent(activeLeaf, forKey: .activeLeaf)
        try c.encode(choices, forKey: .choices)
    }
}
