import AIStudioProtocol
import Agent
import AppKit
import Foundation
import GraphRuntime
import LLMClient
import Observation

/// An open chat tab: the saved transcript plus what only exists while it's open (the draft, the reply streaming in).
@MainActor @Observable
final class ChatDocument: Identifiable {
    let id: UUID
    var transcript: ChatTranscript
    var draft = ""
    /// The assistant turn being written, while a reply streams.
    var liveTurnID: UUID?
    /// The engine's state reported during the request (cold start phases).
    var engine: EngineStatus?
    /// When the request in progress started.
    var requestStarted: Date?
    /// Attachments for the message being written.
    var pending: [PendingAttachment] = []
    /// Videos being made smaller before they're added to `pending`, and why the last one couldn't be.
    var preparingVideos = 0
    var attachNote: String?
    /// The user message being edited in place.
    var editingTurnID: UUID?
    @ObservationIgnored var task: Task<Void, Never>?

    // MARK: coding chats

    /// The `ask_user` questions waiting for the user's answer (their card), and the answer channel.
    var question: PendingQuestion?
    @ObservationIgnored var questionAnswer: CheckedContinuation<[UserAnswer]?, Never>?
    /// The passage the message being written replies to.
    var replyTo: ChatQuote?
    /// Widgets' last reported state (JSON), by tool call id, and what was last sent to the model.
    var widgetStates: [String: String] = [:]
    @ObservationIgnored var sentWidgetStates: [String: String] = [:]
    /// A turn to scroll to and flash (a quote's original).
    var highlightTurnID: UUID?

    /// A tool call waiting for the user's answer, and the question's answer channel.
    var permission: PermissionRequest?
    @ObservationIgnored var permissionAnswer: CheckedContinuation<PermissionAnswer, Never>?
    /// Live output of running commands, by call id.
    var toolOutputs: [String: String] = [:]
    /// The agent's to-do list.
    var todos: [ToolContext.Todo] = []

    var isCoding: Bool { transcript.projectPath != nil }
    /// A graph helper: its tools work on a workflow tab (`workflowID`, else the active one).
    var isGraphHelper: Bool { transcript.graphHelper == true }
    /// Uses tools (and so permission modes).
    var isAgent: Bool { isCoding || isGraphHelper }
    /// The workflow tab a docked helper belongs to.
    var workflowID: UUID?
    /// The graph before and after the helper's last reply that changed it (for its Undo button).
    var lastHelperEdit: (before: Graph, after: Graph, changes: [String])?
    var projectURL: URL? { transcript.projectPath.map { URL(fileURLWithPath: $0) } }
    var mode: PermissionMode {
        get { transcript.permissionMode.flatMap(PermissionMode.init(rawValue:)) ?? .ask }
        set { transcript.permissionMode = newValue.rawValue }
    }
    var rules: AllowRules {
        AllowRules(edits: transcript.allowEdits ?? false, commandPrefixes: transcript.allowCommands ?? [],
                   runs: transcript.allowRuns ?? false)
    }

    /// Answers the pending permission question.
    func answer(_ a: PermissionAnswer) {
        let c = permissionAnswer
        permissionAnswer = nil
        permission = nil
        c?.resume(returning: a)
    }

    /// Answers the open `ask_user` questions (nil: skipped) and lets the reply go on.
    func answerQuestion(_ answers: [UserAnswer]?) {
        let c = questionAnswer
        questionAnswer = nil
        question = nil
        c?.resume(returning: answers)
    }

    init(transcript: ChatTranscript) {
        id = transcript.id
        self.transcript = transcript
    }

    var title: String { transcript.title }
    var isStreaming: Bool { liveTurnID != nil }
    var modelKey: LLMKey? {
        guard let s = transcript.serverID, let m = transcript.modelID else { return nil }
        return LLMKey(serverID: s, modelID: m)
    }
}

/// `ask_user` questions shown as a card until the user answers.
struct PendingQuestion: Equatable {
    var callID: String
    var questions: [UserQuestion]
}

/// An image or video (a file in the chat's folder) or a text file waiting to be sent.
enum PendingAttachment: Hashable {
    case image(String)
    case file(ChatFile)

    var name: String {
        switch self {
        case .image(let n): return n
        case .file(let f): return f.name
        }
    }
}

/// A chat in the Chats panel (read from its file's header fields).
struct ChatSummary: Identifiable, Hashable {
    var id: UUID
    var title: String
    var updated: Date
    var modelID: String?
    var coding = false
    var helper = false
}

/// Saved chats in the library's `chats/` folder: `<id>.json` (the transcript) and `<id>/` (attachments).
struct ChatStore {
    let root: URL

    func url(_ id: UUID) -> URL { root.appendingPathComponent("\(id.uuidString).json") }
    func filesFolder(_ id: UUID) -> URL { root.appendingPathComponent(id.uuidString, isDirectory: true) }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        e.dateEncodingStrategy = .iso8601
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    func load(_ id: UUID) -> ChatTranscript? {
        guard let data = try? Data(contentsOf: url(id)) else { return nil }
        return try? Self.decoder.decode(ChatTranscript.self, from: data)
    }

    func save(_ t: ChatTranscript) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Self.encoder.encode(t).write(to: url(t.id), options: .atomic)
    }

    func list() -> [ChatSummary] {
        let files = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.compactMap { u -> ChatSummary? in
            guard let data = try? Data(contentsOf: u), let t = try? Self.decoder.decode(ChatTranscript.self, from: data)
            else { return nil }
            return ChatSummary(id: t.id, title: t.title, updated: t.updated, modelID: t.modelID, coding: t.projectPath != nil,
                               helper: t.graphHelper == true)
        }.sorted { $0.updated > $1.updated }
    }

    /// To the Trash (deleted outright in automated runs).
    func delete(_ id: UUID, useTrash: Bool) {
        for u in [url(id), filesFolder(id)] where FileManager.default.fileExists(atPath: u.path) {
            if useTrash { try? FileManager.default.trashItem(at: u, resultingItemURL: nil) }
            else { try? FileManager.default.removeItem(at: u) }
        }
    }
}

/// A model on a GPU server.
struct LLMKey: Hashable, Codable {
    var serverID: String
    var modelID: String
}

/// The LLMs on the GPU servers: their listings and engine states, installs, and keeping a chat's model warm.
///
/// Warmth: while a chat tab is in front, the app is active and the user did something in the last 10 minutes, its
/// model gets a keepalive every minute (the server shuts the GPU down a few minutes after the last one). Its status
/// is polled every 10 s while it isn't cold, for the chat header.
@MainActor @Observable
final class LLMCatalog {
    var listings: [String: LLMListing] = [:]
    var errors: [String: String] = [:]
    var statuses: [LLMKey: LLMStatus] = [:]
    var refreshing: Set<String> = []
    @ObservationIgnored let connections: ConnectionStore
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var ticks = 0
    @ObservationIgnored private var monitor: Any?
    @ObservationIgnored var lastActivity = Date()
    /// What should be kept warm now (set by the app model: the chat in front).
    @ObservationIgnored var focus: () -> LLMKey? = { nil }

    init(connections: ConnectionStore) { self.connections = connections }

    func client(_ serverID: String) -> LLMServerClient? {
        guard let c = connections.connection(serverID), c.kind == .gpu, let url = c.url else { return nil }
        return LLMServerClient(baseURL: url, token: ConnectionToken.read(serverID) ?? "")
    }

    var servers: [ServerConnection] { connections.gpuServers }

    /// Installed models, server by server.
    var models: [(server: ServerConnection, model: LLMModel)] {
        servers.flatMap { s in (listings[s.id]?.models ?? []).filter { $0.installed != false }.map { (s, $0) } }
    }

    func model(_ key: LLMKey) -> LLMModel? {
        listings[key.serverID]?.models.first { $0.id == key.modelID }
    }

    func status(_ key: LLMKey) -> LLMStatus? { statuses[key] ?? model(key)?.status }

    func refresh(_ serverID: String) async {
        guard let client = client(serverID) else { return }
        refreshing.insert(serverID)
        defer { refreshing.remove(serverID) }
        do {
            let l = try await client.models()
            listings[serverID] = l
            errors[serverID] = nil
            for m in l.models { if let s = m.status { statuses[LLMKey(serverID: serverID, modelID: m.id)] = s } }
        } catch {
            errors[serverID] = error.localizedDescription
        }
    }

    func refreshAll() async {
        for s in servers { await refresh(s.id) }
    }

    func refreshStatus(_ key: LLMKey) async {
        guard let client = client(key.serverID), let s = try? await client.status(key.modelID) else { return }
        statuses[key] = s
    }

    func warm(_ key: LLMKey) async {
        guard let client = client(key.serverID) else { return }
        do { statuses[key] = try await client.warm(key.modelID) } catch { errors[key.serverID] = error.localizedDescription }
    }

    func stop(_ key: LLMKey) async {
        guard let client = client(key.serverID) else { return }
        if let s = try? await client.stop(key.modelID) { statuses[key] = s }
    }

    func remove(_ key: LLMKey) async {
        guard let client = client(key.serverID) else { return }
        do {
            try await client.remove(key.modelID)
            statuses[key] = nil
        } catch {
            errors[key.serverID] = error.localizedDescription
        }
        await refresh(key.serverID)
    }

    // MARK: warmth

    /// Starts the keepalive / status timer and the activity monitor (once).
    func start() {
        guard timer == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .leftMouseDown, .scrollWheel]) { [weak self] e in
            self?.lastActivity = Date()
            return e
        }
        timer = Timer.scheduledTimer(withTimeInterval: 10, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.tick() }
        }
    }

    func tick() async {
        ticks += 1
        // Installs in progress: refresh their server's listing.
        for (sid, l) in listings where !l.installs.isEmpty { await refresh(sid) }
        guard let key = focus() else { return }
        let st = status(key)
        if st?.isActive == true || st == nil { await refreshStatus(key) }
        let current = status(key)
        // Keepalive once a minute while the user is around and the model is up.
        if ticks % 6 == 0, current?.isActive == true, NSApp.isActive, Date().timeIntervalSince(lastActivity) < 600,
           let client = client(key.serverID), let s = try? await client.keepalive(key.modelID) {
            statuses[key] = s
        }
    }
}
