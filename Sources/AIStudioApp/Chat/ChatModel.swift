import AIStudioLibrary
import AIStudioProtocol
import Agent
import AppKit
import Foundation
import GraphRuntime
import LLMClient
import Observation

/// An open chat tab: the saved transcript plus what only exists while it's open (the draft, the reply streaming in).
/// A coding chat's right-hand panel.
enum SidePanel: String, CaseIterable, Identifiable {
    case files, changes, plans, web, terminal

    var id: String { rawValue }
    var title: String {
        switch self {
        case .files: "Files"
        case .changes: "Changes"
        case .plans: "Plans"
        case .web: "Browser"
        case .terminal: "Terminal"
        }
    }
    var symbol: String {
        switch self {
        case .files: "folder"
        case .changes: "plusminus"
        case .plans: "list.bullet.clipboard"
        case .web: "globe"
        case .terminal: "terminal"
        }
    }
}

@MainActor @Observable
final class ChatDocument: Identifiable {
    let id: UUID
    var transcript: ChatTranscript
    var draft = ""
    /// A message typed while a reply streamed: sent when that reply ends well, else put back in the draft.
    var queued: String?
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
    /// One of the chats the user opened lately, kept loaded and listed in the Window menu. Only the most recent few
    /// stay so (`AppModel.recentChatLimit`); a session that is still working when it drops out stays loaded, hidden.
    var isRecent = true
    /// Called after a permission, question or plan card is answered (the dock badge).
    @ObservationIgnored var onPendingChange: (@MainActor () -> Void)?
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
    /// The to-do list of the run in progress (nil: the transcript's last `todo_write` is the list).
    var liveTodos: [ToolContext.Todo]?
    /// The agent's to-do list: the run's while it works, else derived from the transcript.
    var todos: [ToolContext.Todo] {
        get { liveTodos ?? transcript.todoList }
        set { liveTodos = newValue }
    }
    var todosCollapsed = false
    /// The in-progress step and when it last changed.
    var todoStepStarted: (content: String, at: Date)?
    /// Background commands (dev servers…) the agent started, and the ids of those still running.
    @ObservationIgnored lazy var jobs = BackgroundJobs()
    @ObservationIgnored var jobsObserved = false
    /// Files the agent read or wrote in this chat (path → modification date then), carried from one reply to the next
    /// so an edit doesn't need a fresh read every turn. In memory only: after a restart the agent reads once more.
    @ObservationIgnored var readStamps: [String: Date] = [:]
    var runningJobs: [String] = []
    /// The chat's terminals (the side panel's Terminal tab), in memory only: a shell can't be saved. A chat with
    /// terminals stays loaded; they end when the chat or the app closes.
    var terminals: [TerminalSession] = []
    var activeTerminalID: UUID?
    /// Bumped to put the cursor back in the composer (⌃` from the terminal).
    var composerFocusRequest = 0
    var activeTerminal: TerminalSession? {
        terminals.first { $0.id == activeTerminalID } ?? terminals.last
    }
    /// Where the files the agent changed are saved, for "Rewind to Here".
    @ObservationIgnored var checkpointStore: CheckpointStore?
    /// The plan `exit_plan` waits for approval of (its card), and the answer channel.
    var plan: PendingPlan?
    @ObservationIgnored var planAnswer: CheckedContinuation<PlanAnswer, Never>?
    /// The conversation is being summarized.
    var compacting = false
    /// Tokens of the last request (prompt and reply): how full the context is.
    var contextUsed: Int?
    /// The message a rewind goes back to before (its card), and the files that would be restored.
    var rewindTarget: UUID?
    var rewindFiles: [URL]?

    // MARK: sessions

    /// Session reports that arrived while a reply was streaming; delivered when it ends.
    var pendingEvents: [SessionEvent] = []
    /// How many times a session report made this chat continue by itself since the user last wrote.
    var autoContinues = 0
    /// `propose_session` cards by call id, and what was decided (the started session's id; nil: dismissed).
    var proposals: [String: SessionProposal] = [:]
    var proposalDecisions: [String: UUID?] = [:]

    // MARK: flows

    /// Flow results that arrived while a reply was streaming; delivered when it ends.
    var pendingFlowEvents: [FlowEvent] = []
    /// The run a `flow` tool call started, by call id (in memory; the tool's result text names the run too).
    var flowRunByCall: [String: String] = [:]
    /// The run shown in the flow panel (nil: closed).
    var flowPanelRunID: String?

    // MARK: git

    /// The chat's repository at a glance (nil: not a repository, or not looked at yet).
    var git: GitState?
    var gitRefreshing = false
    @ObservationIgnored var gitRefreshAgain = false
    var gitError: String?
    var changesScope: ChangesScope = .uncommitted
    /// The path of the file whose diff is shown.
    var selectedChange: String?
    /// Diff text by "\(scope)|\(path)"; cleared on every refresh.
    var diffCache: [String: String] = [:]
    var commitDraft = ""
    /// Changed files left out of the next commit (unticked in Changes); every other file goes in.
    var commitExcluded: Set<String> = []
    var drafting = false
    var prSheet: PRDraft?
    /// The side panel's tab, kept while the panel is closed (Files until another is picked).
    var sideTab: SidePanel {
        get { transcript.sidePanel.flatMap(SidePanel.init(rawValue:)) ?? .files }
        set { transcript.sidePanel = newValue.rawValue }
    }
    /// The right-hand panel of a coding chat: shown (on its tab) unless the user closed it; nil: closed. Saved with
    /// the chat.
    var sidePanel: SidePanel? {
        get { transcript.sidePanelClosed == true ? nil : sideTab }
        set {
            if let newValue { sideTab = newValue; transcript.sidePanelClosed = nil } else { transcript.sidePanelClosed = true }
        }
    }
    /// The panel shows Changes. Setting false closes the panel only when it shows Changes.
    var changesOpen: Bool {
        get { sidePanel == .changes }
        set { if newValue { sidePanel = .changes } else if sidePanel == .changes { sidePanel = nil } }
    }

    /// The Browser tab's page (made when the tab is first shown; the address is saved as `transcript.browserURL`).
    var webPage: ChatWebPage?

    // MARK: files (the side panel's Files tab)

    /// The project's files (nil: not listed yet), while a listing runs, and a listing asked for meanwhile.
    var fileTree: FileTree?
    /// The same files for finding paths in replies (links to them).
    var fileIndex: ProjectFileIndex?
    var filesRefreshing = false
    @ObservationIgnored var filesRefreshAgain = false
    var expandedFolders: Set<String> = []
    /// What's opened in the transcript (in memory): a run's folded work (`run:<first reply>`), a long message
    /// (`msg:<id>`), a step's text in a run (`prose:<id>`), a thinking block (`think:<id>`). Kept here, not in the
    /// views, so lazily rebuilt rows don't forget it.
    var openRuns: Set<String> = []
    /// Tool cards opened or closed by hand, by call ID (absent: the card's default).
    var toolOpen: [String: Bool] = [:]

    /// Opens or closes an item of `openRuns`.
    func toggleOpen(_ key: String) {
        if openRuns.contains(key) { openRuns.remove(key) } else { openRuns.insert(key) }
    }
    var fileFilter = ""
    /// The file previewed under the tree (relative to the project) and a line to show.
    var previewPath: String?
    var previewLine: Int?
    /// Bumped when the agent may have changed files: an open preview reads its file again.
    var filesRevision = 0
    /// The plan selected in the Plans tab (its call id; nil: the newest).
    var selectedPlan: String?

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
                   runs: transcript.allowRuns ?? false, domains: transcript.allowDomains ?? [])
    }

    /// Answers the pending permission question.
    func answer(_ a: PermissionAnswer) {
        let c = permissionAnswer
        permissionAnswer = nil
        permission = nil
        c?.resume(returning: a)
        onPendingChange?()
    }

    /// Answers the open `ask_user` questions (nil: skipped) and lets the reply go on.
    func answerQuestion(_ answers: [UserAnswer]?) {
        let c = questionAnswer
        questionAnswer = nil
        question = nil
        c?.resume(returning: answers)
        onPendingChange?()
    }

    /// Answers the pending plan and lets the reply go on.
    func answerPlan(_ a: PlanAnswer) {
        let c = planAnswer
        planAnswer = nil
        plan = nil
        c?.resume(returning: a)
        onPendingChange?()
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

/// An `exit_plan` plan shown as a card until the user answers.
struct PendingPlan: Equatable {
    var callID: String
    var plan: String
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
    var status: SessionStatus = .idle
    var pinned = false
    var archived = false
    var parentID: UUID?
    var branch: String?
    var projectName: String?
    /// The project folder (a worktree session's repository), for "new session in this project".
    var projectPath: String?
    var todoDone: Int?
    var todoTotal: Int?
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
            let progress = t.todoProgress
            return ChatSummary(id: t.id, title: t.title, updated: t.updated, modelID: t.modelID, coding: t.projectPath != nil,
                               helper: t.graphHelper == true, status: t.savedStatus, pinned: t.pinned == true,
                               archived: t.archived == true, parentID: t.parentID, branch: t.worktree?.branch,
                               projectName: (t.worktree?.repo ?? t.projectPath).map { ($0 as NSString).lastPathComponent },
                               projectPath: t.worktree?.repo ?? t.projectPath,
                               todoDone: progress?.done, todoTotal: progress?.total)
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
/// Warmth: sessions that are working or waiting keep their model warm. While a chat tab is in front, the app is active and the user did something in the last 10 minutes, its
/// model gets a keepalive every minute (the server shuts the GPU down a few minutes after the last one). Its status
/// is polled every 10 s while it isn't cold, for the chat header.
@MainActor @Observable
final class LLMCatalog {
    var listings: [String: LLMListing] = [:]
    var errors: [String: String] = [:]
    var statuses: [LLMKey: LLMStatus] = [:] {
        didSet {
            // "$x so far" never goes down within one warm period: the server's figure can step back (Stop sets the end
            // time, a snapshot restore its own start), and the money was spent anyway.
            for (key, s) in statuses {
                switch s.state {
                case "ready", "stopping", "loading":
                    if let c = s.costUSD, c > (peakCost[key] ?? 0) { peakCost[key] = c }
                default:
                    if peakCost[key] != nil { peakCost[key] = nil }
                }
            }
        }
    }
    /// The highest "so far" cost seen in the model's current warm period.
    @ObservationIgnored private var peakCost: [LLMKey: Double] = [:]
    var refreshing: Set<String> = []
    @ObservationIgnored let connections: ConnectionStore
    @ObservationIgnored private var timer: Timer?
    @ObservationIgnored private var ticks = 0
    @ObservationIgnored private var monitor: Any?
    @ObservationIgnored var lastActivity = Date()
    /// What should be kept warm now (set by the app model: the chat in front).
    @ObservationIgnored var focus: () -> LLMKey? = { nil }
    /// The models of sessions that are working or waiting: kept warm whether or not the user is around.
    @ObservationIgnored var working: () -> Set<LLMKey> = { [] }

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

    func status(_ key: LLMKey) -> LLMStatus? {
        guard var s = statuses[key] ?? model(key)?.status else { return nil }
        if let peak = peakCost[key], s.costUSD.map({ $0 < peak }) ?? false { s.costUSD = peak }
        return s
    }

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

    /// Stop: "stopping" at once (the GPU takes ~15–25 s to go), then the server's word every 2 s until it's gone.
    func stop(_ key: LLMKey) async {
        guard let client = client(key.serverID) else { return }
        let before = status(key)
        if var s = before, s.isActive {
            // Still waiting for a GPU: the server cancels that at once. Loading or ready: the container shuts down.
            s.state = s.state == "starting" ? "cold" : "stopping"
            s.phase = s.isStopping ? "Shutting down the GPU" : nil
            s.progress = nil
            statuses[key] = s
        }
        guard let s = try? await client.stop(key.modelID) else {
            if let before { statuses[key] = before }  // the request failed: nothing is stopping
            return
        }
        statuses[key] = s
        for _ in 0..<45 where status(key)?.isStopping == true {
            try? await Task.sleep(nanoseconds: 2_000_000_000)
            await refreshStatus(key)
        }
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
        let focused = focus()
        let busy = working()
        for key in busy.union(focused.map { [$0] } ?? []) {
            let st = status(key)
            // One that's off is checked every 30 s too: it may have been started from another chat, a session or
            // the Servers list, and its chip and menu would otherwise say Cold while it loads.
            if st?.isActive == true || st?.isStopping == true || st == nil || ticks % 3 == 0 { await refreshStatus(key) }
            let current = status(key)
            // Keepalive once a minute: for the chat in front while the user is around, for working sessions always.
            let wanted = busy.contains(key) || (key == focused && NSApp.isActive && Date().timeIntervalSince(lastActivity) < 600)
            if ticks % 6 == 0, current?.isActive == true, wanted,
               let client = client(key.serverID), let s = try? await client.keepalive(key.modelID) {
                statuses[key] = s
            }
        }
    }
}
