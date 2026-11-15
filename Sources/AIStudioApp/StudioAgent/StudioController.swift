import Agent
import AppKit
import MovieCore
import SwiftUI

/// A place in the app the Studio agent works on, and goes back to when you click its step.
enum StudioPlace: Equatable {
    case none
    case panel(SidebarPanel)
    case movie(UUID, MovieStage?)
    case workflow(UUID)
}

/// A free change the agent made that its step row can put back ("Revert").
struct StudioRevert {
    let title: String
    let apply: @MainActor () -> Void
}

/// The Studio agent's action was skipped (Step pace's Skip, or the reply stopped while it waited).
struct StudioSkipped: Error {}

/// The Studio agent: its chat in the floating panel, the pointer, ring and caption it shows in the main window, the gate
/// its actions pass (paused when you use the window; Step pace's Next), and what each step can revisit or revert.
@MainActor @Observable
final class StudioController {
    let gate = StudioGate(pace: StudioPace(rawValue: UserDefaults.standard.string(forKey: "studioPace") ?? "") ?? .watchable)
    @ObservationIgnored weak var model: AppModel?

    /// The panel's chat (nil until the panel is first opened).
    var chat: ChatDocument?
    /// The panel is folded to a pill.
    var folded = false
    var panelShown = false

    // The focus overlay, in the main window's coordinates (points from its top-left corner).
    var pointer: CGPoint?
    var ring: CGRect?
    var ringShown = false
    var caption: String?

    /// A card the pointer is going to that's scrolled out of view: its page scrolls to it (`MoviePage`).
    var scrollRequest: (target: String, id: UUID)?

    /// A step waiting on you (the cost sheet's Run): what the panel's hourglass row says. Clicks in the window don't
    /// pause the agent meanwhile.
    var waitingOnUser: String?

    /// Calls whose change was put back.
    var reverted: Set<String> = []
    @ObservationIgnored var reverts: [String: StudioRevert] = [:]
    @ObservationIgnored var revisits: [String: (place: StudioPlace, target: String?)] = [:]
    /// The last action went through after you used the window: its result says to look again.
    @ObservationIgnored var reread = false
    /// Events the agent posts itself (the fallback's clicks and keys) until then don't pause it.
    @ObservationIgnored var syntheticUntil = Date.distantPast

    @ObservationIgnored private var monitor: Any?
    @ObservationIgnored private var resizeObserver: NSObjectProtocol?
    @ObservationIgnored private var hideTask: Task<Void, Never>?
    @ObservationIgnored lazy var panel = StudioPanelWindow(controller: self)
    @ObservationIgnored lazy var overlay = StudioOverlayWindow(controller: self)

    var pace: StudioPace {
        get { gate.pace }
        set {
            gate.pace = newValue
            if !AppModel.isAutomatedRun { UserDefaults.standard.set(newValue.rawValue, forKey: "studioPace") }
        }
    }

    var isWorking: Bool { chat?.isStreaming == true }

    // MARK: the main window

    /// AIStudio's main window (the one with the sidebar rail), not the panel or a sheet.
    var mainWindow: NSWindow? {
        if let w = MGTestAnchors.window("sidebar-rail") { return w }
        return NSApp.windows.first { !($0 is NSPanel) && $0.isVisible && $0.contentView != nil && $0.title == "AIStudio" }
    }

    /// A screen rectangle in the overlay's coordinates (top-left of the main window).
    func local(_ screen: CGRect, in window: NSWindow) -> CGRect {
        CGRect(x: screen.minX - window.frame.minX, y: window.frame.maxY - screen.maxY, width: screen.width, height: screen.height)
    }

    /// Where a named control is on screen, if it's in the main window now.
    func frame(_ target: String?) -> CGRect? {
        guard let target, let f = MGTestAnchors.frame(target), let main = mainWindow,
              let w = MGTestAnchors.window(target), w === main || w.parent === main else { return nil }
        return f
    }

    // MARK: input from the user

    func installMonitor() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown, .keyDown, .scrollWheel]) { [weak self] e in
            MainActor.assumeIsolated { self?.userInput(e) }
            return e
        }
    }

    /// You clicked, typed or scrolled in the main window while the agent works: it pauses before its next action.
    private func userInput(_ e: NSEvent) {
        guard isWorking, waitingOnUser == nil, Date() > syntheticUntil, let main = mainWindow else { return }
        guard e.window === main || e.window?.parent === main, !(e.window is StudioOverlayWindow) else { return }
        if e.type == .scrollWheel, !(e.phase == .began || (e.phase.isEmpty && e.momentumPhase.isEmpty)) { return }
        gate.pauseByUser()
    }

    // MARK: actions

    /// One action: waits at the gate, goes to `place`, moves the pointer to `step.target` with the caption, does
    /// `work`, then holds the ring a moment. Throws `StudioSkipped` when you skip it.
    func act<T>(_ step: StudioStep, at place: StudioPlace = .none, callID: String? = nil,
                fallback: String? = nil, _ work: () async throws -> T) async throws -> T {
        switch await gate.proceed(step.doing) {
        case .skipped: throw StudioSkipped()
        case .go(let again): if again { reread = true }
        }
        try Task.checkCancellation()
        if place != .none { await go(place, waitFor: step.target) } else { await settle(step.target) }
        if let callID { revisits[callID] = (place == .none ? currentPlace() : place, step.target) }
        await point(at: step.target, fallback: fallback ?? containerTarget(place), caption: step.doing)
        let value = try await work()
        await hold()
        return value
    }

    /// One fallback action at a screen rectangle (an element of the window, not a named control).
    func act<T>(rect: CGRect?, _ step: StudioStep, callID: String? = nil, _ work: () async throws -> T) async throws -> T {
        switch await gate.proceed(step.doing) {
        case .skipped: throw StudioSkipped()
        case .go(let again): if again { reread = true }
        }
        try Task.checkCancellation()
        if let callID { revisits[callID] = (currentPlace(), nil) }
        await point(screen: rect, caption: step.doing)
        let value = try await work()
        await hold()
        return value
    }

    /// Where the app is now, as a place to come back to.
    func currentPlace() -> StudioPlace {
        guard let model else { return .none }
        switch model.frontItem {
        case .movie(let id): return .movie(id, model.activeMovie?.stage)
        case .workflow(let id): return .workflow(id)
        case .chat: return .none
        }
    }

    func containerTarget(_ place: StudioPlace) -> String? {
        switch place {
        case .movie: return "movie-view"
        case .workflow: return "graph-canvas"
        case .panel: return "sidebar-rail"
        case .none:
            if case .movie = currentPlace() { return "movie-view" }
            return "graph-canvas"
        }
    }

    /// Brings `place` to the front and waits (up to 1.5 s) for `target` to be on screen.
    func go(_ place: StudioPlace, waitFor target: String? = nil) async {
        guard let model else { return }
        switch place {
        case .none: break
        case .panel(let p):
            UserDefaults.standard.set(p.rawValue, forKey: "sidebarPanel")
            UserDefaults.standard.set(true, forKey: "sidebarOpen")
        case .movie(let id, let stage):
            if model.activeMovieID != id { model.openMovie(id) }
            if let stage, let doc = model.activeMovie, doc.id == id, doc.stage != stage { doc.stage = stage }
        case .workflow(let id):
            if model.activeTabID != id || !model.workflowInFront { model.activate(id) }
        }
        await settle(target)
    }

    /// Waits for a control to be laid out (a page just switched), at most 1.5 s.
    func settle(_ target: String?) async {
        let deadline = Date().addingTimeInterval(1.5)
        try? await Task.sleep(for: .milliseconds(60))
        guard let target else { return }
        if !onScreen(target) {
            // Below the fold (or not laid out yet, in a lazy list): its page scrolls to it.
            scrollRequest = (target, UUID())
            try? await Task.sleep(for: .milliseconds(350))
        }
        while frame(target) == nil, Date() < deadline {
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    /// Whether a control is in the main window's visible area now.
    func onScreen(_ target: String) -> Bool {
        guard let f = frame(target), let main = mainWindow, let content = main.contentView else { return false }
        let visible = main.convertToScreen(content.convert(content.visibleRect, to: nil))
        return visible.insetBy(dx: 0, dy: 40).intersects(f) && f.midY > visible.minY && f.midY < visible.maxY
    }

    /// Moves the pointer to a control, rings it and shows the caption.
    func point(at target: String?, fallback: String? = nil, caption text: String) async {
        await point(screen: frame(target) ?? frame(fallback), caption: text)
    }

    /// Moves the pointer to a rectangle on screen (nil: just the caption), rings it and shows the caption.
    func point(screen: CGRect?, caption text: String) async {
        guard let main = mainWindow else { return }
        hideTask?.cancel()
        overlay.cover(main)
        let rect = screen.map { local($0, in: main) }
        let pace = gate.pace
        withAnimation(.easeInOut(duration: pace.moveSeconds)) {
            caption = text
            if let rect {
                ring = rect.insetBy(dx: -4, dy: -4)
                ringShown = true
                // The tip just inside the control's lower half, so the caption doesn't cover it.
                pointer = CGPoint(x: rect.minX + min(rect.width / 2, 40), y: rect.midY + min(rect.height / 4, 12))
            } else {
                ringShown = false
                if pointer == nil { pointer = CGPoint(x: main.frame.width / 2, y: main.frame.height - 80) }
            }
        }
        try? await Task.sleep(for: .seconds(pace.moveSeconds))
    }

    /// After an action: the ring stays a moment, then fades.
    func hold() async {
        try? await Task.sleep(for: .seconds(gate.pace.holdSeconds))
        withAnimation(.easeOut(duration: 0.3)) { ringShown = false }
    }

    /// Types `text` into a field through `set`: all at once in Fast pace, else a few characters at a time.
    func type(_ text: String, _ set: (String) -> Void) async {
        let seconds = gate.pace.typingSeconds(text.count)
        guard seconds > 0 else { set(text); return }
        let chars = Array(text)
        let steps = max(1, Int(seconds * 30))
        for i in 1...steps {
            set(String(chars.prefix(chars.count * i / steps)))
            try? await Task.sleep(for: .seconds(seconds / Double(steps)))
        }
        set(text)
    }

    /// The reply ended: anything waiting is let go and the overlay fades after a moment.
    func replyEnded() {
        gate.reset()
        waitingOnUser = nil
        hideTask?.cancel()
        hideTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(1.2))
            guard let self, !Task.isCancelled, !self.isWorking else { return }
            withAnimation(.easeOut(duration: 0.3)) {
                self.ringShown = false
                self.caption = nil
                self.pointer = nil
            }
            try? await Task.sleep(for: .seconds(0.35))
            if !Task.isCancelled, !self.isWorking { self.overlay.orderOut(nil) }
        }
    }

    // MARK: the panel

    func setFolded(_ f: Bool) {
        folded = f
        panel.setFolded(f)
    }

    /// The panel was closed: a reply in progress stops, the chat is saved, the overlay goes.
    func panelClosed() {
        panelShown = false
        if folded { folded = false; panel.setFolded(false) }
        if let chat, let model {
            if chat.isStreaming { model.stopReply(chat) }
            model.saveChat(chat)
        }
        gate.reset()
        overlay.orderOut(nil)
    }

    // MARK: step rows

    /// A step row was clicked: back to where it happened, and the ring on its control again (nothing changes).
    func revisit(_ callID: String) {
        guard let r = revisits[callID] else { return }
        Task { @MainActor in
            await go(r.place, waitFor: r.target)
            await point(at: r.target, fallback: containerTarget(r.place), caption: "Here")
            await hold()
            if !isWorking { replyEnded() }
        }
    }

    func revert(_ callID: String) {
        guard let r = reverts[callID], !reverted.contains(callID) else { return }
        r.apply()
        reverted.insert(callID)
        revisit(callID)
    }
}
