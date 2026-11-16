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

/// Which actor a Studio tool call belongs to: the architect, or the worker whose loop it runs in (set by the worker
/// runner around its loop, so the tools need no actor argument). `prefix` keeps workers' call ids apart.
enum StudioActorScope {
    @TaskLocal static var actor: StudioActor = .architect
    @TaskLocal static var prefix = ""
}

/// One actor's marks on screen: its icon (resting spot, the flight it's on), the ring on its control, its caption, and
/// what it's doing (for the icon's motion). Positions are in the overlay's coordinates (top-left of the display).
@MainActor @Observable
final class StudioCursor: Identifiable {
    let actor: StudioActor
    var id: Int { actor.number }

    /// Where the icon rests (nil: not shown).
    var position: CGPoint?
    var flight: StudioFlight?
    var flightStart = Date.distantPast
    var ring: CGRect?
    var ringShown = false
    var caption: String?
    var state: StudioActorState = .idle
    /// A step waiting on the user (its cost sheet): the hourglass text.
    var waiting: String?
    /// Acting on a page that isn't in front: no pointer, just the log.
    var offscreen = false
    /// Its model is writing text until then (the orb's talking bounce).
    var talkingUntil = Date.distantPast
    /// Which way its eyes look (a unit-ish vector; zero: ahead).
    var gaze: CGVector = .zero
    var isTalking: Bool { Date() < talkingUntil }
    /// What its task's paid runs cost so far (a worker's, reset at each handoff).
    var spent: Double = 0
    /// Bumped to replay the done/failed moment and the pop-in.
    var moment = 0
    var popIn = 0
    /// The panel anchor it docks at (its task row's icon, or the panel header's).
    var homeTarget: String?
    /// The task it's working on (workers).
    var taskID: String?

    init(_ actor: StudioActor) { self.actor = actor }

    func isFlying(at now: Date) -> Bool {
        guard let flight else { return false }
        return now.timeIntervalSince(flightStart) < flight.duration
    }

    /// Where the icon is drawn at `now`: along its flight, else its resting spot (drifting when `drift`).
    func displayed(at now: Date, drift: Bool) -> CGPoint? {
        if let flight, isFlying(at: now) { return flight.point(at: now.timeIntervalSince(flightStart)) }
        guard let p = position else { return nil }
        guard drift, !ringShown else { return p }
        let d = StudioFlight.drift(actor: actor.number, time: now.timeIntervalSinceReferenceDate)
        return CGPoint(x: p.x + d.dx, y: p.y + d.dy)
    }
}

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
    var panelShown = false {
        didSet { if panelShown != oldValue { scheduleWander() } }
    }

    /// The actors on screen: the architect first, then the workers that have worked in this chat.
    var cursors: [StudioCursor] = [StudioCursor(.architect)]
    /// The approved plan (nil until the architect has one approved), shown as the panel's checklist.
    var plan: StudioPlan?
    /// Each task's worker steps (live), for its row in the plan checklist.
    var workerSteps: [String: [StudioWorkerStep]] = [:]
    /// The plan card waiting on the user, and the note they typed for Change….
    var pendingPlan: (plan: StudioPlan, callID: String, answer: (StudioPlanAnswer) -> Void)?
    /// The icons wander around the screen while the app is idle (Studio menu › Wander when idle).
    var wanders: Bool = AppModel.isAutomatedRun ? false : (UserDefaults.standard.object(forKey: "studioWanders") as? Bool ?? true) {
        didSet {
            if !AppModel.isAutomatedRun { UserDefaults.standard.set(wanders, forKey: "studioWanders") }
            scheduleWander()
        }
    }
    /// The overlay's frame on screen (the visible frame of the main window's display).
    var overlayFrame: CGRect = .zero

    func cursor(_ actor: StudioActor) -> StudioCursor {
        if let c = cursors.first(where: { $0.actor == actor }) { return c }
        let c = StudioCursor(actor)
        cursors.append(c)
        cursors.sort { $0.actor < $1.actor }
        return c
    }

    /// `actor`'s model just streamed some text: its orb talks for a moment.
    func talked(_ actor: StudioActor) {
        let c = cursor(actor)
        let was = c.isTalking
        c.talkingUntil = Date().addingTimeInterval(0.4)
        if !was { talkTick &+= 1 }
    }

    /// Bumped when someone starts talking (so views re-read who talks).
    var talkTick = 0

    /// The cursor of the actor whose tool call is running now.
    var current: StudioCursor { cursor(StudioActorScope.actor) }

    // The current actor's focus (the architect's outside a worker), in the overlay's coordinates.
    var pointer: CGPoint? {
        get { current.position }
        set { current.position = newValue }
    }
    var ring: CGRect? {
        get { current.ring }
        set { current.ring = newValue }
    }
    var ringShown: Bool {
        get { current.ringShown }
        set { current.ringShown = newValue }
    }
    var caption: String? {
        get { current.caption }
        set { current.caption = newValue }
    }

    /// A card the pointer is going to that's scrolled out of view: its page scrolls to it (`MoviePage`).
    var scrollRequest: (target: String, id: UUID)?

    /// A step waiting on you (the cost sheet's Run): what the panel's hourglass row says. Clicks in the window don't
    /// pause the agent meanwhile. Set: the current actor's; read: the current actor's, else any actor's.
    var waitingOnUser: String? {
        get { current.waiting ?? cursors.lazy.compactMap(\.waiting).first }
        set {
            let c = current
            c.waiting = newValue
            c.state = newValue == nil ? .working : .waiting
        }
    }
    var anyoneWaiting: Bool { cursors.contains { $0.waiting != nil } || pendingPlan != nil }

    /// Calls whose change was put back.
    var reverted: Set<String> = []
    @ObservationIgnored var reverts: [String: StudioRevert] = [:]
    @ObservationIgnored var revisits: [String: (place: StudioPlace, target: String?)] = [:]
    /// The last action went through after you used the window: its result says to look again (per actor).
    @ObservationIgnored private var rereads: Set<StudioActor> = []
    var reread: Bool {
        get { rereads.contains(StudioActorScope.actor) }
        set { if newValue { rereads.insert(StudioActorScope.actor) } else { rereads.remove(StudioActorScope.actor) } }
    }
    /// The page each actor is acting on now (or just was, for a moment): another actor doesn't navigate away from it.
    @ObservationIgnored private var busy: [StudioActor: (place: StudioPlace, until: Date)] = [:]
    @ObservationIgnored private var wanderTask: Task<Void, Never>?
    @ObservationIgnored private var wanderRandom = StudioRandom(seed: UInt64(Date().timeIntervalSince1970))
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

    /// A screen rectangle in the overlay's coordinates (top-left of the main window's display).
    func local(_ screen: CGRect, in _: NSWindow? = nil) -> CGRect {
        CGRect(x: screen.minX - overlayFrame.minX, y: overlayFrame.maxY - screen.maxY, width: screen.width, height: screen.height)
    }

    func local(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x - overlayFrame.minX, y: overlayFrame.maxY - p.y) }

    /// The display the overlay covers: the main window's (its visible frame, without the menu bar and Dock).
    var displayFrame: CGRect { (mainWindow?.screen ?? NSScreen.main)?.visibleFrame ?? CGRect(x: 0, y: 0, width: 1440, height: 900) }

    /// Where a control in the Studio panel is on screen (a dock spot).
    func panelFrame(_ target: String?) -> CGRect? {
        guard let target, let f = MGTestAnchors.frame(target), MGTestAnchors.window(target) === panel else { return nil }
        return f
    }

    var reduceMotion: Bool { NSWorkspace.shared.accessibilityDisplayShouldReduceMotion }

    func motions(_ c: StudioCursor) -> StudioMotions {
        StudioMotion.motions(state: gate.isPaused && c.state != .idle ? .paused : c.state, pace: pace, reduceMotion: reduceMotion)
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
        guard isWorking, !anyoneWaiting, Date() > syntheticUntil, let main = mainWindow else { return }
        guard e.window === main || e.window?.parent === main, !(e.window is StudioOverlayWindow) else { return }
        if e.type == .scrollWheel, !(e.phase == .began || (e.phase.isEmpty && e.momentumPhase.isEmpty)) { return }
        gate.pauseByUser()
    }

    // MARK: actions

    /// One action: waits at the gate, goes to `place`, moves the pointer to `step.target` with the caption, does
    /// `work`, then holds the ring a moment. Throws `StudioSkipped` when you skip it.
    ///
    /// Several actors share one screen: an action on a page that isn't in front navigates there only when no other
    /// actor is acting on a different page right then; otherwise it acts off screen (no pointer, its row says so).
    func act<T>(_ step: StudioStep, at place: StudioPlace = .none, callID: String? = nil,
                fallback: String? = nil, _ work: () async throws -> T) async throws -> T {
        let actor = StudioActorScope.actor
        let c = cursor(actor)
        switch await gate.proceed(step.doing, actor: actor.isArchitect && cursors.count == 1 ? nil : actor.label) {
        case .skipped: throw StudioSkipped()
        case .go(let again): if again { reread = true }
        }
        try Task.checkCancellation()
        c.state = .working
        let offscreen = place != .none && !isFront(place) && !mayNavigate(actor, to: place)
        c.offscreen = offscreen
        let here = place == .none ? currentPlace() : place
        if let callID { revisits[callID] = (here, step.target) }
        if offscreen {
            c.caption = step.doing + " (off screen)"
            withAnimation(.easeOut(duration: 0.2)) { c.ringShown = false }
            if c.position == nil { await fly(c, to: edgeSpot(for: c)) }
            let value = try await work()
            return value
        }
        busy[actor] = (here, .distantFuture)
        defer { busy[actor] = (here, Date().addingTimeInterval(1.5)) }
        if place != .none { await go(place, waitFor: step.target) } else { await settle(step.target) }
        await point(at: step.target, fallback: fallback ?? containerTarget(place), caption: step.doing)
        let value = try await work()
        await hold()
        return value
    }

    /// Whether `place` is the page in front now (a sidebar panel never changes the page).
    func isFront(_ place: StudioPlace) -> Bool {
        guard let model else { return true }
        switch place {
        case .none: return true
        case .panel(let p): return UserDefaults.standard.string(forKey: "sidebarPanel") == p.rawValue
        case .movie(let id, let stage):
            guard case .movie(let front) = model.frontItem, front == id else { return false }
            return stage == nil || model.activeMovie?.stage == stage
        case .workflow(let id): return model.workflowInFront && model.activeTabID == id
        }
    }

    /// Two places show the same page (sidebar panels sit beside any page).
    static func samePage(_ a: StudioPlace, _ b: StudioPlace) -> Bool {
        switch (a, b) {
        case (.none, _), (_, .none), (.panel, _), (_, .panel): return true
        case (.movie(let x, let s), .movie(let y, let t)): return x == y && (s == nil || t == nil || s == t)
        case (.workflow(let x), .workflow(let y)): return x == y
        default: return false
        }
    }

    /// No other actor is acting on (or has just acted on, or holds) a different page.
    func mayNavigate(_ actor: StudioActor, to place: StudioPlace) -> Bool {
        let now = Date()
        busy = busy.filter { $0.value.until > now }
        return !busy.contains { $0.key != actor && !Self.samePage($0.value.place, place) }
            && !holds.contains { $0.key != actor && !Self.samePage($0.value, place) }
    }

    /// Pages held for a cost sheet: nobody navigates away while it waits for the user.
    @ObservationIgnored private var holds: [StudioActor: StudioPlace] = [:]
    /// Named turns (a movie's jobs, the one cost sheet on screen, the viewer): one actor at a time.
    @ObservationIgnored private var locks: Set<String> = []

    /// Waits until the screen may show `place` for the current actor, then holds it there (a cost sheet needs its page).
    func holdScreen(_ place: StudioPlace) async throws {
        let actor = StudioActorScope.actor
        var said = false
        while !mayNavigate(actor, to: place) {
            if !said { current.caption = "Waiting for the screen"; said = true }
            try await Task.sleep(for: .milliseconds(150))
        }
        holds[actor] = place
    }

    func releaseScreen() { holds[StudioActorScope.actor] = nil }

    /// Waits for the turn called `key`, saying `caption` meanwhile, then takes it.
    func lock(_ key: String, caption: String? = nil) async throws {
        var said = false
        while locks.contains(key) {
            if let caption, !said { current.caption = caption; said = true }
            try await Task.sleep(for: .milliseconds(150))
        }
        locks.insert(key)
    }

    func unlock(_ key: String) { locks.remove(key) }

    /// "Worker 2 · " in a worker's waits and captions; nothing for the architect.
    var actorPrefix: String { StudioActorScope.actor.isArchitect ? "" : StudioActorScope.actor.label + " · " }

    /// Whether the current actor's action is off screen (its viewer stays shut).
    var isOffscreen: Bool { current.offscreen }

    /// One fallback action at a screen rectangle (an element of the window, not a named control).
    func act<T>(rect: CGRect?, _ step: StudioStep, callID: String? = nil, _ work: () async throws -> T) async throws -> T {
        switch await gate.proceed(step.doing) {
        case .skipped: throw StudioSkipped()
        case .go(let again): if again { reread = true }
        }
        try Task.checkCancellation()
        if let callID { revisits[callID] = (currentPlace(), nil) }
        cursor(StudioActorScope.actor).state = .working
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

    /// Moves the current actor's pointer to a rectangle on screen (nil: just the caption), rings it and shows the caption.
    func point(screen: CGRect?, caption text: String) async {
        guard let main = mainWindow else { return }
        hideTask?.cancel()
        showOverlay(main)
        let c = current
        c.offscreen = false
        let rect = screen.map { local($0) }
        withAnimation(.easeInOut(duration: 0.2)) {
            c.caption = text
            if let rect {
                c.ring = rect.insetBy(dx: -4, dy: -4)
                c.ringShown = true
            } else {
                c.ringShown = false
            }
        }
        if let rect {
            // The tip just inside the control's lower half, so the caption doesn't cover it.
            let tip = CGPoint(x: rect.minX + min(rect.width / 2, 40), y: rect.midY + min(rect.height / 4, 12))
            await fly(c, to: tip)
            // Landed: it looks at what it works on.
            withAnimation(.spring(duration: 0.3)) { c.gaze = StudioFace.gaze(from: tip, to: CGPoint(x: rect.midX, y: rect.midY)) }
        } else if c.position == nil {
            await fly(c, to: CGPoint(x: local(main.frame).midX, y: local(main.frame).maxY - 80))
        }
    }

    /// Flies `c` from where it is (or its dock) to `to`: a curve in Watchable and Step, straight and quick in Fast, a
    /// fade with Reduce Motion. Returns when it lands.
    func fly(_ c: StudioCursor, to: CGPoint) async {
        let now = Date()
        let motions = motions(c)
        let from = c.displayed(at: now, drift: motions.contains(.drift)) ?? homeSpot(c) ?? to
        guard motions.contains(.glide) else {
            withAnimation(.easeInOut(duration: 0.2)) { c.position = to }
            c.flight = nil
            return
        }
        let f = StudioFlight(from: from, to: to, pace: motions.contains(.curve) ? (pace == .fast ? .watchable : pace) : .fast)
        c.gaze = StudioFace.gaze(from: from, to: to)
        c.flight = f
        c.flightStart = now
        if c.position == nil { c.popIn += 1 }
        c.position = to
        if f.duration > 0 { try? await Task.sleep(for: .seconds(f.duration)) }
    }

    /// Where `c` docks: its task row's icon in the panel, or the panel header's (the architect).
    func homeSpot(_ c: StudioCursor) -> CGPoint? {
        let target = c.homeTarget ?? (c.actor.isArchitect ? "studio-header-icon" : nil)
        guard let f = panelFrame(target) else { return nil }
        let r = local(f)
        return CGPoint(x: r.midX, y: r.midY)
    }

    /// A spot by the main window's edge for an actor acting off screen (or with nothing to point at).
    func edgeSpot(for c: StudioCursor) -> CGPoint {
        guard let main = mainWindow else { return CGPoint(x: 80, y: 80) }
        let r = local(main.frame)
        let x = min(r.maxX - 30, local(displayFrame).maxX - 30)
        return CGPoint(x: x, y: r.minY + 70 + CGFloat(c.actor.number) * 48)
    }

    /// Flies `c` home and docks it there: it shrinks onto the row's icon and goes.
    func dock(_ c: StudioCursor) async {
        withAnimation(.easeOut(duration: 0.25)) { c.ringShown = false }
        c.caption = nil
        if let home = homeSpot(c) { await fly(c, to: home) }
        withAnimation(.easeIn(duration: 0.2)) { c.position = nil }
        c.state = .idle
    }

    /// A worker's task ended: its done or failed moment where it is, then home to its row.
    func finish(_ actor: StudioActor, ok: Bool) async {
        let c = cursor(actor)
        c.waiting = nil
        c.state = ok ? .done : .failed
        c.moment += 1
        try? await Task.sleep(for: .seconds(0.6))
        await dock(c)
    }

    /// The architect hands a task to a worker: the worker starts at its row (with the handoff sparkle) and pops in.
    func handOff(to actor: StudioActor, task: String) async {
        let c = cursor(actor)
        c.homeTarget = "studio-task-icon:\(task)"
        c.taskID = task
        c.state = .working
        c.waiting = nil
        guard let main = mainWindow else { return }
        showOverlay(main)
        let start = homeSpot(c) ?? edgeSpot(for: c)
        c.flight = nil
        if motions(c).contains(.handoff), let from = cursor(.architect).displayed(at: Date(), drift: false) ?? homeSpot(cursor(.architect)) {
            handoff = (from: from, to: start, at: Date())
            try? await Task.sleep(for: .seconds(0.5))
        }
        c.popIn += 1
        withAnimation(.spring(duration: 0.35)) { c.position = start }
    }

    /// The sparkle flying from the architect to a new worker (the overlay draws it).
    var handoff: (from: CGPoint, to: CGPoint, at: Date)?

    /// Covers the main window's display with the overlay (above the panel; hidden while another app is in front).
    func showOverlay(_ main: NSWindow) {
        let d = displayFrame
        if overlayFrame != d { overlayFrame = d }
        overlay.cover(d, appearance: main.effectiveAppearance)
    }

    // MARK: wandering

    /// While the panel is open, no reply runs and the app is in front, an icon now and then flies to a random spot on
    /// the display, hovers, and flies home.
    func scheduleWander() {
        wanderTask?.cancel()
        guard wanders, panelShown, !reduceMotion else { return }
        wanderTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                let delay = StudioFlight.wanderDelay(using: &self.wanderRandom)
                try? await Task.sleep(for: .seconds(delay))
                guard !Task.isCancelled else { return }
                await self.wanderOnce()
            }
        }
    }

    func wanderOnce() async {
        guard wanders, panelShown, !isWorking, NSApp.isActive, let main = mainWindow, !reduceMotion else { return }
        let who = cursors.filter { $0.actor.isArchitect || $0.taskID != nil }
        guard let c = who.randomElement(using: &wanderRandom), c.position == nil || c.state == .idle else { return }
        showOverlay(main)
        c.state = .idle
        let spot = StudioFlight.wanderSpot(in: CGRect(origin: .zero, size: overlayFrame.size), using: &wanderRandom)
        if c.position == nil { c.popIn += 1 }
        await fly(c, to: spot)
        try? await Task.sleep(for: .seconds(Double.random(in: 2.5...5, using: &wanderRandom)))
        // Home again, unless a reply that began meanwhile has put it to work (then its actions move it).
        guard c.state == .idle, !c.ringShown else { return }
        await dock(c)
        if !isWorking, cursors.allSatisfy({ $0.position == nil }) { overlay.orderOut(nil) }
    }

    /// After an action: the ring stays a moment, then fades.
    func hold() async {
        let c = current
        try? await Task.sleep(for: .seconds(gate.pace.holdSeconds))
        withAnimation(.easeOut(duration: 0.3)) { c.ringShown = false }
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
        holds = [:]
        locks = []
        for c in cursors { c.waiting = nil }
        pendingPlan?.answer(.cancelled)
        pendingPlan = nil
        hideTask?.cancel()
        hideTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(1.2))
            guard let self, !Task.isCancelled, !self.isWorking else { return }
            // Everyone goes home: the architect beside the panel header, workers to their rows.
            for c in self.cursors where c.position != nil { await self.dock(c) }
            for c in self.cursors { c.state = .idle; c.offscreen = false }
            try? await Task.sleep(for: .seconds(0.3))
            if !Task.isCancelled, !self.isWorking { self.overlay.orderOut(nil) }
            self.scheduleWander()
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

/// What the user said on the plan card.
enum StudioPlanAnswer: Equatable {
    case approved
    /// Change…: their note, for the architect to re-plan.
    case change(String)
    case rejected
    /// The reply stopped while the card waited.
    case cancelled
}
