import Foundation
import Observation

// The Studio agent's pacing: how fast its actions play out on screen, and the gate every action passes first
// (paused because the user used the window, or waiting for Next in Step pace). Foundation only, so it's unit-tested.

/// How fast the Studio agent's actions play out in the main window.
public enum StudioPace: String, CaseIterable, Codable, Sendable, Identifiable {
    /// The pointer jumps, the ring flashes, text appears at once.
    case fast
    /// The pointer glides, the ring holds, text types in: a few seconds per action.
    case watchable
    /// Like Watchable, and each action waits for the user's Next.
    case step

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .fast: return "Fast"
        case .watchable: return "Watchable"
        case .step: return "Step by Step"
        }
    }

    public var detail: String {
        switch self {
        case .fast: return "Actions happen at once"
        case .watchable: return "The pointer moves and text types in, so you can follow"
        case .step: return "Each action waits for you to press Next"
        }
    }

    /// Seconds the pointer takes to reach its target.
    public var moveSeconds: Double { self == .fast ? 0.1 : 0.4 }
    /// Seconds the ring stays on the target after the action, before it fades.
    public var holdSeconds: Double { self == .fast ? 0.2 : 0.8 }
    /// Characters typed per second into a field (nil: the text appears at once).
    public var typingRate: Double? { self == .fast ? nil : 60 }

    /// How long typing `count` characters takes, at most `cap` seconds (long prompts type faster).
    public func typingSeconds(_ count: Int, cap: Double = 2.5) -> Double {
        guard let rate = typingRate, count > 0 else { return 0 }
        return min(Double(count) / rate, cap)
    }
}

/// What one Studio action is called: "Typing shot 3's prompt" while it happens (the caption), "Typed shot 3's prompt"
/// in the step list, and the anchor it acts on (`mgTarget` names).
public struct StudioStep: Equatable, Sendable {
    public var doing: String
    public var done: String
    public var target: String?

    public init(doing: String, done: String, target: String? = nil) {
        self.doing = doing
        self.done = done
        self.target = target
    }
}

/// The gate every Studio action passes before it acts: open while running, closed while the user paused it (by using
/// the main window) and, in Step pace, until the user presses Next or Skip. Only actions wait; the model's text keeps
/// streaming.
@MainActor @Observable
public final class StudioGate {
    public enum State: Equatable, Sendable {
        case running
        /// The user clicked, typed or scrolled in the main window.
        case pausedByUser
        /// Step pace: the next action, waiting for Next or Skip.
        case awaitingNext(String)
    }

    /// What an action may do once the gate lets it through.
    public enum Outcome: Equatable, Sendable {
        /// Go ahead. `reread`: the user used the window while it waited, so the page may have changed.
        case go(reread: Bool)
        /// The user pressed Skip (or the reply was stopped): don't act.
        case skipped
    }

    public private(set) var state: State = .running
    public var pace: StudioPace

    /// The user used the window since the last action went through.
    public private(set) var interrupted = false

    @ObservationIgnored private var waiters: [UUID: CheckedContinuation<Bool, Never>] = [:]

    public init(pace: StudioPace = .watchable) {
        self.pace = pace
    }

    public var isPaused: Bool { state == .pausedByUser }

    /// The user used the main window: the next action waits for Resume.
    public func pauseByUser() {
        interrupted = true
        if state == .running { state = .pausedByUser }
    }

    public func resume() {
        guard state == .pausedByUser else { return }
        state = .running
        release(true)
    }

    /// Step pace: the waiting action goes ahead.
    public func next() {
        guard case .awaitingNext = state else { return }
        state = .running
        release(true)
    }

    /// Step pace: the waiting action is skipped.
    public func skip() {
        guard case .awaitingNext = state else { return }
        state = .running
        release(false)
    }

    /// The reply ended: anything waiting is let go without acting, and the gate opens for the next reply.
    public func reset() {
        state = .running
        interrupted = false
        release(false)
    }

    /// Waits until the action titled `doing` may run: while paused, until Resume; in Step pace, until Next or Skip.
    /// A cancelled task gets `.skipped`.
    public func proceed(_ doing: String) async -> Outcome {
        while state == .pausedByUser {
            if await wait() == false { return .skipped }
        }
        if Task.isCancelled { return .skipped }
        if pace == .step {
            state = .awaitingNext(doing)
            if await wait() == false { return .skipped }
            // Paused while waiting for Next (the user clicked the window): Next already let it go.
        }
        if Task.isCancelled { return .skipped }
        let reread = interrupted
        interrupted = false
        return .go(reread: reread)
    }

    private func wait() async -> Bool {
        let id = UUID()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
                if Task.isCancelled { cont.resume(returning: false); return }
                waiters[id] = cont
            }
        } onCancel: {
            Task { @MainActor [weak self] in self?.cancel(id) }
        }
    }

    private func cancel(_ id: UUID) {
        if case .awaitingNext = state { state = .running }
        waiters.removeValue(forKey: id)?.resume(returning: false)
    }

    private func release(_ go: Bool) {
        let all = waiters.values
        waiters.removeAll()
        for w in all { w.resume(returning: go) }
    }

    /// What a tool result starts with after the user used the window while the agent waited.
    nonisolated public static let rereadNote = "The user used the window while you worked, so the page may have changed: check with studio_state before relying on what you saw earlier."
    nonisolated public static let skippedNote = "The user skipped this action: it wasn't done. Go on without it, or ask them."
}

/// Unused file names: "The Last Light.mp4", else "The Last Light 2.mp4", "… 3"… (exports never overwrite).
public enum StudioFiles {
    public static func unused(_ name: String, ext: String, in folder: URL, exists: (URL) -> Bool) -> URL {
        let base = name.trimmingCharacters(in: .whitespacesAndNewlines).replacingOccurrences(of: "/", with: "-")
        let stem = base.isEmpty ? "Movie" : base
        var url = folder.appendingPathComponent(stem).appendingPathExtension(ext)
        var n = 2
        while exists(url) {
            url = folder.appendingPathComponent("\(stem) \(n)").appendingPathExtension(ext)
            n += 1
        }
        return url
    }
}
