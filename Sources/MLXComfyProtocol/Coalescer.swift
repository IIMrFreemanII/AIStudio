import Foundation

/// Hands a stream of events from any thread to the main thread in batches: at most one delivery per `interval`
/// (a display frame by default), however many events arrive. `merge` may fold a new event into the previous one
/// (e.g. two progress steps of the same node: only the newer matters); state changes stay in order.
public final class Coalescer<Event>: @unchecked Sendable {
    public typealias Merge = (_ last: Event, _ new: Event) -> Event?
    public typealias Schedule = (_ after: TimeInterval, _ run: @escaping () -> Void) -> Void

    private let lock = NSLock()
    private var pending: [Event] = []
    private var scheduled = false
    private let interval: TimeInterval
    private let merge: Merge
    private let schedule: Schedule
    private let deliver: ([Event]) -> Void

    public init(interval: TimeInterval = 1.0 / 60,
                schedule: @escaping Schedule = { after, run in DispatchQueue.main.asyncAfter(deadline: .now() + after, execute: run) },
                merge: @escaping Merge = { _, _ in nil },
                deliver: @escaping ([Event]) -> Void) {
        self.interval = interval
        self.schedule = schedule
        self.merge = merge
        self.deliver = deliver
    }

    public func push(_ event: Event) {
        let start: Bool = lock.withLock {
            if let last = pending.last, let merged = merge(last, event) {
                pending[pending.count - 1] = merged
            } else {
                pending.append(event)
            }
            defer { scheduled = true }
            return !scheduled
        }
        if start { schedule(interval) { [weak self] in self?.drain() } }
    }

    /// Delivers whatever is waiting now (call on the delivery thread, e.g. before a run's results are recorded).
    public func drain() {
        let batch: [Event] = lock.withLock {
            defer { pending.removeAll(keepingCapacity: true); scheduled = false }
            return pending
        }
        if !batch.isEmpty { deliver(batch) }
    }

    public var pendingCount: Int { lock.withLock { pending.count } }
}
