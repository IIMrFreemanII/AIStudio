import Foundation

// The Studio agent as an architect: the plan it shows the user, the workers it hands tasks to, and how their icons
// move and fly on screen. Foundation only, so it's unit-tested.

/// Who acts: the architect (0) or one of its workers (1…3).
public struct StudioActor: Hashable, Sendable, Codable, Comparable {
    public var number: Int

    public init(_ number: Int) { self.number = number }

    public static let architect = StudioActor(0)
    public static func worker(_ n: Int) -> StudioActor { StudioActor(n) }

    public var isArchitect: Bool { number == 0 }
    /// "Architect", "Worker 2".
    public var label: String { isArchitect ? "Architect" : "Worker \(number)" }
    public static func < (a: StudioActor, b: StudioActor) -> Bool { a.number < b.number }
}

/// The architect's plan: tasks the user approves before any worker starts, then a checklist the panel keeps showing.
public struct StudioPlan: Codable, Equatable, Sendable {
    public static let maxWorkers = 3
    public static let maxRetries = 2

    public enum Kind: String, Codable, Sendable { case act, review }
    public enum Status: String, Codable, Sendable { case waiting, running, done, failed, needsYou }

    public struct Task: Codable, Equatable, Sendable, Identifiable {
        public var id: String
        public var title: String
        public var brief: String
        public var kind: Kind
        /// The paid stages it runs ("keyframes for shots 1–4"), each through a cost sheet.
        public var paid: [String]
        /// A rough estimate of what its paid stages cost, in dollars.
        public var estimate: Double?
        public var dependsOn: [String]
        public var status: Status = .waiting
        /// The worker it's (to be) given to, 1…3.
        public var worker: Int
        public var retries = 0
        public var spent: Double = 0
        public var report: String?
    }

    public var tasks: [Task]

    public init(tasks: [Task]) { self.tasks = tasks }

    /// Why a task can't start (the tool's error text).
    public struct Refusal: Error, CustomStringConvertible, Equatable {
        public var description: String
        public init(_ d: String) { description = d }
    }

    /// Reads the `studio_plan` tool's `tasks` argument: [{id, title, brief, kind?, paid?, estimate?, depends_on?}].
    public static func parse(_ raw: Any?) throws -> StudioPlan {
        guard let items = raw as? [Any], !items.isEmpty else { throw Refusal("\"tasks\" is missing: a list of tasks") }
        var tasks: [Task] = []
        var seen = Set<String>()
        for (i, item) in items.enumerated() {
            guard let o = item as? [String: Any] else { throw Refusal("task \(i + 1) isn't an object") }
            func str(_ k: String) -> String? {
                (o[k] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
            }
            let id = str("id") ?? "t\(i + 1)"
            guard !seen.contains(id) else { throw Refusal("two tasks have the id “\(id)”") }
            seen.insert(id)
            guard let title = str("title") else { throw Refusal("task “\(id)” has no title") }
            guard let brief = str("brief") else { throw Refusal("task “\(id)” has no brief: workers only see their brief") }
            let kind = Kind(rawValue: str("kind")?.lowercased() ?? "act") ?? .act
            let paid = (o["paid"] as? [Any])?.compactMap { ($0 as? String)?.nilIfEmpty } ?? str("paid").map { [$0] } ?? []
            let estimate = (o["estimate"] as? NSNumber)?.doubleValue ?? Double(str("estimate")?.replacingOccurrences(of: "$", with: "") ?? "")
            let deps = (o["depends_on"] as? [Any] ?? o["dependsOn"] as? [Any])?.compactMap { $0 as? String } ?? []
            tasks.append(Task(id: id, title: title, brief: brief, kind: kind, paid: paid, estimate: estimate, dependsOn: deps,
                              worker: i % maxWorkers + 1))
        }
        for t in tasks {
            for d in t.dependsOn where !seen.contains(d) { throw Refusal("task “\(t.id)” depends on “\(d)”, which isn't in the plan") }
            if t.dependsOn.contains(t.id) { throw Refusal("task “\(t.id)” depends on itself") }
        }
        return StudioPlan(tasks: tasks)
    }

    public func task(_ id: String) -> Task? { tasks.first { $0.id == id } }
    private func index(_ id: String) -> Int? { tasks.firstIndex { $0.id == id } }

    /// The plan's paid stages, all tasks.
    public var estimate: Double { tasks.compactMap(\.estimate).reduce(0, +) }
    public var running: Int { tasks.filter { $0.status == .running }.count }
    public var isFinished: Bool { tasks.allSatisfy { $0.status == .done || $0.status == .failed } }

    /// The tasks `id` waits for that aren't done.
    public func blockers(_ id: String) -> [String] {
        guard let t = task(id) else { return [] }
        return t.dependsOn.filter { task($0)?.status != .done }
    }

    /// Whether a free worker slot exists now.
    public var hasFreeWorker: Bool { running < Self.maxWorkers }

    /// Starts task `id` (or retries it with `retry`): the worker it runs on. Throws when it isn't in the plan, waits on
    /// another task, is already running, or has had its two retries.
    public mutating func start(_ id: String, retry: Bool = false) throws -> StudioActor {
        guard let i = index(id) else {
            throw Refusal("there's no task “\(id)” in the approved plan: tasks are \(tasks.map(\.id).joined(separator: ", "))")
        }
        let t = tasks[i]
        if t.status == .running { throw Refusal("task “\(id)” is already running") }
        let waits = blockers(id)
        if !waits.isEmpty { throw Refusal("task “\(id)” waits for \(waits.map { "“\($0)”" }.joined(separator: ", ")) to be done first") }
        if retry {
            guard t.status != .waiting else { throw Refusal("task “\(id)” hasn't run yet: delegate it without feedback first") }
            guard t.retries < Self.maxRetries else {
                throw Refusal("task “\(id)” has had its \(Self.maxRetries) retries: do it yourself, or ask the user (say what it would cost if it needs another paid run)")
            }
        } else if t.status == .done || t.status == .failed {
            throw Refusal("task “\(id)” has run already: send the worker feedback to redo it")
        }
        guard hasFreeWorker else { throw Refusal("all \(Self.maxWorkers) workers are busy") }
        // Its planned worker, unless that one is busy with another task: then the lowest free one.
        let busy = Set(tasks.filter { $0.status == .running }.map(\.worker))
        let worker = busy.contains(t.worker) ? (1...Self.maxWorkers).first { !busy.contains($0) } ?? t.worker : t.worker
        tasks[i].worker = worker
        tasks[i].status = .running
        if retry { tasks[i].retries += 1 }
        return .worker(worker)
    }

    public mutating func finish(_ id: String, ok: Bool, spent: Double = 0, report: String? = nil) {
        guard let i = index(id) else { return }
        tasks[i].status = ok ? .done : .failed
        tasks[i].spent += spent
        if let report { tasks[i].report = report }
    }

    public mutating func set(_ id: String, _ status: Status) {
        guard let i = index(id) else { return }
        tasks[i].status = status
    }
}

extension String {
    fileprivate var nilIfEmpty: String? { isEmpty ? nil : self }
}

// MARK: icon motion

/// What an actor's icon shows: its moment in the work.
public enum StudioActorState: String, Sendable, CaseIterable {
    case idle, working, waiting, paused, done, failed
}

/// The motions an icon may play.
public struct StudioMotions: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// The sparkle breathes and the small one blinks.
    public static let twinkle = StudioMotions(rawValue: 1 << 0)
    /// An amber ring pulses outwards.
    public static let pulse = StudioMotions(rawValue: 1 << 1)
    /// A steady amber ring (waiting, without motion).
    public static let ring = StudioMotions(rawValue: 1 << 2)
    /// The sparkle bursts into a check mark.
    public static let burst = StudioMotions(rawValue: 1 << 3)
    /// The icon shakes once and shows a cross.
    public static let shake = StudioMotions(rawValue: 1 << 4)
    /// Dimmed and frozen.
    public static let dim = StudioMotions(rawValue: 1 << 5)
    /// Drifting around its home spot when it has nothing to point at.
    public static let drift = StudioMotions(rawValue: 1 << 6)
    /// Moves glide (otherwise the icon fades out and in at the new spot).
    public static let glide = StudioMotions(rawValue: 1 << 7)
    /// Glides curve (otherwise straight).
    public static let curve = StudioMotions(rawValue: 1 << 8)
    /// A sparkle flies from the architect to a new worker, which pops in.
    public static let handoff = StudioMotions(rawValue: 1 << 9)
    /// Wandering around the screen while the app is idle.
    public static let wander = StudioMotions(rawValue: 1 << 10)
}

public enum StudioMotion {
    /// Which motions an icon in `state` plays at `pace`. Fast keeps only the essentials (the waiting pulse, done and
    /// failed); Reduce Motion turns every motion off (waiting shows a steady ring; paused still dims).
    public static func motions(state: StudioActorState, pace: StudioPace, reduceMotion: Bool) -> StudioMotions {
        if reduceMotion {
            switch state {
            case .waiting: return .ring
            case .paused: return .dim
            default: return []
            }
        }
        let full = pace != .fast
        var m: StudioMotions = [.glide]
        if full { m.formUnion([.curve, .handoff]) }
        switch state {
        case .idle: if full { m.formUnion([.drift, .wander]) }
        case .working: if full { m.formUnion([.twinkle, .drift]) }
        case .waiting: m.insert(.pulse)
        case .paused: m = [.dim]
        case .done: m.insert(.burst)
        case .failed: m.insert(.shake)
        }
        return m
    }
}

// MARK: flight

/// A flight across the screen from one spot to another: a gentle curve in Watchable and Step pace, straight and quick
/// in Fast. Points are in any one coordinate space (the overlay's).
public struct StudioFlight: Equatable, Sendable {
    public var from: CGPoint
    public var to: CGPoint
    /// How far the curve bends sideways, as a fraction of the distance (0: straight).
    public var bend: Double
    public var duration: Double

    public init(from: CGPoint, to: CGPoint, bend: Double, duration: Double) {
        self.from = from
        self.to = to
        self.bend = bend
        self.duration = duration
    }

    public init(from: CGPoint, to: CGPoint, pace: StudioPace) {
        let d = Self.distance(from, to)
        self.init(from: from, to: to, bend: pace == .fast ? 0 : 0.15, duration: Self.duration(distance: d, pace: pace))
    }

    public static func distance(_ a: CGPoint, _ b: CGPoint) -> Double { Double(hypot(b.x - a.x, b.y - a.y)) }

    /// Seconds a flight of `distance` points takes: about 0.35–0.9 s, longer for longer flights; 0.15 s in Fast.
    public static func duration(distance: Double, pace: StudioPace) -> Double {
        if pace == .fast { return distance < 1 ? 0 : 0.15 }
        guard distance >= 1 else { return 0 }
        return min(0.9, 0.35 + distance / 1800 * 0.55)
    }

    /// The curve's control point: the midpoint, pushed sideways (to the left of the direction of travel).
    public var control: CGPoint {
        let mid = CGPoint(x: (from.x + to.x) / 2, y: (from.y + to.y) / 2)
        let dx = to.x - from.x, dy = to.y - from.y
        let len = max(Double(hypot(dx, dy)), 0.0001)
        let push = bend * len
        return CGPoint(x: mid.x - dy / len * push, y: mid.y + dx / len * push)
    }

    /// Where the icon is at `t` seconds into the flight (eased in and out).
    public func point(at t: Double) -> CGPoint {
        guard duration > 0 else { return to }
        let u = Self.ease(min(max(t / duration, 0), 1))
        let c = control
        let a = (1 - u) * (1 - u), b = 2 * (1 - u) * u, e = u * u
        return CGPoint(x: a * from.x + b * c.x + e * to.x, y: a * from.y + b * c.y + e * to.y)
    }

    public func isOver(at t: Double) -> Bool { t >= duration }

    static func ease(_ u: Double) -> Double { u < 0.5 ? 2 * u * u : 1 - pow(-2 * u + 2, 2) / 2 }

    /// The idle drift of actor `n` at `time` (seconds): a slow loop about 18 points wide, out of phase per actor.
    public static func drift(actor n: Int, time: Double) -> CGVector {
        let period = 6.0 + Double(n % 4) * 0.7
        let phase = Double(n) * 1.9
        let w = 2 * Double.pi * time / period + phase
        return CGVector(dx: 9 * sin(w), dy: 6 * sin(2 * w + 0.6))
    }

    /// A spot to wander to inside `rect`, at least `inset` points from its edges.
    public static func wanderSpot<R: RandomNumberGenerator>(in rect: CGRect, inset: Double = 80, using rng: inout R) -> CGPoint {
        let r = rect.insetBy(dx: min(inset, rect.width / 2), dy: min(inset, rect.height / 2))
        return CGPoint(x: Double.random(in: Double(r.minX)...Double(r.maxX), using: &rng),
                       y: Double.random(in: Double(r.minY)...Double(r.maxY), using: &rng))
    }

    /// Seconds until the next wander: 10–25.
    public static func wanderDelay<R: RandomNumberGenerator>(using rng: inout R) -> Double {
        Double.random(in: 10...25, using: &rng)
    }
}

/// A repeatable random source (SplitMix64), so wandering can be tested.
public struct StudioRandom: RandomNumberGenerator, Sendable {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
