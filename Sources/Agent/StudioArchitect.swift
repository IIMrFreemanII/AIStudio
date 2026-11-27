import Foundation
import CoreGraphics

// The Studio agent as an architect: the plan it shows the user, the workers it hands tasks to, and how their icons
// move and fly on screen. Foundation only, so it's unit-tested.

/// Who acts: the architect (0) or one of its workers (1…3).
public struct StudioActor: Hashable, Sendable, Codable, Comparable {
    public var number: Int

    public init(_ number: Int) { self.number = number }

    public static let architect = StudioActor(0)
    public static func worker(_ n: Int) -> StudioActor { StudioActor(n) }

    public var isArchitect: Bool { number == 0 }
    /// The workers' names: short, and after their colours (teal, coral, violet).
    public static let workerNames = ["Mint", "Ember", "Plum"]

    /// "Architect", "Mint", "Ember", "Plum".
    public var label: String {
        if isArchitect { return "Architect" }
        return number >= 1 && number <= Self.workerNames.count ? Self.workerNames[number - 1] : "Worker \(number)"
    }
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

    /// How fast and which way the icon moves at `t` seconds into the flight, in points per second (zero outside it).
    public func velocity(at t: Double) -> CGVector {
        guard duration > 0, t >= 0, t <= duration else { return CGVector(dx: 0, dy: 0) }
        let h = min(0.008, duration / 4)
        let a = point(at: max(t - h, 0)), b = point(at: min(t + h, duration))
        let dt = min(t + h, duration) - max(t - h, 0)
        guard dt > 0 else { return CGVector(dx: 0, dy: 0) }
        return CGVector(dx: (b.x - a.x) / dt, dy: (b.y - a.y) / dt)
    }

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

// MARK: trail

/// One ghost echo behind a flying orb: where it is, how big and how visible (both fractions of the orb), and how far
/// back it sits (0 at the front echo, 1 at the back: picks the architect's colour).
public struct StudioEcho: Equatable, Sendable {
    public var point: CGPoint
    public var scale: Double
    public var opacity: Double
    public var band: Double

    public init(point: CGPoint, scale: Double, opacity: Double, band: Double) {
        self.point = point
        self.scale = scale
        self.opacity = opacity
        self.band = band
    }
}

/// A flight and when it began (seconds, any one clock): the path an orb's echoes follow.
public struct StudioTrailLeg: Equatable, Sendable {
    public var flight: StudioFlight
    public var start: Double

    public init(flight: StudioFlight, start: Double) {
        self.flight = flight
        self.start = start
    }

    public var end: Double { start + flight.duration }
}

/// The ghost echoes a flying orb leaves: five glass silhouettes along the last 0.3 s of its path (0.12 s in Fast; at
/// most 150 points), each smaller and fainter. After landing they slide into the orb over 0.35 s and are gone.
public enum StudioTrail {
    public static let count = 5
    /// Seconds the echoes take to pull into the orb after it lands.
    public static let pull = 0.35

    /// The most path (points) the echoes cover: at full speed they overlap into one trail instead of scattering.
    public static let longest = 150.0

    /// How far back along the path (seconds) the last echo sits: shorter for Fast pace's quick flights.
    public static func span(duration: Double) -> Double { duration <= 0.2 ? 0.12 : 0.3 }

    /// The span at `t` seconds into `flight`: `span(duration:)`, shortened while it flies fast so the trail is at most
    /// `longest` points long.
    public static func span(_ flight: StudioFlight, at t: Double) -> Double {
        let base = span(duration: flight.duration)
        let dt = 1.0 / 120
        let tt = min(max(t, dt), flight.duration)
        let speed = StudioFlight.distance(flight.point(at: tt - dt), flight.point(at: tt)) / dt
        return speed > 0 ? min(base, longest / speed) : base
    }

    /// Whether `legs` still show echoes at `now`.
    public static func isActive(_ legs: [StudioTrailLeg], now: Double) -> Bool {
        guard let last = legs.last, last.flight.duration > 0 else { return false }
        return now >= last.start && now < last.end + pull
    }

    /// The echoes at `now`, back to front. Each follows the leg that was flying at its own moment, so a flight replaced
    /// mid-air keeps its old path behind; moments before the first leg have no echo. `offset` is the orb's drift
    /// after landing, which the echoes take on as they pull in.
    public static func echoes(_ legs: [StudioTrailLeg], now: Double, offset: CGVector = .zero) -> [StudioEcho] {
        guard let last = legs.last, last.flight.duration > 0, now >= last.start else { return [] }
        let k = now < last.end ? 1 : max(0, 1 - (now - last.end) / pull)
        guard k > 0 else { return [] }
        let span = span(last.flight, at: now - last.start)
        let drift = 1 - k
        var out: [StudioEcho] = []
        for i in stride(from: count, through: 1, by: -1) {
            let u = Double(i) / Double(count)
            let t = now - span * u * k
            // An older leg counts only while it was still flying (a resting orb leaves no echoes behind its next start).
            guard let leg = legs.last(where: { $0.start <= t && ($0 == last || t <= $0.end) }) else { continue }
            let p = leg.flight.point(at: t - leg.start)
            out.append(StudioEcho(point: CGPoint(x: p.x + offset.dx * drift, y: p.y + offset.dy * drift),
                                  scale: 1 - 0.35 * u, opacity: 0.55 * (1 - 0.85 * u) * k,
                                  band: Double(i - 1) / Double(count - 1)))
        }
        return out
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

// MARK: orb faces

/// The orb characters' eyes.
public enum StudioEyes: String, Sendable, CaseIterable {
    /// Two upright pills.
    case open
    /// Slightly narrowed: concentrating on a step.
    case focus
    /// Taller and a little wider: waiting on the user.
    case wide
    /// Happy arcs (^ ^): a task done.
    case happy
    /// Small crosses: a task failed.
    case cross
    /// Flat lines: paused.
    case closed
}

/// What an orb's face does: its eyes, and which small motions it may play (blinking, glancing, the talking bounce,
/// particles, the architect's orbiting sparkle, the breathing glow). Particles are always on; Fast pace otherwise keeps
/// the moods only; Reduce Motion keeps the eye shapes and nothing that moves.
public struct StudioFace: Equatable, Sendable {
    public var eyes: StudioEyes
    public var blinks = false
    public var glances = false
    public var bounce = false
    public var particles = false
    public var orbit = false
    public var breathe = false

    public init(eyes: StudioEyes, blinks: Bool = false, glances: Bool = false, bounce: Bool = false, particles: Bool = false,
                orbit: Bool = false, breathe: Bool = false) {
        self.eyes = eyes
        self.blinks = blinks
        self.glances = glances
        self.bounce = bounce
        self.particles = particles
        self.orbit = orbit
        self.breathe = breathe
    }

    /// `moving`: flying or wandering (particles trail it).
    public static func face(state: StudioActorState, talking: Bool = false, moving: Bool = false, pace: StudioPace,
                            reduceMotion: Bool) -> StudioFace {
        let eyes: StudioEyes
        switch state {
        case .idle: eyes = .open
        case .working: eyes = talking ? .open : .focus
        case .waiting: eyes = .wide
        case .done: eyes = .happy
        case .failed: eyes = .cross
        case .paused: eyes = .closed
        }
        var f = StudioFace(eyes: eyes)
        guard !reduceMotion else { return f }
        f.breathe = state == .working
        // Sparkles drift around every orb, all the time (the user's choice), whatever the pace or mood.
        f.particles = true
        guard pace != .fast else { return f }
        let awake = state == .idle || state == .working || state == .waiting
        f.blinks = awake
        f.glances = state == .idle && !moving
        f.bounce = talking && state != .paused
        f.orbit = state == .working || moving
        return f
    }

    /// Whether anything moves continuously (the orb redraws every frame); blinks and glances alone are discrete.
    public var animates: Bool { bounce || particles || orbit || breathe }

    static let cycle = 120.0
    static let blinkLength = 0.12

    /// Blink start times within one 120 s cycle for `seed`, 3–6 s apart.
    public static func blinkTimes(seed: UInt64) -> [Double] {
        var rng = StudioRandom(seed: seed &+ 0xB1)
        var t = Double.random(in: 0.5...3, using: &rng)
        var out: [Double] = []
        while t < cycle - blinkLength {
            out.append(t)
            t += Double.random(in: 3...6, using: &rng)
        }
        return out
    }

    /// Whether the eyes are shut for a blink at `time` (seconds, any clock).
    public static func isBlinking(at time: Double, seed: UInt64) -> Bool {
        let u = time.truncatingRemainder(dividingBy: cycle)
        return blinkTimes(seed: seed).contains { u >= $0 && u < $0 + blinkLength }
    }

    /// The idle glance at `time`: mostly straight ahead, now and then a look to one side for a second or two.
    public static func glance(at time: Double, seed: UInt64) -> CGVector {
        let u = time.truncatingRemainder(dividingBy: cycle)
        for g in glanceTimes(seed: seed) where u >= g.start && u < g.start + g.length { return g.direction }
        return .zero
    }

    static func glanceTimes(seed: UInt64) -> [(start: Double, length: Double, direction: CGVector)] {
        var rng = StudioRandom(seed: seed &+ 0x61A)
        let directions = [CGVector(dx: -1, dy: 0), CGVector(dx: 1, dy: 0), CGVector(dx: 0.7, dy: -0.7), CGVector(dx: -0.7, dy: 0.7),
                          CGVector(dx: 0, dy: 1)]
        var t = Double.random(in: 1...4, using: &rng)
        var out: [(Double, Double, CGVector)] = []
        while t < cycle - 2 {
            let length = Double.random(in: 0.8...1.8, using: &rng)
            out.append((t, length, directions[Int.random(in: 0..<directions.count, using: &rng)]))
            t += length + Double.random(in: 2.5...6, using: &rng)
        }
        return out
    }

    /// The moments a still (not continuously animating) face changes in the next `span` seconds after `time`: blink
    /// starts and ends, glance starts and ends. A view redraws only then.
    public static func changes(after time: Double, span: Double = 120, seed: UInt64, blinks: Bool, glances: Bool) -> [Double] {
        let base = time - time.truncatingRemainder(dividingBy: cycle)
        var marks: [Double] = []
        for c in [base, base + cycle] {
            if blinks { for b in blinkTimes(seed: seed) { marks += [c + b, c + b + blinkLength] } }
            if glances { for g in glanceTimes(seed: seed) { marks += [c + g.start, c + g.start + g.length] } }
        }
        return marks.filter { $0 > time && $0 <= time + span }.sorted()
    }

    /// Which way the eyes look from `from` towards `to`: a vector at most `maxShift` long (zero when they're the same).
    public static func gaze(from: CGPoint, to: CGPoint, maxShift: Double = 1) -> CGVector {
        let dx = Double(to.x - from.x), dy = Double(to.y - from.y)
        let len = hypot(dx, dy)
        guard len > 0.5 else { return .zero }
        let k = min(1, len / 40) * maxShift / len
        return CGVector(dx: dx * k, dy: dy * k)
    }

    /// The particles around actor `n`'s orb at `time`: offsets (as fractions of the orb's size, from its centre) and
    /// opacities. They circle slowly between 0.6 and 0.95 of the size out, fading in and out.
    public static func particles(actor n: Int, time: Double, count: Int = 5) -> [(offset: CGVector, opacity: Double)] {
        (0..<count).map { i in
            let k = Double(i) + Double(n) * 0.37
            let speed = 0.35 + 0.12 * Double((i * 7 + n * 3) % 5)
            let angle = k * 2.399 + time * speed * (i % 2 == 0 ? 1 : -1)
            let r = 0.775 + 0.175 * sin(time * 0.6 + k * 1.7)
            let opacity = 0.25 + 0.75 * (0.5 + 0.5 * sin(time * 1.3 + k * 2.1))
            return (CGVector(dx: cos(angle) * r, dy: sin(angle) * r), opacity)
        }
    }
}

// MARK: 3D orb pose

/// The eye shapes the 3D orb's shader draws.
public enum StudioOrbEyeShape: Int, Sendable {
    case pill = 0
    case happy = 1
    case cross = 2
}

/// The look of the 3D orb's glossy dark glass. The defaults are "Close to today" from the mock-up page.
public struct StudioOrbGlass: Equatable, Sendable {
    /// The key-light highlight's half-size, in radians.
    public var highlightSize = 0.16
    public var highlightStrength = 0.85
    /// The rim colour from the viewing angle.
    public var rim = 0.85
    /// The frosted band, as a fraction of the radius.
    public var frost = 0.17
    /// How far the glow inside the glass spreads, in radii.
    public var glowSpread = 0.55
    public var glowStrength = 1.0
    /// The architect's colour-shifting film.
    public var iridescence = 0.7

    public init() {}
    public static let standard = StudioOrbGlass()
}

/// What the 3D orb's shader draws at one moment: how the sphere is turned, the eyes' shape and size on its surface,
/// and how bright the glow inside the glass is. Lengths are in sphere radii, angles in radians, y points up.
public struct StudioOrbPose: Equatable, Sendable {
    public var eyeShape: StudioOrbEyeShape
    public var eyeWidth: Double
    public var eyeHeight: Double
    /// Between the two eye centres.
    public var eyeGap: Double
    /// The talking bounce: moves the eyes up (positive) or down on the surface.
    public var eyeLift: Double
    /// Turned right (positive) or left.
    public var yaw: Double
    /// Turned up (positive) or down.
    public var pitch: Double
    /// Tilted anticlockwise (positive).
    public var roll: Double
    /// The glow inside the glass: 1 normally.
    public var glow: Double

    /// A pill eye at rest.
    public static let eyeWidth = 0.26, eyeHeight = 0.66
    /// How far the sphere turns for a full look (gaze 1) at turn 1.
    public static let maxYaw = 0.7, maxPitch = 0.55
    /// How much the sphere turns toward its gaze (the mock-up's "Turn toward gaze").
    public static let standardTurn = 0.6
    /// How much it tilts into a flight (the mock-up's "Bank when flying"); full tilt at `bankSpeed` points per second.
    public static let standardBank = 0.6, bankSpeed = 700.0
    /// The architect's ring, tilted 18° anticlockwise at rest.
    public static let ringTilt = 0.314
    /// The inner glow's slow drift, in radians per second (the mock-up's "Idle spin" at 0.5).
    public static let spinRate = 0.175

    /// - Parameters:
    ///   - gaze: where it looks, up to 1 long, x right and y down (as `StudioOrb`'s gaze).
    ///   - bounce: the talking bounce's phase, -1…1 (0 when it isn't talking).
    ///   - breathe: the glow's breathing, 0.8…1 while working (1 otherwise).
    ///   - radius: the orb's radius in points, to keep the eyes at least 1.6 pt wide and 1 pt tall.
    ///   - velocity: how fast it flies, in points per second (y down): it tilts into the way it's going.
    public static func pose(face: StudioFace, blinking: Bool = false, gaze: CGVector = CGVector(dx: 0, dy: 0), bounce: Double = 0,
                            breathe: Double = 1, paused: Bool = false, radius: Double = 16,
                            turn: Double = standardTurn, velocity: CGVector = CGVector(dx: 0, dy: 0),
                            bank: Double = standardBank) -> StudioOrbPose {
        let w = eyeWidth, h = eyeHeight
        var shape = StudioOrbEyeShape.pill
        var size: (Double, Double)
        switch face.eyes {
        case .open: size = (w, h)
        case .focus: size = (w, h * 0.72)
        case .wide: size = (w * 1.08, h * 1.15)
        case .closed: size = (w * 1.3, h * 0.14)
        case .happy: shape = .happy; size = (w, h)
        case .cross: shape = .cross; size = (w, h)
        }
        if blinking, shape == .pill, face.eyes != .closed { size = (size.0 * 1.1, h * 0.12) }
        let r = max(radius, 1)
        let g = Self.clampedGaze(gaze)
        let t = min(max(turn, 0), 1)
        return StudioOrbPose(
            eyeShape: shape,
            eyeWidth: max(size.0, 1.6 / r),
            eyeHeight: max(size.1, 1 / r),
            eyeGap: shape == .pill ? 0.42 : 0.54,
            eyeLift: -0.094 * bounce,
            yaw: Double(g.dx) * maxYaw * t,
            pitch: -Double(g.dy) * maxPitch * t,
            roll: -min(max(Double(velocity.dx) / bankSpeed, -1), 1) * min(max(bank, 0), 1) * 0.5,
            glow: breathe * (paused ? 0.35 : 1) * (radius < 12 ? 0.75 : 1))
    }

    /// The architect's ring tilt: its rest tilt, leaning with the body.
    public var ringRoll: Double { Self.ringTilt + roll * 0.6 }

    /// The inner glow's drift angle at `time` (seconds), kept within one turn.
    public static func spin(at time: Double) -> Double {
        (time * spinRate).truncatingRemainder(dividingBy: 2 * .pi)
    }

    /// The talking bounce's squash: how much taller (positive) or wider the orb is at `time`.
    public static func talkStretch(at time: Double) -> Double { 0.021 * sin(time * 2 * .pi * 3) }

    static func clampedGaze(_ v: CGVector) -> CGVector {
        let l = hypot(v.dx, v.dy)
        return l > 1 ? CGVector(dx: v.dx / l, dy: v.dy / l) : v
    }

    /// The sphere's turn as a 3×3 rotation, column by column (body → view): roll · yaw · pitch. The view looks down -z
    /// at the sphere, so the face's front is +z.
    public var rotation: [Float] {
        let cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch), cr = cos(roll), sr = sin(roll)
        // Rz(roll) · Ry(yaw) · Rx(-pitch): a positive pitch turns the front up.
        let rz: [[Double]] = [[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]
        let ry: [[Double]] = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
        let rx: [[Double]] = [[1, 0, 0], [0, cp, sp], [0, -sp, cp]]
        func mul(_ a: [[Double]], _ b: [[Double]]) -> [[Double]] {
            (0..<3).map { (i: Int) -> [Double] in
                (0..<3).map { (j: Int) -> Double in
                    let x: Double = a[i][0] * b[0][j] + a[i][1] * b[1][j]
                    return x + a[i][2] * b[2][j]
                }
            }
        }
        let m = mul(mul(rz, ry), rx)
        return (0..<3).flatMap { c in (0..<3).map { r in Float(m[r][c]) } }
    }
}
