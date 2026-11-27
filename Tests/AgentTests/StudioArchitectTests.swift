import Foundation
import LLMClient
import Testing
@testable import Agent

@MainActor
struct StudioGateQueueTests {
    func spin() async {
        for _ in 0..<20 { await Task.yield() }
        try? await Task.sleep(for: .milliseconds(20))
    }

    @Test func stepPaceQueuesActionsAndNextLetsTheFirstGo() async {
        let gate = StudioGate(pace: .step)
        var a: StudioGate.Outcome?, b: StudioGate.Outcome?
        let ta = Task { @MainActor in a = await gate.proceed("Typing shot 3's prompt", actor: "Worker 1") }
        await spin()
        let tb = Task { @MainActor in b = await gate.proceed("Looking at shot 9", actor: "Worker 2") }
        await spin()
        #expect(gate.state == .awaitingNext("Worker 1 · Typing shot 3's prompt"))
        #expect(gate.queue == ["Worker 1 · Typing shot 3's prompt", "Worker 2 · Looking at shot 9"])
        gate.next()
        await ta.value
        #expect(a == .go(reread: false) && b == nil, "only the first went")
        #expect(gate.state == .awaitingNext("Worker 2 · Looking at shot 9"))
        gate.skip()
        await tb.value
        #expect(b == .skipped && gate.state == .running && gate.queue.isEmpty)
    }

    @Test func resumeReleasesEveryPausedActorAndResetDropsTheQueue() async {
        let gate = StudioGate(pace: .watchable)
        gate.pauseByUser()
        var outs: [StudioGate.Outcome] = []
        let ts = (1...3).map { n in Task { @MainActor in outs.append(await gate.proceed("Act", actor: "Worker \(n)")) } }
        await spin()
        #expect(outs.isEmpty && gate.isPaused)
        gate.resume()
        for t in ts { await t.value }
        #expect(outs.count == 3 && outs.allSatisfy { if case .go = $0 { true } else { false } })

        gate.pace = .step
        var x: StudioGate.Outcome?, y: StudioGate.Outcome?
        let tx = Task { @MainActor in x = await gate.proceed("One") }
        let ty = Task { @MainActor in y = await gate.proceed("Two") }
        await spin()
        gate.reset()
        await tx.value
        await ty.value
        #expect(x == .skipped && y == .skipped && gate.queue.isEmpty && gate.state == .running)
    }

    @Test func aCancelledWaiterLeavesTheQueue() async {
        let gate = StudioGate(pace: .step)
        let t1 = Task { @MainActor in _ = await gate.proceed("First") }
        await spin()
        var second: StudioGate.Outcome?
        let t2 = Task { @MainActor in second = await gate.proceed("Second") }
        await spin()
        t1.cancel()
        await t1.value
        await spin()
        #expect(gate.state == .awaitingNext("Second"))
        gate.next()
        await t2.value
        #expect(second == .go(reread: false))
    }
}

struct StudioPlanTests {
    let raw: [Any] = [
        ["id": "keys", "title": "Keyframes for shots 1–4", "brief": "Write and run keyframes", "paid": ["keyframes"], "estimate": 0.12],
        ["id": "review", "title": "Review keyframes", "brief": "look_at each", "kind": "review", "depends_on": ["keys"]],
        ["title": "Music", "brief": "Write the cue prompts", "estimate": "$0.05"],
        ["title": "Voices", "brief": "Voice samples"],
    ]

    @Test func parsesTasksAndChecksThem() throws {
        let plan = try StudioPlan.parse(raw)
        #expect(plan.tasks.map(\.id) == ["keys", "review", "t3", "t4"])
        #expect(plan.tasks[1].kind == .review && plan.tasks[1].dependsOn == ["keys"])
        #expect(plan.tasks.map(\.worker) == [1, 2, 3, 1], "planned workers go round")
        #expect(abs(plan.estimate - 0.17) < 0.0001)
        #expect(throws: StudioPlan.Refusal.self) { try StudioPlan.parse([["title": "x"]]) }
        #expect(throws: StudioPlan.Refusal.self) { try StudioPlan.parse([["title": "x", "brief": "y", "depends_on": ["nope"]]]) }
        #expect(throws: StudioPlan.Refusal.self) { try StudioPlan.parse([]) }
    }

    @Test func tasksStartWhenReadyOnAFreeWorker() throws {
        var plan = try StudioPlan.parse(raw)
        #expect(throws: StudioPlan.Refusal.self) { try plan.start("review") }
        #expect(try plan.start("keys") == .worker(1))
        #expect(try plan.start("t3") == .worker(3))
        // t4's planned worker (1) is busy: it gets the free one.
        #expect(try plan.start("t4") == .worker(2))
        #expect(plan.running == 3 && !plan.hasFreeWorker)
        plan.finish("keys", ok: true, spent: 0.1)
        #expect(plan.blockers("review").isEmpty)
        #expect(try plan.start("review") == .worker(2) || plan.task("review")?.worker != nil)
        #expect(throws: StudioPlan.Refusal.self) { try plan.start("keys") }
    }

    @Test func twoRetriesThenItsTheArchitects() throws {
        var plan = try StudioPlan.parse(raw)
        #expect(throws: StudioPlan.Refusal.self) { try plan.start("keys", retry: true) }
        _ = try plan.start("keys")
        plan.finish("keys", ok: false)
        _ = try plan.start("keys", retry: true)
        plan.finish("keys", ok: false)
        _ = try plan.start("keys", retry: true)
        plan.finish("keys", ok: true)
        do {
            _ = try plan.start("keys", retry: true)
            Issue.record("a third retry is refused")
        } catch let r as StudioPlan.Refusal {
            #expect(r.description.contains("do it yourself"))
        }
        #expect(plan.task("keys")?.retries == 2)
    }
}

struct StudioMotionTests {
    @Test func motionsFollowStateAndPace() {
        let w = StudioMotion.motions(state: .working, pace: .watchable, reduceMotion: false)
        #expect(w.contains([.twinkle, .glide, .curve, .drift, .handoff]))
        let f = StudioMotion.motions(state: .working, pace: .fast, reduceMotion: false)
        #expect(!f.contains(.twinkle) && !f.contains(.curve) && !f.contains(.handoff) && !f.contains(.drift) && f.contains(.glide))
        #expect(StudioMotion.motions(state: .waiting, pace: .fast, reduceMotion: false).contains(.pulse))
        #expect(StudioMotion.motions(state: .done, pace: .fast, reduceMotion: false).contains(.burst))
        #expect(StudioMotion.motions(state: .failed, pace: .step, reduceMotion: false).contains(.shake))
        #expect(StudioMotion.motions(state: .paused, pace: .watchable, reduceMotion: false) == .dim)
        #expect(StudioMotion.motions(state: .idle, pace: .watchable, reduceMotion: false).contains(.wander))
    }

    @Test func reduceMotionTurnsEveryMotionOff() {
        for state in StudioActorState.allCases {
            for pace in StudioPace.allCases {
                let m = StudioMotion.motions(state: state, pace: pace, reduceMotion: true)
                #expect(m.isDisjoint(with: [.twinkle, .pulse, .burst, .shake, .drift, .glide, .curve, .handoff, .wander]))
            }
        }
        #expect(StudioMotion.motions(state: .waiting, pace: .watchable, reduceMotion: true) == .ring)
    }
}

struct StudioFlightTests {
    @Test func aFlightCurvesFromStartToEnd() {
        let f = StudioFlight(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 1000, y: 0), pace: .watchable)
        #expect(f.point(at: 0) == CGPoint(x: 0, y: 0))
        #expect(f.point(at: f.duration) == CGPoint(x: 1000, y: 0))
        #expect(abs(f.control.y - 150) < 0.001, "bent sideways by 15% of the distance")
        #expect(abs(f.point(at: f.duration / 2).y) > 50)
        let straight = StudioFlight(from: .zero, to: CGPoint(x: 1000, y: 0), pace: .fast)
        #expect(straight.point(at: straight.duration / 2).y == 0 && straight.duration == 0.15)
    }

    @Test func longerFlightsTakeLongerWithinLimits() {
        let short = StudioFlight.duration(distance: 50, pace: .watchable)
        let long = StudioFlight.duration(distance: 3000, pace: .watchable)
        #expect(short >= 0.35 && short < long && long <= 0.9)
        #expect(StudioFlight.duration(distance: 0, pace: .watchable) == 0)
    }

    @Test func driftStaysInItsLoopAndActorsAreOutOfPhase() {
        for t in stride(from: 0.0, to: 20, by: 0.37) {
            let d = StudioFlight.drift(actor: 2, time: t)
            #expect(abs(d.dx) <= 9.0001 && abs(d.dy) <= 6.0001)
        }
        #expect(StudioFlight.drift(actor: 1, time: 3) != StudioFlight.drift(actor: 2, time: 3))
    }

    @Test func wanderingStaysOnScreenAndIsRepeatable() {
        let screen = CGRect(x: 0, y: 25, width: 1440, height: 850)
        var a = StudioRandom(seed: 7), b = StudioRandom(seed: 7)
        for _ in 0..<50 {
            let p = StudioFlight.wanderSpot(in: screen, using: &a)
            #expect(screen.insetBy(dx: 79, dy: 79).contains(p))
            #expect(p == StudioFlight.wanderSpot(in: screen, using: &b))
            let d = StudioFlight.wanderDelay(using: &a)
            #expect(d >= 10 && d <= 25 && d == StudioFlight.wanderDelay(using: &b))
        }
    }

    @Test func ghostEchoesFollowTheFlightAndPullIn() {
        let f = StudioFlight(from: .zero, to: CGPoint(x: 1000, y: 0), pace: .watchable)
        let leg = StudioTrailLeg(flight: f, start: 100)
        // Mid-flight: five echoes, back to front, on the path 0.3 s back at most, smaller and fainter towards the back.
        let mid = 100 + f.duration * 0.7
        let e = StudioTrail.echoes([leg], now: mid)
        #expect(e.count == 5)
        #expect(e.first?.band == 1 && e.last?.band == 0)
        let span = StudioTrail.span(f, at: mid - 100)
        #expect(span > 0.05 && span < 0.3, "fast here: the trail is shortened")
        for (i, echo) in e.enumerated() {
            let lag = span * Double(5 - i) / 5
            let on = f.point(at: mid - 100 - lag)
            #expect(abs(echo.point.x - on.x) < 0.001 && abs(echo.point.y - on.y) < 0.001)
            if i > 0 { #expect(echo.scale > e[i - 1].scale && echo.opacity > e[i - 1].opacity) }
        }
        // At most 150 points of path at full speed (a little more along the curve than straight).
        let mid2 = 100 + f.duration / 2
        let fastest = StudioTrail.echoes([leg], now: mid2)
        #expect(StudioFlight.distance(fastest[0].point, f.point(at: f.duration / 2)) < 155)
        // Slow, near the end: the whole 0.3 s.
        #expect(StudioTrail.span(f, at: f.duration) == 0.3)
        #expect(abs(e[4].scale - 0.93) < 0.001 && abs(e[0].scale - 0.65) < 0.001)
        // Just after take-off: echoes from before the flight aren't there.
        #expect((1..<5).contains(StudioTrail.echoes([leg], now: 100.07).count))
        #expect(StudioTrail.echoes([leg], now: 99).isEmpty)
        // After landing they slide into the landing spot and fade, then are gone.
        let landed = StudioTrail.echoes([leg], now: leg.end + 0.3)
        #expect(landed.count == 5 && landed.allSatisfy { $0.point.x > 980 && $0.opacity < 0.1 })
        #expect(StudioTrail.echoes([leg], now: leg.end + 0.36).isEmpty)
        #expect(StudioTrail.isActive([leg], now: leg.end + 0.34) && !StudioTrail.isActive([leg], now: leg.end + 0.36))
        // The drift after landing pulls them along.
        let drifted = StudioTrail.echoes([leg], now: leg.end + 0.3, offset: CGVector(dx: 0, dy: 8))
        #expect(drifted.allSatisfy { $0.point.y > 6 })
    }

    @Test func fastFlightsLeaveAShortStreakAndReplacedFlightsKeepTheirPath() {
        #expect(StudioTrail.span(duration: 0.15) == 0.12 && StudioTrail.span(duration: 0.5) == 0.3)
        let fast = StudioFlight(from: .zero, to: CGPoint(x: 300, y: 0), pace: .fast)
        let streak = StudioTrail.echoes([StudioTrailLeg(flight: fast, start: 0)], now: 0.14)
        #expect(streak.count == 5 && streak[0].point.x > 0)
        // A flight replaced mid-air: the back echoes are still on the old path (y = 0), the front ones on the new one.
        let old = StudioTrailLeg(flight: StudioFlight(from: .zero, to: CGPoint(x: 1000, y: 0), bend: 0, duration: 0.8), start: 0)
        let new = StudioTrailLeg(flight: StudioFlight(from: CGPoint(x: 500, y: 0), to: CGPoint(x: 500, y: 800), bend: 0, duration: 0.8),
                                 start: 0.4)
        let e = StudioTrail.echoes([old, new], now: 0.47)
        #expect(e.count == 5 && e[0].point.y == 0 && e[4].point.y > 0)
        // An older leg that had landed long before leaves nothing behind the next start.
        let rested = StudioTrailLeg(flight: new.flight, start: 5)
        #expect(StudioTrail.echoes([old, rested], now: 5.13).count == 2)
    }
}

/// Two `delegate` calls in one message run at the same time, like `task`.
private struct SleepyDelegate: AgentTool {
    let name = "delegate"
    let description = "test"
    var parameters: JSONValue { .object(["type": .string("object")]) }
    let effect = ToolEffect.read
    func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview { ToolPreview(kind: .read, title: "delegate") }
    func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        try await Task.sleep(for: .milliseconds(300))
        return ToolOutput("done")
    }
}

private final class DelegatingModel: ChatStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var calls = 0
    func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        let first = lock.withLock { () -> Bool in calls += 1; return calls == 1 }
        return AsyncThrowingStream { c in
            if first {
                c.yield(.toolCall(index: 0, id: "d1", name: "delegate", arguments: "{\"task\":\"a\"}"))
                c.yield(.toolCall(index: 1, id: "d2", name: "delegate", arguments: "{\"task\":\"b\"}"))
            } else {
                c.yield(.content("all done"))
            }
            c.finish()
        }
    }
}

struct DelegateBatchTests {
    @Test func delegateCallsRunInParallel() async throws {
        let model = DelegatingModel()
        var config = AgentConfig(model: "m", tools: [SleepyDelegate()])
        config.mode = .ask
        let start = Date()
        _ = try await AgentLoop.run(client: model, messages: [.user("go")], config: config,
                                    context: ToolContext(root: FileManager.default.temporaryDirectory), ask: { _ in .allow },
                                    events: { _ in })
        #expect(Date().timeIntervalSince(start) < 0.55, "the two 0.3 s workers overlapped")
    }
}

struct StudioFaceTests {
    @Test func eyesFollowTheMood() {
        let shapes = StudioActorState.allCases.map { StudioFace.face(state: $0, pace: .watchable, reduceMotion: false).eyes }
        #expect(shapes == [.open, .focus, .wide, .closed, .happy, .cross])
        #expect(StudioFace.face(state: .working, talking: true, pace: .watchable, reduceMotion: false).eyes == .open)
        let w = StudioFace.face(state: .working, talking: true, pace: .watchable, reduceMotion: false)
        #expect(w.blinks && w.bounce && w.particles && w.orbit && w.breathe && !w.glances && w.animates)
        let idle = StudioFace.face(state: .idle, pace: .step, reduceMotion: false)
        #expect(idle.blinks && idle.glances && idle.particles && idle.animates, "an idle orb blinks, glances and has its sparkles")
        for state in StudioActorState.allCases {
            #expect(StudioFace.face(state: state, pace: .watchable, reduceMotion: false).particles, "sparkles all the time")
        }
    }

    @Test func fastAndReduceMotionKeepMoodsOnly() {
        for state in StudioActorState.allCases {
            let fast = StudioFace.face(state: state, talking: true, moving: true, pace: .fast, reduceMotion: false)
            #expect(!fast.blinks && !fast.glances && !fast.bounce && !fast.orbit && fast.particles, "Fast keeps the sparkles")
            let still = StudioFace.face(state: state, talking: true, moving: true, pace: .watchable, reduceMotion: true)
            #expect(!still.blinks && !still.glances && !still.bounce && !still.particles && !still.orbit && !still.breathe && !still.animates)
            #expect(still.eyes == StudioFace.face(state: state, pace: .watchable, reduceMotion: false).eyes || state == .working)
        }
    }

    @Test func blinksAreRepeatableAndSpaced() {
        let a = StudioFace.blinkTimes(seed: 3), b = StudioFace.blinkTimes(seed: 3)
        #expect(a == b && a.count > 15)
        for (x, y) in zip(a, a.dropFirst()) { #expect(y - x >= 3 && y - x <= 6) }
        #expect(StudioFace.isBlinking(at: a[2] + 0.05, seed: 3) && !StudioFace.isBlinking(at: a[2] + 0.5, seed: 3))
        #expect(StudioFace.isBlinking(at: a[2] + 0.05 + 120, seed: 3), "the schedule repeats every cycle")
        let marks = StudioFace.changes(after: 10, span: 30, seed: 3, blinks: true, glances: true)
        #expect(!marks.isEmpty && marks.allSatisfy { $0 > 10 && $0 <= 40 } && marks == marks.sorted())
    }

    @Test func gazePointsAlongTheWayWithinItsLimit() {
        let g = StudioFace.gaze(from: .zero, to: CGPoint(x: 300, y: 0), maxShift: 1)
        #expect(abs(g.dx - 1) < 0.0001 && g.dy == 0)
        let near = StudioFace.gaze(from: .zero, to: CGPoint(x: 0, y: -10), maxShift: 1)
        #expect(near.dy < 0 && hypot(near.dx, near.dy) < 1, "a near target: a smaller look")
        #expect(StudioFace.gaze(from: .zero, to: .zero) == .zero)
    }

    @Test func particlesStayInTheirRing() {
        for t in stride(from: 0.0, to: 30, by: 0.7) {
            for p in StudioFace.particles(actor: 2, time: t) {
                let r = hypot(p.offset.dx, p.offset.dy)
                #expect(r >= 0.59 && r <= 0.951 && p.opacity >= 0.24 && p.opacity <= 1.0001)
            }
        }
    }
}

struct StudioOrbPoseTests {
    func face(_ s: StudioActorState) -> StudioFace { StudioFace.face(state: s, pace: .watchable, reduceMotion: false) }

    @Test func eyesFollowTheMood() {
        let open = StudioOrbPose.pose(face: face(.idle))
        #expect(open.eyeShape == .pill && open.eyeWidth == 0.26 && open.eyeHeight == 0.66 && open.eyeGap == 0.42)
        #expect(StudioOrbPose.pose(face: face(.working)).eyeHeight < open.eyeHeight, "focus: narrower")
        let wide = StudioOrbPose.pose(face: face(.waiting))
        #expect(wide.eyeHeight > open.eyeHeight && wide.eyeWidth > open.eyeWidth)
        let closed = StudioOrbPose.pose(face: face(.paused), paused: true)
        #expect(closed.eyeHeight < 0.1 && closed.eyeWidth > open.eyeWidth && closed.glow == 0.35)
        #expect(StudioOrbPose.pose(face: face(.done)).eyeShape == .happy && StudioOrbPose.pose(face: face(.done)).eyeGap == 0.54)
        #expect(StudioOrbPose.pose(face: face(.failed)).eyeShape == .cross)
    }

    @Test func blinkShutsPillEyesOnly() {
        let b = StudioOrbPose.pose(face: face(.idle), blinking: true)
        #expect(b.eyeHeight < 0.1 && b.eyeWidth > 0.26)
        #expect(StudioOrbPose.pose(face: face(.done), blinking: true).eyeHeight == 0.66, "happy arcs don't blink")
    }

    @Test func smallOrbsKeepVisibleEyes() {
        let p = StudioOrbPose.pose(face: face(.paused), radius: 8)
        #expect(p.eyeHeight * 8 >= 0.999 && p.eyeWidth * 8 >= 1.599)
        #expect(StudioOrbPose.pose(face: face(.idle), radius: 8).glow == 0.75)
    }

    @Test func gazeTurnsTheSphere() {
        let still = StudioOrbPose.pose(face: face(.idle))
        #expect(still.rotation == [1, 0, 0, 0, 1, 0, 0, 0, 1])
        // Looking right and up: the front (+z in the body) turns toward +x and +y on screen.
        let p = StudioOrbPose.pose(face: face(.idle), gaze: CGVector(dx: 1, dy: -1))
        let m = p.rotation
        let front = (m[6], m[7], m[8])
        #expect(front.0 > 0.2 && front.1 > 0.2 && front.2 > 0.5)
        // Orthonormal columns.
        for c in 0..<3 {
            let col = (m[c * 3], m[c * 3 + 1], m[c * 3 + 2])
            #expect(abs(col.0 * col.0 + col.1 * col.1 + col.2 * col.2 - 1) < 1e-5)
        }
        // A long gaze is clamped to 1; turn 0 keeps it facing front.
        let far = StudioOrbPose.pose(face: face(.idle), gaze: CGVector(dx: 5, dy: 0))
        #expect(abs(far.yaw - StudioOrbPose.maxYaw * StudioOrbPose.standardTurn) < 1e-9)
        #expect(StudioOrbPose.pose(face: face(.idle), gaze: CGVector(dx: 1, dy: 0), turn: 0).yaw == 0)
    }

    @Test func talkingLiftsTheEyes() {
        #expect(StudioOrbPose.pose(face: face(.working), bounce: -1).eyeLift > 0)
        #expect(StudioOrbPose.pose(face: face(.working), bounce: 1).eyeLift < 0)
    }
}
