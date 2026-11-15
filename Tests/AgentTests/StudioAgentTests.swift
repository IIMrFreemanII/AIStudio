import Foundation
import Testing
@testable import Agent

@MainActor
struct StudioAgentTests {
    /// Lets queued main-actor work (a waiting `proceed`) run.
    func spin() async {
        for _ in 0..<20 { await Task.yield() }
        try? await Task.sleep(for: .milliseconds(20))
    }

    @Test func anActionGoesStraightThroughWhileRunning() async {
        let gate = StudioGate(pace: .fast)
        #expect(await gate.proceed("Typing") == .go(reread: false))
        #expect(gate.state == .running)
    }

    @Test func usingTheWindowPausesTheNextActionUntilResume() async {
        let gate = StudioGate(pace: .watchable)
        gate.pauseByUser()
        #expect(gate.state == .pausedByUser)
        var outcome: StudioGate.Outcome?
        let t = Task { @MainActor in outcome = await gate.proceed("Typing the prompt") }
        await spin()
        #expect(outcome == nil, "it waits while paused")
        gate.resume()
        await t.value
        #expect(outcome == .go(reread: true), "and then says to look at the page again")
        #expect(await gate.proceed("Next one") == .go(reread: false), "only once")
    }

    @Test func stepPaceWaitsForNextOrSkip() async {
        let gate = StudioGate(pace: .step)
        var first: StudioGate.Outcome?
        let t1 = Task { @MainActor in first = await gate.proceed("Opening Keyframes") }
        await spin()
        #expect(gate.state == .awaitingNext("Opening Keyframes"))
        #expect(first == nil)
        gate.next()
        await t1.value
        #expect(first == .go(reread: false))

        var second: StudioGate.Outcome?
        let t2 = Task { @MainActor in second = await gate.proceed("Deleting take 2") }
        await spin()
        gate.skip()
        await t2.value
        #expect(second == .skipped)
        #expect(gate.state == .running)
    }

    @Test func resetAndCancelLetWaitingActionsGoWithoutActing() async {
        let gate = StudioGate(pace: .step)
        var outcome: StudioGate.Outcome?
        let t = Task { @MainActor in outcome = await gate.proceed("Approving") }
        await spin()
        gate.reset()
        await t.value
        #expect(outcome == .skipped)
        #expect(gate.state == .running)

        gate.pauseByUser()
        var cancelled: StudioGate.Outcome?
        let t2 = Task { @MainActor in cancelled = await gate.proceed("Typing") }
        await spin()
        t2.cancel()
        await t2.value
        #expect(cancelled == .skipped, "a stopped reply's action never runs")
    }

    @Test func paceTimings() {
        #expect(StudioPace.fast.typingRate == nil)
        #expect(StudioPace.fast.typingSeconds(500) == 0)
        #expect(StudioPace.watchable.typingSeconds(60) == 1)
        #expect(StudioPace.watchable.typingSeconds(10_000) == 2.5, "long prompts type faster")
        #expect(StudioPace.fast.moveSeconds < StudioPace.watchable.moveSeconds)
        #expect(StudioPace.step.moveSeconds == StudioPace.watchable.moveSeconds)
    }

    @Test func exportsNeverOverwrite() {
        let folder = URL(fileURLWithPath: "/tmp/movies")
        let taken: Set<String> = ["/tmp/movies/The Last Light.mp4", "/tmp/movies/The Last Light 2.mp4"]
        let url = StudioFiles.unused("The Last Light", ext: "mp4", in: folder) { taken.contains($0.path) }
        #expect(url.lastPathComponent == "The Last Light 3.mp4")
        #expect(StudioFiles.unused("a/b", ext: "mp4", in: folder) { _ in false }.lastPathComponent == "a-b.mp4")
        #expect(StudioFiles.unused("  ", ext: "mp4", in: folder) { _ in false }.lastPathComponent == "Movie.mp4")
    }

    @Test func studioActionsAreFreeAndDeletesAlwaysAsk() {
        let rules = AllowRules(edits: true, runs: true)
        for mode in [PermissionMode.ask, .acceptEdits, .auto] {
            #expect(PermissionEngine.decide(effect: .studio, inProject: true, mode: mode, rules: rules) == .allow)
            #expect(PermissionEngine.decide(effect: .destructive, inProject: true, mode: mode, rules: rules) == .ask)
        }
        #expect(PermissionEngine.decide(effect: .destructive, inProject: true, mode: .bypass, rules: rules) == .allow)
        if case .deny = PermissionEngine.decide(effect: .studio, inProject: true, mode: .plan, rules: rules) {} else {
            Issue.record("plan mode refuses studio changes")
        }
        if case .deny = PermissionEngine.decide(effect: .destructive, inProject: true, mode: .plan, rules: rules) {} else {
            Issue.record("plan mode refuses deletes")
        }
    }
}
