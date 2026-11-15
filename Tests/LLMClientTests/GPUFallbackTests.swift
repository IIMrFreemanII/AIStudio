import AIStudioProtocol
import Foundation
import Testing
@testable import LLMClient

/// Counts what the stream asked of its GPU-fallback hooks.
final class FallbackLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _switches = 0, _settles = 0
    var switches: Int { lock.withLock { _switches } }
    var settles: Int { lock.withLock { _settles } }
    func switched() { lock.withLock { _switches += 1 } }
    func settled() { lock.withLock { _settles += 1 } }
}

struct GPUFallbackTests {
    func model(gpu: String = "B200", count: Int = 1, load: String? = "processed", snapshot: Bool? = true) -> LLMModel {
        LLMModel(id: "q", name: "Q", url: nil, repo: "Qwen/Q", gpu: gpu, count: count, context: 32768, nativeContext: nil,
                 vision: true, weightBytes: nil, toolParser: "qwen3_coder", reasoningParser: "qwen3", engineArgs: [],
                 installed: true, status: nil, engine: "vllm", source: nil, loadFormat: load, snapshot: snapshot)
    }

    @Test func triesTheListInOrderSkippingWhatWasTried() {
        let b200 = GPUPreset(gpu: "B200", count: 1), h200x2 = GPUPreset(gpu: "H200", count: 2), h100x4 = GPUPreset(gpu: "H100", count: 4)
        let list = [b200, h200x2, h100x4]
        #expect(GPUFallback.next(in: list, current: b200, tried: [b200]) == h200x2)
        #expect(GPUFallback.next(in: list, current: h200x2, tried: [b200, h200x2]) == h100x4)
        #expect(GPUFallback.next(in: list, current: h100x4, tried: [b200, h200x2, h100x4]) == nil, "the list ran out: keep waiting")
        #expect(GPUFallback.next(in: [], current: b200, tried: []) == nil)
    }

    @Test func multiGPUPresetsTurnOffFastStartAndSnapshotsAndTheyComeBack() {
        let m = model()
        let two = GPUFallback.changes(to: GPUPreset(gpu: "H200", count: 2), from: m)
        #expect(two["gpu"] == .string("H200") && two["count"] == .number(2))
        #expect(two["loadFormat"] == .string("auto") && two["snapshot"] == .bool(false))
        let one = GPUFallback.changes(to: GPUPreset(gpu: "H100", count: 1), from: m)
        #expect(one["loadFormat"] == nil && one["snapshot"] == nil, "one GPU keeps them")
        let back = GPUFallback.restore(m)
        #expect(back == ["gpu": .string("B200"), "count": .number(1), "loadFormat": .string("processed"), "snapshot": .bool(true)])
    }

    @Test func waitingIsOverdueAfterTheLimit() {
        let t = Date()
        #expect(!GPUFallback.overdue(since: nil, now: t, limit: 30))
        #expect(!GPUFallback.overdue(since: t, now: t.addingTimeInterval(29), limit: 30))
        #expect(GPUFallback.overdue(since: t, now: t.addingTimeInterval(30), limit: 30))
    }

    @Test func listsAreSavedAsJSON() {
        let lists = ["gpu1|q": [GPUPreset(gpu: "H200", count: 2)]]
        #expect(GPUFallback.decode(GPUFallback.encode(lists)) == lists)
        #expect(GPUFallback.decode(nil).isEmpty && GPUFallback.decode("junk").isEmpty)
        #expect(GPUPreset(gpu: "H200", count: 2).label == "2× H200" && GPUPreset(gpu: "B200", count: 1).label == "B200")
    }
}

/// The stream's side (in the retry suite: they share the stub server).
extension StreamRetryTests {
    func run(_ answers: [StubProtocol.Answer], policy: GPUWaitPolicy) async -> ([ChatEvent], Error?) {
        StubProtocol.lock.withLock {
            StubProtocol.answers = answers
            StubProtocol.requests = 0
        }
        var c = client()
        c.gpuWait = policy
        var events: [ChatEvent] = []
        do {
            for try await e in c.stream(ChatRequest(model: "m", messages: [])) { events.append(e) }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    @Test func aStartWaitingTooLongMovesToTheNextGPUAndSendsAgain() async throws {
        let log = FallbackLog()
        let policy = GPUWaitPolicy(limit: 0, switchGPU: { _ in log.switched(); return "2× H200" }, settle: { _ in log.settled() })
        let (events, error) = await run([.body(Self.status), .body(Self.reply)], policy: policy)
        #expect(error == nil && StubProtocol.requests == 2)
        #expect(log.switches == 1)
        let phases = events.compactMap { if case .status(let s) = $0 { s.phase } else { nil } }
        #expect(phases.contains("Switching GPUs") && phases.contains { $0.contains("waiting for 2× H200") })
        #expect(events.last == .content("ok"))
        try await Task.sleep(for: .milliseconds(100))
        #expect(log.settles == 1, "once the reply began, the model's own GPU goes back")
    }

    @Test func withNothingLeftToTryItKeepsWaiting() async throws {
        let log = FallbackLog()
        let policy = GPUWaitPolicy(limit: 0, switchGPU: { _ in log.switched(); return nil }, settle: { _ in log.settled() })
        let (events, error) = await run([.body(Self.status), .body(Self.status + Self.reply)], policy: policy)
        #expect(error == nil && events.last == .content("ok"))
        #expect(log.switches == 1, "asked once, then it just waits")
        try await Task.sleep(for: .milliseconds(100))
        #expect(log.settles == 0, "nothing was moved")
    }

    @Test func aStartStoppedByAnotherChatsSwitchIsSentAgain() async {
        let stopped = "event: aistudio.error\ndata: {\"error\": \"the model was stopped before it was ready\"}\n\n"
        let policy = GPUWaitPolicy(limit: 60, switchGPU: { _ in nil }, settle: { _ in }, switchedRecently: { _ in true })
        let (events, error) = await run([.body(Self.status + stopped), .body(Self.reply)], policy: policy)
        #expect(error == nil && StubProtocol.requests == 2 && events.last == .content("ok"))
        let (_, plain) = await run([.body(Self.status + stopped)], policy: GPUWaitPolicy(limit: 60, switchGPU: { _ in nil }, settle: { _ in }))
        #expect(plain != nil, "without a switch, a stopped start is an error")
    }
}
