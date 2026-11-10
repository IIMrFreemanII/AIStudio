import Foundation
import GraphRuntime
import Testing
@testable import AIStudioProtocol

func hexData(_ s: String) -> Data {
    var d = Data()
    var i = s.startIndex
    while i < s.endIndex {
        let j = s.index(i, offsetBy: 2)
        d.append(UInt8(s[i..<j], radix: 16)!)
        i = j
    }
    return d
}

func hexString(_ d: Data) -> String { d.map { String(format: "%02x", $0) }.joined() }

struct WireEventTests {
    /// Same bytes as web/dashboard/src/lib/protocol.test.ts `GOLDEN`.
    static let golden = "0500030007000000" + "2a00000005000800" + "10000000" + "000000000000f83f" + "0004000000000000"
        + "0a00010007000000" + "0000000002000800" + "02000000" + "ffd8"

    @Test func encodesTheGoldenFrame() {
        var frame = Data()
        WireEvent(kind: .nodeFinished, node: 3, run: 7, seq: 42, step: 5, total: 8,
                  payload: WireEvent.nodeFinishedPayload(seconds: 1.5, peakMemory: 1024)).encode(into: &frame)
        WireEvent(kind: .preview, node: 1, run: 7, step: 2, total: 8, payload: Data([0xff, 0xd8])).encode(into: &frame)
        #expect(hexString(frame) == Self.golden)
    }

    @Test func decodesTheGoldenFrame() throws {
        let events = try WireEvent.decodeAll(hexData(Self.golden))
        #expect(events.count == 2)
        #expect(events[0].kind == .nodeFinished && events[0].node == 3 && events[0].run == 7 && events[0].seq == 42)
        #expect(events[0].step == 5 && events[0].total == 8)
        #expect(events[0].seconds == 1.5 && events[0].peakMemory == 1024)
        #expect(events[1].kind == .preview && events[1].payload == Data([0xff, 0xd8]))
    }

    /// A status record, as the GPU server (server/gpu/tests/test_wire.py) and the dashboard test encode it.
    static let goldenStatus = "0c00ffff09000000" + "0100000000000000" + "11000000"
        + "5374617274696e672047505520303a3132"

    @Test func statusRecordsMatchTheOtherEncoders() throws {
        let e = WireEvent(kind: .nodeStatus, run: 9, seq: 1, payload: Data("Starting GPU 0:12".utf8))
        #expect(hexString(e.encoded()) == Self.goldenStatus)
        let decoded = try WireEvent.decodeAll(hexData(Self.goldenStatus))
        #expect(decoded.count == 1 && decoded[0].kind == .nodeStatus && decoded[0].message == "Starting GPU 0:12")
    }

    @Test func unknownKindsAreSkipped() throws {
        // A record of a kind this build doesn't know (99) between two it does.
        var frame = WireEvent(kind: .started, run: 1).encoded()
        frame.append(hexData("6300ffff01000000" + "0000000000000000" + "02000000" + "abcd"))
        frame.append(WireEvent(kind: .finished, run: 1, payload: WireEvent.float64(2)).encoded())
        #expect(try WireEvent.decodeAll(frame).map(\.kind) == [.started, .finished])
    }

    @Test func headerIsTwentyBytes() {
        #expect(WireEvent(kind: .started, run: 1).encoded().count == WireEvent.headerSize)
        #expect(WireEvent.headerSize == 20)
    }

    @Test func rejectsTruncatedFramesAndSkipsUnknownKinds() throws {
        let full = hexData(Self.golden)
        #expect(throws: WireError.truncated) { try WireEvent.decodeAll(full.prefix(15)) }
        #expect(throws: WireError.truncated) { try WireEvent.decodeAll(full.dropLast()) }
        var unknown = WireEvent(kind: .started, run: 9).encoded()
        unknown[0] = 99
        let events = try WireEvent.decodeAll(unknown + WireEvent(kind: .finished, run: 9).encoded())
        #expect(events.map(\.kind) == [.finished])
    }

    @Test func unknownPeakIsNil() {
        let e = WireEvent(kind: .nodeFinished, run: 1, payload: WireEvent.nodeFinishedPayload(seconds: 2, peakMemory: nil))
        #expect(e.peakMemory == nil && e.seconds == 2)
    }

    @Test func controlMessagesRoundTrip() throws {
        let g = Graph(nodes: [GraphNode(id: "a", type: "Const", inputs: ["value": .value(.int(1))])])
        let messages: [ClientMessage] = [
            .hello(ClientHello(role: .app, name: "x", previewHz: 2, previewMaxSide: 256)),
            .submit(SubmitRequest(requestID: "r", graph: g, metadataGraph: g, workflow: "w", files: ["f"])),
            .cancel(run: 3), .subscribe(run: 3, lastSeq: 17), .setPreview(hz: 1, maxSide: 128), .ping(id: 5),
            .text(TextRequest(requestID: "t", model: "m", system: "s", user: "u", images: [], maxTokens: 10, temperature: 0.5)),
            .cancelText(requestID: "t"),
        ]
        for m in messages { #expect(try ClientMessage.decode(m.jsonString()) == m) }
        let replies: [ServerMessage] = [
            .welcome(protocolVersion: 1, serverVersion: "v", session: "s"), .submitted(requestID: "r", run: 2, nodes: ["a"]),
            .rejected(requestID: "r", message: "no"), .pong(id: 1), .textDelta(requestID: "t", text: "ab"),
            .textDone(requestID: "t", text: nil, error: "e"), .error(message: "x"),
            .log(LogLine(time: 1, level: .warning, message: "m")),
        ]
        for m in replies { #expect(try ServerMessage.decode(m.jsonString()) == m) }
        // The `type` key is what the dashboard switches on.
        #expect(try ClientMessage.ping(id: 1).jsonString().contains("\"type\":\"ping\""))
    }
}

struct MetricsTests {
    @Test func ringWrapsAndStaysChronological() {
        var ring = MetricsRing(capacity: 3, metrics: 2)
        for t in 1...5 { ring.push(time: Double(t), [Float(t), Float(t * 10)]) }
        let (times, cols) = ring.chronological()
        #expect(times == [3, 4, 5])
        #expect(cols[1] == [30, 40, 50])
        #expect(ring.latest == [5, 50])
    }

    @Test func historyAndSamplePayloadsRoundTrip() throws {
        var ring = MetricsRing(capacity: 4, metrics: MetricID.count)
        for t in 0..<6 { ring.push(time: 100 + Double(t), (0..<MetricID.count).map { Float($0 + t) }) }
        let decoded = try #require(MetricsCodec.decodeHistory(MetricsCodec.history(ring)))
        #expect(decoded.times == [102, 103, 104, 105])
        #expect(decoded.columns[3] == [5, 6, 7, 8])
        let sample = try #require(MetricsCodec.decodeSample(MetricsCodec.sample(time: 7.5, [1, 2, 3])))
        #expect(sample.time == 7.5 && sample.values == [1, 2, 3])
    }

    @Test func latencyPercentiles() {
        var w = LatencyWindow(capacity: 200)
        for i in 1...100 { w.add(Float(i)) }
        let (p50, p99) = w.percentiles()
        #expect(abs(p50 - 50.5) <= 1 && p99 >= 98)
        w.reset()
        #expect(w.percentiles() == (0, 0))
    }
}

struct CoalescerTests {
    @Test func thousandEventsMakeOneDelivery() {
        var scheduled: [() -> Void] = []
        var deliveries: [[Int]] = []
        let c = Coalescer<Int>(schedule: { _, run in scheduled.append(run) }, deliver: { deliveries.append($0) })
        for i in 0..<1000 { c.push(i) }
        #expect(scheduled.count == 1)
        scheduled.removeFirst()()
        #expect(deliveries.count == 1 && deliveries[0].count == 1000 && deliveries[0].last == 999)
        c.push(1)
        #expect(scheduled.count == 1)
    }

    @Test func mergeFoldsRepeatedProgress() {
        var scheduled: [() -> Void] = []
        var delivered: [(String, Int)] = []
        let c = Coalescer<(String, Int)>(schedule: { _, run in scheduled.append(run) },
                                         merge: { last, new in last.0 == new.0 ? new : nil },
                                         deliver: { delivered += $0 })
        for step in 1...50 { c.push(("sampler", step)) }
        c.push(("decode", 1))
        c.push(("sampler", 51))
        c.drain()
        #expect(delivered.map(\.0) == ["sampler", "decode", "sampler"])
        #expect(delivered.map(\.1) == [50, 1, 51])
    }
}
