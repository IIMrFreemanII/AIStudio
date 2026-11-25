import AIStudioProtocol
import Foundation
import Testing
@testable import LLMClient

/// Serves canned answers in turn: a body to stream, or an error to fail with (after the status line, if any).
final class StubProtocol: URLProtocol, @unchecked Sendable {
    enum Answer { case body(String), dropAfter(String, URLError.Code), http(Int) }
    nonisolated(unsafe) static var answers: [Answer] = []
    nonisolated(unsafe) static var requests = 0
    static let lock = NSLock()

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func stopLoading() {}

    override func startLoading() {
        let answer: Answer = Self.lock.withLock {
            Self.requests += 1
            return Self.answers.isEmpty ? .body("data: [DONE]\n\n") : Self.answers.removeFirst()
        }
        let url = request.url!
        func respond(_ code: Int) {
            let r = HTTPURLResponse(url: url, statusCode: code, httpVersion: "HTTP/1.1",
                                    headerFields: ["Content-Type": "text/event-stream"])!
            client?.urlProtocol(self, didReceive: r, cacheStoragePolicy: .notAllowed)
        }
        switch answer {
        case .body(let s):
            respond(200)
            client?.urlProtocol(self, didLoad: Data(s.utf8))
            client?.urlProtocolDidFinishLoading(self)
        case .dropAfter(let s, let code):
            respond(200)
            client?.urlProtocol(self, didLoad: Data(s.utf8))
            // Later, as a real connection drops: data and failure together would lose the data.
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.05) { [self] in
                client?.urlProtocol(self, didFailWithError: URLError(code))
            }
        case .http(let code):
            respond(code)
            client?.urlProtocol(self, didLoad: Data("{\"error\": \"bad gateway\"}".utf8))
            client?.urlProtocolDidFinishLoading(self)
        }
    }
}

@Suite(.serialized)
struct StreamRetryTests {
    static let status = "event: aistudio.status\ndata: {\"state\": \"starting\", \"phase\": \"Waiting for a GPU\"}\n\n"
    static let reply = "data: {\"choices\": [{\"delta\": {\"content\": \"ok\"}}]}\n\ndata: [DONE]\n\n"

    func client() -> OpenAIChatClient {
        let cfg = URLSessionConfiguration.ephemeral
        cfg.protocolClasses = [StubProtocol.self]
        var c = OpenAIChatClient(http: HTTPService(baseURL: URL(string: "https://gpu.test")!, token: "t",
                                                   session: URLSession(configuration: cfg)))
        c.retryDelays = [0.01, 0.01]
        return c
    }

    func run(_ answers: [StubProtocol.Answer]) async -> ([ChatEvent], Error?) {
        StubProtocol.lock.withLock {
            StubProtocol.answers = answers
            StubProtocol.requests = 0
        }
        var events: [ChatEvent] = []
        do {
            for try await e in client().stream(ChatRequest(model: "m", messages: [])) { events.append(e) }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    @Test func aConnectionLostWhileStartingIsRetried() async {
        let (events, error) = await run([.dropAfter(Self.status, .networkConnectionLost), .http(504), .body(Self.reply)])
        #expect(error == nil)
        #expect(StubProtocol.requests == 3)
        let phases = events.compactMap { if case .status(let s) = $0 { s.phase } else { nil } }
        #expect(phases == ["Waiting for a GPU", "Reconnecting", "Reconnecting"])
        #expect(events.last == .content("ok"))
    }

    @Test func noRetryOnceTheReplyBegan() async {
        let partial = "data: {\"choices\": [{\"delta\": {\"content\": \"o\"}}]}\n\n"
        let (events, error) = await run([.dropAfter(partial, .networkConnectionLost), .body(Self.reply)])
        #expect((error as? URLError)?.code == .networkConnectionLost)
        #expect(StubProtocol.requests == 1 && events == [.content("o")])
    }

    @Test func aStreamClosedBeforeDoneIsADroppedConnection() async {
        // Closed cleanly while the model was starting: sent again, not taken as an empty reply.
        let (events, error) = await run([.body(Self.status), .body(Self.reply)])
        #expect(error == nil && StubProtocol.requests == 2)
        #expect(events.last == .content("ok"))
        // Closed mid-reply: an error, not a reply cut short without a word.
        let partial = "data: {\"choices\": [{\"delta\": {\"content\": \"o\"}}]}\n\n"
        let (cut, cutError) = await run([.body(partial), .body(Self.reply)])
        #expect((cutError as? URLError)?.code == .networkConnectionLost && cut == [.content("o")])
        // A server that sends its finish reason and closes without `[DONE]` ended the reply well.
        let finished = "data: {\"choices\": [{\"delta\": {\"content\": \"ok\"}, \"finish_reason\": \"stop\"}]}\n\n"
        let (done, doneError) = await run([.body(finished)])
        #expect(doneError == nil && StubProtocol.requests == 1 && done.contains(.content("ok")))
    }

    @Test func givesUpAfterItsRetriesAndOnOtherErrors() async {
        let (_, error) = await run([.dropAfter("", .timedOut), .dropAfter("", .timedOut), .dropAfter("", .timedOut)])
        #expect((error as? URLError)?.code == .timedOut && StubProtocol.requests == 3)
        let (_, refused) = await run([.http(401), .body(Self.reply)])
        #expect(refused != nil && StubProtocol.requests == 1)
    }
}
