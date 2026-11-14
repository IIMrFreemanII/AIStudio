import AIStudioProtocol
import Foundation

/// Streams chat completions from an OpenAI-compatible endpoint: the GPU server's `v1/llm/oai/v1` (with its
/// `aistudio.*` events for cold starts, usage and cost), or any other.
public struct OpenAIChatClient: Sendable {
    public let http: HTTPService
    /// The path of the OpenAI API under the server's URL.
    public let apiPath: String

    public init(http: HTTPService, apiPath: String = "v1/llm/oai/v1") {
        self.http = http
        self.apiPath = apiPath
    }

    /// Seconds to wait before each new attempt after the connection dropped (see `stream`); its count is the number of
    /// retries.
    public var retryDelays: [Double] = [1, 2, 4, 8, 15]

    /// The reply as events. Cancel the consuming task (or stop iterating) to stop generation.
    ///
    /// A connection lost before the reply began (a long cold start, a network blip, a gateway error) is retried with the
    /// same request: nothing was shown or run yet, and the server is still starting the model, so the reply would
    /// otherwise be lost and the GPU would start for nothing. A `Reconnecting` status shows meanwhile.
    public func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                var lastStatus: EngineStatus?
                var attempt = 0
                while true {
                    var began = false
                    do {
                        try await once(request) { e in
                            switch e {
                            case .status(let s): lastStatus = s
                            default: began = true
                            }
                            continuation.yield(e)
                        }
                        continuation.finish()
                        return
                    } catch {
                        if error is CancellationError || Task.isCancelled {
                            continuation.finish(throwing: CancellationError())
                            return
                        }
                        guard !began, attempt < retryDelays.count, Self.isTransient(error) else {
                            continuation.finish(throwing: error)
                            return
                        }
                        continuation.yield(.status(EngineStatus(state: lastStatus?.state ?? "starting", phase: "Reconnecting",
                                                                gpu: lastStatus?.gpu, count: lastStatus?.count)))
                        do {
                            try await Task.sleep(for: .seconds(retryDelays[attempt]))
                        } catch {
                            continuation.finish(throwing: CancellationError())
                            return
                        }
                        attempt += 1
                    }
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// A dropped or refused connection, or a gateway error: worth sending the request again.
    static func isTransient(_ error: Error) -> Bool {
        if case LLMError.http(let code, _) = error { return [502, 503, 504].contains(code) }
        guard let u = error as? URLError else { return false }
        return [.networkConnectionLost, .timedOut, .notConnectedToInternet, .cannotConnectToHost, .dataNotAllowed,
                .cannotFindHost, .dnsLookupFailed].contains(u.code)
    }

    /// One request: its events, until `[DONE]` or the end of the stream.
    private func once(_ request: ChatRequest, yield: (ChatEvent) -> Void) async throws {
        var request = request
        request.stream = true
        var r = http.request("\(apiPath)/chat/completions", method: "POST")
        r.httpBody = try JSONEncoder().encode(request)
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        r.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        // The server pings every few seconds, also through a cold start.
        r.timeoutInterval = 120
        let (bytes, response) = try await http.session.bytes(for: r)
        if let status = (response as? HTTPURLResponse)?.statusCode, !(200..<300).contains(status) {
            var body = Data()
            for try await b in bytes {
                body.append(b)
                if body.count > 65536 { break }
            }
            throw LLMError.http(status, Self.errorText(body))
        }
        var parser = SSEParser()
        var line: [UInt8] = []
        line.reserveCapacity(4096)
        for try await b in bytes {
            if b == 10 {
                if let e = parser.feed(line: String(decoding: line, as: UTF8.self)), try Self.handle(e, yield: yield) { return }
                line.removeAll(keepingCapacity: true)
            } else {
                line.append(b)
            }
        }
        if !line.isEmpty, let e = parser.feed(line: String(decoding: line, as: UTF8.self)) { _ = try Self.handle(e, yield: yield) }
        if let e = parser.flush() { _ = try Self.handle(e, yield: yield) }
    }

    /// Yields the event's chat events; true at `[DONE]`.
    static func handle(_ e: SSEEvent, yield: (ChatEvent) -> Void) throws -> Bool {
        switch e.event {
        case "aistudio.status":
            if let s = try? JSONDecoder().decode(EngineStatus.self, from: Data(e.data.utf8)) { yield(.status(s)) }
            return false
        case "aistudio.usage":
            if let s = try? JSONDecoder().decode(RequestStats.self, from: Data(e.data.utf8)) { yield(.stats(s)) }
            return false
        case "aistudio.error":
            let obj = try? JSONSerialization.jsonObject(with: Data(e.data.utf8)) as? [String: Any]
            throw LLMError.server(obj?["error"] as? String ?? e.data)
        case nil, "message":
            if e.data == "[DONE]" { return true }
            guard let obj = try? JSONSerialization.jsonObject(with: Data(e.data.utf8)) as? [String: Any] else {
                throw LLMError.badReply(String(e.data.prefix(200)))
            }
            for ev in try ChunkDecoder.events(obj) { yield(ev) }
            return false
        default:
            return false
        }
    }

    static func errorText(_ data: Data) -> String {
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let e = obj["error"] as? String { return e }
            if let e = obj["error"] as? [String: Any], let m = e["message"] as? String { return m }
            if let d = obj["detail"] as? String { return d }
        }
        return String(decoding: data.prefix(500), as: UTF8.self)
    }
}

/// A reply put together from its events: what a chat keeps once the stream ends.
public struct ReplyAccumulator: Sendable {
    public var content = ""
    public var reasoning = ""
    public var tools = ToolCallAssembler()
    public var finishReason: String?
    public var stats: RequestStats?
    public var promptTokens: Int?
    public var completionTokens: Int?
    public var status: EngineStatus?
    /// When the first token arrived, for a live tokens-per-second figure.
    public var firstToken: Date?
    private var splitter: ThinkSplitter?
    private var sawReasoning = false

    /// `inlineThinking`: split `<think>` tags out of the content (when the server has no reasoning parser).
    public init(inlineThinking: Bool = true) {
        splitter = inlineThinking ? ThinkSplitter() : nil
    }

    public mutating func add(_ e: ChatEvent, now: Date = Date()) {
        switch e {
        case .content(let t):
            if firstToken == nil { firstToken = now }
            if !sawReasoning, var s = splitter {
                for p in s.feed(t) {
                    switch p {
                    case .content(let c): content += c
                    case .reasoning(let r): reasoning += r
                    }
                }
                splitter = s
            } else {
                content += t
            }
        case .reasoning(let t):
            if firstToken == nil { firstToken = now }
            sawReasoning = true
            reasoning += t
        case .toolCall(let i, let id, let name, let args):
            if firstToken == nil { firstToken = now }
            tools.add(index: i, id: id, name: name, arguments: args)
        case .finish(let reason): finishReason = reason
        case .usage(let p, let c): promptTokens = p; completionTokens = c
        case .status(let s): status = s
        case .stats(let s): stats = s
        }
    }

    /// Whether the stream ended (`finish` was called).
    public private(set) var finished = false

    /// Flushes text the think splitter held back.
    public mutating func finish() {
        finished = true
        guard var s = splitter else { return }
        for p in s.finish() {
            switch p {
            case .content(let c): content += c
            case .reasoning(let r): reasoning += r
            }
        }
        splitter = s
    }
}
