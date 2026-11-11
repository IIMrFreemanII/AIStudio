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

    /// The reply as events. Cancel the consuming task (or stop iterating) to stop generation.
    public func stream(_ request: ChatRequest) -> AsyncThrowingStream<ChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
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
                    func handle(_ e: SSEEvent) throws -> Bool {
                        try Self.handle(e) { continuation.yield($0) }
                    }
                    var done = false
                    for try await b in bytes {
                        if b == 10 {
                            if let e = parser.feed(line: String(decoding: line, as: UTF8.self)), try handle(e) {
                                done = true
                                break
                            }
                            line.removeAll(keepingCapacity: true)
                        } else {
                            line.append(b)
                        }
                    }
                    if !done {
                        if !line.isEmpty, let e = parser.feed(line: String(decoding: line, as: UTF8.self)) {
                            _ = try handle(e)
                        }
                        if let e = parser.flush() { _ = try handle(e) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error is CancellationError ? CancellationError() : error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
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

    /// Flushes text the think splitter held back.
    public mutating func finish() {
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
