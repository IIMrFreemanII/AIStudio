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
    /// Moves the model to another GPU when its own doesn't come in time (nil: the stream waits).
    public var gpuWait: GPUWaitPolicy?

    /// Thrown inside an attempt when its start has waited past `gpuWait.limit`: the attempt ends, the model moves to
    /// another GPU, and the request goes again.
    struct WaitedTooLong: Error {}

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
                // GPU fallback: when the start began waiting for a GPU, whether this stream moved the model (its own
                // GPU goes back once the reply begins), and whether there's nothing left to move it to.
                var waitingSince: Date?
                var switched = false
                var settled = false
                var noMoreGPUs = false
                let policy = gpuWait
                func settle() {
                    guard switched, !settled, let policy else { return }
                    settled = true
                    let model = request.model
                    Task { await policy.settle(model) }
                }
                while true {
                    var began = false
                    do {
                        // Checked on every line (the server's pings come every few seconds; a status only when
                        // the phase changes).
                        func checkWait() throws {
                            guard let policy, !noMoreGPUs, !began, lastStatus?.state == "starting" else { return }
                            if waitingSince == nil { waitingSince = Date() }
                            if GPUFallback.overdue(since: waitingSince, now: Date(), limit: policy.limit) { throw WaitedTooLong() }
                        }
                        try await once(request, tick: checkWait) { e in
                            switch e {
                            case .status(let s):
                                lastStatus = s
                                if s.state != "starting" { waitingSince = nil }
                                try checkWait()
                            default:
                                began = true
                                settle()
                            }
                            continuation.yield(e)
                        }
                        settle()
                        continuation.finish()
                        return
                    } catch is WaitedTooLong {
                        let from = lastStatus?.gpu.map { GPUPreset(gpu: $0, count: lastStatus?.count ?? 1).label }
                        continuation.yield(.status(EngineStatus(state: "starting", phase: "Switching GPUs",
                                                                gpu: lastStatus?.gpu, count: lastStatus?.count)))
                        if let label = await policy?.switchGPU(request.model) {
                            switched = true
                            continuation.yield(.status(EngineStatus(
                                state: "starting",
                                phase: "\(from ?? "Its GPU") didn't come in \(Int(policy?.limit ?? 30)) s: waiting for \(label)")))
                        } else {
                            noMoreGPUs = true  // keep waiting where it is
                        }
                        // The next attempt's wait counts from the server's own status for it.
                        waitingSince = nil
                        lastStatus = nil
                        if Task.isCancelled {
                            settle()
                            continuation.finish(throwing: CancellationError())
                            return
                        }
                    } catch {
                        if error is CancellationError || Task.isCancelled {
                            settle()
                            continuation.finish(throwing: CancellationError())
                            return
                        }
                        // Another chat moved the model to another GPU while this one waited for it: wait again.
                        if !began, let policy, GPUWaitPolicy.isStoppedStart(error), await policy.switchedRecently(request.model) {
                            waitingSince = nil
                            continue
                        }
                        guard !began, attempt < retryDelays.count, Self.isTransient(error) else {
                            settle()
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
    private func once(_ request: ChatRequest, tick: () throws -> Void = {}, yield: (ChatEvent) throws -> Void) async throws {
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
        var finished = false
        func pass(_ e: ChatEvent) throws {
            if case .finish = e { finished = true }
            try yield(e)
        }
        for try await b in bytes {
            if b == 10 {
                try tick()
                if let e = parser.feed(line: String(decoding: line, as: UTF8.self)), try Self.handle(e, yield: pass) { return }
                line.removeAll(keepingCapacity: true)
            } else {
                line.append(b)
            }
        }
        if !line.isEmpty, let e = parser.feed(line: String(decoding: line, as: UTF8.self)), try Self.handle(e, yield: pass) {
            return
        }
        if let e = parser.flush(), try Self.handle(e, yield: pass) { return }
        // Closed before `[DONE]` and before the model finished: the connection dropped (retried when nothing came
        // yet), not an empty reply. A server that ends without `[DONE]` after its finish reason is fine.
        if !finished { throw URLError(.networkConnectionLost) }
    }

    /// Yields the event's chat events; true at `[DONE]`.
    static func handle(_ e: SSEEvent, yield: (ChatEvent) throws -> Void) throws -> Bool {
        switch e.event {
        case "aistudio.status":
            if let s = try? JSONDecoder().decode(EngineStatus.self, from: Data(e.data.utf8)) { try yield(.status(s)) }
            return false
        case "aistudio.usage":
            if let s = try? JSONDecoder().decode(RequestStats.self, from: Data(e.data.utf8)) { try yield(.stats(s)) }
            return false
        case "aistudio.error":
            let obj = try? JSONSerialization.jsonObject(with: Data(e.data.utf8)) as? [String: Any]
            throw LLMError.server(obj?["error"] as? String ?? e.data)
        case nil, "message":
            if e.data == "[DONE]" { return true }
            guard let obj = try? JSONSerialization.jsonObject(with: Data(e.data.utf8)) as? [String: Any] else {
                throw LLMError.badReply(String(e.data.prefix(200)))
            }
            for ev in try ChunkDecoder.events(obj) { try yield(ev) }
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
