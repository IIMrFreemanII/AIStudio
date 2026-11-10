import Foundation
import GraphRuntime

/// A connection to `mlxcomfy-cli serve`: REST calls, streamed uploads and downloads, and one WebSocket for control and
/// run events that reconnects on its own and replays what it missed.
///
/// One long-lived `URLSession` and one socket per server; nothing is set up per run.
public final class RemoteClient: @unchecked Sendable {
    public struct Endpoint: Sendable, Equatable {
        public var baseURL: URL
        public var token: String
        public init(baseURL: URL, token: String) { self.baseURL = baseURL; self.token = token }
    }

    public enum ClientError: Error, CustomStringConvertible, Equatable {
        case http(Int, String)
        case rejected(String)
        case disconnected
        case protocolMismatch(String)
        public var description: String {
            switch self {
            case .http(let code, let body): return code == 401 ? "the server refused the token (401)" : "server error \(code): \(body)"
            case .rejected(let m): return m
            case .disconnected: return "lost the connection to the server"
            case .protocolMismatch(let m): return m
            }
        }
    }

    public let endpoint: Endpoint
    public let hello: ClientHello
    let session: URLSession
    private let lock = NSLock()
    private var socket: URLSessionWebSocketTask?
    private var connecting: Task<Void, Error>?
    private var closed = false
    private var pendingSubmits: [String: CheckedContinuation<(UInt32, [String]), Error>] = [:]
    private var runHandlers: [UInt32: (WireEvent) -> Void] = [:]
    /// Records of runs whose `submitted` reply hasn't been handled yet (the run can start before it arrives).
    private var orphans: [UInt32: [WireEvent]] = [:]
    private var lastSeq: [UInt32: UInt32] = [:]
    private var textHandlers: [String: (ServerMessage) -> Void] = [:]
    private var pongs: [Int: CheckedContinuation<Void, Never>] = [:]
    /// Called with every server message not handled above (errors, welcome…).
    public var onMessage: ((ServerMessage) -> Void)?
    /// Connection state changes (true = connected).
    public var onConnection: ((Bool) -> Void)?

    public init(endpoint: Endpoint, hello: ClientHello) {
        self.endpoint = endpoint
        self.hello = hello
        let cfg = URLSessionConfiguration.default
        cfg.httpMaximumConnectionsPerHost = 6
        cfg.timeoutIntervalForRequest = 60
        cfg.timeoutIntervalForResource = 24 * 3600
        cfg.requestCachePolicy = .reloadIgnoringLocalCacheData
        cfg.urlCache = nil
        session = URLSession(configuration: cfg)
    }

    deinit { close() }

    public func close() {
        let s: URLSessionWebSocketTask? = lock.withLock {
            closed = true
            defer { socket = nil }
            return socket
        }
        s?.cancel(with: .goingAway, reason: nil)
        failPending(ClientError.disconnected)
    }

    // MARK: REST

    func request(_ path: String, method: String = "GET") -> URLRequest {
        var r = URLRequest(url: endpoint.baseURL.appendingPathComponent(path))
        r.httpMethod = method
        r.setValue("Bearer \(endpoint.token)", forHTTPHeaderField: "Authorization")
        return r
    }

    func check(_ response: URLResponse, _ data: Data?) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            throw ClientError.http(http.statusCode, String(decoding: (data ?? Data()).prefix(500), as: UTF8.self))
        }
    }

    public func info() async throws -> ServerInfo {
        let (data, response) = try await session.data(for: request("v1/info"))
        try check(response, data)
        return try JSONDecoder().decode(ServerInfo.self, from: data)
    }

    public func post<T: Encodable>(_ path: String, _ body: T) async throws {
        var r = request(path, method: "POST")
        r.httpBody = try JSONEncoder().encode(body)
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await session.data(for: r)
        try check(response, data)
    }

    /// Uploads an input file unless the server has it. Streams from disk (constant memory) and resumes a partial
    /// upload from where the server says it stopped.
    public func upload(_ upload: GraphTransport.Upload, progress: @escaping (Int) -> Void = { _ in }) async throws {
        var attempt = 0
        while true {
            let (_, head) = try await session.data(for: request("v1/files/\(upload.name)", method: "HEAD"))
            let status = (head as? HTTPURLResponse)?.statusCode ?? 0
            if status == 200 { progress(upload.bytes); return }
            if status == 401 { throw ClientError.http(401, "") }
            let offset = ((head as? HTTPURLResponse)?.value(forHTTPHeaderField: "X-Upload-Offset")).flatMap(Int.init) ?? 0
            var r = request("v1/files/\(upload.name)", method: "PUT")
            r.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            r.setValue(String(upload.bytes - offset), forHTTPHeaderField: "Content-Length")
            if offset > 0 {
                r.setValue("bytes \(offset)-\(upload.bytes - 1)/\(upload.bytes)", forHTTPHeaderField: "Content-Range")
            }
            guard let stream = InputStream(url: upload.url) else { throw CocoaError(.fileReadNoSuchFile) }
            if offset > 0 { stream.setProperty(NSNumber(value: offset), forKey: .fileCurrentOffsetKey) }
            r.httpBodyStream = stream
            do {
                let (data, response) = try await session.data(for: r)
                try check(response, data)
                progress(upload.bytes)
                return
            } catch let e as ClientError {
                throw e
            } catch {
                // A dropped connection: ask the server how far it got and continue from there.
                attempt += 1
                if attempt >= 4 { throw error }
                try await Task.sleep(nanoseconds: UInt64(attempt) * 500_000_000)
            }
        }
    }

    /// Downloads a run output to `destination`, resuming after a dropped connection.
    public func download(run: UInt32, name: String, to destination: URL) async throws {
        let req = request("v1/runs/\(run)/outputs/\(name)")
        var resumeData: Data?
        for attempt in 0..<4 {
            do {
                let (temp, response): (URL, URLResponse)
                if let resumeData { (temp, response) = try await session.download(resumeFrom: resumeData) }
                else { (temp, response) = try await session.download(for: req) }
                try check(response, nil)
                try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                        withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: temp, to: destination)
                return
            } catch let e as ClientError {
                throw e
            } catch {
                resumeData = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data
                if attempt == 3 { throw error }
                try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 500_000_000)
            }
        }
    }

    // MARK: WebSocket

    /// Opens the socket (once; later calls wait for the same connection).
    public func connect() async throws {
        let task: Task<Void, Error> = lock.withLock {
            if let c = connecting { return c }
            let t = Task { try await self.openSocket() }
            connecting = t
            return t
        }
        do { try await task.value } catch {
            lock.withLock { connecting = nil }
            throw error
        }
    }

    private func openSocket() async throws {
        var comps = URLComponents(url: endpoint.baseURL.appendingPathComponent("v1/ws"), resolvingAgainstBaseURL: false)!
        comps.scheme = comps.scheme == "https" ? "wss" : "ws"
        var r = URLRequest(url: comps.url!)
        r.setValue("Bearer \(endpoint.token)", forHTTPHeaderField: "Authorization")
        let ws = session.webSocketTask(with: r)
        ws.maximumMessageSize = 64 << 20
        lock.withLock { socket = ws }
        ws.resume()
        try await sendRaw(.hello(hello), on: ws)
        // Wait for the welcome before anything else.
        let first = try await ws.receive()
        guard case .string(let text) = first, case .welcome(let version, _, _) = try ServerMessage.decode(text) else {
            if case .string(let text) = first, case .error(let m)? = try? ServerMessage.decode(text) {
                throw ClientError.protocolMismatch(m)
            }
            throw ClientError.protocolMismatch("unexpected first message from the server")
        }
        guard version == RemoteProtocol.version else {
            throw ClientError.protocolMismatch("server speaks protocol \(version), this app \(RemoteProtocol.version)")
        }
        // Re-attach to runs still in flight (after a reconnect).
        let resume = lock.withLock { runHandlers.keys.map { ($0, lastSeq[$0] ?? 0) } }
        for (run, seq) in resume { try await sendRaw(.subscribe(run: run, lastSeq: seq), on: ws) }
        onConnection?(true)
        Task { await self.receiveLoop(ws) }
    }

    private func receiveLoop(_ ws: URLSessionWebSocketTask) async {
        while true {
            let message: URLSessionWebSocketTask.Message
            do { message = try await ws.receive() } catch { break }
            switch message {
            case .data(let data):
                guard let events = try? WireEvent.decodeAll(data) else { continue }
                for e in events { dispatch(e) }
            case .string(let text):
                guard let msg = try? ServerMessage.decode(text) else { continue }
                dispatch(msg)
            @unknown default: break
            }
        }
        onConnection?(false)
        let reconnect: Bool = lock.withLock {
            if socket === ws { socket = nil; connecting = nil }
            return !closed && !runHandlers.isEmpty
        }
        failPending(ClientError.disconnected, keepRuns: true)
        guard reconnect else { return }
        // Runs continue on the server while we're away: reconnect with backoff and replay their events.
        for attempt in 1...30 {
            try? await Task.sleep(nanoseconds: UInt64(min(attempt, 10)) * 1_000_000_000)
            if lock.withLock({ closed }) { return }
            if (try? await connect()) != nil { return }
        }
        let handlers: [(WireEvent) -> Void] = lock.withLock {
            defer { runHandlers.removeAll() }
            return Array(runHandlers.values)
        }
        // Give up: tell every waiting run it failed.
        for h in handlers {
            h(WireEvent(kind: .runFailed, run: 0, payload: Data("lost the connection to the server".utf8)))
        }
    }

    private func dispatch(_ e: WireEvent) {
        let handler: ((WireEvent) -> Void)? = lock.withLock {
            if e.seq > 0 { lastSeq[e.run] = max(lastSeq[e.run] ?? 0, e.seq) }
            if let h = runHandlers[e.run] { return h }
            if e.kind != .metrics && e.kind != .metricsHistory { orphans[e.run, default: []].append(e) }
            return nil
        }
        handler?(e)
    }

    private func dispatch(_ msg: ServerMessage) {
        switch msg {
        case .submitted(let id, let run, let nodes):
            lock.withLock { pendingSubmits.removeValue(forKey: id) }?.resume(returning: (run, nodes))
        case .rejected(let id, let message):
            lock.withLock { pendingSubmits.removeValue(forKey: id) }?.resume(throwing: ClientError.rejected(message))
        case .pong(let id):
            lock.withLock { pongs.removeValue(forKey: id) }?.resume()
        case .textDelta(let id, _), .textDone(let id, _, _):
            lock.withLock { textHandlers[id] }?(msg)
        default:
            onMessage?(msg)
        }
    }

    private func failPending(_ error: Error, keepRuns: Bool = false) {
        let (submits, texts, ps): ([CheckedContinuation<(UInt32, [String]), Error>], [(ServerMessage) -> Void], [CheckedContinuation<Void, Never>]) = lock.withLock {
            defer {
                pendingSubmits.removeAll()
                pongs.removeAll()
                if !keepRuns { runHandlers.removeAll() }
            }
            return (Array(pendingSubmits.values), Array(textHandlers.values), Array(pongs.values))
        }
        submits.forEach { $0.resume(throwing: error) }
        ps.forEach { $0.resume() }
        for t in texts { t(.textDone(requestID: "", text: nil, error: String(describing: error))) }
    }

    private func sendRaw(_ msg: ClientMessage, on ws: URLSessionWebSocketTask) async throws {
        try await ws.send(.string(try msg.jsonString()))
    }

    public func send(_ msg: ClientMessage) async throws {
        try await connect()
        guard let ws = lock.withLock({ socket }) else { throw ClientError.disconnected }
        try await sendRaw(msg, on: ws)
    }

    /// Queues a run. `onEvent` gets every record of it (in order, starting with any that arrived before the reply)
    /// until a terminal record (finished, cancelled, runFailed).
    public func submit(_ req: SubmitRequest, onEvent: @escaping (WireEvent) -> Void) async throws -> (run: UInt32, nodes: [String]) {
        try await connect()
        let (run, nodes) = try await withCheckedThrowingContinuation { (c: CheckedContinuation<(UInt32, [String]), Error>) in
            lock.withLock { pendingSubmits[req.requestID] = c }
            Task {
                do { try await self.send(.submit(req)) } catch {
                    self.lock.withLock { self.pendingSubmits.removeValue(forKey: req.requestID) }?.resume(throwing: error)
                }
            }
        }
        let early: [WireEvent] = lock.withLock {
            runHandlers[run] = { [weak self] e in
                if [.finished, .cancelled, .runFailed].contains(e.kind) {
                    _ = self?.lock.withLock { self?.runHandlers.removeValue(forKey: run) }
                }
                onEvent(e)
            }
            return orphans.removeValue(forKey: run) ?? []
        }
        // Deliver through the handler so a terminal record among them also unregisters it.
        let handler = lock.withLock { runHandlers[run] }
        for e in early { handler?(e) }
        return (run, nodes)
    }

    public func cancel(run: UInt32) async { try? await send(.cancel(run: run)) }

    /// Round-trip time of one ping, in seconds.
    public func ping() async throws -> Double {
        try await connect()
        let id = Int.random(in: 1...Int(Int32.max))
        let start = DispatchTime.now().uptimeNanoseconds
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            lock.withLock { pongs[id] = c }
            Task { try? await self.send(.ping(id: id)) }
        }
        return Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
    }

    /// Runs the prompt assistant on the server; `onText` gets the text so far. Returns the final text.
    public func generateText(_ req: TextRequest, onText: @escaping (String) -> Void) async throws -> String {
        try await connect()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (c: CheckedContinuation<String, Error>) in
                var done = false
                lock.withLock {
                    textHandlers[req.requestID] = { [weak self] msg in
                        switch msg {
                        case .textDelta(_, let t): onText(t)
                        case .textDone(_, let t, let e):
                            guard !done else { return }
                            done = true
                            _ = self?.lock.withLock { self?.textHandlers.removeValue(forKey: req.requestID) }
                            if let t { c.resume(returning: t) } else { c.resume(throwing: ClientError.rejected(e ?? "failed")) }
                        default: break
                        }
                    }
                }
                Task { try? await self.send(.text(req)) }
            }
        } onCancel: {
            Task { try? await self.send(.cancelText(requestID: req.requestID)) }
        }
    }
}
