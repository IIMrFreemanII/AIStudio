import Foundation
import HTTPTypes
import GraphRuntime
import Hummingbird
import HummingbirdWebSocket
import MLX
import AIStudioProtocol
import MLXDiffusionCore
import NIOCore
import NodesStandard

/// Request context: WebSocket upgrades plus the peer address (shown in the dashboard's client list).
public struct ServerContext: RequestContext, WebSocketRequestContext, RemoteAddressRequestContext {
    public var coreContext: CoreRequestContextStorage
    public let webSocket: WebSocketHandlerReference<ServerContext>
    public let remoteAddress: SocketAddress?

    public init(source: ApplicationRequestContextSource) {
        coreContext = .init(source: source)
        webSocket = .init()
        remoteAddress = source.channel.remoteAddress
    }
}

/// `aistudio-cli serve`: runs graphs for remote AIStudio apps and serves the dashboard.
public final class RemoteServer: @unchecked Sendable {
    public static let version = "0.4.0"

    let config: ServerConfigBox
    let auth: Auth
    let hub = Hub()
    let files: FileStore
    let scheduler: Scheduler
    let log = ServerLog.shared
    let started = Date()
    private var metrics = MetricsRing()
    private let metricsLock = NSLock()
    private var cachedModels: (at: Date, list: [InstalledModel])?
    private var cachedDisk: (at: Date, inputBytes: Int, disks: [DiskInfo])?
    private let cacheLock = NSLock()

    public init(config: ServerConfig) {
        let box = ServerConfigBox(config)
        self.config = box
        auth = Auth(token: config.token)
        files = FileStore(directory: config.inputsDirectory)
        scheduler = Scheduler(config: box, hub: hub, files: files)
        // In server mode a graph may only touch uploads, installed models and its run folder.
        PathSandbox.roots = .init(inputs: config.inputsDirectory, models: config.modelsDirectory)
        ModelPaths.directory = config.modelsDirectory
        let hub = self.hub
        log.onLine = { line in hub.sendToDashboards(.log(line)) }
    }

    // MARK: running

    /// Builds the application (also used by tests, which drive it without a socket).
    public func makeApplication() -> some ApplicationProtocol {
        let router = Router(context: ServerContext.self)
        router.add(middleware: SecurityMiddleware(auth: auth, tls: false))
        addRoutes(router)
        let cfg = config.value
        return Application(
            router: router,
            server: .http1WebSocketUpgrade(webSocketRouter: router, configuration: .init(
                ws: .init(maxFrameSize: 64 << 20, autoPing: .enabled(timePeriod: .seconds(20))))),
            configuration: .init(address: .hostname(cfg.host, port: cfg.port), serverName: "AIStudio/\(Self.version)"),
            onServerRunning: { [log] channel in
                log.info("listening on \(channel.localAddress?.description ?? "\(cfg.host):\(cfg.port)")")
            })
    }

    public func run() async throws {
        await scheduler.start()
        startBackgroundTasks()
        let cfg = config.value
        log.info("AIStudio server \(Self.version) on \(SystemInfo.chip), \(SystemInfo.physicalMemory / (1 << 30)) GB")
        log.info("models: \(cfg.modelsDirectory.path)")
        log.info("data: \(cfg.dataDirectory.path)")
        try await makeApplication().runService()
    }

    func startBackgroundTasks() {
        // Model sizes take a folder walk each; have them ready before the first client asks.
        Task.detached { [self] in _ = await installedModels(refresh: true) }
        // Metrics: one sample per second into the ring, pushed to dashboards.
        Task.detached { [self] in
            var lastCPU = SystemInfo.cpuSeconds(), lastTime = Date()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                let now = Date()
                let cpu = SystemInfo.cpuSeconds()
                let dt = max(0.001, now.timeIntervalSince(lastTime))
                let cpuPercent = (cpu - lastCPU) / dt * 100
                lastCPU = cpu; lastTime = now
                let (events, previews, out, inn, p50, p99) = hub.takeCounters()
                let queue = await scheduler.queueLength
                var sample = [Float](repeating: 0, count: MetricID.count)
                sample[MetricID.memoryActive.rawValue] = Float(Memory.activeMemory)
                sample[MetricID.memoryCache.rawValue] = Float(Memory.cacheMemory)
                sample[MetricID.memoryPeak.rawValue] = Float(Memory.peakMemory)
                sample[MetricID.gpuUtilization.rawValue] = Float(SystemInfo.gpuUtilization() ?? 0)
                sample[MetricID.cpuProcess.rawValue] = Float(cpuPercent)
                sample[MetricID.eventsPerSecond.rawValue] = Float(Double(events) / dt)
                sample[MetricID.previewsPerSecond.rawValue] = Float(Double(previews) / dt)
                sample[MetricID.bytesOutPerSecond.rawValue] = Float(Double(out) / dt)
                sample[MetricID.bytesInPerSecond.rawValue] = Float(Double(inn) / dt)
                sample[MetricID.eventLatencyP50.rawValue] = p50
                sample[MetricID.eventLatencyP99.rawValue] = p99
                sample[MetricID.queueLength.rawValue] = Float(queue)
                let t = now.timeIntervalSince1970
                metricsLock.withLock { metrics.push(time: t, sample) }
                hub.publishToDashboards(WireEvent(kind: .metrics, run: 0, payload: MetricsCodec.sample(time: t, sample)))
            }
        }
        // Dashboard state: rebuilt a few times per second while a dashboard is open, sent only when it changed.
        Task.detached { [self] in
            var last = Data()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 250_000_000)
                guard hub.hasDashboards else { last = Data(); continue }
                let state = await dashboardState()
                // Live memory streams in the binary metrics; resend the state only when something else changed.
                var stable = state
                stable.info.memory = MemoryInfo(total: 0, active: 0, cache: 0, peak: 0, budget: state.info.memory.budget)
                guard let data = try? JSONEncoder().encode(stable), data != last else { continue }
                last = data
                hub.sendToDashboards(.state(state))
            }
        }
    }

    // MARK: state

    func installedModels(refresh: Bool = false) async -> [InstalledModel] {
        let loaded = Set(await scheduler.loadedModels().map(\.name))
        let cached: [InstalledModel]? = cacheLock.withLock {
            guard !refresh, let c = cachedModels, Date().timeIntervalSince(c.at) < 30 else { return nil }
            return c.list
        }
        let list = cached ?? {
            let dir = config.value.modelsDirectory
            let names = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
                .filter { !$0.hasPrefix(".") }.sorted()
            let list = names.compactMap { name -> InstalledModel? in
                var isDir: ObjCBool = false
                let url = dir.appendingPathComponent(name)
                guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else { return nil }
                return InstalledModel(name: name, bytes: ModelRegistry.weightBytes(at: url), loaded: false)
            }
            cacheLock.withLock { cachedModels = (Date(), list) }
            return list
        }()
        return list.map { var m = $0; m.loaded = loaded.contains(m.name); return m }
    }

    func info() async -> ServerInfo {
        let cfg = config.value
        return ServerInfo(
            protocolVersion: RemoteProtocol.version, serverVersion: Self.version, hostName: SystemInfo.hostName,
            chip: SystemInfo.chip, osVersion: SystemInfo.osVersion, gpu: SystemInfo.gpu,
            memory: MemoryInfo(total: SystemInfo.physicalMemory, active: Memory.activeMemory, cache: Memory.cacheMemory,
                               peak: Memory.peakMemory,
                               budget: Int(Double(SystemInfo.physicalMemory) * min(max(cfg.budgetFraction, 0.3), 0.95))),
            nodeTypes: await scheduler.nodeTypes, models: await installedModels())
    }

    func dashboardState() async -> DashboardState {
        let cfg = config.value
        let (queue, history) = await scheduler.snapshot()
        let disk: (Int, [DiskInfo]) = {
            if let c = cacheLock.withLock({ cachedDisk }), Date().timeIntervalSince(c.at) < 5 { return (c.inputBytes, c.disks) }
            let dirs = [("models", cfg.modelsDirectory), ("data", cfg.dataDirectory)]
            let disks = dirs.compactMap { name, url in
                SystemInfo.disk(url).map { DiskInfo(path: "\(name): \(url.path)", free: $0.free, total: $0.total) }
            }
            let bytes = files.totalBytes
            cacheLock.withLock { cachedDisk = (Date(), bytes, disks) }
            return (bytes, disks)
        }()
        return DashboardState(
            info: await info(), started: started.timeIntervalSince1970, thermalState: SystemInfo.thermalState,
            queue: queue, history: Array(history.prefix(100)), loadedModels: await scheduler.loadedModels(),
            clients: hub.clientInfos, transfers: files.activeTransfers, inputCacheBytes: disk.0, disks: disk.1,
            settings: ServerSettings(budgetFraction: cfg.budgetFraction, previewHz: cfg.previewHz,
                                     previewMaxSide: cfg.previewMaxSide, host: cfg.host, port: cfg.port,
                                     dataDirectory: cfg.dataDirectory.path, modelsDirectory: cfg.modelsDirectory.path))
    }

    // MARK: routes

    func addRoutes(_ router: Router<ServerContext>) {
        router.get("/v1/health") { _, _ in "ok" }

        router.get("/v1/info") { [self] _, _ in try json(await info()) }

        router.post("/v1/session") { [self] request, context in
            struct Login: Decodable { var token: String }
            let body: Login = try await decode(request)
            guard let (id, csrf) = await auth.login(token: body.token) else { throw HTTPError(.unauthorized) }
            var response = Response(status: .noContent)
            let secure = Self.isHTTPS(request.headers, directTLS: request.uri.scheme == .https)
            response.setCookie(Cookie(name: Auth.sessionCookie, value: id, maxAge: Int(Auth.sessionLifetime), path: "/",
                                      secure: secure, httpOnly: true, sameSite: .strict))
            response.setCookie(Cookie(name: Auth.csrfCookie, value: csrf, maxAge: Int(Auth.sessionLifetime), path: "/",
                                      secure: secure, httpOnly: false, sameSite: .strict))
            return response
        }

        router.delete("/v1/session") { [self] request, _ in
            if case .session(let id, _)? = auth.principal(for: request) { auth.logout(id) }
            var response = Response(status: .noContent)
            response.setCookie(Cookie(name: Auth.sessionCookie, value: "", maxAge: 0, path: "/"))
            return response
        }

        router.get("/v1/state") { [self] _, _ in try json(await dashboardState()) }

        // Uploads: HEAD says whether a file is there (200) or how much of it arrived (404 + X-Upload-Offset).
        router.head("/v1/files/:name") { [self] _, context in
            let name = try context.parameters.require("name")
            guard GraphTransport.isServerName(name) else { throw HTTPError(.badRequest) }
            if files.exists(name) { return Response(status: .ok) }
            var r = Response(status: .notFound)
            r.headers[Self.uploadOffset] = String(files.partialSize(name))
            return r
        }

        router.put("/v1/files/:name") { [self] request, context in
            let name = try context.parameters.require("name")
            guard GraphTransport.isServerName(name) else { throw HTTPError(.badRequest, message: "bad file name") }
            var offset = 0
            var total = request.headers[.contentLength].flatMap(Int.init)
            if let range = request.headers[.contentRange], let parsed = Self.parseContentRange(range) {
                offset = parsed.start
                total = parsed.total
            }
            do {
                let complete = try await files.receive(name, offset: offset, total: total, body: request.body) { n in
                    hub.countTransfer(inBytes: n)
                }
                var r = Response(status: complete ? .created : .accepted)
                r.headers[Self.uploadOffset] = String(complete ? (total ?? 0) : files.partialSize(name))
                return r
            } catch let e as FileStore.UploadError {
                if case .offsetMismatch(let expected) = e {
                    var r = Response(status: .conflict)
                    r.headers[Self.uploadOffset] = String(expected)
                    return r
                }
                throw HTTPError(.badRequest, message: e.description)
            }
        }

        router.delete("/v1/files") { [self] _, _ in
            let freed = files.purge(keeping: await scheduler.neededFiles)
            log.info("input cache purged: \(freed / (1 << 20)) MB freed")
            return try json(["freed": freed])
        }

        // Outputs, with Range support so big videos can resume.
        router.get("/v1/runs/:run/outputs/:name") { [self] request, context in
            let run = try context.parameters.require("run", as: UInt32.self)
            let name = try context.parameters.require("name")
            guard let folder = await scheduler.folder(of: run) else { throw HTTPError(.notFound) }
            guard let url = try? PathSandbox.contained(name, in: folder.resolvingSymlinksInPath()) else {
                throw HTTPError(.notFound)
            }
            return try fileResponse(url, request: request)
        }

        router.post("/v1/runs/:run/cancel") { [self] _, context in
            await scheduler.cancel(try context.parameters.require("run", as: UInt32.self))
            return Response(status: .noContent)
        }

        router.delete("/v1/queue") { [self] _, _ in
            await scheduler.clearWaiting()
            return Response(status: .noContent)
        }

        router.get("/v1/models") { [self] _, _ in try json(await installedModels(refresh: true)) }

        router.get("/v1/models/loaded") { [self] _, _ in try json(await scheduler.loadedModels()) }

        router.post("/v1/models/preload") { [self] request, _ in
            struct Body: Decodable { var name: String }
            let body: Body = try await decode(request)
            do { try await scheduler.preload(body.name) } catch {
                throw HTTPError(.badRequest, message: String(describing: error))
            }
            return Response(status: .noContent)
        }

        router.post("/v1/models/unload") { [self] request, _ in
            struct Body: Decodable { var key: String? }
            let body: Body = try await decode(request)
            if let key = body.key {
                await scheduler.unload(key)
            } else {
                for m in await scheduler.loadedModels() where !m.pinned { await scheduler.unload(m.key) }
            }
            return Response(status: .noContent)
        }

        router.get("/v1/settings") { [self] _, _ in try json(await dashboardState().settings) }

        router.post("/v1/settings") { [self] request, _ in
            struct Body: Decodable { var budgetFraction: Double?; var previewHz: Double?; var previewMaxSide: Int? }
            let b: Body = try await decode(request)
            config.update { c in
                if let v = b.budgetFraction { c.budgetFraction = min(max(v, 0.3), 0.95) }
                if let v = b.previewHz { c.previewHz = min(max(v, 0), 30) }
                if let v = b.previewMaxSide { c.previewMaxSide = min(max(v, 64), 2048) }
            }
            if let v = b.budgetFraction { await scheduler.setBudget(fraction: v) }
            log.info("settings changed from the dashboard")
            return try json(await dashboardState().settings)
        }

        router.ws("/v1/ws", shouldUpgrade: { [self] request, _ in
            guard let who = auth.principal(for: request) else { throw HTTPError(.unauthorized) }
            if case .session = who {
                // Cookies ride along on cross-site requests too: a cookie session must come from this origin.
                guard let origin = request.headers[.origin],
                      Self.isSameOrigin(origin, authority: request.head.authority, headers: request.headers)
                else { throw HTTPError(.forbidden) }
            }
            return .upgrade([:])
        }, onUpgrade: { [self] inbound, outbound, context in
            let fullAccess = auth.principal(for: context.request) == .bearer
            let address = context.requestContext.remoteAddress?.ipAddress ?? "unknown"
            await session(inbound: inbound, outbound: outbound, address: address, fullAccess: fullAccess)
        })

        if config.value.dashboard { addDashboardRoutes(router) }
    }

    static let uploadOffset = HTTPField.Name("X-Upload-Offset")!

    /// One WebSocket client, from hello to close.
    func session(inbound: WebSocketInboundStream, outbound: WebSocketOutboundWriter, address: String,
                 fullAccess: Bool) async {
        let client = Hub.Client(session: ServerConfig.generateToken().prefix(16).description, address: address)
        hub.add(client)
        let pump = Task { await hub.pump(client, outbound: outbound) }
        defer {
            hub.remove(client.session)
            client.signal.finish()
            pump.cancel()
            log.info("\(client.name) (\(address)) disconnected")
        }
        do {
            for try await message in inbound.messages(maxSize: 64 << 20) {
                guard case .text(let text) = message else { continue }
                hub.countIn(client.session, bytes: text.utf8.count)
                let msg: ClientMessage
                do { msg = try ClientMessage.decode(text) } catch {
                    hub.send(.error(message: "bad message: \(error)"), to: client.session)
                    continue
                }
                await handle(msg, client: client, fullAccess: fullAccess)
            }
        } catch {
            log.warning("connection from \(address) ended: \(error)")
        }
    }

    func handle(_ msg: ClientMessage, client: Hub.Client, fullAccess: Bool) async {
        let session = client.session
        switch msg {
        case .hello(let h):
            guard h.protocolVersion == RemoteProtocol.version else {
                hub.send(.error(message: "protocol \(h.protocolVersion) not supported (server speaks \(RemoteProtocol.version))"),
                         to: session)
                return
            }
            // Only bearer clients may act as apps; a dashboard session stays a dashboard.
            let role = fullAccess ? h.role : .dashboard
            hub.update(session) { c in
                c.role = role
                c.name = String(h.name.prefix(80))
                c.previewHz = min(max(h.previewHz, 0), 30)
                c.previewMaxSide = min(max(h.previewMaxSide, 64), 2048)
            }
            hub.send(.welcome(protocolVersion: RemoteProtocol.version, serverVersion: Self.version, session: session),
                     to: session)
            log.info("\(h.name) connected from \(client.address) as \(role.rawValue)")
            if role == .dashboard {
                let history = metricsLock.withLock { MetricsCodec.history(metrics) }
                hub.update(session) { c in
                    c.pending.append(WireEvent(kind: .metricsHistory, run: 0, payload: history).encoded())
                    c.urgent = true
                }
                for line in log.recent.suffix(200) { hub.send(.log(line), to: session) }
                hub.send(.state(await dashboardState()), to: session)
            }
        case .submit(let req):
            guard fullAccess else { return hub.send(.rejected(requestID: req.requestID, message: "not allowed"), to: session) }
            do {
                let (run, nodes) = try await scheduler.submit(req, session: session, client: client.name)
                hub.send(.submitted(requestID: req.requestID, run: run, nodes: nodes), to: session)
            } catch {
                hub.send(.rejected(requestID: req.requestID, message: String(describing: error)), to: session)
            }
        case .cancel(let run):
            await scheduler.cancel(run)
        case .subscribe(let run, let lastSeq):
            hub.subscribe(session, run: run, after: lastSeq)
        case .setPreview(let hz, let side):
            hub.update(session) { c in c.previewHz = min(max(hz, 0), 30); c.previewMaxSide = min(max(side, 64), 2048) }
        case .ping(let id):
            hub.send(.pong(id: id), to: session)
        case .text(let req):
            guard fullAccess else {
                return hub.send(.textDone(requestID: req.requestID, text: nil, error: "not allowed"), to: session)
            }
            await scheduler.submitText(req, session: session)
        case .cancelText(let id):
            await scheduler.cancelText(id)
        }
    }

    // MARK: dashboard files

    static var dashboardDirectory: URL? {
        Bundle.module.url(forResource: "dashboard", withExtension: nil)
    }

    func addDashboardRoutes(_ router: Router<ServerContext>) {
        guard let root = Self.dashboardDirectory?.resolvingSymlinksInPath() else {
            log.warning("dashboard files not found in the bundle")
            return
        }
        router.get("/") { [self] request, _ in try staticFile(root.appendingPathComponent("index.html"), request: request, immutable: false) }
        router.get("/assets/**") { [self] request, context in
            let rel = context.parameters.getCatchAll().joined(separator: "/")
            guard let url = try? PathSandbox.contained("assets/" + rel, in: root) else { throw HTTPError(.notFound) }
            return try staticFile(url, request: request, immutable: true)
        }
        router.get("/favicon.svg") { [self] request, _ in
            try staticFile(root.appendingPathComponent("favicon.svg"), request: request, immutable: false)
        }
    }

    /// A dashboard file, pre-compressed variant first (`.br`, `.gz`) when the browser accepts it. Vite names assets
    /// by content hash, so they are cached forever; `index.html` revalidates.
    func staticFile(_ url: URL, request: Request, immutable: Bool) throws -> Response {
        guard FileManager.default.fileExists(atPath: url.path) else { throw HTTPError(.notFound) }
        let accept = request.headers[.acceptEncoding] ?? ""
        var chosen = url, encoding: String?
        for (ext, enc) in [("br", "br"), ("gz", "gzip")] where accept.contains(enc) {
            let candidate = URL(fileURLWithPath: url.path + "." + ext)
            if FileManager.default.fileExists(atPath: candidate.path) { chosen = candidate; encoding = enc; break }
        }
        let data = try Data(contentsOf: chosen)
        var r = Response(status: .ok, body: .init(byteBuffer: ByteBuffer(bytes: data)))
        r.headers[.contentType] = Self.mimeType(url.pathExtension)
        if let encoding { r.headers[.contentEncoding] = encoding }
        r.headers[.vary] = "Accept-Encoding"
        r.headers[.cacheControl] = immutable ? "public, max-age=31536000, immutable" : "no-cache"
        return r
    }

    static func mimeType(_ ext: String) -> String {
        switch ext.lowercased() {
        case "html": return "text/html; charset=utf-8"
        case "js", "mjs": return "text/javascript; charset=utf-8"
        case "css": return "text/css; charset=utf-8"
        case "svg": return "image/svg+xml"
        case "png": return "image/png"
        case "jpg", "jpeg": return "image/jpeg"
        case "json": return "application/json"
        case "woff2": return "font/woff2"
        case "mp4": return "video/mp4"
        case "mov": return "video/quicktime"
        case "wav": return "audio/wav"
        case "m4a": return "audio/mp4"
        case "mp3": return "audio/mpeg"
        case "flac": return "audio/flac"
        default: return "application/octet-stream"
        }
    }

    // MARK: helpers

    func json<T: Encodable>(_ value: T, status: HTTPResponse.Status = .ok) throws -> Response {
        let data = try JSONEncoder().encode(value)
        var r = Response(status: status, body: .init(byteBuffer: ByteBuffer(bytes: data)))
        r.headers[.contentType] = "application/json"
        return r
    }

    func decode<T: Decodable>(_ request: Request) async throws -> T {
        var request = request
        let buffer = try await request.collectBody(upTo: 1 << 20)
        do {
            return try JSONDecoder().decode(T.self, from: Data(buffer.readableBytesView))
        } catch {
            throw HTTPError(.badRequest, message: "bad JSON: \(error)")
        }
    }

    /// Streams a file (or the requested byte range) in 1 MB reads; nothing is held whole in memory.
    func fileResponse(_ url: URL, request: Request) throws -> Response {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: url.path), let size = attrs[.size] as? Int else {
            throw HTTPError(.notFound)
        }
        let modified = (attrs[.modificationDate] as? Date) ?? Date()
        var start = 0, end = max(0, size - 1)
        var status: HTTPResponse.Status = .ok
        if let header = request.headers[.range] {
            guard let r = Self.parseRange(header, size: size) else {
                var resp = Response(status: .rangeNotSatisfiable)
                resp.headers[.contentRange] = "bytes */\(size)"
                return resp
            }
            (start, end) = r
            status = .partialContent
        }
        let length = size == 0 ? 0 : end - start + 1
        let offset = start
        let hub = self.hub
        let id = UUID().uuidString
        files.beginTransfer(id: id, name: url.lastPathComponent, total: length, direction: .download)
        let store = files
        var response = Response(status: status, body: ResponseBody(contentLength: length) { writer in
            defer { store.endTransfer(id) }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: UInt64(offset))
            var remaining = length, sent = 0
            while remaining > 0 {
                let chunk = try autoreleasepool { try handle.read(upToCount: min(1 << 20, remaining)) } ?? Data()
                if chunk.isEmpty { break }
                try await writer.write(ByteBuffer(bytes: chunk))
                remaining -= chunk.count
                sent += chunk.count
                hub.countTransfer(outBytes: chunk.count)
                store.updateTransfer(id, bytes: sent)
            }
            try await writer.finish(nil)
        })
        response.headers[.contentType] = Self.mimeType(url.pathExtension)
        response.headers[.acceptRanges] = "bytes"
        // Validators let URLSession resume an interrupted download (outputs never change once written).
        response.headers[.eTag] = "\"\(size)-\(Int(modified.timeIntervalSince1970))\""
        response.headers[HTTPField.Name("Last-Modified")!] = Self.httpDate(modified)
        if status == .partialContent { response.headers[.contentRange] = "bytes \(start)-\(end)/\(size)" }
        return response
    }

    static func httpDate(_ d: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = TimeZone(identifier: "GMT")
        f.dateFormat = "EEE, dd MMM yyyy HH:mm:ss 'GMT'"
        return f.string(from: d)
    }

    /// `bytes=a-b`, `bytes=a-`, `bytes=-n` (one range).
    static let forwardedProto = HTTPField.Name("X-Forwarded-Proto")!
    static let forwardedHost = HTTPField.Name("X-Forwarded-Host")!

    /// Whether the browser reached us over https: directly, or through a proxy such as `tailscale serve` or Caddy that
    /// terminates TLS and says so in `X-Forwarded-Proto`. (A client could fake the header, but it only decides whether
    /// that client's own cookies get the `Secure` flag.)
    static func isHTTPS(_ headers: HTTPFields, directTLS: Bool) -> Bool {
        directTLS || headers[forwardedProto]?.split(separator: ",").first?
            .trimmingCharacters(in: .whitespaces).lowercased() == "https"
    }

    /// Whether a browser's `Origin` is this server: the `Host` it asked for, or the original host a proxy forwarded in
    /// `X-Forwarded-Host` (proxies such as nginx replace `Host`). Page scripts can't set either header on a WebSocket.
    static func isSameOrigin(_ origin: String, authority: String?, headers: HTTPFields) -> Bool {
        let hosts = [authority, headers[forwardedHost]?.split(separator: ",").first.map {
            $0.trimmingCharacters(in: .whitespaces)
        }].compactMap { $0 }.filter { !$0.isEmpty }
        return hosts.contains { origin == "http://\($0)" || origin == "https://\($0)" }
    }

    static func parseRange(_ header: String, size: Int) -> (Int, Int)? {
        guard header.hasPrefix("bytes="), !header.contains(","), size > 0 else { return nil }
        let spec = header.dropFirst(6)
        let parts = spec.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2 else { return nil }
        if parts[0].isEmpty {
            guard let n = Int(parts[1]), n > 0 else { return nil }
            return (max(0, size - n), size - 1)
        }
        guard let a = Int(parts[0]), a < size else { return nil }
        let b = parts[1].isEmpty ? size - 1 : min(Int(parts[1]) ?? -1, size - 1)
        return b >= a ? (a, b) : nil
    }

    /// `bytes a-b/total` (an upload chunk).
    static func parseContentRange(_ header: String) -> (start: Int, end: Int, total: Int?)? {
        guard header.hasPrefix("bytes ") else { return nil }
        let rest = header.dropFirst(6)
        let halves = rest.split(separator: "/", maxSplits: 1)
        guard halves.count == 2 else { return nil }
        let bounds = halves[0].split(separator: "-")
        guard bounds.count == 2, let a = Int(bounds[0]), let b = Int(bounds[1]), b >= a else { return nil }
        return (a, b, halves[1] == "*" ? nil : Int(halves[1]))
    }
}
