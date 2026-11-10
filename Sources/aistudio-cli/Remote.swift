import ArgumentParser
import CoreGraphics
import Foundation
import GraphRuntime
import ImageIO
import AIStudioProtocol
import AIStudioServer
import MLXDiffusionCore
import NodesStandard
import UniformTypeIdentifiers

/// `~/.config/aistudio/token`. A token made before the rename (`~/.config/mlxcomfy/token`) is moved there.
let defaultTokenFile: URL = {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let url = home.appendingPathComponent(".config/aistudio/token")
    let legacy = home.appendingPathComponent(".config/mlxcomfy/token")
    let fm = FileManager.default
    if !fm.fileExists(atPath: url.path), fm.fileExists(atPath: legacy.path) {
        try? fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                attributes: [.posixPermissions: 0o700])
        try? fm.moveItem(at: legacy, to: url)
    }
    return url
}()

func expand(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath).standardizedFileURL }

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Run graphs for remote AIStudio apps, with a web dashboard at the same address.",
        discussion: """
            Binds to 127.0.0.1 by default. To use it from another machine, reach it through Tailscale, an SSH tunnel \
            or a TLS reverse proxy; see docs/remote-server.md. Every client needs the token.
            """)

    @Option(help: "Address to listen on (127.0.0.1, a Tailscale IP, or 0.0.0.0 for all interfaces).") var host = "127.0.0.1"
    @Option var port = RemoteProtocol.defaultPort
    @Option(help: "File holding the bearer token (created with a new token if missing). AISTUDIO_TOKEN overrides it.")
    var tokenFile = defaultTokenFile.path
    @Option(help: "Models folder (default: the app's).") var models: String?
    @Option(help: "Server data folder: uploads and run outputs.") var data: String?
    @Option(help: "Fraction of RAM models may use before the least recently used is unloaded.") var budget = 0.8
    @Option(help: "Default preview frames per second.") var previewHz = 4.0
    @Option(help: "Default longest side of preview frames.") var previewSize = 512
    @Option(help: "Finished runs whose output folders are kept.") var keepRuns = 200
    @Flag(help: "Don't serve the web dashboard.") var noDashboard = false
    @Flag(help: "Print the token (also when it already existed) and exit.") var showToken = false

    func run() async throws {
        let token: String
        if let env = LegacyNames.environment("AISTUDIO_TOKEN"), !env.isEmpty {
            guard env.count >= 16 else { throw ValidationError("AISTUDIO_TOKEN must be at least 16 characters") }
            token = env
        } else {
            let (t, created) = try ServerConfig.token(fromFile: expand(tokenFile), create: true)
            token = t
            if created || showToken {
                print("Token (\(created ? "new, " : "")saved in \(expand(tokenFile).path)):\n\(t)")
                if created { print("Paste it into AIStudio › Connections. It is not printed again.") }
            }
        }
        if showToken { return }
        var config = ServerConfig(host: host, port: port, token: token)
        if let models { config.modelsDirectory = expand(models) }
        if let data { config.dataDirectory = expand(data) }
        config.budgetFraction = budget
        config.previewHz = previewHz
        config.previewMaxSide = previewSize
        config.keepRuns = max(1, keepRuns)
        config.dashboard = !noDashboard
        if !config.isLoopback {
            ServerLog.shared.warning("listening on \(host): anyone who can reach this address and has the token can run "
                                     + "graphs. Prefer a Tailscale IP or an SSH tunnel; don't expose it to the internet "
                                     + "without TLS.")
        }
        try await RemoteServer(config: config).run()
    }
}

struct Download: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Download a model into the models folder (catalog id, folder name, or a Hugging Face repo).",
        discussion: "Run without arguments to list the catalog. Downloads resume where they stopped.")

    @Argument(help: "Catalog id (see the list), its folder name, or owner/repo.") var model: String?
    @Option(help: "Models folder (default: the app's).") var models: String?
    @Option(help: "File with a Hugging Face token, for gated repos.") var hfTokenFile: String?
    @Option(help: "Branch or commit (for a plain repo).") var revision = "main"

    func run() async throws {
        let entries = ModelCatalog.entries.filter(\.supported)
        guard let model else {
            for e in entries {
                print(String(format: "%-28@ %-24@ ~%3.0f GB  %@%@", e.id as NSString, e.folder as NSString, e.approxGB,
                             e.name, e.gated ? " (gated)" : ""))
            }
            return
        }
        let root = models.map(expand) ?? ModelPaths.directory
        let hfToken = try hfTokenFile.map { try String(contentsOf: expand($0), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines) }
            ?? ProcessInfo.processInfo.environment["HF_TOKEN"]
        let entry = entries.first { $0.id == model || $0.folder == model }
        let parts: [(CatalogPart, URL)]
        let dest: URL
        if let entry {
            dest = root.appendingPathComponent(entry.folder)
            parts = entry.parts.map { ($0, $0.subfolder.isEmpty ? dest : dest.appendingPathComponent($0.subfolder)) }
        } else {
            guard model.split(separator: "/").count == 2 else { throw ValidationError("unknown model \(model); run without arguments for the list") }
            dest = root.appendingPathComponent(String(model.split(separator: "/")[1]))
            parts = [(CatalogPart(repo: model, revision: revision, include: [], exclude: [".gitattributes"], subfolder: "",
                                  license: "", licenseURL: "", gated: hfToken != nil), dest)]
        }
        for (part, folder) in parts {
            let source = HTTPByteSource(repo: part.repo, revision: part.revision, token: part.gated ? hfToken : nil)
            let files = HFAPI.select(try await source.list(), include: part.include, exclude: part.exclude)
            let total = files.reduce(Int64(0)) { $0 + $1.size }
            print("\(part.repo) → \(folder.path) (\(String(format: "%.1f", Double(total) / 1e9)) GB)")
            var last = Date.distantPast
            try await ModelDownloader.download(files, to: folder, from: source) { p in
                guard Date().timeIntervalSince(last) > 1 || p.done == p.total else { return }
                last = Date()
                FileHandle.standardError.write(Data(String(format: "\r  %5.1f%%  %@", p.fraction * 100, p.file).utf8))
            }
            FileHandle.standardError.write(Data("\n".utf8))
        }
        if let entry, let manifest = entry.manifest, ModelRegistry.className(at: dest) == nil,
           let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]) {
            try data.write(to: dest.appendingPathComponent("aistudio.json"))
        }
        print("installed \(dest.lastPathComponent)")
    }
}

struct BenchRemote: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bench-remote",
        abstract: "Measure a server: request latency, event round trip, upload/download speed, and a model-free run.")

    @Argument(help: "Server URL, e.g. http://100.64.0.2:8188") var url: String
    @Option(help: "File with the bearer token (AISTUDIO_TOKEN overrides it).") var tokenFile = defaultTokenFile.path
    @Option(help: "Size of the upload test in MB.") var uploadMB = 256
    @Option var pings = 200

    func run() async throws {
        guard let base = URL(string: url) else { throw ValidationError("bad URL") }
        let token = try LegacyNames.environment("AISTUDIO_TOKEN")
            ?? ServerConfig.token(fromFile: expand(tokenFile), create: false).token
        let client = RemoteClient(endpoint: .init(baseURL: base, token: token),
                                  hello: ClientHello(role: .bench, name: "bench-remote", previewHz: 0))
        defer { client.close() }

        func ms(_ s: Double) -> String { String(format: "%.2f ms", s * 1000) }
        func pct(_ v: [Double], _ q: Double) -> Double { let s = v.sorted(); return s[min(s.count - 1, Int(Double(s.count - 1) * q + 0.5))] }

        var t = Date()
        let info = try await client.info()
        print("server \(info.serverVersion) on \(info.hostName) — \(info.chip), \(info.memory.total >> 30) GB, \(info.models.count) models")
        var infoTimes: [Double] = []
        for _ in 0..<10 { t = Date(); _ = try await client.info(); infoTimes.append(Date().timeIntervalSince(t)) }
        print("GET /v1/info            p50 \(ms(pct(infoTimes, 0.5)))  p99 \(ms(pct(infoTimes, 0.99)))")

        try await client.connect()
        var rtts: [Double] = []
        for _ in 0..<pings { rtts.append(try await client.ping()) }
        print("WebSocket ping          p50 \(ms(pct(rtts, 0.5)))  p99 \(ms(pct(rtts, 0.99)))")

        // Upload: random bytes (incompressible), then the same file again (should be skipped by hash).
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("aistudio-bench-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let blob = dir.appendingPathComponent("blob.bin")
        FileManager.default.createFile(atPath: blob.path, contents: nil)
        let h = try FileHandle(forWritingTo: blob)
        var rng = SystemRandomNumberGenerator()
        for _ in 0..<uploadMB { h.write(Data((0..<(1 << 20) / 8).flatMap { _ in withUnsafeBytes(of: rng.next()) { Array($0) } })) }
        try h.close()
        t = Date()
        let sha = try FileHasher.hashFile(blob)
        let hashSecs = Date().timeIntervalSince(t)
        let up = GraphTransport.Upload(name: GraphTransport.serverName(sha256: sha, for: blob), url: blob, bytes: uploadMB << 20)
        t = Date()
        try await client.upload(up)
        let upSecs = Date().timeIntervalSince(t)
        t = Date()
        try await client.upload(up)
        let againSecs = Date().timeIntervalSince(t)
        print(String(format: "hash %d MB              %.0f MB/s", uploadMB, Double(uploadMB) / hashSecs))
        print(String(format: "upload %d MB            %.0f MB/s (%.1f s); again (dedupe) %@", uploadMB,
                     Double(uploadMB) / upSecs, upSecs, ms(againSecs)))

        // A model-free run: an uploaded image through Load/Save, and a test-pattern video.
        let png = dir.appendingPathComponent("noise.png")
        try writeNoisePNG(png, side: 1536)
        var g = Graph(nodes: [
            GraphNode(id: "load", type: "LoadImage", inputs: ["path": .value(.string(png.path))]),
            GraphNode(id: "save", type: "SaveImage", inputs: ["image": .link(node: "load", output: "image"),
                                                               "filename_prefix": .value(.string("bench"))]),
            GraphNode(id: "pattern", type: "TestPattern", inputs: ["width": .value(.int(1280)), "height": .value(.int(720)),
                                                                     "seconds": .value(.double(6)), "fps": .value(.double(30))]),
            GraphNode(id: "video", type: "SaveVideo", inputs: ["video": .link(node: "pattern", output: "video"),
                                                                "filename_prefix": .value(.string("bench"))]),
        ])
        g.metadata = ["workflow": "bench-remote"]
        let registry = StandardNodes.makeRegistry()
        let hasher = FileHasher(cacheURL: nil)
        let prepared = try GraphTransport.prepare(g, schema: registry.schema, resolveInput: { expand($0) },
                                                  hash: hasher.sha256(of:))
        for u in prepared.uploads { try await client.upload(u) }
        var events = 0
        var outputs: [OutputDescriptor] = []
        var firstEvent: Double?
        let submitted = Date()
        let done = AsyncStream<WireEvent>.makeStream()
        let (run, _) = try await client.submit(SubmitRequest(requestID: UUID().uuidString, graph: prepared.graph,
                                                             metadataGraph: g, workflow: "bench-remote",
                                                             files: prepared.uploads.map(\.name))) { e in
            if firstEvent == nil { firstEvent = Date().timeIntervalSince(submitted) }
            events += 1
            if e.kind == .outputReady, let d = try? JSONDecoder().decode(OutputDescriptor.self, from: e.payload) { outputs.append(d) }
            if [.finished, .cancelled, .runFailed].contains(e.kind) { done.continuation.yield(e); done.continuation.finish() }
        }
        var final: WireEvent?
        for await e in done.stream { final = e }
        let runSecs = Date().timeIntervalSince(submitted)
        guard final?.kind == .finished else { throw ValidationError("run \(run) failed: \(final?.message ?? "?")") }
        print("run \(run)                 \(events) events, first after \(ms(firstEvent ?? 0)), done in \(ms(runSecs))")
        t = Date()
        var bytes = 0
        for o in outputs {
            try await client.download(run: run, name: o.name, to: dir.appendingPathComponent(o.name))
            bytes += o.bytes
        }
        let dlSecs = Date().timeIntervalSince(t)
        print(String(format: "download %d files        %.1f MB at %.0f MB/s", outputs.count, Double(bytes) / 1e6,
                     Double(bytes) / 1e6 / max(dlSecs, 1e-6)))
        if let img = outputs.first(where: { $0.kind == .image }),
           let text = StandardNodes.embeddedGraph(ImageArray.readPNGText(dir.appendingPathComponent(img.name))),
           let embedded = try? JSONDecoder().decode(Graph.self, from: Data(text.utf8)) {
            // Compare with the graph as JSON reads it back (6.0 and 6 decode the same way).
            let sent = try JSONDecoder().decode(Graph.self, from: JSONEncoder().encode(g))
            print("embedded graph          \(embedded == sent ? "is the original (local paths)" : "DIFFERS from the original")")
        }
    }

    func writeNoisePNG(_ url: URL, side: Int) throws {
        var bytes = [UInt8](repeating: 0, count: side * side * 4)
        var rng = SystemRandomNumberGenerator()
        for i in stride(from: 0, to: bytes.count, by: 4) {
            let v = rng.next() as UInt64
            bytes[i] = UInt8(truncatingIfNeeded: v); bytes[i + 1] = UInt8(truncatingIfNeeded: v >> 8)
            bytes[i + 2] = UInt8(truncatingIfNeeded: v >> 16); bytes[i + 3] = 255
        }
        let ctx = CGContext(data: &bytes, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        let image = ctx.makeImage()!
        let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
