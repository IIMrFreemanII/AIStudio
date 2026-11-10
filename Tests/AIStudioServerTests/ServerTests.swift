import CryptoKit
import Foundation
import GraphRuntime
import HTTPTypes
import Hummingbird
import HummingbirdTesting
import AIStudioProtocol
import NIOCore
import Testing
@testable import AIStudioServer

let testToken = String(repeating: "t", count: 40)

func makeServer() throws -> RemoteServer {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("server-\(UUID().uuidString)")
    let models = root.appendingPathComponent("models")
    try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
    return RemoteServer(config: ServerConfig(port: 0, token: testToken, dataDirectory: root.appendingPathComponent("data"),
                                             modelsDirectory: models))
}

let bearer: HTTPFields = [.authorization: "Bearer \(testToken)"]

func buffer(_ d: Data) -> ByteBuffer { ByteBuffer(bytes: d) }
func string(_ b: ByteBuffer) -> String { String(buffer: b) }

// The server installs a process-wide path sandbox, so these run one at a time.
@Suite(.serialized) struct ServerHTTPTests {
    @Test func everyAPIRouteNeedsTheToken() async throws {
        let app = try makeServer().makeApplication()
        try await app.test(.router) { client in
            #expect(try await client.execute(uri: "/v1/health", method: .get).status == .ok)
            #expect(try await client.execute(uri: "/v1/info", method: .get).status == .unauthorized)
            #expect(try await client.execute(uri: "/v1/info", method: .get,
                                             headers: [.authorization: "Bearer wrong-token-xxxxxxxx"]).status == .unauthorized)
            let info = try await client.execute(uri: "/v1/info", method: .get, headers: bearer)
            #expect(info.status == .ok)
            let decoded = try JSONDecoder().decode(ServerInfo.self, from: Data(info.body.readableBytesView))
            #expect(decoded.protocolVersion == RemoteProtocol.version && decoded.nodeTypes.contains("SaveImage"))
            #expect(try await client.execute(uri: "/v1/state", method: .get).status == .unauthorized)
            #expect(try await client.execute(uri: "/v1/files", method: .delete).status == .unauthorized)
        }
    }

    @Test func securityHeadersOnEveryResponse() async throws {
        let app = try makeServer().makeApplication()
        try await app.test(.router) { client in
            for uri in ["/v1/health", "/"] {
                let r = try await client.execute(uri: uri, method: .get)
                #expect(r.headers[.contentSecurityPolicy]?.contains("script-src 'self'") == true)
                #expect(r.headers[.xContentTypeOptions] == "nosniff")
                #expect(r.headers[HTTPField.Name("X-Frame-Options")!] == "DENY")
            }
        }
    }

    @Test func dashboardSessionCookieAndCSRF() async throws {
        let app = try makeServer().makeApplication()
        try await app.test(.router) { client in
            let wrong = try await client.execute(uri: "/v1/session", method: .post,
                                                 body: buffer(Data(#"{"token":"nope-nope-nope-nope"}"#.utf8)))
            #expect(wrong.status == .unauthorized)
            let ok = try await client.execute(uri: "/v1/session", method: .post,
                                              body: buffer(Data("{\"token\":\"\(testToken)\"}".utf8)))
            #expect(ok.status == .noContent)
            let cookies = ok.headers[values: .setCookie]
            let session = try #require(cookies.first { $0.hasPrefix("mlxc_session=") })
            let csrfCookie = try #require(cookies.first { $0.hasPrefix("mlxc_csrf=") })
            #expect(session.contains("HttpOnly") && session.contains("SameSite=Strict"))
            #expect(!csrfCookie.contains("HttpOnly"))
            func value(_ c: String) -> String { String(c.split(separator: ";")[0].split(separator: "=", maxSplits: 1)[1]) }
            let cookieHeader = "mlxc_session=\(value(session)); mlxc_csrf=\(value(csrfCookie))"
            // Reading with the cookie works.
            #expect(try await client.execute(uri: "/v1/state", method: .get, headers: [.cookie: cookieHeader]).status == .ok)
            // Changing something needs the CSRF header as well.
            let body = buffer(Data(#"{"previewHz":2}"#.utf8))
            #expect(try await client.execute(uri: "/v1/settings", method: .post, headers: [.cookie: cookieHeader],
                                             body: body).status == .forbidden)
            var withCSRF: HTTPFields = [.cookie: cookieHeader]
            withCSRF[Auth.csrfHeader] = value(csrfCookie)
            #expect(try await client.execute(uri: "/v1/settings", method: .post, headers: withCSRF, body: body).status == .ok)
            // A bearer client needs no CSRF header.
            #expect(try await client.execute(uri: "/v1/settings", method: .post, headers: bearer, body: body).status == .ok)
        }
    }

    /// Behind `tailscale serve` (or another TLS-terminating proxy) the session cookies are `Secure`.
    @Test func cookiesAreSecureBehindAnHTTPSProxy() async throws {
        let app = try makeServer().makeApplication()
        try await app.test(.router) { client in
            let login = buffer(Data("{\"token\":\"\(testToken)\"}".utf8))
            let direct = try await client.execute(uri: "/v1/session", method: .post, body: login)
            #expect(direct.headers[values: .setCookie].allSatisfy { !$0.contains("Secure") })
            let proxied = try await client.execute(uri: "/v1/session", method: .post,
                                                   headers: [RemoteServer.forwardedProto: "https"], body: login)
            let cookies = proxied.headers[values: .setCookie]
            #expect(cookies.count == 2 && cookies.allSatisfy { $0.contains("Secure") })
        }
    }

    @Test func uploadsAreContentAddressedVerifiedAndResumable() async throws {
        let app = try makeServer().makeApplication()
        let bytes = Data((0..<20).map { UInt8($0) })
        let name = FileHasher.hex(SHA256.hash(data: bytes)) + ".bin"
        try await app.test(.router) { client in
            var h = bearer
            #expect(try await client.execute(uri: "/v1/files/\(name)", method: .head, headers: h).status == .notFound)
            // First half, then the rest.
            h[.contentRange] = "bytes 0-9/20"
            let first = try await client.execute(uri: "/v1/files/\(name)", method: .put, headers: h, body: buffer(bytes.prefix(10)))
            #expect(first.status == .accepted && first.headers[RemoteServer.uploadOffset] == "10")
            let head = try await client.execute(uri: "/v1/files/\(name)", method: .head, headers: bearer)
            #expect(head.status == .notFound && head.headers[RemoteServer.uploadOffset] == "10")
            // Resuming from the wrong offset is refused with the right one.
            h[.contentRange] = "bytes 5-19/20"
            let wrongOffset = try await client.execute(uri: "/v1/files/\(name)", method: .put, headers: h, body: buffer(bytes.suffix(15)))
            #expect(wrongOffset.status == .conflict && wrongOffset.headers[RemoteServer.uploadOffset] == "10")
            h[.contentRange] = "bytes 10-19/20"
            let rest = try await client.execute(uri: "/v1/files/\(name)", method: .put, headers: h, body: buffer(bytes.suffix(10)))
            #expect(rest.status == .created)
            #expect(try await client.execute(uri: "/v1/files/\(name)", method: .head, headers: bearer).status == .ok)
            // Bytes that don't match the name are rejected; so are names that aren't hashes.
            let lie = FileHasher.hex(SHA256.hash(data: Data("other".utf8))) + ".bin"
            #expect(try await client.execute(uri: "/v1/files/\(lie)", method: .put, headers: bearer,
                                             body: buffer(Data("not other".utf8))).status == .badRequest)
            #expect(try await client.execute(uri: "/v1/files/..%2F..%2Fx", method: .put, headers: bearer,
                                             body: buffer(Data("x".utf8))).status == .badRequest)
        }
    }

    @Test func dashboardFilesAndTraversal() async throws {
        let app = try makeServer().makeApplication()
        try await app.test(.router) { client in
            let index = try await client.execute(uri: "/", method: .get)
            #expect(index.status == .ok && string(index.body).contains("<div id=\"app\">"))
            #expect(index.headers[.cacheControl] == "no-cache")
            #expect(try await client.execute(uri: "/assets/../../../../etc/passwd", method: .get).status == .notFound)
            #expect(try await client.execute(uri: "/v1/runs/1/outputs/..%2F..%2Fsecret", method: .get, headers: bearer).status == .notFound)
        }
    }

    @Test func runIsRejectedWhenAModelIsMissing() async throws {
        let server = try makeServer()
        let g = Graph(nodes: [
            GraphNode(id: "m", type: "LoadModel", inputs: ["path": .value(.string("not-installed"))]),
            GraphNode(id: "p", type: "TestPattern"),
            GraphNode(id: "s", type: "SaveVideo", inputs: ["video": .link(node: "p", output: "video")]),
        ])
        await #expect(throws: (any Error).self) {
            _ = try await server.scheduler.submit(SubmitRequest(requestID: "r", graph: g, metadataGraph: nil, workflow: "",
                                                                files: []), session: "s", client: "test")
        }
        let bad = Graph(nodes: [GraphNode(id: "x", type: "NoSuchNode")])
        await #expect(throws: (any Error).self) {
            _ = try await server.scheduler.submit(SubmitRequest(requestID: "r", graph: bad, metadataGraph: nil, workflow: "",
                                                                files: []), session: "s", client: "test")
        }
    }
}

struct RangeParsingTests {
    @Test func dashboardOriginCheck() {
        let none = HTTPFields()
        // Direct, and through tailscale serve (which keeps Host).
        #expect(RemoteServer.isSameOrigin("http://127.0.0.1:8188", authority: "127.0.0.1:8188", headers: none))
        #expect(RemoteServer.isSameOrigin("https://studio.tail1234.ts.net", authority: "studio.tail1234.ts.net", headers: none))
        // A proxy that replaces Host but forwards the original.
        let forwarded: HTTPFields = [RemoteServer.forwardedHost: "studio.example.com"]
        #expect(RemoteServer.isSameOrigin("https://studio.example.com", authority: "127.0.0.1:8188", headers: forwarded))
        // Another site, or no host at all, is refused.
        #expect(!RemoteServer.isSameOrigin("https://evil.example", authority: "127.0.0.1:8188", headers: forwarded))
        #expect(!RemoteServer.isSameOrigin("https://evil.example", authority: nil, headers: none))
        #expect(!RemoteServer.isSameOrigin("null", authority: "127.0.0.1:8188", headers: none))
    }

    @Test func ranges() {
        #expect(RemoteServer.parseRange("bytes=0-99", size: 1000)! == (0, 99))
        #expect(RemoteServer.parseRange("bytes=900-", size: 1000)! == (900, 999))
        #expect(RemoteServer.parseRange("bytes=-100", size: 1000)! == (900, 999))
        #expect(RemoteServer.parseRange("bytes=500-5000", size: 1000)! == (500, 999))
        #expect(RemoteServer.parseRange("bytes=1000-", size: 1000) == nil)
        #expect(RemoteServer.parseRange("bytes=0-1,5-6", size: 1000) == nil)
        #expect(RemoteServer.parseRange("items=0-1", size: 1000) == nil)
        #expect(RemoteServer.parseContentRange("bytes 10-19/20")! == (10, 19, 20))
        #expect(RemoteServer.parseContentRange("bytes 0-9/*")!.total == nil)
        #expect(RemoteServer.parseContentRange("bytes 9-0/20") == nil)
    }
}

struct HubTests {
    @Test func recordsGoToSubscribersAndReplayAfterReconnect() {
        let hub = Hub()
        let app = Hub.Client(session: "app", address: "x")
        let other = Hub.Client(session: "other", address: "y")
        let dash = Hub.Client(session: "dash", address: "z")
        dash.role = .dashboard
        [app, other, dash].forEach(hub.add)
        hub.subscribe("app", run: 1, after: .max)
        for step in 1...5 { hub.publish(WireEvent(kind: .progress, run: 1, step: UInt16(step), total: 5)) }
        let appRecords = (try? WireEvent.decodeAll(app.pending)) ?? []
        #expect(appRecords.map(\.seq) == [1, 2, 3, 4, 5])
        #expect(other.pending.isEmpty)
        #expect(((try? WireEvent.decodeAll(dash.pending)) ?? []).count == 5)
        // A client re-attaching after seq 3 gets 4 and 5 only.
        let back = Hub.Client(session: "back", address: "x")
        hub.add(back)
        hub.subscribe("back", run: 1, after: 3)
        #expect(((try? WireEvent.decodeAll(back.pending)) ?? []).map(\.seq) == [4, 5])
    }

    @Test func previewsAreLatestWinsPerClient() {
        let hub = Hub()
        let c = Hub.Client(session: "c", address: "x")
        hub.add(c)
        hub.subscribe("c", run: 2, after: .max)
        #expect(hub.previewDemand(run: 2).hz == 4)
        for step in 1...30 { hub.publishPreview(run: 2, node: 0, step: step, total: 30, jpeg: Data(repeating: UInt8(step), count: 100)) }
        #expect(c.previews.count == 1)
        let kept = try? WireEvent.decodeAll(c.previews[2]!)
        #expect(kept?.first?.step == 30)
        c.previewHz = 0
        #expect(hub.previewDemand(run: 2).hz == 0)
    }
}

struct DashboardBundleTests {
    /// The committed dashboard bundle must be built from the current sources (`scripts/build-dashboard.sh`).
    @Test func bundleMatchesSources() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let web = root.appendingPathComponent("web/dashboard")
        let fm = FileManager.default
        var files: [String] = []
        for entry in ["index.html", "package.json", "vite.config.ts", "src", "public"] {
            let url = web.appendingPathComponent(entry)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: url.path, isDirectory: &isDir) else { continue }
            if isDir.boolValue {
                let e = fm.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey])!
                for case let f as URL in e where (try? f.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
                    files.append(String(f.standardizedFileURL.path.dropFirst(web.standardizedFileURL.path.count + 1)))
                }
            } else {
                files.append(entry)
            }
        }
        var h = SHA256()
        for f in files.filter({ !$0.hasSuffix(".test.ts") }).sorted() {
            h.update(data: Data(f.utf8))
            h.update(data: Data([0]))
            h.update(data: try Data(contentsOf: web.appendingPathComponent(f)))
            h.update(data: Data([0]))
        }
        let info = try JSONSerialization.jsonObject(with: Data(contentsOf: root
            .appendingPathComponent("Sources/AIStudioServer/Resources/dashboard/build-info.json"))) as? [String: Any]
        #expect(info?["sourceHash"] as? String == FileHasher.hex(h.finalize()),
                "the dashboard bundle is stale: run scripts/build-dashboard.sh")
    }
}
