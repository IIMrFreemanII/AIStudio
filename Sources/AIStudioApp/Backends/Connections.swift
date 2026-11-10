import Foundation
import GraphRuntime
import AIStudioProtocol
import NodesRemote
import Observation
import Security

/// A server the app can send work to. The token lives in the Keychain, not in this record.
struct ServerConnection: Codable, Identifiable, Hashable {
    enum Kind: String, Codable, CaseIterable {
        /// `aistudio-cli serve` on another Mac: runs whole graphs (a tab's run target).
        case aistudio
        /// The GPU server (server/gpu, on Modal or a GPU box): runs jobs for the Cloud GPU nodes.
        case gpu
        var label: String { self == .aistudio ? "Mac server" : "GPU server" }
    }

    var id: String = UUID().uuidString
    var name: String
    var kind: Kind = .aistudio
    var baseURL: String

    var url: URL? {
        guard var c = URLComponents(string: baseURL.trimmingCharacters(in: .whitespaces)),
              let scheme = c.scheme?.lowercased(), ["http", "https"].contains(scheme), c.host != nil else { return nil }
        if c.path.hasSuffix("/") { c.path.removeLast() }
        return c.url
    }

    /// Plain HTTP to a host that isn't loopback, a private LAN range or Tailscale: the token would cross the network
    /// unencrypted.
    var isInsecure: Bool {
        guard let u = url, u.scheme == "http", let host = u.host else { return false }
        return !Self.isPrivate(host)
    }

    static func isPrivate(_ host: String) -> Bool {
        if ["localhost", "127.0.0.1", "::1"].contains(host) || host.hasSuffix(".local") || host.hasSuffix(".ts.net") {
            return true
        }
        let p = host.split(separator: ".").compactMap { Int($0) }
        guard p.count == 4 else { return false }
        switch (p[0], p[1]) {
        case (10, _), (192, 168), (127, _): return true
        case (172, 16...31): return true
        case (100, 64...127): return true  // carrier-grade NAT range Tailscale uses
        default: return false
        }
    }
}

/// Tokens of server connections, kept in the login Keychain and read at most once per launch each.
enum ConnectionToken {
    static let service = "AIStudio.connection"
    private static var cache: [String: String] = [:]
    private static let lock = NSLock()

    static func read(_ id: String) -> String? {
        if let t = lock.withLock({ cache[id] }) { return t }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: id, kSecReturnData as String: true,
                                kSecMatchLimit as String: kSecMatchLimitOne]
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data,
              let t = String(data: d, encoding: .utf8) else {
            // Saved before the rename: move it to the new service name.
            guard let old = LegacyKeychain.take(service, account: id) else { return nil }
            save(old, for: id)
            return old
        }
        lock.withLock { cache[id] = t }
        return t
    }

    /// `keychain` false keeps it in memory only (automated runs must not touch the user's Keychain).
    static func save(_ token: String, for id: String, keychain: Bool = true) {
        if keychain { delete(id) } else { _ = lock.withLock { cache.removeValue(forKey: id) } }
        guard !token.isEmpty else { return }
        lock.withLock { cache[id] = token }
        guard keychain else { return }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: id, kSecValueData as String: Data(token.utf8)]
        SecItemAdd(q as CFDictionary, nil)
    }

    static func delete(_ id: String) {
        _ = lock.withLock { cache.removeValue(forKey: id) }
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: id]
        SecItemDelete(q as CFDictionary)
        LegacyKeychain.delete(service, account: id)
    }
}

/// The saved connections, their live status, and the backends made from them.
@MainActor @Observable
final class ConnectionStore {
    enum Status: Equatable {
        case unknown
        case checking
        case online(ServerInfo)
        case gpuOnline(GPUServerInfo)
        case offline(String)
    }

    private(set) var connections: [ServerConnection] = []
    var status: [String: Status] = [:]
    /// Whether the Servers panel checks every server when it appears (off for snapshots, which set statuses).
    @ObservationIgnored var checksOnAppear = true
    @ObservationIgnored private var backends: [String: RemoteMLXBackend] = [:]
    @ObservationIgnored let fileURL: URL?
    @ObservationIgnored let registry: NodeRegistry
    @ObservationIgnored let hasher: FileHasher

    /// `fileURL` nil: nothing is read or written (snapshots, smoke tests).
    init(fileURL: URL?, registry: NodeRegistry) {
        self.fileURL = fileURL
        self.registry = registry
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        hasher = FileHasher(cacheURL: fileURL == nil ? nil : caches.appendingPathComponent("AIStudio/file-hashes.json"))
        if let fileURL, let data = try? Data(contentsOf: fileURL),
           let list = try? JSONDecoder().decode([ServerConnection].self, from: data) {
            connections = list
        }
        publishEndpoints()
    }

    static var defaultFileURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AIStudio/connections.json")
    }

    func connection(_ id: String) -> ServerConnection? { connections.first { $0.id == id } }

    /// Mac servers: run targets for whole tabs.
    var macServers: [ServerConnection] { connections.filter { $0.kind == .aistudio } }
    /// GPU servers: where the Cloud GPU nodes send their jobs.
    var gpuServers: [ServerConnection] { connections.filter { $0.kind == .gpu } }

    /// Tells the remote nodes which GPU servers exist (tokens are read from the Keychain when a node runs).
    private func publishEndpoints() {
        let servers = gpuServers.compactMap { c in c.url.map { (c.id, c.name, $0) } }
        RemoteEndpoints.install { id in
            guard let s = id.isEmpty ? servers.first : servers.first(where: { $0.0 == id }),
                  let token = ConnectionToken.read(s.0) else { return nil }
            return RemoteEndpoints.Endpoint(id: s.0, name: s.1, baseURL: s.2, token: token)
        }
    }

    func save(_ c: ServerConnection, token: String?) {
        if let i = connections.firstIndex(where: { $0.id == c.id }) { connections[i] = c } else { connections.append(c) }
        if let token { ConnectionToken.save(token, for: c.id, keychain: fileURL != nil) }
        backends.removeValue(forKey: c.id)?.client.close()
        status[c.id] = .unknown
        persist()
        publishEndpoints()
    }

    func remove(_ id: String) {
        connections.removeAll { $0.id == id }
        if fileURL != nil { ConnectionToken.delete(id) }
        backends.removeValue(forKey: id)?.client.close()
        status.removeValue(forKey: id)
        persist()
        publishEndpoints()
    }

    private func persist() {
        guard let fileURL else { return }
        try? FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? JSONEncoder().encode(connections).write(to: fileURL, options: .atomic)
    }

    /// The backend for a connection (made on first use; one per connection for the app's lifetime).
    func backend(_ id: String) -> RemoteMLXBackend? {
        if let b = backends[id] { return b }
        guard let c = connection(id), c.kind == .aistudio, let url = c.url, let token = ConnectionToken.read(id) else {
            return nil
        }
        let b = RemoteMLXBackend(id: id, name: c.name, endpoint: .init(baseURL: url, token: token), registry: registry,
                                 hasher: hasher)
        backends[id] = b
        return b
    }

    /// Asks the server for `/v1/info` with the saved (or given) token.
    func test(_ id: String, token: String? = nil) async {
        guard let c = connection(id), let url = c.url else {
            status[id] = .offline("Not a valid http(s) URL")
            return
        }
        guard let token = token ?? ConnectionToken.read(id) else {
            status[id] = .offline("No token saved")
            return
        }
        status[id] = .checking
        do {
            status[id] = try await Self.check(kind: c.kind, url: url, token: token)
        } catch {
            status[id] = .offline(Self.describe(error))
        }
    }

    /// Asks a server for `/v1/info` (a GPU server answers from its front desk: no GPU starts).
    static func check(kind: ServerConnection.Kind, url: URL, token: String) async throws -> Status {
        switch kind {
        case .aistudio:
            let client = RemoteClient(endpoint: .init(baseURL: url, token: token), hello: ClientHello(role: .app, name: "test"))
            defer { client.close() }
            return .online(try await client.info())
        case .gpu:
            return .gpuOnline(try await GPUJobClient(baseURL: url, token: token).info())
        }
    }

    func refreshAll() async {
        for c in connections { await test(c.id) }
    }

    static func describe(_ error: Error) -> String {
        if let e = error as? URLError {
            switch e.code {
            case .cannotConnectToHost: return "Connection refused: is the server running, and on this address?"
            case .timedOut: return "Timed out: check the address and your tunnel or Tailscale."
            case .cannotFindHost: return "Unknown host"
            case .notConnectedToInternet: return "Offline"
            case .appTransportSecurityRequiresSecureConnection:
                // macOS blocks plain http from apps to Tailscale (100.x, *.ts.net) and public addresses.
                return "macOS only allows https to this address. With Tailscale, publish the server with `tailscale serve` "
                    + "and use https://<mac>.<tailnet>.ts.net; or use an SSH tunnel (http://127.0.0.1:…) or a LAN address."
            default: return e.localizedDescription
            }
        }
        return String(describing: error)
    }

    /// The server's models (for showing what a remote target can run).
    func info(_ id: String) -> ServerInfo? {
        if case .online(let i)? = status[id] { return i }
        return nil
    }
}
