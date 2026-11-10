import Foundation

/// The GPU servers remote nodes may call, installed by the app (from its Servers panel and the Keychain) or the CLI.
/// Nodes refer to a server by id; an empty id means the first one. Lock-protected: nodes run off the main actor.
public enum RemoteEndpoints {
    public struct Endpoint: Sendable, Equatable {
        public var id: String
        public var name: String
        public var baseURL: URL
        public var token: String
        public init(id: String, name: String, baseURL: URL, token: String) {
            self.id = id
            self.name = name
            self.baseURL = baseURL
            self.token = token
        }
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var lookup: @Sendable (String) -> Endpoint? = { _ in nil }

    /// A fixed list of servers (the CLI, tests).
    public static func install(_ endpoints: [Endpoint]) {
        install { id in id.isEmpty ? endpoints.first : endpoints.first { $0.id == id } }
    }

    /// A lookup the app keeps current; it may read tokens lazily (only when a node runs).
    public static func install(lookup: @escaping @Sendable (String) -> Endpoint?) { lock.withLock { self.lookup = lookup } }

    /// The server with this id; the first one for an empty id.
    public static func resolve(_ id: String) -> Endpoint? { lock.withLock { lookup }(id) }
}
