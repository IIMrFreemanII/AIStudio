import Foundation
import MLXComfyProtocol
import MLXDiffusionCore
import Security

/// How `mlxcomfy-cli serve` runs.
public struct ServerConfig: Sendable {
    public var host: String
    public var port: Int
    /// The bearer token every client must send.
    public var token: String
    /// Root of the server's own files: `inputs/` (uploads), `runs/<id>/` (outputs).
    public var dataDirectory: URL
    public var modelsDirectory: URL
    /// Fraction of RAM models may occupy before the least recently used is unloaded.
    public var budgetFraction: Double
    public var previewHz: Double
    public var previewMaxSide: Int
    /// Finished runs whose folders are kept (older ones are deleted).
    public var keepRuns: Int
    /// Serve the web dashboard at `/`.
    public var dashboard: Bool

    public init(host: String = "127.0.0.1", port: Int = RemoteProtocol.defaultPort, token: String,
                dataDirectory: URL = ServerConfig.defaultDataDirectory, modelsDirectory: URL = ModelPaths.directory,
                budgetFraction: Double = 0.8, previewHz: Double = 4, previewMaxSide: Int = 512, keepRuns: Int = 200,
                dashboard: Bool = true) {
        self.host = host
        self.port = port
        self.token = token
        self.dataDirectory = dataDirectory
        self.modelsDirectory = modelsDirectory
        self.budgetFraction = budgetFraction
        self.previewHz = previewHz
        self.previewMaxSide = previewMaxSide
        self.keepRuns = keepRuns
        self.dashboard = dashboard
    }

    /// `~/Library/Application Support/MLXComfyUI/server`.
    public static var defaultDataDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("MLXComfyUI/server")
    }

    public var inputsDirectory: URL { dataDirectory.appendingPathComponent("inputs") }
    public var runsDirectory: URL { dataDirectory.appendingPathComponent("runs") }

    public var isLoopback: Bool { ["127.0.0.1", "localhost", "::1"].contains(host) }

    /// 256 random bits as 64 hex characters.
    public static func generateToken() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        return FileHasher.hex(bytes)
    }

    /// Reads a token file (first line, trimmed); creates it with a new token (mode 600) when `create` and missing.
    public static func token(fromFile url: URL, create: Bool) throws -> (token: String, created: Bool) {
        if let text = try? String(contentsOf: url, encoding: .utf8) {
            let t = text.split(whereSeparator: \.isNewline).first.map(String.init)?
                .trimmingCharacters(in: .whitespaces) ?? ""
            guard t.count >= 16 else { throw ConfigError.weakToken(url.path) }
            return (t, false)
        }
        guard create else { throw ConfigError.missingToken(url.path) }
        let t = generateToken()
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        FileManager.default.createFile(atPath: url.path, contents: Data((t + "\n").utf8),
                                       attributes: [.posixPermissions: 0o600])
        return (t, true)
    }

    public enum ConfigError: Error, CustomStringConvertible {
        case missingToken(String)
        case weakToken(String)
        public var description: String {
            switch self {
            case .missingToken(let p): return "token file not found: \(p)"
            case .weakToken(let p): return "the token in \(p) is shorter than 16 characters"
            }
        }
    }
}
