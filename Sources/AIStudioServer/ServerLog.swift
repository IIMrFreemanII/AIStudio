import Foundation
import AIStudioProtocol

/// The server's log: printed to stderr (launchd sends it to the log file) and kept in a ring for the dashboard.
public final class ServerLog: @unchecked Sendable {
    public static let shared = ServerLog()

    private var lines: [LogLine] = []
    private let capacity = 500
    private let lock = NSLock()
    /// Called for every new line (the hub forwards it to dashboards).
    var onLine: (@Sendable (LogLine) -> Void)?

    public func log(_ level: LogLine.Level, _ message: String) {
        let line = LogLine(time: Date().timeIntervalSince1970, level: level, message: message)
        let stamp = ISO8601DateFormatter.string(from: Date(), timeZone: .current,
                                                formatOptions: [.withInternetDateTime])
        FileHandle.standardError.write(Data("\(stamp) [\(level.rawValue)] \(message)\n".utf8))
        let hook: (@Sendable (LogLine) -> Void)? = lock.withLock {
            lines.append(line)
            if lines.count > capacity { lines.removeFirst(lines.count - capacity) }
            return onLine
        }
        hook?(line)
    }

    public func info(_ m: String) { log(.info, m) }
    public func warning(_ m: String) { log(.warning, m) }
    public func error(_ m: String) { log(.error, m) }

    var recent: [LogLine] { lock.withLock { lines } }
}
