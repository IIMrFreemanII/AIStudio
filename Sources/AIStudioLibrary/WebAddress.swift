import Foundation

/// What the side panel's address field makes of typed text.
public enum WebAddress {
    /// A web address for `text`: as typed when it names http(s), file or about; `localhost:3000`, `127.0.0.1` and other
    /// local hosts over http; anything else that looks like a host (`example.com/docs`) over https. Nil for text with
    /// spaces or without a dot that isn't a local host (not an address).
    public static func url(from text: String) -> URL? {
        let s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !s.isEmpty, !s.contains(where: \.isWhitespace) else { return nil }
        if let scheme = s.range(of: "://").map({ s[..<$0.lowerBound].lowercased() }) {
            guard ["http", "https", "file"].contains(scheme), let u = URL(string: s), u.host != nil || scheme == "file"
            else { return nil }
            return u
        }
        if s.lowercased().hasPrefix("about:") { return URL(string: s) }
        let host = String(s.prefix { $0 != "/" && $0 != ":" && $0 != "?" && $0 != "#" }).lowercased()
        if isLocal(host) { return URL(string: "http://" + s) }
        guard host.contains("."), !host.hasPrefix("."), !host.hasSuffix(".") else { return nil }
        return URL(string: "https://" + s)
    }

    /// localhost, *.localhost, *.local, 127.x / 0.0.0.0 / 10.x / 192.168.x addresses.
    public static func isLocal(_ host: String) -> Bool {
        if host == "localhost" || host.hasSuffix(".localhost") || host.hasSuffix(".local") || host == "0.0.0.0" { return true }
        let parts = host.split(separator: ".")
        guard parts.count == 4, parts.allSatisfy({ UInt8($0) != nil }) else { return false }
        return parts[0] == "127" || parts[0] == "10" || (parts[0] == "192" && parts[1] == "168")
    }

    /// The address as the field shows it: without `https://` and a lone trailing slash.
    public static func display(_ url: URL) -> String {
        var s = url.absoluteString
        if url.scheme == "https", s.hasPrefix("https://") { s.removeFirst(8) }
        if s.hasSuffix("/"), s.filter({ $0 == "/" }).count == 1 { s.removeLast() }
        return s
    }
}
