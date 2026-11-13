import Foundation
import Security

/// The app's tokens (server connections, the search key, the Hugging Face token, the Civitai key) in a file only the
/// user can read: `~/Library/Application Support/AIStudio/tokens.json`, mode 0600 (a folder it makes is 0700). The user enters
/// them; the app never fills them.
///
/// They used to live in the login Keychain, but macOS trusts an app signed without a team ID per build (its code
/// hash), so every rebuild asked for Keychain access again. A token still in the Keychain moves here the first time
/// it's read (one last prompt) and is deleted there. Thread-safe: downloads read tokens off the main thread.
enum TokenFile {
    private static let lock = NSLock()
    private static var tokens: [String: String]?
    /// Where they're kept; tests point it elsewhere.
    nonisolated(unsafe) static var url: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("AIStudio/tokens.json")

    static func get(_ key: String) -> String? {
        lock.withLock { load()[key] }
    }

    /// Saves (or with nil / "" removes) a token. `persist` false keeps it in memory only (automated runs and tests
    /// must not touch the user's file).
    static func set(_ key: String, _ value: String?, persist: Bool = true) {
        lock.withLock {
            var all = load()
            if let value, !value.isEmpty { all[key] = value } else { all.removeValue(forKey: key) }
            tokens = all
            if persist { write(all) }
        }
    }

    /// The token, else one still in the Keychain (moved here, persisted only when `persist`), else nil.
    static func get(_ key: String, keychain: KeychainItem, persist: Bool = true) -> String? {
        if let t = get(key) { return t }
        // Left in the Keychain unless it can be saved here.
        guard let t = keychain.take(delete: persist) else { return nil }
        set(key, t, persist: persist)
        return t
    }

    /// Forgets what was read (tests).
    static func reset() { lock.withLock { tokens = nil } }

    private static func load() -> [String: String] {
        if let tokens { return tokens }
        let read = (try? Data(contentsOf: url)).flatMap { try? JSONDecoder().decode([String: String].self, from: $0) } ?? [:]
        tokens = read
        return read
    }

    private static func write(_ all: [String: String]) {
        let fm = FileManager.default
        let dir = url.deletingLastPathComponent()
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        if all.isEmpty { try? fm.removeItem(at: url); return }
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? enc.encode(all) else { return }
        // Written private from the start, then moved over the old file.
        let tmp = dir.appendingPathComponent(".tokens-\(UUID().uuidString).json")
        guard fm.createFile(atPath: tmp.path, contents: data, attributes: [.posixPermissions: 0o600]) else { return }
        if rename(tmp.path, url.path) != 0 { try? fm.removeItem(at: tmp) }
    }
}

/// A token's old place in the login Keychain (and its name from before the rename), to move it to `TokenFile`.
struct KeychainItem {
    let service: String
    var account: String?

    private var query: [String: Any] {
        var q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service]
        if let account { q[kSecAttrAccount as String] = account }
        return q
    }

    /// Whether it's still there, without reading it (never asks).
    var exists: Bool {
        var q = query
        q[kSecReturnAttributes as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        return SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess || LegacyKeychain.exists(service, account: account)
    }

    /// Its text, deleted from the Keychain when `delete` (macOS may ask once).
    func take(delete: Bool = true) -> String? {
        guard exists else { return nil }
        var q = query
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        if SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data,
           let text = String(data: d, encoding: .utf8) {
            if delete { self.delete() }
            return text
        }
        return delete ? LegacyKeychain.take(service, account: account) : nil
    }

    func delete() {
        SecItemDelete(query as CFDictionary)
        LegacyKeychain.delete(service, account: account)
    }
}
