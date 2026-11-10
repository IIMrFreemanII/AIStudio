import Foundation
import MLXDiffusionCore
import Security

/// Carries an install from before the rename (MLXComfyUI) over to AIStudio: the folders, then the settings. Runs
/// before anything reads either. Keychain tokens move lazily, when they're first read (see `LegacyKeychain`).
enum LegacyMigration {
    static let doneKey = "migratedFromMLXComfyUI"

    static func run() {
        LegacyNames.migrateFolders()
        migrateDefaults()
    }

    /// Copies the settings the old app saved (keys this app hasn't set yet), with paths under the old folders
    /// pointed at the new ones.
    static func migrateDefaults() {
        let defaults = UserDefaults.standard
        guard !defaults.bool(forKey: doneKey) else { return }
        let own = Bundle.main.bundleIdentifier ?? ProcessInfo.processInfo.processName
        let current = defaults.persistentDomain(forName: own) ?? [:]
        for domain in LegacyNames.defaultsDomains where domain != own {
            for (key, value) in defaults.persistentDomain(forName: domain) ?? [:] where current[key] == nil {
                defaults.set(upgraded(value), forKey: key)
            }
        }
        defaults.set(true, forKey: doneKey)
    }

    private static func upgraded(_ value: Any) -> Any {
        switch value {
        case let s as String: return LegacyNames.upgradedPath(s)
        case let a as [String]: return a.map(LegacyNames.upgradedPath)
        case let d as Data:
            // JSON-encoded settings (the added model folders) hold paths too, with `/` escaped as `\/`.
            guard var text = String(data: d, encoding: .utf8) else { return d }
            for (old, new) in LegacyNames.folderMoves {
                text = text.replacingOccurrences(of: old.path + "/", with: new.path + "/")
                    .replacingOccurrences(of: old.path.replacingOccurrences(of: "/", with: "\\/") + "\\/",
                                          with: new.path.replacingOccurrences(of: "/", with: "\\/") + "\\/")
            }
            return Data(text.utf8)
        default: return value
        }
    }
}

/// Keychain items saved under the old service names (`MLXComfyUI.huggingface`…).
enum LegacyKeychain {
    /// `AIStudio.x` → `MLXComfyUI.x`.
    static func service(for service: String) -> String? {
        guard let dot = service.firstIndex(of: ".") else { return nil }
        return LegacyNames.keychainPrefix + service[service.index(after: dot)...]
    }

    private static func query(_ service: String, _ account: String?) -> [String: Any] {
        var q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service]
        if let account { q[kSecAttrAccount as String] = account }
        return q
    }

    /// Whether an old item exists, without reading it (no Keychain prompt).
    static func exists(_ service: String, account: String? = nil) -> Bool {
        guard let old = self.service(for: service) else { return false }
        var q = query(old, account)
        q[kSecReturnAttributes as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        return SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess
    }

    /// The old item's text, removed from the Keychain; the caller saves it under the new name.
    static func take(_ service: String, account: String? = nil) -> String? {
        guard let old = self.service(for: service) else { return nil }
        var q = query(old, account)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var out: AnyObject?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data,
              let text = String(data: d, encoding: .utf8) else { return nil }
        delete(service, account: account)
        return text
    }

    static func delete(_ service: String, account: String? = nil) {
        guard let old = self.service(for: service) else { return }
        SecItemDelete(query(old, account) as CFDictionary)
    }
}
