import Foundation
import Security

extension SmokeTest {
    /// `--smoke-tokens`: the token file in a temporary folder (private permissions, removal, moving a token out of
    /// the Keychain). The Keychain part uses a throwaway item of its own, never the user's tokens.
    static func runTokens() -> Bool {
        func say(_ s: String) { print("[smoke-tokens] \(s)"); fflush(stdout) }
        var ok = true
        func check(_ cond: Bool, _ what: String) {
            if cond { say("ok: \(what)") } else { say("FAIL: \(what)"); ok = false }
        }
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent("smoke-tokens-\(UUID().uuidString.prefix(8))/AIStudio")
        let saved = TokenFile.url
        TokenFile.url = dir.appendingPathComponent("tokens.json")
        TokenFile.reset()
        defer {
            TokenFile.url = saved
            TokenFile.reset()
            try? fm.removeItem(at: dir.deletingLastPathComponent())
        }
        func mode(_ url: URL) -> Int { ((try? fm.attributesOfItem(atPath: url.path))?[.posixPermissions] as? NSNumber)?.intValue ?? -1 }

        TokenFile.set("connection:a", "secret-a")
        TokenFile.set("huggingface", "hf_x")
        check(mode(TokenFile.url) == 0o600, "the file is private (\(String(mode(TokenFile.url), radix: 8)))")
        check(mode(dir) == 0o700, "its folder is private (\(String(mode(dir), radix: 8)))")
        TokenFile.reset()
        check(TokenFile.get("connection:a") == "secret-a" && TokenFile.get("huggingface") == "hf_x", "tokens are read back")
        TokenFile.set("connection:a", "")
        TokenFile.reset()
        check(TokenFile.get("connection:a") == nil && TokenFile.get("huggingface") == "hf_x", "a removed token is gone")
        TokenFile.set("huggingface", nil)
        check(!fm.fileExists(atPath: TokenFile.url.path), "the file goes when it's empty")
        TokenFile.set("civitai", "kept in memory", persist: false)
        check(TokenFile.get("civitai") == "kept in memory" && !fm.fileExists(atPath: TokenFile.url.path),
              "persist false keeps a token in memory only")
        TokenFile.set("civitai", nil, persist: false)

        // A token still in the Keychain moves to the file and leaves the Keychain.
        let item = KeychainItem(service: "AIStudio.smoke-\(UUID().uuidString.prefix(8))", account: "smoke")
        check(!item.exists && TokenFile.get("connection:smoke", keychain: item) == nil, "no token anywhere reads nil")
        SecItemAdd([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: item.service,
                    kSecAttrAccount as String: "smoke", kSecValueData as String: Data("from-keychain".utf8)] as CFDictionary, nil)
        check(item.exists, "the throwaway Keychain item was made")
        let kept = TokenFile.get("connection:smoke", keychain: item, persist: false)
        check(kept == "from-keychain" && item.exists, "a run that can't save leaves it in the Keychain")
        TokenFile.set("connection:smoke", nil, persist: false)
        let moved = TokenFile.get("connection:smoke", keychain: item)
        check(moved == "from-keychain" && !item.exists, "it moves out of the Keychain")
        TokenFile.reset()
        check(TokenFile.get("connection:smoke") == "from-keychain", "and is in the file")
        item.delete()

        say(ok ? "PASS" : "FAILED")
        return ok
    }
}
