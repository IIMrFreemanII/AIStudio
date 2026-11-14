import AppKit
import CoreGraphics
import Agent

/// Running GUI apps and their on-screen windows.
enum AppList {
    static func running() -> [UIApp] {
        let front = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let windows = onScreenWindows()
        let apps = NSWorkspace.shared.runningApplications
            .filter { $0.activationPolicy == .regular && !$0.isTerminated }
            .map { app -> UIApp in
                let pid = app.processIdentifier
                return UIApp(name: app.localizedName ?? app.bundleIdentifier ?? "pid \(pid)",
                             bundleID: app.bundleIdentifier, pid: pid, isFrontmost: pid == front,
                             windows: windows[pid] ?? [])
            }
        return apps.sorted { a, b in
            if a.isFrontmost != b.isFrontmost { return a.isFrontmost }
            if a.windows.isEmpty != b.windows.isEmpty { return !a.windows.isEmpty }
            return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending
        }
    }

    /// Layer-0 windows per owner pid, in the system's front-to-back order. Titles are empty without Screen Recording.
    static func onScreenWindows() -> [Int32: [UIWindowInfo]] {
        guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
                as? [[String: Any]] else { return [:] }
        var out: [Int32: [UIWindowInfo]] = [:]
        for w in list {
            guard (w[kCGWindowLayer as String] as? Int) == 0,
                  let pid = w[kCGWindowOwnerPID as String] as? Int32 ?? (w[kCGWindowOwnerPID as String] as? Int).map(Int32.init),
                  let bounds = w[kCGWindowBounds as String] as? [String: Any],
                  let r = CGRect(dictionaryRepresentation: bounds as CFDictionary), !isBadge(r) else { continue }
            let title = w[kCGWindowName as String] as? String ?? ""
            out[pid, default: []].append(UIWindowInfo(title: title, frame: UIRect(x: r.origin.x, y: r.origin.y, width: r.width, height: r.height)))
        }
        return out
    }

    /// A window too small to be one the user works in. While a window is captured, macOS adds a 66×20 badge window,
    /// titled "Window", to the captured app, on top of its title bar: it must not count as the app's front window.
    static func isBadge(_ r: CGRect) -> Bool { r.width < 80 || r.height < 40 }

    /// Window ids, front to back (all on-screen windows).
    static func zOrder() -> [CGWindowID] {
        guard let list = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
                as? [[String: Any]] else { return [] }
        return list.compactMap { ($0[kCGWindowNumber as String] as? Int).map { CGWindowID($0) } }
    }
}
