import AppKit
import SwiftUI

/// `AIStudioApp --ui-fixture`: a tiny standalone app for testing the `ui_*` tools (`--smoke-ui` starts it as a child
/// process). It runs before any of AIStudio's startup: no models, stores, servers or user data.
enum UIFixture {
    /// The list's rows, long enough to scroll: Apple, Banana, Cherry, then Item 4…Item 40.
    static let rows = ["Apple", "Banana", "Cherry"] + (4...40).map { "Item \($0)" }

    static func run() -> Never {
        let app = NSApplication.shared
        app.setActivationPolicy(.regular)
        app.mainMenu = menu()
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 420, height: 300),
                              styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "UI Fixture"
        window.contentView = NSHostingView(rootView: UIFixtureView())
        window.center()
        window.isReleasedWhenClosed = false
        window.makeKeyAndOrderFront(nil)
        app.activate(ignoringOtherApps: true)
        app.run()
        exit(0)
    }

    /// The app and Edit menus a real app has: without them ⌘A, ⌘C, ⌘V and ⌘Z do nothing in a text field.
    private static func menu() -> NSMenu {
        let bar = NSMenu()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit UI Fixture", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        for sub in [appMenu, edit] {
            let item = NSMenuItem()
            item.submenu = sub
            bar.addItem(item)
        }
        return bar
    }
}

private struct UIFixtureView: View {
    @State private var name = ""
    @State private var count = 0
    @State private var enabled = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            TextField("Name", text: $name)
                .accessibilityLabel("Name")
            HStack {
                Button("Count") { count += 1 }
                Text("Count: \(count)")
            }
            Toggle("Enabled", isOn: $enabled)
            List(UIFixture.rows, id: \.self) { Text($0) }
                .frame(height: 100)
        }
        .padding(16)
        .frame(width: 420, height: 300)
    }
}
