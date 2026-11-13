import AppKit
import GraphRuntime
import AIStudioLibrary
import SwiftUI

/// `--smoke-chrome` (no model needed; a throwaway library): the design-system chrome driven with real events posted
/// to the app's queue — context menus and their keyboard, a pop-up picker, the title bar, alerts, the save picker,
/// tooltips and the text edit menu — and that no system menu ever opens.
@MainActor
enum SmokeChrome {
    static func run(_ model: AppModel) async -> Bool {
        func say(_ s: String) { print("[smoke] \(s)"); fflush(stdout) }
        var failures: [String] = []
        func check(_ ok: Bool, _ what: String) {
            say((ok ? "✓ " : "✗ ") + what)
            if !ok { failures.append(what) }
        }
        func settle(_ s: Double = 0.35) async { try? await Task.sleep(nanoseconds: UInt64(s * 1e9)) }
        var nativeMenus = 0
        let observer = NotificationCenter.default.addObserver(forName: NSMenu.didBeginTrackingNotification, object: nil,
                                                              queue: .main) { note in
            // The menu bar tracks too when it's used; only context and pop-up menus count.
            if let m = note.object as? NSMenu, m !== NSApp.mainMenu, m.supermenu == nil { nativeMenus += 1 }
        }
        defer { NotificationCenter.default.removeObserver(observer) }

        UserDefaults.standard.set(SidebarPanel.workflows.rawValue, forKey: "sidebarPanel")
        UserDefaults.standard.set(true, forKey: "sidebarOpen")
        _ = model.saveAs(model.active, name: "Chrome Test")
        model.reloadWorkflows()
        await settle(1.5)
        guard let window = NSApp.windows.first(where: { !($0 is MGFloatingPanel) && $0.isVisible && $0.frame.width >= 900 }) else {
            say("no main window"); return false
        }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        await settle(0.6)
        say("window \(Int(window.frame.width))×\(Int(window.frame.height)), active \(NSApp.isActive)")

        // MARK: events

        func post(_ type: NSEvent.EventType, _ screen: CGPoint, flags: NSEvent.ModifierFlags = [], clicks: Int = 1) {
            let p = window.convertPoint(fromScreen: screen)
            if let e = NSEvent.mouseEvent(with: type, location: p, modifierFlags: flags,
                                          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                                          context: nil, eventNumber: 0, clickCount: clicks, pressure: 1) {
                NSApp.postEvent(e, atStart: false)
            }
        }
        func click(_ p: CGPoint, clicks: Int = 1) async {
            post(.leftMouseDown, p, clicks: clicks); post(.leftMouseUp, p, clicks: clicks); await settle()
        }
        func rightClick(_ p: CGPoint) async { post(.rightMouseDown, p); post(.rightMouseUp, p); await settle() }
        func key(_ code: UInt16, _ chars: String) async {
            for type in [NSEvent.EventType.keyDown, .keyUp] {
                if let e = NSEvent.keyEvent(with: type, location: .zero, modifierFlags: [],
                                            timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                                            context: nil, characters: chars, charactersIgnoringModifiers: chars,
                                            isARepeat: false, keyCode: code) {
                    NSApp.postEvent(e, atStart: false)
                }
            }
            await settle(0.25)
        }
        func center(_ r: CGRect) -> CGPoint { CGPoint(x: r.midX, y: r.midY) }
        var menuTitles: [String] { MGFloating.shared.menu?.root.items.map(\.title) ?? [] }

        // MARK: context menu on a sidebar row

        guard var row = MGTestAnchors.frame("workflow-row:Chrome Test") else { say("no workflow row"); return false }
        await rightClick(center(row))
        check(MGFloating.shared.isMenuOpen && menuTitles.contains("Rename…") && menuTitles.contains("Move to Trash"),
              "right-click on a workflow opens its design-system menu (\(menuTitles.filter { !$0.isEmpty }.joined(separator: ", ")))")
        await key(125, "\u{F701}")
        check(MGFloating.shared.menu?.root.highlighted == 0, "↓ highlights the first item")
        await key(53, "\u{1B}")
        check(!MGFloating.shared.isMenuOpen, "Esc closes the menu")
        await rightClick(center(row))
        await click(CGPoint(x: window.frame.midX, y: window.frame.minY + 40))
        check(!MGFloating.shared.isMenuOpen, "a click outside closes the menu")

        // Left-clicks go through the catcher: opening the workflow from its row.
        let blank = model.newTab()
        check(model.activeTabID == blank.id, "a new tab is active")
        // The new workflow joins the Open list above the saved ones, so the row moved down.
        await settle()
        row = MGTestAnchors.frame("workflow-row:Chrome Test") ?? row
        check(MGTestAnchors.frame("open-workflow:\(blank.name)") != nil, "the new workflow is listed under Open")
        await click(center(row))
        check(model.active.fileURL?.lastPathComponent == "Chrome Test.json", "a left-click on the row still opens the workflow")

        // ↓ ↓ Return runs the second item: Rename… shows the prompt.
        await rightClick(center(row))
        await key(125, "\u{F701}"); await key(125, "\u{F701}")
        await key(36, "\r")
        await settle()
        check(model.presenter.top?.id == "rename", "↓ ↓ Return chooses Rename…, which shows the rename prompt")
        model.presenter.dismiss("rename")
        await settle()

        // A SwiftUI overlay on top (the viewer) keeps right-clicks from reaching the catchers under it.
        model.viewerResult = NodeResult(node: "save", kind: .image, image: nil, url: nil)
        await settle(0.5)
        await rightClick(center(row))
        check(!MGFloating.shared.isMenuOpen, "the viewer covers the sidebar: no row menu opens through it")
        MGFloating.shared.closeAll()
        model.viewerResult = nil
        await settle(0.4)

        // MARK: the title bar

        if let results = MGTestAnchors.frame("results-button") {
            let before = model.showResultsPane
            await click(center(results))
            await settle(0.4)
            check(model.showResultsPane != before, "the title bar's Results button takes clicks")
            if model.showResultsPane != before { await click(center(results)); await settle(0.4) }
        } else {
            check(false, "the Results button is in the title bar")
        }

        // MARK: the graph: node, wire and canvas menus

        if let canvas = MGTestAnchors.frame("graph-canvas"),
           let sampler = model.editorGraph["sampler"], let sf = model.frame(of: sampler),
           let latent = model.editorGraph["latent"], let ls = model.schema(for: latent),
           let ss = model.schema(for: sampler),
           let a = NodeLayout.outputPoint(latent, ls, "latent"), let b = NodeLayout.inputPoint(sampler, ss, "latent") {
            let v = model.active.viewport ?? (1, .zero)
            func screen(_ p: CGPoint) -> CGPoint {
                CGPoint(x: canvas.minX + p.x * v.zoom + v.pan.width, y: canvas.maxY - (p.y * v.zoom + v.pan.height))
            }
            await rightClick(screen(CGPoint(x: sf.minX + 40, y: sf.minY + 12)))
            check(menuTitles.contains("Duplicate") && menuTitles.contains("Delete"), "right-click on a node opens the node menu")
            await key(53, "\u{1B}")
            _ = a
            await rightClick(screen(CGPoint(x: b.x - 7, y: b.y)))
            check(menuTitles.contains("Delete Wire"), "right-click on a wire opens the wire menu")
            await key(53, "\u{1B}")
            await rightClick(CGPoint(x: canvas.maxX - 30, y: canvas.maxY - 90))
            check(menuTitles.contains("Add Node…") && menuTitles.contains("Add to Workflow"),
                  "right-click on empty canvas opens the canvas menu")
            // → opens Add to Workflow's submenu once it's highlighted.
            if let i = MGFloating.shared.menu?.root.visible.firstIndex(where: { $0.title == "Add to Workflow" }) {
                MGFloating.shared.menu?.root.highlighted = i
                await key(124, "\u{F703}")
                check((MGFloating.shared.menu?.levels.count ?? 0) == 2, "→ opens a submenu")
                await key(123, "\u{F702}")
                check((MGFloating.shared.menu?.levels.count ?? 0) == 1, "← closes it again")
            }
            await key(53, "\u{1B}")
        } else {
            check(false, "the graph canvas and its default nodes are there")
        }

        // MARK: a pop-up picker

        UserDefaults.standard.set(SidebarPanel.assets.rawValue, forKey: "sidebarPanel")
        UserDefaults.standard.set("input", forKey: "assetsTab")
        await settle(0.8)
        if let filter = MGTestAnchors.frame("assets-filter") {
            await click(center(filter))
            check(menuTitles == ["All", "Images", "Masks", "Videos", "Sounds"]
                  && MGFloating.shared.menu?.root.items.first?.checked == true, "the picker opens its options with All checked")
            await key(125, "\u{F701}"); await key(125, "\u{F701}"); await key(36, "\r")
            check(!MGFloating.shared.isMenuOpen, "Return picks an option and closes the menu")
            await click(center(filter))
            check(MGFloating.shared.menu?.root.items.first(where: \.checked)?.title == "Images", "the picker now shows Images")
            await key(53, "\u{1B}")
        } else {
            check(false, "the assets filter is on screen")
        }
        UserDefaults.standard.set(SidebarPanel.workflows.rawValue, forKey: "sidebarPanel")
        await settle(0.5)

        // MARK: alerts

        guard let te = model.firstNode(ofType: "TextEncode") else { say("no TextEncode"); return false }
        model.setLiteral(.string("dirty"), node: te.id, input: "prompt", undoManager: model.activeUndoManager)
        let dirty = model.active
        TabCommands.close(dirty, model: model)
        await settle()
        let alert = model.presenter.top
        check(alert?.kind == .alert && alert?.actions.map(\.label) == ["Save", "Don't Save", "Cancel"],
              "closing a dirty tab asks Save / Don't Save / Cancel")
        await rightClick(center(row))
        check(!MGFloating.shared.isMenuOpen, "context menus stay shut under an alert")
        await key(53, "\u{1B}")
        await settle()
        check(model.presenter.top == nil && model.tabs.contains { $0.id == dirty.id }, "Esc cancels the alert and keeps the tab")
        TabCommands.close(dirty, model: model)
        await settle()
        if let top = model.presenter.top, let dont = top.actions.first(where: { $0.label == "Don't Save" }) {
            model.presenter.choose(dont, in: top.id)
        }
        await settle()
        check(!model.tabs.contains { $0.id == dirty.id }, "Don't Save closes the tab")

        // MARK: the save picker

        let sub = model.library.workflows.appendingPathComponent("Sub", isDirectory: true)
        try? FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        var saved: URL?
        model.presenter.chooseFile(title: "Save Workflow", roots: [MGFileRoot(title: "Workflows", url: model.library.workflows)],
                                   mode: .save(name: "Picked Name", type: .json), startIn: sub, finder: { nil },
                                   onChoose: { saved = $0.first })
        await settle(0.6)
        check(model.presenter.top?.id == "file-picker", "Save As shows the in-app picker")
        await key(36, "\r")
        await settle()
        check(saved?.standardizedFileURL.path == sub.appendingPathComponent("Picked Name.json").standardizedFileURL.path,
              "Return saves into the folder it opened in (\(saved?.path ?? "nothing"))")
        check(model.presenter.top == nil, "the picker closes")

        // MARK: tooltips

        MGFloating.shared.tooltip.enter(UUID(), text: "Queue the current workflow (⌘↩)", window: window)
        await settle(1.1)
        if NSApp.isActive {
            check(MGFloating.shared.tooltip.shownText == "Queue the current workflow (⌘↩)", "a tooltip shows after the delay")
            post(.leftMouseDown, CGPoint(x: window.frame.midX, y: window.frame.minY + 40))
            post(.leftMouseUp, CGPoint(x: window.frame.midX, y: window.frame.minY + 40))
            await settle()
            check(!MGFloating.shared.tooltip.isShown, "a click hides it")
        } else {
            say("– tooltip check skipped: the app is in the background")
        }

        // MARK: the text edit menu

        model.selection = []
        model.selectedGroup = nil
        MGFloating.shared.closeAll()
        await settle(0.6)

        if let field = MGTestAnchors.frame("inspector-name") {
            let pb = NSPasteboard.general
            let savedItems = pb.pasteboardItems?.map { item -> [NSPasteboard.PasteboardType: Data] in
                Dictionary(uniqueKeysWithValues: item.types.compactMap { t in item.data(forType: t).map { (t, $0) } })
            } ?? []
            await rightClick(center(field))
            if !MGFloating.shared.isMenuOpen {
                await settle(0.5)
                say("edit menu not open: open now \(MGFloating.shared.isMenuOpen), last close “\(MGFloating.shared.lastCloseReason)”, responder \(window.firstResponder.map { String(describing: type(of: $0)) } ?? "nil"), target \(MGTestAnchors.frame("inspector-name").map { "\($0)" } ?? "gone")")
            }
            check(menuTitles.contains("Select All") && menuTitles.contains("Paste"), "right-click in a text field opens the edit menu")
            if let session = MGFloating.shared.menu, let i = session.root.visible.firstIndex(where: { $0.title == "Select All" }) {
                session.activate(session.root, i)
                await settle()
                await rightClick(center(field))
                if let s2 = MGFloating.shared.menu, let c = s2.root.visible.firstIndex(where: { $0.title == "Copy" }) {
                    check(s2.root.visible[c].enabled, "Copy is enabled once the text is selected")
                    s2.activate(s2.root, c)
                    await settle()
                }
                // (The clipboard's contents aren't printed: they're restored below and may be the user's.)
                check(pb.string(forType: .string) == model.active.name, "Select All then Copy copies the field")
            }
            MGFloating.shared.closeAll()
            pb.clearContents()
            pb.writeObjects(savedItems.map { d in
                let item = NSPasteboardItem()
                for (t, data) in d { item.setData(data, forType: t) }
                return item
            })
        } else {
            check(false, "the inspector's name field is on screen")
        }

        check(nativeMenus == 0, "no system context or pop-up menu opened (\(nativeMenus))")
        say(failures.isEmpty ? "all chrome checks passed" : "\(failures.count) failed: \(failures.joined(separator: "; "))")
        return failures.isEmpty
    }
}
