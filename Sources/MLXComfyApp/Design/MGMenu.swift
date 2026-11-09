import AppKit
import MLXComfyLibrary
import SwiftUI

// The design system's Menu: one description of a menu's items, drawn either by the design system (context menus,
// pop-up buttons) or as the system's menu items (the menu bar, which macOS draws itself).

// MARK: - items

struct MGMenuItem {
    enum Kind {
        case action(@MainActor () -> Void)
        case submenu([MGMenuItem])
        case separator
        case header
    }

    var kind: Kind
    var title = ""
    var icon: String? = nil
    var shortcut: KeyboardShortcut? = nil
    var checked = false
    var enabled = true
    var destructive = false

    static func button(_ title: String, icon: String? = nil, shortcut: KeyboardShortcut? = nil, checked: Bool = false,
                       enabled: Bool = true, destructive: Bool = false,
                       action: @escaping @MainActor () -> Void) -> MGMenuItem {
        MGMenuItem(kind: .action(action), title: title, icon: icon, shortcut: shortcut, checked: checked,
                   enabled: enabled, destructive: destructive)
    }

    static func menu(_ title: String, icon: String? = nil, enabled: Bool = true,
                     @MGMenuBuilder _ items: () -> [MGMenuItem]) -> MGMenuItem {
        let sub = items()
        return MGMenuItem(kind: .submenu(sub), title: title, icon: icon, enabled: enabled && !sub.isEmpty)
    }

    static var divider: MGMenuItem { MGMenuItem(kind: .separator) }
    static func header(_ title: String) -> MGMenuItem { MGMenuItem(kind: .header, title: title) }

    var isSeparator: Bool { if case .separator = kind { return true } else { return false } }
    var isHeader: Bool { if case .header = kind { return true } else { return false } }
    var submenu: [MGMenuItem]? { if case .submenu(let s) = kind { return s } else { return nil } }
    /// Whether the keyboard and the pointer can land on it.
    var isSelectable: Bool { enabled && !isSeparator && !isHeader }

    /// The shortcut as the menu shows it (⌃⌥⇧⌘ and the key).
    var shortcutText: String? { shortcut.map(Self.text(for:)) }

    static func text(for s: KeyboardShortcut) -> String {
        var t = ""
        if s.modifiers.contains(.control) { t += "⌃" }
        if s.modifiers.contains(.option) { t += "⌥" }
        if s.modifiers.contains(.shift) { t += "⇧" }
        if s.modifiers.contains(.command) { t += "⌘" }
        switch s.key {
        case .return: t += "↩"
        case .delete: t += "⌫"
        case .deleteForward: t += "⌦"
        case .escape: t += "⎋"
        case .tab: t += "⇥"
        case .space: t += "Space"
        case .upArrow: t += "↑"
        case .downArrow: t += "↓"
        case .leftArrow: t += "←"
        case .rightArrow: t += "→"
        default: t += String(s.key.character).uppercased()
        }
        return t
    }

    /// Without separators at either end or two in a row.
    static func tidy(_ items: [MGMenuItem]) -> [MGMenuItem] {
        var out: [MGMenuItem] = []
        for item in items {
            if item.isSeparator, out.isEmpty || out.last!.isSeparator { continue }
            out.append(item)
        }
        while out.last?.isSeparator == true { out.removeLast() }
        return out
    }
}

@resultBuilder
enum MGMenuBuilder {
    static func buildExpression(_ item: MGMenuItem) -> [MGMenuItem] { [item] }
    static func buildExpression(_ items: [MGMenuItem]) -> [MGMenuItem] { items }
    static func buildBlock(_ parts: [MGMenuItem]...) -> [MGMenuItem] { parts.flatMap { $0 } }
    static func buildOptional(_ part: [MGMenuItem]?) -> [MGMenuItem] { part ?? [] }
    static func buildEither(first part: [MGMenuItem]) -> [MGMenuItem] { part }
    static func buildEither(second part: [MGMenuItem]) -> [MGMenuItem] { part }
    static func buildArray(_ parts: [[MGMenuItem]]) -> [MGMenuItem] { parts.flatMap { $0 } }
}

/// The items as the system's menu items, for the menu bar.
struct MGNativeMenuItems: View {
    let items: [MGMenuItem]

    init(_ items: [MGMenuItem]) { self.items = items }
    init(@MGMenuBuilder _ items: () -> [MGMenuItem]) { self.items = items() }

    var body: some View {
        let tidy = MGMenuItem.tidy(items)
        ForEach(tidy.indices, id: \.self) { i in
            let item = tidy[i]
            switch item.kind {
            case .action(let run):
                if item.checked {
                    Toggle(item.title, isOn: Binding(get: { true }, set: { _ in run() }))
                        .keyboardShortcut(item.shortcut)
                        .disabled(!item.enabled)
                } else {
                    Button(item.title, role: item.destructive ? .destructive : nil) { run() }
                        .keyboardShortcut(item.shortcut)
                        .disabled(!item.enabled)
                }
            case .submenu(let sub):
                Menu(item.title) { MGNativeMenuItems(sub) }.disabled(!item.enabled)
            case .separator:
                Divider()
            case .header:
                Text(item.title)
            }
        }
    }
}

// MARK: - metrics

/// The design system's menu metrics: 5 padding, 22 rows, an 18-wide check column, 10.5 for a separator.
enum MGMenuMetrics {
    static let padding: CGFloat = 5
    static let row: CGFloat = 22
    static let separator: CGFloat = 10.5
    static let check: CGFloat = 18
    static let icon: CGFloat = 22
    static let search: CGFloat = 28
    static let minWidth: CGFloat = 160
    static let maxWidth: CGFloat = 440

    static func height(_ item: MGMenuItem) -> CGFloat { item.isSeparator ? separator : row }

    @MainActor static func textWidth(_ s: String) -> CGFloat {
        ceil((s as NSString).size(withAttributes: [.font: MGType.body.nsFont]).width)
    }
}

// MARK: - a menu on screen

/// One menu of an open chain (the root, or a submenu), with its highlight and, for long pop-ups, a search query.
@MainActor @Observable
final class MGMenuLevel {
    let items: [MGMenuItem]
    let searchable: Bool
    var query = ""
    /// Index into `visible`.
    var highlighted: Int?
    /// True when the keyboard moved the highlight (so the list scrolls to it).
    var keyboardMoved = false
    var minWidth: CGFloat = 0
    @ObservationIgnored var frame: CGRect = .zero
    @ObservationIgnored let panel = MGFloatingPanel()
    /// The index (in `visible`) of the item whose submenu is open.
    var openSubmenu: Int?

    init(items: [MGMenuItem], searchable: Bool = false) {
        self.items = MGMenuItem.tidy(items)
        self.searchable = searchable
    }

    var visible: [MGMenuItem] {
        let q = query.trimmingCharacters(in: .whitespaces)
        guard searchable, !q.isEmpty else { return items }
        return items.filter { !$0.isSeparator && !$0.isHeader && $0.title.localizedCaseInsensitiveContains(q) }
    }

    var hasIcons: Bool { items.contains { $0.icon != nil } }

    /// The menu's size: its widest row, and all its rows up to `maxHeight` (it scrolls past that).
    func size(maxHeight: CGFloat) -> CGSize {
        let icons = hasIcons
        var widest: CGFloat = 0
        for item in items where !item.isSeparator {
            var w = MGMenuMetrics.check + (icons ? MGMenuMetrics.icon : 0) + MGMenuMetrics.textWidth(item.title) + 10
            if let s = item.shortcutText { w += 24 + MGMenuMetrics.textWidth(s) }
            if item.submenu != nil { w += 22 }
            widest = max(widest, w)
        }
        let width = min(max(widest + MGMenuMetrics.padding * 2, MGMenuMetrics.minWidth, minWidth), MGMenuMetrics.maxWidth)
        let content = items.reduce(0) { $0 + MGMenuMetrics.height($1) }
        let height = MGMenuMetrics.padding * 2 + (searchable ? MGMenuMetrics.search : 0) + max(content, MGMenuMetrics.row)
        return CGSize(width: width, height: min(height, maxHeight))
    }

    /// Row `index` (of `visible`) on screen, as laid out before any scrolling.
    func rowFrame(_ index: Int) -> CGRect {
        let rows = visible
        var top = MGMenuMetrics.padding + (searchable ? MGMenuMetrics.search : 0)
        for i in 0..<min(index, rows.count) { top += MGMenuMetrics.height(rows[i]) }
        let h = index < rows.count ? MGMenuMetrics.height(rows[index]) : MGMenuMetrics.row
        return CGRect(x: frame.minX + MGMenuMetrics.padding, y: frame.maxY - top - h,
                      width: frame.width - MGMenuMetrics.padding * 2, height: h)
    }
}

/// An open menu chain and its keyboard and pointer handling.
@MainActor
final class MGMenuSession {
    let root: MGMenuLevel
    private(set) var levels: [MGMenuLevel] = []
    weak var window: NSWindow?
    var onClose: (() -> Void)?
    private var typed = ""
    private var typedAt = Date.distantPast
    private var timer: Timer?

    init(items: [MGMenuItem], window: NSWindow?, searchable: Bool = false) {
        root = MGMenuLevel(items: items, searchable: searchable)
        self.window = window
    }

    func present(_ level: MGMenuLevel, at frame: CGRect) {
        level.frame = frame
        level.panel.show(MGMenuView(level: level, session: self), surface: frame, parent: window)
        levels.append(level)
    }

    func closeAll() {
        timer?.invalidate()
        for level in levels { level.panel.hide() }
        levels.removeAll()
    }

    private func close(after level: MGMenuLevel) {
        guard let i = levels.firstIndex(where: { $0 === level }) else { return }
        for l in levels[(i + 1)...] { l.panel.hide() }
        levels.removeSubrange((i + 1)...)
        level.openSubmenu = nil
    }

    // MARK: pointer

    func hover(_ level: MGMenuLevel, _ index: Int?) {
        timer?.invalidate()
        guard let index else { return }
        level.keyboardMoved = false
        level.highlighted = level.visible[index].isSelectable ? index : nil
        guard level.openSubmenu != index else { return }
        // Open (or close) a submenu after a short rest, so moving across rows towards one doesn't flicker.
        timer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: false) { [weak self, weak level] _ in
            MainActor.assumeIsolated {
                guard let self, let level, level.highlighted == index else { return }
                self.close(after: level)
                if level.visible[index].submenu != nil { self.openSubmenu(level, index) }
            }
        }
    }

    func activate(_ level: MGMenuLevel, _ index: Int) {
        let item = level.visible[index]
        guard item.isSelectable else { return }
        switch item.kind {
        case .action(let run):
            MGFloating.shared.closeMenu()
            // After the menu has gone, so whatever the action shows (a sheet, an alert) comes up cleanly.
            DispatchQueue.main.async { run() }
        case .submenu:
            timer?.invalidate()
            close(after: level)
            openSubmenu(level, index)
        default:
            break
        }
    }

    private func openSubmenu(_ level: MGMenuLevel, _ index: Int, highlightFirst: Bool = false) {
        guard let items = level.visible[index].submenu else { return }
        let sub = MGMenuLevel(items: items)
        let row = level.rowFrame(index)
        let screen = mgScreen(containing: CGPoint(x: row.midX, y: row.midY))
        let size = sub.size(maxHeight: SurfacePlacement.maxMenuHeight(screen: screen))
        level.openSubmenu = index
        if highlightFirst {
            sub.highlighted = MenuNavigation.move(from: nil, step: 1, selectable: sub.visible.map(\.isSelectable))
            sub.keyboardMoved = true
        }
        present(sub, at: SurfacePlacement.submenu(row: row, parent: level.frame, size: size, screen: screen))
    }

    // MARK: keyboard

    func key(_ e: NSEvent) {
        guard let level = levels.last else { return }
        let selectable = level.visible.map(\.isSelectable)
        func move(_ step: Int) {
            level.highlighted = MenuNavigation.move(from: level.highlighted, step: step, selectable: selectable)
            level.keyboardMoved = true
        }
        switch e.keyCode {
        case 125: move(1)
        case 126: move(-1)
        case 124:
            if let h = level.highlighted, level.visible[h].submenu != nil {
                close(after: level)
                openSubmenu(level, h, highlightFirst: true)
            }
        case 123:
            if levels.count > 1, let parent = levels.dropLast().last { close(after: parent) }
        case 36, 76:
            if let h = level.highlighted { activate(level, h) }
        case 53:
            if level.searchable, !level.query.isEmpty { level.query = ""; level.highlighted = nil } else { MGFloating.shared.closeMenu() }
        case 51 where level.searchable:
            if !level.query.isEmpty { level.query.removeLast() }
            level.highlighted = MenuNavigation.move(from: nil, step: 1, selectable: level.visible.map(\.isSelectable))
        case 49 where !level.searchable && typed.isEmpty:
            if let h = level.highlighted { activate(level, h) }
        default:
            guard let chars = e.charactersIgnoringModifiers, !chars.isEmpty,
                  chars.unicodeScalars.allSatisfy({ !CharacterSet.controlCharacters.contains($0) }) else { return }
            if level.searchable {
                level.query += chars
                level.highlighted = MenuNavigation.move(from: nil, step: 1, selectable: level.visible.map(\.isSelectable))
                level.keyboardMoved = true
                return
            }
            if Date().timeIntervalSince(typedAt) > 1 { typed = "" }
            typed += chars
            typedAt = Date()
            if let i = MenuNavigation.match(typed, titles: level.visible.map(\.title), selectable: selectable,
                                            from: level.highlighted) {
                level.highlighted = i
                level.keyboardMoved = true
            }
        }
    }
}

// MARK: - drawing

/// Menu glass with the rows: the hovered row takes the selection colour at radius 5, with an 18-wide check column.
struct MGMenuView: View {
    let level: MGMenuLevel
    let session: MGMenuSession

    var body: some View {
        let rows = level.visible
        let icons = level.hasIcons
        VStack(spacing: 0) {
            if level.searchable {
                MGFieldChrome(icon: "magnifyingglass") {
                    Text(level.query.isEmpty ? "Type to search" : level.query)
                        .foregroundStyle(level.query.isEmpty ? MG.placeholder : MG.label)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(.bottom, 6)
            }
            ScrollViewReader { proxy in
                ScrollView(.vertical) {
                    VStack(spacing: 0) {
                        if rows.isEmpty {
                            Text("No matches").foregroundStyle(MG.secondaryLabel)
                                .frame(maxWidth: .infinity, minHeight: MGMenuMetrics.row)
                        }
                        ForEach(rows.indices, id: \.self) { i in
                            MGMenuRow(item: rows[i], icons: icons,
                                      highlighted: level.highlighted == i || level.openSubmenu == i)
                                .id(i)
                                .onHover { inside in if inside { session.hover(level, i) } }
                                .onTapGesture { session.activate(level, i) }
                        }
                    }
                }
                .scrollIndicators(.hidden)
                .scrollDisabled(level.frame.height >= level.size(maxHeight: .infinity).height)
                .onChange(of: level.highlighted) { _, h in
                    if let h, level.keyboardMoved { proxy.scrollTo(h) }
                }
            }
        }
        .padding(MGMenuMetrics.padding)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .mgGlass(.menu, radius: MGRadius.lg)
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous))
        .onHover { inside in if !inside { level.highlighted = level.openSubmenu } }
    }
}

private struct MGMenuRow: View {
    let item: MGMenuItem
    let icons: Bool
    let highlighted: Bool

    var body: some View {
        switch item.kind {
        case .separator:
            Rectangle().fill(MG.separator).frame(height: 0.5).padding(.horizontal, 6).padding(.vertical, 5)
        case .header:
            Text(item.title).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel)
                .frame(maxWidth: .infinity, minHeight: MGMenuMetrics.row, alignment: .leading)
                .padding(.leading, MGMenuMetrics.check)
        default:
            HStack(spacing: 0) {
                Group {
                    if item.checked { Image(systemName: "checkmark").font(.system(size: 10, weight: .bold)) }
                }
                .frame(width: MGMenuMetrics.check)
                if icons {
                    Group {
                        if let icon = item.icon { Image(systemName: icon).font(.system(size: 12)) }
                    }
                    .frame(width: 16, alignment: .center)
                    .padding(.trailing, 6)
                }
                Text(item.title).lineLimit(1).truncationMode(.middle)
                    .foregroundStyle(item.destructive ? MG.destructive : MG.label)
                Spacer(minLength: 0)
                if let s = item.shortcutText {
                    Text(s).foregroundStyle(MG.secondaryLabel).padding(.leading, 24)
                }
                if item.submenu != nil {
                    Image(systemName: "chevron.right").font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(MG.secondaryLabel).padding(.leading, 12)
                }
            }
            .font(.mgBody)
            .padding(.trailing, 10)
            .frame(height: MGMenuMetrics.row)
            .background {
                if highlighted && item.isSelectable {
                    RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous).fill(MG.selection)
                }
            }
            .contentShape(Rectangle())
            .opacity(item.enabled ? 1 : 0.4)
        }
    }
}

// MARK: - context menus

/// Catches right-clicks (and ⌃-clicks) over a view and opens a design-system menu. Every other event goes through
/// to the SwiftUI view underneath, because `hitTest` only claims the point for a right-click.
final class MGContextCatcherView: NSView {
    var items: ((CGPoint) -> [MGMenuItem])?
    /// While set, catchers let every click through (to find what's under them).
    @MainActor static var suspended = false

    override var isFlipped: Bool { true }

    static func isContextClick(_ e: NSEvent?) -> Bool {
        guard let e else { return false }
        return e.type == .rightMouseDown || (e.type == .leftMouseDown && e.modifierFlags.contains(.control))
    }

    /// Off while the view is disabled (under an alert or sheet).
    var enabled = true

    override func hitTest(_ point: NSPoint) -> NSView? {
        enabled && !Self.suspended && Self.isContextClick(NSApp.currentEvent) ? super.hitTest(point) : nil
    }

    override func rightMouseDown(with e: NSEvent) { open(e) }
    override func mouseDown(with e: NSEvent) {
        if e.modifierFlags.contains(.control) { open(e) } else { super.mouseDown(with: e) }
    }
    override func menu(for event: NSEvent) -> NSMenu? { nil }

    func open(_ e: NSEvent) {
        guard let items, let window else { return }
        let local = convert(e.locationInWindow, from: nil)
        let list = items(local)
        guard !MGMenuItem.tidy(list).isEmpty else { return }
        MGFloating.shared.openMenu(list, at: window.convertPoint(toScreen: e.locationInWindow), in: window)
    }
}

private struct MGContextCatcher: NSViewRepresentable {
    let items: (CGPoint) -> [MGMenuItem]
    func makeNSView(context: Context) -> MGContextCatcherView {
        let v = MGContextCatcherView()
        v.items = items
        return v
    }
    func updateNSView(_ v: MGContextCatcherView, context: Context) {
        v.items = items
        v.enabled = context.environment.isEnabled
    }
}

extension View {
    /// A design-system context menu; the items are built when the menu opens.
    func mgContextMenu(@MGMenuBuilder _ items: @escaping () -> [MGMenuItem]) -> some View {
        overlay(MGContextCatcher(items: { _ in items() }))
    }

    /// A context menu that depends on where the click was (in this view's coordinates, y down).
    func mgContextMenu(at items: @escaping (CGPoint) -> [MGMenuItem]) -> some View {
        overlay(MGContextCatcher(items: items))
    }
}

// MARK: - pop-up button and menu button

/// The design system's pop-up button: the control fill with a hairline, the label, and the accent up-down tile.
private struct MGPopUpLabel: View {
    let title: String
    var chevrons = true
    var pressed = false
    @Environment(\.isEnabled) private var enabled

    var body: some View {
        HStack(spacing: 8) {
            Text(title).lineLimit(1).truncationMode(.middle)
            if chevrons {
                RoundedRectangle(cornerRadius: 4, style: .continuous).fill(MG.accent)
                    .overlay {
                        Image(systemName: "chevron.up.chevron.down").font(.system(size: 8, weight: .bold))
                            .foregroundStyle(MG.accentForeground)
                    }
                    .frame(width: 16, height: 16)
            }
        }
        .font(.mgBody)
        .foregroundStyle(MG.label)
        .padding(.leading, 10)
        .padding(.trailing, chevrons ? 3 : 10)
        .frame(height: 22)
        .background {
            RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                .fill(pressed ? MG.fillPressed : MG.controlButton)
                .mgShadow(MGShadow.control)
        }
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .contentShape(RoundedRectangle(cornerRadius: MGRadius.md))
        .opacity(enabled ? 1 : 0.4)
    }
}

/// A pop-up picker: the current option on a pop-up button, the options in a design-system menu under it (with a
/// search field when there are many).
struct MGPicker<Value: Hashable>: View {
    @Binding var selection: Value
    let options: [(value: Value, title: String)]
    var placeholder = "Choose…"
    var maxWidth: CGFloat? = nil
    @State private var anchor = MGScreenAnchor()
    @State private var open = false
    @Environment(\.isEnabled) private var enabled

    init(selection: Binding<Value>, options: [(value: Value, title: String)], placeholder: String = "Choose…",
         maxWidth: CGFloat? = nil) {
        _selection = selection
        self.options = options
        self.placeholder = placeholder
        self.maxWidth = maxWidth
    }

    var title: String { options.first { $0.value == selection }?.title ?? placeholder }

    var body: some View {
        MGPopUpLabel(title: title, pressed: open)
            .frame(maxWidth: maxWidth, alignment: .leading)
            .fixedSize(horizontal: maxWidth == nil, vertical: true)
            .mgAnchor(anchor)
            .onTapGesture { if enabled { show() } }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(title)
            .accessibilityAddTraits(.isButton)
            .accessibilityAction { show() }
    }

    func show() {
        guard let frame = anchor.screenFrame else { return }
        let binding = $selection
        let items = options.map { o in
            MGMenuItem.button(o.title, checked: o.value == selection) { binding.wrappedValue = o.value }
        }
        open = true
        MGFloating.shared.openMenu(items, below: frame, in: anchor.window, searchable: options.count > 12) {
            open = false
        }
    }
}

/// A button that opens a design-system menu under itself.
struct MGMenuButton<Label: View>: View {
    let items: () -> [MGMenuItem]
    @ViewBuilder var label: Label
    var kind: MGButtonKind = .bordered
    @State private var anchor = MGScreenAnchor()

    init(kind: MGButtonKind = .bordered, @MGMenuBuilder items: @escaping () -> [MGMenuItem],
         @ViewBuilder label: () -> Label) {
        self.kind = kind
        self.items = items
        self.label = label()
    }

    var body: some View {
        Button {
            guard let frame = anchor.screenFrame else { return }
            MGFloating.shared.openMenu(items(), below: frame, in: anchor.window)
        } label: {
            label
        }
        .buttonStyle(.mg(kind))
        .mgAnchor(anchor)
    }
}

extension MGMenuButton where Label == AnyView {
    /// A titled menu button with a down chevron.
    init(_ title: String, kind: MGButtonKind = .bordered, @MGMenuBuilder items: @escaping () -> [MGMenuItem]) {
        self.init(kind: kind, items: items) {
            AnyView(HStack(spacing: 5) {
                Text(title)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
            })
        }
    }
}
