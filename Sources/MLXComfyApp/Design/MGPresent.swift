import AppKit
import SwiftUI

// Alerts, confirmation dialogs, prompts and sheets drawn by the design system inside the window: a Scrim dims the
// window and centres the Alert or Sheet on it. The window underneath is disabled while one is up, so its buttons and
// keyboard shortcuts can't fire through.

struct MGAlertAction {
    enum Role { case cancel, destructive }
    var label: String
    var role: Role? = nil
    var action: @MainActor () -> Void = {}

    static func cancel(_ label: String = "Cancel", action: @escaping @MainActor () -> Void = {}) -> MGAlertAction {
        MGAlertAction(label: label, role: .cancel, action: action)
    }
    static func destructive(_ label: String, action: @escaping @MainActor () -> Void) -> MGAlertAction {
        MGAlertAction(label: label, role: .destructive, action: action)
    }
}

/// Closes the sheet it's read in (the presenter's version of SwiftUI's `dismiss`).
struct MGDismissAction {
    var run: @MainActor () -> Void = {}
    @MainActor func callAsFunction() { run() }
}

extension EnvironmentValues {
    @Entry var mgDismiss = MGDismissAction()
}

@MainActor @Observable
final class MGPresenter {
    enum Kind {
        /// Two actions or fewer side by side (Cancel on the left), more stacked.
        case alert
        /// Always stacked; Cancel is added when there's none.
        case dialog
    }

    struct Entry: Identifiable {
        let id: String
        var kind: Kind? = nil
        var title = ""
        var message: String? = nil
        var actions: [MGAlertAction] = []
        var sheet: AnyView? = nil
        var onDismiss: (@MainActor () -> Void)? = nil
    }

    private(set) var stack: [Entry] = []

    var isPresenting: Bool { !stack.isEmpty }
    var top: Entry? { stack.last }

    /// Shows an alert. The first action without a role is the default (prominent, Return); Cancel answers Esc.
    func alert(_ title: String, message: String? = nil, actions: [MGAlertAction]) {
        push(Entry(id: "alert-\(UUID().uuidString)", kind: .alert, title: title, message: message,
                   actions: actions.isEmpty ? [MGAlertAction(label: "OK")] : actions))
    }

    func confirm(_ title: String, message: String? = nil, actions: [MGAlertAction]) {
        var all = actions
        if !all.contains(where: { $0.role == .cancel }) { all.append(.cancel()) }
        push(Entry(id: "dialog-\(UUID().uuidString)", kind: .dialog, title: title, message: message, actions: all))
    }

    /// Shows `content` as a sheet; `id` keeps one sheet per id.
    func present<Content: View>(id: String, onDismiss: (@MainActor () -> Void)? = nil, @ViewBuilder _ content: () -> Content) {
        guard !stack.contains(where: { $0.id == id }) else { return }
        push(Entry(id: id, sheet: AnyView(content()), onDismiss: onDismiss))
    }

    func dismiss(_ id: String) {
        guard let i = stack.firstIndex(where: { $0.id == id }) else { return }
        let e = stack.remove(at: i)
        e.onDismiss?()
    }

    /// Runs an alert's action after taking the alert down.
    func choose(_ action: MGAlertAction, in id: String) {
        stack.removeAll { $0.id == id }
        action.action()
    }

    private func push(_ e: Entry) {
        MGFloating.shared.closeAll()
        MGFloating.shared.tooltip.hide()
        withAnimation(.easeOut(duration: 0.16)) { stack.append(e) }
    }
}

/// Puts a presenter in the environment and draws its alerts and sheets over the content.
private struct MGPresentationHost: ViewModifier {
    let presenter: MGPresenter

    func body(content: Content) -> some View {
        content
            .disabled(presenter.isPresenting)
            .overlay {
                if !presenter.stack.isEmpty {
                    ZStack {
                        MG.scrim.ignoresSafeArea()
                            .contentShape(Rectangle())
                            .onTapGesture {}
                        ForEach(presenter.stack) { entry in
                            let topmost = entry.id == presenter.top?.id
                            Group {
                                if let kind = entry.kind {
                                    MGAlertView(entry: entry, kind: kind, presenter: presenter)
                                } else if let sheet = entry.sheet {
                                    sheet
                                        .environment(\.mgDismiss, MGDismissAction { presenter.dismiss(entry.id) })
                                        .mgSheetChrome()
                                }
                            }
                            .disabled(!topmost)
                            .opacity(topmost ? 1 : 0)
                            .transition(.scale(scale: 0.96).combined(with: .opacity))
                        }
                        .padding(40)
                    }
                }
            }
            .environment(presenter)
    }
}

extension View {
    /// The design system's Sheet around this content: sheet glass, radius 12, a hairline and the sheet shadow.
    func mgSheetChrome() -> some View {
        mgGlass(.sheet, radius: MGRadius.xl)
            .clipShape(RoundedRectangle(cornerRadius: MGRadius.xl, style: .continuous))
            .mgHairline(RoundedRectangle(cornerRadius: MGRadius.xl, style: .continuous))
            .background(RoundedRectangle(cornerRadius: MGRadius.xl).fill(MG.windowBackground.opacity(0.01)).mgShadow(MGShadow.sheet))
    }

    /// The window's presenter for alerts and sheets (put on each window's content, inside its environment).
    func mgPresentationHost(_ presenter: MGPresenter) -> some View {
        modifier(MGPresentationHost(presenter: presenter))
    }
}

/// The design system's Alert and ConfirmationDialog: a 260-wide sheet, title centred, message, full-width buttons.
struct MGAlertView: View {
    let entry: MGPresenter.Entry
    let kind: MGPresenter.Kind
    let presenter: MGPresenter

    var stacked: Bool { kind == .dialog || entry.actions.count > 2 }

    /// Cancel on the left when side by side; last when stacked.
    var ordered: [MGAlertAction] {
        let cancel = entry.actions.filter { $0.role == .cancel }
        let rest = entry.actions.filter { $0.role != .cancel }
        return stacked ? rest + cancel : cancel + rest
    }

    var defaultIndex: Int? { ordered.firstIndex { $0.role == nil } }

    var body: some View {
        VStack(spacing: 12) {
            if !entry.title.isEmpty {
                Text(entry.title).font(.mgHeadline).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let message = entry.message {
                Text(message).font(.mgCallout).foregroundStyle(MG.secondaryLabel).multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            let layout = stacked ? AnyLayout(VStackLayout(spacing: 6)) : AnyLayout(HStackLayout(spacing: 8))
            layout {
                ForEach(ordered.indices, id: \.self) { i in
                    let a = ordered[i]
                    let isDefault = i == defaultIndex
                    Button { presenter.choose(a, in: entry.id) } label: { Text(a.label).frame(maxWidth: .infinity) }
                        .buttonStyle(.mg(isDefault ? .prominent : .bordered, destructive: a.role == .destructive))
                        .keyboardShortcut(isDefault ? .defaultAction : a.role == .cancel ? .cancelAction : nil)
                }
            }
            .padding(.top, 4)
        }
        .padding(20)
        .frame(width: 260)
        .mgGlass(.sheet, radius: MGRadius.xl)
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.xl, style: .continuous))
    }
}

// MARK: - declarative presentation

private struct MGSheetModifier<Sheet: View>: ViewModifier {
    @Binding var isPresented: Bool
    let id: String
    var onDismiss: (() -> Void)?
    @ViewBuilder var sheet: () -> Sheet
    @Environment(MGPresenter.self) private var presenter: MGPresenter?

    func body(content: Content) -> some View {
        content
            .onChange(of: isPresented, initial: true) { _, shown in
                guard let presenter else { return }
                if shown {
                    presenter.present(id: id, onDismiss: {
                        isPresented = false
                        onDismiss?()
                    }, sheet)
                } else {
                    presenter.dismiss(id)
                }
            }
    }
}

private struct MGItemSheetModifier<Item: Identifiable, Sheet: View>: ViewModifier {
    @Binding var item: Item?
    let id: String
    @ViewBuilder var sheet: (Item) -> Sheet
    @Environment(MGPresenter.self) private var presenter: MGPresenter?

    func body(content: Content) -> some View {
        content
            .onChange(of: item?.id, initial: true) { old, new in
                guard let presenter else { return }
                if old != nil, old != new { presenter.dismiss(id) }
                if let current = item {
                    presenter.present(id: id, onDismiss: { if item?.id == current.id { item = nil } }) { sheet(current) }
                } else {
                    presenter.dismiss(id)
                }
            }
    }
}

private struct MGAlertModifier: ViewModifier {
    @Binding var isPresented: Bool
    let title: String
    let message: String?
    let kind: MGPresenter.Kind
    let actions: () -> [MGAlertAction]
    @Environment(MGPresenter.self) private var presenter: MGPresenter?

    func body(content: Content) -> some View {
        content
            .onChange(of: isPresented) { _, shown in
                guard shown, let presenter else { return }
                let wrapped = actions().map { a in
                    MGAlertAction(label: a.label, role: a.role) { isPresented = false; a.action() }
                }
                if kind == .alert { presenter.alert(title, message: message, actions: wrapped) }
                else { presenter.confirm(title, message: message, actions: wrapped) }
            }
    }
}

extension View {
    func mgSheet<Content: View>(_ id: String, isPresented: Binding<Bool>, onDismiss: (() -> Void)? = nil,
                                @ViewBuilder content: @escaping () -> Content) -> some View {
        modifier(MGSheetModifier(isPresented: isPresented, id: id, onDismiss: onDismiss, sheet: content))
    }

    func mgSheet<Item: Identifiable, Content: View>(_ id: String, item: Binding<Item?>,
                                                    @ViewBuilder content: @escaping (Item) -> Content) -> some View {
        modifier(MGItemSheetModifier(item: item, id: id, sheet: content))
    }

    func mgAlert(_ title: String, isPresented: Binding<Bool>, message: String? = nil,
                 actions: @escaping () -> [MGAlertAction]) -> some View {
        modifier(MGAlertModifier(isPresented: isPresented, title: title, message: message, kind: .alert, actions: actions))
    }

    func mgConfirmationDialog(_ title: String, isPresented: Binding<Bool>, message: String? = nil,
                              actions: @escaping () -> [MGAlertAction]) -> some View {
        modifier(MGAlertModifier(isPresented: isPresented, title: title, message: message, kind: .dialog, actions: actions))
    }
}

// MARK: - prompt

/// A Sheet asking for one line of text (rename and the like).
struct MGPromptSheet: View {
    let title: String
    var message: String? = nil
    var actionTitle = "OK"
    @State var text: String
    let onCommit: (String) -> Void
    @Environment(\.mgDismiss) private var dismiss
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.mgHeadline)
            if let message { Text(message).font(.mgCallout).foregroundStyle(MG.secondaryLabel) }
            TextField("", text: $text)
                .focused($focused)
                .onSubmit { commit() }
            HStack(spacing: 8) {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(actionTitle) { commit() }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                    .disabled(text.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 320)
        .onAppear { focused = true }
    }

    func commit() {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return }
        dismiss()
        onCommit(t)
    }
}
