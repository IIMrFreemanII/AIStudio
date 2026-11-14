import AppKit
import Agent
import SwiftUI
import UIControl

/// The pill over every Space while the agent uses or records another app: what it is doing and a Stop button, so the
/// user can take back the mouse and keyboard at any moment.
@MainActor
final class UIControlPanel {
    static let shared = UIControlPanel()
    private var panel: NSPanel?
    private var host: NSHostingView<UIControlPill>?

    /// Shows the pill ("AIStudio is controlling <app>", or `verb` "recording"); `stop` ends the reply.
    func show(app: String, verb: String = "controlling", stop: @escaping () -> Void) {
        let pill = UIControlPill(app: app, verb: verb, stop: stop)
        if let panel, let host {
            host.rootView = pill
            place(panel, host)
            panel.orderFrontRegardless()
            return
        }
        let host = NSHostingView(rootView: pill)
        let panel = NSPanel(contentRect: .zero, styleMask: [.nonactivatingPanel, .borderless],
                            backing: .buffered, defer: false)
        panel.contentView = host
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.hidesOnDeactivate = false
        panel.isMovableByWindowBackground = true
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.isReleasedWhenClosed = false
        self.panel = panel
        self.host = host
        place(panel, host)
        panel.orderFrontRegardless()
    }

    func hide() { panel?.orderOut(nil) }

    /// Horizontally centred, 8 pt below the top of the main screen's usable area.
    private func place(_ panel: NSPanel, _ host: NSHostingView<UIControlPill>) {
        let size = host.fittingSize
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let f = screen.visibleFrame
        panel.setFrame(NSRect(x: f.midX - size.width / 2, y: f.maxY - 8 - size.height, width: size.width, height: size.height),
                       display: true)
    }
}

struct UIControlPill: View {
    let app: String
    let verb: String
    let stop: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            MGStateDot(kind: .busy)
            Text("AIStudio is \(verb) \(app)").font(.mgFootnote).foregroundStyle(MG.label).lineLimit(1)
            Button("Stop", action: stop)
                .buttonStyle(.mg(.bordered)).controlSize(.small)
                .mgTestAnchor("ui-control-stop")
        }
        .padding(.horizontal, 12)
        .frame(height: 30)
        .environment(\.mgGlassBlending, .behindWindow)
        .mgGlass(.floatingPanel, in: Capsule())
        .fixedSize()
        .padding(4)
        .mgTestAnchor("ui-control-panel")
    }
}

/// Above the composer when a `ui_*` tool found a macOS permission missing: what it is for, and the way to turn it on.
struct UIAccessCard: View {
    let doc: ChatDocument
    let kind: UIAccessKind

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top) {
                Text("Let AIStudio see and use apps").font(.mgCallout.weight(.semibold))
                Spacer(minLength: 4)
                Button { doc.uiAccessNeeded = nil } label: { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon)).controlSize(.small)
                    .accessibilityLabel("Close")
                    .mgTestAnchor("ui-access-close")
            }
            Text(kind == .accessibility
                 ? "Accessibility lets the agent read and click other apps' windows."
                 : "Screen Recording lets the agent take screenshots and recordings of a window.")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            Text("After turning it on, macOS may ask you to quit and reopen AIStudio.")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            HStack(spacing: 8) {
                Button("Open System Settings") {
                    UIDriver.requestAccess(accessibility: kind == .accessibility, screenRecording: kind == .screenRecording)
                    let pane = kind == .accessibility ? "Privacy_Accessibility" : "Privacy_ScreenCapture"
                    if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?" + pane) {
                        NSWorkspace.shared.open(url)
                    }
                }
                .buttonStyle(.mg(.prominent)).controlSize(.small)
                .mgTestAnchor("ui-access-open")
                Button("Check Again") {
                    let access = UIDriver(filesFolder: FileManager.default.temporaryDirectory).access()
                    if kind == .accessibility ? access.accessibility : access.screenRecording { doc.uiAccessNeeded = nil }
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small)
                .mgTestAnchor("ui-access-check")
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.controlBackground))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous), color: MG.border)
        .padding(.horizontal, 14).padding(.bottom, 6)
        .mgTestAnchor("ui-access-card")
    }
}
