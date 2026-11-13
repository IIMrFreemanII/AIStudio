import SwiftUI

/// One status language for servers and model engines, wherever they show (the header's model chip, the sidebar's
/// server and model rows): a grey ring when idle or cold, an amber pulsing dot while something starts, loads, stops or
/// is checked, green when ready or online, red when it failed or is offline.
struct MGStateDot: View {
    enum Kind { case idle, busy, ready, failed }

    let kind: Kind
    var size: CGFloat = 7
    @State private var dim = false

    var body: some View {
        Group {
            switch kind {
            case .idle:
                Circle().strokeBorder(MG.secondaryLabel, lineWidth: 1.2)
            case .busy:
                Circle().fill(MG.warning)
                    .opacity(dim ? 0.35 : 1)
                    .onAppear { withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) { dim = true } }
            case .ready:
                Circle().fill(MG.success)
            case .failed:
                Circle().fill(MG.destructive)
            }
        }
        .frame(width: size, height: size)
        .accessibilityLabel(label)
    }

    var label: String {
        switch kind {
        case .idle: "Idle"
        case .busy: "Working"
        case .ready: "Ready"
        case .failed: "Failed"
        }
    }

    /// A model engine's state as the server reports it ("cold", "starting", "loading", "ready", "stopping", "failed").
    static func engine(_ state: String) -> Kind {
        switch state {
        case "starting", "loading", "stopping": .busy
        case "ready": .ready
        case "failed": .failed
        default: .idle
        }
    }
}
