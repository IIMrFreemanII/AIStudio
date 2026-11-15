import AppKit
import SwiftUI

/// Zoom and pan of an image shown fitted in a box. `zoom` 1 is fitted; `fit` is the fitted size's points per image
/// pixel, so the true scale is `zoom * fit` (1 = actual size, an image pixel per point).
struct ImageZoom: Equatable {
    var zoom: CGFloat = 1
    /// From the centre, in points.
    var offset: CGSize = .zero
    var fit: CGFloat = 1

    /// The true scale in percent: 100 is actual size.
    var percent: Int { Int((zoom * fit * 100).rounded()) }
    var isFitted: Bool { zoom == 1 && offset == .zero }

    /// From a quarter of the fitted size to 16× actual size (or 8× the fitted size, for a small image).
    var zoomRange: ClosedRange<CGFloat> { 0.25...max(8, 16 / max(fit, 0.001)) }

    /// Zooms to `newZoom`, keeping the point under `anchor` (from the centre, in points) where it is. Back at the
    /// fitted size or smaller, the image is centred again.
    mutating func set(_ newZoom: CGFloat, anchor: CGPoint = .zero) {
        let z = min(max(newZoom, zoomRange.lowerBound), zoomRange.upperBound)
        if z <= 1 {
            offset = .zero
        } else {
            offset = CGSize(width: anchor.x - (anchor.x - offset.width) * z / zoom,
                            height: anchor.y - (anchor.y - offset.height) * z / zoom)
        }
        zoom = z
    }

    /// A step in or out (⌘+ / ⌘−, the buttons): ×1.25, landing on actual size when it passes it.
    mutating func step(_ direction: Int, anchor: CGPoint = .zero) {
        var next = zoom * (direction > 0 ? 1.25 : 0.8)
        let actual = 1 / fit
        if (zoom - actual) * (next - actual) < 0 { next = actual }
        set(next, anchor: anchor)
    }

    mutating func fitToBox() { zoom = 1; offset = .zero }
    mutating func actualSize(anchor: CGPoint = .zero) { set(1 / fit, anchor: anchor) }
}

/// An image fitted to the space that zooms and pans:
/// - pinch, ⌘-scroll or a mouse's scroll wheel zooms around the pointer; two-finger scroll or a drag pans;
/// - double-click toggles between fitted and 2× (actual size for a small image) at the pointer;
/// - a click outside the image calls `onBackgroundTap`.
struct ZoomableImage: View {
    let image: CGImage
    @Binding var state: ImageZoom
    var padding: CGFloat = 24
    var onBackgroundTap: () -> Void = {}

    /// The pointer, from the centre; nil when it's elsewhere.
    @State private var hover: CGPoint?
    @State private var pinchStart: ImageZoom?
    @State private var dragStart: CGSize?
    @State private var monitor: Any?

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            let fit = min(max(1, size.width - 2 * padding) / CGFloat(image.width),
                          max(1, size.height - 2 * padding) / CGFloat(image.height))
            let fitted = CGSize(width: CGFloat(image.width) * fit, height: CGFloat(image.height) * fit)
            let fromCentre = { (p: CGPoint) in CGPoint(x: p.x - size.width / 2, y: p.y - size.height / 2) }
            ZStack {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .interpolation(state.zoom * fit > 2 ? .none : .high)  // crisp pixels when magnified
                    .frame(width: fitted.width, height: fitted.height)
                    .shadow(color: .black.opacity(0.35), radius: 12, y: 4)
                    .scaleEffect(state.zoom)
                    .offset(state.offset)
            }
            .frame(width: size.width, height: size.height)
            .contentShape(Rectangle())
            .onContinuousHover { phase in
                if case .active(let p) = phase { hover = fromCentre(p) } else { hover = nil }
            }
            .gesture(MagnifyGesture()
                .onChanged { v in
                    if pinchStart == nil { pinchStart = state }
                    var s = pinchStart!
                    s.set(pinchStart!.zoom * v.magnification, anchor: fromCentre(v.startLocation))
                    state = s
                }
                .onEnded { _ in pinchStart = nil })
            .simultaneousGesture(DragGesture(minimumDistance: 3)
                .onChanged { v in
                    guard state.zoom > 1 else { return }
                    if dragStart == nil { dragStart = state.offset }
                    state.offset = CGSize(width: dragStart!.width + v.translation.width,
                                          height: dragStart!.height + v.translation.height)
                }
                .onEnded { _ in dragStart = nil })
            .onTapGesture(count: 2) { p in
                withAnimation(.easeOut(duration: 0.15)) {
                    if state.isFitted { state.set(max(2, 1 / fit), anchor: fromCentre(p)) } else { state.fitToBox() }
                }
            }
            .onTapGesture { p in
                let shown = CGRect(x: state.offset.width - fitted.width * state.zoom / 2,
                                   y: state.offset.height - fitted.height * state.zoom / 2,
                                   width: fitted.width * state.zoom, height: fitted.height * state.zoom)
                if !shown.contains(fromCentre(p)) { onBackgroundTap() }
            }
            .onChange(of: fit, initial: true) { _, f in state.fit = f }
        }
        .onAppear(perform: installScrollMonitor)
        .onDisappear(perform: removeScrollMonitor)
    }

    // MARK: scroll wheel

    /// SwiftUI has no scroll-wheel gesture on macOS. A mouse's wheel (or ⌘ with any scroll) zooms around the pointer,
    /// like the graph's ⌘-scroll; a trackpad's two-finger scroll pans a magnified image.
    func installScrollMonitor() {
        guard monitor == nil else { return }
        monitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) { event in
            guard let at = hover else { return event }
            if !event.hasPreciseScrollingDeltas || event.modifierFlags.contains(.command) {
                let k: CGFloat = event.hasPreciseScrollingDeltas ? 0.01 : 0.1
                state.set(state.zoom * exp(event.scrollingDeltaY * k), anchor: at)
            } else if state.zoom > 1 {
                state.offset = CGSize(width: state.offset.width + event.scrollingDeltaX,
                                      height: state.offset.height + event.scrollingDeltaY)
            }
            return nil
        }
    }

    func removeScrollMonitor() {
        if let monitor { NSEvent.removeMonitor(monitor) }
        monitor = nil
    }
}
