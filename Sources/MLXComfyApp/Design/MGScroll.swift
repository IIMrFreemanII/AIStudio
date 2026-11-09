import AppKit
import MLXComfyLibrary
import SwiftUI

// The design system's ScrollIndicator in place of the system's scrollers: a 5 pt thumb with square ends, 2 pt from
// the edge, shown while scrolling or hovering and faded a second after. It can be dragged.

/// Snapshots show the thumbs without scrolling first.
@MainActor enum MGScrollDebug { static var alwaysVisible = false }

struct MGScrollMetrics: Equatable {
    var content: CGSize
    var viewport: CGSize
    var offset: CGPoint
}

/// The thumbs over a scrolling view.
private struct MGThumbOverlay: View {
    let metrics: MGScrollMetrics?
    let axes: Axis.Set
    let flash: Int
    let scrollTo: (CGPoint) -> Void
    @State private var visible = false
    @State private var hovering = false
    @State private var dragStart: CGFloat?
    @State private var hideTask: Task<Void, Never>?

    var body: some View {
        GeometryReader { geo in
            if let m = metrics {
                if axes.contains(.vertical),
                   let t = ScrollThumb.make(content: m.content.height, viewport: m.viewport.height, offset: m.offset.y,
                                            track: geo.size.height - 4) {
                    thumb(width: 5, height: t.length)
                        .position(x: geo.size.width - 4.5, y: 2 + t.start + t.length / 2)
                        .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                            if dragStart == nil { dragStart = t.start }
                            let y = ScrollThumb.offset(forStart: dragStart! + g.translation.height, content: m.content.height,
                                                       viewport: m.viewport.height, track: geo.size.height - 4)
                            scrollTo(CGPoint(x: m.offset.x, y: y))
                        }.onEnded { _ in dragStart = nil; scheduleHide() })
                }
                if axes.contains(.horizontal),
                   let t = ScrollThumb.make(content: m.content.width, viewport: m.viewport.width, offset: m.offset.x,
                                            track: geo.size.width - 4) {
                    thumb(width: t.length, height: 5)
                        .position(x: 2 + t.start + t.length / 2, y: geo.size.height - 4.5)
                        .gesture(DragGesture(minimumDistance: 0).onChanged { g in
                            if dragStart == nil { dragStart = t.start }
                            let x = ScrollThumb.offset(forStart: dragStart! + g.translation.width, content: m.content.width,
                                                       viewport: m.viewport.width, track: geo.size.width - 4)
                            scrollTo(CGPoint(x: x, y: m.offset.y))
                        }.onEnded { _ in dragStart = nil; scheduleHide() })
                }
            }
        }
        .onChange(of: flash) { show() }
    }

    func thumb(width: CGFloat, height: CGFloat) -> some View {
        Rectangle()
            .fill(MG.scrollIndicator)
            .frame(width: width, height: height)
            .opacity(visible || hovering || dragStart != nil || MGScrollDebug.alwaysVisible ? 1 : 0)
            .animation(.easeOut(duration: 0.25), value: visible)
            .contentShape(Rectangle().inset(by: -3))
            .onHover { h in hovering = h; if !h { scheduleHide() } }
            .allowsHitTesting(visible || hovering || dragStart != nil)
    }

    func show() {
        visible = true
        scheduleHide()
    }

    func scheduleHide() {
        hideTask?.cancel()
        hideTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(1))
            if !Task.isCancelled, dragStart == nil, !hovering { visible = false }
        }
    }
}

/// For SwiftUI's ScrollView: its geometry drives the thumbs.
private struct MGScrollIndicators: ViewModifier {
    let axes: Axis.Set
    @State private var metrics: MGScrollMetrics?
    @State private var flash = 0
    @State private var position = ScrollPosition()

    func body(content: Content) -> some View {
        content
            .scrollIndicators(.never)
            .scrollPosition($position)
            .onScrollGeometryChange(for: MGScrollMetrics.self) { g in
                MGScrollMetrics(
                    content: g.contentSize,
                    viewport: CGSize(width: g.containerSize.width - g.contentInsets.leading - g.contentInsets.trailing,
                                     height: g.containerSize.height - g.contentInsets.top - g.contentInsets.bottom),
                    offset: CGPoint(x: g.contentOffset.x + g.contentInsets.leading, y: g.contentOffset.y + g.contentInsets.top))
            } action: { old, new in
                metrics = new
                if old.offset != new.offset { flash += 1 }
            }
            .overlay { MGThumbOverlay(metrics: metrics, axes: axes, flash: flash) { position.scrollTo(point: $0) } }
            .onHover { if $0 { flash += 1 } }
    }
}

/// For views AppKit scrolls (List, TextEditor): finds the NSScrollView under this view, hides its scrollers and
/// follows its clip view.
@MainActor @Observable
final class MGAppKitScrollModel {
    var metrics: MGScrollMetrics?
    var flash = 0
    @ObservationIgnored weak var scrollView: NSScrollView?
    @ObservationIgnored private var observers: [NSObjectProtocol] = []

    func attach(_ sv: NSScrollView) {
        if sv !== scrollView {
            observers.forEach(NotificationCenter.default.removeObserver)
            observers.removeAll()
            scrollView = sv
            sv.contentView.postsBoundsChangedNotifications = true
            sv.documentView?.postsFrameChangedNotifications = true
            for (name, obj) in [(NSView.boundsDidChangeNotification, sv.contentView as NSView?),
                                (NSView.frameDidChangeNotification, sv.documentView),
                                (NSView.frameDidChangeNotification, sv as NSView?)] {
                observers.append(NotificationCenter.default.addObserver(forName: name, object: obj, queue: .main) { [weak self] _ in
                    MainActor.assumeIsolated { self?.update(scrolled: name == NSView.boundsDidChangeNotification) }
                })
            }
        }
        sv.hasVerticalScroller = false
        sv.hasHorizontalScroller = false
        update(scrolled: false)
    }

    func update(scrolled: Bool) {
        guard let sv = scrollView, let doc = sv.documentView else { return }
        let insets = sv.contentInsets
        let clip = sv.contentView.bounds
        let new = MGScrollMetrics(content: doc.frame.size,
                                  viewport: CGSize(width: clip.width - insets.left - insets.right,
                                                   height: clip.height - insets.top - insets.bottom),
                                  offset: CGPoint(x: clip.origin.x + insets.left, y: clip.origin.y + insets.top))
        if new != metrics { metrics = new }
        if scrolled { flash += 1 }
        if sv.hasVerticalScroller || sv.hasHorizontalScroller {
            sv.hasVerticalScroller = false
            sv.hasHorizontalScroller = false
        }
    }

    func scroll(to p: CGPoint) {
        guard let sv = scrollView else { return }
        let insets = sv.contentInsets
        sv.contentView.scroll(to: NSPoint(x: p.x - insets.left, y: p.y - insets.top))
        sv.reflectScrolledClipView(sv.contentView)
    }
}

private struct MGScrollFinder: NSViewRepresentable {
    let model: MGAppKitScrollModel

    final class FinderView: MGPassthroughView {
        var model: MGAppKitScrollModel?
        override func viewDidMoveToWindow() { super.viewDidMoveToWindow(); find() }
        override func layout() { super.layout(); find() }

        func find() {
            guard let root = window?.contentView, let model else { return }
            let me = convert(bounds, to: nil)
            guard me.width > 0, me.height > 0 else { return }
            var best: (NSScrollView, CGFloat)?
            func walk(_ v: NSView) {
                if let sv = v as? NSScrollView {
                    let r = sv.convert(sv.bounds, to: nil)
                    let overlap = r.intersection(me)
                    if !overlap.isNull, overlap.width * overlap.height > me.width * me.height * 0.8 {
                        let area = r.width * r.height
                        if best == nil || area < best!.1 { best = (sv, area) }
                    }
                }
                v.subviews.forEach(walk)
            }
            walk(root)
            if let sv = best?.0 { DispatchQueue.main.async { model.attach(sv) } }
        }
    }

    func makeNSView(context: Context) -> FinderView {
        let v = FinderView()
        v.model = model
        return v
    }
    func updateNSView(_ v: FinderView, context: Context) { v.model = model }
}

private struct MGAppKitScrollIndicators: ViewModifier {
    let axes: Axis.Set
    @State private var model = MGAppKitScrollModel()

    func body(content: Content) -> some View {
        content
            .scrollIndicators(.never)
            .overlay {
                ZStack {
                    MGScrollFinder(model: model).allowsHitTesting(false)
                    MGThumbOverlay(metrics: model.metrics, axes: axes, flash: model.flash) { model.scroll(to: $0) }
                }
            }
            .onHover { if $0 { model.flash += 1 } }
    }
}

extension View {
    /// The design system's scroll indicators on a SwiftUI ScrollView.
    func mgScrollIndicators(_ axes: Axis.Set = .vertical) -> some View {
        modifier(MGScrollIndicators(axes: axes))
    }

    /// The design system's scroll indicators on a view AppKit scrolls (List, TextEditor).
    func mgAppKitScrollIndicators(_ axes: Axis.Set = .vertical) -> some View {
        modifier(MGAppKitScrollIndicators(axes: axes))
    }
}
