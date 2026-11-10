import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

/// Paint an inpainting mask over an image. White (shown red) = regenerate. Saves a grayscale PNG at image size.
struct MaskEditorView: View {
    struct Stroke {
        var points: [CGPoint]  // normalised 0…1, origin top-left
        var width: CGFloat  // fraction of image width
        var erase: Bool
    }

    let imageURL: URL
    let existingMask: URL?
    let saveDirectory: URL
    let onSave: (URL) -> Void

    @Environment(\.mgDismiss) private var dismiss
    @State private var image: CGImage?
    @State private var baseMask: CGImage?
    @State private var baseOverlay: CGImage?
    @State private var strokes: [Stroke] = []
    @State private var current: Stroke?
    @State private var brush: Double = 0.06
    @State private var erase = false
    @State private var error: String?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                MGSegmentedPicker(selection: $erase, options: [(false, "Brush"), (true, "Eraser")])
                    .frame(width: 180)
                HStack(spacing: 6) {
                    Image(systemName: "circle.fill").font(.system(size: 6)).foregroundStyle(MG.secondaryLabel)
                    MGSlider(value: $brush, in: 0.01...0.25).frame(width: 140)
                    Image(systemName: "circle.fill").font(.system(size: 14)).foregroundStyle(MG.secondaryLabel)
                }
                .mgHelp("Brush size")
                Button("Undo Stroke") { _ = strokes.popLast() }
                    .disabled(strokes.isEmpty)
                    .keyboardShortcut("z")
                Button("Clear") { strokes = []; baseMask = nil; baseOverlay = nil }
                Spacer()
                if let error { Text(error).foregroundStyle(MG.destructive).font(.mgSubheadline) }
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Use Mask") { save() }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                    .disabled(strokes.isEmpty && baseMask == nil)
            }
            .padding(12)
            .mgChrome(MG.barTint, material: .headerView)
            MGDivider()
            GeometryReader { geo in
                if let image {
                    let rect = fitted(CGSize(width: image.width, height: image.height), in: geo.size)
                    ZStack(alignment: .topLeading) {
                        Image(decorative: image, scale: 1).resizable()
                            .frame(width: rect.width, height: rect.height)
                        Canvas { ctx, size in drawOverlay(&ctx, size: size) }
                            .frame(width: rect.width, height: rect.height)
                            .allowsHitTesting(false)
                    }
                    .frame(width: rect.width, height: rect.height)
                    .contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .local).onChanged { v in
                        let p = CGPoint(x: v.location.x / rect.width, y: v.location.y / rect.height)
                        if current == nil { current = Stroke(points: [], width: brush, erase: erase) }
                        current?.points.append(p)
                    }.onEnded { _ in
                        if let c = current { strokes.append(c) }
                        current = nil
                    })
                    .position(x: geo.size.width / 2, y: geo.size.height / 2)
                } else {
                    MGSpinner().frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            .background(MG.contentBackground)
            Text("Paint over the area to regenerate. Everything outside the mask stays pixel-identical.")
                .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).padding(8)
        }
        .frame(minWidth: 760, minHeight: 640)
        .task {
            image = Self.load(imageURL)
            if let existingMask {
                baseMask = Self.load(existingMask)
                baseOverlay = baseMask.flatMap { Self.redOverlay($0, maxSize: 1600) }
            }
        }
    }

    func fitted(_ size: CGSize, in box: CGSize) -> CGSize {
        let k = min((box.width - 24) / size.width, (box.height - 24) / size.height)
        return CGSize(width: size.width * k, height: size.height * k)
    }

    /// Mask shown as translucent red: opaque red drawn in a layer, composited at 50% so overlaps don't darken.
    func drawOverlay(_ ctx: inout GraphicsContext, size: CGSize) {
        let rect = CGRect(origin: .zero, size: size)
        ctx.opacity = 0.5
        ctx.drawLayer { layer in
            if let baseOverlay { layer.draw(Image(decorative: baseOverlay, scale: 1), in: rect) }
            for s in strokes + (current.map { [$0] } ?? []) {
                guard let first = s.points.first else { continue }
                var path = Path()
                path.move(to: CGPoint(x: first.x * size.width, y: first.y * size.height))
                for p in s.points.dropFirst() { path.addLine(to: CGPoint(x: p.x * size.width, y: p.y * size.height)) }
                if s.points.count == 1 { path.addLine(to: CGPoint(x: first.x * size.width + 0.1, y: first.y * size.height)) }
                var l = layer
                l.blendMode = s.erase ? .clear : .normal
                l.stroke(path, with: .color(.red),
                         style: StrokeStyle(lineWidth: s.width * size.width, lineCap: .round, lineJoin: .round))
            }
        }
    }

    /// Rasterises base mask + strokes at the image's pixel size.
    func save() {
        guard let image else { return }
        let (w, h) = (image.width, image.height)
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
        else { return }
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        if let baseMask { ctx.draw(baseMask, in: CGRect(x: 0, y: 0, width: w, height: h)) }
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)
        for s in strokes {
            guard let first = s.points.first else { continue }
            ctx.setStrokeColor(gray: s.erase ? 0 : 1, alpha: 1)
            ctx.setLineWidth(s.width * CGFloat(w))
            func pt(_ p: CGPoint) -> CGPoint { CGPoint(x: p.x * CGFloat(w), y: (1 - p.y) * CGFloat(h)) }
            ctx.move(to: pt(first))
            for p in s.points.dropFirst() { ctx.addLine(to: pt(p)) }
            if s.points.count == 1 { ctx.addLine(to: CGPoint(x: pt(first).x + 0.1, y: pt(first).y)) }
            ctx.strokePath()
        }
        guard let mask = ctx.makeImage() else { return }
        do {
            try FileManager.default.createDirectory(at: saveDirectory, withIntermediateDirectories: true)
            let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
            let url = saveDirectory.appendingPathComponent("mask-\(stamp).png")
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw CocoaError(.fileWriteUnknown) }
            CGImageDestinationAddImage(dest, mask, nil)
            guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
            onSave(url)
            dismiss()
        } catch {
            self.error = "Could not save mask: \(error.localizedDescription)"
        }
    }

    /// A red image whose alpha is the mask's luminance (white → opaque red, black → transparent).
    static func redOverlay(_ mask: CGImage, maxSize: Int? = nil) -> CGImage? {
        var w = mask.width, h = mask.height
        if let maxSize, max(w, h) > maxSize {
            let k = Double(maxSize) / Double(max(w, h))
            w = max(1, Int(Double(w) * k)); h = max(1, Int(Double(h) * k))
        }
        var gray = [UInt8](repeating: 0, count: w * h)
        gray.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            ctx?.interpolationQuality = .medium
            ctx?.draw(mask, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            let a = gray[i]
            rgba[i * 4] = a  // premultiplied red
            rgba[i * 4 + 3] = a
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
        return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    static func load(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}
