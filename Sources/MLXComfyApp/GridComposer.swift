import AppKit
import CoreGraphics
import GraphRuntime

/// Lays out XY plot cells in a labelled grid: X values across the top, Y values down the left side.
enum GridComposer {
    struct Layout: Equatable {
        var cell: CGSize
        var left: CGFloat
        var top: CGFloat
        var gap: CGFloat
        var size: CGSize
        var fontSize: CGFloat
    }

    /// Cell size fits the largest image, scaled so the grid stays under `maxWidth` pixels wide.
    static func layout(columns: Int, rows: Int, imageSizes: [CGSize], hasRowLabels: Bool,
                       maxWidth: CGFloat = 6144) -> Layout {
        let w = imageSizes.map(\.width).max() ?? 512, h = imageSizes.map(\.height).max() ?? 512
        let scale = min(1, maxWidth / (CGFloat(columns) * w))
        let cell = CGSize(width: (w * scale).rounded(), height: (h * scale).rounded())
        let font = min(max(cell.width * 0.04, 16), 44).rounded()
        let gap = (font * 0.5).rounded()
        let top = (font * 2.2).rounded()
        let left = hasRowLabels ? (cell.width * 0.32).rounded() : gap
        let size = CGSize(width: left + CGFloat(columns) * cell.width + CGFloat(columns + 1) * gap,
                          height: top + CGFloat(rows) * cell.height + CGFloat(rows + 1) * gap)
        return Layout(cell: cell, left: left, top: top, gap: gap, size: size, fontSize: font)
    }

    /// Origin (top-left, in a top-left-origin canvas) of cell (`column`, `row`).
    static func origin(_ l: Layout, column: Int, row: Int) -> CGPoint {
        CGPoint(x: l.left + l.gap + CGFloat(column) * (l.cell.width + l.gap),
                y: l.top + l.gap + CGFloat(row) * (l.cell.height + l.gap))
    }

    static func compose(_ spec: PlotSpec, images: [Int: CGImage]) -> CGImage {
        let sizes = images.values.map { CGSize(width: $0.width, height: $0.height) }
        let l = layout(columns: spec.columns, rows: spec.rows, imageSizes: sizes, hasRowLabels: spec.y != nil)
        let (W, H) = (Int(l.size.width), Int(l.size.height))
        let ctx = CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        // Flip so drawing uses a top-left origin like the layout.
        ctx.translateBy(x: 0, y: CGFloat(H))
        ctx.scaleBy(x: 1, y: -1)
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))

        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: true)
        let para = NSMutableParagraphStyle()
        para.alignment = .center
        para.lineBreakMode = .byTruncatingTail
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: l.fontSize, weight: .semibold),
            .foregroundColor: NSColor(white: 0.12, alpha: 1), .paragraphStyle: para,
        ]
        for c in 0..<spec.columns {
            let o = origin(l, column: c, row: 0)
            (spec.x.labels[c] as NSString).draw(
                in: CGRect(x: o.x, y: (l.top - l.fontSize * 1.3) / 2, width: l.cell.width, height: l.fontSize * 1.4),
                withAttributes: attrs)
        }
        if let y = spec.y {
            let rowPara = NSMutableParagraphStyle()
            rowPara.alignment = .right
            rowPara.lineBreakMode = .byWordWrapping
            var rowAttrs = attrs
            rowAttrs[.paragraphStyle] = rowPara
            for r in 0..<spec.rows {
                let o = origin(l, column: 0, row: r)
                let box = CGRect(x: l.gap, y: o.y, width: l.left - l.gap * 1.5, height: l.cell.height)
                let text = y.labels[r] as NSString
                let h = text.boundingRect(with: box.size, options: [.usesLineFragmentOrigin], attributes: rowAttrs).height
                text.draw(with: CGRect(x: box.minX, y: box.midY - h / 2, width: box.width, height: h),
                          options: [.usesLineFragmentOrigin], attributes: rowAttrs)
            }
        }
        NSGraphicsContext.restoreGraphicsState()

        for r in 0..<spec.rows {
            for c in 0..<spec.columns {
                let o = origin(l, column: c, row: r)
                let rect = CGRect(origin: o, size: l.cell)
                guard let img = images[r * spec.columns + c] else {
                    ctx.setFillColor(CGColor(gray: 0.9, alpha: 1))
                    ctx.fill(rect)
                    continue
                }
                // Aspect-fit, drawn upright inside the flipped context.
                let s = min(rect.width / CGFloat(img.width), rect.height / CGFloat(img.height))
                let dw = CGFloat(img.width) * s, dh = CGFloat(img.height) * s
                let dst = CGRect(x: rect.midX - dw / 2, y: rect.midY - dh / 2, width: dw, height: dh)
                ctx.saveGState()
                ctx.translateBy(x: dst.minX, y: dst.maxY)
                ctx.scaleBy(x: 1, y: -1)
                ctx.interpolationQuality = .high
                ctx.draw(img, in: CGRect(origin: .zero, size: dst.size))
                ctx.restoreGState()
            }
        }
        return ctx.makeImage()!
    }
}
