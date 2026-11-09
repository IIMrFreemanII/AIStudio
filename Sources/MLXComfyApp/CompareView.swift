import CoreImage
import SwiftUI

/// Two images against each other: a draggable split, side by side, or their difference.
struct CompareView: View {
    @Environment(AppModel.self) private var model
    let a: CGImage
    let b: CGImage
    @State private var split: CGFloat = 0.5
    @State private var difference: CGImage?

    var body: some View {
        @Bindable var model = model
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                MGSegmentedPicker(selection: $model.compareMode, options: CompareMode.allCases.map { ($0, $0.rawValue) })
                    .frame(maxWidth: 340)
                Spacer()
                Text(caption).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(1).truncationMode(.middle)
                Button("Done") { model.endCompare() }.keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(MG.barOverContent)
            .overlay(alignment: .bottom) { MGDivider() }

            GeometryReader { geo in
                let frame = fitted(a, in: geo.size.insetBy(24))
                ZStack {
                    switch model.compareMode {
                    case .split:
                        // Drag (or click) anywhere on the image to move the divider: A on the left, B on the right.
                        ZStack {
                            image(a)
                            image(b).mask(alignment: .trailing) {
                                Rectangle().frame(width: frame.width * (1 - split))
                            }
                            splitHandle(frame)
                        }
                        .frame(width: frame.width, height: frame.height)
                        .contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                            split = min(max(v.location.x / frame.width, 0), 1)
                        })
                    case .sideBySide:
                        HStack(spacing: 12) {
                            image(a)
                            image(b)
                        }
                        .padding(24)
                    case .difference:
                        if let difference {
                            image(difference).frame(width: frame.width, height: frame.height)
                        } else {
                            ProgressView()
                        }
                    }
                }
                .frame(width: geo.size.width, height: geo.size.height)
            }
            .overlay(alignment: .bottom) {
                if model.compareMode == .split {
                    HStack {
                        Label("A", systemImage: "a.square").padding(.horizontal, 8).padding(.vertical, 4).mgGlass(.tooltip, in: Capsule())
                        Spacer()
                        Label("B", systemImage: "b.square").padding(.horizontal, 8).padding(.vertical, 4).mgGlass(.tooltip, in: Capsule())
                    }
                    .font(.mgSubheadline)
                    .padding(20)
                }
            }
        }
        .task(id: model.compareMode == .difference ? ObjectIdentifier(b) : nil) {
            guard model.compareMode == .difference, difference == nil else { return }
            let (a, b) = (a, b)
            difference = await Task.detached(priority: .userInitiated) { Self.difference(a, b) }.value
        }
    }

    var caption: String {
        let names = [model.displayedURL, model.compareURL].map { $0?.lastPathComponent ?? "current image" }
        return "A: \(names[0])   B: \(names[1])"
    }

    func image(_ img: CGImage) -> some View {
        Image(decorative: img, scale: 1).resizable().interpolation(.high).aspectRatio(contentMode: .fit)
    }

    func fitted(_ img: CGImage, in size: CGSize) -> CGSize {
        let s = min(size.width / CGFloat(img.width), size.height / CGFloat(img.height))
        return CGSize(width: CGFloat(img.width) * s, height: CGFloat(img.height) * s)
    }

    func splitHandle(_ frame: CGSize) -> some View {
        ZStack {
            Rectangle().fill(.white).frame(width: 2, height: frame.height).shadow(radius: 2)
            Circle().fill(.white).frame(width: 28, height: 28).shadow(radius: 3)
                .overlay(Image(systemName: "arrow.left.and.right").font(.caption.bold()).foregroundStyle(.black))
        }
        .offset(x: (split - 0.5) * frame.width)
        .allowsHitTesting(false)
    }

    /// |A − B| per pixel, B resized to A, brightened ×4 so small differences show.
    nonisolated static func difference(_ a: CGImage, _ b: CGImage) -> CGImage? {
        let ia = CIImage(cgImage: a)
        var ib = CIImage(cgImage: b)
        if a.width != b.width || a.height != b.height {
            ib = ib.transformed(by: CGAffineTransform(scaleX: CGFloat(a.width) / CGFloat(b.width),
                                                      y: CGFloat(a.height) / CGFloat(b.height)))
        }
        let diff = ia.applyingFilter("CIDifferenceBlendMode", parameters: [kCIInputBackgroundImageKey: ib])
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 4, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 4, z: 0, w: 0),
                "inputBVector": CIVector(x: 0, y: 0, z: 4, w: 0), "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 0),
            ])
        return CIContext().createCGImage(diff, from: ia.extent)
    }
}

private extension CGSize {
    func insetBy(_ d: CGFloat) -> CGSize { CGSize(width: max(width - 2 * d, 1), height: max(height - 2 * d, 1)) }
}
