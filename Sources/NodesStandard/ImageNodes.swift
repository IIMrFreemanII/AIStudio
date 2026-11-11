import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore

/// Shared `tiled` input of VAE Encode / VAE Decode.
let tilingInput = InputSpec("tiled", .string, default: .string(VAETiling.auto.rawValue),
                            widget: .choice(VAETiling.allCases.map(\.rawValue)))

func tiling(_ inputs: NodeInputs) throws -> VAETiling {
    VAETiling(rawValue: try inputs.string("tiled")) ?? .auto
}

/// Grows (or shrinks, negative) a mask and softens its edge, so a composite blends into the original instead of
/// leaving a seam.
public struct FeatherMaskNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "FeatherMask", displayName: "Feather Mask", category: "image",
        inputs: [
            InputSpec("mask", .mask),
            InputSpec("grow", .int, default: .int(0), widget: .number(min: -256, max: 256, step: 1)),
            InputSpec("feather", .int, default: .int(16), widget: .number(min: 0, max: 256, step: 1)),
        ],
        outputs: [OutputSpec("mask", .mask)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let mask: MLXArray = try inputs.object("mask")
        let out = MaskOps.feather(mask, grow: try inputs.int("grow"), radius: Float(try inputs.int("feather")))
        return ["mask": .object(out)]
    }
}

/// Extends the canvas for outpainting; the mask is white over the new border.
public struct PadImageNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "PadImage", displayName: "Pad Image", category: "image",
        inputs: [
            InputSpec("image", .image),
            InputSpec("left", .int, default: .int(0), widget: .number(min: 0, max: 2048, step: 16)),
            InputSpec("top", .int, default: .int(0), widget: .number(min: 0, max: 2048, step: 16)),
            InputSpec("right", .int, default: .int(0), widget: .number(min: 0, max: 2048, step: 16)),
            InputSpec("bottom", .int, default: .int(0), widget: .number(min: 0, max: 2048, step: 16)),
            InputSpec("fill", .string, default: .string(MaskOps.Fill.edge.rawValue),
                      widget: .choice(MaskOps.Fill.allCases.map(\.rawValue))),
        ],
        outputs: [OutputSpec("image", .image), OutputSpec("mask", .mask)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let image: MLXArray = try inputs.object("image")
        let (out, mask) = MaskOps.pad(
            image, left: try inputs.int("left"), top: try inputs.int("top"), right: try inputs.int("right"),
            bottom: try inputs.int("bottom"), fill: MaskOps.Fill(rawValue: try inputs.string("fill")) ?? .edge)
        return ["image": .object(out), "mask": .object(mask)]
    }
}

/// Resizes an image by `factor`, or to `width` × `height` (0 keeps the aspect ratio from the other side).
public struct ImageScaleNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: "ImageScale", displayName: "Scale Image", category: "image",
        inputs: [
            InputSpec("image", .image),
            // > 0 wins over width/height.
            InputSpec("factor", .float, default: .double(1.5), widget: .number(min: 0, max: 4, step: 0.25)),
            InputSpec("width", .int, default: .int(0), widget: .number(min: 0, max: 4096, step: 16), zeroLabel: "auto"),
            InputSpec("height", .int, default: .int(0), widget: .number(min: 0, max: 4096, step: 16), zeroLabel: "auto"),
        ],
        outputs: [OutputSpec("image", .image)])
    public init() {}

    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        let image: MLXArray = try inputs.object("image")
        let (h, w) = (image.dim(0), image.dim(1))
        let size = Self.targetSize(width: w, height: h, factor: try inputs.float("factor"),
                                   targetWidth: try inputs.int("width"), targetHeight: try inputs.int("height"))
        return ["image": .object(try ImageArray.resized(image, width: size.width, height: size.height))]
    }

    static func targetSize(width w: Int, height h: Int, factor: Double, targetWidth: Int, targetHeight: Int)
        -> (width: Int, height: Int)
    {
        if factor > 0 { return (max(1, Int((Double(w) * factor).rounded())), max(1, Int((Double(h) * factor).rounded()))) }
        switch (targetWidth, targetHeight) {
        case (0, 0): return (w, h)
        case (let tw, 0): return (tw, max(1, Int((Double(h) * Double(tw) / Double(w)).rounded())))
        case (0, let th): return (max(1, Int((Double(w) * Double(th) / Double(h)).rounded())), th)
        case (let tw, let th): return (tw, th)
        }
    }
}
