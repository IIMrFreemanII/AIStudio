import Foundation
import ImageIO
import MLX
import MLXDiffusionCore

/// Conditioning-image preprocessing of the reference image-to-video path (`load_image_and_preprocess`):
///
/// 1. decode to oriented sRGB RGB uint8 (``loadRGB(_:)``),
/// 2. H.264 4:2:0 round-trip at the checkpoint's image CRF (33 for LTX-2.3) — **not ported**: the reference shells out
///    to ffmpeg/libx264; pass a `compress` closure to ``conditioningPixels(rgb:height:width:compress:)`` to supply one
///    (CRF 0 = the closure omitted = passthrough),
/// 3. aspect-preserving bilinear resize (PyTorch `align_corners=False`, no antialias) to cover the target, center
///    crop (``resizeAndCenterCrop(_:height:width:)``),
/// 4. `x / 127.5 - 1`, bf16, shaped `[1, 1, H, W, 3]` for ``LTXVideoEncoder/encode(_:)``.
public enum LTXImagePreprocess {
    /// Loads an image as `[H, W, 3]` float32 holding uint8 values `0...255` (sRGB, EXIF rotation applied as the
    /// reference does: orientations 3, 6, 8).
    public static func loadRGB(_ url: URL) throws -> MLXArray {
        let cg = try ImageArray.load(url)
        var x = round(ImageArray.array(from: cg)[0..., 0..., 0 ..< 3] * 255)
        if let src = CGImageSourceCreateWithURL(url as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let o = props[kCGImagePropertyOrientation] as? Int
        {
            switch o {
            case 3: x = x[.stride(by: -1), .stride(by: -1)]  // 180°
            case 6: x = x.transposed(1, 0, 2)[0..., .stride(by: -1)]  // 90° clockwise
            case 8: x = x.transposed(1, 0, 2)[.stride(by: -1)]  // 90° counter-clockwise
            default: break
            }
        }
        eval(x)
        return x
    }

    /// One axis of PyTorch bilinear interpolation (`align_corners=False`, source clamped at 0, no antialias).
    static func resizeAxis(_ x: MLXArray, to outSize: Int, axis: Int) -> MLXArray {
        let inSize = x.dim(axis)
        let scale = Float(Double(inSize) / Double(outSize))
        var i0 = [Int32](), i1 = [Int32](), lam = [Float]()
        for i in 0..<outSize {
            let src = max((Float(i) + Float(0.5)) * scale - Float(0.5), 0)
            let a = min(Int(src.rounded(.down)), inSize - 1)
            i0.append(Int32(a))
            i1.append(Int32(min(a + 1, inSize - 1)))
            lam.append(src - Float(a))
        }
        var shape = [Int](repeating: 1, count: x.ndim)
        shape[axis] = outSize
        let l = MLXArray(lam).reshaped(shape)
        let a = x.take(MLXArray(i0), axis: axis)
        let b = x.take(MLXArray(i1), axis: axis)
        return a * (Float(1) - l) + b * l
    }

    /// Resize `[H, W, C]` (float, values `0...255`) to cover `height × width` keeping the aspect ratio (size rounded
    /// up), then center crop. Returns float32, not re-quantized.
    public static func resizeAndCenterCrop(_ image: MLXArray, height: Int, width: Int) -> MLXArray {
        let x = image.asType(.float32)
        let (sh, sw) = (x.dim(0), x.dim(1))
        let scale = max(Double(height) / Double(sh), Double(width) / Double(sw))
        let nh = Int((Double(sh) * scale).rounded(.up))
        let nw = Int((Double(sw) * scale).rounded(.up))
        let r = resizeAxis(resizeAxis(x, to: nh, axis: 0), to: nw, axis: 1)
        let top = (nh - height) / 2, left = (nw - width) / 2
        return r[top ..< (top + height), left ..< (left + width)]
    }

    /// Full conditioning preprocessing: `[H, W, 3]` uint8-valued RGB → `[1, 1, height, width, 3]` bf16 in `[-1, 1]`.
    /// `compress` stands in for the reference's H.264 CRF round-trip (it receives and returns uint8-valued RGB).
    public static func conditioningPixels(rgb: MLXArray, height: Int, width: Int,
                                          compress: ((MLXArray) throws -> MLXArray)? = nil) rethrows -> MLXArray {
        let src = try compress?(rgb) ?? rgb
        let f = resizeAndCenterCrop(src, height: height, width: width) / Float(127.5) - Float(1)
        return f.asType(.bfloat16).reshaped(1, 1, height, width, 3)
    }

    /// Loads, preprocesses and encodes a conditioning image to a normalized latent `[1, 1, height/32, width/32, 128]`
    /// (reference `combined_image_conditionings` per image; tokens are this reshaped to `[1, -1, 128]`).
    public static func encodeConditioning(url: URL, height: Int, width: Int, encoder: LTXVideoEncoder,
                                          compress: ((MLXArray) throws -> MLXArray)? = nil) throws -> MLXArray {
        let px = try conditioningPixels(rgb: loadRGB(url), height: height, width: width, compress: compress)
        let lat = encoder.encode(px)
        eval(lat)
        return lat
    }
}
