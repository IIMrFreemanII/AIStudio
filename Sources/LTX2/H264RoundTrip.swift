import CoreMedia
import CoreVideo
import Foundation
import MLX
import VideoToolbox

/// Encodes one frame with H.264 and decodes it back (VideoToolbox, synchronous), so a conditioning image carries the
/// compression the model saw in training. The reference uses libx264 at CRF 33 (yuv420p); VideoToolbox quality 0.28
/// gives the same PSNR on photographic images (~30 dB).
enum H264RoundTrip {
    /// `rgb`: `[H, W, 3]` with uint8 values. Odd sizes are cropped to even, as the reference does. Returns the input
    /// unchanged if VideoToolbox fails.
    static func apply(_ rgb: MLXArray, quality: Float = 0.28) -> MLXArray {
        let h = rgb.dim(0) / 2 * 2, w = rgb.dim(1) / 2 * 2
        guard h >= 16, w >= 16 else { return rgb }
        let src = rgb[0..<h, 0..<w, 0...]
        guard let input = pixelBuffer(from: src), let encoded = encode(input, width: w, height: h, quality: quality),
              let decoded = decode(encoded) else { return src }
        return array(from: decoded, width: w, height: h) ?? src
    }

    static func pixelBuffer(from rgb: MLXArray) -> CVPixelBuffer? {
        let h = rgb.dim(0), w = rgb.dim(1)
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, attrs, &pb) == kCVReturnSuccess, let pb else { return nil }
        let alpha = MLXArray.full([h, w, 1], values: MLXArray(Float(255)))
        let bgra = concatenated([rgb[0..., 0..., 2..<3], rgb[0..., 0..., 1..<2], rgb[0..., 0..., 0..<1], alpha], axis: 2)
        let bytes = clip(bgra, min: 0, max: 255).asType(.uint8).asArray(UInt8.self)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        bytes.withUnsafeBytes { raw in
            for y in 0..<h {
                memcpy(base + y * stride, raw.baseAddress! + y * w * 4, w * 4)
            }
        }
        return pb
    }

    static func encode(_ frame: CVPixelBuffer, width: Int, height: Int, quality: Float) -> CMSampleBuffer? {
        var session: VTCompressionSession?
        guard VTCompressionSessionCreate(allocator: nil, width: Int32(width), height: Int32(height),
                                         codecType: kCMVideoCodecType_H264, encoderSpecification: nil,
                                         imageBufferAttributes: nil, compressedDataAllocator: nil, outputCallback: nil,
                                         refcon: nil, compressionSessionOut: &session) == noErr, let session else { return nil }
        defer { VTCompressionSessionInvalidate(session) }
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_RealTime, value: kCFBooleanFalse)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_AllowFrameReordering, value: kCFBooleanFalse)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_ProfileLevel, value: kVTProfileLevel_H264_High_AutoLevel)
        VTSessionSetProperty(session, key: kVTCompressionPropertyKey_Quality, value: NSNumber(value: quality))
        var out: CMSampleBuffer?
        let status = VTCompressionSessionEncodeFrame(session, imageBuffer: frame, presentationTimeStamp: .zero,
                                                     duration: .invalid, frameProperties: nil, infoFlagsOut: nil) {
            status, _, sample in if status == noErr { out = sample }
        }
        guard status == noErr else { return nil }
        VTCompressionSessionCompleteFrames(session, untilPresentationTimeStamp: .invalid)
        return out
    }

    static func decode(_ sample: CMSampleBuffer) -> CVPixelBuffer? {
        guard let format = CMSampleBufferGetFormatDescription(sample) else { return nil }
        let attrs = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA] as CFDictionary
        var session: VTDecompressionSession?
        guard VTDecompressionSessionCreate(allocator: nil, formatDescription: format, decoderSpecification: nil,
                                           imageBufferAttributes: attrs, outputCallback: nil,
                                           decompressionSessionOut: &session) == noErr, let session else { return nil }
        defer { VTDecompressionSessionInvalidate(session) }
        var out: CVPixelBuffer?
        // No asynchronous-decompression flag: the handler runs before DecodeFrame returns.
        let status = VTDecompressionSessionDecodeFrame(session, sampleBuffer: sample, flags: [], infoFlagsOut: nil) {
            status, _, image, _, _ in if status == noErr { out = image }
        }
        guard status == noErr else { return nil }
        VTDecompressionSessionWaitForAsynchronousFrames(session)
        return out
    }

    static func array(from pb: CVPixelBuffer, width w: Int, height h: Int) -> MLXArray? {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb), CVPixelBufferGetWidth(pb) >= w,
              CVPixelBufferGetHeight(pb) >= h else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes { raw in
            for y in 0..<h { memcpy(raw.baseAddress! + y * w * 4, base + y * stride, w * 4) }
        }
        let bgra = MLXArray(bytes, [h, w, 4]).asType(.float32)
        return concatenated([bgra[0..., 0..., 2..<3], bgra[0..., 0..., 1..<2], bgra[0..., 0..., 0..<1]], axis: 2)
    }
}
