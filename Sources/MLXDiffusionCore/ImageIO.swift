import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

public enum ImageIOError: Error, CustomStringConvertible {
    case cannotRead(URL)
    case cannotWrite(URL)
    case badArray([Int])

    public var description: String {
        switch self {
        case .cannotRead(let u): return "cannot read image at \(u.path)"
        case .cannotWrite(let u): return "cannot write image to \(u.path)"
        case .badArray(let s): return "expected [H, W, C] image array, got \(s)"
        }
    }
}

public enum ImageResize {
    /// Resize with Core Graphics' high-quality (Lanczos-like) filter.
    case highQuality
}

/// Images as MLX tensors: `[H, W, C]` float32 in `[0, 1]`, C = 4 (RGBA, alpha premultiplication undone).
public enum ImageArray {

    public static func load(_ url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw ImageIOError.cannotRead(url) }
        return img
    }

    /// Renders `image` (optionally resized) into straight RGBA8 and returns `[H, W, 4]` float32 in `[0, 1]`.
    public static func array(from image: CGImage, width: Int? = nil, height: Int? = nil) -> MLXArray {
        let w = width ?? image.width
        let h = height ?? image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        bytes.withUnsafeMutableBytes { buf in
            let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        // Undo premultiplication so RGB matches PIL's straight-alpha convention.
        var floats = [Float](repeating: 0, count: w * h * 4)
        for i in 0..<(w * h) {
            let a = Float(bytes[i * 4 + 3]) / 255
            for c in 0..<3 {
                let v = Float(bytes[i * 4 + c]) / 255
                floats[i * 4 + c] = a > 0 ? min(v / a, 1) : 0
            }
            floats[i * 4 + 3] = a
        }
        return MLXArray(floats, [h, w, 4])
    }

    /// `[H, W, C]` (C = 3 or 4) float in `[0, 1]` → CGImage (RGB, or RGBA when `keepAlpha`).
    public static func cgImage(from array: MLXArray, keepAlpha: Bool = false) throws -> CGImage {
        guard array.ndim == 3, [3, 4].contains(array.dim(2)) else { throw ImageIOError.badArray(array.shape) }
        let h = array.dim(0), w = array.dim(1), c = array.dim(2)
        var rgba = clip(array.asType(.float32), min: 0, max: 1)
        if c == 3 { rgba = concatenated([rgba, MLXArray.ones([h, w, 1])], axis: 2) }
        if !keepAlpha { rgba = concatenated([rgba[0..., 0..., ..<3], MLXArray.ones([h, w, 1])], axis: 2) }
        let u8 = (rgba * 255 + 0.5).asType(.uint8)
        eval(u8)
        var bytes = u8.asArray(UInt8.self)
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let info: CGImageAlphaInfo = keepAlpha ? .last : .noneSkipLast
        let provider = CGDataProvider(data: Data(bytes: &bytes, count: bytes.count) as CFData)!
        guard let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                                space: cs, bitmapInfo: CGBitmapInfo(rawValue: info.rawValue), provider: provider,
                                decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw ImageIOError.badArray(array.shape) }
        return img
    }

    /// Writes a PNG with one uncompressed `iTXt` chunk per `text` entry (keyword → UTF-8 value).
    ///
    /// ImageIO would fold PNG comments into XMP, so the chunks are spliced in by hand; any PNG reader that
    /// understands `iTXt` (including ComfyUI-style tooling) can see them.
    public static func savePNG(_ image: CGImage, to url: URL, text: [String: String] = [:]) throws {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw ImageIOError.cannotWrite(url) }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw ImageIOError.cannotWrite(url) }
        var png = data as Data
        if !text.isEmpty { png = try PNGText.insert(text, into: png) }
        do { try png.write(to: url, options: .atomic) } catch { throw ImageIOError.cannotWrite(url) }
    }

    /// Reads the `tEXt`/`iTXt` entries of a PNG (empty if none or not a PNG).
    public static func readPNGText(_ url: URL) -> [String: String] {
        guard let data = try? Data(contentsOf: url) else { return [:] }
        return PNGText.read(data)
    }

    public static func savePNG(_ array: MLXArray, to url: URL, keepAlpha: Bool = false,
                               text: [String: String] = [:]) throws {
        try savePNG(cgImage(from: array, keepAlpha: keepAlpha), to: url, text: text)
    }
}

/// Minimal PNG text-chunk reader/writer.
public enum PNGText {
    static let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

    struct Chunk {
        let type: String
        let range: Range<Int>  // whole chunk including length/type/crc
        let body: Data
    }

    static func chunks(_ data: Data) -> [Chunk]? {
        let bytes = [UInt8](data)
        guard bytes.count > 8, Array(bytes[0..<8]) == signature else { return nil }
        var out: [Chunk] = []
        var i = 8
        while i + 12 <= bytes.count {
            let len = Int(bytes[i]) << 24 | Int(bytes[i + 1]) << 16 | Int(bytes[i + 2]) << 8 | Int(bytes[i + 3])
            guard i + 12 + len <= bytes.count else { return nil }
            let type = String(bytes: bytes[(i + 4)..<(i + 8)], encoding: .ascii) ?? "????"
            out.append(Chunk(type: type, range: i..<(i + 12 + len), body: Data(bytes[(i + 8)..<(i + 8 + len)])))
            i += 12 + len
            if type == "IEND" { break }
        }
        return out
    }

    public static func read(_ data: Data) -> [String: String] {
        var result: [String: String] = [:]
        for c in chunks(data) ?? [] {
            let b = [UInt8](c.body)
            guard let nul = b.firstIndex(of: 0) else { continue }
            let key = String(decoding: b[..<nul], as: UTF8.self)
            if c.type == "tEXt" {
                result[key] = String(decoding: b[(nul + 1)...], as: UTF8.self)  // Latin-1 in theory; UTF-8 in practice
            } else if c.type == "iTXt" {
                // keyword\0 compressionFlag compressionMethod languageTag\0 translatedKeyword\0 text
                var j = nul + 1
                guard j + 2 <= b.count, b[j] == 0 else { continue }  // compressed iTXt not supported
                j += 2
                guard let langEnd = b[j...].firstIndex(of: 0) else { continue }
                guard let transEnd = b[(langEnd + 1)...].firstIndex(of: 0) else { continue }
                result[key] = String(decoding: b[(transEnd + 1)...], as: UTF8.self)
            }
        }
        return result
    }

    public static func insert(_ text: [String: String], into data: Data) throws -> Data {
        guard let cs = chunks(data), let iend = cs.last(where: { $0.type == "IEND" }) else {
            throw ImageIOError.badArray([])
        }
        var out = Data(data[..<iend.range.lowerBound])
        for (k, v) in text.sorted(by: { $0.key < $1.key }) {
            var body = Data(k.utf8.prefix(79))
            body.append(contentsOf: [0, 0, 0, 0, 0])  // nul, uncompressed, method 0, empty lang\0, empty translated\0
            body.append(Data(v.utf8))
            out.append(chunk(type: "iTXt", body: body))
        }
        out.append(data[iend.range])
        return out
    }

    static func chunk(type: String, body: Data) -> Data {
        var d = Data()
        let n = UInt32(body.count)
        d.append(contentsOf: [UInt8(n >> 24), UInt8((n >> 16) & 0xFF), UInt8((n >> 8) & 0xFF), UInt8(n & 0xFF)])
        let typeAndBody = Data(type.utf8) + body
        d.append(typeAndBody)
        let c = crc32(typeAndBody)
        d.append(contentsOf: [UInt8(c >> 24), UInt8((c >> 16) & 0xFF), UInt8((c >> 8) & 0xFF), UInt8(c & 0xFF)])
        return d
    }

    static let crcTable: [UInt32] = (0..<256).map { n -> UInt32 in
        var c = UInt32(n)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in data { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }
}
