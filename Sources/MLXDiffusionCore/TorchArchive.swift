import Foundation
import MLX

/// Reads raw tensor storage out of a PyTorch `torch.save` file (a stored, uncompressed zip with `data.pkl` and
/// `data/<n>` storage entries) without unpickling anything. The caller supplies the layout; only float32 storage
/// is supported. Used for ACE-Step's `silence_latent.pt`.
public enum TorchArchive {
    public enum Failure: Error, CustomStringConvertible {
        case notZip, missing(String), compressed(String), badSize(String, Int)
        public var description: String {
            switch self {
            case .notZip: "not a zip archive"
            case .missing(let n): "no entry ending in \(n)"
            case .compressed(let n): "entry \(n) is compressed"
            case .badSize(let n, let s): "entry \(n) has \(s) bytes, not a multiple of 4"
            }
        }
    }

    /// Entry names and their stored bytes.
    static func entries(_ data: Data) throws -> [String: Data] {
        let bytes = [UInt8](data)
        func u16(_ o: Int) -> Int { Int(bytes[o]) | Int(bytes[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        // End of central directory: the last "PK\u{5}\u{6}".
        guard bytes.count >= 22, let eocd = stride(from: bytes.count - 22, through: max(0, bytes.count - 65_557), by: -1)
            .first(where: { u32($0) == 0x0605_4B50 }) else { throw Failure.notZip }
        let count = u16(eocd + 10)
        var cd = u32(eocd + 16)
        var out: [String: Data] = [:]
        for _ in 0..<count {
            guard u32(cd) == 0x0201_4B50 else { throw Failure.notZip }
            let method = u16(cd + 10), size = u32(cd + 20)
            let nameLen = u16(cd + 28), extraLen = u16(cd + 30), commentLen = u16(cd + 32)
            let local = u32(cd + 42)
            let name = String(decoding: bytes[(cd + 46) ..< (cd + 46 + nameLen)], as: UTF8.self)
            if method != 0 { throw Failure.compressed(name) }
            let start = local + 30 + u16(local + 26) + u16(local + 28)
            out[name] = data.subdata(in: start ..< start + size)
            cd += 46 + nameLen + extraLen + commentLen
        }
        return out
    }

    /// The float32 storage `data/<index>` reshaped to `shape` (row-major, as saved for a contiguous tensor).
    public static func float32(url: URL, storage index: Int = 0, shape: [Int]) throws -> MLXArray {
        let all = try entries(Data(contentsOf: url, options: .mappedIfSafe))
        let suffix = "/data/\(index)"
        guard let (name, raw) = all.first(where: { $0.key.hasSuffix(suffix) }) else { throw Failure.missing(suffix) }
        guard raw.count % 4 == 0 else { throw Failure.badSize(name, raw.count) }
        let floats = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        if shape.reduce(1, *) != floats.count { throw Failure.badSize(name, raw.count) }
        return MLXArray(floats, shape)
    }

    /// Element count of storage `data/<index>` (float32), to infer one free dimension.
    public static func float32Count(url: URL, storage index: Int = 0) throws -> Int {
        let all = try entries(Data(contentsOf: url, options: .mappedIfSafe))
        guard let raw = all.first(where: { $0.key.hasSuffix("/data/\(index)") })?.value else {
            throw Failure.missing("/data/\(index)")
        }
        return raw.count / 4
    }
}
