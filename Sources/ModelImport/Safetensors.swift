import Foundation
import MLX

/// The header of a safetensors file: tensor names with their stored dtype tag and shape, plus the metadata.
///
/// Read without touching the tensor data, so a 20 GB checkpoint is inspected in milliseconds. The dtype tags are the
/// file's own (`F8_E4M3`, `BF16`, …): MLX loads FP8 as plain `uint8`, so the tag is the only way to tell them apart.
public struct SafetensorsHeader: Sendable {
    public struct Entry: Sendable, Equatable {
        public var dtype: String
        public var shape: [Int]
        public var byteCount: Int
    }

    public var url: URL
    public var tensors: [String: Entry]
    public var metadata: [String: String]

    public init(url: URL) throws {
        self.url = url
        let h = try FileHandle(forReadingFrom: url)
        defer { try? h.close() }
        let lenData = try h.read(upToCount: 8) ?? Data()
        guard lenData.count == 8 else { throw ModelImportError.notSafetensors(url.lastPathComponent) }
        if lenData.prefix(4) == Data("GGUF".utf8) { throw ModelImportError.gguf(url.lastPathComponent) }
        let n = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard n > 1, n < 200_000_000, let json = try h.read(upToCount: Int(n)), json.count == Int(n),
              json.first == UInt8(ascii: "{"),
              let obj = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw ModelImportError.notSafetensors(url.lastPathComponent)
        }
        var tensors: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        for (k, v) in obj {
            if k == "__metadata__" {
                for (mk, mv) in v as? [String: Any] ?? [:] { metadata[mk] = "\(mv)" }
                continue
            }
            guard let d = v as? [String: Any], let dtype = d["dtype"] as? String,
                  let shape = (d["shape"] as? [NSNumber])?.map(\.intValue),
                  let off = (d["data_offsets"] as? [NSNumber])?.map(\.intValue), off.count == 2 else {
                throw ModelImportError.notSafetensors(url.lastPathComponent)
            }
            tensors[k] = Entry(dtype: dtype, shape: shape, byteCount: off[1] - off[0])
        }
        self.tensors = tensors
        self.metadata = metadata
    }

    /// The headers of every `*.safetensors` in `directory` (sharded checkpoints), by file name.
    public static func directory(_ directory: URL) throws -> [SafetensorsHeader] {
        try safetensorsFiles(in: directory).map { try SafetensorsHeader(url: $0) }
    }
}

func safetensorsFiles(in directory: URL) throws -> [URL] {
    try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        .filter { $0.pathExtension == "safetensors" && !$0.lastPathComponent.hasPrefix(".") }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// One output tensor: a lazy MLX array (nothing is computed until it is written) and an optional dtype tag override
/// (to write `uint8` bytes as `F8_E4M3`, which tests use to build FP8 checkpoints).
public struct PlannedTensor {
    public var name: String
    public var array: MLXArray
    public var dtypeTag: String?

    public init(name: String, array: MLXArray, dtypeTag: String? = nil) {
        self.name = name
        self.array = array
        self.dtypeTag = dtypeTag
    }

    var tag: String { dtypeTag ?? Self.tag(array.dtype) }
    var byteCount: Int { array.size * array.dtype.size }

    static func tag(_ d: DType) -> String {
        switch d {
        case .bool: "BOOL"
        case .uint8: "U8"
        case .uint16: "U16"
        case .uint32: "U32"
        case .uint64: "U64"
        case .int8: "I8"
        case .int16: "I16"
        case .int32: "I32"
        case .int64: "I64"
        case .float16: "F16"
        case .bfloat16: "BF16"
        case .float32: "F32"
        case .float64: "F64"
        default: "UNKNOWN"
        }
    }
}

/// Writes a safetensors file one tensor at a time.
///
/// The header is built first from the arrays' shapes (known without evaluating them), then each tensor is evaluated,
/// appended and released, so memory peaks at about one source tensor plus its result instead of the whole file.
public enum SafetensorsWriter {
    /// `tensors` is taken `inout` and emptied as it is written: the caller must hold no other copy of the plan, or
    /// the evaluated tensors stay referenced until the call returns.
    ///
    /// - Parameter written: called after each tensor with the bytes written so far; may throw to cancel.
    public static func write(_ tensors: inout [PlannedTensor], metadata: [String: String] = [:], to url: URL,
                             written: (Int) throws -> Void = { _ in }) throws {
        var header: [String: Any] = [:]
        var offset = 0
        for t in tensors {
            precondition(header[t.name] == nil, "duplicate tensor \(t.name)")
            let n = t.byteCount
            header[t.name] = ["dtype": t.tag, "shape": t.array.shape, "data_offsets": [offset, offset + n]] as [String: Any]
            offset += n
        }
        if !metadata.isEmpty { header["__metadata__"] = metadata }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // Pad with spaces so the data starts 8-byte aligned (as the reference writer does).
        json.append(contentsOf: [UInt8](repeating: 0x20, count: (8 - json.count % 8) % 8))

        FileManager.default.createFile(atPath: url.path, contents: nil)
        let h = try FileHandle(forWritingTo: url)
        defer { try? h.close() }
        var len = UInt64(json.count).littleEndian
        try h.write(contentsOf: Data(bytes: &len, count: 8))
        try h.write(contentsOf: json)

        defer { tensors.removeAll() }
        var done = 0, sinceClear = 0
        for i in tensors.indices {
            let a = tensors[i].array
            eval(a)
            let data = a.asData(access: .copy).data
            precondition(data.count == tensors[i].byteCount, "size mismatch for \(tensors[i].name)")
            try h.write(contentsOf: data)
            // Drop this entry's array so its buffers (and, once siblings are written, its source) are released.
            tensors[i].array = MLXArray(0)
            done += data.count
            sinceClear += data.count
            if sinceClear > 1 << 30 {
                Memory.clearCache()
                sinceClear = 0
            }
            try written(done)
        }
        Memory.clearCache()
    }
}
