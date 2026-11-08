import Foundation
import MLX

/// Loads tensors from a PyTorch `torch.save` file (state dicts, nested dicts of state dicts, or a single tensor)
/// with a minimal pickle *interpreter* that never executes code: it understands only the opcodes `torch.save`
/// emits for plain tensors and dicts, and only the globals `torch._utils._rebuild_tensor_v2`,
/// `torch._utils._rebuild_parameter`, `collections.OrderedDict` and `torch.<Type>Storage`. Anything else is an
/// error. Float32, float16 and bfloat16 storages are supported.
public enum TorchPickle {
    public enum Failure: Error, CustomStringConvertible {
        case unsupportedOpcode(UInt8, Int), unsupportedGlobal(String), malformed(String), noDataPickle
        public var description: String {
            switch self {
            case .unsupportedOpcode(let o, let p): String(format: "unsupported pickle opcode 0x%02x at %d", o, p)
            case .unsupportedGlobal(let g): "refusing pickle global \(g)"
            case .malformed(let m): "malformed pickle: \(m)"
            case .noDataPickle: "no data.pkl in the archive"
            }
        }
    }

    indirect enum Value {
        case none, bool(Bool), int(Int), float(Double), string(String)
        case tuple([Value]), list([Value]), dict([(Value, Value)])
        case global(String)
        case storage(key: String, type: String)
        case tensor(storage: String, type: String, offset: Int, shape: [Int], strides: [Int])
        case mark
    }

    /// Every tensor in the file, keyed by its dotted path through nested dicts (a lone tensor gets the key "").
    public static func load(url: URL) throws -> [String: MLXArray] {
        let entries = try TorchArchive.entries(Data(contentsOf: url, options: .mappedIfSafe))
        guard let (pklName, pkl) = entries.first(where: { $0.key.hasSuffix("/data.pkl") }) else { throw Failure.noDataPickle }
        let root = String(pklName.dropLast("data.pkl".count))  // "archive/"
        let value = try interpret([UInt8](pkl))
        var out: [String: MLXArray] = [:]
        func walk(_ v: Value, _ path: String) throws {
            switch v {
            case .dict(let items):
                for (k, item) in items {
                    guard case .string(let name) = k else { continue }
                    try walk(item, path.isEmpty ? name : path + "." + name)
                }
            case .tensor(let key, let type, let offset, let shape, let strides):
                guard let raw = entries[root + "data/" + key] else { throw Failure.malformed("missing storage \(key)") }
                out[path] = try tensor(raw, type: type, offset: offset, shape: shape, strides: strides)
            default:
                break
            }
        }
        try walk(value, "")
        return out
    }

    static func tensor(_ raw: Data, type: String, offset: Int, shape: [Int], strides: [Int]) throws -> MLXArray {
        let flat: MLXArray
        switch type {
        case "FloatStorage":
            flat = MLXArray(raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) })
        case "HalfStorage":
            flat = MLXArray(raw.withUnsafeBytes { Array($0.bindMemory(to: Float16.self)) })
        case "BFloat16Storage":
            let u16 = raw.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
            flat = MLXArray(u16.map { Float(bitPattern: UInt32($0) << 16) }).asType(.bfloat16)
        default:
            throw Failure.unsupportedGlobal("torch.\(type)")
        }
        let count = shape.reduce(1, *)
        // Contiguous row-major tensors (the usual case) are a slice and a reshape; others go through asStrided.
        var contiguous = true, expect = 1
        for (n, s) in zip(shape, strides).reversed() where n != 1 {
            if s != expect { contiguous = false; break }
            expect *= n
        }
        if contiguous { return flat[offset ..< offset + count].reshaped(shape) }
        return asStrided(flat, shape, strides: strides, offset: offset)
    }

    /// Interprets the pickle stream into a `Value`.
    static func interpret(_ b: [UInt8]) throws -> Value {
        var stack: [Value] = []
        var memo: [Int: Value] = [:]
        var p = 0
        func u8() throws -> Int { guard p < b.count else { throw Failure.malformed("eof") }; p += 1; return Int(b[p - 1]) }
        func le(_ n: Int) throws -> Int {
            guard p + n <= b.count else { throw Failure.malformed("eof") }
            var v = 0
            for i in 0..<n { v |= Int(b[p + i]) << (8 * i) }
            p += n
            return v
        }
        func bytes(_ n: Int) throws -> [UInt8] {
            guard p + n <= b.count else { throw Failure.malformed("eof") }
            p += n
            return Array(b[(p - n) ..< p])
        }
        func line() throws -> String {
            guard let e = b[p...].firstIndex(of: 0x0A) else { throw Failure.malformed("unterminated line") }
            let s = String(decoding: b[p..<e], as: UTF8.self)
            p = e + 1
            return s
        }
        func pop() throws -> Value { guard let v = stack.popLast() else { throw Failure.malformed("stack underflow") }; return v }
        func popMark() throws -> [Value] {
            guard let m = stack.lastIndex(where: { if case .mark = $0 { true } else { false } }) else {
                throw Failure.malformed("no mark")
            }
            let items = Array(stack[(m + 1)...])
            stack.removeSubrange(m...)
            return items
        }
        func setItems(_ pairs: [Value]) throws {
            guard case .dict(var d) = try pop() else { throw Failure.malformed("setitems on non-dict") }
            var i = 0
            while i + 1 < pairs.count { d.append((pairs[i], pairs[i + 1])); i += 2 }
            stack.append(.dict(d))
        }
        let allowed: Set<String> = ["torch._utils _rebuild_tensor_v2", "torch._utils _rebuild_parameter",
                                    "collections OrderedDict"]
        while p < b.count {
            let start = p
            let op = try u8()
            switch op {
            case 0x80: _ = try u8()                                         // PROTO
            case 0x95: _ = try le(8)                                        // FRAME
            case 0x2E: return try pop()                                     // STOP
            case 0x28: stack.append(.mark)                                  // MARK
            case 0x4E: stack.append(.none)                                  // NONE
            case 0x88: stack.append(.bool(true))                            // NEWTRUE
            case 0x89: stack.append(.bool(false))                           // NEWFALSE
            case 0x4A: let v = try le(4); stack.append(.int(Int(Int32(truncatingIfNeeded: v))))  // BININT
            case 0x4B: stack.append(.int(try u8()))                         // BININT1
            case 0x4D: stack.append(.int(try le(2)))                        // BININT2
            case 0x8A:                                                      // LONG1
                let n = try u8()
                let raw = try bytes(n)
                var v = 0
                for (i, x) in raw.enumerated() where i < 8 { v |= Int(x) << (8 * i) }
                if n > 0, n < 8, raw[n - 1] & 0x80 != 0 { v -= 1 << (8 * n) }
                stack.append(.int(v))
            case 0x47:                                                      // BINFLOAT (big-endian)
                let raw = try bytes(8)
                stack.append(.float(Double(bitPattern: raw.reduce(UInt64(0)) { $0 << 8 | UInt64($1) })))
            case 0x58: let n = try le(4); stack.append(.string(String(decoding: try bytes(n), as: UTF8.self)))  // BINUNICODE
            case 0x8C: let n = try u8(); stack.append(.string(String(decoding: try bytes(n), as: UTF8.self)))   // SHORT_BINUNICODE
            case 0x29: stack.append(.tuple([]))                             // EMPTY_TUPLE
            case 0x74: stack.append(.tuple(try popMark()))                  // TUPLE
            case 0x85: stack.append(.tuple([try pop()]))                    // TUPLE1
            case 0x86: let b2 = try pop(), a = try pop(); stack.append(.tuple([a, b2]))                  // TUPLE2
            case 0x87: let c = try pop(), b2 = try pop(), a = try pop(); stack.append(.tuple([a, b2, c]))  // TUPLE3
            case 0x7D: stack.append(.dict([]))                              // EMPTY_DICT
            case 0x5D: stack.append(.list([]))                              // EMPTY_LIST
            case 0x73: let v = try pop(), k = try pop(); try setItems([k, v])  // SETITEM
            case 0x75: try setItems(try popMark())                          // SETITEMS
            case 0x61:                                                      // APPEND
                let v = try pop()
                guard case .list(var l) = try pop() else { throw Failure.malformed("append on non-list") }
                l.append(v); stack.append(.list(l))
            case 0x65:                                                      // APPENDS
                let items = try popMark()
                guard case .list(var l) = try pop() else { throw Failure.malformed("appends on non-list") }
                l += items; stack.append(.list(l))
            case 0x71: memo[try u8()] = stack.last                          // BINPUT
            case 0x72: memo[try le(4)] = stack.last                         // LONG_BINPUT
            case 0x94: memo[memo.count] = stack.last                        // MEMOIZE
            case 0x68: guard let v = memo[try u8()] else { throw Failure.malformed("memo") }; stack.append(v)   // BINGET
            case 0x6A: guard let v = memo[try le(4)] else { throw Failure.malformed("memo") }; stack.append(v)  // LONG_BINGET
            case 0x63:                                                      // GLOBAL
                let name = try line() + " " + line()
                if allowed.contains(name) || (name.hasPrefix("torch ") && name.hasSuffix("Storage")) {
                    stack.append(.global(name))
                } else {
                    throw Failure.unsupportedGlobal(name)
                }
            case 0x51:                                                      // BINPERSID
                guard case .tuple(let t) = try pop(), t.count >= 3, case .global(let g) = t[1],
                      case .string(let key) = t[2] else { throw Failure.malformed("persistent id") }
                stack.append(.storage(key: key, type: String(g.dropFirst("torch ".count))))
            case 0x52:                                                      // REDUCE
                let args = try pop(), fn = try pop()
                guard case .global(let g) = fn, case .tuple(let a) = args else { throw Failure.malformed("reduce") }
                switch g {
                case "collections OrderedDict": stack.append(.dict([]))
                case "torch._utils _rebuild_parameter":
                    guard let t = a.first else { throw Failure.malformed("parameter") }
                    stack.append(t)
                case "torch._utils _rebuild_tensor_v2":
                    guard a.count >= 4, case .storage(let key, let type) = a[0], case .int(let off) = a[1],
                          case .tuple(let sz) = a[2], case .tuple(let st) = a[3] else { throw Failure.malformed("tensor") }
                    func ints(_ v: [Value]) throws -> [Int] {
                        try v.map { if case .int(let i) = $0 { return i } else { throw Failure.malformed("shape") } }
                    }
                    stack.append(.tensor(storage: key, type: type, offset: off, shape: try ints(sz), strides: try ints(st)))
                default: throw Failure.unsupportedGlobal(g)
                }
            case 0x62: _ = try pop()                                        // BUILD: drop the state (dict metadata)
            default: throw Failure.unsupportedOpcode(UInt8(op), start)
            }
        }
        throw Failure.malformed("no STOP")
    }
}
