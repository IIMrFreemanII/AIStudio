import CryptoKit
import Foundation
import GraphRuntime

/// Turns a local graph into one a remote server can run, and back.
///
/// Every `.filePath` input is one of three roles:
/// - **model** (`LoadModel.path`): sent as the folder's name; the server looks it up in its models folder.
/// - **output** (`directory` of Save nodes): the server writes into its run folder; the client keeps the local
///   directory so downloaded results land where a local run would have put them.
/// - **input** (every other file: images, masks, video, audio, LoRAs): uploaded once, content-addressed as
///   `<sha256>.<ext>`, and the input rewritten to that name.
public enum GraphTransport {
    public enum Role: String, Sendable { case input, model, output }

    public struct Reference: Equatable, Sendable {
        public var node: String
        public var input: String
        public var role: Role
        public var value: String
    }

    public struct Upload: Equatable, Sendable {
        /// Server-side name, `<sha256>.<ext>`.
        public var name: String
        public var url: URL
        public var bytes: Int
        public init(name: String, url: URL, bytes: Int) { self.name = name; self.url = url; self.bytes = bytes }
    }

    public struct Prepared: Sendable {
        /// Flattened, with server-side names.
        public var graph: Graph
        /// The document graph as given (embedded into the outputs).
        public var original: Graph
        /// Distinct files the run needs.
        public var uploads: [Upload]
        /// Model folder names the run loads.
        public var models: [String]
        /// Save node id → the local directory its results go to.
        public var outputDirectories: [String: String]
    }

    public enum TransportError: Error, CustomStringConvertible, Equatable {
        case missingFile(node: String, path: String)
        public var description: String {
            switch self {
            case .missingFile(let n, let p): return "node \(n): file not found: \(p)"
            }
        }
    }

    public static func role(nodeType: String, input: String) -> Role {
        if nodeType == "LoadModel" { return .model }
        if input == "directory" { return .output }
        return .input
    }

    /// The file references of `graph` (which should be flat), in node order.
    public static func references(in graph: Graph, schema: (String) -> NodeSchema?) -> [Reference] {
        var refs: [Reference] = []
        for node in graph.nodes {
            guard let s = schema(node.type) else { continue }
            for spec in s.inputs where spec.widget == .filePath {
                let value: String
                switch node.inputs[spec.name] {
                case .value(.string(let v))?: value = v
                case nil: if case .string(let d)? = spec.defaultValue { value = d } else { continue }
                default: continue
                }
                refs.append(Reference(node: node.id, input: spec.name, role: role(nodeType: node.type, input: spec.name),
                                      value: value))
            }
        }
        return refs
    }

    /// The server-side name of an input file: its content hash plus its extension (decoders such as AVFoundation
    /// pick the container by extension).
    public static func serverName(sha256: String, for url: URL) -> String {
        let ext = url.pathExtension.lowercased().filter { $0.isLetter || $0.isNumber }.prefix(8)
        return ext.isEmpty ? sha256 : "\(sha256).\(ext)"
    }

    /// Whether `name` is a well-formed server-side file name (the server's upload routes accept nothing else).
    public static func isServerName(_ name: String) -> Bool {
        let parts = name.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 1 || parts.count == 2 else { return false }
        let hex = parts[0]
        guard hex.count == 64, hex.allSatisfy({ $0.isHexDigit && !$0.isUppercase }) else { return false }
        if parts.count == 2 {
            let ext = parts[1]
            guard (1...8).contains(ext.count), ext.allSatisfy({ ($0.isLetter && $0.isLowercase) || $0.isNumber }),
                  ext.allSatisfy(\.isASCII) else { return false }
        }
        return true
    }

    /// Rewrites `graph` for a server. `resolveInput` maps a Load node's path to a local file (the app's input
    /// folder lookup); `hash` returns a file's SHA-256 (hex).
    public static func prepare(_ graph: Graph, schema: (String) -> NodeSchema?,
                               resolveInput: (String) -> URL,
                               hash: (URL) throws -> String) throws -> Prepared {
        var flat = try graph.flattened()
        var uploads: [String: Upload] = [:]
        var order: [String] = []
        var models: [String] = []
        var outputs: [String: String] = [:]
        for ref in references(in: flat, schema: schema) {
            switch ref.role {
            case .model:
                let name = URL(fileURLWithPath: (ref.value as NSString).expandingTildeInPath).lastPathComponent
                if !models.contains(name) { models.append(name) }
                try flat.setValue(.string(name), node: ref.node, input: ref.input)
            case .output:
                outputs[ref.node] = ref.value
                try flat.setValue(.string("outputs"), node: ref.node, input: ref.input)
            case .input:
                // Optional file inputs (an empty LoRA path, an unset image) stay empty.
                guard !ref.value.isEmpty else { continue }
                let url = resolveInput(ref.value)
                guard let size = (try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int else {
                    throw TransportError.missingFile(node: ref.node, path: ref.value)
                }
                let name = serverName(sha256: try hash(url), for: url)
                if uploads[name] == nil {
                    uploads[name] = Upload(name: name, url: url, bytes: size)
                    order.append(name)
                }
                try flat.setValue(.string(name), node: ref.node, input: ref.input)
            }
        }
        return Prepared(graph: flat, original: graph, uploads: order.map { uploads[$0]! }, models: models,
                        outputDirectories: outputs)
    }
}

/// Streaming SHA-256 of files, remembered by (path, size, modification date) so a multi-gigabyte LoRA or video is
/// hashed once, not on every run.
public final class FileHasher: @unchecked Sendable {
    private struct Entry: Codable { var size: Int; var mtime: Double; var sha: String }

    public static let chunkSize = 4 << 20

    private let cacheURL: URL?
    private var entries: [String: Entry] = [:]
    private var dirty = false
    private let lock = NSLock()

    /// `cacheURL` nil keeps the cache in memory only.
    public init(cacheURL: URL?) {
        self.cacheURL = cacheURL
        if let cacheURL, let data = try? Data(contentsOf: cacheURL),
           let e = try? JSONDecoder().decode([String: Entry].self, from: data) {
            entries = e
        }
    }

    public func sha256(of url: URL) throws -> String {
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = (attrs[.size] as? Int) ?? 0
        let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = url.standardizedFileURL.path
        if let e = lock.withLock({ entries[key] }), e.size == size, e.mtime == mtime { return e.sha }
        let sha = try Self.hashFile(url)
        lock.withLock {
            entries[key] = Entry(size: size, mtime: mtime, sha: sha)
            dirty = true
        }
        return sha
    }

    /// Writes the cache if it changed.
    public func save() {
        guard let cacheURL else { return }
        let data: Data? = lock.withLock {
            guard dirty else { return nil }
            dirty = false
            return try? JSONEncoder().encode(entries)
        }
        guard let data else { return }
        try? FileManager.default.createDirectory(at: cacheURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: cacheURL, options: .atomic)
    }

    /// SHA-256 of a file read in ``chunkSize`` pieces (constant memory).
    public static func hashFile(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = try autoreleasepool { try handle.read(upToCount: chunkSize) }
            guard let chunk, !chunk.isEmpty else { break }
            hasher.update(data: chunk)
        }
        return hex(hasher.finalize())
    }

    public static func hex<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
        let table = Array("0123456789abcdef".utf8)
        var out = [UInt8]()
        out.reserveCapacity(64)
        for b in digest {
            out.append(table[Int(b >> 4)])
            out.append(table[Int(b & 0x0f)])
        }
        return String(decoding: out, as: UTF8.self)
    }
}
