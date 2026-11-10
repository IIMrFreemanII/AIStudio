import CryptoKit
import Foundation
import NIOCore
import MLXComfyProtocol

/// Uploaded input files, stored by content: `inputs/<sha256>.<ext>`. Uploads stream to `<name>.partial` and are
/// renamed once their hash checks out, so a half-written file is never visible to a run.
final class FileStore: @unchecked Sendable {
    let directory: URL
    private let lock = NSLock()
    private var waiters: [(names: Set<String>, continuation: CheckedContinuation<Void, Never>)] = []
    private var transfers: [String: Transfer] = [:]

    struct Transfer {
        var info: TransferInfo
        var started: Date
        var lastUpdate: Date
        var lastBytes: Int
    }

    enum UploadError: Error, CustomStringConvertible {
        case badName(String)
        case offsetMismatch(expected: Int)
        case hashMismatch(String)
        public var description: String {
            switch self {
            case .badName(let n): return "not a content-addressed file name: \(n)"
            case .offsetMismatch(let e): return "upload must resume at byte \(e)"
            case .hashMismatch(let n): return "uploaded bytes don't match \(n)"
            }
        }
    }

    init(directory: URL) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    func url(_ name: String) -> URL { directory.appendingPathComponent(name) }
    func partialURL(_ name: String) -> URL { directory.appendingPathComponent(name + ".partial") }

    func exists(_ name: String) -> Bool { FileManager.default.fileExists(atPath: url(name).path) }

    /// Bytes of an unfinished upload (resume offset), or 0.
    func partialSize(_ name: String) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: partialURL(name).path))?[.size] as? Int) ?? 0
    }

    var totalBytes: Int { SystemInfo.folderBytes(directory) }

    /// Waits until every name exists (returns at once when they do).
    func wait(for names: [String]) async {
        let set = Set(names)
        if set.allSatisfy(exists) { return }
        await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in
            lock.lock()
            // Re-check under the lock: the upload may have finished meanwhile.
            if set.allSatisfy(exists) { lock.unlock(); c.resume(); return }
            waiters.append((set, c))
            lock.unlock()
        }
    }

    private func fileArrived() {
        let ready: [CheckedContinuation<Void, Never>] = lock.withLock {
            var out: [CheckedContinuation<Void, Never>] = []
            waiters.removeAll { w in
                guard w.names.allSatisfy(exists) else { return false }
                out.append(w.continuation)
                return true
            }
            return out
        }
        ready.forEach { $0.resume() }
    }

    /// Wakes every waiter (used when a waiting run is cancelled, so its task can notice).
    func wakeAll() {
        let all: [CheckedContinuation<Void, Never>] = lock.withLock {
            defer { waiters.removeAll() }
            return waiters.map(\.continuation)
        }
        all.forEach { $0.resume() }
    }

    /// Streams `body` into the file `name`, starting at `offset` (0, or the partial size when resuming). `total` is
    /// the full size when known. Returns true when the file is complete and verified.
    func receive<S: AsyncSequence>(_ name: String, offset: Int, total: Int?, body: S,
                                   onBytes: (Int) -> Void) async throws -> Bool where S.Element == ByteBuffer {
        guard GraphTransport.isServerName(name) else { throw UploadError.badName(name) }
        if exists(name) {
            for try await _ in body {}
            return true
        }
        let partial = partialURL(name)
        let have = partialSize(name)
        guard offset == have else { throw UploadError.offsetMismatch(expected: have) }
        if offset == 0 { FileManager.default.createFile(atPath: partial.path, contents: nil) }
        let handle = try FileHandle(forWritingTo: partial)
        var hasher: SHA256? = offset == 0 ? SHA256() : nil
        var written = offset
        let id = UUID().uuidString
        beginTransfer(id: id, name: name, total: total, bytes: offset)
        defer { endTransfer(id) }
        do {
            try handle.seek(toOffset: UInt64(offset))
            for try await buffer in body {
                buffer.withUnsafeReadableBytes { raw in
                    guard raw.count > 0 else { return }
                    hasher?.update(bufferPointer: raw)
                    handle.write(Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                                      count: raw.count, deallocator: .none))
                }
                written += buffer.readableBytes
                onBytes(buffer.readableBytes)
                updateTransfer(id, bytes: written)
            }
            try handle.close()
        } catch {
            try? handle.close()
            throw error
        }
        if let total, written < total { return false }
        // Verify: the streamed hash, or (after a resume) a hash of the whole file.
        let sha = try hasher.map { FileHasher.hex($0.finalize()) } ?? FileHasher.hashFile(partial)
        guard name.hasPrefix(sha) else {
            try? FileManager.default.removeItem(at: partial)
            throw UploadError.hashMismatch(name)
        }
        try? FileManager.default.removeItem(at: url(name))
        try FileManager.default.moveItem(at: partial, to: url(name))
        fileArrived()
        return true
    }

    /// Deletes stored inputs not in `keep`; returns the bytes freed.
    func purge(keeping keep: Set<String>) -> Int {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        var freed = 0
        for n in names where !keep.contains(n) && !keep.contains(n.replacingOccurrences(of: ".partial", with: "")) {
            let u = directory.appendingPathComponent(n)
            freed += ((try? FileManager.default.attributesOfItem(atPath: u.path))?[.size] as? Int) ?? 0
            try? FileManager.default.removeItem(at: u)
        }
        return freed
    }

    // MARK: transfers (dashboard)

    func beginTransfer(id: String, name: String, total: Int?, bytes: Int = 0,
                       direction: TransferInfo.Direction = .upload) {
        let now = Date()
        lock.withLock {
            transfers[id] = Transfer(info: TransferInfo(id: id, name: name, direction: direction, bytes: bytes,
                                                        total: total, bytesPerSecond: 0),
                                     started: now, lastUpdate: now, lastBytes: bytes)
        }
    }

    func updateTransfer(_ id: String, bytes: Int) {
        let now = Date()
        lock.withLock {
            guard var t = transfers[id] else { return }
            t.info.bytes = bytes
            let dt = now.timeIntervalSince(t.lastUpdate)
            if dt >= 0.25 {
                let rate = Double(bytes - t.lastBytes) / dt
                t.info.bytesPerSecond = t.info.bytesPerSecond == 0 ? rate : t.info.bytesPerSecond * 0.6 + rate * 0.4
                t.lastUpdate = now
                t.lastBytes = bytes
            }
            transfers[id] = t
        }
    }

    func endTransfer(_ id: String) { lock.withLock { _ = transfers.removeValue(forKey: id) } }

    var activeTransfers: [TransferInfo] { lock.withLock { transfers.values.map(\.info).sorted { $0.name < $1.name } } }
}
