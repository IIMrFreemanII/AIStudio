import Foundation
import Testing
@testable import MLXDiffusionCore

/// A folder source that fails after `limit` bytes of its first stream.
final class FlakySource: ByteSource, @unchecked Sendable {
    let base: FolderByteSource
    let limit: Int
    var failed = false
    init(base: FolderByteSource, limit: Int) { self.base = base; self.limit = limit }

    func list() async throws -> [RemoteFile] { try await base.list() }

    func open(_ file: RemoteFile, from offset: Int64) -> AsyncThrowingStream<Data, Error> {
        let inner = base.open(file, from: offset)
        let limit = limit
        let shouldFail = !failed
        failed = true
        return AsyncThrowingStream { cont in
            Task {
                var sent = 0
                do {
                    for try await d in inner {
                        if shouldFail && sent + d.count > limit {
                            cont.yield(d.prefix(limit - sent))
                            throw URLError(.networkConnectionLost)
                        }
                        cont.yield(d)
                        sent += d.count
                    }
                    cont.finish()
                } catch {
                    cont.finish(throwing: error)
                }
            }
        }
    }
}

func makeRepo() throws -> URL {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("repo-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("transformer"), withIntermediateDirectories: true)
    var big = Data(count: 300_000)
    for i in 0..<big.count { big[i] = UInt8(truncatingIfNeeded: i &* 31 &+ 7) }
    try big.write(to: root.appendingPathComponent("transformer/model.safetensors"))
    try Data("{\"a\":1}".utf8).write(to: root.appendingPathComponent("config.json"))
    try Data("# readme".utf8).write(to: root.appendingPathComponent("README.md"))
    return root
}

@Test func parsesHFTreeAndSelectsByGlob() throws {
    let json = """
        [{"type":"directory","path":"vae"},
         {"type":"file","path":"vae/config.json","size":512},
         {"type":"file","path":"vae/model.safetensors","size":134,"lfs":{"oid":"abc123","size":644000000}},
         {"type":"file","path":"README.md","size":2000}]
        """
    let files = try HFAPI.parseTree(Data(json.utf8))
    #expect(files.count == 3)
    #expect(files[1] == RemoteFile(path: "vae/model.safetensors", size: 644_000_000, sha256: "abc123"))
    let picked = HFAPI.select(files, include: [], exclude: ["*.md"])
    #expect(picked.map(\.path) == ["vae/config.json", "vae/model.safetensors"])
    #expect(HFAPI.select(files, include: ["*.safetensors"], exclude: []).count == 1)
}

@Test func downloadsVerifiesAndSkipsFinishedFiles() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let source = FolderByteSource(root: repo)
    let files = HFAPI.select(try await source.list(), include: [], exclude: ["*.md"])
    var last = ModelDownloader.Progress(done: 0, total: 0, file: "")
    try await ModelDownloader.download(files, to: dest, from: source) { last = $0 }
    #expect(last.done == last.total && last.total == 300_000 + 7)
    #expect(try Data(contentsOf: dest.appendingPathComponent("transformer/model.safetensors"))
        == Data(contentsOf: repo.appendingPathComponent("transformer/model.safetensors")))
    #expect(!FileManager.default.fileExists(atPath: dest.appendingPathComponent("README.md").path))
    #expect(ModelDownloader.remainingBytes(files, in: dest) == 0)
    // A second run has nothing to fetch.
    var calls = 0
    try await ModelDownloader.download(files, to: dest, from: source) { _ in calls += 1 }
    #expect(calls == 0)
}

@Test func interruptedDownloadResumesFromThePart() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let base = FolderByteSource(root: repo)
    let files = HFAPI.select(try await base.list(), include: ["*.safetensors"], exclude: [])
    let flaky = FlakySource(base: base, limit: 100_000)
    await #expect(throws: URLError.self) { try await ModelDownloader.download(files, to: dest, from: flaky) }
    let part = dest.appendingPathComponent("transformer/model.safetensors.part")
    #expect(ModelDownloader.size(of: part) == 100_000)
    #expect(ModelDownloader.remainingBytes(files, in: dest) == 200_000)
    var first: Int64 = -1
    try await ModelDownloader.download(files, to: dest, from: flaky) { p in if first < 0 { first = p.done } }
    #expect(first > 100_000)  // resumed, not restarted
    #expect(try ModelDownloader.sha256(of: dest.appendingPathComponent("transformer/model.safetensors")) == files[0].sha256)
    #expect(!FileManager.default.fileExists(atPath: part.path))
}

@Test func corruptedPartFailsTheChecksumAndIsDiscarded() async throws {
    let repo = try makeRepo()
    let dest = FileManager.default.temporaryDirectory.appendingPathComponent("dl-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: repo); try? FileManager.default.removeItem(at: dest) }
    let source = FolderByteSource(root: repo)
    let files = HFAPI.select(try await source.list(), include: ["*.safetensors"], exclude: [])
    let part = dest.appendingPathComponent("transformer/model.safetensors.part")
    try FileManager.default.createDirectory(at: part.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(repeating: 0xFF, count: 1000).write(to: part)  // wrong bytes
    await #expect(throws: DownloadError.checksumMismatch("transformer/model.safetensors")) {
        try await ModelDownloader.download(files, to: dest, from: source)
    }
    #expect(!FileManager.default.fileExists(atPath: part.path))
    try await ModelDownloader.download(files, to: dest, from: source)  // clean retry succeeds
}
