import Foundation
import Testing
@testable import Agent

struct CheckpointTests {
    func makeStore() -> (CheckpointStore, URL, URL) {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("checkpoint-test-\(UUID().uuidString)")
        let work = base.appendingPathComponent("work")
        try? FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        return (CheckpointStore(folder: base.appendingPathComponent("store")), base.appendingPathComponent("store"), work)
    }

    func text(_ url: URL) -> String? { try? String(contentsOf: url, encoding: .utf8) }

    @Test func secondSaveForSameTurnKeepsFirstCopy() async throws {
        let (store, _, work) = makeStore()
        let f = work.appendingPathComponent("a.txt")
        let t = UUID()
        try "one".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t)
        try "two".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t)
        #expect(await store.entries().count == 1)
        try await store.restore(turns: [t])
        #expect(text(f) == "one")
    }

    @Test func restoresModifiedFile() async throws {
        let (store, _, work) = makeStore()
        let f = work.appendingPathComponent("a.txt")
        let t = UUID()
        try "before".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t)
        try "after".write(to: f, atomically: true, encoding: .utf8)
        let r = try await store.restore(turns: [t])
        #expect(r.restored.map(\.path) == [f.standardizedFileURL.path])
        #expect(r.skipped.isEmpty)
        #expect(text(f) == "before")
    }

    @Test func deletesFileTheAgentCreated() async throws {
        let (store, _, work) = makeStore()
        let f = work.appendingPathComponent("new.txt")
        let t = UUID()
        try await store.save(f, turn: t)
        #expect(await store.entries().first?.blob == nil)
        try "made".write(to: f, atomically: true, encoding: .utf8)
        let r = try await store.restore(turns: [t])
        #expect(r.restored.count == 1)
        #expect(!FileManager.default.fileExists(atPath: f.path))
    }

    @Test func tooLargeFileIsSkipped() async throws {
        let (store, _, work) = makeStore()
        let f = work.appendingPathComponent("big.bin")
        let t = UUID()
        try Data(count: CheckpointStore.maxBytes + 1).write(to: f)
        try await store.save(f, turn: t)
        let e = await store.entries()
        #expect(e.first?.tooLarge == true)
        #expect(e.first?.blob == nil)
        try Data(count: 3).write(to: f)
        let r = try await store.restore(turns: [t])
        #expect(r.restored.isEmpty)
        #expect(r.skipped.count == 1)
        #expect(try Data(contentsOf: f).count == 3)
    }

    @Test func newStoreSeesSavedEntries() async throws {
        let (store, folder, work) = makeStore()
        let f = work.appendingPathComponent("a.txt")
        let t = UUID()
        try "x".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t)
        let again = CheckpointStore(folder: folder)
        #expect(await again.entries() == store.entries())
        #expect(await again.entries().count == 1)
        #expect(await again.changedPaths(in: [t]).map(\.path) == [f.standardizedFileURL.path])
    }

    @Test func restoreOfLaterTurnGoesBackToStateBeforeIt() async throws {
        let (store, folder, work) = makeStore()
        let f = work.appendingPathComponent("a.txt")
        let t1 = UUID(), t2 = UUID()
        try "v0".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t1)
        try "v1".write(to: f, atomically: true, encoding: .utf8)
        try await store.save(f, turn: t2)
        try "v2".write(to: f, atomically: true, encoding: .utf8)
        try await store.restore(turns: [t2])
        #expect(text(f) == "v1")
        let left = await store.entries()
        #expect(left.count == 1)
        #expect(left.first?.turn == t1)
        let blobs = try FileManager.default.contentsOfDirectory(atPath: folder.appendingPathComponent("blobs").path)
        #expect(blobs == [left[0].blob!])
        // Both turns together go back to the oldest copy.
        try "v2".write(to: f, atomically: true, encoding: .utf8)
        try await store.restore(turns: [t1])
        #expect(text(f) == "v0")
        #expect(await store.entries().isEmpty)
        #expect(try FileManager.default.contentsOfDirectory(atPath: folder.appendingPathComponent("blobs").path).isEmpty)
    }
}
