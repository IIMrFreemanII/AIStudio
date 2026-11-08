import Foundation
import Testing
@testable import GraphRuntime

final class EvictionLog: @unchecked Sendable {
    var keys: [String] = []
}

@Test func evictsLeastRecentlyUsedToFitTheBudget() async throws {
    let cache = ResourceCache(budgetBytes: 100)
    let log = EvictionLog()
    cache.onEvict = { log.keys += $0 }
    let _: String = try await cache.get("a", estimatedBytes: 40) { "A" }
    let _: String = try await cache.get("b", estimatedBytes: 40) { "B" }
    let _: String = try await cache.get("a") { "never" }  // touch a: b is now the oldest
    let _: String = try await cache.get("c", estimatedBytes: 40) { "C" }
    #expect(log.keys == ["b"])
    #expect(Set(cache.keys) == ["a", "c"])
    #expect(cache.residentBytes == 80)
    // Something bigger than everything else evicts all of it.
    let _: String = try await cache.get("huge", estimatedBytes: 90) { "H" }
    #expect(Set(log.keys) == ["a", "b", "c"])
    #expect(cache.keys == ["huge"])
}

@Test func resourcesUsedByTheRunArePinned() async throws {
    let cache = ResourceCache(budgetBytes: 100)
    let _: Int = try await cache.get("old", estimatedBytes: 60) { 1 }
    cache.beginRun()
    let _: Int = try await cache.get("old") { 0 }  // used by this run
    let _: Int = try await cache.get("new", estimatedBytes: 60) { 2 }  // can't evict "old": over budget, allowed
    #expect(Set(cache.keys) == ["old", "new"])
    let allPinned = cache.entries.allSatisfy { $0.pinned }
    #expect(allPinned)
    cache.endRun()
    let _: Int = try await cache.get("third", estimatedBytes: 60) { 3 }
    #expect(cache.keys.contains("third"))
    #expect(cache.residentBytes <= 120)
    #expect(!cache.keys.contains("old"))  // the least recently used went first

    cache.pin("third")
    let _: Int = try await cache.get("fourth", estimatedBytes: 60) { 4 }
    #expect(cache.keys.contains("third"))
    cache.unpin("third")
}

@Test func unlimitedBudgetNeverEvicts() async throws {
    let cache = ResourceCache()
    for i in 0..<5 { let _: Int = try await cache.get("k\(i)", estimatedBytes: 1 << 40) { i } }
    #expect(cache.keys.count == 5)
    cache.remove("k0")
    #expect(cache.keys.count == 4)
}
