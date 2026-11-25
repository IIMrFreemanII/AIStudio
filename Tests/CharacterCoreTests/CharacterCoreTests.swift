import Agent
import Foundation
import LLMClient
import Testing
@testable import CharacterCore

private let epoch = Date(timeIntervalSince1970: 1_700_000_000)

private func sample(_ name: String = "Ava", age: Int = 28) -> Character {
    Character(name: name, age: age, gender: "woman", look: "red hair", personality: "kind", date: epoch)
}

@Suite struct CharacterModelTests {
    @Test func decodeDefaults() throws {
        let id = UUID()
        let json = "{\"id\":\"\(id.uuidString)\",\"name\":\"X\"}"
        let c = try JSONDecoder().decode(Character.self, from: Data(json.utf8))
        #expect(c.id == id && c.name == "X" && c.age == 18 && c.gender == "")
        #expect(c.initiative == 3 && c.temperature == 0.9 && c.topP == 0.95)
        #expect(c.artModel == ArtModel() && c.loras.isEmpty && c.portraits.isEmpty)
        #expect(c.created == Date(timeIntervalSince1970: 0))
    }

    @Test func roundTrip() throws {
        var c = sample()
        c.loras = [CharacterLoRA(link: "civitai:1", strength: 0.5)]
        c.artModel = ArtModel(family: .chroma, style: .anime, variant: "v1")
        c.selectedPortrait = "a.png"
        let data = try CharacterStore.encoder.encode(c)
        #expect(try CharacterStore.decoder.decode(Character.self, from: data) == c)
    }
}

@Suite struct MemoryTests {
    @Test func applyDedupeForgetClamp() {
        var m = CharacterMemory()
        m.facts = [.init(text: "Has a cat", date: epoch), .init(text: "Likes tea", date: epoch)]
        m.relationship.affection = 95
        let u = MemoryUpdate(newFacts: ["  has a CAT ", "Lives in Rome", ""], forget: [" likes TEA "],
                             affectionDelta: 50, stage: "Dating", mood: " happy ", summary: " The story ")
        m.apply(u, chatID: nil, date: epoch.addingTimeInterval(10))
        #expect(m.facts.map(\.text) == ["Has a cat", "Lives in Rome"])
        #expect(m.relationship.affection == 100)
        #expect(m.relationship.stage == .dating && m.relationship.mood == "happy" && m.summary == "The story")
        #expect(m.relationship.lastSeen == epoch.addingTimeInterval(10) && m.turnsSinceUpdate == 0)
    }

    @Test func invalidStageIgnoredAndNegativeClamp() {
        var m = CharacterMemory()
        m.apply(MemoryUpdate(affectionDelta: -50, stage: "married"), chatID: nil)
        #expect(m.relationship.stage == .strangers)
        #expect(m.relationship.affection == 10)
    }

    @Test func maxFactsDropsOldest() {
        var m = CharacterMemory()
        m.facts = (0..<CharacterMemory.maxFacts).map { .init(text: "f\($0)", date: epoch.addingTimeInterval(Double($0))) }
        m.apply(MemoryUpdate(newFacts: ["new"]), chatID: nil, date: epoch.addingTimeInterval(1000))
        #expect(m.facts.count == CharacterMemory.maxFacts)
        #expect(!m.facts.contains { $0.text == "f0" } && m.facts.last?.text == "new")
    }

    @Test func updateDecodesTolerantly() throws {
        let u = try JSONDecoder().decode(MemoryUpdate.self, from: Data("{\"new_facts\":[\"a\"],\"affection_delta\":3}".utf8))
        #expect(u.newFacts == ["a"] && u.affectionDelta == 3 && u.forget.isEmpty && u.stage == nil)
        let e = try JSONDecoder().decode(MemoryUpdate.self, from: Data("{}".utf8))
        #expect(e == MemoryUpdate())
        let m = try JSONDecoder().decode(CharacterMemory.self, from: Data("{}".utf8))
        #expect(m == CharacterMemory())
    }
}

@Suite struct MemoryPassTests {
    @Test func parseFencesAndProse() throws {
        let t = "Sure! Here you go:\n```json\n{\"new_facts\":[\"The user has a cat\"],\"mood\":\"warm\"}\n```\nDone."
        let u = try MemoryPass.parse(t)
        #expect(u.newFacts == ["The user has a cat"] && u.mood == "warm")
        #expect(throws: (any Error).self) { try MemoryPass.parse("no json here") }
    }

    @Test func userFirstLine() {
        let s = MemoryPass.user(sample(), memory: CharacterMemory(), recent: [("User", "hi"), ("Ava", "hello")])
        #expect(s.components(separatedBy: "\n").first == "Task: character memory")
        #expect(s.contains("User: hi") && s.contains("Ava: hello"))
        #expect(MemoryPass.system(sample()).contains("new_facts"))
    }
}

@Suite struct MediaTagsTests {
    @Test func parseTwoTagsUnknownAndUnclosed() {
        let r = MediaTags.parse("Hi <photo>a red dress</photo> there <voice>good night</voice>  ok <b>x</b>")
        #expect(r.requests == [MediaRequest(kind: .photo, text: "a red dress", index: 0),
                               MediaRequest(kind: .voice, text: "good night", index: 1)])
        #expect(r.text == "Hi there ok <b>x</b>")
        let u = MediaTags.parse("Look <photo>half")
        #expect(u.requests.isEmpty && u.text == "Look <photo>half")
    }

    @Test func segmentsWhileStreaming() {
        let full = "Hi <photo>a red dress</photo> ok"
        for n in 0...full.count {
            let prefix = String(full.prefix(n))
            let segs = MediaTags.segments(prefix)
            let shown = segs.compactMap { if case .text(let t) = $0 { t } else { nil } }.joined(separator: "|")
            #expect(!shown.contains("<") && !shown.contains("red dress"), "prefix: \(prefix)")
            if prefix.hasSuffix("<photo>") || prefix.hasSuffix("a red dress") || prefix.hasSuffix("</photo") {
                #expect(segs.last == .pending(.photo), "prefix: \(prefix)")
            }
        }
        #expect(MediaTags.segments(full) == [.text("Hi"), .media(MediaRequest(kind: .photo, text: "a red dress", index: 0)), .text("ok")])
    }

    @Test func instructionsOnlyRequested() {
        #expect(MediaTags.instructions([]) == "")
        let s = MediaTags.instructions([.photo])
        #expect(s.contains("<photo>") && !s.contains("<voice>") && !s.contains("<video>") && !s.contains("<talk>"))
    }
}

@Suite struct AgeGuardTests {
    @Test func rejects() {
        #expect(!AgeGuard.problems(sample(age: 17)).isEmpty)
        #expect(AgeGuard.problems(sample(age: 18)).isEmpty)
        #expect(AgeGuard.youthTerm(in: "a 16-year-old with") != nil)
        #expect(AgeGuard.youthTerm(in: "She is 15 yo") != nil)
        #expect(AgeGuard.youthTerm(in: "Teenager") != nil)
        #expect(AgeGuard.youthTerm(in: "went to high school student council") != nil)
        var c = sample()
        c.backstory = "A teenager from Rome"
        #expect(AgeGuard.problems(c).first?.hasPrefix("Backstory describes someone underage") == true)
    }

    @Test func allows() {
        for s in ["my girlfriend", "a young woman", "petite", "a girl", "she is 25 years old", "21 yo", "kidnapped?", "Mother of children's author".replacingOccurrences(of: "children's", with: "novel")] {
            #expect(AgeGuard.youthTerm(in: s) == nil, "\(s)")
        }
    }

    @Test func templatesPass() {
        #expect(CharacterTemplates.all.count == 5)
        for t in CharacterTemplates.all {
            #expect(AgeGuard.problems(t).isEmpty, "\(t.name)")
            #expect((24...35).contains(t.age) && !t.greeting.isEmpty && !t.voice.isEmpty && t.templateID != nil)
        }
        #expect(Set(CharacterTemplates.all.map(\.id)).count == 5)
        let m = CharacterTemplates.make(CharacterTemplates.all[0], date: epoch)
        #expect(m.id != CharacterTemplates.all[0].id && m.templateID == "mia" && m.created == epoch)
    }
}

@Suite struct CharacterPromptTests {
    @Test func containsParts() {
        var m = CharacterMemory()
        m.facts = [.init(text: "The user has a cat called Miso")]
        m.summary = "We met online."
        let scene = Scenario.builtIns[1]
        var cal = Calendar(identifier: .gregorian)
        cal.timeZone = TimeZone(identifier: "UTC")!
        let other = sample("Bella")
        let p = CharacterPrompt.system(sample(), memory: m, scene: scene, others: [other], media: [.photo],
                                       initiativeLeft: 2, now: Date(timeIntervalSince1970: 1_700_000_000), calendar: cal)
        #expect(p.hasPrefix("You are Ava, a 28-year-old woman."))
        #expect(p.contains("Looks: red hair") && p.contains("- The user has a cat called Miso"))
        #expect(p.contains("Coffee date") && p.contains("It's Tuesday, 10:13 PM."))
        #expect(p.contains("Also in this chat: Bella.") && p.contains("<photo>") && !p.contains("<voice>"))
        #expect(p.contains("at most 2 more today"))
        let q = CharacterPrompt.system(sample(), memory: CharacterMemory(), scene: nil)
        #expect(!q.contains("<photo>") && !q.contains("Right now"))
        #expect(q.contains(CharacterPrompt.noMedia) && !p.contains(CharacterPrompt.noMedia))
    }
}

@Suite struct CharacterStoreTests {
    private func tempStore() -> CharacterStore {
        CharacterStore(root: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
    }

    @Test func roundTrips() throws {
        let store = tempStore()
        defer { try? FileManager.default.removeItem(at: store.root) }
        let a = sample("A")
        var b = sample("B")
        b.updated = epoch.addingTimeInterval(100)
        try store.save(a)
        try store.save(b)
        #expect(store.list().map(\.name) == ["B", "A"])
        #expect(store.load(a.id) == a)
        #expect(store.memory(a.id) == CharacterMemory())
        var m = CharacterMemory()
        m.facts = [.init(text: "x", date: epoch)]
        m.spentUSD = 1.5
        try store.saveMemory(m, for: a.id)
        #expect(store.memory(a.id) == m)
        #expect(store.scenarios() == Scenario.builtIns)
        let custom = Scenario(id: "c1", title: "Gym", place: "gym")
        try store.saveCustomScenarios([custom])
        #expect(store.scenarios() == Scenario.builtIns + [custom])
        store.delete(a.id, useTrash: false)
        #expect(store.load(a.id) == nil && store.list().count == 1)
    }

    @Test func invalidThrows() {
        let store = tempStore()
        #expect(throws: CharacterError.self) { try store.save(sample(age: 16)) }
        #expect(store.list().isEmpty)
    }
}
