import Foundation
import Testing
@testable import CharacterCore

/// A repeatable generator (SplitMix64) so a roll can be checked.
private struct Seeded: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@Suite struct CharacterSurpriseTests {
    @Test func aRollPicksEveryCategoryWithinItsLimitAndClearsTheText() {
        var base = Character(name: "Old", age: 30, gender: "woman", look: "x", personality: "y", backstory: "z",
                             speakingStyle: "s", likes: "l", greeting: "hi", voice: "v")
        base.portraits = ["p.png"]
        for seed in 0..<50 {
            var rng = Seeded(state: UInt64(seed))
            let c = CharacterSurprise.roll(base, using: &rng)
            #expect(c.id == base.id && c.portraits == ["p.png"])
            #expect(c.gender == "woman")
            #expect(CharacterSurprise.names["woman"]!.contains(c.name))
            #expect(CharacterSurprise.ages.contains(c.age))
            for category in CharacterOptions.all {
                let picks = c.traits[category.id] ?? []
                #expect(!picks.isEmpty && picks.count <= category.max, "\(category.id)")
                let allowed = Set(CharacterOptions.options(category, gender: "woman").map(\.id))
                #expect(picks.allSatisfy(allowed.contains), "\(category.id) \(picks)")
            }
            #expect(c.look.isEmpty && c.personality.isEmpty && c.backstory.isEmpty && c.speakingStyle.isEmpty
                    && c.likes.isEmpty && c.voice.isEmpty && c.greeting.isEmpty)
            #expect(AgeGuard.problems(c).isEmpty)
        }
    }

    @Test func anUnknownGenderGetsWomanOrMan() {
        var rng = Seeded(state: 7)
        let c = CharacterSurprise.roll(Character(name: "", age: 25, gender: "robot"), using: &rng)
        #expect(["woman", "man"].contains(c.gender))
        #expect(CharacterSurprise.names[c.gender]!.contains(c.name))
        var rng2 = Seeded(state: 7)
        #expect(CharacterSurprise.roll(Character(name: "", age: 25, gender: "robot"), using: &rng2).traits == c.traits)
    }

    @Test func detailsParseAndApply() throws {
        let reply = """
            Sure! ```json
            {"name": "Juno", "look": "a gold ring", "personality": "Quick to laugh.", "backstory": "Grew up by the sea.",
             "speaking_style": "dry jokes", "likes": "vinyl, ramen", "greeting": "Hey!"}
            ```
            """
        let d = try CharacterWriter.parse(reply)
        #expect(d.name == "Juno" && d.speakingStyle == "dry jokes" && d.greeting == "Hey!")
        let c = Character(name: "Mia", age: 27, gender: "woman", likes: "tea")
        let kept = d.apply(to: c, name: false)
        #expect(kept.name == "Mia" && kept.backstory == "Grew up by the sea." && kept.likes == "vinyl, ramen")
        #expect(d.apply(to: c, name: true).name == "Juno")
        // Empty fields leave the character's own text.
        #expect(CharacterDetails(likes: "  ").apply(to: c, name: true).likes == "tea")
        #expect(throws: (any Error).self) { try CharacterWriter.parse("no json here") }
    }

    @Test func theRequestCarriesThePicks() {
        var c = Character(name: "Mia", age: 27, gender: "woman")
        c.traits = CharacterOptions.toggle(c.traits, category: "hairColor", option: "auburn")
        c.traits = CharacterOptions.toggle(c.traits, category: "rhythm", option: "night-owl")
        let u = CharacterWriter.user(c, newName: true)
        #expect(u.hasPrefix("Task: character details\n"))
        #expect(u.contains("auburn hair") && u.contains("suggest a new one") && u.contains("Habits: "))
    }
}
