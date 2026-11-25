import Foundation
import Testing
@testable import CharacterCore

private func who(_ name: String) -> Character {
    Character(name: name, age: 25, gender: "woman", date: Date(timeIntervalSince1970: 0))
}

@Suite struct GroupChatTests {
    let mia = who("Mia"), bo = who("Bo"), ada = who("Ada Lovelace"), eve = who("Eve")

    @Test func namedComeFirstInTextOrder() {
        let r = GroupTurns.speakers(for: "bo, and then MIA?", cast: [mia, bo, eve], lastFirst: nil)
        #expect(r == [bo.id, mia.id])
    }

    @Test func wholeWordOnly() {
        #expect(GroupTurns.speakers(for: "Mia's idea", cast: [mia, bo], lastFirst: nil) == [mia.id])
        // "Miami" does not name Mia: nobody named, rotation from the start.
        #expect(GroupTurns.speakers(for: "I love Miami", cast: [mia, bo], lastFirst: nil) == [mia.id, bo.id])
    }

    @Test func firstWordOfLongName() {
        #expect(GroupTurns.speakers(for: "hey ada!", cast: [mia, ada], lastFirst: nil) == [ada.id])
        #expect(GroupTurns.speakers(for: "ada lovelace?", cast: [mia, ada], lastFirst: nil) == [ada.id])
    }

    @Test func everyoneAndRotation() {
        let cast = [mia, bo, eve]
        #expect(GroupTurns.speakers(for: "hi everyone", cast: cast, lastFirst: nil) == [mia.id, bo.id, eve.id])
        #expect(GroupTurns.speakers(for: "what do you two think", cast: [mia, bo], lastFirst: mia.id) == [bo.id, mia.id])
        #expect(GroupTurns.speakers(for: "hello", cast: cast, lastFirst: bo.id) == [eve.id, mia.id, bo.id])
        #expect(GroupTurns.speakers(for: "hello", cast: cast, lastFirst: eve.id) == [mia.id, bo.id, eve.id])
        #expect(GroupTurns.speakers(for: "hello", cast: cast, lastFirst: UUID()) == [mia.id, bo.id, eve.id])
    }

    @Test func capAtThree() {
        let cast = [mia, bo, eve, ada]
        #expect(GroupTurns.speakers(for: "hello all", cast: cast, lastFirst: nil).count == 3)
        #expect(GroupTurns.speakers(for: "everybody", cast: cast, lastFirst: mia.id) == [bo.id, eve.id, ada.id])
        #expect(GroupTurns.speakers(for: "Mia Bo Eve Ada", cast: cast, lastFirst: nil) == [mia.id, bo.id, eve.id])
    }

    @Test func singleAndEmpty() {
        #expect(GroupTurns.speakers(for: "hi", cast: [], lastFirst: nil).isEmpty)
        #expect(GroupTurns.speakers(for: "hi Bo", cast: [mia], lastFirst: nil) == [mia.id])
    }

    @Test func stripNameForms() {
        #expect(GroupTurns.stripName("Mia: hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("**Mia:** hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("**Mia**: hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("[Mia]: hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("[Mia] hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("  mia:\n hello", name: "Mia") == "hello")
        #expect(GroupTurns.stripName("Hello Mia: hi", name: "Mia") == "Hello Mia: hi")
        #expect(GroupTurns.stripName("Mia is here", name: "Mia") == "Mia is here")
    }

    @Test func plainAndLine() {
        #expect(MediaTags.plain("<photo> on a beach </photo>") == "[sent a photo: on a beach]")
        #expect(MediaTags.plain("<video>waves</video>") == "[sent a video: waves]")
        #expect(MediaTags.plain("<voice>Hi you</voice>") == "[sent a voice message: \"Hi you\"]")
        #expect(MediaTags.plain("<talk>Miss you</talk>") == "[sent a video message: \"Miss you\"]")
        #expect(MediaTags.plain("Look <photo>me</photo> nice?") == "Look [sent a photo: me] nice?")
        #expect(MediaTags.plain("  plain  ") == "plain")
        #expect(GroupTurns.line("Bo", "hey <voice>hello</voice>") == "Bo: hey [sent a voice message: \"hello\"]")
    }

    @Test func checkInDue() {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let eight = now.addingTimeInterval(-8 * 3600)
        #expect(CheckIn.due(hours: 8, last: eight, lastIsUser: false, now: now))
        #expect(!CheckIn.due(hours: 8, last: eight.addingTimeInterval(1), lastIsUser: false, now: now))
        #expect(!CheckIn.due(hours: 8, last: eight, lastIsUser: true, now: now))
        #expect(!CheckIn.due(hours: 0, last: eight, lastIsUser: false, now: now))
        #expect(!CheckIn.due(hours: 8, last: nil, lastIsUser: false, now: now))
    }

    @Test func nudgeAndLabel() {
        #expect(CheckIn.nudge(name: "Mia", away: 3600) == "[About an hour since your last messages. Text them first, as Mia: say what you've been up to, or pick up where you left off. One or two short messages; don't mention this note.]")
        #expect(CheckIn.nudge(name: "Mia", away: 9 * 3600 + 5).hasPrefix("[About 9 hours since"))
        #expect(CheckIn.nudge(name: "Mia", away: 3 * 86400 + 5).hasPrefix("[About 3 days since"))
        #expect(CheckIn.choices == [0, 4, 8, 24])
        #expect(CheckIn.label(0) == "Never" && CheckIn.label(24) == "After a day")
        #expect(CheckIn.label(1) == "After 1 hour" && CheckIn.label(4) == "After 4 hours")
    }

    @Test func scenarioLike() {
        let item = AlbumItem(file: "a.png", kind: .photo, prompt: "  red dress, smiling  ")
        let base = Scenario(id: "x", title: "X", place: "a cafe", time: "noon", mood: "happy")
        let a = Scenario.like(item, base: base, date: "Nov 3")
        #expect(a.id == "album-\(item.id.uuidString)" && a.title == "Like the photo of Nov 3")
        #expect(a.place == "a cafe" && a.outfit == "red dress, smiling" && a.time == "noon" && a.mood == "happy")
        let b = Scenario.like(item, base: nil, date: "Nov 3")
        #expect(b.place == "the same place as in that photo" && b.time == "" && b.mood == "")
    }

    @Test func checkInHoursDefaultAndDecoding() throws {
        #expect(mia.checkInHours == 8)
        var c = mia
        c.checkInHours = 24
        let data = try JSONEncoder().encode(c)
        #expect(try JSONDecoder().decode(Character.self, from: data).checkInHours == 24)
        let old = try JSONDecoder().decode(Character.self, from: Data(#"{"id":"8F1C1B2A-0000-4000-8000-000000000001","name":"Old"}"#.utf8))
        #expect(old.checkInHours == 8)
    }

    @Test func groupSentenceInPrompt() {
        let p = CharacterPrompt.system(mia, memory: CharacterMemory(), scene: nil, others: [bo, eve])
        #expect(p.contains("This is a group chat. Also in it: Bo, Eve."))
        #expect(p.contains("Write only your own message as Mia, without your name in front"))
        let q = CharacterPrompt.system(mia, memory: CharacterMemory(), scene: nil)
        #expect(!q.contains("group chat"))
    }
}
