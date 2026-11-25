import Foundation
import Testing
@testable import LLMClient

@Suite struct GroupHistoryTests {
    let when = Date(timeIntervalSince1970: 1_800_000_000)
    let a = UUID(), b = UUID()

    func transcript() -> ChatTranscript {
        var t = ChatTranscript(title: "Group", serverID: "s", modelID: "m", date: when)
        t.append(ChatTurn(role: .user, text: "Hi all", date: when))
        var ta = ChatTurn(role: .assistant, text: "Hey from A", date: when)
        ta.speaker = a
        var tb = ChatTurn(role: .assistant, text: "B here", date: when)
        tb.speaker = b
        t.append(ta)
        t.append(tb)
        t.append(ChatTurn(role: .user, text: "Nice", date: when))
        return t
    }

    func others(_ t: ChatTurn) -> String? { t.speaker == b ? "B: \(t.text)" : "A: \(t.text)" }

    func shape(_ m: [ChatMessage]) -> [String] { m.map { "\($0.role.rawValue):\($0.text)" } }

    @Test func seenAsA() {
        let m = transcript().messages(system: false, speaker: a, others: others)
        #expect(shape(m) == ["user:Hi all", "assistant:Hey from A", "user:B: B here\n\nNice"])
        #expect(m[2].parts.count == 1)
    }

    @Test func seenAsB() {
        let m = transcript().messages(system: false, speaker: b, others: others)
        #expect(shape(m) == ["user:Hi all\n\nA: Hey from A", "assistant:B here", "user:Nice"])
    }

    @Test func nilDropsTurn() {
        let m = transcript().messages(system: false, speaker: a, others: { _ in nil })
        #expect(shape(m) == ["user:Hi all", "assistant:Hey from A", "user:Nice"])
    }

    @Test func noSpeakerIsUnchanged() {
        let m = transcript().messages(system: false)
        #expect(shape(m) == ["user:Hi all", "assistant:Hey from A", "assistant:B here", "user:Nice"])
    }

    @Test func imagePartsKeptWhenJoining() {
        var t = ChatTranscript(title: "G", serverID: "s", modelID: "m", date: when)
        var ta = ChatTurn(role: .assistant, text: "x", date: when)
        ta.speaker = a
        t.append(ta)
        t.append(ChatTurn(role: .user, text: "Look", images: ["p.png"], date: when))
        let m = t.messages(system: false, image: { _ in "data:x" }, speaker: b, others: { _ in "A: x" })
        #expect(m.count == 1)
        #expect(m[0].parts == [.text("A: x"), .image(url: "data:x"), .text("Look")])
    }
}
