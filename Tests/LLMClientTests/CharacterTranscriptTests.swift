import Foundation
import Testing
@testable import LLMClient

/// A character chat's fields: the characters, the scene, and a turn's speaker. Saved chats without them still load.
struct CharacterTranscriptTests {
    let when = Date(timeIntervalSince1970: 1_800_000_000)

    func coder() -> JSONEncoder {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        return e
    }

    func decoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }

    @Test func oldChatWithoutCharacterKeysDecodesAsNotACharacterChat() throws {
        let json = """
        {"id":"8F1C1B2A-0000-4000-8000-000000000001","title":"Old chat",
         "created":"2026-11-20T10:00:00Z","updated":"2026-11-20T10:05:00Z"}
        """
        let t = try decoder().decode(ChatTranscript.self, from: Data(json.utf8))
        #expect(t.title == "Old chat")
        #expect(t.characterIDs == nil)
        #expect(t.sceneID == nil)
    }

    @Test func characterFieldsAndSpeakerRoundTrip() throws {
        let ada = UUID(), bo = UUID()
        var t = ChatTranscript(title: "Scene", serverID: "s", modelID: "m", date: when)
        t.characterIDs = [ada, bo]
        t.sceneID = "harbor-night"
        var reply = ChatTurn(role: .assistant, text: "Hello.", date: when)
        reply.speaker = bo
        t.append(ChatTurn(role: .user, text: "Hi", date: when))
        t.append(reply)

        let data = try coder().encode(t)
        let back = try decoder().decode(ChatTranscript.self, from: data)
        #expect(back == t)
        #expect(back.characterIDs == [ada, bo])
        #expect(back.sceneID == "harbor-night")
        #expect(back.turns.last?.speaker == bo)
    }

    @Test func forkKeepsCharacterIDsAndScene() throws {
        let ada = UUID()
        var t = ChatTranscript(title: "Scene", serverID: "s", modelID: "m", date: when)
        t.characterIDs = [ada]
        t.sceneID = "market"
        let first = ChatTurn(role: .user, text: "Hi", date: when)
        t.append(first)
        t.append(ChatTurn(role: .assistant, text: "Hey", date: when))

        let copy = try #require(t.fork(upTo: first.id))
        #expect(copy.characterIDs == [ada])
        #expect(copy.sceneID == "market")
    }

    @Test func mediaRoundTripsAndOldTurnsDecodeWithoutIt() throws {
        var turn = ChatTurn(role: .assistant, text: "Here you go", date: when)
        turn.media = [
            ChatMedia(kind: "photo", prompt: "at the beach", status: .done, file: "a.png", estimateUSD: 0.02, costUSD: 0.01),
            ChatMedia(kind: "video", prompt: "waving", status: .waiting, estimateUSD: 0.5),
        ]
        let back = try decoder().decode(ChatTurn.self, from: try coder().encode(turn))
        #expect(back == turn)
        #expect(back.media?.count == 2 && back.media?[1].status == .waiting)
        let json = """
        {"id":"8F1C1B2A-0000-4000-8000-000000000002","role":"assistant","text":"hi","images":[],"toolCalls":[],
         "date":"2026-11-20T10:00:00Z"}
        """
        #expect(try decoder().decode(ChatTurn.self, from: Data(json.utf8)).media == nil)
    }
}
