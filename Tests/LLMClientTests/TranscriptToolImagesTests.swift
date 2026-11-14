import Foundation
import Testing
@testable import LLMClient

/// A coding chat's `ui_*` screenshots: kept with the tool turns, only the last two sent again.
struct TranscriptToolImagesTests {
    func chat(shots: [[String]]) -> ChatTranscript {
        var t = ChatTranscript(title: "t", serverID: "s", modelID: "m")
        t.append(ChatTurn(role: .user, text: "go"))
        for (i, images) in shots.enumerated() {
            let call = ToolCall(id: "c\(i)", name: "ui_screenshot", arguments: "{}")
            t.append(ChatTurn(role: .assistant, text: "", toolCalls: [call]))
            var tool = ChatTurn(role: .tool, text: "result \(i)", toolCallID: "c\(i)")
            tool.images = images
            t.append(tool)
        }
        return t
    }

    func isShots(_ m: ChatMessage) -> Bool {
        m.role == .user && m.parts.first == .text("The images from the tool results above, in order:")
    }

    @Test func onlyTheLastTwoToolTurnsKeepTheirImages() {
        let t = chat(shots: [["a.jpg"], ["b.jpg", "c.jpg"], ["d.jpg"]])
        let msgs = t.messages(system: false) { "data:" + $0 }
        let tools = msgs.filter { $0.role == .tool }
        #expect(tools[0].parts == [.text("result 0 [screenshot omitted]")])
        #expect(tools[1].parts == [.text("result 1")])
        let shots = msgs.filter(isShots)
        #expect(shots.count == 2)
        #expect(shots[0].parts == [.text("The images from the tool results above, in order:"), .image(url: "data:b.jpg"), .image(url: "data:c.jpg")])
        #expect(shots[1].parts.count == 2)
        // Each images message follows its tool message.
        let i = msgs.firstIndex(where: isShots)!
        #expect(msgs[i - 1].role == .tool && msgs[i - 1].toolCallID == "c1")
    }

    @Test func aRunOfToolTurnsGetsOneImagesMessage() {
        var t = chat(shots: [["a.jpg"]])
        let call = ToolCall(id: "x", name: "ui_screenshot", arguments: "{}")
        // Two tool results in a row after one assistant message.
        t.update(t.turns.last!.parent!) { $0.toolCalls.append(call) }
        var second = ChatTurn(role: .tool, text: "again", toolCallID: "x")
        second.images = ["e.jpg"]
        t.append(second)
        let msgs = t.messages(system: false) { "data:" + $0 }
        let shots = msgs.filter(isShots)
        #expect(shots.count == 1)
        #expect(shots[0].parts.count == 3)
    }

    @Test func aToolTurnWithoutImagesIsUnchanged() {
        let t = chat(shots: [[]])
        let msgs = t.messages(system: false) { "data:" + $0 }
        #expect(msgs.last?.role == .tool)
        #expect(msgs.last?.parts == [.text("result 0")])
        #expect(!msgs.contains(where: isShots))
    }

    @Test func recordingsAreNotAttachments() {
        var t = chat(shots: [["a.jpg"]])
        let id = t.turns.last!.id
        t.update(id) { $0.toolFiles = ["rec.mp4"] }
        #expect(t.attachments == ["a.jpg"])
        #expect(!t.messages(system: false) { "data:" + $0 }.contains { $0.parts.contains(.image(url: "data:rec.mp4")) })
    }

    @Test func toolFilesAndRulesSurviveSaving() throws {
        var t = chat(shots: [["a.jpg"]])
        t.update(t.turns.last!.id) { $0.toolFiles = ["rec.mp4"] }
        t.allowObserveApps = ["com.x.app"]
        t.allowControlApps = ["com.y.app"]
        let back = try JSONDecoder().decode(ChatTranscript.self, from: JSONEncoder().encode(t))
        #expect(back.turns.last?.toolFiles == ["rec.mp4"])
        #expect(back.allowObserveApps == ["com.x.app"])
        #expect(back.allowControlApps == ["com.y.app"])
        #expect(t.fork(upTo: t.turns.last!.id)?.allowControlApps == ["com.y.app"])
    }
}
