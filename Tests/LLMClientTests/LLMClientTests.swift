import Foundation
import Testing
@testable import LLMClient

/// The pieces of a chat stream: SSE framing, OpenAI chunks, the server's `aistudio.*` events, `<think>` splitting,
/// tool-call assembly, and what requests and transcripts encode to.
struct LLMClientTests {
    func parse(_ text: String) -> [SSEEvent] {
        var p = SSEParser()
        var out = text.components(separatedBy: "\n").compactMap { p.feed(line: $0) }
        if let e = p.flush() { out.append(e) }
        return out
    }

    @Test func sseFramesEventsCommentsAndMultilineData() {
        let events = parse(": ping\n\nevent: aistudio.status\ndata: {\"state\":\"loading\"}\n\ndata: a\ndata: b\r\n\ndata: [DONE]")
        #expect(events == [SSEEvent(event: "aistudio.status", data: "{\"state\":\"loading\"}"),
                           SSEEvent(data: "a\nb"), SSEEvent(data: "[DONE]")])
    }

    @Test func chunksBecomeEventsAndDoneStops() throws {
        var got: [ChatEvent] = []
        let chunk = #"{"choices":[{"index":0,"delta":{"reasoning":"hm ","content":"Hi"},"finish_reason":null}]}"#
        #expect(try !OpenAIChatClient.handle(SSEEvent(data: chunk)) { got.append($0) })
        let tool = #"{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read","arguments":"{\"a\""}}]},"finish_reason":"tool_calls"}]}"#
        _ = try OpenAIChatClient.handle(SSEEvent(data: tool)) { got.append($0) }
        let usage = #"{"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":7}}"#
        _ = try OpenAIChatClient.handle(SSEEvent(data: usage)) { got.append($0) }
        _ = try OpenAIChatClient.handle(SSEEvent(event: "aistudio.status", data: #"{"state":"loading","phase":"Loading weights","progress":0.41}"#)) { got.append($0) }
        _ = try OpenAIChatClient.handle(SSEEvent(event: "aistudio.usage", data: #"{"seconds":2.5,"ttftMs":300,"tokPerSec":41.2,"completionTokens":7,"costUSD":0.0063}"#)) { got.append($0) }
        #expect(try OpenAIChatClient.handle(SSEEvent(data: "[DONE]")) { got.append($0) })
        #expect(got == [
            .reasoning("hm "), .content("Hi"),
            .toolCall(index: 0, id: "c1", name: "read", arguments: "{\"a\""), .finish(reason: "tool_calls"),
            .usage(promptTokens: 12, completionTokens: 7),
            .status(EngineStatus(state: "loading", phase: "Loading weights", progress: 0.41)),
            .stats(RequestStats(seconds: 2.5, ttftMs: 300, tokPerSec: 41.2, completionTokens: 7, costUSD: 0.0063)),
        ])
    }

    @Test func serverErrorsThrow() {
        #expect(throws: LLMError.server("the GPU ran out")) {
            try OpenAIChatClient.handle(SSEEvent(event: "aistudio.error", data: #"{"error":"the GPU ran out"}"#)) { _ in }
        }
        #expect(throws: LLMError.server("bad model")) {
            try OpenAIChatClient.handle(SSEEvent(data: #"{"error":{"message":"bad model"}}"#)) { _ in }
        }
    }

    @Test func thinkTagsSplitAcrossChunks() {
        var s = ThinkSplitter()
        var pieces: [ThinkSplitter.Piece] = []
        for chunk in ["<th", "ink>let me ", "see</thi", "nk>\n\nThe answer", " is 4. <", "b>"] {
            pieces += s.feed(chunk)
        }
        pieces += s.finish()
        let reasoning = pieces.compactMap { if case .reasoning(let r) = $0 { return r }; return nil }.joined()
        let content = pieces.compactMap { if case .content(let c) = $0 { return c }; return nil }.joined()
        #expect(reasoning == "let me see")
        #expect(content == "The answer is 4. <b>")
    }

    @Test func accumulatorPrefersTheServersReasoningField() {
        var a = ReplyAccumulator()
        a.add(.reasoning("thinking"))
        a.add(.content("<think>not thinking</think>text"))
        a.finish()
        #expect(a.reasoning == "thinking" && a.content == "<think>not thinking</think>text")

        var b = ReplyAccumulator()
        for t in ["<think>", "plan", "</think>", "done"] { b.add(.content(t)) }
        b.finish()
        #expect(b.reasoning == "plan" && b.content == "done")
    }

    @Test func toolCallsAssembleByIndex() {
        var t = ToolCallAssembler()
        t.add(index: 1, id: "b", name: "grep", arguments: "{}")
        t.add(index: 0, id: "a", name: "read_file", arguments: "{\"path\": ")
        t.add(index: 0, id: nil, name: nil, arguments: "\"README.md\"}")
        t.add(index: 2, id: nil, name: "glob", arguments: "{}")
        #expect(t.calls == [ToolCall(id: "a", name: "read_file", arguments: "{\"path\": \"README.md\"}"),
                            ToolCall(id: "b", name: "grep", arguments: "{}"),
                            ToolCall(id: "call_2", name: "glob", arguments: "{}")])
    }

    @Test func requestEncodesLikeOpenAI() throws {
        var r = ChatRequest(model: "qwen38", messages: [
            .system("be brief"),
            ChatMessage(role: .user, parts: [.image(url: "data:image/jpeg;base64,AA=="), .text("what is this?")]),
            ChatMessage(role: .assistant, parts: [.text("")], toolCalls: [ToolCall(id: "c1", name: "f", arguments: "{}")]),
            ChatMessage(role: .tool, parts: [.text("ok")], toolCallID: "c1"),
        ])
        r.reasoningEffort = "low"
        r.chatTemplateKwargs = ["enable_thinking": .bool(false)]
        r.tools = [ToolSpec(name: "f", description: "does f", parameters: .object(["type": .string("object")]))]
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(r)) as! [String: Any]
        #expect(json["model"] as? String == "qwen38" && json["stream"] as? Bool == true)
        #expect((json["stream_options"] as? [String: Bool])?["include_usage"] == true)
        #expect(json["reasoning_effort"] as? String == "low")
        #expect((json["chat_template_kwargs"] as? [String: Bool])?["enable_thinking"] == false)
        #expect(json["temperature"] == nil)
        let msgs = json["messages"] as! [[String: Any]]
        #expect(msgs[0]["content"] as? String == "be brief")
        let parts = msgs[1]["content"] as! [[String: Any]]
        #expect(parts[0]["type"] as? String == "image_url" && parts[1]["text"] as? String == "what is this?")
        let call = (msgs[2]["tool_calls"] as! [[String: Any]])[0]
        #expect(call["type"] as? String == "function" && (call["function"] as? [String: String])?["name"] == "f")
        #expect(msgs[3]["tool_call_id"] as? String == "c1")
        let tool = (json["tools"] as! [[String: Any]])[0]
        #expect((tool["function"] as? [String: Any])?["description"] as? String == "does f")
    }

    @Test func transcriptRoundTripsAndSkipsFailedReplies() throws {
        var t = ChatTranscript(serverID: "s", modelID: "m")
        t.systemPrompt = "sys"
        t.append(ChatTurn(role: .user, text: "hi", images: ["a.jpg"], files: [ChatFile(name: "x.py", text: "print(1)")]))
        var failed = ChatTurn(role: .assistant, text: "")
        failed.error = "cancelled"
        t.append(failed)
        var ok = ChatTurn(role: .assistant, text: "hello")
        ok.stats = RequestStats(seconds: 1, costUSD: 0.01)
        t.append(ok)
        let data = try JSONEncoder().encode(t)
        let back = try JSONDecoder().decode(ChatTranscript.self, from: data)
        #expect(back == t && back.costUSD == 0.01)
        let msgs = back.messages { $0 == "a.jpg" ? "data:image/jpeg;base64,AA==" : nil }
        #expect(msgs.map(\.role) == [.system, .user, .assistant])
        #expect(msgs[1].parts.first == .image(url: "data:image/jpeg;base64,AA=="))
        #expect(msgs[1].text == "hi\n\n`x.py`:\n```py\nprint(1)\n```")
        #expect(ChatTranscript.title(from: "  \nFix the CLI flag parsing\nplease") == "Fix the CLI flag parsing")
    }

    @Test func messagesStartFromTheLatestSummary() throws {
        var t = ChatTranscript(serverID: "s", modelID: "m")
        t.systemPrompt = "sys"
        t.append(ChatTurn(role: .user, text: "old question"))
        t.append(ChatTurn(role: .assistant, text: "old answer"))
        var s1 = ChatTurn(role: .user, text: "first summary")
        s1.summary = true
        t.append(s1)
        t.append(ChatTurn(role: .assistant, text: "middle"))
        var s2 = ChatTurn(role: .user, text: "second summary")
        s2.summary = true
        t.append(s2)
        t.append(ChatTurn(role: .assistant, text: "after"))
        t.append(ChatTurn(role: .user, text: "next"))
        let msgs = t.messages()
        #expect(msgs.map(\.role) == [.system, .user, .assistant, .user])
        #expect(msgs[1].text == ChatTurn.summaryPrefix + "second summary")
        #expect(msgs[2].text == "after" && msgs[3].text == "next")
        // Old files have no `summary` key.
        let old = try JSONDecoder().decode(ChatTurn.self, from: Data("""
            {"id":"\(UUID().uuidString)","role":"user","text":"x","images":[],"toolCalls":[],"date":0}
            """.utf8))
        #expect(old.summary == nil)
    }

    @Test func videosGoAsVideoURLParts() throws {
        var t = ChatTranscript(serverID: "s", modelID: "m")
        let video = String(repeating: "a", count: 64) + ".mp4"
        t.append(ChatTurn(role: .user, text: "what happens?", images: ["b.jpg", video]))
        #expect(ChatTurn.isVideo(video) && !ChatTurn.isVideo("b.jpg") && t.attachments == ["b.jpg", video])
        let msgs = t.messages { $0 == video ? "upload:" + $0 : "data:image/jpeg;base64,AA==" }
        #expect(msgs[0].parts.prefix(2) == [.image(url: "data:image/jpeg;base64,AA=="), .video(url: "upload:" + video)])
        let json = try JSONSerialization.jsonObject(with: JSONEncoder().encode(ContentPart.video(url: "upload:x.mp4")))
            as! [String: Any]
        #expect(json["type"] as? String == "video_url")
        #expect((json["video_url"] as? [String: Any])?["url"] as? String == "upload:x.mp4")
        let back = try JSONDecoder().decode(ContentPart.self, from: JSONEncoder().encode(ContentPart.video(url: "u")))
        #expect(back == .video(url: "u"))
        // A message with a video goes in the list form (plain text would drop it).
        let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(
            ChatMessage(role: .user, parts: [.video(url: "upload:x.mp4"), .text("hi")]))) as! [String: Any]
        #expect((encoded["content"] as? [[String: Any]])?.first?["type"] as? String == "video_url")
        // Without vision nothing is sent for them.
        #expect(t.messages { _ in nil }[0].parts == [.text("what happens?")])
    }

    @Test func followUpsComeOutOfTheText() {
        let full = "Use 0.8.\n\n<followups>\n- Try 1.0\n* What's CFG?\n3. Compare them\n</followups>"
        let (text, items) = FollowUps.split(full)
        #expect(text == "Use 0.8." && items == ["Try 1.0", "What's CFG?", "Compare them"])
        #expect(FollowUps.split("Use 0.8.\n\n<followups>\n- Try", streaming: true) == ("Use 0.8.", []))
        #expect(FollowUps.split("Use 0.8.\n<follo", streaming: true).text == "Use 0.8.")
        #expect(FollowUps.split("a < b").text == "a < b" && FollowUps.split("no block").items.isEmpty)
        let many = "x<followups>" + (1...6).map { "- item \($0) " + String(repeating: "y", count: 100) }.joined(separator: "\n") + "</followups>"
        let capped = FollowUps.split(many).items
        #expect(capped.count == 4 && capped.allSatisfy { $0.count <= 80 })
    }

    @Test func quotesAndHiddenContextGoToTheModel() throws {
        var t = ChatTranscript()
        let reply = ChatTurn(role: .assistant, text: "strength 0.8")
        t.append(ChatTurn(role: .user, text: "hi"))
        t.append(reply)
        var turn = ChatTurn(role: .user, text: "why?")
        turn.quote = ChatQuote(turnID: reply.id, role: .assistant, text: "strength\n0.8")
        turn.context = "[Widget \"S\" state: {\"strength\":0.8}]"
        t.append(turn)
        var withFollowUps = ChatTurn(role: .assistant, text: "because")
        withFollowUps.followUps = ["More"]
        t.append(withFollowUps)
        let back = try JSONDecoder().decode(ChatTranscript.self, from: JSONEncoder().encode(t))
        #expect(back == t && back.turns[2].quote?.turnID == reply.id && back.turns[3].followUps == ["More"])
        let msgs = back.messages()
        #expect(msgs[2].text == "Replying to this part of your reply:\n> strength\n> 0.8\n\nwhy?\n\n[Widget \"S\" state: {\"strength\":0.8}]")
        #expect(msgs[3].text == "because")
        #expect(ChatQuote(turnID: nil, role: .user, text: String(repeating: "a", count: 3000)).text.count == ChatQuote.maxLength)
        var sys = ChatTranscript()
        sys.systemPrompt = "be brief"
        sys.append(ChatTurn(role: .user, text: "x"))
        #expect(sys.messages().count == 2 && sys.messages(system: false).count == 1)
    }

    @Test func editsAndRetriesBranch() {
        var t = ChatTranscript()
        let q1 = ChatTurn(role: .user, text: "q1")
        t.append(q1)
        let a1 = ChatTurn(role: .assistant, text: "a1")
        t.append(a1)
        t.append(ChatTurn(role: .user, text: "q2"))
        let a2 = ChatTurn(role: .assistant, text: "a2")
        t.append(a2)
        // Retry the first answer: a sibling, and the branch after it goes out of view.
        t.branch(from: a1.id, with: ChatTurn(role: .assistant, text: "a1 again"))
        #expect(t.turns.map(\.text) == ["q1", "a1 again"])
        #expect(t.siblings(of: a1.id).map(\.text) == ["a1", "a1 again"])
        // Back to the first answer: its newest descendants come back.
        t.select(a1.id)
        #expect(t.turns.map(\.text) == ["q1", "a1", "q2", "a2"])
        // Edit the first question.
        let e = ChatTurn(role: .user, text: "q1 edited")
        t.branch(from: q1.id, with: e)
        #expect(t.turns.map(\.text) == ["q1 edited"] && t.siblings(of: q1.id).count == 2)
        t.append(ChatTurn(role: .assistant, text: "a for edit"))
        t.select(q1.id)
        #expect(t.turns.last?.text == "a2")
        // Removing a turn removes what follows it and falls back to its parent's newest branch.
        t.remove(a2.id)
        #expect(t.turns.map(\.text) == ["q1", "a1", "q2"])
        #expect(t.nodes.count == 6)
    }

    @Test func versionOneFilesStillOpen() throws {
        let json = #"""
        {"version":1,"id":"6E0F6B6E-6D53-4C49-9C3A-3B7E2B1A0A11","title":"Old","created":0,"updated":0,
         "turns":[{"id":"1E0F6B6E-6D53-4C49-9C3A-3B7E2B1A0A11","role":"user","text":"hi","images":[],"toolCalls":[],"date":0},
                  {"id":"2E0F6B6E-6D53-4C49-9C3A-3B7E2B1A0A11","role":"assistant","text":"hello","images":[],"toolCalls":[],"date":0}]}
        """#
        let t = try JSONDecoder().decode(ChatTranscript.self, from: Data(json.utf8))
        #expect(t.turns.map(\.text) == ["hi", "hello"] && t.turns[1].parent == t.turns[0].id)
    }

    @Test func listingDecodes() throws {
        let json = #"""
        {"models":[{"id":"qwen38","name":"Qwen3.8","url":"Qwen/x","repo":"Qwen/x","gpu":"H200","count":2,
         "context":131072,"nativeContext":262144,"vision":true,"installed":true,"engine":"vllm",
         "status":{"id":"qwen38","state":"ready","phase":"Ready","progress":1,"uptime":12.5,"usdPerSecond":0.0025,
                   "costUSD":0.03,"idleSeconds":180,"leaseSeconds":1800,"lastChat":null,"error":null,"gpu":"H200","count":2}}],
         "installs":{"tiny":{"state":"downloading","bytes":10,"total":100}}}
        """#
        let l = try JSONDecoder().decode(LLMListing.self, from: Data(json.utf8))
        #expect(l.models[0].gpuLabel == "2× H200" && l.models[0].status?.isActive == true)
        #expect(l.installs["tiny"]?.total == 100)
    }

    @Test func statusDecodesTheColdStartSummary() throws {
        let json = """
        {"id":"m","state":"loading","phase":"Loading weights","progress":0.4,"bootSeconds":75.2,
         "coldStart":{"last":{"at":1.0,"seconds":192.4,"waitSeconds":20,"how":"cold","gpu":"H200","count":2,
           "loadFormat":"runai","phases":[{"name":"Waiting for a GPU","seconds":20},{"name":"Starting container","seconds":0.3},
           {"name":"Loading weights","seconds":130.1},{"name":"Compiling","seconds":42}]},
         "typicalSeconds":185,"count":3}}
        """
        let s = try JSONDecoder().decode(LLMStatus.self, from: Data(json.utf8))
        #expect(s.bootSeconds == 75.2 && s.coldStart?.typicalSeconds == 185 && s.coldStart?.count == 3)
        let last = try #require(s.coldStart?.last)
        #expect(last.loadFormat == "runai" && last.phases.count == 4)
        // The short phase is left out of the breakdown.
        #expect(last.breakdown == "Waiting for a GPU 20 s · Loading weights 2 min 10 s · Compiling 42 s")
        #expect(ColdStartRecord.duration(192.4) == "3 min 12 s" && ColdStartRecord.duration(120) == "2 min"
                && ColdStartRecord.duration(48.4) == "48 s")
        // Older servers send neither.
        let old = try JSONDecoder().decode(LLMStatus.self, from: Data(#"{"state":"cold"}"#.utf8))
        #expect(old.coldStart == nil && old.bootSeconds == nil)
        let model = try JSONDecoder().decode(LLMModel.self, from: Data(#"{"id":"m","name":"M","repo":"o/m","gpu":"H100","count":1,"context":8192,"loadFormat":"runai","snapshot":true}"#.utf8))
        #expect(model.loadFormat == "runai" && model.snapshot == true)
    }
}
