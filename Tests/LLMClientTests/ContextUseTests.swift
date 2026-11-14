import Foundation
import Testing
@testable import LLMClient

/// The context meter's numbers: the server's count for the last reply, estimates for what came after it.
struct ContextUseTests {
    /// A text of exactly `tokens` estimated tokens.
    func words(_ tokens: Int) -> String { String(repeating: "abcd", count: tokens) }

    func reply(_ tokens: Int, prompt: Int, completion: Int) -> ChatTurn {
        var t = ChatTurn(role: .assistant, text: words(tokens))
        t.stats = RequestStats(seconds: 1, promptTokens: prompt, completionTokens: completion)
        return t
    }

    @Test func aCountedReplyIsExact() {
        var t = ChatTranscript()
        t.append(ChatTurn(role: .user, text: words(10)))
        t.append(reply(20, prompt: 1_000, completion: 25))
        #expect(t.contextUse() == ContextUse(tokens: 1_025, exact: true))
    }

    @Test func whatCameAfterItIsEstimated() {
        var t = ChatTranscript()
        t.append(ChatTurn(role: .user, text: words(10)))
        t.append(reply(20, prompt: 1_000, completion: 25))
        // The message just sent, and a reply still streaming (no stats yet).
        t.append(ChatTurn(role: .user, text: words(30)))
        t.append(ChatTurn(role: .assistant, text: words(5)))
        #expect(t.contextUse() == ContextUse(tokens: 1_025 + 30 + 5, exact: false))
    }

    @Test func toolResultsAndKeptScreenshotsAreEstimated() {
        var t = ChatTranscript()
        t.append(ChatTurn(role: .user, text: words(10)))
        var call = reply(0, prompt: 500, completion: 10)
        call.toolCalls = [ToolCall(id: "c", name: "ui_screenshot", arguments: words(3))]
        t.append(call)
        var tool = ChatTurn(role: .tool, text: words(40), toolCallID: "c")
        tool.images = ["a.jpg"]
        t.append(tool)
        let use = t.contextUse(breakdown: true)
        #expect(use.tokens == 510 + 40 + 1_000 && !use.exact)
        #expect(use.breakdown?.tools == 3 + 40 && use.breakdown?.media == 1_000)
    }

    @Test func withoutACountTheOverheadAndEstimateAreUsed() {
        var t = ChatTranscript()
        var user = ChatTurn(role: .user, text: words(10))
        user.images = ["a.jpg", "b.mp4"]
        t.append(user)
        let use = t.contextUse(overhead: 300, breakdown: true)
        #expect(use.tokens == 300 + 10 + 2_000 && !use.exact)
        #expect(use.breakdown == ContextUse.Breakdown(system: 300, messages: 10, tools: 0, media: 2_000))
    }

    @Test func onlyWhatFollowsTheLatestSummaryCounts() {
        var t = ChatTranscript()
        t.append(ChatTurn(role: .user, text: words(10)))
        t.append(reply(20, prompt: 9_000, completion: 25))
        var summary = ChatTurn(role: .user, text: words(50))
        summary.summary = true
        t.append(summary)
        let prefix = TokenEstimate.text(ChatTurn.summaryPrefix + words(50))
        #expect(t.contextUse(overhead: 100) == ContextUse(tokens: 100 + prefix, exact: false))
    }

    @Test func aRewoundBranchUsesItsOwnCount() {
        var t = ChatTranscript()
        let first = ChatTurn(role: .user, text: words(10))
        t.append(first)
        t.append(reply(20, prompt: 1_000, completion: 25))
        t.append(ChatTurn(role: .user, text: words(10)))
        t.append(reply(20, prompt: 5_000, completion: 50))
        #expect(t.contextUse().tokens == 5_050)
        // Edit the first message: the new branch has no counted reply yet.
        t.branch(from: first.id, with: ChatTurn(role: .user, text: words(12)))
        #expect(t.contextUse(overhead: 200) == ContextUse(tokens: 212, exact: false))
    }

    @Test func theSystemPartIsWhatTheCountHasBeyondTheMessages() {
        var t = ChatTranscript()
        t.append(ChatTurn(role: .user, text: words(10)))
        t.append(reply(20, prompt: 1_000, completion: 25))
        let b = t.contextUse(breakdown: true).breakdown
        #expect(b == ContextUse.Breakdown(system: 1_025 - 30, messages: 30, tools: 0, media: 0))
    }
}
