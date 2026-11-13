import Testing
@testable import AIStudioLibrary

struct ReplyErrorTests {
    @Test func contextLengthSaysTheLimit() {
        let e = "The model server answered HTTP 400: This model's maximum context length is 32768 tokens. However, your prompt contains at least 32769 input tokens."
        #expect(ReplyError.summary(e) == "The chat is longer than this model's 32K context")
    }

    @Test func commonFailuresGetOneLine() {
        #expect(ReplyError.summary("The request timed out.") == "The model server took too long to answer")
        #expect(ReplyError.summary("Could not connect to the server.") == "Couldn't reach the model server")
        #expect(ReplyError.summary("The model server answered HTTP 503: upstream unavailable, try again later") == "The model server failed (HTTP 503)")
    }

    @Test func authFailuresNeedAStatus() {
        #expect(ReplyError.summary("The model server answered HTTP 401: invalid token") == "The model server refused the token")
        #expect(ReplyError.summary("Request failed with status code 403") == "The model server refused the token")
        #expect(ReplyError.summary("Unauthorized") == "The model server refused the token")
        // A count that happens to contain 401 or 403 is not an auth failure.
        #expect(ReplyError.summary("The prompt has 14030 tokens") == nil)
        #expect(ReplyError.summary("Generation stopped at token 4017") == nil)
    }

    @Test func aShortErrorStaysAsItIs() {
        #expect(ReplyError.summary("Couldn't compact: no model") == nil)
    }

    @Test func aLongUnknownErrorKeepsItsFirstSentence() {
        let e = "Something odd happened in the engine. " + String(repeating: "More detail. ", count: 20)
        #expect(ReplyError.summary(e) == "Something odd happened in the engine")
    }

    @Test func tokenCounts() {
        #expect(ReplyError.tokens(131072) == "128K")
        #expect(ReplyError.tokens(8000) == "8K")
    }
}
