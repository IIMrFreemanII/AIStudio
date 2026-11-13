import Foundation
import Testing
@testable import Agent
@testable import LLMClient

/// A fake network: records the requests and answers with one canned response.
final class FakeWeb: @unchecked Sendable {
    private let lock = NSLock()
    private var _requests: [URLRequest] = []
    var requests: [URLRequest] { lock.withLock { _requests } }
    let status: Int
    let type: String?
    let body: Data

    init(status: Int = 200, type: String? = "text/html; charset=utf-8", body: Data) {
        self.status = status
        self.type = type
        self.body = body
    }

    convenience init(status: Int = 200, type: String? = "text/html; charset=utf-8", _ body: String) {
        self.init(status: status, type: type, body: Data(body.utf8))
    }

    var fetcher: WebFetcher {
        { [self] request in
            lock.withLock { _requests.append(request) }
            var headers: [String: String] = [:]
            if let type { headers["Content-Type"] = type }
            let url = request.url ?? URL(string: "https://example.com")!
            let response = HTTPURLResponse(url: url, statusCode: status, httpVersion: "HTTP/1.1", headerFields: headers)!
            return (body, response)
        }
    }
}

private let webCtx = ToolContext(root: URL(fileURLWithPath: NSTemporaryDirectory()))

private func fetchArgs(_ url: String) -> ToolArgs { ToolArgs(["url": url]) }

struct HTMLTextTests {
    static let page = """
    <!DOCTYPE html>
    <html><head><title> My &amp; Page </title><meta charset="utf-8"><style>body { color: red }</style></head>
    <body>
    <script>var x = "<p>nope</p>";</script>
    <!-- a comment -->
    <h1>Welcome</h1>
    <p>Hello   <b>big</b>
       world &amp; &lt;friends&gt; &quot;q&quot; &#39;s&#39; &#65;&#x42; &apos;a&apos;.</p>
    <p>See <a href="https://example.com/docs?a=1&amp;b=2">the docs</a>, <a href="#top">top</a>, <a href="javascript:void(0)">js</a>.</p>
    <h3 class="x">Steps</h3>
    <ul><li>one</li><li>two</li></ul>
    <table><tr><td>A</td><td>B</td></tr></table>
    <pre>let a = 1 &lt; 2
      indented <span>tag</span>
    end</pre>
    <p>Call <code>foo()</code> now.</p>
    <noscript>enable js</noscript>
    </body></html>
    """

    @Test func convertsTheFixture() {
        let t = HTMLText.text(Self.page)
        #expect(t.hasPrefix("# My & Page\n"))
        #expect(!t.contains("color: red"))
        #expect(!t.contains("nope"))
        #expect(!t.contains("a comment"))
        #expect(!t.contains("enable js"))
        #expect(t.contains("\n# Welcome\n"))
        #expect(t.contains("### Steps"))
        #expect(t.contains("Hello big world & <friends> \"q\" 's' AB 'a'."))
        #expect(t.contains("[the docs](https://example.com/docs?a=1&b=2)"))
        #expect(t.contains(", top, js."))
        #expect(!t.contains("](#top)"))
        #expect(t.contains("- one\n- two"))
        #expect(t.contains("| A | B"))
        #expect(t.contains("```\nlet a = 1 < 2\n  indented tag\nend\n```"))
        #expect(t.contains("Call `foo()` now."))
        #expect(!t.contains("\n\n\n"))
        #expect(t == t.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    @Test func noTitleNoHeadingLine() {
        #expect(HTMLText.text("<html><head><title> </title></head><body><p>x</p></body></html>") == "x")
    }

    @Test func plainTextPassesThrough() {
        #expect(HTMLText.text("just  some\ttext") == "just some text")
    }

    @Test func entitiesAreDecodedOnce() {
        #expect(HTMLText.text("<p>&amp;lt; &nbsp;x</p>") == "&lt; x")
    }
}

struct WebFetchTests {
    @Test func htmlBecomesText() async throws {
        let web = FakeWeb("<html><head><title>Hi</title></head><body><h2>Sub</h2><p>Body</p><script>x()</script></body></html>")
        let out = try await WebFetchTool(fetch: web.fetcher).run(fetchArgs("https://docs.example.com/a/b"), webCtx)
        #expect(!out.isError)
        #expect(out.text == "URL: https://docs.example.com/a/b\n\n# Hi\n\n## Sub\n\nBody")
    }

    @Test func plainTextAndJSONAreKept() async throws {
        let plain = try await WebFetchTool(fetch: FakeWeb(type: "text/plain", "line 1\n<b>raw</b>").fetcher)
            .run(fetchArgs("https://example.com/r.txt"), webCtx)
        #expect(plain.text == "URL: https://example.com/r.txt\n\nline 1\n<b>raw</b>")
        let json = try await WebFetchTool(fetch: FakeWeb(type: "application/json", "{\"a\": 1}").fetcher)
            .run(fetchArgs("https://example.com/a.json"), webCtx)
        #expect(json.text.hasSuffix("{\"a\": 1}"))
        #expect(!json.isError)
    }

    @Test func notFoundIsAnError() async throws {
        let web = FakeWeb(status: 404, "<html><body><p>Nothing here</p></body></html>")
        let out = try await WebFetchTool(fetch: web.fetcher).run(fetchArgs("https://example.com/missing"), webCtx)
        #expect(out.isError)
        #expect(out.text == "HTTP 404 from https://example.com/missing\nNothing here")
    }

    @Test func binaryIsLeftOut() async throws {
        let web = FakeWeb(type: "image/png", body: Data([0x89, 0x50, 0x4E, 0x47, 0, 1, 2]))
        let out = try await WebFetchTool(fetch: web.fetcher).run(fetchArgs("https://example.com/x.png"), webCtx)
        #expect(out.text == "(image/png, 7 bytes: not text, left out)")
        #expect(!out.isError)
    }

    @Test func missingTypeIsSniffed() async throws {
        let text = try await WebFetchTool(fetch: FakeWeb(type: nil, "hello").fetcher).run(fetchArgs("https://example.com/"), webCtx)
        #expect(text.text.hasSuffix("hello"))
        let bin = try await WebFetchTool(fetch: FakeWeb(type: nil, body: Data([0xFF, 0xFE, 0xFD])).fetcher)
            .run(fetchArgs("https://example.com/"), webCtx)
        #expect(bin.text.contains("not text"))
    }

    @Test func longPagesAreCut() async throws {
        let web = FakeWeb(type: "text/plain", String(repeating: "x", count: 50_000))
        let out = try await WebFetchTool(fetch: web.fetcher).run(fetchArgs("https://example.com/big"), webCtx)
        #expect(out.text.contains("characters left out"))
        #expect(out.text.count < 31_000)
    }

    @Test func sendsTheRequestHeaders() async throws {
        let web = FakeWeb("<p>x</p>")
        _ = try await WebFetchTool(fetch: web.fetcher).run(fetchArgs("https://example.com/p?q=1"), webCtx)
        let r = try #require(web.requests.first)
        #expect(r.url?.absoluteString == "https://example.com/p?q=1")
        #expect(r.timeoutInterval == 30)
        #expect(r.value(forHTTPHeaderField: "User-Agent") == "AIStudio-Agent/1.0")
        #expect(r.value(forHTTPHeaderField: "Accept") == "text/html, text/plain, application/json;q=0.9, */*;q=0.5")
    }

    @Test func previewNamesTheDomain() throws {
        let tool = WebFetchTool(fetch: FakeWeb("").fetcher)
        let p = try tool.preview(fetchArgs("https://Docs.Example.com/guide/intro"), webCtx)
        #expect(p.domain == "docs.example.com")
        #expect(p.kind == .read)
        #expect(p.title == "Fetch Docs.Example.com/guide/intro")
        #expect(p.detail == "https://Docs.Example.com/guide/intro")
        #expect(p.paths.isEmpty)
        #expect(tool.effect == .network)
        #expect(tool.name == "web_fetch")
    }

    @Test func refusesOtherSchemes() async throws {
        let web = FakeWeb("")
        let tool = WebFetchTool(fetch: web.fetcher)
        for bad in ["file:///etc/passwd", "ftp://x", "https://", "not a url", ""] {
            #expect(throws: ToolError("web_fetch needs an http or https URL")) { try tool.preview(fetchArgs(bad), webCtx) }
            await #expect(throws: ToolError.self) { try await tool.run(fetchArgs(bad), webCtx) }
        }
        #expect(web.requests.isEmpty)
    }
}

struct WebPermissionTests {
    private func decide(_ mode: PermissionMode, _ domain: String = "docs.example.com", _ rules: AllowRules = AllowRules()) -> PermissionDecision {
        PermissionEngine.decide(effect: .network, inProject: true, domain: domain, mode: mode, rules: rules)
    }

    @Test func askModeAsks() { #expect(decide(.ask) == .ask) }

    @Test func projectPagesDontAsk() {
        // open_page on the project's own HTML file: no host, a path in the project.
        #expect(PermissionEngine.decide(effect: .network, inProject: true, domain: nil, mode: .ask, rules: AllowRules()) == .allow)
        #expect(PermissionEngine.decide(effect: .network, inProject: false, domain: nil, mode: .ask, rules: AllowRules()) == .ask)
    }

    @Test func otherModesAllow() {
        #expect(decide(.acceptEdits) == .allow)
        #expect(decide(.plan) == .allow)
        #expect(decide(.bypass) == .allow)
    }

    @Test func alwaysAllowedDomains() {
        let rules = AllowRules(domains: ["example.com"])
        #expect(decide(.ask, "docs.example.com", rules) == .allow)
        #expect(decide(.ask, "example.com", rules) == .allow)
        #expect(decide(.ask, "badexample.com", rules) == .ask)
    }
}

struct WebSearchTests {
    static let brave = """
    {"web": {"results": [
      {"title": "Swift <strong>Concurrency</strong>", "url": "https://swift.org/c", "description": "Learn <strong>async</strong> &amp; await."},
      {"title": "Second", "url": "https://example.com/2", "description": ""}
    ]}}
    """

    @Test func sendsKeyAndEncodedQuery() async throws {
        let web = FakeWeb(type: "application/json", Self.brave)
        let tool = WebSearchTool(apiKey: "KEY123", fetch: web.fetcher)
        _ = try await tool.run(ToolArgs(["query": "swift & c++ tips", "count": 3]), webCtx)
        let r = try #require(web.requests.first)
        #expect(r.httpMethod == nil || r.httpMethod == "GET")
        #expect(r.url?.absoluteString == "https://api.search.brave.com/res/v1/web/search?q=swift%20%26%20c%2B%2B%20tips&count=3")
        #expect(r.value(forHTTPHeaderField: "X-Subscription-Token") == "KEY123")
        #expect(r.value(forHTTPHeaderField: "Accept") == "application/json")
    }

    @Test func countDefaultsAndClamps() async throws {
        let web = FakeWeb(type: "application/json", Self.brave)
        let tool = WebSearchTool(apiKey: "k", fetch: web.fetcher)
        _ = try await tool.run(ToolArgs(["query": "a"]), webCtx)
        _ = try await tool.run(ToolArgs(["query": "a", "count": 99]), webCtx)
        _ = try await tool.run(ToolArgs(["query": "a", "count": 0]), webCtx)
        #expect(web.requests.map { $0.url?.query } == ["q=a&count=5", "q=a&count=10", "q=a&count=1"])
    }

    @Test func formatsTheResults() async throws {
        let web = FakeWeb(type: "application/json", Self.brave)
        let out = try await WebSearchTool(apiKey: "k", fetch: web.fetcher).run(ToolArgs(["query": "swift"]), webCtx)
        #expect(!out.isError)
        #expect(out.text == """
        1. Swift Concurrency
           https://swift.org/c
           Learn async & await.

        2. Second
           https://example.com/2
        """)
    }

    @Test func noResults() async throws {
        let web = FakeWeb(type: "application/json", "{\"web\": {\"results\": []}}")
        let out = try await WebSearchTool(apiKey: "k", fetch: web.fetcher).run(ToolArgs(["query": "zzz"]), webCtx)
        #expect(out.text == "No results for “zzz”.")
        let none = try await WebSearchTool(apiKey: "k", fetch: FakeWeb(type: "application/json", "{}").fetcher)
            .run(ToolArgs(["query": "zzz"]), webCtx)
        #expect(none.text == "No results for “zzz”.")
    }

    @Test func refusedKey() async throws {
        for code in [401, 403] {
            let web = FakeWeb(status: code, type: "application/json", "{\"error\": \"bad\"}")
            let out = try await WebSearchTool(apiKey: "bad", fetch: web.fetcher).run(ToolArgs(["query": "x"]), webCtx)
            #expect(out.isError)
            #expect(out.text == "The search API key was refused (HTTP \(code)).")
        }
    }

    @Test func otherHTTPErrors() async throws {
        let web = FakeWeb(status: 429, type: "application/json", "slow down")
        let out = try await WebSearchTool(apiKey: "k", fetch: web.fetcher).run(ToolArgs(["query": "x"]), webCtx)
        #expect(out.isError)
        #expect(out.text.contains("429"))
        #expect(out.text.contains("slow down"))
    }

    @Test func previewAndSpec() throws {
        let tool = WebSearchTool(apiKey: "k", fetch: FakeWeb("").fetcher)
        let p = try tool.preview(ToolArgs(["query": "swift"]), webCtx)
        #expect(p.title == "Search the web for “swift”")
        #expect(p.domain == "api.search.brave.com")
        #expect(p.kind == .read)
        #expect(tool.effect == .network)
        #expect(tool.name == "web_search")
    }
}

struct OpenPageTests {
    func args(_ d: [String: Any]) -> ToolArgs { ToolArgs(d) }

    func project() throws -> ToolContext {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("open-page-\(UUID().uuidString.prefix(6))")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("site"), withIntermediateDirectories: true)
        try "<title>T</title>".write(to: root.appendingPathComponent("site/index.html"), atomically: true, encoding: .utf8)
        return ToolContext(root: root)
    }

    @Test func addressesAndTheirPermission() throws {
        let ctx = try project()
        let web = try OpenPageTool().preview(args(["url": "https://github.com/a/b/pull/3"]), ctx)
        #expect(web.domain == "github.com" && web.title == "Open github.com/a/b/pull/3 in the browser")
        let local = try OpenPageTool().preview(args(["url": "localhost:3000"]), ctx)
        #expect(local.domain == "localhost" && local.detail == "http://localhost:3000")
        let file = try OpenPageTool().preview(args(["url": "site/index.html"]), ctx)
        #expect(file.domain == nil && file.paths.first?.path.hasSuffix("site/index.html") == true)
        #expect(file.title == "Open site/index.html in the browser")
        #expect(throws: ToolError.self) { try OpenPageTool().preview(self.args(["url": "ftp://example.com"]), ctx) }
        #expect(throws: ToolError.self) { try OpenPageTool().preview(self.args(["url": "missing.html"]), ctx) }
    }

    @Test func returnsWhatTheTabLoaded() async throws {
        let ctx = try project()
        let asked = Asked()
        ctx.openPage = { _, url, shot in
            await asked.set(url, shot)
            return PageView(url: url.absoluteString, title: "Docs", text: "Hello page", image: shot ? "data:image/jpeg;base64,AA==" : nil)
        }
        let out = try await OpenPageTool().run(args(["url": "https://example.com/docs", "screenshot": true]), ctx)
        #expect(out.text == "Opened in the Browser tab.\nURL: https://example.com/docs\nTitle: Docs\n\nHello page")
        #expect(out.images == ["data:image/jpeg;base64,AA=="] && out.detail == "https://example.com/docs" && !out.isError)
        let (askedURL, askedShot) = (await asked.url, await asked.shot)
        #expect(askedURL?.absoluteString == "https://example.com/docs" && askedShot == true)
        let quiet = try await OpenPageTool().run(args(["url": "https://example.com", "read": false]), ctx)
        #expect(!quiet.text.contains("Hello page") && quiet.images.isEmpty)
    }

    @Test func failuresAndNoBrowser() async throws {
        let ctx = try project()
        let none = try await OpenPageTool().run(args(["url": "https://example.com"]), ctx)
        #expect(none.isError && none.text.contains("web_fetch"))
        ctx.openPage = { _, url, _ in PageView(url: url.absoluteString, error: "Could not connect to the server.") }
        let down = try await OpenPageTool().run(args(["url": "localhost:5173"]), ctx)
        #expect(down.isError && down.text == "Couldn't load http://localhost:5173: Could not connect to the server.")
    }
}

private actor Asked {
    var url: URL?
    var shot: Bool?
    func set(_ u: URL, _ s: Bool) { url = u; shot = s }
}
