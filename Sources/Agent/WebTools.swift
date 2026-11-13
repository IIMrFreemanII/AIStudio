import Foundation
import LLMClient

/// How the web tools reach the network (tests pass a fake).
public typealias WebFetcher = @Sendable (URLRequest) async throws -> (Data, URLResponse)

public enum WebAccess {
    public static let urlSession: WebFetcher = { try await URLSession.shared.data(for: $0) }
}

// MARK: HTML to text

/// A readable text version of an HTML page: headings, lists, links and code blocks kept as markdown-ish text.
public enum HTMLText {
    public static func text(_ html: String) -> String {
        var s = html
        s = sub(s, "<!--[\\s\\S]*?-->", "")
        s = sub(s, "<(script|style|noscript|svg|template)\\b[^>]*>[\\s\\S]*?</\\1\\s*>", "")

        // The title becomes the first line; the rest of the head goes.
        var title = ""
        if let m = first(s, "<title\\b[^>]*>([\\s\\S]*?)</title\\s*>") {
            title = collapse(decode(stripTags(m)))
        }
        s = sub(s, "<head\\b[^>]*>[\\s\\S]*?</head\\s*>", "")
        s = sub(s, "<title\\b[^>]*>[\\s\\S]*?</title\\s*>", "")

        // <pre> blocks are set aside and put back verbatim at the end.
        var pres: [String] = []
        s = replace(s, "<pre\\b[^>]*>([\\s\\S]*?)</pre\\s*>") { groups in
            var inner = sub(groups[1], "<br\\s*/?>", "\n")
            inner = decode(stripTags(inner))
            if inner.hasPrefix("\n") { inner.removeFirst() }
            while inner.hasSuffix("\n") { inner.removeLast() }
            pres.append(inner)
            return "\n\n\(marker)\(pres.count - 1)\(marker)\n\n"
        }

        // Line breaks in the source are plain whitespace.
        s = sub(s, "\\s+", " ")

        s = replace(s, "<code\\b[^>]*>([\\s\\S]*?)</code\\s*>") { g in
            let inner = stripTags(g[1])
            return collapse(inner).isEmpty ? "" : "`" + inner + "`"
        }

        s = replace(s, "<a\\b[^>]*?\\bhref\\s*=\\s*(?:\"([^\"]*)\"|'([^']*)'|([^\\s>\"']+))[^>]*>([\\s\\S]*?)</a\\s*>") { g in
            let href = (g[1].isEmpty ? (g[2].isEmpty ? g[3] : g[2]) : g[1]).trimmingCharacters(in: .whitespacesAndNewlines)
            let label = collapse(stripTags(g[4]))
            let lower = href.lowercased()
            if href.isEmpty || href.hasPrefix("#") || lower.hasPrefix("javascript:") { return label }
            return label.isEmpty ? "" : "[\(label)](\(href))"
        }

        s = replace(s, "<h([1-6])\\b[^>]*>([\\s\\S]*?)</h\\1\\s*>") { g in
            let text = collapse(stripTags(g[2]))
            return text.isEmpty ? "\n" : "\n\n" + String(repeating: "#", count: Int(g[1]) ?? 1) + " " + text + "\n\n"
        }

        s = sub(s, "<br\\b[^>]*>", "\n")
        s = sub(s, "<li\\b[^>]*>", "\n- ")
        s = sub(s, "</?(p|section|article|header|footer|main|table|ul|ol|blockquote)\\b[^>]*>", "\n\n")
        s = sub(s, "</?(div|tr|thead|tbody|tfoot)\\b[^>]*>", "\n")
        s = sub(s, "<(td|th)\\b[^>]*>", " | ")
        s = stripTags(s)
        s = decode(s)

        // Whitespace: spaces and tabs collapse, lines trim, blank runs shrink.
        s = sub(s, "[ \\t\\x{00A0}\\r\\f]+", " ")
        s = s.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.joined(separator: "\n")
        s = sub(s, "\\n{3,}", "\n\n")

        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if !title.isEmpty { s = s.isEmpty ? "# \(title)" : "# \(title)\n\n" + s }

        for (i, code) in pres.enumerated() {
            s = s.replacingOccurrences(of: "\(marker)\(i)\(marker)", with: "```\n\(code)\n```")
        }
        return s
    }

    // Private-use characters that cannot be in a decoded page's text in practice.
    private static let marker = "\u{E000}"

    private static func regex(_ pattern: String) -> NSRegularExpression {
        // The patterns are literals in this file.
        try! NSRegularExpression(pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators])
    }

    private static func sub(_ s: String, _ pattern: String, _ template: String) -> String {
        let re = regex(pattern)
        return re.stringByReplacingMatches(in: s, range: NSRange(location: 0, length: (s as NSString).length),
                                           withTemplate: NSRegularExpression.escapedTemplate(for: template))
    }

    private static func first(_ s: String, _ pattern: String) -> String? {
        let ns = s as NSString
        guard let m = regex(pattern).firstMatch(in: s, range: NSRange(location: 0, length: ns.length)) else { return nil }
        return m.numberOfRanges > 1 && m.range(at: 1).location != NSNotFound ? ns.substring(with: m.range(at: 1)) : ""
    }

    /// Replaces each match with `f(groups)`; groups[0] is the whole match, unmatched groups are "".
    private static func replace(_ s: String, _ pattern: String, _ f: ([String]) -> String) -> String {
        let ns = s as NSString
        let re = regex(pattern)
        var out = ""
        var last = 0
        for m in re.matches(in: s, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let groups = (0..<m.numberOfRanges).map { i -> String in
                let r = m.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }
            out += f(groups)
            last = m.range.location + m.range.length
        }
        return out + ns.substring(from: last)
    }

    private static func stripTags(_ s: String) -> String { sub(s, "</?[a-zA-Z!][^>]*>", "") }

    private static func collapse(_ s: String) -> String {
        sub(s, "\\s+", " ").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func decode(_ s: String) -> String {
        replace(s, "&(#[0-9]+|#x[0-9a-f]+|amp|lt|gt|quot|apos|nbsp);") { g in
            let name = g[1]
            switch name.lowercased() {
            case "amp": return "&"
            case "lt": return "<"
            case "gt": return ">"
            case "quot": return "\""
            case "apos": return "'"
            case "nbsp": return " "
            default:
                let lower = name.lowercased()
                let code = lower.hasPrefix("#x") ? UInt32(lower.dropFirst(2), radix: 16) : UInt32(name.dropFirst())
                if let code, let scalar = Unicode.Scalar(code) { return String(Character(scalar)) }
                return g[0]
            }
        }
    }
}

// MARK: web_fetch

public struct WebFetchTool: AgentTool {
    let fetch: WebFetcher
    public init(fetch: @escaping WebFetcher = WebAccess.urlSession) { self.fetch = fetch }

    public let name = "web_fetch"
    public let description = "Fetch a web page (http or https) and return it as text. Use it for documentation, issues and pages the user links. Long pages are cut."
    public var parameters: JSONValue { schema(["url": ("string", "The page's address (http or https)")], required: ["url"]) }
    public let effect = ToolEffect.network

    static func parse(_ args: ToolArgs) throws -> URL {
        let raw = try args.required("url").trimmingCharacters(in: .whitespacesAndNewlines)
        guard let url = URL(string: raw), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https",
              let host = url.host, !host.isEmpty else {
            throw ToolError("web_fetch needs an http or https URL")
        }
        return url
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = try Self.parse(args)
        let host = url.host ?? ""
        var p = ToolPreview(kind: .read, title: "Fetch \(host)\(String(url.path.prefix(60)))", detail: url.absoluteString, paths: [])
        p.domain = host.lowercased()
        return p
    }

    /// The page as text, or nil when it isn't text.
    static func render(_ data: Data, type: String?) -> String? {
        let body = data.prefix(5_000_000)
        let text = String(decoding: body, as: UTF8.self)
        guard let type, !type.isEmpty else {
            return text.contains("\u{FFFD}") ? nil : text
        }
        if type.contains("html") { return HTMLText.text(text) }
        if type.hasPrefix("text/") || ["json", "xml", "javascript"].contains(where: type.contains) { return text }
        return nil
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let url = try Self.parse(args)
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue("AIStudio-Agent/1.0", forHTTPHeaderField: "User-Agent")
        request.setValue("text/html, text/plain, application/json;q=0.9, */*;q=0.5", forHTTPHeaderField: "Accept")
        let (data, response) = try await fetch(request)
        let http = response as? HTTPURLResponse
        let type = http?.value(forHTTPHeaderField: "Content-Type")?.lowercased()
        if let http, http.statusCode >= 400 {
            let body = Self.render(data, type: type) ?? ""
            return ToolOutput("HTTP \(http.statusCode) from \(url.absoluteString)\n" + clip(body, max: 2000), isError: true)
        }
        guard let text = Self.render(data, type: type) else {
            return ToolOutput("(\(type?.ifEmpty("unknown type") ?? "unknown type"), \(data.count) bytes: not text, left out)")
        }
        let shown = (response.url ?? url).absoluteString
        return ToolOutput("URL: \(shown)\n\n" + clip(text, max: 30_000))
    }
}

// MARK: open_page

/// A page the Browser tab loaded for `open_page`.
public struct PageView: Equatable, Sendable {
    public var url: String
    public var title: String
    /// The page's visible text (nil when it didn't load).
    public var text: String?
    /// Why it didn't load.
    public var error: String?
    /// A screenshot of what the tab shows (a JPEG `data:` URL), when asked for.
    public var image: String?
    public init(url: String, title: String = "", text: String? = nil, error: String? = nil, image: String? = nil) {
        self.url = url
        self.title = title
        self.text = text
        self.error = error
        self.image = image
    }
}

public struct OpenPageTool: AgentTool {
    public init() {}
    public let name = "open_page"
    public let description = """
        Open a web page in the Browser tab of the chat's side panel, where the user sees it, and get back its title \
        and visible text once it has loaded. Use it to show the user a page (documentation, a pull request, a \
        preview) or to check a dev server you started (http://localhost:3000). `url` is an http(s) address, or the \
        path of an HTML file in the project. The tab keeps the user's sign-ins: to just read a public page, \
        web_fetch is enough.
        """
    public var parameters: JSONValue {
        schema(["url": ("string", "An http(s) address, or an HTML file's path in the project"),
                "read": ("boolean", "Return the page's text (default true); false only opens it for the user"),
                "screenshot": ("boolean", "Also return a screenshot of the page as the user sees it (vision models)")],
               required: ["url"])
    }
    public let effect = ToolEffect.network

    /// The address: http(s) as given, `localhost:3000` over http, else a file in the project.
    static func parse(_ args: ToolArgs, _ ctx: ToolContext) throws -> URL {
        let raw = try args.required("url").trimmingCharacters(in: .whitespacesAndNewlines)
        if let url = URL(string: raw), let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" {
            guard url.host?.isEmpty == false else { throw ToolError("\(raw) has no host") }
            return url
        }
        if raw.lowercased().hasPrefix("localhost:") || raw.hasPrefix("127.0.0.1:"), let url = URL(string: "http://" + raw) {
            return url
        }
        if raw.contains("://") && !raw.lowercased().hasPrefix("file://") {
            throw ToolError("open_page opens http(s) addresses and the project's HTML files")
        }
        let file = raw.lowercased().hasPrefix("file://") ? URL(string: raw).map { $0.standardizedFileURL } : ctx.resolve(raw)
        guard let file, FileManager.default.fileExists(atPath: file.path) else { throw ToolError("no such file: \(raw)") }
        return file
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let url = try Self.parse(args, ctx)
        if url.isFileURL {
            let rel = url.path.hasPrefix(ctx.root.path + "/") ? String(url.path.dropFirst(ctx.root.path.count + 1)) : url.path
            return ToolPreview(kind: .read, title: "Open \(rel) in the browser", detail: url.absoluteString, paths: [url])
        }
        let host = url.host ?? ""
        let port = url.port.map { ":\($0)" } ?? ""
        var p = ToolPreview(kind: .read, title: "Open \(host)\(port)\(String(url.path.prefix(60))) in the browser",
                            detail: url.absoluteString)
        p.domain = host.lowercased()
        return p
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let url = try Self.parse(args, ctx)
        guard let open = ctx.openPage else {
            return ToolOutput("There's no browser in this chat: use web_fetch to read the page.", isError: true)
        }
        let page = await open(ctx.callID, url, args.bool("screenshot"))
        try Task.checkCancellation()
        if let e = page.error {
            return ToolOutput("Couldn't load \(page.url): \(e)", isError: true, detail: e)
        }
        var out = "Opened in the Browser tab.\nURL: \(page.url)\nTitle: \(page.title.isEmpty ? "(none)" : page.title)"
        if args.raw["read"] == nil || args.bool("read"), let text = page.text {
            out += "\n\n" + (text.isEmpty ? "(the page shows no text)" : clip(text, max: 20_000))
        }
        return ToolOutput(out, images: page.image.map { [$0] } ?? [], detail: page.url)
    }
}

// MARK: web_search

public struct WebSearchTool: AgentTool {
    let apiKey: String
    let fetch: WebFetcher
    public init(apiKey: String, fetch: @escaping WebFetcher = WebAccess.urlSession) {
        self.apiKey = apiKey
        self.fetch = fetch
    }

    public let name = "web_search"
    public let description = "Search the web (Brave Search) and return the top results with title, URL and a snippet. Read a result with web_fetch."
    public var parameters: JSONValue {
        schema(["query": ("string", "What to search for"), "count": ("integer", "How many results, 1 to 10 (default 5)")],
               required: ["query"])
    }
    public let effect = ToolEffect.network

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        var p = ToolPreview(kind: .read, title: "Search the web for “\(try args.required("query"))”")
        p.domain = "api.search.brave.com"
        return p
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let query = try args.required("query")
        let count = min(10, max(1, args.int("count") ?? 5))
        var comps = URLComponents()
        comps.scheme = "https"
        comps.host = "api.search.brave.com"
        comps.path = "/res/v1/web/search"
        // Encoded by hand: queryItems leaves "+" and "&" in a value as they are.
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        let q = query.addingPercentEncoding(withAllowedCharacters: allowed) ?? ""
        comps.percentEncodedQuery = "q=\(q)&count=\(count)"
        guard let url = comps.url else { throw ToolError("can't search for that query") }
        var request = URLRequest(url: url)
        request.timeoutInterval = 30
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await fetch(request)
        if let code = (response as? HTTPURLResponse)?.statusCode, code >= 400 {
            if code == 401 || code == 403 {
                return ToolOutput("The search API key was refused (HTTP \(code)).", isError: true)
            }
            return ToolOutput("HTTP \(code) from the search API\n" + clip(String(decoding: data, as: UTF8.self), max: 2000), isError: true)
        }
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ToolError("the search API's answer isn't JSON")
        }
        let results = ((root["web"] as? [String: Any])?["results"] as? [[String: Any]]) ?? []
        let items = results.compactMap { r -> String? in
            guard let link = r["url"] as? String else { return nil }
            let title = HTMLText.text(r["title"] as? String ?? "").ifEmpty(link)
            let snippet = HTMLText.text(r["description"] as? String ?? "")
            var item = "\(title)\n   \(link)"
            if !snippet.isEmpty { item += "\n   " + snippet.replacingOccurrences(of: "\n", with: " ") }
            return item
        }
        guard !items.isEmpty else { return ToolOutput("No results for “\(query)”.") }
        let numbered = items.enumerated().map { "\($0.offset + 1). \($0.element)" }
        return ToolOutput(numbered.joined(separator: "\n\n"))
    }
}
