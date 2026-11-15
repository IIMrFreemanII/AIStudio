import Foundation
import XCTest
@testable import CodeEditing

final class HighlighterTests: XCTestCase {
    /// The range of the `n`th (0-based) occurrence of `needle` in `text`, in UTF-16 units.
    private func range(of needle: String, in text: String, nth: Int = 0) -> NSRange {
        var found = NSRange(location: NSNotFound, length: 0)
        var search = NSRange(location: 0, length: (text as NSString).length)
        for _ in 0...nth {
            found = (text as NSString).range(of: needle, options: [], range: search)
            if found.location == NSNotFound { break }
            search = NSRange(location: found.upperBound, length: (text as NSString).length - found.upperBound)
        }
        return found
    }

    private func assertSpan(_ spans: [HighlightSpan], _ needle: String, in text: String, _ token: SyntaxToken, nth: Int = 0,
                            file: StaticString = #filePath, line: UInt = #line) {
        let r = range(of: needle, in: text, nth: nth)
        XCTAssertNotEqual(r.location, NSNotFound, "sample lacks \(needle)", file: file, line: line)
        XCTAssertTrue(spans.contains(HighlightSpan(range: r, token: token)),
                      "no \(token) span at \(r) for \"\(needle)\"; spans: \(spans.map { "\($0.token)@\($0.range.location),\($0.range.length)" })",
                      file: file, line: line)
    }

    private func highlight(_ text: String, _ language: CodeLanguage) async -> [HighlightSpan] {
        await SyntaxHighlighter.highlight(text, language: language)
    }

    // MARK: Registry

    func testEveryLanguageHasAConfiguration() {
        for language in CodeLanguage.allCases {
            XCTAssertNotNil(GrammarRegistry.configuration(for: language), "\(language)")
        }
        print("[CodeEditing] dropped query files: \(GrammarRegistry.droppedQueryFiles)")
    }

    func testDisplayNames() {
        XCTAssertEqual(CodeLanguage.allCases.map(\.displayName),
                       ["Swift", "Python", "JavaScript", "TypeScript", "TSX", "JSON", "Shell", "C", "C++", "Rust", "Go",
                        "HTML", "CSS", "YAML", "TOML", "Markdown", "Markdown"])
    }

    func testCaptureTable() {
        let cases: [(String, SyntaxToken?)] = [
            ("keyword", .keyword), ("keyword.function", .keyword), ("keyword.directive", .directive), ("keyword.return", .keyword),
            ("conditional", .keyword), ("repeat", .keyword), ("exception", .keyword), ("storageclass", .keyword),
            ("include", .directive), ("preproc", .directive), ("define", .directive),
            ("type", .type), ("type.builtin", .type), ("constructor", .type), ("namespace", .type), ("module", .type), ("tag", .type),
            ("function", .function), ("function.method", .function), ("function.builtin", .function), ("method", .function),
            ("function.call", .function), ("function.macro", .function),
            ("string", .string), ("string.special", .string), ("character", .string), ("string.regexp", .string),
            ("string.escape", .constant), ("escape", .constant),
            ("number", .number), ("float", .number), ("number.float", .number),
            ("boolean", .constant), ("constant", .constant), ("constant.builtin", .constant),
            ("comment", .comment), ("comment.documentation", .comment),
            ("attribute", .attribute), ("decorator", .attribute), ("tag.attribute", .attribute), ("label", .attribute),
            ("operator", .operatorSymbol),
            ("punctuation", .punctuation), ("punctuation.delimiter", .punctuation), ("punctuation.bracket", .punctuation),
            ("punctuation.special", .punctuation),
            ("property", .variable), ("field", .variable), ("variable.member", .variable), ("variable.parameter", .variable),
            ("parameter", .variable),
            ("variable", nil), ("variable.builtin", .keyword), ("variable.other", nil),
            ("markup.heading", .heading), ("markup.heading.1", .heading), ("text.title", .heading),
            ("markup.raw", .code), ("markup.raw.block", .code), ("text.literal", .code),
            ("markup.link", .link), ("markup.link.url", .link), ("text.uri", .link), ("text.reference", .link),
            ("markup.quote", .quote), ("markup.list", .listMarker), ("markup.list.checked", .listMarker),
            ("markup.strikethrough", .strikethrough), ("text.strike", .strikethrough),
            ("markup", .markup), ("text.emphasis", .markup), ("text.strong", .markup), ("markup.italic", .markup),
            ("markup.bold", .markup),
            ("spell", nil), ("none", nil), ("nospell", nil), ("embedded", nil), ("_name", nil), ("_parent.keyword", nil),
            ("unknown", nil), ("", nil),
        ]
        for (name, token) in cases {
            XCTAssertEqual(SyntaxCaptures.token(for: name), token, "capture \"\(name)\"")
        }
    }

    // MARK: Languages

    func testSwift() async {
        let s = "import Foundation\n// note\nlet name = \"x\"\nfunc f() -> Int { 1 }\n"
        let spans = await highlight(s, .swift)
        assertSpan(spans, "import", in: s, .keyword)
        assertSpan(spans, "// note", in: s, .comment)
        // The Swift grammar splits a string literal into its quotes and content, which together cover it.
        let literal = range(of: "\"x\"", in: s)
        let covered = spans.filter { $0.token == .string && NSIntersectionRange($0.range, literal).length == $0.range.length }
            .reduce(0) { $0 + $1.range.length }
        XCTAssertEqual(covered, literal.length, "string literal covered")
        assertSpan(spans, "func", in: s, .keyword)
        assertSpan(spans, "1", in: s, .number, nth: 0)
    }

    func testPython() async {
        let s = "def greet(name):\n    # hi\n    return \"hello\" + name\n"
        let spans = await highlight(s, .python)
        assertSpan(spans, "def", in: s, .keyword)
        assertSpan(spans, "# hi", in: s, .comment)
        assertSpan(spans, "\"hello\"", in: s, .string)
    }

    func testJavaScript() async {
        let s = "const x = 42;\n// hi\nfunction f() { return \"a\"; }\n"
        let spans = await highlight(s, .javascript)
        assertSpan(spans, "const", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
        assertSpan(spans, "42", in: s, .number)
    }

    func testTypeScript() async {
        let s = "interface A { n: number }\n// hi\nconst x: string = \"a\";\n"
        let spans = await highlight(s, .typescript)
        assertSpan(spans, "interface", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
        assertSpan(spans, "const", in: s, .keyword)
    }

    func testTSX() async {
        let s = "const App = () => <div className=\"a\">hi</div>;\n// hi\n"
        let spans = await highlight(s, .tsx)
        assertSpan(spans, "const", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
        assertSpan(spans, "div", in: s, .type)
    }

    func testJSON() async {
        let s = "{\n  \"name\": \"x\",\n  \"n\": 12,\n  \"ok\": true\n}\n"
        let spans = await highlight(s, .json)
        assertSpan(spans, "\"x\"", in: s, .string)
        assertSpan(spans, "12", in: s, .number)
    }

    func testBash() async {
        let s = "if [ -f a ]; then\n  echo \"hi\" # note\nfi\n"
        let spans = await highlight(s, .bash)
        assertSpan(spans, "if", in: s, .keyword)
        assertSpan(spans, "\"hi\"", in: s, .string)
        assertSpan(spans, "# note", in: s, .comment)
    }

    func testC() async {
        let s = "#include <stdio.h>\n// hi\nint main(void) { return 0; }\nchar *s = \"a\";\n"
        let spans = await highlight(s, .c)
        assertSpan(spans, "return", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
        assertSpan(spans, "int", in: s, .type)
    }

    func testCpp() async {
        let s = "#include <vector>\n// hi\nclass A { public: int x = 1; };\nauto s = \"a\";\n"
        let spans = await highlight(s, .cpp)
        assertSpan(spans, "class", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
    }

    func testRust() async {
        let s = "fn main() {\n    // hi\n    let s = \"a\";\n}\n"
        let spans = await highlight(s, .rust)
        assertSpan(spans, "fn", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
    }

    func testGo() async {
        let s = "package main\n// hi\nfunc main() {\n\ts := \"a\"\n}\n"
        let spans = await highlight(s, .go)
        assertSpan(spans, "func", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
        assertSpan(spans, "\"a\"", in: s, .string)
    }

    func testHTML() async {
        let s = "<!-- hi -->\n<div class=\"a\">text</div>\n"
        let spans = await highlight(s, .html)
        assertSpan(spans, "div", in: s, .type)
        assertSpan(spans, "<!-- hi -->", in: s, .comment)
        assertSpan(spans, "class", in: s, .attribute)
        // The attribute value's string span is the text between the quotes.
        let quoted = range(of: "\"a\"", in: s)
        XCTAssertTrue(spans.contains(HighlightSpan(range: NSRange(location: quoted.location + 1, length: 1), token: .string)))
    }

    func testHTMLInjectsCSSAndJavaScript() async {
        let s = "<style>\na { color: red; }\n</style>\n<script>\nconst x = 1;\n</script>\n"
        let spans = await highlight(s, .html)
        assertSpan(spans, "const", in: s, .keyword)
        XCTAssertTrue(spans.contains { $0.range.location >= range(of: "a {", in: s).location
            && $0.range.upperBound <= range(of: "</style>", in: s).location && $0.token != .punctuation },
            "no CSS span inside <style>")
    }

    func testCSS() async {
        let s = "/* hi */\na.b { color: red; width: 10px; content: \"x\"; }\n"
        let spans = await highlight(s, .css)
        assertSpan(spans, "/* hi */", in: s, .comment)
        assertSpan(spans, "\"x\"", in: s, .string)
        XCTAssertTrue(spans.contains { $0.token == .variable || $0.token == .type || $0.token == .attribute }, "property/selector tokens")
        assertSpan(spans, "10px", in: s, .number)
    }

    func testYAML() async {
        let s = "# hi\nname: \"x\"\ncount: 3\nlist:\n  - a\n"
        let spans = await highlight(s, .yaml)
        assertSpan(spans, "# hi", in: s, .comment)
        assertSpan(spans, "\"x\"", in: s, .string)
        assertSpan(spans, "3", in: s, .number)
        XCTAssertTrue(spans.contains { $0.token == .variable }, "keys")
    }

    func testTOML() async {
        let s = "# hi\n[server]\nname = \"x\"\nport = 80\n"
        let spans = await highlight(s, .toml)
        assertSpan(spans, "# hi", in: s, .comment)
        assertSpan(spans, "\"x\"", in: s, .string)
        assertSpan(spans, "80", in: s, .number)
        XCTAssertTrue(spans.contains { $0.token == .variable || $0.token == .type }, "keys or tables")
    }

    func testMarkdownHeadingAndInline() async {
        let s = "# Title\n\nSome `code` and [a link](http://x.y).\n"
        let spans = await highlight(s, .markdown)
        XCTAssertTrue(spans.contains { $0.token == .heading && NSIntersectionRange($0.range, range(of: "Title", in: s)).length > 0 }, "heading")
        let code = range(of: "`code`", in: s)
        XCTAssertTrue(spans.contains { $0.token == .code && NSIntersectionRange($0.range, code).length > 0 }, "inline code via injection")
        let link = range(of: "[a link](http://x.y)", in: s)
        XCTAssertTrue(spans.contains { $0.token == .link && NSIntersectionRange($0.range, link).length > 0 }, "link via injection")
    }

    func testMarkdownFencedSwiftBlockIsHighlightedAsSwift() async {
        let s = "Text\n\n```swift\nlet x = 1 // hi\n```\n"
        let spans = await highlight(s, .markdown)
        assertSpan(spans, "let", in: s, .keyword)
        assertSpan(spans, "// hi", in: s, .comment)
    }

    func testMarkdownInlineAlone() async {
        let s = "a `b` c"
        let spans = await highlight(s, .markdownInline)
        XCTAssertTrue(spans.contains { $0.token == .code })
    }

    // MARK: Offsets and ordering

    func testNonASCIIOffsetsAreUTF16() async {
        let s = "let s = \"héllo 🎉\" // ok"
        let spans = await highlight(s, .swift)
        let comment = range(of: "// ok", in: s)
        // The emoji is two UTF-16 units: the comment starts at 19 (18 in Characters, 22 in UTF-8 bytes).
        XCTAssertEqual(comment.location, 19)
        assertSpan(spans, "// ok", in: s, .comment)
        // The string's content (h é l l o, a space, 🎉 = 8 UTF-16 units) sits between the quotes at 8 and 17.
        XCTAssertTrue(spans.contains(HighlightSpan(range: NSRange(location: 9, length: 8), token: .string)))
        assertSpan(spans, "let", in: s, .keyword)
    }

    func testSpansAreFilteredAndOrdered() async {
        let s = "let a = 1\nlet b = \"two\" // c\nlet c = 3\n"
        let h = SyntaxHighlighter(language: .swift)
        await h.update(text: s)
        let full = await h.spans(in: NSRange(location: 0, length: (s as NSString).length))
        XCTAssertFalse(full.isEmpty)
        for (a, b) in zip(full, full.dropFirst()) {
            XCTAssertTrue(a.range.location < b.range.location
                          || (a.range.location == b.range.location && a.range.length >= b.range.length), "order \(a) \(b)")
        }
        let line2 = range(of: "let b = \"two\" // c", in: s)
        let part = await h.spans(in: line2)
        XCTAssertFalse(part.isEmpty)
        XCTAssertTrue(part.allSatisfy { NSIntersectionRange($0.range, line2).length > 0 })
        XCTAssertTrue(full.contains { NSIntersectionRange($0.range, line2).length == 0 }, "sample has spans outside the range")
        XCTAssertTrue(part.count < full.count)
        let none = await h.spans(in: NSRange(location: 0, length: 0))
        XCTAssertTrue(none.isEmpty)
    }
}
