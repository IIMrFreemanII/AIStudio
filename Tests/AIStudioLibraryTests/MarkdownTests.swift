import Testing
@testable import AIStudioLibrary

/// The chat view's Markdown blocks.
struct MarkdownTests {
    @Test func blocksOfATypicalReply() {
        let md = """
        # Plan

        First paragraph
        continues here.

        - one
        - two
          - nested
        1. first
        2. second

        > a quote
        > more

        | a | b |
        |---|:-:|
        | 1 | 2 |
        | 3 |

        ---
        ```swift
        let x = 1
        ```
        - [x] done
        """
        #expect(Markdown.blocks(md) == [
            .heading(level: 1, text: "Plan"),
            .paragraph("First paragraph\ncontinues here."),
            .list(items: [MarkdownListItem(depth: 0, number: nil, text: "one"),
                          MarkdownListItem(depth: 0, number: nil, text: "two"),
                          MarkdownListItem(depth: 1, number: nil, text: "nested"),
                          MarkdownListItem(depth: 0, number: 1, text: "first"),
                          MarkdownListItem(depth: 0, number: 2, text: "second")]),
            .quote("a quote\nmore"),
            .table(header: ["a", "b"], rows: [["1", "2"], ["3", ""]]),
            .rule,
            .code(language: "swift", text: "let x = 1", closed: true),
            .list(items: [MarkdownListItem(depth: 0, number: nil, text: "done", checked: true)]),
        ])
    }

    @Test func anUnclosedFenceRunsToTheEnd() {
        #expect(Markdown.blocks("Here:\n```py\nprint(1)\n") == [
            .paragraph("Here:"), .code(language: "py", text: "print(1)\n", closed: false),
        ])
    }

    @Test func notHeadingsOrLists() {
        #expect(Markdown.blocks("#hashtag\n-not a list\n3.5 is a number") == [
            .paragraph("#hashtag\n-not a list\n3.5 is a number"),
        ])
    }
}
