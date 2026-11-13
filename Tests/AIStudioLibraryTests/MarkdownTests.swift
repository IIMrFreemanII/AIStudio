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

    @Test func shellBlocksGiveTheirCommands() {
        #expect(ShellBlock.commands(language: "bash", code: "$ ls -la\n$ git status\n") == "ls -la\ngit status")
        #expect(ShellBlock.commands(language: "Bash title=x", code: "\ngh auth login\n\n") == "gh auth login")
        #expect(ShellBlock.commands(language: "console", code: "$ echo hi\nhi\n% pwd\n/tmp") == "echo hi\npwd")
        #expect(ShellBlock.commands(language: "sh", code: "for f in *; do\n  echo $f\ndone") == "for f in *; do\n  echo $f\ndone")
        #expect(ShellBlock.commands(language: "python", code: "print(1)") == nil)
        #expect(ShellBlock.commands(language: "", code: "ls") == nil)
        #expect(ShellBlock.commands(language: "bash", code: "\n  \n") == nil)
        #expect(ShellBlock.commands(language: "console", code: "just output") == nil)
    }

    @Test func notHeadingsOrLists() {
        #expect(Markdown.blocks("#hashtag\n-not a list\n3.5 is a number") == [
            .paragraph("#hashtag\n-not a list\n3.5 is a number"),
        ])
    }

    @Test func anImageOnItsOwnLineIsMedia() {
        #expect(Markdown.blocks("Look:\n![the chart](out/chart.png)\n![clip](https://x.io/a.mp4 \"A clip\")\nDone") == [
            .paragraph("Look:"), .media(alt: "the chart", src: "out/chart.png"),
            .media(alt: "clip", src: "https://x.io/a.mp4"), .paragraph("Done"),
        ])
        // Inside a sentence it stays inline text.
        #expect(Markdown.blocks("See ![x](a.png) here") == [.paragraph("See ![x](a.png) here")])
    }
}
