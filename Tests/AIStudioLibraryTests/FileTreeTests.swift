import Testing
@testable import AIStudioLibrary

/// The coding chat's file explorer: the tree from a file list, and the rows shown.
struct FileTreeTests {
    let tree = FileTree(paths: ["README.md", "Sources/App/main.swift", "Sources/App/View10.swift",
                                "Sources/App/View2.swift", "Sources/Lib.swift", "docs/a.md", "Package.swift"])

    @Test func foldersFirstThenFilesInNaturalOrder() {
        #expect(tree.rows(expanded: []).map(\.name) == ["docs", "Sources", "Package.swift", "README.md"])
        let open = tree.rows(expanded: ["Sources", "Sources/App"])
        #expect(open.map(\.path) == ["docs", "Sources", "Sources/App", "Sources/App/main.swift",
                                     "Sources/App/View2.swift", "Sources/App/View10.swift", "Sources/Lib.swift",
                                     "Package.swift", "README.md"])
        #expect(open.map(\.depth) == [0, 0, 1, 2, 2, 2, 1, 0, 0])
        #expect(open[1].open && open[1].isFolder && !open[0].open)
    }

    @Test func aFilterShowsMatchesInsideOpenFolders() {
        let rows = tree.rows(expanded: [], filter: "view")
        #expect(rows.map(\.path) == ["Sources", "Sources/App", "Sources/App/View2.swift", "Sources/App/View10.swift"])
        #expect(rows.allSatisfy { !$0.isFolder || $0.open })
        #expect(tree.rows(expanded: [], filter: "nothing").isEmpty)
    }

    @Test func filesAndParents() {
        #expect(tree.files.contains("Sources/Lib.swift") && !tree.files.contains("Sources"))
        #expect(FileTree.parents(of: "a/b/c.swift") == ["a", "a/b"])
        #expect(FileTree.parents(of: "c.swift").isEmpty)
    }
}
