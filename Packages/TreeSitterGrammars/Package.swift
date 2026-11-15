// swift-tools-version: 6.0
import PackageDescription

// Vendored tree-sitter grammars (src/, highlight and injection queries, LICENSE) for grammars whose upstream
// Package.swift cannot be used from Xcode: it decides whether to compile src/scanner.c with
// `FileManager.default.fileExists(atPath: "src/scanner.c")`, which is relative to the working directory and so false
// under xcodebuild, leaving the external scanner unlinked. The sources are unmodified copies of the tags noted below.
//   python 0.25.0, javascript 0.25.0, css 0.25.0 (tree-sitter/tree-sitter-*), yaml 0.7.2 (tree-sitter-grammars)
func grammar(_ name: String) -> Target {
    .target(
        name: "TreeSitter\(name)",
        path: "Sources/TreeSitter\(name)",
        sources: ["src/parser.c", "src/scanner.c"],
        resources: [.copy("queries")],
        publicHeadersPath: "include",
        cSettings: [.headerSearchPath("src")]
    )
}

let package = Package(
    name: "TreeSitterGrammars",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "TreeSitterPython", targets: ["TreeSitterPython"]),
        .library(name: "TreeSitterJavaScript", targets: ["TreeSitterJavaScript"]),
        .library(name: "TreeSitterCSS", targets: ["TreeSitterCSS"]),
        .library(name: "TreeSitterYAML", targets: ["TreeSitterYAML"]),
    ],
    targets: [grammar("Python"), grammar("JavaScript"), grammar("CSS"), grammar("YAML")],
    cLanguageStandard: .c11
)
