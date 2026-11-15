import Foundation

/// A language the editor can highlight. Detection by file extension or fence name is not part of this type.
public enum CodeLanguage: String, CaseIterable, Sendable, Codable {
    case swift, python, javascript, typescript, tsx, json, bash, c, cpp, rust, go, html, css, yaml, toml, markdown
    /// Used only through injection from markdown.
    case markdownInline = "markdown_inline"

    public var displayName: String {
        switch self {
        case .swift: "Swift"
        case .python: "Python"
        case .javascript: "JavaScript"
        case .typescript: "TypeScript"
        case .tsx: "TSX"
        case .json: "JSON"
        case .bash: "Shell"
        case .c: "C"
        case .cpp: "C++"
        case .rust: "Rust"
        case .go: "Go"
        case .html: "HTML"
        case .css: "CSS"
        case .yaml: "YAML"
        case .toml: "TOML"
        case .markdown, .markdownInline: "Markdown"
        }
    }
}
