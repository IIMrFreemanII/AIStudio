import Foundation

extension CodeLanguage {
    /// Picks a language from a file path (extension, then exact file name), falling back to the shebang in `firstLine`.
    /// Extensions are matched case-insensitively; exact file names case-sensitively.
    /// Choices the brief left open: `.prettierrc` and `.eslintrc` → json; `Dockerfile` and `Makefile` → nil
    /// (no grammar), without trying the shebang.
    public static func detect(path: String, firstLine: String? = nil) -> CodeLanguage? {
        let name = path.split(separator: "/", omittingEmptySubsequences: true).last.map(String.init) ?? path
        switch name {
        case ".bashrc", ".zshrc", ".bash_profile", ".zprofile", ".profile", ".envrc": return .bash
        case ".prettierrc", ".eslintrc": return .json
        case "Cargo.lock", "uv.lock", "poetry.lock", "Pipfile": return .toml
        case "Dockerfile", "Makefile": return nil
        default: break
        }
        if let dot = name.lastIndex(of: "."), dot != name.startIndex {
            let ext = name[name.index(after: dot)...].lowercased()
            if let language = byExtension[ext] { return language }
        }
        if let firstLine, let language = fromShebang(firstLine) { return language }
        return nil
    }

    /// The language for the first word of a Markdown fence info string ("swift", "python title=x"); nil if unknown.
    public static func fromFence(_ info: String) -> CodeLanguage? {
        let word = info.split(whereSeparator: { $0.isWhitespace }).first.map { $0.lowercased() } ?? ""
        return byFence[word]
    }

    private static let byExtension: [String: CodeLanguage] = {
        var map: [String: CodeLanguage] = [:]
        func add(_ language: CodeLanguage, _ exts: [String]) { for e in exts { map[e] = language } }
        add(.swift, ["swift"])
        add(.python, ["py", "pyw", "pyi"])
        add(.javascript, ["js", "mjs", "cjs", "jsx"])
        add(.typescript, ["ts", "mts", "cts"])
        add(.tsx, ["tsx"])
        add(.json, ["json", "jsonc", "json5", "geojson", "resolved", "code-workspace"])
        add(.bash, ["sh", "bash", "zsh", "ksh", "command"])
        add(.c, ["c", "h"])
        add(.cpp, ["cc", "cpp", "cxx", "c++", "hpp", "hh", "hxx", "h++", "ipp", "m", "mm"])
        add(.rust, ["rs"])
        add(.go, ["go"])
        add(.html, ["html", "htm", "xhtml"])
        add(.css, ["css"])
        add(.yaml, ["yml", "yaml"])
        add(.toml, ["toml"])
        add(.markdown, ["md", "markdown", "mdx"])
        return map
    }()

    private static let byFence: [String: CodeLanguage] = {
        var map: [String: CodeLanguage] = [:]
        func add(_ language: CodeLanguage, _ names: [String]) { for n in names { map[n] = language } }
        add(.swift, ["swift"])
        add(.python, ["py", "python", "python3"])
        add(.javascript, ["js", "javascript", "jsx", "mjs"])
        add(.typescript, ["ts", "typescript"])
        add(.tsx, ["tsx"])
        add(.json, ["json", "jsonc"])
        add(.bash, ["sh", "bash", "zsh", "shell", "console", "shellscript"])
        add(.c, ["c", "h"])
        add(.cpp, ["cpp", "c++", "cc", "hpp", "objc", "objective-c"])
        add(.rust, ["rs", "rust"])
        add(.go, ["go", "golang"])
        add(.html, ["html"])
        add(.css, ["css"])
        add(.yaml, ["yml", "yaml"])
        add(.toml, ["toml"])
        add(.markdown, ["md", "markdown"])
        return map
    }()

    private static func fromShebang(_ line: String) -> CodeLanguage? {
        guard line.hasPrefix("#!") else { return nil }
        let parts = line.dropFirst(2).split(whereSeparator: { $0.isWhitespace }).map(String.init)
        guard var program = parts.first.map({ $0.split(separator: "/").last.map(String.init) ?? $0 }) else { return nil }
        if program == "env" {
            guard let next = parts.dropFirst().first(where: { !$0.hasPrefix("-") && !$0.contains("=") }) else { return nil }
            program = next.split(separator: "/").last.map(String.init) ?? next
        }
        switch program {
        case "sh", "bash", "zsh", "ksh", "dash": return .bash
        case "node": return .javascript
        case "swift": return .swift
        case _ where program.hasPrefix("python"): return .python
        default: return nil
        }
    }
}
