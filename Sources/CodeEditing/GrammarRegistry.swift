import Foundation
import SwiftTreeSitter
import SwiftTreeSitterLayer
import TreeSitterBash
import TreeSitterC
import TreeSitterCPP
import TreeSitterCSS
import TreeSitterGo
import TreeSitterHTML
import TreeSitterJSON
import TreeSitterJavaScript
import TreeSitterMarkdown
import TreeSitterMarkdownInline
import TreeSitterPython
import TreeSitterRust
import TreeSitterSwift
import TreeSitterTOML
import TreeSitterTSX
import TreeSitterTypeScript
import TreeSitterYAML

/// Builds and caches one tree-sitter `LanguageConfiguration` per language. Query files are read from the grammar
/// packages' SwiftPM resource bundles (`<Package>_<Target>.bundle/queries`), in the app's `Contents/Resources` and next
/// to the test bundle under `xcodebuild test`.
public enum GrammarRegistry {
    /// The configuration for a language, or nil when its grammar or queries can't be built (show plain text).
    public static func configuration(for language: CodeLanguage) -> LanguageConfiguration? {
        cache.configuration(for: language)
    }

    /// The language a tree-sitter injection name ("javascript", "sh", "ts", "markdown_inline"…) stands for.
    public static func language(forInjection name: String) -> CodeLanguage? {
        let key = name.trimmingCharacters(in: .whitespaces).lowercased()
        if let direct = CodeLanguage(rawValue: key) { return direct }
        switch key {
        case "js", "jsx", "mjs", "cjs": return .javascript
        case "ts": return .typescript
        case "sh", "shell", "zsh", "console": return .bash
        case "py", "python3": return .python
        case "rs": return .rust
        case "golang": return .go
        case "c++", "cc", "cxx", "hpp", "h": return .cpp
        case "yml": return .yaml
        case "md": return .markdown
        case "htm": return .html
        case "jsonc", "json5": return .json
        default: return nil
        }
    }

    /// Query files left out because they did not compile against their grammar, as "language/file" (diagnostics).
    public static var droppedQueryFiles: [String] { cache.dropped() }

    private static let cache = Cache()

    private final class Cache: @unchecked Sendable {
        private let lock = NSLock()
        private var built: [CodeLanguage: LanguageConfiguration?] = [:]
        private var droppedFiles: [String] = []

        func dropped() -> [String] { lock.lock(); defer { lock.unlock() }; return droppedFiles }

        func configuration(for language: CodeLanguage) -> LanguageConfiguration? {
            lock.lock(); defer { lock.unlock() }
            if let hit = built[language] { return hit }
            let (config, dropped) = build(language)
            droppedFiles.append(contentsOf: dropped.map { "\(language.rawValue)/\($0)" })
            built[language] = .some(config)
            return config
        }
    }

    private struct Source {
        var bundle: String
        var files: [String]
    }

    /// The grammar's parser, the highlight query files to concatenate (in order), and the injections source.
    private static func spec(for language: CodeLanguage) -> (language: Language, highlights: [(String, String)], injections: Source?)? {
        func lang(_ p: OpaquePointer?) -> Language? { p.map { Language($0) } }
        func bundle(_ package: String, _ target: String? = nil) -> String { "\(package)_\(target ?? package)" }
        let js = bundle("TreeSitterJavaScript")
        let ts = bundle("TreeSitterTypeScript")
        switch language {
        case .swift:
            guard let l = lang(tree_sitter_swift()) else { return nil }
            let b = bundle("TreeSitterSwift")
            return (l, [(b, "highlights.scm")], Source(bundle: b, files: ["injections.scm"]))
        case .python:
            guard let l = lang(tree_sitter_python()) else { return nil }
            return (l, [(bundle("TreeSitterPython"), "highlights.scm")], nil)
        case .javascript:
            guard let l = lang(tree_sitter_javascript()) else { return nil }
            return (l, [(js, "highlights.scm"), (js, "highlights-jsx.scm")], Source(bundle: js, files: ["injections.scm"]))
        case .typescript:
            guard let l = lang(tree_sitter_typescript()) else { return nil }
            return (l, [(ts, "highlights.scm"), (js, "highlights.scm")], nil)
        case .tsx:
            guard let l = lang(tree_sitter_tsx()) else { return nil }
            let tsx = bundle("TreeSitterTypeScript", "TreeSitterTSX")
            return (l, [(tsx, "highlights.scm"), (js, "highlights-jsx.scm"), (js, "highlights.scm")], nil)
        case .json:
            guard let l = lang(tree_sitter_json()) else { return nil }
            return (l, [(bundle("TreeSitterJSON"), "highlights.scm")], nil)
        case .bash:
            guard let l = lang(tree_sitter_bash()) else { return nil }
            return (l, [(bundle("TreeSitterBash"), "highlights.scm")], nil)
        case .c:
            guard let l = lang(tree_sitter_c()) else { return nil }
            return (l, [(bundle("TreeSitterC"), "highlights.scm")], nil)
        case .cpp:
            guard let l = lang(tree_sitter_cpp()) else { return nil }
            let cpp = bundle("TreeSitterCPP")
            return (l, [(bundle("TreeSitterC"), "highlights.scm"), (cpp, "highlights.scm")],
                    Source(bundle: cpp, files: ["injections.scm"]))
        case .rust:
            guard let l = lang(tree_sitter_rust()) else { return nil }
            let b = bundle("TreeSitterRust")
            return (l, [(b, "highlights.scm")], Source(bundle: b, files: ["injections.scm"]))
        case .go:
            guard let l = lang(tree_sitter_go()) else { return nil }
            return (l, [(bundle("TreeSitterGo"), "highlights.scm")], nil)
        case .html:
            guard let l = lang(tree_sitter_html()) else { return nil }
            let b = bundle("TreeSitterHTML")
            return (l, [(b, "highlights.scm")], Source(bundle: b, files: ["injections.scm"]))
        case .css:
            guard let l = lang(tree_sitter_css()) else { return nil }
            return (l, [(bundle("TreeSitterCSS"), "highlights.scm")], nil)
        case .yaml:
            guard let l = lang(tree_sitter_yaml()) else { return nil }
            return (l, [(bundle("TreeSitterYAML"), "highlights.scm")], nil)
        case .toml:
            guard let l = lang(tree_sitter_toml()) else { return nil }
            return (l, [(bundle("TreeSitterTOML"), "highlights.scm")], nil)
        case .markdown:
            guard let l = lang(tree_sitter_markdown()) else { return nil }
            let b = bundle("TreeSitterMarkdown")
            return (l, [(b, "highlights.scm")], Source(bundle: b, files: ["injections.scm"]))
        case .markdownInline:
            guard let l = lang(tree_sitter_markdown_inline()) else { return nil }
            let b = bundle("TreeSitterMarkdown", "TreeSitterMarkdownInline")
            return (l, [(b, "highlights.scm")], Source(bundle: b, files: ["injections.scm"]))
        }
    }

    private static func build(_ language: CodeLanguage) -> (LanguageConfiguration?, dropped: [String]) {
        guard let spec = spec(for: language) else { return (nil, []) }
        var dropped: [String] = []
        var queries: [Query.Definition: Query] = [:]

        // Highlights: read each file, keep the ones that compile alone, then compile the concatenation.
        var parts: [(label: String, text: String)] = []
        for (bundle, file) in spec.highlights {
            guard let text = readQuery(bundle: bundle, file: file) else { dropped.append("\(bundle)/\(file) (missing)"); continue }
            parts.append(("\(bundle)/\(file)", text))
        }
        if let q = compileDroppingFailures(parts, language: spec.language, dropped: &dropped) { queries[.highlights] = q }

        if let inj = spec.injections {
            let injParts = inj.files.compactMap { f in readQuery(bundle: inj.bundle, file: f).map { ("\(inj.bundle)/\(f)", $0) } }
            var ignored: [String] = []
            if let q = compileDroppingFailures(injParts, language: spec.language, dropped: &ignored) { queries[.injections] = q }
            dropped.append(contentsOf: ignored)
        }
        // A language with no highlight query would show plain text; report it as unavailable.
        guard queries[.highlights] != nil else { return (nil, dropped) }
        return (LanguageConfiguration(spec.language, name: language.rawValue, queries: queries), dropped)
    }

    private static func compileDroppingFailures(_ parts: [(label: String, text: String)], language: Language,
                                                dropped: inout [String]) -> Query? {
        guard !parts.isEmpty else { return nil }
        var kept = parts
        // Drop files that don't compile on their own.
        if kept.count > 1 {
            kept = kept.filter { part in
                if (try? Query(language: language, data: Data(part.text.utf8))) != nil { return true }
                dropped.append(part.label)
                return false
            }
        }
        while !kept.isEmpty {
            let data = Data(kept.map(\.text).joined(separator: "\n").utf8)
            if let q = try? Query(language: language, data: data) { return q }
            // The files compile alone but not together: drop the last (the inherited ones come last).
            dropped.append(kept.removeLast().label)
            if kept.count == 1, let only = kept.first {
                if let q = try? Query(language: language, data: Data(only.text.utf8)) { return q }
                dropped.append(only.label)
                return nil
            }
        }
        if parts.count == 1 { dropped.append(parts[0].label) }
        return nil
    }

    // MARK: Bundle lookup

    private final class BundleAnchor {}

    /// Directories that may hold the `*.bundle` resource bundles.
    private static let searchDirectories: [URL] = {
        var dirs: [URL] = []
        func add(_ url: URL?) { if let url, !dirs.contains(url) { dirs.append(url) } }
        add(Bundle.main.resourceURL)
        add(Bundle.main.bundleURL)
        let anchor = Bundle(for: BundleAnchor.self)
        add(anchor.resourceURL)
        add(anchor.bundleURL.deletingLastPathComponent())
        for b in Bundle.allBundles + Bundle.allFrameworks {
            add(b.resourceURL)
            add(b.bundleURL.deletingLastPathComponent())
        }
        return dirs
    }()

    private static func readQuery(bundle name: String, file: String) -> String? {
        let fm = FileManager.default
        // The vendored grammars (Packages/TreeSitterGrammars) are named TreeSitterGrammars_<Target>.
        let target = name.split(separator: "_").last.map(String.init) ?? name
        for dir in searchDirectories {
          for bundleName in [name, "TreeSitterGrammars_\(target)"] {
            let root = dir.appendingPathComponent("\(bundleName).bundle", isDirectory: true)
            for sub in ["queries", "Contents/Resources/queries"] {
                let url = root.appendingPathComponent(sub, isDirectory: true).appendingPathComponent(file)
                if fm.isReadableFile(atPath: url.path), let text = try? String(contentsOf: url, encoding: .utf8) {
                    return text
                }
            }
          }
        }
        return nil
    }
}
