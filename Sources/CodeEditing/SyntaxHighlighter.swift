import Foundation
import SwiftTreeSitter
import SwiftTreeSitterLayer

/// Parses text with a tree-sitter grammar and returns coloured spans (UTF-16 / NSString ranges).
public actor SyntaxHighlighter {
    private let language: CodeLanguage
    private var text = ""
    private var layer: LanguageLayer?
    private var all: [HighlightSpan] = []
    private var loaded = false

    public init(language: CodeLanguage) {
        self.language = language
    }

    /// Replaces the text and reparses (full parse; incremental is a later optimisation).
    public func update(text: String) {
        self.text = text
        loaded = false
        all = []
        guard let config = GrammarRegistry.configuration(for: language) else { layer = nil; return }
        let configuration = LanguageLayer.Configuration(maximumLanguageDepth: 4) { name in
            GrammarRegistry.language(forInjection: name).flatMap { GrammarRegistry.configuration(for: $0) }
        }
        guard let layer = try? LanguageLayer(languageConfig: config, configuration: configuration) else { self.layer = nil; return }
        layer.replaceContent(with: text)
        self.layer = layer
        let length = (text as NSString).length
        if length > 0, let named = try? layer.highlights(in: NSRange(location: 0, length: length), provider: text.predicateTextProvider) {
            all = Self.spans(from: named)
        }
        loaded = true
    }

    /// Spans that intersect `range`, sorted by location, then by length descending (so applying them in order lets an
    /// inner, more specific capture override an outer one). Injected languages are included; captures with no token
    /// are skipped.
    public func spans(in range: NSRange) -> [HighlightSpan] {
        guard loaded else { return [] }
        return all.filter { span in
            if range.length == 0 { return false }
            return NSIntersectionRange(span.range, range).length > 0
        }
    }

    /// One-shot, for chat code blocks.
    public static func highlight(_ text: String, language: CodeLanguage) async -> [HighlightSpan] {
        let highlighter = SyntaxHighlighter(language: language)
        await highlighter.update(text: text)
        return await highlighter.spans(in: NSRange(location: 0, length: (text as NSString).length))
    }

    private static func spans(from named: [NamedRange]) -> [HighlightSpan] {
        var result: [(offset: Int, span: HighlightSpan)] = []
        for (i, item) in named.enumerated() {
            guard let token = SyntaxCaptures.token(for: item.name), item.range.length > 0 else { continue }
            result.append((i, HighlightSpan(range: item.range, token: token)))
        }
        // Location ascending, length descending; equal ranges keep the query's order (a later capture wins).
        result.sort {
            if $0.span.range.location != $1.span.range.location { return $0.span.range.location < $1.span.range.location }
            if $0.span.range.length != $1.span.range.length { return $0.span.range.length > $1.span.range.length }
            return $0.offset < $1.offset
        }
        return result.map(\.span)
    }
}
