import Foundation

/// The design system's syntax colours (tokens `mg-syntax-<rawValue>`).
public enum SyntaxToken: String, CaseIterable, Sendable {
    case keyword, type, function, string, number, comment, attribute, directive
    case operatorSymbol = "operator-symbol", punctuation, variable, constant
    case heading = "heading-1", code, link, markup, quote, listMarker = "list-marker", strikethrough
}

/// A coloured run in UTF-16 units (NSString ranges).
public struct HighlightSpan: Sendable, Equatable {
    public var range: NSRange
    public var token: SyntaxToken
    public init(range: NSRange, token: SyntaxToken) {
        self.range = range
        self.token = token
    }
}

public enum SyntaxCaptures {
    /// Entries map to a token; `nil` means "default text colour" and stops the prefix search there.
    private static let table: [String: SyntaxToken?] = [
        "keyword": .keyword, "conditional": .keyword, "repeat": .keyword, "exception": .keyword, "storageclass": .keyword,
        "include": .directive, "preproc": .directive, "keyword.directive": .directive, "define": .directive,
        "type": .type, "type.builtin": .type, "constructor": .type, "namespace": .type, "module": .type, "tag": .type,
        "function": .function, "function.method": .function, "function.builtin": .function, "method": .function,
        "function.call": .function, "function.macro": .function,
        "string": .string, "string.special": .string, "character": .string,
        "string.escape": .constant, "escape": .constant,
        "number": .number, "float": .number,
        "boolean": .constant, "constant": .constant, "constant.builtin": .constant,
        "comment": .comment,
        "attribute": .attribute, "decorator": .attribute, "tag.attribute": .attribute,
        "operator": .operatorSymbol,
        "punctuation": .punctuation, "punctuation.delimiter": .punctuation, "punctuation.bracket": .punctuation,
        "punctuation.special": .punctuation,
        "property": .variable, "field": .variable, "variable.member": .variable, "variable.parameter": .variable,
        "parameter": .variable,
        "variable": nil, "variable.builtin": .keyword,
        "label": .attribute,
        "markup.heading": .heading, "text.title": .heading,
        "markup.raw": .code, "text.literal": .code,
        "markup.link": .link, "markup.link.url": .link, "text.uri": .link, "text.reference": .link,
        "markup.quote": .quote,
        "markup.list": .listMarker,
        "markup.strikethrough": .strikethrough, "text.strike": .strikethrough,
        "markup": .markup, "text.emphasis": .markup, "text.strong": .markup, "markup.italic": .markup, "markup.bold": .markup,
        "spell": nil, "none": nil, "nospell": nil, "embedded": nil,
    ]

    /// The token for a tree-sitter capture name ("keyword.function" → .keyword); nil = default text colour.
    /// Looks up the full name, then drops the last ".segment" and retries (the longest prefix wins).
    public static func token(for capture: String) -> SyntaxToken? {
        if capture.hasPrefix("_") { return nil }
        var name = Substring(capture)
        while true {
            if let hit = table[String(name)] { return hit }
            guard let dot = name.lastIndex(of: ".") else { return nil }
            name = name[..<dot]
        }
    }
}
