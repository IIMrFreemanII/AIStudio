import AppKit
import CodeEditing

/// Syntax token → design-system colour. (Strikethrough also gets a line; see `SyntaxStyler`.)
enum SyntaxTheme {
    static func color(_ t: SyntaxToken) -> NSColor {
        switch t {
        case .keyword: MGSyntaxColor.keyword
        case .type: MGSyntaxColor.type
        case .function: MGSyntaxColor.function
        case .string: MGSyntaxColor.string
        case .number: MGSyntaxColor.number
        case .comment: MGSyntaxColor.comment
        case .attribute: MGSyntaxColor.attribute
        case .directive: MGSyntaxColor.directive
        case .operatorSymbol: MGSyntaxColor.operatorSymbol
        case .punctuation: MGSyntaxColor.punctuation
        case .variable: MGSyntaxColor.variable
        case .constant: MGSyntaxColor.constant
        case .heading: MGSyntaxColor.heading1
        case .code: MGSyntaxColor.code
        case .link: MGSyntaxColor.link
        case .markup: MGSyntaxColor.markup
        case .quote: MGSyntaxColor.quote
        case .listMarker: MGSyntaxColor.listMarker
        case .strikethrough: MGSyntaxColor.strikethrough
        }
    }
}
