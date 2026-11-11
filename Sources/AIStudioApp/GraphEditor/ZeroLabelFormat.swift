import Foundation
import GraphRuntime

/// A number format that shows a word ("auto") for 0 when 0 is a sentinel, and reads that word, or an empty field,
/// back as 0. Without a label it's the base format.
struct ZeroLabelFormat<Base: ParseableFormatStyle>: ParseableFormatStyle
where Base.FormatOutput == String, Base.FormatInput: Numeric, Base.Strategy.ParseInput == String,
      Base.Strategy.ParseOutput == Base.FormatInput {
    var base: Base
    var label: String?

    func format(_ value: Base.FormatInput) -> String {
        if let label, value == .zero { return label }
        return base.format(value)
    }

    var parseStrategy: Strategy { Strategy(base: base.parseStrategy, label: label) }

    struct Strategy: ParseStrategy {
        var base: Base.Strategy
        var label: String?

        func parse(_ value: String) throws -> Base.FormatInput {
            let text = value.trimmingCharacters(in: .whitespaces)
            if let label, text.isEmpty || text.caseInsensitiveCompare(label) == .orderedSame { return .zero }
            return try base.parse(text)
        }
    }
}

extension InputSpec {
    /// Number formats for this input's fields: 0 shows as `zeroLabel` when it has one.
    var intFormat: ZeroLabelFormat<IntegerFormatStyle<Int>> { ZeroLabelFormat(base: .number.grouping(.never), label: zeroLabel) }
    var doubleFormat: ZeroLabelFormat<FloatingPointFormatStyle<Double>> { ZeroLabelFormat(base: .number, label: zeroLabel) }
}
