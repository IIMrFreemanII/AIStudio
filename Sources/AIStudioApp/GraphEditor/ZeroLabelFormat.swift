import Foundation
import GraphRuntime

/// A number format that shows a word ("auto") for 0 when 0 is a sentinel, and reads that word, or an empty field,
/// back as 0. Without a label it's the base format.
///
/// With `dotIsDecimal`, a "." also reads as the decimal point where the Mac's region uses a comma (Ukraine, Germany…):
/// the base format would read "1.5" as 1 and drop the rest without a word.
struct ZeroLabelFormat<Base: ParseableFormatStyle>: ParseableFormatStyle
where Base.FormatOutput == String, Base.FormatInput: Numeric, Base.Strategy.ParseInput == String,
      Base.Strategy.ParseOutput == Base.FormatInput {
    var base: Base
    var label: String?
    var dotIsDecimal = false

    func format(_ value: Base.FormatInput) -> String {
        if let label, value == .zero { return label }
        return base.format(value)
    }

    var parseStrategy: Strategy { Strategy(base: base.parseStrategy, label: label, dotIsDecimal: dotIsDecimal) }

    struct Strategy: ParseStrategy {
        var base: Base.Strategy
        var label: String?
        var dotIsDecimal = false

        func parse(_ value: String) throws -> Base.FormatInput {
            let text = value.trimmingCharacters(in: .whitespaces)
            if let label, text.isEmpty || text.caseInsensitiveCompare(label) == .orderedSame { return .zero }
            return try base.parse(dotIsDecimal ? Self.localDecimal(text) : text)
        }

        /// "1.5" → "1,5" where the decimal separator is a comma and "." isn't the thousands separator; text that
        /// already has the local separator is left alone.
        static func localDecimal(_ text: String, locale: Locale = .current) -> String {
            guard let sep = locale.decimalSeparator, sep != ".", locale.groupingSeparator != ".",
                  text.contains("."), !text.contains(sep) else { return text }
            return text.replacingOccurrences(of: ".", with: sep)
        }
    }
}

extension InputSpec {
    /// Number formats for this input's fields: 0 shows as `zeroLabel` when it has one.
    var intFormat: ZeroLabelFormat<IntegerFormatStyle<Int>> { ZeroLabelFormat(base: .number.grouping(.never), label: zeroLabel) }
    var doubleFormat: ZeroLabelFormat<FloatingPointFormatStyle<Double>> { ZeroLabelFormat(base: .number, label: zeroLabel, dotIsDecimal: true) }
}
