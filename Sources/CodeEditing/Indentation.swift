import Foundation

public enum Indentation: Equatable, Sendable {
    case tabs
    case spaces(Int)

    /// "\t" or n spaces.
    public var unit: String {
        switch self {
        case .tabs: return "\t"
        case .spaces(let n): return String(repeating: " ", count: n)
        }
    }

    /// Looks at up to the first 2000 lines. More tab-indented than space-indented lines → tabs. Otherwise the most
    /// common increase in width between consecutive non-blank lines (an unindented line counts as width 0) among
    /// 2, 4 and 8; ties and no data → 4.
    public static func detect(_ text: String) -> Indentation {
        var tabLines = 0, spaceLines = 0
        var votes: [Int: Int] = [2: 0, 4: 0, 8: 0]
        var previous = 0
        for line in text.split(separator: "\n", maxSplits: 2000, omittingEmptySubsequences: false).prefix(2000) {
            guard let first = line.first else { continue }
            if first == "\t" {
                tabLines += 1
                previous = 0
            } else if first == " " {
                let width = line.prefix(while: { $0 == " " }).count
                if width == line.count || line.dropFirst(width).allSatisfy({ $0 == "\r" }) { continue }  // blank
                spaceLines += 1
                if width > previous, votes[width - previous] != nil { votes[width - previous]! += 1 }
                previous = width
            } else if first == "\r" {
                continue
            } else {
                previous = 0
            }
        }
        if tabLines > spaceLines { return .tabs }
        var best = 4
        for candidate in [4, 2, 8] where votes[candidate]! > votes[best]! { best = candidate }
        return .spaces(best)
    }
}
