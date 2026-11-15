import Foundation

/// Line index over a text, in UTF-16 units (NSString ranges). Line breaks are "\n", "\r\n" (one break) and a lone "\r".
/// Lines are 1-based; a text always has at least one line and a trailing break starts an empty last line
/// ("a\n" has 2 lines), matching editors' line numbers.
public struct TextLines: Sendable, Equatable {
    private let starts: [Int]
    private let ends: [Int]      // end of each line, before its break
    public let length: Int       // UTF-16 length of the text

    public init(_ text: String) {
        var starts = [0], ends: [Int] = []
        var i = 0
        let units = Array(text.utf16)
        while i < units.count {
            let c = units[i]
            if c == 10 {
                ends.append(i); starts.append(i + 1)
            } else if c == 13 {
                ends.append(i)
                if i + 1 < units.count, units[i + 1] == 10 { i += 1 }
                starts.append(i + 1)
            }
            i += 1
        }
        ends.append(units.count)
        self.starts = starts
        self.ends = ends
        self.length = units.count
    }

    public init(_ text: NSString) { self.init(text as String) }

    /// Number of lines (at least 1).
    public var count: Int { starts.count }

    /// The 1-based line containing `offset` (clamped to 0...length); an offset right after a break is on the next line.
    public func line(at offset: Int) -> Int {
        let o = min(max(offset, 0), length)
        var lo = 0, hi = starts.count - 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if starts[mid] <= o { lo = mid } else { hi = mid - 1 }
        }
        return lo + 1
    }

    /// The range of line `n` without its break; `n` is clamped to 1...count.
    public func range(ofLine n: Int) -> NSRange {
        let i = min(max(n, 1), count) - 1
        return NSRange(location: starts[i], length: ends[i] - starts[i])
    }

    /// From the start of the first line to the end of the last one, without the final break.
    public func range(ofLines lines: ClosedRange<Int>) -> NSRange {
        let first = range(ofLine: lines.lowerBound), last = range(ofLine: lines.upperBound)
        return NSRange(location: first.location, length: max(0, last.location + last.length - first.location))
    }

    /// The lines a selection covers. An empty selection is the caret's line; a selection that ends exactly at
    /// column 0 of a later line doesn't count that line.
    public func lineSpan(of selection: NSRange) -> ClosedRange<Int> {
        let first = line(at: selection.location)
        guard selection.length > 0 else { return first...first }
        let end = selection.location + selection.length
        var last = line(at: end)
        if last > first, min(max(end, 0), length) == starts[last - 1] { last -= 1 }
        return first...last
    }

    /// 1-based line and 0-based UTF-16 column of an offset (clamped).
    public func position(of offset: Int) -> (line: Int, column: Int) {
        let o = min(max(offset, 0), length)
        let l = line(at: o)
        return (l, o - starts[l - 1])
    }

    /// The offset of (line, column), with both clamped (the column to the line's length).
    public func offset(line: Int, column: Int) -> Int {
        let i = min(max(line, 1), count) - 1
        return starts[i] + min(max(column, 0), ends[i] - starts[i])
    }

    /// Keeps the same (line, column) of a selection's start and end, clamped into the new text (for a reload from disk).
    public static func remap(_ selection: NSRange, from old: TextLines, to new: TextLines) -> NSRange {
        let s = old.position(of: selection.location)
        let e = old.position(of: selection.location + selection.length)
        let start = new.offset(line: s.line, column: s.column)
        let end = max(start, new.offset(line: e.line, column: e.column))
        return NSRange(location: start, length: end - start)
    }
}
