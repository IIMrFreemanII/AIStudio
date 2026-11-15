import Foundation

/// Three-way merge by lines. Each side's changes against `base` become hunks (a base line range replaced by new
/// lines); hunks that don't overlap or touch are combined, identical ones are applied once, anything else is a
/// conflict (nil). Two non-empty replacements that merely sit next to each other are not a conflict; an insertion
/// touching another hunk is.
public enum LineMerge {
    public static func merge(base: String, mine: String, theirs: String) -> String? {
        if mine == theirs { return mine }
        if mine == base { return theirs }
        if theirs == base { return mine }
        let b = lines(base)
        let a = hunks(base: b, other: lines(mine))
        let t = hunks(base: b, other: lines(theirs))
        var out: [String] = []
        var pos = 0, i = 0, j = 0
        func apply(_ h: Hunk) {
            out.append(contentsOf: b[pos..<h.lo])
            out.append(contentsOf: h.lines)
            pos = h.hi
        }
        while i < a.count || j < t.count {
            if i < a.count, j < t.count {
                let x = a[i], y = t[j]
                if conflicts(x, y) {
                    guard x == y else { return nil }
                    apply(x); i += 1; j += 1
                } else if x.lo < y.lo {
                    apply(x); i += 1
                } else {
                    apply(y); j += 1
                }
            } else if i < a.count {
                apply(a[i]); i += 1
            } else {
                apply(t[j]); j += 1
            }
        }
        out.append(contentsOf: b[pos...])
        return out.joined()
    }

    private struct Hunk: Equatable {
        var lo: Int
        var hi: Int
        var lines: [String]
        var isEmpty: Bool { lo == hi }
    }

    private static func conflicts(_ x: Hunk, _ y: Hunk) -> Bool {
        if x.lo < y.hi && y.lo < x.hi { return true }
        if x.isEmpty || y.isEmpty { return x.lo <= y.hi && y.lo <= x.hi }
        return false
    }

    /// Lines with their endings kept, so a missing final newline survives.
    private static func lines(_ text: String) -> [String] {
        var result: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ch == "\n" || ch == "\r\n" { result.append(current); current = "" }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    private static func hunks(base: [String], other: [String]) -> [Hunk] {
        let diff = other.difference(from: base)
        var removed = Set<Int>(), inserted = Set<Int>()
        for change in diff {
            switch change {
            case .remove(let offset, _, _): removed.insert(offset)
            case .insert(let offset, _, _): inserted.insert(offset)
            }
        }
        var result: [Hunk] = []
        var i = 0, j = 0
        while i < base.count || j < other.count {
            if i < base.count, j < other.count, !removed.contains(i), !inserted.contains(j) {
                i += 1; j += 1
                continue
            }
            let (si, sj) = (i, j)
            while i < base.count, removed.contains(i) { i += 1 }
            while j < other.count, inserted.contains(j) { j += 1 }
            if i == si && j == sj { break }  // unreachable for a valid diff; avoids looping forever
            result.append(Hunk(lo: si, hi: i, lines: Array(other[sj..<j])))
        }
        return result
    }
}
