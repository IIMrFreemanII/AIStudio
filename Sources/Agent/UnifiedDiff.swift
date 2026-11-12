import Foundation

/// A line diff for the edit cards (longest common subsequence; big files fall back to the changed middle).
public enum UnifiedDiff {
    public enum Line: Equatable, Sendable {
        case same(String)
        case removed(String)
        case added(String)
    }

    public static func lines(_ old: String, _ new: String) -> [Line] {
        let a = old.components(separatedBy: "\n"), b = new.components(separatedBy: "\n")
        // Common head and tail first, so the LCS only covers the changed middle.
        var head = 0
        while head < a.count, head < b.count, a[head] == b[head] { head += 1 }
        var tail = 0
        while tail < a.count - head, tail < b.count - head, a[a.count - 1 - tail] == b[b.count - 1 - tail] { tail += 1 }
        let am = Array(a[head..<(a.count - tail)]), bm = Array(b[head..<(b.count - tail)])
        var out = a[..<head].map(Line.same)
        if am.count * bm.count > 4_000_000 {
            out += am.map(Line.removed) + bm.map(Line.added)
        } else {
            var dp = Array(repeating: Array(repeating: 0, count: bm.count + 1), count: am.count + 1)
            for i in stride(from: am.count - 1, through: 0, by: -1) {
                for j in stride(from: bm.count - 1, through: 0, by: -1) {
                    dp[i][j] = am[i] == bm[j] ? dp[i + 1][j + 1] + 1 : max(dp[i + 1][j], dp[i][j + 1])
                }
            }
            var i = 0, j = 0
            while i < am.count || j < bm.count {
                if i < am.count, j < bm.count, am[i] == bm[j] { out.append(.same(am[i])); i += 1; j += 1 }
                else if i < am.count, j == bm.count || dp[i + 1][j] >= dp[i][j + 1] { out.append(.removed(am[i])); i += 1 }
                else { out.append(.added(bm[j])); j += 1 }
            }
        }
        out += a[(a.count - tail)...].map(Line.same)
        return out
    }

    /// Unified-diff text with `context` unchanged lines around each change, each hunk headed
    /// "@@ -oldStart,oldCount +newStart,newCount @@" (1-based line numbers).
    public static func text(_ old: String, _ new: String, path: String = "", context: Int = 3) -> String {
        let ls = lines(old, new)
        let changed = ls.indices.filter { if case .same = ls[$0] { return false }; return true }
        guard !changed.isEmpty else { return "" }
        // Each entry's line number in the old and the new text.
        var numbers: [(old: Int, new: Int)] = []
        var o = 1, n = 1
        for l in ls {
            numbers.append((o, n))
            switch l {
            case .same: o += 1; n += 1
            case .removed: o += 1
            case .added: n += 1
            }
        }
        var keep = Set<Int>()
        for c in changed { for k in max(0, c - context)...min(ls.count - 1, c + context) { keep.insert(k) } }
        // Runs of consecutive kept entries are the hunks.
        var hunks: [[Int]] = []
        for k in keep.sorted() {
            if let last = hunks.last?.last, last == k - 1 { hunks[hunks.count - 1].append(k) } else { hunks.append([k]) }
        }
        var out = path.isEmpty ? [] : ["--- \(path)", "+++ \(path)"]
        for hunk in hunks {
            let oldCount = hunk.filter { if case .added = ls[$0] { return false }; return true }.count
            let newCount = hunk.filter { if case .removed = ls[$0] { return false }; return true }.count
            // An empty side starts at the line before it (diff's convention).
            let first = numbers[hunk[0]]
            out.append("@@ -\(oldCount == 0 ? first.old - 1 : first.old),\(oldCount) "
                       + "+\(newCount == 0 ? first.new - 1 : first.new),\(newCount) @@")
            for k in hunk {
                switch ls[k] {
                case .same(let s): out.append(" " + s)
                case .removed(let s): out.append("-" + s)
                case .added(let s): out.append("+" + s)
                }
            }
        }
        return out.joined(separator: "\n")
    }

    /// (+added, −removed) line counts.
    public static func counts(_ old: String, _ new: String) -> (added: Int, removed: Int) {
        lines(old, new).reduce(into: (0, 0)) { acc, l in
            if case .added = l { acc.0 += 1 } else if case .removed = l { acc.1 += 1 }
        }
    }
}

/// A unified diff split into rows with line numbers (new-file numbers; old ones for removed lines), read from the
/// "@@ -a,b +c,d @@" hunk headers. Diffs saved before the headers had numbers ("@@" alone) get no numbers.
public enum DiffNumbers {
    public struct Row: Identifiable, Sendable {
        public let id: Int
        public var number: Int?
        public var sign: Character?
        public var text: String
        public var gap = false
    }

    public static func rows(_ diff: String) -> [Row] {
        var rows: [Row] = []
        var old: Int?, new: Int?
        // True at the start and after "diff --git": git's file header lines (index, mode, ---/+++ …) are skipped
        // until a hunk begins.
        var header = true
        for (i, line) in diff.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let s = String(line)
            if s.hasPrefix("diff --git") { header = true; continue }
            if header, !s.hasPrefix("@@") { continue }
            if s.hasPrefix("@@") {
                header = false
                (old, new) = starts(s)
                // No ⋯ before a first hunk that starts at the top of the file.
                if !(rows.isEmpty && (new ?? 0) <= 1 && new != nil) { rows.append(Row(id: i, text: "⋯", gap: true)) }
                continue
            }
            if s.hasPrefix("\\ ") { continue }  // "\ No newline at end of file"
            switch s.first {
            case "+":
                rows.append(Row(id: i, number: new, sign: "+", text: String(s.dropFirst())))
                new = new.map { $0 + 1 }
            case "-":
                rows.append(Row(id: i, number: old, sign: "-", text: String(s.dropFirst())))
                old = old.map { $0 + 1 }
            default:
                rows.append(Row(id: i, number: new, sign: nil, text: String(s.dropFirst())))
                old = old.map { $0 + 1 }
                new = new.map { $0 + 1 }
            }
        }
        return rows
    }

    /// The old and new start lines in "@@ -3,4 +3,5 @@" (nil for a bare "@@").
    public static func starts(_ header: String) -> (Int?, Int?) {
        var old: Int?, new: Int?
        // Only the text between the first "@@" and the next one (a trailing function name may hold anything).
        guard let open = header.range(of: "@@") else { return (nil, nil) }
        let rest = header[open.upperBound...]
        let body = rest.range(of: "@@").map { rest[..<$0.lowerBound] } ?? rest
        for part in body.split(separator: " ") {
            let number = Int(part.dropFirst().prefix { $0 != "," })
            if part.hasPrefix("-") { old = number } else if part.hasPrefix("+") { new = number }
        }
        return (old, new)
    }
}
