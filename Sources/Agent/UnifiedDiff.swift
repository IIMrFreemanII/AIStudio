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

    /// Unified-diff text with `context` unchanged lines around each change ("@@" between distant hunks).
    public static func text(_ old: String, _ new: String, path: String = "", context: Int = 3) -> String {
        let ls = lines(old, new)
        let changed = ls.indices.filter { if case .same = ls[$0] { return false }; return true }
        guard !changed.isEmpty else { return "" }
        var keep = Set<Int>()
        for c in changed { for k in max(0, c - context)...min(ls.count - 1, c + context) { keep.insert(k) } }
        var out = path.isEmpty ? [] : ["--- \(path)", "+++ \(path)"]
        var last = -2
        for k in keep.sorted() {
            if k != last + 1 { out.append("@@") }
            switch ls[k] {
            case .same(let s): out.append(" " + s)
            case .removed(let s): out.append("-" + s)
            case .added(let s): out.append("+" + s)
            }
            last = k
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
