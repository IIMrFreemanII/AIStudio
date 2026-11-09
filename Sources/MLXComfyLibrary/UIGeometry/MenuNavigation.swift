import Foundation

/// Keyboard movement in a menu: arrows skip separators, headers and disabled items; typing jumps to an item.
public enum MenuNavigation {
    /// The next selectable index after `current` moving by `step` (+1 down, -1 up), wrapping round. With no current
    /// item, down starts at the first and up at the last.
    public static func move(from current: Int?, step: Int, selectable: [Bool]) -> Int? {
        let n = selectable.count
        guard n > 0, selectable.contains(true) else { return nil }
        var i = current ?? (step > 0 ? -1 : n)
        for _ in 0..<n {
            i = ((i + step) % n + n) % n
            if selectable[i] { return i }
        }
        return nil
    }

    /// The first selectable item whose title starts with `prefix` (ignoring case and accents), searching from the
    /// current item when the prefix is still being typed and from the one after it when it's a single new letter.
    public static func match(_ prefix: String, titles: [String], selectable: [Bool], from current: Int?) -> Int? {
        let p = fold(prefix)
        guard !p.isEmpty, titles.count == selectable.count, !titles.isEmpty else { return nil }
        let n = titles.count
        let first = current.map { p.count == 1 ? $0 + 1 : $0 } ?? 0
        for k in 0..<n {
            let i = (first + k) % n
            if selectable[i], fold(titles[i]).hasPrefix(p) { return i }
        }
        return nil
    }

    static func fold(_ s: String) -> String {
        s.folding(options: [.caseInsensitive, .diacriticInsensitive], locale: nil).trimmingCharacters(in: .whitespaces)
    }
}
