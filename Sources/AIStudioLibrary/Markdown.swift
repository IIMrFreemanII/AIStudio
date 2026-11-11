import Foundation

/// A Markdown document split into blocks for the chat view (inline styling — emphasis, code spans, links — is left
/// to `AttributedString(markdown:)` per block). Covers what models write: headings, paragraphs, fenced code (an
/// unclosed fence runs to the end, as while a reply streams), bullet and numbered lists with nesting, quotes,
/// tables and rules.
public enum MarkdownBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case code(language: String, text: String, closed: Bool)
    /// Items with their nesting depth (0 = top) and, for numbered lists, their number.
    case list(items: [MarkdownListItem])
    case quote(String)
    case table(header: [String], rows: [[String]])
    case rule
}

public struct MarkdownListItem: Equatable, Sendable {
    public var depth: Int
    public var number: Int?
    public var text: String
    /// A task list item's box: `- [ ]` / `- [x]`.
    public var checked: Bool?
}

public enum Markdown {
    public static func blocks(_ source: String) -> [MarkdownBlock] {
        let lines = source.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var out: [MarkdownBlock] = []
        var para: [String] = []
        var i = 0

        func flushPara() {
            if !para.isEmpty { out.append(.paragraph(para.joined(separator: "\n"))); para = [] }
        }

        while i < lines.count {
            let line = lines[i]
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Fenced code.
            if let fence = fenceMarker(trimmed) {
                flushPara()
                let lang = String(trimmed.dropFirst(fence.count)).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                var closed = false
                i += 1
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if t.hasPrefix(fence), t.allSatisfy({ $0 == fence.first }) { closed = true; i += 1; break }
                    body.append(lines[i])
                    i += 1
                }
                out.append(.code(language: lang, text: body.joined(separator: "\n"), closed: closed))
                continue
            }
            if trimmed.isEmpty { flushPara(); i += 1; continue }
            // Heading.
            if let h = heading(trimmed) { flushPara(); out.append(h); i += 1; continue }
            // Rule.
            if isRule(trimmed) { flushPara(); out.append(.rule); i += 1; continue }
            // Table: a row, then a separator row.
            if trimmed.contains("|"), i + 1 < lines.count, isTableSeparator(lines[i + 1]) {
                flushPara()
                let header = cells(trimmed)
                var rows: [[String]] = []
                i += 2
                while i < lines.count, lines[i].contains("|"), !lines[i].trimmingCharacters(in: .whitespaces).isEmpty {
                    var r = cells(lines[i])
                    if r.count < header.count { r += Array(repeating: "", count: header.count - r.count) }
                    rows.append(Array(r.prefix(header.count)))
                    i += 1
                }
                out.append(.table(header: header, rows: rows))
                continue
            }
            // Quote.
            if trimmed.hasPrefix(">") {
                flushPara()
                var body: [String] = []
                while i < lines.count {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    guard t.hasPrefix(">") else { break }
                    body.append(String(t.dropFirst()).trimmingCharacters(in: .whitespaces))
                    i += 1
                }
                out.append(.quote(body.joined(separator: "\n")))
                continue
            }
            // List.
            if listItem(line) != nil {
                flushPara()
                var items: [MarkdownListItem] = []
                var indents: [Int] = []
                while i < lines.count {
                    if let item = listItem(lines[i]) {
                        while let last = indents.last, item.indent < last { indents.removeLast() }
                        if indents.isEmpty || item.indent > indents[indents.count - 1] { indents.append(item.indent) }
                        items.append(MarkdownListItem(depth: max(0, indents.count - 1), number: item.number,
                                                      text: item.text, checked: item.checked))
                        i += 1
                    } else if !lines[i].trimmingCharacters(in: .whitespaces).isEmpty,
                              lines[i].hasPrefix("  ") || lines[i].hasPrefix("\t"), !items.isEmpty {
                        // A continuation line of the item above.
                        items[items.count - 1].text += "\n" + lines[i].trimmingCharacters(in: .whitespaces)
                        i += 1
                    } else {
                        break
                    }
                }
                out.append(.list(items: items))
                continue
            }
            para.append(line)
            i += 1
        }
        flushPara()
        return out
    }

    static func fenceMarker(_ t: String) -> String? {
        for f in ["```", "~~~"] where t.hasPrefix(f) {
            let ch = f.first!
            return String(t.prefix { $0 == ch })
        }
        return nil
    }

    static func heading(_ t: String) -> MarkdownBlock? {
        let hashes = t.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes), t.count > hashes, t[t.index(t.startIndex, offsetBy: hashes)] == " " else {
            return nil
        }
        var text = String(t.dropFirst(hashes)).trimmingCharacters(in: .whitespaces)
        while text.hasSuffix("#") { text.removeLast() }
        return .heading(level: hashes, text: text.trimmingCharacters(in: .whitespaces))
    }

    static func isRule(_ t: String) -> Bool {
        let s = t.replacingOccurrences(of: " ", with: "")
        guard s.count >= 3, let c = s.first, "-*_".contains(c) else { return false }
        return s.allSatisfy { $0 == c }
    }

    static func isTableSeparator(_ line: String) -> Bool {
        let t = line.trimmingCharacters(in: .whitespaces)
        guard t.contains("-"), t.contains("|") || t.hasPrefix("-") else { return false }
        return t.allSatisfy { "|-: ".contains($0) }
    }

    static func cells(_ line: String) -> [String] {
        var t = line.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|") { t.removeLast() }
        return t.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    struct RawItem { var indent: Int; var number: Int?; var text: String; var checked: Bool? }

    static func listItem(_ line: String) -> RawItem? {
        let indent = line.prefix { $0 == " " || $0 == "\t" }.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) }
        let t = line.trimmingCharacters(in: .whitespaces)
        var number: Int?
        var rest: Substring
        if let c = t.first, "-*+".contains(c), t.dropFirst().first == " " {
            rest = t.dropFirst(2)
        } else {
            let digits = t.prefix { $0.isNumber }
            guard !digits.isEmpty, digits.count <= 9 else { return nil }
            let after = t.dropFirst(digits.count)
            guard let d = after.first, d == "." || d == ")", after.dropFirst().first == " " else { return nil }
            number = Int(digits)
            rest = after.dropFirst(2)
        }
        var checked: Bool?
        if rest.hasPrefix("[ ] ") { checked = false; rest = rest.dropFirst(4) }
        else if rest.hasPrefix("[x] ") || rest.hasPrefix("[X] ") { checked = true; rest = rest.dropFirst(4) }
        return RawItem(indent: indent, number: number, text: String(rest), checked: checked)
    }
}
