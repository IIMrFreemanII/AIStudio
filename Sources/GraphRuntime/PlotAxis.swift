import Foundation

/// One axis of an XY plot: a literal node input and the values it takes, one per row or column.
public struct PlotAxis: Codable, Hashable, Sendable {
    public var node: String
    public var input: String
    public var values: [JSONValue]
    /// Short labels drawn on the grid, one per value.
    public var labels: [String]

    public init(node: String, input: String, values: [JSONValue], labels: [String]) {
        self.node = node
        self.input = input
        self.values = values
        self.labels = labels
    }

    public enum ParseError: Error, CustomStringConvertible, Equatable {
        case empty
        case badValue(String)
        case tooMany(Int)
        case searchTermMissing(String)

        public var description: String {
            switch self {
            case .empty: return "enter at least one value"
            case .badValue(let v): return "can't read \"\(v)\""
            case .tooMany(let n): return "\(n) values is too many (at most \(PlotAxis.maxValues) per axis)"
            case .searchTermMissing(let t): return "\"\(t)\" doesn't appear in the current text"
            }
        }
    }

    public static let maxValues = 16

    /// Parses the values typed for an input of type `type`.
    ///
    /// - Numbers: a comma list; ranges `1-4` (step 1) and `4-12:4` / `0.2-0.6:0.2` (with a step).
    /// - Text: options separated by `|` (or commas when there is no `|`). With `searchReplace`, the first option
    ///   must appear in `current`; each option replaces it there ("prompt S/R").
    /// - Booleans: `true, false`.
    public static func parse(_ text: String, type: SocketType, node: String, input: String,
                             current: JSONValue? = nil, searchReplace: Bool = false) throws -> PlotAxis {
        let raw = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { throw ParseError.empty }
        var values: [JSONValue] = []
        var labels: [String] = []
        switch type {
        case .int, .float:
            for item in raw.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces) }) where !item.isEmpty {
                let nums = try numbers(item, integer: type == .int)
                values += nums.map { type == .int ? .int(Int($0)) : .double($0) }
                if values.count > maxValues { throw ParseError.tooMany(values.count) }
            }
            labels = values.map { "\(input) \(format($0))" }
        case .bool:
            for item in raw.split(separator: ",").map({ $0.trimmingCharacters(in: .whitespaces).lowercased() }) {
                guard item == "true" || item == "false" else { throw ParseError.badValue(item) }
                values.append(.bool(item == "true"))
            }
            labels = values.map { "\(input) \($0)" }
        default:
            let sep: Character = raw.contains("|") ? "|" : ","
            let items = raw.split(separator: sep).map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
            if searchReplace {
                let base = current?.description ?? ""
                guard let term = items.first, base.contains(term) else {
                    throw ParseError.searchTermMissing(items.first ?? raw)
                }
                values = items.map { .string(base.replacingOccurrences(of: term, with: $0)) }
                labels = items
            } else {
                values = items.map { .string($0) }
                labels = items.map { $0.count > 40 ? String($0.prefix(39)) + "…" : $0 }
            }
        }
        guard !values.isEmpty else { throw ParseError.empty }
        guard values.count <= maxValues else { throw ParseError.tooMany(values.count) }
        return PlotAxis(node: node, input: input, values: values, labels: labels)
    }

    static func numbers(_ item: String, integer: Bool) throws -> [Double] {
        // "a-b" or "a-b:step"; a leading "-" belongs to the first number.
        let pattern = #"^(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)(?:\s*:\s*(\d+(?:\.\d+)?))?$"#
        let ns = item as NSString
        if let match = try! NSRegularExpression(pattern: pattern)
            .firstMatch(in: item, range: NSRange(location: 0, length: ns.length)) {
            let a = Double(ns.substring(with: match.range(at: 1)))!
            let b = Double(ns.substring(with: match.range(at: 2)))!
            let step = match.range(at: 3).location != NSNotFound ? Double(ns.substring(with: match.range(at: 3)))! : 1
            guard step > 0 else { throw ParseError.badValue(item) }
            let count = Int(((abs(b - a)) / step + 1e-9).rounded(.down)) + 1
            guard count <= maxValues else { throw ParseError.tooMany(count) }
            let dir: Double = b >= a ? 1 : -1
            return (0..<count).map { ((a + dir * step * Double($0)) * 1e6).rounded() / 1e6 }
        }
        guard let v = Double(item), !integer || v == v.rounded() else { throw ParseError.badValue(item) }
        return [v]
    }

    static func format(_ v: JSONValue) -> String {
        if case .double(let d) = v { return String(format: "%g", d) }
        return v.description
    }
}

/// An XY plot: one or two axes over a base graph. Cells are row-major (`y` rows of `x` columns).
public struct PlotSpec: Codable, Hashable, Sendable {
    public var x: PlotAxis
    public var y: PlotAxis?

    public init(x: PlotAxis, y: PlotAxis?) {
        self.x = x
        self.y = y
    }

    public var columns: Int { x.values.count }
    public var rows: Int { y?.values.count ?? 1 }
    public var cellCount: Int { columns * rows }

    /// The base graph with cell (`column`, `row`)'s values set. Setting a Sampler's seed sets every Sampler's
    /// seed, so a hi-res pass follows the base one.
    public func graph(for base: Graph, column: Int, row: Int) -> Graph {
        var g = base
        func set(_ axis: PlotAxis, _ i: Int) {
            let v = axis.values[i]
            if axis.input == "seed", g[axis.node]?.type == "Sampler" {
                for n in g.nodes where n.type == "Sampler" { try? g.setValue(v, node: n.id, input: "seed") }
            } else {
                try? g.setValue(v, node: axis.node, input: axis.input)
            }
        }
        set(x, column)
        if let y { set(y, row) }
        return g
    }

    public func label(column: Int, row: Int) -> String {
        [x.labels[column], y.map { $0.labels[row] }].compactMap { $0 }.joined(separator: " · ")
    }
}
