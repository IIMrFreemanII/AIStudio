import Foundation
import GraphRuntime

/// A graph a movie renders with, and the named inputs it fills per shot ("prompt" → node 3's text): the stage's
/// default, or the user's edited copy in the movie's `templates/` folder.
public struct MovieTemplate: Codable, Sendable, Equatable {
    public struct Slot: Codable, Sendable, Equatable {
        public var node: String
        public var input: String
        public init(node: String, input: String) {
            self.node = node
            self.input = input
        }
    }

    public var name: String
    public var graph: Graph
    public var slots: [String: Slot]

    public init(name: String, graph: Graph, slots: [String: Slot]) {
        self.name = name
        self.graph = graph
        self.slots = slots
    }

    public enum BindError: Error, CustomStringConvertible, Equatable {
        case unknownSlot(String)
        case missingNode(slot: String, node: String)

        public var description: String {
            switch self {
            case .unknownSlot(let s): "The template has no “\(s)” input"
            case .missingNode(let s, let n): "The template's “\(s)” input points at node \(n), which isn't in the graph"
            }
        }
    }

    /// Slots whose node is gone from the graph (the user deleted it while editing the template).
    public var brokenSlots: [String] {
        slots.filter { graph[$0.value.node] == nil && !$0.value.node.contains("/") }.map(\.key).sorted()
    }

    /// The graph with `values` set into their slots. Values for slots the template doesn't have are an error unless
    /// listed in `optional` (a template may leave out, say, the negative prompt).
    public func bind(_ values: [String: JSONValue], optional: Set<String> = []) throws -> Graph {
        var g = graph
        for key in values.keys.sorted() {
            guard let slot = slots[key] else {
                if optional.contains(key) { continue }
                throw BindError.unknownSlot(key)
            }
            guard g[slot.node] != nil || slot.node.contains("/") else {
                throw BindError.missingNode(slot: key, node: slot.node)
            }
            try g.setValue(values[key]!, node: slot.node, input: slot.input)
        }
        return g
    }

    public static func load(_ url: URL) throws -> MovieTemplate {
        try JSONDecoder().decode(MovieTemplate.self, from: Data(contentsOf: url))
    }

    public func save(_ url: URL) throws {
        let e = JSONEncoder()
        e.outputFormatting = [.prettyPrinted, .sortedKeys]
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try e.encode(self).write(to: url, options: .atomic)
    }
}
