import Foundation

/// A sticky note on the canvas. It has no outputs and is not an output node, so it never runs.
public struct NoteNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: Graph.noteType, displayName: "Note", category: "utils",
        inputs: [InputSpec("text", .string, default: .string(""), widget: .text(multiline: true))],
        outputs: [])
    public init() {}
    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] { [:] }
}

/// A bend point for a wire. Spliced out before execution, so types are checked between its real endpoints.
public struct RerouteNode: NodeImplementation {
    public static let schema = NodeSchema(
        type: Graph.rerouteType, displayName: "Reroute", category: "utils",
        inputs: [InputSpec("in", .any, optional: true)],
        outputs: [OutputSpec("out", .any)])
    public init() {}
    public func execute(_ inputs: NodeInputs, context: ExecutionContext) async throws -> [String: Value] {
        // Not reached in normal runs (flattening removes reroutes); pass the value through for direct callers.
        guard case .object(let o)? = inputs.values["in"] else { return [:] }
        return ["out": .object(o)]
    }
}
