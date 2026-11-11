import Foundation
import GraphRuntime

extension GraphOps {
    /// Operations from the model's JSON (`[{"op": "add", "type": "…", "inputs": {…}}, …]`).
    public static func parse(_ raw: [Any]) throws -> [Op] {
        try raw.enumerated().map { i, item in
            guard let o = item as? [String: Any] else { throw Failure(index: i, "each operation is a JSON object") }
            func str(_ k: String) throws -> String {
                guard let s = o[k] as? String, !s.isEmpty else { throw Failure(index: i, "\"\(k)\" is missing") }
                return s
            }
            let kind = (o["op"] as? String)?.lowercased() ?? ""
            switch kind {
            case "add":
                var inputs: [String: NodeInput] = [:]
                for (k, v) in (o["inputs"] as? [String: Any]) ?? [:] {
                    if let link = v as? [String: Any], let node = link["node"] as? String {
                        inputs[k] = .link(node: node, output: (link["output"] as? String) ?? "")
                    } else if let value = jsonValue(v) {
                        inputs[k] = .value(value)
                    } else {
                        throw Failure(index: i, "input \"\(k)\" is neither a value nor {\"node\", \"output\"}")
                    }
                }
                return .add(id: o["id"] as? String, type: try str("type"), title: o["title"] as? String, inputs: inputs)
            case "connect":
                return .connect(from: try str("from"), output: (o["output"] as? String).flatMap { $0.isEmpty ? nil : $0 },
                                to: try str("to"), input: try str("input"))
            case "disconnect":
                return .disconnect(node: try str("node"), input: try str("input"))
            case "set":
                guard let v = o["value"].flatMap(jsonValue) else { throw Failure(index: i, "\"value\" is missing") }
                return .set(node: try str("node"), input: try str("input"), value: v)
            case "remove", "delete":
                return .remove(node: try str("node"))
            case "rename":
                return .rename(node: try str("node"), title: (o["title"] as? String) ?? "")
            default:
                throw Failure(index: i, "unknown op \"\(kind)\" (add, connect, disconnect, set, remove, rename)")
            }
        }
    }

    /// A JSONSerialization value as a literal (booleans first: they're NSNumbers too).
    public static func jsonValue(_ v: Any) -> JSONValue? {
        if let n = v as? NSNumber {
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return .bool(n.boolValue) }
            let d = n.doubleValue
            if CFNumberIsFloatType(n) { return d == d.rounded() && abs(d) < 1e15 ? .int(Int(d)) : .double(d) }
            return .int(n.intValue)
        }
        if let s = v as? String { return .string(s) }
        return nil
    }
}
