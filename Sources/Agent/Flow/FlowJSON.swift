import Foundation
import LLMClient

/// Pulling a JSON value out of an agent's reply, and checking it against a small JSON Schema subset.
public enum FlowJSON {
    /// The JSON value in a reply: the last ```json (or ```) fenced block, else the last balanced top-level {...} or [...].
    public static func extract(_ text: String) -> JSONValue? {
        if let re = try? NSRegularExpression(pattern: "```[A-Za-z]*[ \\t]*\\n(.*?)```", options: [.dotMatchesLineSeparators]) {
            let ns = text as NSString
            let matches = re.matches(in: text, range: NSRange(location: 0, length: ns.length))
            for m in matches.reversed() {
                if let v = parse(ns.substring(with: m.range(at: 1))) { return v }
            }
        }
        for candidate in balanced(text).reversed() {
            if let v = parse(candidate) { return v }
        }
        return nil
    }

    private static func parse(_ s: String) -> JSONValue? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard t.hasPrefix("{") || t.hasPrefix("[") || t.hasPrefix("\"") || Double(t) != nil || ["true", "false", "null"].contains(t),
              let data = t.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(JSONValue.self, from: data)
    }

    /// Top-level balanced {...} and [...] substrings, in order (string literals respected).
    private static func balanced(_ text: String) -> [String] {
        let chars = Array(text)
        var out: [String] = []
        var i = 0
        while i < chars.count {
            guard chars[i] == "{" || chars[i] == "[" else { i += 1; continue }
            var depth = 0
            var inString = false
            var escaped = false
            var j = i
            var end: Int?
            while j < chars.count {
                let c = chars[j]
                if inString {
                    if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
                } else if c == "\"" {
                    inString = true
                } else if c == "{" || c == "[" {
                    depth += 1
                } else if c == "}" || c == "]" {
                    depth -= 1
                    if depth == 0 { end = j; break }
                }
                j += 1
            }
            if let end {
                out.append(String(chars[i...end]))
                i = end + 1
            } else {
                i += 1
            }
        }
        return out
    }

    /// nil when `value` matches `schema`, else a short problem ("findings[0].title: expected string").
    /// Supports: type (object, array, string, number, integer, boolean, null; or an array of types), properties,
    /// required, items, enum. Unknown keywords are ignored. Extra properties are allowed.
    public static func validate(_ value: JSONValue, schema: JSONValue) -> String? {
        check(value, schema, path: "")
    }

    private static func typeName(_ v: JSONValue) -> String {
        switch v {
        case .null: return "null"
        case .bool: return "boolean"
        case .number(let n): return n == n.rounded() && abs(n) < 1e15 ? "integer" : "number"
        case .string: return "string"
        case .array: return "array"
        case .object: return "object"
        }
    }

    private static func matches(_ v: JSONValue, type: String) -> Bool {
        let actual = typeName(v)
        if type == "number" { return actual == "number" || actual == "integer" }
        return actual == type
    }

    private static func check(_ value: JSONValue, _ schema: JSONValue, path: String) -> String? {
        guard case .object(let s) = schema else { return nil }
        let at = path.isEmpty ? "reply" : path
        if let t = s["type"] {
            var types: [String] = []
            if case .string(let one) = t { types = [one] }
            if case .array(let many) = t { types = many.compactMap { if case .string(let x) = $0 { return x } else { return nil } } }
            if !types.isEmpty, !types.contains(where: { matches(value, type: $0) }) {
                return "\(at): expected \(types.joined(separator: " or "))"
            }
        }
        if case .array(let options)? = s["enum"], !options.contains(value) {
            return "\(at): not one of the allowed values"
        }
        if case .object(let obj) = value {
            if case .array(let req)? = s["required"] {
                for case .string(let key) in req where obj[key] == nil {
                    return "\(path.isEmpty ? "" : path + ".")\(key): required but missing"
                }
            }
            if case .object(let props)? = s["properties"] {
                for key in props.keys.sorted() {
                    guard let sub = obj[key] else { continue }
                    if let p = check(sub, props[key]!, path: path.isEmpty ? key : "\(path).\(key)") { return p }
                }
            }
        }
        if case .array(let items) = value, let itemSchema = s["items"] {
            for (i, item) in items.enumerated() {
                if let p = check(item, itemSchema, path: "\(path)[\(i)]") { return p }
            }
        }
        return nil
    }
}
