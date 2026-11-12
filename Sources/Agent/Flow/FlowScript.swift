import Foundation
import JavaScriptCore

/// Reading a flow script before it runs: its meta (for the approval card) and the body the runtime executes.
public enum FlowScript {
    public struct ScriptError: Error, LocalizedError, Equatable {
        public let message: String
        public init(_ message: String) { self.message = message }
        public var errorDescription: String? { message }
    }

    /// The `export const meta = {…}` literal, evaluated alone (it must be a pure literal: no variables or calls).
    public static func meta(_ script: String) throws -> FlowMeta {
        guard let (literal, _) = metaLiteral(script) else {
            throw ScriptError("The script must start with `export const meta = { name, description, phases }`.")
        }
        guard let ctx = JSContext() else { throw ScriptError("JavaScript isn't available.") }
        var thrown: String?
        ctx.exceptionHandler = { _, e in thrown = e?.toString() }
        // Evaluated alone: a variable or a call to the script's own functions throws (not defined here).
        let value = ctx.evaluateScript("'use strict'; (function(){ return (\(literal)); })()")
        if let thrown { throw ScriptError("meta isn't a plain literal: \(thrown)") }
        guard let obj = value?.toDictionary() else { throw ScriptError("meta must be an object literal.") }
        guard let name = obj["name"] as? String, !name.isEmpty else { throw ScriptError("meta.name is missing.") }
        let description = obj["description"] as? String ?? ""
        let phases: [String] = (obj["phases"] as? [Any] ?? []).compactMap { p in
            if let s = p as? String { return s }
            return (p as? [String: Any])?["title"] as? String
        }
        return FlowMeta(name: name, description: description, phases: phases)
    }

    /// The script without its `export const meta = …` statement (still declares `meta` as a plain const).
    public static func body(_ script: String) -> String {
        guard let (_, range) = metaLiteral(script) else { return script }
        var s = script
        s.replaceSubrange(range, with: "const meta = " + String(script[range]).drop { $0 != "{" })
        return s
    }

    /// How many `mode: 'edit'` agents the script asks for (for the approval card; a lower bound with loops).
    public static func editAgentMentions(_ script: String) -> Int {
        let re = try! NSRegularExpression(pattern: #"mode\s*:\s*['"`]edit['"`]"#)
        return re.numberOfMatches(in: script, range: NSRange(script.startIndex..., in: script))
            + script.components(separatedBy: "implementAndCheck(").count - 1
    }

    /// The meta object literal and the range of the whole `export const meta = {…}` (through the closing brace).
    static func metaLiteral(_ script: String) -> (String, Range<String.Index>)? {
        guard let start = script.range(of: #"export\s+const\s+meta\s*=\s*"#, options: .regularExpression),
              script[start.upperBound...].first == "{" else { return nil }
        var depth = 0
        var quote: Character?
        var escaped = false
        var i = start.upperBound
        while i < script.endIndex {
            let c = script[i]
            if let q = quote {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == q { quote = nil }
            } else if c == "'" || c == "\"" || c == "`" {
                quote = c
            } else if c == "{" {
                depth += 1
            } else if c == "}" {
                depth -= 1
                if depth == 0 {
                    let end = script.index(after: i)
                    return (String(script[start.upperBound..<end]), start.lowerBound..<end)
                }
            }
            i = script.index(after: i)
        }
        return nil
    }
}
