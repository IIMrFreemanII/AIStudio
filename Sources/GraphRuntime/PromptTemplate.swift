import Foundation

/// Prompt templating applied when a job is queued, so the saved graph always holds the concrete prompt.
///
/// - `{red|green|blue}` picks one option; options can nest (`{a {big|small} dog|a cat}`).
///   Braces without a `|` are left as written.
/// - `__name__` picks a random line from the wildcard list `name` (lines may use choices and other wildcards).
/// - Picks come from a generator seeded by the job's seed, so the same seed gives the same prompt.
public enum PromptTemplate {
    /// Whether `text` uses choices or wildcards.
    public static func isTemplate(_ text: String) -> Bool {
        text.range(of: #"__[A-Za-z0-9_\-/]+__"#, options: .regularExpression) != nil || hasChoice(text)
    }

    public static func expand(_ text: String, seed: UInt64, wildcards: (String) -> [String]? = { _ in nil })
        -> String
    {
        var rng = SplitMix64(seed: seed)
        return expand(text, rng: &rng, wildcards: wildcards, depth: 0)
    }

    static func expand(_ text: String, rng: inout SplitMix64, wildcards: (String) -> [String]?, depth: Int)
        -> String
    {
        guard depth < 12 else { return text }
        let chars = Array(text)
        var out = ""
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if c == "{", let close = matchingBrace(chars, from: i) {
                let inner = String(chars[(i + 1)..<close])
                let options = splitTopLevel(inner)
                if options.count > 1 {
                    let pick = options[Int(rng.next() % UInt64(options.count))]
                    out += expand(pick, rng: &rng, wildcards: wildcards, depth: depth + 1)
                } else {
                    out += "{" + expand(inner, rng: &rng, wildcards: wildcards, depth: depth + 1) + "}"
                }
                i = close + 1
            } else if c == "_", i + 1 < chars.count, chars[i + 1] == "_",
                      let (name, end) = wildcardName(chars, from: i),
                      let lines = wildcards(name), !lines.isEmpty {
                let pick = lines[Int(rng.next() % UInt64(lines.count))]
                out += expand(pick, rng: &rng, wildcards: wildcards, depth: depth + 1)
                i = end
            } else {
                out.append(c)
                i += 1
            }
        }
        return out
    }

    static func hasChoice(_ text: String) -> Bool {
        let chars = Array(text)
        for (i, c) in chars.enumerated() where c == "{" {
            if let close = matchingBrace(chars, from: i), splitTopLevel(String(chars[(i + 1)..<close])).count > 1 {
                return true
            }
        }
        return false
    }

    static func matchingBrace(_ chars: [Character], from start: Int) -> Int? {
        var depth = 0
        for j in start..<chars.count {
            if chars[j] == "{" { depth += 1 }
            if chars[j] == "}" { depth -= 1; if depth == 0 { return j } }
        }
        return nil
    }

    static func splitTopLevel(_ s: String) -> [String] {
        var parts: [String] = []
        var cur = ""
        var depth = 0
        for c in s {
            if c == "{" { depth += 1 }
            if c == "}" { depth -= 1 }
            if c == "|" && depth == 0 { parts.append(cur); cur = "" } else { cur.append(c) }
        }
        parts.append(cur)
        return parts
    }

    /// `__name__` starting at `start`: the name and the index after the closing `__`.
    static func wildcardName(_ chars: [Character], from start: Int) -> (String, Int)? {
        var j = start + 2
        var name = ""
        while j + 1 < chars.count {
            if chars[j] == "_" && chars[j + 1] == "_" {
                return name.isEmpty ? nil : (name, j + 2)
            }
            let c = chars[j]
            guard c.isLetter || c.isNumber || c == "-" || c == "/" || c == "_" else { return nil }
            name.append(c)
            j += 1
        }
        return nil
    }

    /// Reads wildcard lists from `directory/<name>.txt` (one option per line, `#` comments and blanks skipped).
    public static func wildcardFiles(in directory: URL) -> (String) -> [String]? {
        { name in
            let url = directory.appendingPathComponent(name).appendingPathExtension("txt")
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
            return text.split(whereSeparator: \.isNewline)
                .map { $0.trimmingCharacters(in: .whitespaces) }
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        }
    }
}

/// A named prompt style: a template containing `{prompt}`.
public struct PromptStyle: Codable, Hashable, Identifiable, Sendable {
    public var name: String
    public var template: String
    public var id: String { name }

    public init(name: String, template: String) {
        self.name = name
        self.template = template
    }

    /// The prompt wrapped in this style (appended after a comma when the template has no `{prompt}`).
    public func apply(to prompt: String) -> String {
        let p = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if template.contains("{prompt}") { return template.replacingOccurrences(of: "{prompt}", with: p) }
        return p.isEmpty ? template : p + ", " + template
    }

    public static let builtIn: [PromptStyle] = [
        PromptStyle(name: "Photograph", template: "{prompt}, professional photograph, natural light, sharp focus, 35mm"),
        PromptStyle(name: "Cinematic", template: "cinematic film still of {prompt}, dramatic lighting, shallow depth of field, anamorphic, color graded"),
        PromptStyle(name: "Watercolor", template: "watercolor painting of {prompt}, soft washes, paper texture, loose brushwork"),
        PromptStyle(name: "Oil Painting", template: "oil painting of {prompt}, visible brushstrokes, rich colors, classical composition"),
        PromptStyle(name: "Anime", template: "anime illustration of {prompt}, clean line art, cel shading, vibrant colors"),
        PromptStyle(name: "3D Render", template: "3D render of {prompt}, soft global illumination, physically based materials, studio lighting"),
        PromptStyle(name: "Line Art", template: "minimal black ink line art of {prompt}, white background, clean strokes"),
    ]
}

/// Small, fast, seedable generator (SplitMix64) so expansions are reproducible across runs and platforms.
public struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    public init(seed: UInt64) { state = seed }
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
