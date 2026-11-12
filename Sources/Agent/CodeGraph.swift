import CryptoKit
import Foundation
import LLMClient

/// Builds and caches code graphs of projects with the `graphify` CLI (local AST parsing, no LLM).
public actor ProjectGraphs {
    public static let shared = ProjectGraphs()

    private let cacheRoot: URL
    private let explicitBinary: String?
    private var binaryLooked = false
    private var binaryPath: String?
    private var stale: Set<String> = []
    private var inflight: [String: Task<URL, Error>] = [:]

    public init(cacheRoot: URL? = nil, binary: String? = nil) {
        self.cacheRoot = cacheRoot ?? FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("AIStudio/code-graphs")
        self.explicitBinary = binary
    }

    /// The graphify executable: the injected path, else the first one on the login shell's PATH (looked up once, a
    /// miss too, until `resetBinary`).
    public func binary() async -> String? {
        if let explicitBinary { return explicitBinary }
        if binaryLooked { return binaryPath }
        binaryPath = Self.findBinary()
        binaryLooked = true
        return binaryPath
    }

    /// graphify on the login shell's PATH, without running a shell (cheap enough to call when building a chat's tools).
    public static func findBinary() -> String? {
        ShellRunner.loginPath.split(separator: ":").lazy
            .map { URL(fileURLWithPath: String($0)).appendingPathComponent("graphify").path }
            .first { FileManager.default.isExecutableFile(atPath: $0) }
    }

    private var hubsCache: (graph: String, modified: Date, line: String)?

    /// "Main hubs: AppModel, ChatDocument, …" for the system prompt, from an already-built graph ("" when there is none
    /// yet; never builds). Cached until graph.json changes.
    public func hubs(root: URL) async -> String {
        guard let g = existingGraph(root: root), let bin = await binary() else { return "" }
        let modified = (try? g.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        if let c = hubsCache, c.graph == g.path, c.modified == modified { return c.line }
        let r = await ShellRunner().run("\(shellQuote(bin)) god-nodes --top 8 --graph \(shellQuote(g.path))", cwd: root, timeout: 15)
        let line = r.status == 0 ? GraphOutput.hubsLine(r.output) : ""
        hubsCache = (g.path, modified, line)
        return line
    }

    public func resetBinary() {
        binaryLooked = false
        binaryPath = nil
    }

    public func markStale(root: URL) { stale.insert(root.standardizedFileURL.path) }

    /// Ensures a current graph exists and returns its graph.json.
    public func graphFile(root: URL) async throws -> URL {
        let key = root.standardizedFileURL.path
        if let running = inflight[key] { return try await running.value }
        let own = Self.ownGraph(root)
        let cached = cachedGraph(root)
        let graph = own ?? cached
        if FileManager.default.fileExists(atPath: graph.path), !stale.contains(key) { return graph }
        let out = outDir(root)
        if own == nil { try? FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true) }
        stale.remove(key)  // a markStale during the build stays set
        // Registered before any await, so concurrent callers for this root join this build instead of starting another.
        let task = Task {
            guard let bin = await self.binary() else { throw ToolError("graphify isn't installed (uv tool install graphifyy)") }
            let command = own != nil
                ? "\(shellQuote(bin)) update \(shellQuote(root.path))"
                : "\(shellQuote(bin)) extract \(shellQuote(root.path)) --code-only --out \(shellQuote(out.path))"
            return try await Self.build(command: command, root: root, graph: graph)
        }
        inflight[key] = task
        do {
            let url = try await task.value
            inflight[key] = nil
            return url
        } catch {
            inflight[key] = nil
            stale.insert(key)
            throw error
        }
    }

    public func warm(root: URL) async { _ = try? await graphFile(root: root) }

    /// The graph.json if one is already built; never builds.
    public func existingGraph(root: URL) -> URL? {
        let candidates = [Self.ownGraph(root), cachedGraph(root)].compactMap { $0 }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) }
    }

    private static func ownGraph(_ root: URL) -> URL? {
        let g = root.appendingPathComponent("graphify-out/graph.json")
        return FileManager.default.fileExists(atPath: g.path) ? g : nil
    }

    private func outDir(_ root: URL) -> URL {
        let digest = Insecure.SHA1.hash(data: Data(root.standardizedFileURL.path.utf8))
        return cacheRoot.appendingPathComponent(digest.map { String(format: "%02x", $0) }.joined())
    }

    private func cachedGraph(_ root: URL) -> URL { outDir(root).appendingPathComponent("graphify-out/graph.json") }

    private static func build(command: String, root: URL, graph: URL) async throws -> URL {
        let start = Date()
        let r = await ShellRunner().run(command, cwd: root, timeout: 120)
        guard r.status == 0 else {
            let tail = r.output.split(separator: "\n", omittingEmptySubsequences: true).suffix(5).joined(separator: "\n")
            throw ToolError("couldn't build the project map: \(tail)")
        }
        guard FileManager.default.fileExists(atPath: graph.path) else {
            throw ToolError("couldn't build the project map: no graph was written")
        }
        print("[code-graph] built \(root.lastPathComponent) in \(String(format: "%.1f", Date().timeIntervalSince(start)))s")
        return graph
    }
}

/// Answers "where is X / what calls Y / what breaks if I change Z" from the project's code graph.
public struct CodeGraphTool: AgentTool {
    let graphs: ProjectGraphs

    public init(graphs: ProjectGraphs = .shared) { self.graphs = graphs }

    public let name = "code_graph"
    public let description = "Find where code lives and how it connects, without reading files. Use it first for 'where is X', 'what calls Y', 'what breaks if I change Z'. Returns symbols with file:line; then read_file just those lines (offset/limit). The graph is a map: verify in the file before editing."
    public var parameters: JSONValue {
        schema(["target": ("string", "The question (query), or a symbol name (explain/path/affected)."),
                "to": ("string", "The second symbol, for path."),
                "budget": ("integer", "Max tokens of output for query (default 1200, max 4000).")],
               required: ["action"],
               extra: ["action": .object(["type": .string("string"),
                                          "description": .string("What to ask the graph."),
                                          "enum": .array(Self.actions.map(JSONValue.string))])])
    }
    public let effect = ToolEffect.read

    static let actions = ["query", "explain", "path", "affected", "hubs"]

    private func parse(_ args: ToolArgs) throws -> (action: String, target: String, to: String) {
        let action = args.string("action") ?? ""
        guard Self.actions.contains(action) else { throw ToolError("action must be one of query, explain, path, affected, hubs") }
        let target = args.string("target")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if action != "hubs", target.isEmpty { throw ToolError("missing \"target\"") }
        let to = args.string("to")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if action == "path", to.isEmpty { throw ToolError("missing \"to\"") }
        return (action, target, to)
    }

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let (action, target, to) = try parse(args)
        let title: String
        switch action {
        case "hubs": title = "Map: hubs"
        case "path": title = "Map: path \(target) → \(to)"
        default: title = "Map: \(action) \(target)"
        }
        return ToolPreview(kind: .read, title: title, paths: [ctx.root])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let (action, target, to) = try parse(args)
        let g = try await graphs.graphFile(root: ctx.root)
        guard let bin = await graphs.binary() else { throw ToolError("graphify isn't installed (uv tool install graphifyy)") }
        let graphArg = "--graph \(shellQuote(g.path))"
        func exec(_ sub: String) async throws -> String {
            let r = await ShellRunner().run("\(shellQuote(bin)) \(sub) \(graphArg)", cwd: ctx.root, timeout: 30)
            guard r.status == 0 else { throw ToolError(clip(r.output.trimmingCharacters(in: .whitespacesAndNewlines), max: 2_000)) }
            return r.output
        }
        var raw: String
        switch action {
        case "query":
            let budget = min(max(args.int("budget") ?? 1200, 200), 4000)
            raw = try await exec("query \(shellQuote(target)) --budget \(budget)")
        case "explain": raw = try await exec("explain \(shellQuote(target))")
        case "path":
            let sub = "path \(shellQuote(target)) \(shellQuote(to))"
            raw = try await exec(sub)
            if raw.contains("No directed path found") { raw = try await exec(sub + " --undirected") }
        case "affected": raw = try await exec("affected \(shellQuote(target)) --depth 2")
        default: raw = try await exec("god-nodes --top 15")
        }
        let text = GraphOutput.compact(raw, root: ctx.root)
        if text.isEmpty || text.hasPrefix("No matching nodes") || text.hasPrefix("No node matching") {
            return ToolOutput("No match in the project map for '\(target.isEmpty ? action : target)'. Try other words, or fall back to search.")
        }
        return ToolOutput(text)
    }
}

/// Shrinks graphify's output for the model: file:line instead of metadata, no noise nodes, no truncation prose.
public enum GraphOutput {
    private static func rx(_ p: String) -> NSRegularExpression { try! NSRegularExpression(pattern: p) }
    private static let header = rx(#"^Graph: .*?Start: \[(.*?)\] \| (\d+) nodes found"#)
    private static let node = rx(#"^NODE (.+) \[src=(\S*) loc=(?:L(\d+))?\s*(?:community=.*)?\]\s*$"#)
    private static let edge = rx(#"^EDGE (.+?) (<?--)(\S+) \[[^\]]*\](-->?|--) (.+?)(?: at=(.+?):L(\d+))?\s*$"#)
    private static let tags = rx(#" ?\[(?:EXTRACTED|INFERRED|AMBIGUOUS)\]"#)
    private static let lineRef = rx(#"(?<=\w):L(\d+)\b"#)
    private static let source = rx(#"^(\s*Source:)\s+(\S+) L(\d+)\s*$"#)
    private static let cutNodes = rx(#"(\d+) cut nodes"#)
    private static let moreNodes = rx(#"truncated\D{0,4}(\d+) more nodes"#)
    private static let pathRef = rx(#"\S:\d+$"#)

    private static func firstGroup(_ re: NSRegularExpression, _ s: String) -> String? {
        guard let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)), m.numberOfRanges > 1,
              let r = Range(m.range(at: 1), in: s) else { return nil }
        return String(s[r])
    }

    private static func groups(_ re: NSRegularExpression, _ s: String) -> [String]? {
        guard let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) else { return nil }
        return (1..<m.numberOfRanges).map { i in Range(m.range(at: i), in: s).map { String(s[$0]) } ?? "" }
    }

    private static func sub(_ re: NSRegularExpression, _ s: String, _ template: String) -> String {
        re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: template)
    }

    public static func compact(_ raw: String, root: URL) -> String {
        let lines = raw.components(separatedBy: "\n")
        // Noise: nodes with no source file (String, Sendable, Codable, ...), dropped with their edges.
        var noise = Set<String>()
        for l in lines where l.hasPrefix("NODE ") {
            if let g = groups(node, l), g[1].isEmpty { noise.insert(g[0]) }
        }
        var out: [String] = []
        var cut: String?
        var inTruncated = false
        for line in lines {
            if inTruncated {
                if line.trimmingCharacters(in: .whitespaces).isEmpty { inTruncated = false } else { continue }
            }
            if line.hasPrefix("[!] TRUNCATED:") {
                inTruncated = true
                cut = cut ?? firstGroup(cutNodes, line)
                continue
            }
            if line.hasPrefix("... (truncated") || line.hasPrefix("… (truncated") {
                cut = firstGroup(moreNodes, line) ?? cut
                continue
            }
            if line.hasPrefix("Graph: ") {
                if let g = groups(header, line) {
                    let start = g[0].replacingOccurrences(of: "'", with: "").replacingOccurrences(of: "\"", with: "")
                    out.append("Start: \(start) (\(g[1]) nodes)")
                }
                continue
            }
            if line.hasPrefix("NODE ") {
                if let g = groups(node, line), !g[1].isEmpty { out.append(g[2].isEmpty ? "\(g[0])  \(g[1])" : "\(g[0])  \(g[1]):\(g[2])") }
                continue
            }
            if line.hasPrefix("EDGE ") {
                guard let g = groups(edge, line), !noise.contains(g[0]), !noise.contains(g[4]) else { continue }
                var s = "\(g[0]) \(g[1])\(g[2])\(g[3]) \(g[4])"
                if !g[5].isEmpty { s += "  \(g[5]):\(g[6])" }
                out.append(s)
                continue
            }
            let t = line.trimmingCharacters(in: .whitespaces)
            if ["ID:", "Type:", "Community:", "Degree:", "Relations:"].contains(where: { t.hasPrefix($0) }) { continue }
            var l = sub(tags, line, "")
            l = sub(source, l, "$1 $2:$3")
            l = sub(lineRef, l, ":$1")
            if t.hasPrefix("-->") || t.hasPrefix("<--") {
                if pathRef.firstMatch(in: l, range: NSRange(l.startIndex..., in: l)) == nil { continue }  // target has no file
            }
            out.append(l)
        }
        if let cut { out.append("… \(cut) more; narrow the question or raise budget") }
        var text = out.joined(separator: "\n").replacingOccurrences(of: root.path + "/", with: "")
        while text.contains("\n\n\n") { text = text.replacingOccurrences(of: "\n\n\n", with: "\n\n") }
        return clip(text.trimmingCharacters(in: .whitespacesAndNewlines), max: 8_000)
    }

    /// god-nodes output as one line: "Main hubs: AppModel, ChatDocument, ..." (up to 8; empty when there are none).
    public static func hubsLine(_ compactHubs: String) -> String {
        let re = rx(#"^\s*\d+\.\s+(.+?) - \d+ edges"#)
        let names = compactHubs.components(separatedBy: "\n").compactMap { firstGroup(re, $0) }
        return names.isEmpty ? "" : "Main hubs: " + names.prefix(8).joined(separator: ", ")
    }
}
