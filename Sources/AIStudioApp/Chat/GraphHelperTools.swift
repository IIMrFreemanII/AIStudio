import Agent
import AppKit
import AIStudioLibrary
import Foundation
import GraphRuntime
import ImageIO
import LLMClient
import MLXDiffusionCore
import NodesStandard

/// The graph helper's side of a reply: which workflow it works on, the graph before the reply (one undo step at the
/// end), and what it changed. Its tools call into it on the main actor.
@MainActor
final class GraphHelperSession {
    weak var app: AppModel?
    weak var workflow: WorkflowDocument?
    let start: Graph
    private(set) var changes: [String] = []

    init(app: AppModel, workflow: WorkflowDocument) {
        self.app = app
        self.workflow = workflow
        start = workflow.graph
    }

    func doc() throws -> (AppModel, WorkflowDocument) {
        guard let app, let workflow, app.tabs.contains(where: { $0 === workflow }) else {
            throw ToolError("the workflow tab was closed")
        }
        return (app, workflow)
    }

    func types(_ app: AppModel) -> GraphOps.Schemas { { app.registry.schema($0) } }

    /// Ends the reply: the edits become one undo step on the workflow ("Graph Helper").
    func commit() -> (before: Graph, after: Graph, changes: [String])? {
        guard !changes.isEmpty, let app, let workflow, workflow.graph != start else { return nil }
        app.commitLiveEdit(from: start, in: workflow, actionName: GraphHelper.undoName, undoManager: workflow.undoManager)
        return (start, workflow.graph, changes)
    }

    // MARK: tools

    func getGraph() throws -> String {
        let (app, doc) = try doc()
        var out = "Workflow “\(doc.name)” (runs on \(app.backendName(doc.backendID))):\n"
        out += GraphOps.describe(doc.graph, schemas: types(app))
        out += "\n\n" + (app.validationError(of: doc.graph).map { "It can't run yet: \($0)" } ?? "It's valid and can run.")
        return out
    }

    func edit(_ ops: [GraphOps.Op]) throws -> String {
        let (app, doc) = try doc()
        let r: GraphOps.Result
        do {
            r = try GraphOps.apply(ops, to: doc.graph, schemas: types(app)) { node in
                app.registry.schema(node.type).map { NodeLayout.frame(node, $0).size } ?? CGSize(width: 250, height: 160)
            }
        } catch let f as GraphOps.Failure {
            throw ToolError("\(f) — nothing was changed")
        }
        doc.graph = r.graph
        // Live, no undo yet (the reply's edits are committed as one step); show what's new.
        let newIDs = Set(r.added.values)
        doc.selection = newIDs.isEmpty ? doc.selection.filter { r.graph[$0] != nil } : newIDs
        changes += r.changes
        var out = r.changes.joined(separator: "\n")
        // Wished ids that were taken: say what the new nodes are called.
        let renamed = ops.compactMap { op -> String? in
            guard case .add(let wish?, _, _, _) = op, let actual = r.added[wish], actual != wish else { return nil }
            return "\(wish) is \(actual)"
        }
        if !renamed.isEmpty { out += "\nids: " + renamed.joined(separator: ", ") }
        out += "\n" + (app.validationError(of: doc.graph).map { "Not runnable yet: \($0)" } ?? "The workflow is valid.")
        return out
    }

    func setPrompt(_ text: String, node: String?) throws -> String {
        let (_, doc) = try doc()
        let candidates = doc.graph.nodes.filter { AppModel.promptFields[$0.type] != nil }
        let target: GraphNode
        if let node {
            guard let n = doc.graph[node], AppModel.promptFields[n.type] != nil else {
                throw ToolError("\(node) has no prompt; nodes with one: \(candidates.map(\.id).joined(separator: ", "))")
            }
            target = n
        } else {
            guard candidates.count == 1 else {
                throw ToolError(candidates.isEmpty ? "the workflow has no prompt node"
                                : "say which node: \(candidates.map(\.id).joined(separator: ", "))")
            }
            target = candidates[0]
        }
        let field = AppModel.promptFields[target.type]!
        return try edit([.set(node: target.id, input: field, value: .string(text))])
            .replacingOccurrences(of: "set \(target.id).\(field)", with: "set the prompt of \(target.id)")
    }

    /// Queues the workflow once and waits for it; returns how it went and what it made.
    func run() async throws -> String {
        let (app, doc) = try doc()
        if let err = app.validationError(of: doc.graph) { throw ToolError("the workflow can't run: \(err)") }
        guard let id = app.enqueue(doc: doc, count: 1, quiet: true).first else { throw ToolError("it couldn't be queued") }
        let before = doc.outputs.count
        while true {
            do { try await Task.sleep(for: .milliseconds(300)) } catch {
                // The reply was stopped: don't leave the job waiting.
                if let i = app.queue.firstIndex(where: { $0.id == id }), app.queue[i].status == .pending {
                    app.queue[i].status = .cancelled
                }
                throw error
            }
            guard let item = app.queue.first(where: { $0.id == id }) else { throw ToolError("the job was removed from the queue") }
            switch item.status {
            case .pending, .running: continue
            case .cancelled: return "The run was cancelled."
            case .failed(let why): return "The run failed: \(why)\nget_errors shows which node."
            case .done(let seconds):
                let made = doc.outputs.dropFirst(before)
                var out = String(format: "Finished in %.1f s", seconds)
                if let seed = item.seed { out += " (seed \(seed))" }
                out += made.isEmpty ? ". It produced no outputs." : ". Outputs:\n" + made.map { r in
                    "- \(r.node): \(r.kind.rawValue)\(r.url.map { " " + $0.lastPathComponent } ?? "")\(r.isPreview ? " (preview)" : "")"
                }.joined(separator: "\n")
                if made.contains(where: { $0.image != nil }) { out += "\nget_last_outputs shows the images." }
                return out
            }
        }
    }

    func lastOutputs(count: Int, node: String?) throws -> ToolOutput {
        let (_, doc) = try doc()
        let shown = doc.outputs.filter { node == nil || $0.node == node }.suffix(max(1, min(count, 4)))
        guard !shown.isEmpty else { return ToolOutput("No outputs yet: queue_run first.") }
        var images: [String] = []
        var lines: [String] = []
        for r in shown {
            var l = "- \(r.node): \(r.kind.rawValue)\(r.url.map { " " + $0.path } ?? "")"
            if let img = r.image {
                l += " (\(img.width)×\(img.height))"
                if let url = GraphHelper.dataURL(img) { images.append(url) }
            }
            lines.append(l)
        }
        return ToolOutput("The latest outputs, oldest first:\n" + lines.joined(separator: "\n"), images: images)
    }

    func errors() throws -> String {
        let (app, doc) = try doc()
        var out: [String] = []
        if let v = app.validationError(of: doc.graph) { out.append("Validation: \(v)") }
        for (id, state) in doc.nodeStates.sorted(by: { $0.key < $1.key }) {
            if case .failed(let why) = state { out.append("Node \(id) failed in the last run: \(why)") }
        }
        if let last = app.queue.last(where: { $0.docID == doc.id && $0.isFinished }), case .failed(let why) = last.status {
            out.append("Last run: failed: \(why)")
        }
        return out.isEmpty ? "No errors: the workflow is valid and its last run (if any) didn't fail." : out.joined(separator: "\n")
    }

    func models() throws -> String {
        let (app, _) = try doc()
        var out: [String] = []
        if let store = app.store {
            let installed = ModelCatalog.entries.compactMap { e in store.installedURL(e.id).map { "- \(e.name) (\(e.kind.rawValue)): \($0.path)" } }
            out.append("Installed models (folders for a LoadModel-style path input):")
            out += installed.isEmpty ? ["- none"] : installed
            let lib = store.library.items
            out.append("\nLibrary items (ids for inputs that pick a lora or model):")
            out += lib.isEmpty ? ["- none"] : lib.map { e in
                "- \(e.id): \(e.title) (\(e.kind.rawValue)\(e.family.map { ", " + $0 } ?? ""))"
                    + (e.triggerWords.isEmpty ? "" : "; trigger words: \(e.triggerWords.joined(separator: ", "))")
            }
        } else {
            out.append("The model list isn't available.")
        }
        out.append("\nThe Qwen-Image folder setting: \(app.modelPath)")
        return out.joined(separator: "\n")
    }
}

/// The graph helper's tools, prompt and image helpers.
enum GraphHelper {
    static let undoName = "Graph Helper"

    @MainActor
    static func tools(_ s: GraphHelperSession) -> [any AgentTool] {
        [TodoWriteTool(),
            HelperTool(name: "get_graph", effect: .read,
                       description: "The open workflow: every node with its id, type, literal inputs, links (input ← node.output) and outputs, and whether it can run.",
                       parameters: params([:])) { _ in ToolPreview(kind: .read, title: "Read the workflow") }
                run: { _ in ToolOutput(try await s.getGraph()) },
            HelperTool(name: "list_node_types", effect: .read,
                       description: "The node types that can be added, by category, with their linked inputs and outputs.",
                       parameters: params(["category": ("string", "Only this category (optional)")])) { _ in
                ToolPreview(kind: .read, title: "List node types")
            } run: { a in
                let cat = a.string("category")
                return ToolOutput(await MainActor.run {
                    GraphOps.typeList(s.app?.schemas ?? [], category: cat?.isEmpty == false ? cat : nil)
                })
            },
            HelperTool(name: "describe_node_type", effect: .read,
                       description: "One node type in full: each input's type, default, allowed values and range, and the outputs.",
                       parameters: params(["type": ("string", "The node type, e.g. Sampler")], required: ["type"])) { a in
                ToolPreview(kind: .read, title: "Describe \(a.string("type") ?? "a node type")")
            } run: { a in
                let type = try a.required("type")
                let text: String? = await MainActor.run { s.app?.registry.schema(type).map(GraphOps.describeType) }
                guard let text else { throw ToolError("no node type \(type) (list_node_types shows them)") }
                return ToolOutput(text)
            },
            HelperTool(name: "edit_graph", effect: .graphEdit,
                       description: """
                       Changes the workflow with a list of operations, applied all or nothing (an error changes nothing \
                       and says which operation failed). Operations:
                       {"op":"add","id":"upscale","type":"ImageScale","inputs":{"image":{"node":"decode","output":"image"},"factor":2}} \
                       (id is a wish; inputs take literals or {"node","output"} links; defaults fill the rest);
                       {"op":"connect","from":"upscale","output":"image","to":"save","input":"image"} (output may be left out when only one fits);
                       {"op":"disconnect","node":"save","input":"image"};
                       {"op":"set","node":"sampler","input":"steps","value":8};
                       {"op":"remove","node":"upscale"} (its links are cut);
                       {"op":"rename","node":"sampler","title":"Main sampler"}.
                       New nodes are placed next to what they connect to. All of a reply's edits are one undo step.
                       """,
                       parameters: .object([
                           "type": .string("object"),
                           "properties": .object(["operations": .object([
                               "type": .string("array"), "description": .string("The operations, in order"),
                               "items": .object(["type": .string("object")]),
                           ])]),
                           "required": .array([.string("operations")]),
                       ])) { a in
                let ops = (a.raw["operations"] as? [Any]) ?? []
                let parsed = (try? GraphOps.parse(ops)) ?? []
                return ToolPreview(kind: .graph, title: ops.count == 1 ? "Edit the workflow (1 change)" : "Edit the workflow (\(ops.count) changes)",
                                   detail: parsed.map(describe).joined(separator: "\n"))
            } run: { a in
                guard let ops = a.raw["operations"] as? [Any], !ops.isEmpty else { throw ToolError("\"operations\" is missing") }
                let parsed: [GraphOps.Op]
                do { parsed = try GraphOps.parse(ops) } catch { throw ToolError("\(error) — nothing was changed") }
                return ToolOutput(try await s.edit(parsed))
            },
            HelperTool(name: "set_prompt", effect: .graphEdit,
                       description: "Sets the prompt text of a prompt node (Text Encode, a cloud node's prompt…). node may be left out when the workflow has one.",
                       parameters: params(["text": ("string", "The new prompt"), "node": ("string", "The node id (optional)")], required: ["text"])) { a in
                ToolPreview(kind: .graph, title: "Set the prompt", detail: a.string("text"))
            } run: { a in
                let text = try a.required("text")
                let node = a.string("node")
                return ToolOutput(try await s.setPrompt(text, node: node?.isEmpty == false ? node : nil))
            },
            HelperTool(name: "queue_run", effect: .run,
                       description: "Runs the workflow once (with a new seed when the toolbar randomizes seeds) and waits for it to finish; says how long it took and what it produced.",
                       parameters: params([:])) { _ in ToolPreview(kind: .run, title: "Run the workflow") }
                run: { _ in ToolOutput(try await s.run()) },
            HelperTool(name: "get_last_outputs", effect: .read,
                       description: "The workflow's latest results (files, sizes), with the images to look at.",
                       parameters: params(["count": ("integer", "How many, 1–4 (default 1)"), "node": ("string", "Only this output node (optional)")])) { _ in
                ToolPreview(kind: .read, title: "Look at the latest outputs")
            } run: { a in
                let count = a.int("count") ?? 1
                let node = a.string("node")
                return try await s.lastOutputs(count: count, node: node?.isEmpty == false ? node : nil)
            },
            HelperTool(name: "get_errors", effect: .read,
                       description: "Why the workflow can't run, and which nodes failed in its last run.",
                       parameters: params([:])) { _ in ToolPreview(kind: .read, title: "Check for errors") }
                run: { _ in ToolOutput(try await s.errors()) },
            HelperTool(name: "list_models", effect: .read,
                       description: "The installed models and the library's LoRAs and models (with ids and trigger words).",
                       parameters: params([:])) { _ in ToolPreview(kind: .read, title: "List models") }
                run: { _ in ToolOutput(try await s.models()) },
        ]
    }

    static func describe(_ op: GraphOps.Op) -> String {
        switch op {
        case .add(let id, let type, _, let inputs):
            return "add \(id ?? type) (\(type))" + (inputs.isEmpty ? "" : ": " + inputs.keys.sorted().joined(separator: ", "))
        case .connect(let from, let output, let to, let input): return "connect \(from)\(output.map { "." + $0 } ?? "") → \(to).\(input)"
        case .disconnect(let node, let input): return "disconnect \(node).\(input)"
        case .set(let node, let input, let value): return "set \(node).\(input) = \(value)"
        case .remove(let node): return "remove \(node)"
        case .rename(let node, let title): return "rename \(node) to “\(title)”"
        }
    }

    static func params(_ props: [String: (String, String)], required: [String] = []) -> LLMClient.JSONValue {
        var p: [String: LLMClient.JSONValue] = [:]
        for (k, (type, desc)) in props { p[k] = .object(["type": .string(type), "description": .string(desc)]) }
        return .object(["type": .string("object"), "properties": .object(p), "required": .array(required.map { .string($0) })])
    }

    /// The system prompt: what the helper is, the workflow as it is now, the node types, the mode.
    @MainActor
    static func systemPrompt(_ s: GraphHelperSession, mode: PermissionMode) -> String {
        let graph = (try? s.getGraph()) ?? "(the workflow tab was closed)"
        let types = GraphOps.typeList(s.app?.schemas ?? [])
        let modeLine: String
        switch mode {
        case .plan: modeLine = "You are in plan mode: look and explain, but don't edit or run the workflow; finish with the changes you'd make."
        case .ask, .acceptEdits: modeLine = "Edits apply right away (the user can undo them as one step); the user approves each run."
        case .bypass: modeLine = "Edits and runs happen without asking."
        }
        return """
        You are the graph helper in AIStudio, a node-based app for image, video and audio generation on a Mac. You \
        build and fix the user's open workflow with the tools: read it (get_graph), look up node types \
        (list_node_types, describe_node_type), change it (edit_graph, set_prompt), run it (queue_run), then look \
        at the results (get_last_outputs) and errors (get_errors), and iterate. Keep edits minimal and use the \
        existing node ids. Check a node type's inputs with describe_node_type before adding it if you're unsure. \
        After editing, say briefly what you changed. Write prompts in the style the model expects (descriptive \
        English for image models). \(modeLine)

        The workflow now:
        \(graph)

        Node types:
        \(types)
        """
    }

    /// A JPEG `data:` URL at most 768 px on the long side, for the model to look at.
    static func dataURL(_ image: CGImage, maxSide: Int = 768) -> String? {
        let scale = min(1, Double(maxSide) / Double(max(image.width, image.height)))
        let w = max(1, Int(Double(image.width) * scale)), h = max(1, Int(Double(image.height) * scale))
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let small = ctx.makeImage() else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, small, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return "data:image/jpeg;base64," + (data as Data).base64EncodedString()
    }
}

/// A tool made of closures (the graph helper's tools all call into its session).
struct HelperTool: AgentTool {
    let name: String
    let effect: ToolEffect
    let description: String
    let parameters: LLMClient.JSONValue
    let makePreview: @Sendable (ToolArgs) -> ToolPreview
    let body: @Sendable (ToolArgs) async throws -> ToolOutput

    init(name: String, effect: ToolEffect, description: String, parameters: LLMClient.JSONValue,
         preview: @escaping @Sendable (ToolArgs) -> ToolPreview,
         run: @escaping @Sendable (ToolArgs) async throws -> ToolOutput) {
        self.name = name
        self.effect = effect
        self.description = description
        self.parameters = parameters
        makePreview = preview
        body = run
    }

    func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview { makePreview(args) }
    func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput { try await body(args) }
}
