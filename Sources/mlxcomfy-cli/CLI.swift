import ArgumentParser
import Foundation
import GraphRuntime
import MLX
import MLXDiffusionCore
import NodesStandard
import QwenImage21
import Tokenizers

@main
struct MLXComfyCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlxcomfy-cli",
        abstract: "MLXComfyUI engine command line.",
        subcommands: [Run.self, Generate.self, Nodes.self, GraphFromPNG.self, Compare.self, VAERoundTrip.self,
                      Tokenize.self])
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func memoryString() -> String {
    String(format: "active %.1f GB, peak %.1f GB", Double(Memory.activeMemory) / 1e9, Double(Memory.peakMemory) / 1e9)
}

func parseSize(_ s: String) throws -> (Int, Int) {
    let p = s.lowercased().split(separator: "x").compactMap { Int($0) }
    guard p.count == 2 else { throw ValidationError("size must look like 1024x1024") }
    return (p[0], p[1])
}

/// `node.input=value` overrides.
func applyOverrides(_ overrides: [String], to graph: inout Graph) throws {
    for o in overrides {
        guard let eq = o.firstIndex(of: "="), let dot = o[..<eq].firstIndex(of: ".") else {
            throw ValidationError("override '\(o)' must look like node.input=value")
        }
        try graph.setValue(JSONValue.parse(String(o[o.index(after: eq)...])), node: String(o[..<dot]),
                           input: String(o[o.index(after: dot)..<eq]))
    }
}

struct Run: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Execute a graph JSON file.")

    @Argument(help: "Graph JSON (or a PNG saved by SaveImage).") var graph: String
    @Option(name: .customLong("set"), help: "Override a literal input: node.input=value (repeatable).")
    var overrides: [String] = []
    @Option(name: .customLong("then-set"),
            help: "Run a second time in the same process with these overrides, to exercise the cache.")
    var secondOverrides: [String] = []

    func run() async throws {
        let url = URL(fileURLWithPath: graph)
        var g: Graph
        if url.pathExtension.lowercased() == "png" {
            guard let json = ImageArray.readPNGText(url)[StandardNodes.graphMetadataKey] else {
                throw ValidationError("\(graph) carries no graph metadata")
            }
            g = try JSONDecoder().decode(Graph.self, from: Data(json.utf8))
        } else {
            g = try Graph.load(url)
        }
        try applyOverrides(overrides, to: &g)

        let executor = GraphExecutor(registry: StandardNodes.makeRegistry(), hooks: StandardNodes.mlxHooks)
        try await runOnce(executor, g, label: "run 1")
        if !secondOverrides.isEmpty {
            try applyOverrides(secondOverrides, to: &g)
            try await runOnce(executor, g, label: "run 2")
        }
    }

    func runOnce(_ executor: GraphExecutor, _ g: Graph, label: String) async throws {
        log("── \(label)")
        let result = try await executor.run(g) { event in
            switch event {
            case .started(let nodes): log("plan: \(nodes.joined(separator: " → "))")
            case .nodeStarted(let n): log("▶︎ \(n)")
            case .nodeCached(let n): log("✓ \(n) (cached)")
            case .nodeProgress(let n, let s, let t, _): log("  \(n) \(s)/\(t)")
            case .nodeFinished(let n, let secs, let peak):
                log(String(format: "✓ %@ %.2fs%@", n, secs, peak.map { String(format: ", peak %.1f GB", Double($0) / 1e9) } ?? ""))
            case .nodeFailed(let n, let e): log("✗ \(n): \(e)")
            case .finished(let secs): log(String(format: "done in %.1fs (%@)", secs, memoryString()))
            case .cancelled: log("cancelled")
            }
        }
        for (node, out) in result.outputs.sorted(by: { $0.key < $1.key }) {
            if case .object(let o)? = out["images"], let urls = o as? [URL] {
                urls.forEach { log("\(node) wrote \($0.path)") }
            }
        }
        log("executed \(result.executed.count), cached \(result.cached.count)")
    }
}

struct Nodes: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "List available node types.")

    func run() async throws {
        for s in StandardNodes.makeRegistry().schemas {
            let ins = s.inputs.map { i in
                "\(i.name): \(i.type)" + (i.defaultValue.map { " = \($0)" } ?? "")
            }.joined(separator: ", ")
            let outs = s.outputs.map { "\($0.name): \($0.type)" }.joined(separator: ", ")
            print("\(s.type) [\(s.category)]\(s.isOutput ? " (output)" : "")\n  in:  \(ins)\n  out: \(outs)")
        }
    }
}

struct GraphFromPNG: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "graph-from-png", abstract: "Print the graph embedded in a saved PNG.")
    @Argument var png: String

    func run() async throws {
        guard let json = ImageArray.readPNGText(URL(fileURLWithPath: png))[StandardNodes.graphMetadataKey] else {
            throw ValidationError("no graph metadata in \(png)")
        }
        let g = try JSONDecoder().decode(Graph.self, from: Data(json.utf8))
        print(String(data: try g.jsonData(), encoding: .utf8)!)
    }
}

struct Generate: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Text-to-image with Qwen-Image 2.1 (no graph).")

    @Option(help: "Model folder (diffusers layout).") var model = "qwen-turbo-mlx"
    @Option(help: "Prompt text.") var prompt: String
    @Option(help: "Output size WxH.") var size = "1024x1024"
    @Option(help: "Random seed.") var seed: UInt64 = 0
    @Option(help: "Override the checkpoint's step count.") var steps: Int?
    @Option(help: "Output PNG path.") var out = "out.png"
    @Flag(help: "Save every step's x0 estimate as out.stepN.png.") var saveSteps = false
    @Flag(help: "Expand the 8-bit transformer to bf16 at load (more memory, faster steps).") var dequantize = false

    func run() async throws {
        let (w, h) = try parseSize(size)
        let t0 = Date()
        let pipe = try await QwenImage21Pipeline(directory: URL(fileURLWithPath: model),
                                                 dequantizeTransformer: dequantize, log: log)
        log(String(format: "loaded in %.1fs (%@)", Date().timeIntervalSince(t0), memoryString()))

        let req = QwenImage21Pipeline.Request(prompt: prompt, width: w, height: h, seed: seed, steps: steps)
        let snapped = pipe.snapSize(width: w, height: h)
        let outURL = URL(fileURLWithPath: out)
        let t1 = Date()
        let image = try pipe.generate(req) { step in
            log(String(format: "step %d/%d  sigma %.3f  %.2fs", step.index + 1, step.count, step.sigma,
                       Date().timeIntervalSince(t1)))
            if saveSteps {
                let img = pipe.decode(step.predicted, width: snapped.width, height: snapped.height)
                let u = outURL.deletingPathExtension().appendingPathExtension("step\(step.index + 1).png")
                try? ImageArray.savePNG(img, to: u)
            }
        }
        log(String(format: "generated in %.1fs (%@)", Date().timeIntervalSince(t1), memoryString()))
        try ImageArray.savePNG(image, to: outURL, text: [
            "mlxcomfy.prompt": prompt, "mlxcomfy.seed": String(seed), "mlxcomfy.model": "Qwen-Image-2.1",
        ])
        log("wrote \(outURL.path)")
    }
}

struct VAERoundTrip: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "vae-roundtrip", abstract: "Encode and decode an image through the VAE.")

    @Option var model = "qwen-turbo-mlx"
    @Option var input: String
    @Option var out = "roundtrip.png"
    @Option(help: "Resize to this size (multiple of 32), e.g. 768x1024.") var size: String?

    func run() async throws {
        let vae = try QwenImage21VAE(directory: URL(fileURLWithPath: model).appendingPathComponent("vae"))
        let cg = try ImageArray.load(URL(fileURLWithPath: input))
        var w = cg.width / 32 * 32, h = cg.height / 32 * 32
        if let size { (w, h) = try parseSize(size) }
        let x = ImageArray.array(from: cg, width: w, height: h)
        let t = Date()
        let z = vae.encode((x * 2 - 1).expandedDimensions(axis: 0))
        eval(z)
        log("latents \(z.shape)  mean \(z.mean().item(Float.self))  std \(sqrt(z.variance().item(Float.self)))")
        let y = (vae.decode(normalizedLatents: z)[0].asType(.float32) + 1) / 2
        eval(y)
        let diff = abs(y[0..., 0..., ..<3] - x[0..., 0..., ..<3])
        let mse = (diff * diff).mean().item(Float.self)
        log(String(format: "%.1fs  mean abs err %.4f  PSNR %.2f dB  (%@)", Date().timeIntervalSince(t),
                   diff.mean().item(Float.self), 10 * log10(1 / mse), memoryString()))
        try ImageArray.savePNG(y, to: URL(fileURLWithPath: out))
        log("wrote \(out)")
    }
}

struct Compare: AsyncParsableCommand {
    static let configuration = CommandConfiguration(abstract: "Compare the pixels of two images.")
    @Argument var a: String
    @Argument var b: String

    func run() async throws {
        let x = ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: a)))
        let y = ImageArray.array(from: try ImageArray.load(URL(fileURLWithPath: b)))
        guard x.shape == y.shape else { print("different sizes: \(x.shape) vs \(y.shape)"); throw ExitCode(1) }
        let d = abs(x - y)
        let maxDiff = d.max().item(Float.self) * 255
        let differing = (d .> 0).asType(.float32).mean().item(Float.self) * 100
        print(String(format: "max diff %.0f/255, %.2f%% of values differ", maxDiff, differing))
        if maxDiff > 0 { throw ExitCode(1) }
        print("identical")
    }
}

struct Tokenize: AsyncParsableCommand {
    @Option var model = "qwen-turbo-mlx"
    @Argument var text: String

    func run() async throws {
        let url = URL(fileURLWithPath: model).appendingPathComponent("processor")
        let tok = try await AutoTokenizer.from(modelFolder: url)
        print(tok.encode(text: text, addSpecialTokens: false))
    }
}
