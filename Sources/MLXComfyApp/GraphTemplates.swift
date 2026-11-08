import Foundation
import GraphRuntime

enum GraphTemplates {
    /// The graph behind Simple mode: model → prompt → empty latent → sampler → decode → save.
    static func textToImage(modelPath: String, outputDirectory: String,
                            prompt: String = "A red fox standing in fresh snow at sunrise, detailed fur, photograph")
        -> Graph
    {
        func link(_ n: String, _ o: String) -> NodeInput { .link(node: n, output: o) }
        return Graph(nodes: [
            GraphNode(id: "model", type: "LoadModel", inputs: ["path": .value(.string(modelPath))],
                      position: GraphPoint(x: 40, y: 140)),
            GraphNode(id: "prompt", type: "TextEncode",
                      inputs: ["model": link("model", "model"), "prompt": .value(.string(prompt))],
                      position: GraphPoint(x: 340, y: 40)),
            GraphNode(id: "latent", type: "EmptyLatent",
                      inputs: ["width": .value(.int(1024)), "height": .value(.int(1024))],
                      position: GraphPoint(x: 340, y: 300)),
            GraphNode(id: "sampler", type: "Sampler",
                      inputs: ["model": link("model", "model"), "conditioning": link("prompt", "conditioning"),
                               "latent": link("latent", "latent"), "seed": .value(.int(1)),
                               "steps": .value(.int(0))],
                      position: GraphPoint(x: 640, y: 120)),
            GraphNode(id: "decode", type: "VAEDecode",
                      inputs: ["model": link("model", "model"), "latent": link("sampler", "latent")],
                      position: GraphPoint(x: 940, y: 140)),
            GraphNode(id: "save", type: "SaveImage",
                      inputs: ["image": link("decode", "image"), "filename_prefix": .value(.string("mlxcomfy")),
                               "directory": .value(.string(outputDirectory))],
                      position: GraphPoint(x: 1220, y: 140)),
        ])
    }

    static let sizePresets: [(name: String, width: Int, height: Int)] = [
        ("Square 1:1", 1024, 1024),
        ("Portrait 3:4", 896, 1184),
        ("Landscape 4:3", 1184, 896),
        ("Portrait 9:16", 768, 1344),
        ("Landscape 16:9", 1344, 768),
    ]
}
