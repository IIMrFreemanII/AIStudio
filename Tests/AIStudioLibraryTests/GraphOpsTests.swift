import CoreGraphics
import Foundation
import GraphRuntime
import Testing
@testable import AIStudioLibrary

/// The graph helper's edits: checked against schemas, all or nothing, placed next to their neighbours.
struct GraphOpsTests {
    static let types: [String: NodeSchema] = {
        let list = [
            NodeSchema(type: "Load", category: "loaders", inputs: [InputSpec("path", .string, default: .string("m"))],
                       outputs: [OutputSpec("model", .model)]),
            NodeSchema(type: "Encode", category: "conditioning",
                       inputs: [InputSpec("model", .model), InputSpec("prompt", .string, default: .string(""), widget: .text(multiline: true))],
                       outputs: [OutputSpec("conditioning", .conditioning)]),
            NodeSchema(type: "Sampler", category: "sampling",
                       inputs: [InputSpec("model", .model), InputSpec("positive", .conditioning),
                                InputSpec("steps", .int, default: .int(8), widget: .number(min: 1, max: 100)),
                                InputSpec("cfg", .float, default: .double(1)),
                                InputSpec("scheduler", .string, default: .string("normal"), widget: .choice(["normal", "karras"]))],
                       outputs: [OutputSpec("latent", .latent)]),
            NodeSchema(type: "Decode", category: "latent", inputs: [InputSpec("model", .model), InputSpec("latent", .latent)],
                       outputs: [OutputSpec("image", .image)]),
            NodeSchema(type: "Upscale", category: "image",
                       inputs: [InputSpec("image", .image), InputSpec("scale", .float, default: .double(2))],
                       outputs: [OutputSpec("image", .image)]),
            NodeSchema(type: "Save", category: "image", inputs: [InputSpec("images", .image)], outputs: [], isOutput: true),
        ]
        return Dictionary(uniqueKeysWithValues: list.map { ($0.type, $0) })
    }()

    static func schemas(_ t: String) -> NodeSchema? { types[t] }

    static var base: Graph {
        Graph(nodes: [
            GraphNode(id: "load", type: "Load", inputs: ["path": .value(.string("m"))], position: GraphPoint(x: 0, y: 0)),
            GraphNode(id: "enc", type: "Encode", inputs: ["model": .link(node: "load", output: "model"), "prompt": .value(.string("a cat"))],
                      position: GraphPoint(x: 300, y: 0)),
            GraphNode(id: "sampler", type: "Sampler", inputs: ["model": .link(node: "load", output: "model"),
                                                                "positive": .link(node: "enc", output: "conditioning")],
                      position: GraphPoint(x: 600, y: 0)),
            GraphNode(id: "decode", type: "Decode", inputs: ["model": .link(node: "load", output: "model"),
                                                              "latent": .link(node: "sampler", output: "latent")],
                      position: GraphPoint(x: 900, y: 0)),
            GraphNode(id: "save", type: "Save", inputs: ["images": .link(node: "decode", output: "image")],
                      position: GraphPoint(x: 1200, y: 0)),
        ])
    }

    @Test func insertsAnUpscaleBeforeSave() throws {
        let ops = try GraphOps.parse([
            ["op": "add", "id": "up", "type": "Upscale", "inputs": ["image": ["node": "decode", "output": "image"], "scale": 1.5]],
            ["op": "connect", "from": "up", "to": "save", "input": "images"],
        ])
        let r = try GraphOps.apply(ops, to: Self.base, schemas: Self.schemas)
        let up = try #require(r.graph["up"])
        #expect(up.inputs["image"] == .link(node: "decode", output: "image"))
        #expect(up.inputs["scale"] == .value(.double(1.5)))
        #expect(r.graph["save"]?.inputs["images"] == .link(node: "up", output: "image"))
        #expect(r.added["up"] == "up")
        // Right of decode, in line; Save moves right to make room.
        let p = try #require(up.position)
        #expect(p.x == 900 + 250 + 60 && p.y == 0)
        #expect(r.graph["save"]?.position == GraphPoint(x: 1200 + 320, y: 0))
        #expect(r.graph["sampler"]?.position == GraphPoint(x: 600, y: 0))
        #expect(r.changes == ["added up (Upscale)", "connected up.image → save.images"])
    }

    @Test func clashingIDsAreRenamedAndLaterOpsFollow() throws {
        let ops: [GraphOps.Op] = [
            .add(id: "save", type: "Save", title: nil, inputs: [:]),
            .connect(from: "decode", output: nil, to: "save", input: "images"),
        ]
        let r = try GraphOps.apply(ops, to: Self.base, schemas: Self.schemas)
        // The wished id was taken: the new node is save2, but "save" still names the existing node first.
        #expect(r.added["save"] == "save2")
        #expect(r.graph["save2"] != nil)
        #expect(r.graph.nodes.count == 6)
    }

    @Test func setCoercesAndChecksValues() throws {
        var r = try GraphOps.apply([.set(node: "sampler", input: "steps", value: .double(12))], to: Self.base, schemas: Self.schemas)
        #expect(r.graph["sampler"]?.inputs["steps"] == .value(.int(12)))
        r = try GraphOps.apply([.set(node: "sampler", input: "cfg", value: .int(3))], to: Self.base, schemas: Self.schemas)
        #expect(r.graph["sampler"]?.inputs["cfg"] == .value(.double(3)))

        #expect(throws: GraphOps.Failure.self) {
            try GraphOps.apply([.set(node: "sampler", input: "scheduler", value: .string("exotic"))], to: Self.base, schemas: Self.schemas)
        }
        #expect(throws: GraphOps.Failure.self) {
            try GraphOps.apply([.set(node: "sampler", input: "steps", value: .int(500))], to: Self.base, schemas: Self.schemas)
        }
        #expect(throws: GraphOps.Failure.self) {
            try GraphOps.apply([.set(node: "sampler", input: "positive", value: .string("x"))], to: Self.base, schemas: Self.schemas)
        }
    }

    @Test func badLinksAreRefusedWithReasons() {
        func failure(_ ops: [GraphOps.Op]) -> String {
            do { _ = try GraphOps.apply(ops, to: Self.base, schemas: Self.schemas); return "" }
            catch { return "\(error)" }
        }
        #expect(failure([.connect(from: "load", output: "model", to: "save", input: "images")]).contains("takes IMAGE"))
        #expect(failure([.connect(from: "nope", output: nil, to: "save", input: "images")]).contains("no node \"nope\""))
        #expect(failure([.connect(from: "decode", output: "pixels", to: "save", input: "images")]).contains("no output \"pixels\""))
        #expect(failure([.add(id: nil, type: "Teleport", title: nil, inputs: [:])]).contains("unknown node type"))
        #expect(failure([.set(node: "sampler", input: "stepz", value: .int(1))]).contains("its inputs"))
    }

    @Test func aBatchWithAnErrorChangesNothing() throws {
        let g = Self.base
        #expect(throws: GraphOps.Failure.self) {
            try GraphOps.apply([.set(node: "sampler", input: "steps", value: .int(4)), .remove(node: "ghost")], to: g, schemas: Self.schemas)
        }
        #expect(g == Self.base)
    }

    @Test func cyclesAreRefused() {
        let ops: [GraphOps.Op] = [
            .add(id: "u1", type: "Upscale", title: nil, inputs: ["image": .link(node: "decode", output: "image")]),
            .add(id: "u2", type: "Upscale", title: nil, inputs: ["image": .link(node: "u1", output: "image")]),
            .connect(from: "u2", output: nil, to: "u1", input: "image"),
        ]
        do {
            _ = try GraphOps.apply(ops, to: Self.base, schemas: Self.schemas)
            Issue.record("expected a cycle error")
        } catch let f as GraphOps.Failure {
            #expect(f.index == nil)
            #expect(f.message.contains("loop"))
        } catch { Issue.record("\(error)") }
    }

    @Test func removeCutsLinksAndDisconnectRenameWork() throws {
        let r = try GraphOps.apply([.remove(node: "decode"), .rename(node: "sampler", title: "Main"),
                                    .disconnect(node: "sampler", input: "positive")],
                                   to: Self.base, schemas: Self.schemas)
        #expect(r.graph["decode"] == nil)
        #expect(r.graph["save"]?.inputs["images"] == nil)
        #expect(r.graph["sampler"]?.title == "Main")
        #expect(r.graph["sampler"]?.inputs["positive"] == nil)
        #expect(r.changes[0] == "removed decode (disconnecting save.images)")
    }

    @Test func describesTheGraphAndTypes() {
        var g = Self.base
        g["save"]?.inputs["images"] = nil
        let text = GraphOps.describe(g, schemas: Self.schemas)
        #expect(text.contains("sampler [Sampler]: model ← load.model; positive ← enc.conditioning"))
        #expect(text.contains("save [Save]: images=MISSING"))
        #expect(text.contains("enc [Encode]: model ← load.model; prompt=“a cat”; outputs: conditioning:CONDITIONING"))

        let list = GraphOps.typeList(Array(Self.types.values))
        #expect(list.contains("# sampling\nSampler — in: model, positive; out: latent"))
        let one = GraphOps.describeType(Self.types["Sampler"]!)
        #expect(one.contains("- scheduler: STRING, default “normal”, one of: normal, karras"))
        #expect(one.contains("- steps: INT, default 8, min 1, max 100"))
    }

    @Test func parsesNumbersAndBooleans() throws {
        let raw = try JSONSerialization.jsonObject(with: Data(#"[{"op":"set","node":"a","input":"b","value":true},{"op":"set","node":"a","input":"b","value":3},{"op":"set","node":"a","input":"b","value":0.25}]"#.utf8)) as! [Any]
        let ops = try GraphOps.parse(raw)
        #expect(ops == [.set(node: "a", input: "b", value: .bool(true)), .set(node: "a", input: "b", value: .int(3)),
                        .set(node: "a", input: "b", value: .double(0.25))])
        #expect(throws: GraphOps.Failure.self) { try GraphOps.parse([["op": "explode"]]) }
    }
}
