import Foundation
import Testing
@testable import GraphRuntime

@Test func choicesAreSeededAndCoverAllOptions() {
    let t = "a {red|green|blue} car"
    #expect(PromptTemplate.expand(t, seed: 7) == PromptTemplate.expand(t, seed: 7))
    let seen = Set((0..<64).map { PromptTemplate.expand(t, seed: UInt64($0)) })
    #expect(seen == ["a red car", "a green car", "a blue car"])
}

@Test func nestedChoicesAndLiteralBraces() {
    let t = "{a {big|small} dog|a cat}"
    let seen = Set((0..<64).map { PromptTemplate.expand(t, seed: UInt64($0)) })
    #expect(seen == ["a big dog", "a small dog", "a cat"])
    // Braces without a choice stay as written; unmatched braces too.
    #expect(PromptTemplate.expand("json {x} and {y", seed: 1) == "json {x} and {y")
    #expect(!PromptTemplate.isTemplate("plain {x} text"))
    #expect(PromptTemplate.isTemplate("a {b|c}"))
    #expect(PromptTemplate.isTemplate("a __animal__"))
}

@Test func wildcardsExpandRecursively() {
    let lists = ["animal": ["{red|blue} fox", "__bird__"], "bird": ["owl"]]
    let seen = Set((0..<64).map { PromptTemplate.expand("a __animal__", seed: UInt64($0)) { lists[$0] } })
    #expect(seen == ["a red fox", "a blue fox", "a owl"])
    // Unknown wildcards are left alone.
    #expect(PromptTemplate.expand("a __nope__ here", seed: 1) { lists[$0] } == "a __nope__ here")
}

@Test func wildcardFilesSkipCommentsAndBlanks() throws {
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try "# colors\nred\n\n  green  \n".write(to: dir.appendingPathComponent("color.txt"), atomically: true, encoding: .utf8)
    let lookup = PromptTemplate.wildcardFiles(in: dir)
    #expect(lookup("color") == ["red", "green"])
    #expect(lookup("missing") == nil)
}

@Test func stylesWrapPrompt() {
    let s = PromptStyle(name: "Film", template: "film still of {prompt}, grain")
    #expect(s.apply(to: " a dog ") == "film still of a dog, grain")
    #expect(PromptStyle(name: "x", template: "oil painting").apply(to: "a dog") == "a dog, oil painting")
}

@Test func plotAxisParsesNumbersAndRanges() throws {
    let a = try PlotAxis.parse("1-3, 7", type: .int, node: "s", input: "seed")
    #expect(a.values == [.int(1), .int(2), .int(3), .int(7)])
    #expect(a.labels.first == "seed 1")
    let b = try PlotAxis.parse("4-12:4", type: .int, node: "s", input: "steps")
    #expect(b.values == [.int(4), .int(8), .int(12)])
    let c = try PlotAxis.parse("0.2-0.6:0.2", type: .float, node: "s", input: "denoise")
    #expect(c.values == [.double(0.2), .double(0.4), .double(0.6)])
    #expect(c.labels == ["denoise 0.2", "denoise 0.4", "denoise 0.6"])
    #expect(throws: PlotAxis.ParseError.self) { try PlotAxis.parse("1.5", type: .int, node: "s", input: "steps") }
    #expect(throws: PlotAxis.ParseError.self) { try PlotAxis.parse("1-100", type: .int, node: "s", input: "seed") }
    #expect(throws: PlotAxis.ParseError.self) { try PlotAxis.parse(" ", type: .int, node: "s", input: "seed") }
}

@Test func plotAxisSearchReplaceAndText() throws {
    let cur = JSONValue.string("a red fox in snow")
    let a = try PlotAxis.parse("red, arctic, grey", type: .string, node: "p", input: "prompt", current: cur,
                               searchReplace: true)
    #expect(a.values == [.string("a red fox in snow"), .string("a arctic fox in snow"), .string("a grey fox in snow")])
    #expect(a.labels == ["red", "arctic", "grey"])
    #expect(throws: PlotAxis.ParseError.self) {
        try PlotAxis.parse("blue, green", type: .string, node: "p", input: "prompt", current: cur, searchReplace: true)
    }
    let b = try PlotAxis.parse("a cat, sitting | a dog", type: .string, node: "p", input: "prompt")
    #expect(b.values == [.string("a cat, sitting"), .string("a dog")])
}

@Test func plotSpecBuildsCellGraphs() throws {
    let base = Graph(nodes: [
        GraphNode(id: "s1", type: "Sampler", inputs: ["seed": .value(.int(1)), "steps": .value(.int(8))]),
        GraphNode(id: "s2", type: "Sampler", inputs: ["seed": .value(.int(1))]),
    ])
    let spec = PlotSpec(x: try PlotAxis.parse("10, 20", type: .int, node: "s1", input: "seed"),
                        y: try PlotAxis.parse("4, 8", type: .int, node: "s1", input: "steps"))
    #expect(spec.cellCount == 4)
    let g = spec.graph(for: base, column: 1, row: 0)
    #expect(g["s1"]?.inputs["seed"] == .value(.int(20)))
    #expect(g["s2"]?.inputs["seed"] == .value(.int(20)))  // seeds follow together
    #expect(g["s1"]?.inputs["steps"] == .value(.int(4)))
    #expect(spec.label(column: 1, row: 0) == "seed 20 · steps 4")
}

@Test func graphMetadataIsOptional() throws {
    let old = #"{"version":1,"nodes":[]}"#
    let g = try JSONDecoder().decode(Graph.self, from: Data(old.utf8))
    #expect(g.metadata == nil)
    #expect(!g.jsonString().contains("metadata"))
    var h = g
    h.metadata = ["k": "v"]
    let back = try JSONDecoder().decode(Graph.self, from: try h.jsonData())
    #expect(back.metadata == ["k": "v"])
}
