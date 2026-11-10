import CoreGraphics
import Foundation
import GraphRuntime
import NodesRemote
import NodesStandard
import Testing
@testable import MLXComfyLibrary

@Suite struct WireRouterTests {
    let src = CGRect(x: 0, y: 0, width: 250, height: 120)

    @Test func everyStyleStartsAndEndsHorizontally() {
        let a = CGPoint(x: 250, y: 40), b = CGPoint(x: 600, y: 200)
        for style in WireStyle.allCases {
            let r = WireRouter.route(from: a, to: b, style: style, sourceFrame: src,
                                     targetFrame: CGRect(x: 600, y: 160, width: 250, height: 100))
            #expect(r.start == a && r.end == b, "\(style)")
            #expect(r.points[1].y == a.y && r.points[r.points.count - 2].y == b.y, "\(style)")
            #expect(r.points[1].x > a.x && r.points[r.points.count - 2].x < b.x, "\(style)")
        }
    }

    @Test func forwardRightAngleHasOneVerticalRunInTheMiddle() {
        let r = WireRouter.route(from: CGPoint(x: 250, y: 40), to: CGPoint(x: 650, y: 200), style: .rightAngle)
        #expect(r.kind == .polyline)
        #expect(r.points.count == 4)
        #expect(r.points[1].x == 450 && r.points[2].x == 450)
        let shifted = WireRouter.route(from: CGPoint(x: 250, y: 40), to: CGPoint(x: 650, y: 200), style: .rightAngle, lane: 2)
        #expect(shifted.points[1].x == 450 + 2 * WireRouter.laneSpacing)
        // Level sockets: one straight run.
        #expect(WireRouter.route(from: CGPoint(x: 250, y: 40), to: CGPoint(x: 650, y: 40), style: .rightAngle).points.count == 2)
    }

    @Test func backwardLinksGoAroundBothNodes() {
        // Target to the left of the source and overlapping it vertically.
        let target = CGRect(x: -100, y: 30, width: 250, height: 200)
        let a = CGPoint(x: src.maxX, y: 40), b = CGPoint(x: target.minX, y: 60)
        for style in [WireStyle.rightAngle, .curved] {
            let r = WireRouter.route(from: a, to: b, style: style, sourceFrame: src, targetFrame: target)
            let inner = r.sampled(step: 2).dropFirst(3).dropLast(3)
            for p in inner {
                #expect(!src.insetBy(dx: 1, dy: 1).contains(p), "\(style) crosses the source at \(p)")
                #expect(!target.insetBy(dx: 1, dy: 1).contains(p), "\(style) crosses the target at \(p)")
            }
        }
    }

    /// Two nodes side by side with a narrow gap: the wire stays in the gap instead of detouring under the nodes.
    @Test func closeNeighboursConnectStraightAcross() {
        let target = CGRect(x: 265, y: 0, width: 250, height: 260)
        for style in WireStyle.allCases {
            for dy: CGFloat in [0, 30] {
                let a = CGPoint(x: src.maxX, y: 40), b = CGPoint(x: target.minX, y: 40 + dy)
                let r = WireRouter.route(from: a, to: b, style: style, sourceFrame: src, targetFrame: target)
                for p in r.sampled(step: 1) {
                    #expect(p.x >= a.x - 0.01 && p.x <= b.x + 0.01, "\(style) dy \(dy) leaves the gap at \(p)")
                    #expect(p.y >= min(a.y, b.y) - 0.01 && p.y <= max(a.y, b.y) + 0.01, "\(style) dy \(dy) detours to \(p)")
                }
            }
        }
    }

    @Test func backwardLinkUsesTheGapBetweenNodes() {
        let target = CGRect(x: -100, y: 400, width: 250, height: 100)
        let r = WireRouter.route(from: CGPoint(x: 250, y: 40), to: CGPoint(x: -100, y: 420), style: .rightAngle,
                                 sourceFrame: src, targetFrame: target)
        #expect(r.points[2].y == (src.maxY + target.minY) / 2)
    }

    @Test func distanceFindsPointsOnTheWire() {
        let r = WireRouter.route(from: CGPoint(x: 0, y: 0), to: CGPoint(x: 400, y: 100), style: .rightAngle)
        #expect(r.distance(to: CGPoint(x: 200, y: 50)) < 1)
        #expect(r.distance(to: CGPoint(x: 100, y: 80)) > 20)
    }
}

@Suite struct GraphArrangerTests {
    let sizes: [String: CGSize] = [
        "model": CGSize(width: 250, height: 70), "text": CGSize(width: 250, height: 140),
        "latent": CGSize(width: 250, height: 90), "sampler": CGSize(width: 250, height: 200),
        "decode": CGSize(width: 250, height: 90), "save": CGSize(width: 250, height: 260),
        "note": CGSize(width: 250, height: 150),
    ]
    var edges: [GraphArranger.Edge] {
        [.init("model", "text", weak: true), .init("model", "sampler", weak: true), .init("model", "decode", weak: true),
         .init("text", "sampler"), .init("latent", "sampler"), .init("sampler", "decode"), .init("decode", "save")]
    }

    func frames(_ r: GraphArranger.Result) -> [String: CGRect] {
        Dictionary(uniqueKeysWithValues: r.positions.map { ($0.key, CGRect(origin: $0.value, size: sizes[$0.key]!)) })
    }

    @Test func linksPointRightAndNothingOverlaps() {
        let r = GraphArranger.arrange(sizes: sizes, edges: edges)
        let f = frames(r)
        #expect(f.count == sizes.count)
        for e in edges { #expect(f[e.source]!.maxX < f[e.target]!.minX, "\(e.source) → \(e.target)") }
        let all = f.sorted { $0.key < $1.key }
        for i in all.indices { for j in all.indices where j > i {
            #expect(!all[i].value.intersects(all[j].value), "\(all[i].key) overlaps \(all[j].key)")
        } }
        // The source with one consumer sits in the column right before it.
        #expect(f["latent"]!.minX == f["text"]!.minX)
        // The unconnected note goes underneath everything.
        #expect(f["note"]!.minY > f.filter { $0.key != "note" }.map(\.value.maxY).max()!)
    }

    @Test func isDeterministicAndHonoursOrigin() {
        let a = GraphArranger.arrange(sizes: sizes, edges: edges, origin: CGPoint(x: 100, y: 50))
        let b = GraphArranger.arrange(sizes: sizes, edges: edges.reversed(), origin: CGPoint(x: 100, y: 50))
        #expect(a.positions == b.positions)
        let box = GraphArranger.bounds(a.positions, sizes)
        #expect(box.minX == 100 && box.minY == 50)
    }

    @Test func groupsAreArrangedAsBlocks() {
        let r = GraphArranger.arrange(sizes: sizes, edges: edges, groups: ["g": ["sampler", "decode"]])
        let f = frames(r)
        let g = r.groupFrames["g"]!
        #expect(g.contains(f["sampler"]!) && g.contains(f["decode"]!))
        for (id, frame) in f where id != "sampler" && id != "decode" { #expect(!g.intersects(frame), "\(id) inside the group") }
        #expect(f["sampler"]!.minY >= g.minY + GraphArranger.Spacing().groupHeader)
    }

    @Test func freeSpotAvoidsEverything() {
        let taken = [CGRect(x: 0, y: 0, width: 300, height: 300), CGRect(x: 320, y: 0, width: 300, height: 300)]
        let p = GraphArranger.freeSpot(size: CGSize(width: 250, height: 100), near: CGPoint(x: 100, y: 100), avoiding: taken)
        let r = CGRect(origin: p, size: CGSize(width: 250, height: 100)).insetBy(dx: -40, dy: -40)
        #expect(!taken.contains { $0.intersects(r) })
        #expect(GraphArranger.freeSpot(size: CGSize(width: 10, height: 10), near: CGPoint(x: 1000, y: 1000), avoiding: taken)
                == CGPoint(x: 1000, y: 1000))
    }

    @Test func everyBuiltInPresetArrangesWithoutOverlaps() {
        let registry = RemoteNodes.makeRegistry()
        for p in PresetCatalog.builtIn() {
            var sz: [String: CGSize] = [:]
            var es: [GraphArranger.Edge] = []
            for n in p.graph.nodes {
                let count = registry.schema(n.type).map { $0.inputs.count + $0.outputs.count } ?? 2
                sz[n.id] = CGSize(width: 250, height: 30 + 26 * CGFloat(count))
                for case .link(let src, _) in n.inputs.values { es.append(.init(src, n.id)) }
            }
            let r = GraphArranger.arrange(sizes: sz, edges: es)
            let f = r.positions.map { CGRect(origin: $0.value, size: sz[$0.key]!) }
            for i in f.indices { for j in f.indices where j > i { #expect(!f[i].intersects(f[j]), "\(p.id)") } }
            for e in es { #expect(r.positions[e.source]!.x < r.positions[e.target]!.x, "\(p.id): \(e.source) → \(e.target)") }
        }
    }
}

@Suite struct AlignmentTests {
    let frames: [String: CGRect] = [
        "a": CGRect(x: 0, y: 0, width: 100, height: 50),
        "b": CGRect(x: 300, y: 40, width: 200, height: 80),
        "c": CGRect(x: 120, y: 200, width: 50, height: 50),
    ]

    @Test func alignsEdgesAndCentres() {
        let left = Alignment.apply(.left, to: frames)
        #expect(left["b"] == CGPoint(x: 0, y: 40) && left["c"] == CGPoint(x: 0, y: 200) && left["a"] == nil)
        let bottom = Alignment.apply(.bottom, to: frames)
        #expect(bottom["a"] == CGPoint(x: 0, y: 200) && bottom["b"] == CGPoint(x: 300, y: 170))
        let mid = Alignment.apply(.centerX, to: frames)
        #expect(mid["a"]!.x == 200 && mid["b"]!.x == 150)
    }

    @Test func distributesWithEqualGaps() {
        let d = Alignment.apply(.distributeX, to: frames)
        // a: 0–100, c moves so the gaps (span 500 − widths 350 = 150, two gaps of 75) are equal, b stays.
        #expect(d["c"] == CGPoint(x: 175, y: 200))
        #expect(d["a"] == nil && d["b"] == nil)
        #expect(Alignment.apply(.distributeX, to: ["a": frames["a"]!, "b": frames["b"]!]).isEmpty)
    }

    @Test func snapsToTheGrid() {
        #expect(Alignment.snap(CGPoint(x: 13, y: 35)) == CGPoint(x: 24, y: 24))
        #expect(Alignment.snap(CGPoint(x: -13, y: 11)) == CGPoint(x: -24, y: 0))
    }
}

@Suite struct NodeClipboardTests {
    let level = Graph(nodes: [
        GraphNode(id: "1", type: "LoadModel", position: GraphPoint(x: 0, y: 0)),
        GraphNode(id: "2", type: "TextEncode", inputs: ["model": .link(node: "1", output: "model")],
                  position: GraphPoint(x: 300, y: 0)),
        GraphNode(id: "3", type: "Sampler", inputs: ["model": .link(node: "1", output: "model"),
                                                     "positive": .link(node: "2", output: "conditioning")],
                  position: GraphPoint(x: 600, y: 100)),
    ], groups: [GraphGroup(id: "g", title: "G", x: 280, y: -20, width: 600, height: 400)])

    @Test func pastingInTheSameGraphRenamesAndOffsets() {
        let clip = NodeClipboard.copy(["2", "3"], groupIDs: ["g"], from: level, definitions: [:])
        #expect(clip.origin == GraphPoint(x: 280, y: -20))
        let p = clip.pasted(into: level, definitions: [:], at: GraphPoint(x: 1000, y: 1000), keepExternalLinks: false)
        #expect(p.nodeIDs == ["2_2", "3_2"])
        #expect(p.nodes[0].position == GraphPoint(x: 1020, y: 1020))
        #expect(p.nodes[1].inputs["positive"] == .link(node: "2_2", output: "conditioning"))
        #expect(p.nodes[1].inputs["model"] == nil, "external link dropped")
        #expect(p.groups.map(\.id) == ["g_2"] && p.groups[0].x == 1000)
        let kept = clip.pasted(into: level, definitions: [:], at: GraphPoint(x: 0, y: 0), keepExternalLinks: true)
        #expect(kept.nodes[1].inputs["model"] == .link(node: "1", output: "model"))
    }

    @Test func pastingIntoAnotherGraphKeepsFreeIDsAndDropsMissingSources() {
        let clip = NodeClipboard.copy(["3"], from: level, definitions: [:])
        let p = clip.pasted(into: Graph(), definitions: [:], at: GraphPoint(x: 0, y: 0), keepExternalLinks: true)
        #expect(p.nodeIDs == ["3"])
        #expect(p.nodes[0].inputs.isEmpty)
    }

    @Test func carriesSubgraphDefinitionsAndRenamesClashes() throws {
        let inner = Graph(nodes: [GraphNode(id: "n", type: "Note")])
        let def = SubgraphDef(name: "Inner", graph: inner, inputs: [], outputs: [])
        let other = SubgraphDef(name: "Other", graph: Graph(), inputs: [], outputs: [])
        let g = Graph(nodes: [GraphNode(id: "s", type: Graph.subgraphType, inputs: ["def": .value(.string("sg1"))])])
        let clip = NodeClipboard.copy(["s"], from: g, definitions: ["sg1": def])
        #expect(clip.subgraphs["sg1"] == def)
        // Same definition already there: reused.
        #expect(clip.pasted(into: g, definitions: ["sg1": def], at: GraphPoint(x: 0, y: 0), keepExternalLinks: false)
            .subgraphs.isEmpty)
        // A different definition under that id: renamed.
        let p = clip.pasted(into: Graph(), definitions: ["sg1": other], at: GraphPoint(x: 0, y: 0), keepExternalLinks: false)
        #expect(p.subgraphs.keys.sorted() == ["sg1_2"])
        #expect(p.nodes[0].inputs["def"] == .value(.string("sg1_2")))
        // Round-trips through JSON (the pasteboard format).
        let data = try JSONEncoder().encode(clip)
        #expect(try JSONDecoder().decode(NodeClipboard.self, from: data) == clip)
    }
}

@Suite struct CollapsedNodeTests {
    @Test func expandedNodesEncodeWithoutTheField() throws {
        let n = GraphNode(id: "1", type: "Note", position: GraphPoint(x: 1, y: 2))
        let json = String(decoding: try JSONEncoder().encode(n), as: UTF8.self)
        #expect(!json.contains("collapsed"))
        var c = n
        c.collapsed = true
        let back = try JSONDecoder().decode(GraphNode.self, from: try JSONEncoder().encode(c))
        #expect(back.isCollapsed)
    }
}

@Suite struct GroupLoopTests {
    /// Outpainting-style: the group feeds the sampler and takes the decoded image back.
    @Test func aGroupInALoopGoesAfterWhatFeedsIt() {
        let s = CGSize(width: 250, height: 100)
        let sizes = ["load": s, "pad": s, "mask": s, "comp": s, "sampler": s, "decode": s, "save": s]
        let edges: [GraphArranger.Edge] = [
            .init("load", "pad"), .init("pad", "mask"), .init("mask", "sampler"), .init("sampler", "decode"),
            .init("decode", "comp"), .init("pad", "comp"), .init("comp", "save"),
        ]
        let r = GraphArranger.arrange(sizes: sizes, edges: edges, groups: ["out": ["pad", "mask", "comp"]])
        let p = r.positions, g = r.groupFrames["out"]!
        #expect(p["sampler"]!.x < p["decode"]!.x && p["decode"]!.x < g.minX && g.maxX < p["save"]!.x)
        // A source sits just left of what it feeds.
        #expect(p["load"]!.x < g.minX && p["load"]!.x > p["sampler"]!.x)
    }
}
