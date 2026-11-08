import Foundation
import GraphRuntime

/// Simple-mode graph rewiring: reference image, inpainting, outpainting and the hi-res fix.
///
/// Each feature is a chain of nodes with fixed IDs spliced into the standard
/// `LoadModel → TextEncode → EmptyLatent → Sampler → VAEDecode → SaveImage` graph, so it shows up in the node editor
/// like any other nodes and is removed again without touching the rest of the graph.
/// Inpainting and outpainting both replace the Sampler's latent and composite over the reference, so they are
/// exclusive. The hi-res pass sits between the final image and Save Image and works with either.
extension AppModel {
    static let inpaintIDs = (encode: "inpaint_encode", mask: "inpaint_mask", feather: "inpaint_feather",
                             latent: "inpaint_latent", composite: "inpaint_composite")
    static let outpaintIDs = (pad: "outpaint_pad", feather: "outpaint_feather", encode: "outpaint_encode",
                              latent: "outpaint_latent", composite: "outpaint_composite")
    static let hiresIDs = (scale: "hires_scale", encode: "hires_encode", sampler: "hires_sampler",
                           decode: "hires_decode")

    static var inpaintChain: [String] {
        let i = inpaintIDs
        return [i.encode, i.mask, i.feather, i.latent, i.composite]
    }

    static var outpaintChain: [String] {
        let o = outpaintIDs
        return [o.pad, o.feather, o.encode, o.latent, o.composite]
    }

    static var hiresChain: [String] {
        let h = hiresIDs
        return [h.scale, h.encode, h.sampler, h.decode]
    }

    // MARK: state

    /// The LoadImage node feeding the first TextEncode's `image` input, if any.
    var referenceImageNode: GraphNode? {
        guard let te = firstNode(ofType: "TextEncode"), case .link(let src, _)? = te.inputs["image"],
              let n = graph[src], n.type == "LoadImage" else { return nil }
        return n
    }

    var referenceImagePath: String? {
        guard let n = referenceImageNode, case .value(.string(let p))? = n.inputs["path"], !p.isEmpty else { return nil }
        return p
    }

    var maskPath: String? {
        guard let n = graph[Self.inpaintIDs.mask], case .value(.string(let p))? = n.inputs["path"] else { return nil }
        return p
    }

    var isOutpainting: Bool { graph[Self.outpaintIDs.pad] != nil }
    var isHiResEnabled: Bool { graph[Self.hiresIDs.sampler] != nil }

    /// Where the final image enters the hi-res pass, or Save Image when there is none: the input that inpaint /
    /// outpaint composites are spliced in front of.
    func finalImageConsumer(in g: Graph) -> (node: String, input: String)? {
        if g[Self.hiresIDs.scale] != nil { return (Self.hiresIDs.scale, "image") }
        guard let save = g.nodes.first(where: { $0.type == "SaveImage" }) else { return nil }
        return (save.id, "image")
    }

    // MARK: reference image

    /// Uses `url` as the condition image (editing / reference) and makes the output match its aspect ratio.
    func setReferenceImage(_ url: URL, undoManager: UndoManager?) {
        guard let te = firstNode(ofType: "TextEncode") else { lastError = "This graph has no TextEncode node."; return }
        mutate("Set Reference Image", undoManager: undoManager) { g in
            if let existing = referenceImageNode {
                try? g.setValue(.string(url.path), node: existing.id, input: "path")
                // A mask painted for the previous image no longer fits.
                if g[Self.inpaintIDs.mask] != nil { g = Self.removingInpaint(from: g) }
            } else {
                let id = g["refimage"] == nil ? "refimage" : "refimage_\(g.nodes.count)"
                let p = te.position ?? GraphPoint(x: 340, y: 40)
                g.nodes.append(GraphNode(id: id, type: "LoadImage", inputs: ["path": .value(.string(url.path))],
                                         position: GraphPoint(x: p.x - 300, y: p.y - 120)))
                var t = g[te.id]!
                t.inputs["image"] = .link(node: id, output: "image")
                g[te.id] = t
            }
            if let l = g.nodes.first(where: { $0.type == "EmptyLatent" }) {
                try? g.setValue(.int(0), node: l.id, input: "width")
                try? g.setValue(.int(0), node: l.id, input: "height")
            }
        }
    }

    func clearReferenceImage(undoManager: UndoManager?) {
        guard let ref = referenceImageNode else { return }
        mutate("Remove Reference Image", undoManager: undoManager) { g in
            g = Self.removingOutpaint(from: Self.removingInpaint(from: g))
            g.nodes.removeAll { $0.id == ref.id }
            for i in g.nodes.indices {
                for (k, v) in g.nodes[i].inputs { if case .link(ref.id, _) = v { g.nodes[i].inputs[k] = nil } }
            }
            if let l = g.nodes.first(where: { $0.type == "EmptyLatent" }) {
                try? g.setValue(.int(1024), node: l.id, input: "width")
                try? g.setValue(.int(1024), node: l.id, input: "height")
            }
        }
    }

    // MARK: inpainting

    /// Switches the simple graph to inpainting with the mask at `url` (white = regenerate):
    /// reference → VAE Encode → Set Latent Mask (feathered mask) → Sampler, and the result composited back over the
    /// original with the same feathered mask, so the edit blends in without a seam.
    func setMask(_ url: URL, feather: Int = 16, undoManager: UndoManager?) {
        guard let ref = referenceImageNode, let sampler = firstNode(ofType: "Sampler"),
              let model = firstNode(ofType: "LoadModel"), let consumer = finalImageConsumer(in: graph),
              case .link(let decodeID, let decodeOut)? = graph[consumer.node]?.inputs[consumer.input] else {
            lastError = "Inpainting needs a reference image and the standard Sampler → VAE Decode → Save Image chain."
            return
        }
        let ids = Self.inpaintIDs
        mutate("Set Mask", undoManager: undoManager) { g in
            if g[ids.mask] != nil {
                try? g.setValue(.string(url.path), node: ids.mask, input: "path")
                return
            }
            g = Self.removingOutpaint(from: g)
            let sp = sampler.position ?? GraphPoint(x: 640, y: 120)
            let link = { (n: String, o: String) in NodeInput.link(node: n, output: o) }
            let savePos = g[consumer.node]?.position ?? GraphPoint(x: 1220, y: 140)
            g.nodes += [
                GraphNode(id: ids.encode, type: "VAEEncode",
                          inputs: ["model": link(model.id, "model"), "image": link(ref.id, "image"),
                                   "megapixels": .value(.double(1))],
                          position: GraphPoint(x: sp.x - 300, y: sp.y + 300)),
                GraphNode(id: ids.mask, type: "LoadMask",
                          inputs: ["path": .value(.string(url.path)), "channel": .value(.string("luminance")),
                                   "invert": .value(.bool(false))],
                          position: GraphPoint(x: sp.x - 600, y: sp.y + 500)),
                GraphNode(id: ids.feather, type: "FeatherMask",
                          inputs: ["mask": link(ids.mask, "mask"), "grow": .value(.int(0)),
                                   "feather": .value(.int(feather))],
                          position: GraphPoint(x: sp.x - 300, y: sp.y + 500)),
                GraphNode(id: ids.latent, type: "SetLatentMask",
                          inputs: ["model": link(model.id, "model"), "latent": link(ids.encode, "latent"),
                                   "mask": link(ids.feather, "mask")],
                          position: GraphPoint(x: sp.x, y: sp.y + 380)),
                GraphNode(id: ids.composite, type: "ImageCompositeMasked",
                          inputs: ["destination": link(ref.id, "image"), "source": link(decodeID, decodeOut),
                                   "mask": link(ids.feather, "mask")],
                          position: GraphPoint(x: savePos.x - 20, y: savePos.y + 260)),
            ]
            var s = g[sampler.id]!
            s.inputs["latent"] = link(ids.latent, "latent")
            s.inputs["denoise"] = .value(.double(1))
            g[sampler.id] = s
            var c = g[consumer.node]!
            c.inputs[consumer.input] = link(ids.composite, "image")
            g[consumer.node] = c
        }
    }

    func clearMask(undoManager: UndoManager?) {
        guard graph[Self.inpaintIDs.mask] != nil else { return }
        mutate("Clear Mask", undoManager: undoManager) { g in g = Self.removingInpaint(from: g) }
    }

    static func removingInpaint(from graph: Graph) -> Graph {
        removingChain(from: graph, ids: inpaintChain, latent: inpaintIDs.latent, composite: inpaintIDs.composite)
    }

    // MARK: outpainting

    /// Extends the reference image's canvas by the given pixels per side and generates the new border:
    /// reference → Pad Image → (feathered border mask) → VAE Encode → Set Latent Mask → Sampler, composited over the
    /// padded original. The prompt still sees the unpadded reference.
    func setOutpaint(left: Int, top: Int, right: Int, bottom: Int, feather: Int = 32, undoManager: UndoManager?) {
        guard let ref = referenceImageNode, let sampler = firstNode(ofType: "Sampler"),
              let model = firstNode(ofType: "LoadModel"), let consumer0 = finalImageConsumer(in: graph),
              graph[consumer0.node]?.inputs[consumer0.input] != nil else {
            lastError = "Outpainting needs a reference image and the standard Sampler → VAE Decode → Save Image chain."
            return
        }
        let ids = Self.outpaintIDs
        mutate("Extend Canvas", undoManager: undoManager) { g in
            if g[ids.pad] != nil {
                for (k, v) in ["left": left, "top": top, "right": right, "bottom": bottom] {
                    try? g.setValue(.int(v), node: ids.pad, input: k)
                }
                return
            }
            g = Self.removingInpaint(from: g)
            guard let consumer = finalImageConsumer(in: g),
                  case .link(let decodeID, let decodeOut)? = g[consumer.node]?.inputs[consumer.input] else { return }
            let sp = sampler.position ?? GraphPoint(x: 640, y: 120)
            let savePos = g[consumer.node]?.position ?? GraphPoint(x: 1220, y: 140)
            let link = { (n: String, o: String) in NodeInput.link(node: n, output: o) }
            g.nodes += [
                GraphNode(id: ids.pad, type: "PadImage",
                          inputs: ["image": link(ref.id, "image"), "left": .value(.int(left)), "top": .value(.int(top)),
                                   "right": .value(.int(right)), "bottom": .value(.int(bottom)),
                                   "fill": .value(.string("edge"))],
                          position: GraphPoint(x: sp.x - 600, y: sp.y + 320)),
                GraphNode(id: ids.feather, type: "FeatherMask",
                          inputs: ["mask": link(ids.pad, "mask"), "grow": .value(.int(0)),
                                   "feather": .value(.int(feather))],
                          position: GraphPoint(x: sp.x - 300, y: sp.y + 520)),
                GraphNode(id: ids.encode, type: "VAEEncode",
                          inputs: ["model": link(model.id, "model"), "image": link(ids.pad, "image"),
                                   "megapixels": .value(.double(0))],
                          position: GraphPoint(x: sp.x - 300, y: sp.y + 300)),
                GraphNode(id: ids.latent, type: "SetLatentMask",
                          inputs: ["model": link(model.id, "model"), "latent": link(ids.encode, "latent"),
                                   "mask": link(ids.feather, "mask")],
                          position: GraphPoint(x: sp.x, y: sp.y + 380)),
                GraphNode(id: ids.composite, type: "ImageCompositeMasked",
                          inputs: ["destination": link(ids.pad, "image"), "source": link(decodeID, decodeOut),
                                   "mask": link(ids.feather, "mask")],
                          position: GraphPoint(x: savePos.x - 20, y: savePos.y + 260)),
            ]
            var s = g[sampler.id]!
            s.inputs["latent"] = link(ids.latent, "latent")
            s.inputs["denoise"] = .value(.double(1))
            g[sampler.id] = s
            var c = g[consumer.node]!
            c.inputs[consumer.input] = link(ids.composite, "image")
            g[consumer.node] = c
        }
    }

    func clearOutpaint(undoManager: UndoManager?) {
        guard isOutpainting else { return }
        mutate("Stop Extending Canvas", undoManager: undoManager) { g in g = Self.removingOutpaint(from: g) }
    }

    static func removingOutpaint(from graph: Graph) -> Graph {
        removingChain(from: graph, ids: outpaintChain, latent: outpaintIDs.latent, composite: outpaintIDs.composite)
    }

    // MARK: hi-res fix

    /// Adds (or removes) a second pass: final image → Scale Image → VAE Encode → Sampler (low denoise, same prompt and
    /// seed) → VAE Decode → Save Image. Adds detail at a size the model can't start from directly.
    func setHiRes(_ enabled: Bool, undoManager: UndoManager?) {
        if !enabled {
            guard isHiResEnabled else { return }
            mutate("Turn Off Hi-Res Fix", undoManager: undoManager) { g in g = Self.removingHiRes(from: g) }
            return
        }
        guard !isHiResEnabled, let sampler = firstNode(ofType: "Sampler"), let model = firstNode(ofType: "LoadModel"),
              let save = firstNode(ofType: "SaveImage"), let finalImage = save.inputs["image"],
              case .link(let condNode, let condOut)? = sampler.inputs["conditioning"] else {
            if !isHiResEnabled { lastError = "The hi-res fix needs the standard Sampler → VAE Decode → Save Image chain." }
            return
        }
        let ids = Self.hiresIDs
        mutate("Turn On Hi-Res Fix", undoManager: undoManager) { g in
            let sv = save.position ?? GraphPoint(x: 1220, y: 140)
            let link = { (n: String, o: String) in NodeInput.link(node: n, output: o) }
            g.nodes += [
                GraphNode(id: ids.scale, type: "ImageScale",
                          inputs: ["image": finalImage, "factor": .value(.double(1.5))],
                          position: GraphPoint(x: sv.x, y: sv.y - 260)),
                GraphNode(id: ids.encode, type: "VAEEncode",
                          inputs: ["model": link(model.id, "model"), "image": link(ids.scale, "image"),
                                   "megapixels": .value(.double(0))],
                          position: GraphPoint(x: sv.x + 280, y: sv.y - 260)),
                GraphNode(id: ids.sampler, type: "Sampler",
                          inputs: ["model": link(model.id, "model"), "conditioning": link(condNode, condOut),
                                   "latent": link(ids.encode, "latent"),
                                   "seed": sampler.inputs["seed"] ?? .value(.int(0)),
                                   // An evenly spaced schedule: the checkpoint's distilled one ends at σ 0.41, so any
                                   // denoise below 0.7 would get a single step. 8 steps × denoise 0.4 = 3 refinement steps.
                                   "steps": .value(.int(8)),
                                   "denoise": .value(.double(0.4))],
                          position: GraphPoint(x: sv.x + 560, y: sv.y - 280)),
                GraphNode(id: ids.decode, type: "VAEDecode",
                          inputs: ["model": link(model.id, "model"), "latent": link(ids.sampler, "latent")],
                          position: GraphPoint(x: sv.x + 840, y: sv.y - 260)),
            ]
            var s = g[save.id]!
            s.inputs["image"] = link(ids.decode, "image")
            s.position = GraphPoint(x: sv.x + 1120, y: sv.y)
            g[save.id] = s
        }
    }

    static func removingHiRes(from graph: Graph) -> Graph {
        var g = graph
        let ids = hiresIDs
        let upstream = g[ids.scale]?.inputs["image"]
        g.nodes.removeAll { hiresChain.contains($0.id) }
        for i in g.nodes.indices {
            for (k, v) in g.nodes[i].inputs {
                guard case .link(let src, _) = v, hiresChain.contains(src) else { continue }
                g.nodes[i].inputs[k] = src == ids.decode ? upstream : nil
            }
        }
        return g
    }

    // MARK: helpers

    /// Removes a mask chain and restores Sampler ← Empty Latent and whatever consumed the composite ← its source.
    static func removingChain(from graph: Graph, ids: [String], latent: String, composite: String) -> Graph {
        var g = graph
        guard g.nodes.contains(where: { ids.contains($0.id) }) else { return g }
        let decodeLink = g[composite]?.inputs["source"]
        g.nodes.removeAll { ids.contains($0.id) }
        let emptyLatent = g.nodes.first { $0.type == "EmptyLatent" }
        for i in g.nodes.indices {
            for (k, v) in g.nodes[i].inputs {
                guard case .link(let src, _) = v, ids.contains(src) else { continue }
                if src == latent, let l = emptyLatent { g.nodes[i].inputs[k] = .link(node: l.id, output: "latent") }
                else if src == composite, let d = decodeLink { g.nodes[i].inputs[k] = d }
                else { g.nodes[i].inputs[k] = nil }
            }
        }
        return g
    }

    /// Where the mask editor saves masks.
    var maskDirectory: URL {
        URL(fileURLWithPath: (outputDirectory as NSString).expandingTildeInPath).appendingPathComponent("masks")
    }
}
