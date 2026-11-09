import Foundation
import GraphRuntime

/// Graph rewiring behind Add to Workflow (and the built-in presets): reference image, inpainting, outpainting and
/// the hi-res fix.
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
    /// Image-to-image for models without reference editing (Flux): reference → VAE Encode → Sampler at a denoise < 1.
    static let img2imgEncodeID = "img2img_encode"
    static let defaultImg2ImgStrength = 0.75

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
        return resolvedPath(p)
    }

    var maskPath: String? {
        guard let n = graph[Self.inpaintIDs.mask], case .value(.string(let p))? = n.inputs["path"] else { return nil }
        return resolvedPath(p)
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
                try? g.setValue(.string(storedPath(url)), node: existing.id, input: "path")
                // A mask painted for the previous image no longer fits.
                if g[Self.inpaintIDs.mask] != nil { g = Self.removingInpaint(from: g) }
            } else {
                let id = g["refimage"] == nil ? "refimage" : "refimage_\(g.nodes.count)"
                let p = te.position ?? GraphPoint(x: 340, y: 40)
                g.nodes.append(GraphNode(id: id, type: "LoadImage", inputs: ["path": .value(.string(storedPath(url)))],
                                         position: GraphPoint(x: p.x - 300, y: p.y - 120)))
                var t = g[te.id]!
                t.inputs["image"] = .link(node: id, output: "image")
                g[te.id] = t
            }
            if let l = g.nodes.first(where: { $0.type == "EmptyLatent" }) {
                try? g.setValue(.int(0), node: l.id, input: "width")
                try? g.setValue(.int(0), node: l.id, input: "height")
            }
            g = Self.syncingImg2Img(g)
        }
    }

    func clearReferenceImage(undoManager: UndoManager?) {
        guard let ref = referenceImageNode else { return }
        mutate("Remove Reference Image", undoManager: undoManager) { g in
            g = Self.removingImg2Img(from: Self.removingOutpaint(from: Self.removingInpaint(from: g)))
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
                try? g.setValue(.string(storedPath(url)), node: ids.mask, input: "path")
                return
            }
            g = Self.removingImg2Img(from: Self.removingOutpaint(from: g))
            let sp = sampler.position ?? GraphPoint(x: 640, y: 120)
            let link = { (n: String, o: String) in NodeInput.link(node: n, output: o) }
            let savePos = g[consumer.node]?.position ?? GraphPoint(x: 1220, y: 140)
            g.nodes += [
                GraphNode(id: ids.encode, type: "VAEEncode",
                          inputs: ["model": link(model.id, "model"), "image": link(ref.id, "image"),
                                   "megapixels": .value(.double(1))],
                          position: GraphPoint(x: sp.x - 300, y: sp.y + 300)),
                GraphNode(id: ids.mask, type: "LoadMask",
                          inputs: ["path": .value(.string(storedPath(url))), "channel": .value(.string("luminance")),
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
        mutate("Clear Mask", undoManager: undoManager) { g in g = Self.syncingImg2Img(Self.removingInpaint(from: g)) }
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
            g = Self.removingImg2Img(from: Self.removingInpaint(from: g))
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
        mutate("Stop Extending Canvas", undoManager: undoManager) { g in g = Self.syncingImg2Img(Self.removingOutpaint(from: g)) }
    }

    static func removingOutpaint(from graph: Graph) -> Graph {
        removingChain(from: graph, ids: outpaintChain, latent: outpaintIDs.latent, composite: outpaintIDs.composite)
    }

    // MARK: image-to-image (Flux)

    var isImg2Img: Bool { graph[Self.img2imgEncodeID] != nil }

    /// Adds or removes the image-to-image chain so it is present exactly when the graph's model can't use a reference
    /// image for editing (Flux), there is a reference image, and no inpaint or outpaint chain owns the Sampler's latent.
    static func syncingImg2Img(_ graph: Graph) -> Graph {
        var g = graph
        guard let model = g.nodes.first(where: { $0.type == "LoadModel" }),
              let sampler = g.nodes.first(where: { $0.type == "Sampler" }) else { return g }
        let path: String = { if case .value(.string(let p))? = model.inputs["path"] { return p }; return "" }()
        let ref: String? = g.nodes.first(where: { $0.type == "TextEncode" }).flatMap { te in
            if case .link(let src, _)? = te.inputs["image"], g[src]?.type == "LoadImage" { return src }
            return nil
        }
        let want = ImageFamily.of(path: path) == .flux && ref != nil && g[inpaintIDs.mask] == nil
            && g[outpaintIDs.pad] == nil
        if !want { return g[img2imgEncodeID] == nil ? g : removingImg2Img(from: g) }
        guard g[img2imgEncodeID] == nil, let ref else { return g }
        let sp = sampler.position ?? GraphPoint(x: 640, y: 120)
        g.nodes.append(GraphNode(id: img2imgEncodeID, type: "VAEEncode",
                                 inputs: ["model": .link(node: model.id, output: "model"),
                                          "image": .link(node: ref, output: "image"),
                                          "megapixels": .value(.double(1))],
                                 position: GraphPoint(x: sp.x - 300, y: sp.y + 300)))
        var s = g[sampler.id]!
        s.inputs["latent"] = .link(node: img2imgEncodeID, output: "latent")
        if case .value(.double(let d))? = s.inputs["denoise"], d < 1 {} else {
            s.inputs["denoise"] = .value(.double(defaultImg2ImgStrength))
        }
        g[sampler.id] = s
        return g
    }

    /// Removes the image-to-image chain: Sampler ← Empty Latent at full denoise.
    static func removingImg2Img(from graph: Graph) -> Graph {
        var g = graph
        guard g[img2imgEncodeID] != nil else { return g }
        g.nodes.removeAll { $0.id == img2imgEncodeID }
        let emptyLatent = g.nodes.first { $0.type == "EmptyLatent" }
        for i in g.nodes.indices {
            for (k, v) in g.nodes[i].inputs {
                guard case .link(img2imgEncodeID, _) = v else { continue }
                g.nodes[i].inputs[k] = emptyLatent.map { .link(node: $0.id, output: "latent") }
                if g.nodes[i].type == "Sampler" { g.nodes[i].inputs["denoise"] = .value(.double(1)) }
            }
        }
        return g
    }

    /// Points the image graph at another image model, rewiring image-to-image as that model needs.
    func setImageModel(path: String, undoManager: UndoManager?) {
        guard firstNode(ofType: "LoadModel").map({ n in
            if case .value(.string(let p))? = n.inputs["path"] { return p != path }; return true
        }) == true else { return }
        mutate("Change Image Model", undoManager: undoManager) { g in
            for n in g.nodes where n.type == "LoadModel" { try? g.setValue(.string(path), node: n.id, input: "path") }
            g = Self.syncingImg2Img(g)
        }
    }

    // MARK: LoRAs

    /// The LoRA nodes between the model loader and the Sampler, in order from the loader.
    var simpleLoRAs: [GraphNode] {
        guard let sampler = firstNode(ofType: "Sampler") else { return [] }
        var chain: [GraphNode] = []
        var input = sampler.inputs["model"]
        while case .link(let id, _)? = input, let n = graph[id], n.type == "LoRA" {
            chain.insert(n, at: 0)
            input = n.inputs["model"]
        }
        return chain
    }

    /// Adds a LoRA after the last one (or after the model loader) for every Sampler that used that model.
    func addLoRA(_ url: URL, strength: Double = 1, undoManager: UndoManager?) {
        guard let sampler = firstNode(ofType: "Sampler"), let head = sampler.inputs["model"] else {
            lastError = "This graph has no Sampler to apply a LoRA to."; return
        }
        mutate("Add LoRA", undoManager: undoManager) { g in
            var n = 1
            while g["lora\(n)"] != nil { n += 1 }
            let id = "lora\(n)"
            let p = sampler.position ?? GraphPoint(x: 640, y: 120)
            g.nodes.append(GraphNode(id: id, type: "LoRA",
                                     inputs: ["model": head, "path": .value(.string(url.path)),
                                              "strength": .value(.double(strength))],
                                     position: GraphPoint(x: p.x - 300 + Double(n - 1) * 40, y: p.y - 200 - Double(n - 1) * 40)))
            for i in g.nodes.indices where g.nodes[i].type == "Sampler" && g.nodes[i].inputs["model"] == head {
                g.nodes[i].inputs["model"] = .link(node: id, output: "model")
            }
        }
    }

    func removeLoRA(_ id: String, undoManager: UndoManager?) {
        guard let node = graph[id], node.type == "LoRA" else { return }
        mutate("Remove LoRA", undoManager: undoManager) { g in
            g.nodes.removeAll { $0.id == id }
            for i in g.nodes.indices {
                for (k, v) in g.nodes[i].inputs { if case .link(id, _) = v { g.nodes[i].inputs[k] = node.inputs["model"] } }
            }
        }
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
                          inputs: ["model": sampler.inputs["model"] ?? link(model.id, "model"),
                                   "conditioning": link(condNode, condOut),
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

}

// MARK: - video

extension AppModel {
    /// The Load Image node feeding the Video Sampler's first frame, if any.
    var firstFrameNode: GraphNode? {
        guard let s = firstNode(ofType: "VideoSampler"), case .link(let id, _)? = s.inputs["image"] else { return nil }
        return graph[id]
    }

    var firstFramePath: String? {
        guard let n = firstFrameNode, case .value(.string(let p))? = n.inputs["path"], !p.isEmpty else { return nil }
        return resolvedPath(p)
    }

    /// Pins the clip's first frame to an image (image-to-video).
    func setFirstFrame(_ url: URL, undoManager: UndoManager?) {
        guard let s = firstNode(ofType: "VideoSampler") else { lastError = "This graph has no Video Sampler node."; return }
        mutate("Set First Frame", undoManager: undoManager) { g in
            if let existing = firstFrameNode {
                try? g.setValue(.string(storedPath(url)), node: existing.id, input: "path")
            } else {
                let id = g["first_frame"] == nil ? "first_frame" : "first_frame_\(g.nodes.count)"
                let p = s.position ?? GraphPoint(x: 640, y: 120)
                g.nodes.append(GraphNode(id: id, type: "LoadImage", inputs: ["path": .value(.string(storedPath(url)))],
                                         position: GraphPoint(x: p.x - 300, y: p.y + 260)))
                var n = g[s.id]!
                n.inputs["image"] = .link(node: id, output: "image")
                g[s.id] = n
            }
        }
    }

    func clearFirstFrame(undoManager: UndoManager?) {
        guard let ref = firstFrameNode, let s = firstNode(ofType: "VideoSampler") else { return }
        mutate("Remove First Frame", undoManager: undoManager) { g in
            var n = g[s.id]!
            n.inputs["image"] = nil
            g[s.id] = n
            // Keep the Load Image node if something else uses it.
            let used = g.nodes.contains { $0.inputs.values.contains { if case .link(ref.id, _) = $0 { return true }; return false } }
            if !used { g.nodes.removeAll { $0.id == ref.id } }
        }
    }

    /// Points the Video graph at another video model. Switching family resets the clip settings to that model's
    /// (Wan: 832×480 at 16 fps, up to 5 s; LTX: 768×512 at 24 fps).
    func setVideoModel(path: String, undoManager: UndoManager?) {
        guard let loader = firstNode(ofType: "LoadModel"), case .value(.string(let old))? = loader.inputs["path"],
              old != path else { return }
        let family = VideoFamily.of(path: path), changed = family != VideoFamily.of(path: old)
        let latent = firstNode(ofType: "EmptyVideoLatent")
        UserDefaults.standard.set(family.rawValue, forKey: "videoFamily")
        mutate(changed ? "Change Video Model" : "Switch Wan Variant", undoManager: undoManager) { g in
            try? g.setValue(.string(path), node: loader.id, input: "path")
            guard changed, let latent else { return }
            let (w, h, fps) = family == .wan ? (832, 480, 16.0) : (768, 512, 24.0)
            try? g.setValue(.int(w), node: latent.id, input: "width")
            try? g.setValue(.int(h), node: latent.id, input: "height")
            try? g.setValue(.double(fps), node: latent.id, input: "fps")
            if family == .wan, case .value(.double(let s))? = latent.inputs["seconds"], s > 5 {
                try? g.setValue(.double(5), node: latent.id, input: "seconds")
            }
        }
    }
}
