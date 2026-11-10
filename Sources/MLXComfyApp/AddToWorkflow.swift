import AppKit
import GraphRuntime
import MLXComfyLibrary
import SwiftUI
import UniformTypeIdentifiers

/// Ready-made chains the Graph menu, the canvas and the inspector can splice into the current workflow
/// (see SimpleChains.swift). The multi-node ones arrive as a labelled group.
@MainActor
enum AddOn: String, CaseIterable {
    case reference, inpaint, outpaint, hires, lora, firstFrame

    var title: String {
        switch self {
        case .reference: return "Reference Image…"
        case .inpaint: return "Inpaint Mask…"
        case .outpaint: return "Outpaint (Extend Canvas)"
        case .hires: return "Hi-res Fix"
        case .lora: return "LoRA"
        case .firstFrame: return "First Frame…"
        }
    }

    /// The group title for chains of several nodes.
    var groupTitle: String? {
        switch self {
        case .inpaint: return "Inpainting"
        case .outpaint: return "Outpainting"
        case .hires: return "Hi-res Fix"
        default: return nil
        }
    }

    var chain: [String] {
        switch self {
        case .inpaint: return AppModel.inpaintChain
        case .outpaint: return AppModel.outpaintChain
        case .hires: return AppModel.hiresChain
        default: return []
        }
    }
}

extension AppModel {
    // MARK: what applies

    var hasImageSampler: Bool { firstNode(ofType: "Sampler") != nil }
    var hasVideoSampler: Bool { firstNode(ofType: "VideoSampler") != nil }
    /// A local Wan 2.2 video graph (its LoRAs go on the Video Sampler's model; LTX doesn't take LoRAs here).
    var isWanGraph: Bool {
        guard let m = firstNode(ofType: "LoadModel"), case .value(.string(let p))? = m.inputs["path"] else { return false }
        return VideoFamily.of(path: p) == .wan
    }
    /// A Cloud GPU image node (FLUX.1 dev, Qwen-Image) and no local sampler: LoRAs go to the cloud node.
    var usesCloudLoRAs: Bool { !hasImageSampler && !hasVideoSampler && cloudImageNode != nil }

    func canAdd(_ a: AddOn) -> Bool {
        switch a {
        case .reference: return firstNode(ofType: "TextEncode") != nil && hasImageSampler
        case .inpaint: return referenceImageNode != nil && hasImageSampler
        case .outpaint: return referenceImageNode != nil && hasImageSampler && !isOutpainting
        case .hires: return hasImageSampler && firstNode(ofType: "SaveImage") != nil
        case .lora: return hasImageSampler || (hasVideoSampler && isWanGraph) || usesCloudLoRAs
        case .firstFrame: return hasVideoSampler
        }
    }

    func isActive(_ a: AddOn) -> Bool {
        switch a {
        case .reference: return referenceImageNode != nil
        case .inpaint: return maskPath != nil
        case .outpaint: return isOutpainting
        case .hires: return isHiResEnabled
        case .lora: return !simpleLoRAs.isEmpty || !cloudLoRAs.isEmpty
        case .firstFrame: return firstFrameNode != nil
        }
    }

    // MARK: doing it

    /// Runs an add-on; the ones that need a file or a mask ask for it first.
    func add(_ a: AddOn) {
        let undo = activeUndoManager
        switch a {
        case .reference:
            // Copied into the input folder first, so the workflow keeps working when the original moves.
            chooseAsset([.image], title: "Reference Image", message: "Choose the image to edit, redraw or use as a reference.") { copy in
                self.setReferenceImage(copy, undoManager: undo)
                self.tidyChainGroups()
            }
        case .inpaint:
            guard let ref = referenceImageNode, let path = referenceImagePath else {
                lastError = "Add a reference image first: inpainting paints over it."
                return
            }
            maskEditorTarget = MaskEditorTarget(imageNode: ref.id, imagePath: path)
        case .outpaint:
            setOutpaint(left: 256, top: 0, right: 256, bottom: 0, undoManager: undo)
            groupAddedChain(.outpaint)
        case .hires:
            setHiRes(!isHiResEnabled, undoManager: undo)
            if isHiResEnabled { groupAddedChain(.hires) }
        case .lora:
            // A LoRA node, selected: its menu (inspector, node) picks from the library or adds from a link or a file.
            let before = Set(graph.nodes.map(\.id))
            if usesCloudLoRAs { addCloudLoRA(nil, undoManager: undo) } else { addLoRA(nil, undoManager: undo) }
            if let id = graph.nodes.map(\.id).first(where: { !before.contains($0) }) { selection = [id] }
        case .firstFrame:
            chooseAsset([.image], title: "First Frame", message: "Choose the image the clip starts from.") { copy in
                self.setFirstFrame(copy, undoManager: undo)
                self.tidyChainGroups()
            }
        }
        tidyChainGroups()
    }

    /// Removes an add-on again (where that makes sense).
    func remove(_ a: AddOn) {
        let undo = activeUndoManager
        switch a {
        case .reference: clearReferenceImage(undoManager: undo)
        case .inpaint: clearMask(undoManager: undo)
        case .outpaint: clearOutpaint(undoManager: undo)
        case .hires: setHiRes(false, undoManager: undo)
        case .lora:
            for l in simpleLoRAs { removeLoRA(l.id, undoManager: undo) }
            for l in cloudLoRAs { removeCloudLoRA(l.id, undoManager: undo) }
        case .firstFrame: clearFirstFrame(undoManager: undo)
        }
        tidyChainGroups()
    }

    /// Lays out a chain that was just spliced in, puts it in free space, and wraps it in a group titled after it (once).
    func groupAddedChain(_ a: AddOn, undoManager: UndoManager? = nil) {
        guard let title = a.groupTitle, !(graph.groups ?? []).contains(where: { $0.title == title }) else { return }
        var k = (graph.groups?.count ?? 0) + 1
        while (graph.groups ?? []).contains(where: { $0.id == "group\(k)" }) { k += 1 }
        let color = a == .hires ? "purple" : a == .inpaint ? "green" : "yellow"
        let id = "group\(k)"
        mutate("Group \(title)", undoManager: undoManager ?? activeUndoManager) { g in
            guard let box = placeChain(a.chain, in: &g) else { return }
            g.groups = (g.groups ?? []) + [GraphGroup(id: id, title: title, x: box.minX, y: box.minY,
                                                      width: box.width, height: box.height, color: color)]
        }
    }

    /// Drops the labelled groups of chains that are no longer in the graph.
    func tidyChainGroups() {
        let stale = (graph.groups ?? []).filter { grp in
            guard let a = AddOn.allCases.first(where: { $0.groupTitle == grp.title }) else { return false }
            return !graph.nodes.contains { a.chain.contains($0.id) }
        }.map(\.id)
        guard !stale.isEmpty else { return }
        mutate("Remove Group", undoManager: activeUndoManager) { g in
            g.groups?.removeAll { stale.contains($0.id) }
            if g.groups?.isEmpty == true { g.groups = nil }
        }
    }

    /// A file for a Load node from Assets › Input or Output in the app's picker, or from anywhere through Finder;
    /// files from outside the input folder are copied into it first, so the workflow keeps working when they move.
    func chooseAsset(_ types: [UTType], title: String, message: String, asMask: Bool = false,
                     then: @escaping @MainActor (URL) -> Void) {
        presenter.chooseFile(
            title: title, message: message,
            roots: [MGFileRoot(title: "Input", url: library.input),
                    MGFileRoot(title: "Output", url: URL(fileURLWithPath: outputDirectory))],
            mode: .open(types: types, multiple: false),
            finder: { Self.chooseFile(types, message: message).map { [$0] } },
            onChoose: { urls in
                guard let url = urls.first else { return }
                if self.assetStore.relativePath(url) != nil { then(url) }
                else if let copy = self.importAssets([url], asMask: asMask).first { then(copy) }
            })
    }

    static func chooseFile(_ types: [UTType], message: String) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.message = message
        return panel.runModal() == .OK ? panel.url : nil
    }
}

extension AppModel {
    /// The Add to Workflow items, for the Graph menu, the canvas menu and the inspector.
    func addToWorkflowItems() -> [MGMenuItem] {
        AddOn.allCases.map { a in
            if isActive(a) && [.outpaint, .hires].contains(a) {
                return .button("Remove \(a.groupTitle ?? a.title)") { self.remove(a) }
            }
            return .button(a == .inpaint && isActive(a) ? "Edit Inpaint Mask…" : a.title, enabled: canAdd(a)) { self.add(a) }
        }
    }
}

extension MGMenuItem {
    /// Align and Distribute, enabled for how many nodes are selected.
    static func alignItems(count: Int, _ align: @escaping @MainActor (AlignMode) -> Void) -> [MGMenuItem] {
        AlignMode.allCases.flatMap { mode -> [MGMenuItem] in
            let item = MGMenuItem.button(mode.title, enabled: count >= mode.minimumCount) { align(mode) }
            return mode == .right || mode == .bottom ? [item, .divider] : [item]
        }
    }
}
