import AppKit
import GraphRuntime
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
        case .lora: return "LoRA…"
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

    func canAdd(_ a: AddOn) -> Bool {
        switch a {
        case .reference: return firstNode(ofType: "TextEncode") != nil && hasImageSampler
        case .inpaint: return referenceImageNode != nil && hasImageSampler
        case .outpaint: return referenceImageNode != nil && hasImageSampler && !isOutpainting
        case .hires: return hasImageSampler && firstNode(ofType: "SaveImage") != nil
        case .lora: return hasImageSampler || hasVideoSampler
        case .firstFrame: return hasVideoSampler
        }
    }

    func isActive(_ a: AddOn) -> Bool {
        switch a {
        case .reference: return referenceImageNode != nil
        case .inpaint: return maskPath != nil
        case .outpaint: return isOutpainting
        case .hires: return isHiResEnabled
        case .lora: return !simpleLoRAs.isEmpty
        case .firstFrame: return firstFrameNode != nil
        }
    }

    // MARK: doing it

    /// Runs an add-on; the ones that need a file or a mask ask for it first.
    func add(_ a: AddOn) {
        let undo = activeUndoManager
        switch a {
        case .reference:
            if let url = Self.chooseFile([.image], message: "Choose the image to edit, redraw or use as a reference.") {
                setReferenceImage(url, undoManager: undo)
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
            if let url = Self.chooseFile([UTType(filenameExtension: "safetensors") ?? .data],
                                         message: "Choose a LoRA (.safetensors) made for this workflow's model.") {
                addLoRA(url, undoManager: undo)
            }
        case .firstFrame:
            if let url = Self.chooseFile([.image], message: "Choose the image the clip starts from.") {
                setFirstFrame(url, undoManager: undo)
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
        case .lora: for l in simpleLoRAs { removeLoRA(l.id, undoManager: undo) }
        case .firstFrame: clearFirstFrame(undoManager: undo)
        }
        tidyChainGroups()
    }

    /// Wraps a chain that was just spliced in in a group titled after it (once).
    func groupAddedChain(_ a: AddOn, undoManager: UndoManager? = nil) {
        guard let title = a.groupTitle, !(graph.groups ?? []).contains(where: { $0.title == title }) else { return }
        let frames = graph.nodes.filter { a.chain.contains($0.id) }.compactMap { n in
            schema(for: n).map { NodeLayout.frame(n, $0) }
        }
        guard let first = frames.first else { return }
        let box = frames.dropFirst().reduce(first) { $0.union($1) }.insetBy(dx: -24, dy: -24)
        var k = (graph.groups?.count ?? 0) + 1
        while (graph.groups ?? []).contains(where: { $0.id == "group\(k)" }) { k += 1 }
        let color = a == .hires ? "purple" : a == .inpaint ? "green" : "yellow"
        let group = GraphGroup(id: "group\(k)", title: title, x: box.minX, y: box.minY - 30,
                               width: box.width, height: box.height + 30, color: color)
        mutate("Group \(title)", undoManager: undoManager ?? activeUndoManager) { g in g.groups = (g.groups ?? []) + [group] }
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

    static func chooseFile(_ types: [UTType], message: String) -> URL? {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = types
        panel.message = message
        return panel.runModal() == .OK ? panel.url : nil
    }
}

/// The Add to Workflow items, for menus and context menus.
struct AddToWorkflowItems: View {
    let model: AppModel

    var body: some View {
        ForEach(AddOn.allCases, id: \.self) { a in
            if model.isActive(a) && [.outpaint, .hires].contains(a) {
                Button("Remove \(a.groupTitle ?? a.title)") { model.remove(a) }
            } else {
                Button(a == .inpaint && model.isActive(a) ? "Edit Inpaint Mask…" : a.title) { model.add(a) }
                    .disabled(!model.canAdd(a))
            }
        }
    }
}
