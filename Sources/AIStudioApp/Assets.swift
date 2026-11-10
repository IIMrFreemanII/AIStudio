import AppKit
import Foundation
import GraphRuntime
import AIStudioLibrary
import NodesStandard

/// The input folder (Assets › Input): importing files, Load-node paths relative to it, and files dropped on the graph.
extension AppModel {
    var assetStore: AssetStore { AssetStore(root: library.input, useTrash: !Self.isAutomatedRun) }

    func reloadAssets() { assets = assetStore.list() }

    /// Copies files into the input folder (masks into `masks/`); returns the copies.
    @discardableResult
    func importAssets(_ urls: [URL], asMask: Bool = false) -> [URL] {
        var out: [URL] = []
        for url in urls {
            do { out.append(try assetStore.importFile(url, asMask: asMask)) }
            catch { lastError = "Could not import \(url.lastPathComponent): \(error.localizedDescription)" }
        }
        reloadAssets()
        return out
    }

    /// What a Load node stores for a file: its path relative to the input folder, or the absolute path.
    func storedPath(_ url: URL) -> String {
        assetStore.relativePath(url) ?? url.standardizedFileURL.path
    }

    /// A stored Load path as a file on disk (relative ones are looked up in the input folder, then the working folder).
    func resolvedPath(_ stored: String) -> String {
        stored.isEmpty ? stored : StandardNodes.resolveInputPath(stored).path
    }

    func renameAsset(_ url: URL, to name: String) {
        do {
            let old = storedPath(url)
            let new = try assetStore.rename(url, to: name)
            // Load nodes in open tabs follow the renamed file.
            let newPath = storedPath(new)
            for doc in tabs {
                var g = doc.graph
                for n in g.nodes where AssetKind.allCases.map(\.loadNodeType).contains(n.type) {
                    if case .value(.string(let p))? = n.inputs["path"], p == old { try? g.setValue(.string(newPath), node: n.id, input: "path") }
                }
                if g != doc.graph { setGraph(g, in: doc, actionName: "Rename Asset", undoManager: doc.undoManager) }
            }
            reloadAssets()
        } catch { lastError = "Could not rename: \(error.localizedDescription)" }
    }

    func trashAsset(_ url: URL) {
        do { try assetStore.delete(url); reloadAssets() } catch { lastError = "Could not move to the Trash: \(error.localizedDescription)" }
    }

    /// The node types that can take a file of this kind.
    static func loadTypes(for kind: AssetKind) -> [String] {
        switch kind {
        case .image: return ["LoadImage", "LoadMask"]
        case .mask: return ["LoadMask", "LoadImage"]
        case .video: return ["LoadVideo"]
        case .audio: return ["LoadAudio"]
        }
    }

    /// Adds a Load node for an asset at `point` (graph coordinates) on the editing level, selecting it.
    @discardableResult
    func addLoadNode(for url: URL, at point: GraphPoint) -> String? {
        guard let kind = AssetStore.kind(of: url) else { return nil }
        let type = kind.loadNodeType
        addNode(type: type, at: point, undoManager: activeUndoManager)
        guard let id = selection.first, editorGraph[id]?.type == type else { return nil }
        setEditorLiteral(.string(storedPath(url)), node: id, input: "path", undoManager: activeUndoManager)
        return id
    }

    /// Files dropped on the graph: workflows (and outputs carrying a graph, unless ⌥ is held) open in a tab; media are
    /// imported into the input folder and either replace the file of the Load node they land on, or get a new one.
    func dropFiles(_ urls: [URL], at point: GraphPoint, onto node: String?, forceAsInput: Bool = false) {
        var offset = 0.0
        for url in urls {
            let ext = url.pathExtension.lowercased()
            if ext == "json" { open(url); continue }
            if !forceAsInput, node == nil, ext == "png", Self.embeddedGraph(in: url) != nil { open(url); continue }
            if !forceAsInput, node == nil, (AssetKind.videoExtensions.contains(ext) || AssetKind.audioExtensions.contains(ext)),
               history.contains(where: { $0.url == url.standardizedFileURL }) {
                open(url); continue  // our own video/sound: restore its graph
            }
            guard let kind = AssetStore.kind(of: url) else {
                lastError = "\(url.lastPathComponent) isn't an image, video or sound file."
                NSSound.beep()
                continue
            }
            let asMask = node.flatMap { editorGraph[$0]?.type } == "LoadMask" && kind == .image
            guard let copy = importAssets([url], asMask: asMask).first else { continue }
            if let node, let n = editorGraph[node], Self.loadTypes(for: kind).contains(n.type) {
                setEditorLiteral(.string(storedPath(copy)), node: node, input: "path", undoManager: activeUndoManager)
                selection = [node]
            } else {
                addLoadNode(for: copy, at: GraphPoint(x: point.x + offset, y: point.y + offset))
                offset += 40
            }
        }
    }

    /// Where the mask editor saves masks.
    var maskDirectory: URL { assetStore.masks }
}

extension AppModel {
    /// Shows a file (an input asset) in the results pane.
    func showFile(_ url: URL, kind: AssetKind) {
        let media: MediaKind = kind == .video ? .video : kind == .audio ? .audio : .image
        endCompare()
        show(HistoryItem(url: url, date: Date(), kind: media))
        showResultsPane = true
    }
}
