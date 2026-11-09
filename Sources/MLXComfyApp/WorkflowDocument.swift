import AppKit
import Foundation
import GraphRuntime
import Observation

/// One output a node produced in a run: shown on the node, in the results pane and in the viewer.
struct NodeResult: Identifiable {
    let id = UUID()
    let node: String
    let kind: MediaKind
    let image: CGImage?
    let url: URL?
    var isPreview = false
    var date = Date()
}

/// One open workflow tab: its graph, editor state, undo history and the results of the runs it queued.
@MainActor @Observable
final class WorkflowDocument: Identifiable {
    let id: UUID
    var name: String
    /// The workflow file it was opened from or saved to (nil: never saved).
    var fileURL: URL?
    /// The preset it started from, if any.
    var presetID: String?

    var graph: Graph
    /// The graph as last saved (or as the tab was created); `isDirty` compares against it.
    var cleanGraph: Graph
    var selection: Set<String> = []
    /// Subgraph instances entered in the editor, outermost first (empty = the top-level graph).
    var editingPath: [String] = []
    /// The group selected in the editor, if any (at the current editing level).
    var selectedGroup: String?
    /// Each tab has its own undo history.
    @ObservationIgnored let undoManager = UndoManager()

    // MARK: run results

    var nodeStates: [String: NodeRunState] = [:]
    /// The live step preview of the run in progress, and the node sending it.
    var previewImage: CGImage?
    var previewNode: String?
    /// The latest result of each output node (flattened node id).
    var nodeResults: [String: NodeResult] = [:]
    /// Every result this tab produced, oldest first (the viewer steps through them).
    var outputs: [NodeResult] = []
    /// The result shown in the results pane and viewer.
    var displayedImage: CGImage?
    var displayedURL: URL?
    var displayedMediaURL: URL?
    var displayedMediaKind: MediaKind = .image
    /// A/B compare: the image shown against the displayed one.
    var compareURL: URL?
    var compareImage: CGImage?

    // MARK: editor & session

    /// Zoom and pan of the graph editor, kept while the tab is in the background.
    @ObservationIgnored var viewport: (zoom: CGFloat, pan: CGSize)?
    /// The graph last written as this tab's draft.
    @ObservationIgnored var lastDraft: Graph?

    var isDirty: Bool { graph != cleanGraph }
    var isUntitled: Bool { fileURL == nil }

    init(id: UUID = UUID(), name: String, graph: Graph, fileURL: URL? = nil, presetID: String? = nil,
         cleanGraph: Graph? = nil) {
        self.id = id
        self.name = name
        self.graph = graph
        self.cleanGraph = cleanGraph ?? graph
        self.fileURL = fileURL
        self.presetID = presetID
    }

    func markSaved(to url: URL) {
        fileURL = url
        name = url.deletingPathExtension().lastPathComponent
        cleanGraph = graph
    }

    /// Records a finished run's outputs.
    func record(_ outputs: [EngineOutput]) {
        for out in outputs {
            let r = NodeResult(node: out.node, kind: out.kind, image: out.image, url: out.url, isPreview: out.isPreview)
            nodeResults[out.node] = r
            self.outputs.append(r)
        }
        if self.outputs.count > 200 { self.outputs.removeFirst(self.outputs.count - 200) }
        if let out = outputs.last {
            displayedImage = out.image
            displayedURL = out.url
            displayedMediaURL = out.kind == .image ? nil : out.url
            displayedMediaKind = out.kind
        }
    }
}
