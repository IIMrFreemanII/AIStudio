import Agent
import Foundation
import LLMClient

/// The graph helper docked on workflow tabs: one chat per tab, made on first use and saved with the session.
extension AppModel {
    /// The tab's helper chat if it's open.
    func existingHelper(_ doc: WorkflowDocument) -> ChatDocument? {
        helperChats.first { $0.id == doc.helperChatID }
    }

    /// The tab's helper chat: open, loaded from its file, or new (with the last model used).
    @discardableResult
    func ensureHelper(_ doc: WorkflowDocument) -> ChatDocument {
        if let c = existingHelper(doc) { return c }
        let chat: ChatDocument
        if let id = doc.helperChatID, let t = chatStore.load(id) {
            chat = ChatDocument(transcript: t)
        } else {
            let k = defaultChatModel
            var t = ChatTranscript(serverID: k?.serverID, modelID: k?.modelID)
            t.graphHelper = true
            t.title = "Helper: " + doc.name
            t.permissionMode = PermissionMode.ask.rawValue
            chat = ChatDocument(transcript: t)
            doc.helperChatID = chat.id
        }
        chat.workflowID = doc.id
        helperChats.append(chat)
        return chat
    }

    /// The workflow a helper works on: its tab, else (opened from the Chats panel as a tab) the active workflow.
    func helperWorkflow(_ chat: ChatDocument) -> WorkflowDocument? {
        chat.workflowID.map { document(id: $0) } ?? active
    }

    /// Undoes the helper's last reply (its one undo step), if nothing changed the workflow since.
    func undoHelperEdit(_ chat: ChatDocument) {
        guard let edit = chat.lastHelperEdit, let wf = helperWorkflow(chat), wf.graph == edit.after else { return }
        if wf.undoManager.undoActionName == GraphHelper.undoName {
            wf.undoManager.undo()
        } else {
            setGraph(edit.before, in: wf, actionName: "Undo \(GraphHelper.undoName)", undoManager: wf.undoManager)
        }
        chat.lastHelperEdit = nil
    }

    /// Saves and closes a tab's helper (the tab is closing).
    func closeHelper(of doc: WorkflowDocument) {
        guard let chat = existingHelper(doc) else { return }
        chat.task?.cancel()
        saveChat(chat)
        helperChats.removeAll { $0 === chat }
    }

    /// The model new chats start with: the last one used (if still there), else the first installed one.
    var defaultChatModel: LLMKey? {
        lastChatModel.flatMap { llm.model($0) != nil ? $0 : nil }
            ?? llm.models.first.map { LLMKey(serverID: $0.server.id, modelID: $0.model.id) }
    }
}

extension AppModel {
    /// Saves the tab's helper conversation (it stays in Chats) and starts a new one.
    func startNewHelper(_ doc: WorkflowDocument) {
        closeHelper(of: doc)
        doc.helperChatID = nil
        ensureHelper(doc)
    }
}
