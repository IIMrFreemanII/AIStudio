import Agent
import LLMClient

/// The to-do list as the transcript has it: the last `todo_write` call on the shown branch.
extension ChatTranscript {
    private var lastTodoCall: (call: ToolCall, index: Int)? {
        let shown = turns
        for (i, t) in shown.enumerated().reversed() {
            if t.role == .assistant, let c = t.toolCalls.last(where: { $0.name == "todo_write" }) { return (c, i) }
        }
        return nil
    }

    /// The list the agent last wrote (empty when it never did).
    var todoList: [ToolContext.Todo] {
        lastTodoCall.map { TodoWriteTool.list(json: $0.call.arguments) } ?? []
    }

    /// The list was written after the user's last real message.
    var todoListIsCurrent: Bool {
        guard let last = lastTodoCall else { return false }
        let shown = turns
        guard let u = shown.lastIndex(where: { $0.role == .user && $0.summary != true && $0.sessionEvent == nil && $0.flowEvent == nil }) else { return true }
        return last.index > u
    }

    /// Done and total, while the list has something left to do.
    var todoProgress: (done: Int, total: Int)? {
        let l = todoList
        guard l.contains(where: { $0.status != "completed" }) else { return nil }
        return (l.filter { $0.status == "completed" }.count, l.count)
    }
}
