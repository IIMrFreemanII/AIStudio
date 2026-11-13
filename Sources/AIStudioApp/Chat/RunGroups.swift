import Agent
import Foundation
import LLMClient

/// Folding an agent run's work: a run (the replies between two of your messages) with tool calls shows its thinking,
/// the text between its steps and its tool calls as one summary row, opened into a timeline on a click; while it works,
/// as a small box of its steps. What needs you stays outside at full size (pinned), and so does the run's answer.
struct RunFolding {
    /// The runs and items opened (`ChatDocument.openRuns`).
    var open: Set<String>
    /// Tool results by call ID.
    var results: [String: ChatTurn]
    /// The agent is working: the last run is live.
    var streaming: Bool
    /// The call waiting for permission.
    var asking: String?

    /// Cards that stay outside the fold: questions, plans, widgets, session proposals, flows (they run in the
    /// background with their own card), and the call waiting for your permission.
    static let pinnedTools: Set<String> = ["ask_user", "exit_plan", "show_widget", "propose_session", "flow"]

    func pinned(_ call: ToolCall) -> Bool { Self.pinnedTools.contains(call.name) || call.id == asking }

    static func key(_ run: [ChatTurn]) -> String { "run:\(run[0].id.uuidString)" }

    /// Whether a run folds: it has a tool call.
    static func folds(_ run: [ChatTurn]) -> Bool { run.contains { !$0.toolCalls.isEmpty } }

    /// The run is still going: it's the last one and the agent is working, or one of its calls still runs.
    func isLive(_ run: [ChatTurn], last: Bool) -> Bool {
        guard last else { return false }
        return streaming || run.contains { $0.toolCalls.contains { results[$0.id]?.toolStatus == "running" } }
    }
}

/// A run's header row.
struct RunHeaderInfo {
    let key: String
    let summary: RunSummary
    let open: Bool
}

/// A live run's box: its steps so far.
struct LiveRunInfo {
    let key: String
    let summary: RunSummary
    let steps: [ChatTurn]
    /// The last step's text shows under the box (it may be the answer), not in it.
    let lastTextOutside: Bool
}

/// What a run did, for its header: "✓ Worked 2 min 14 s · 12 actions" over "read 5 · edited 2 +18 −4 · ran 3".
struct RunSummary: Equatable {
    enum Status: Equatable { case running, done, stopped, failed }

    var status: Status
    var started: Date
    /// From the first step to the last thing that happened (finished runs).
    var seconds: Double
    /// Tool calls, not counting to-do list updates.
    var actions: Int
    /// What kinds of things it did, in a fixed order ("read 5", "ran 3", "1 task").
    var kinds: [String]
    /// Edited files' lines added and removed.
    var added = 0
    var removed = 0
    /// Calls that failed or weren't allowed.
    var failed = 0

    static func of(_ steps: [ChatTurn], results: [String: ChatTurn], live: Bool) -> RunSummary {
        var reads = 0, searches = 0, ran = 0, tasks = 0, web = 0, sessions = 0, flows = 0, skills = 0, other = 0
        var edited: Set<String> = []
        var added = 0, removed = 0, failed = 0, actions = 0
        var end = steps.first?.date ?? Date()
        for s in steps {
            end = max(end, s.date.addingTimeInterval(s.stats?.seconds ?? 0))
            for call in s.toolCalls {
                let result = results[call.id]
                if let r = result { end = max(end, r.date) }
                if result?.toolStatus == "failed" || result?.toolStatus == "denied" { failed += 1 }
                // Bookkeeping and questions to you aren't work done (the questions show at full size anyway).
                if ["todo_write", "ask_user", "exit_plan", "show_widget"].contains(call.name) { continue }
                actions += 1
                switch call.name {
                case "read_file": reads += 1
                case "search", "list_files", "code_graph", "command_output": searches += 1
                case "edit_file", "write_file":
                    edited.insert((try? ToolArgs(json: call.arguments))?.string("path") ?? call.id)
                    if let d = result?.toolDetail {
                        let (a, r) = ToolCard.diffCounts(d)
                        added += a
                        removed += r
                    }
                case "run_command", "stop_command": ran += 1
                case "task": tasks += 1
                case "web_fetch", "web_search", "open_page": web += 1
                case "start_session", "send_to_session", "hand_off", "list_sessions", "propose_session": sessions += 1
                case "flow": flows += 1
                case "use_skill": skills += 1
                default: other += 1
                }
            }
        }
        func n(_ count: Int, _ one: String, _ many: String) -> String? {
            count == 0 ? nil : "\(count) \(count == 1 ? one : many)"
        }
        let kinds = [
            reads > 0 ? "read \(reads)" : nil,
            searches > 0 ? "searched \(searches)" : nil,
            edited.isEmpty ? nil : "edited \(edited.count)",
            ran > 0 ? "ran \(ran)" : nil,
            n(tasks, "task", "tasks"),
            web > 0 ? "web \(web)" : nil,
            n(sessions, "session", "sessions"),
            n(flows, "flow", "flows"),
            n(skills, "skill", "skills"),
            n(other, "other", "other"),
        ].compactMap { $0 }
        let last = steps.last
        let status: Status = live ? .running : last?.error == "Stopped" ? .stopped : last?.error != nil ? .failed : .done
        let started = steps.first?.date ?? Date()
        return RunSummary(status: status, started: started, seconds: max(0, end.timeIntervalSince(started)),
                          actions: actions, kinds: kinds, added: added, removed: removed, failed: failed)
    }

    /// "2 min 14 s", "8 s".
    static func duration(_ s: Double) -> String {
        s < 60 ? "\(max(1, Int(s.rounded()))) s" : "\(Int(s) / 60) min \(Int(s) % 60) s"
    }

    /// "0:42" for a running clock.
    static func clock(_ s: Double) -> String {
        let t = max(0, Int(s))
        return "\(t / 60):\(String(format: "%02d", t % 60))"
    }
}
