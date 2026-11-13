import Foundation

/// A plan the agent proposed (an `exit_plan` call) and how it was answered.
public struct ChatPlan: Equatable, Sendable, Identifiable {
    public var id: String { callID }
    public var callID: String
    public var plan: String
    /// When the reply holding the call was written.
    public var date: Date
    /// "running" while waiting for an answer, "done" once answered (`detail` says how), else how it ended.
    public var status: String
    /// "Approved — Auto-accept Edits", "Keep planning: …", nil while waiting.
    public var detail: String?

    public var title: String { PlanFiles.title(plan) }

    public init(callID: String, plan: String, date: Date, status: String, detail: String?) {
        self.callID = callID
        self.plan = plan
        self.date = date
        self.status = status
        self.detail = detail
    }
}

extension ChatTranscript {
    /// Every plan on the shown branch, oldest first.
    public var plans: [ChatPlan] {
        let shown = turns
        var results: [String: ChatTurn] = [:]
        for t in shown where t.role == .tool { if let id = t.toolCallID { results[id] = t } }
        var out: [ChatPlan] = []
        for t in shown where t.role == .assistant {
            for c in t.toolCalls where c.name == "exit_plan" {
                guard let plan = PlanFiles.plan(fromArguments: c.arguments), !plan.isEmpty else { continue }
                let r = results[c.id]
                out.append(ChatPlan(callID: c.id, plan: plan, date: t.date, status: r?.toolStatus ?? "running",
                                    detail: r?.toolDetail))
            }
        }
        return out
    }
}

/// Plans saved as Markdown files in a chat's folder: `plans/2026-11-13 0840 Add a file explorer.md`.
public enum PlanFiles {
    /// The `plan` argument of an `exit_plan` call.
    public static func plan(fromArguments json: String) -> String? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj["plan"] as? String
    }

    /// The plan's first heading, else its first line, without Markdown marks; at most 60 characters.
    public static func title(_ plan: String) -> String {
        let lines = plan.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let line = lines.first { $0.hasPrefix("#") } ?? lines.first ?? ""
        var t = line.drop { $0 == "#" || $0 == " " }.replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "`", with: "").trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("Plan: ") { t = String(t.dropFirst(6)) }
        if t.isEmpty { return "Plan" }
        return t.count > 60 ? String(t.prefix(59)).trimmingCharacters(in: .whitespaces) + "…" : t
    }

    /// The file name for a plan proposed at `date`: the date and time, then the title made safe for a file name.
    public static func name(date: Date, plan: String, timeZone: TimeZone = .current) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.timeZone = timeZone
        f.dateFormat = "yyyy-MM-dd HHmm"
        let bad = CharacterSet(charactersIn: "/\\:*?\"<>|\n\r\t")
        let safe = title(plan).replacingOccurrences(of: "…", with: "")
            .components(separatedBy: bad).joined(separator: " ")
            .split(separator: " ").joined(separator: " ")
        let cut = safe.count > 50 ? String(safe.prefix(50)).trimmingCharacters(in: .whitespaces) : safe
        return "\(f.string(from: date)) \(cut.isEmpty ? "Plan" : cut).md"
    }

    /// The names for a chat's plans in order: the same name twice gets " 2", " 3", … (stable as plans are added).
    public static func names(_ plans: [ChatPlan], timeZone: TimeZone = .current) -> [String: String] {
        var used: [String: Int] = [:]
        var out: [String: String] = [:]
        for p in plans {
            let base = name(date: p.date, plan: p.plan, timeZone: timeZone)
            let n = (used[base] ?? 0) + 1
            used[base] = n
            out[p.callID] = n == 1 ? base : String(base.dropLast(3)) + " \(n).md"
        }
        return out
    }
}
