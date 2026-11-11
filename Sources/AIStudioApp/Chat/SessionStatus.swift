import AppKit
import Foundation
import LLMClient
import UserNotifications

/// What a session is doing, for its tab, the sidebar and the dock badge.
enum SessionStatus: String, Equatable {
    case working, waiting, failed, unread, idle
}

extension ChatTranscript {
    /// For sessions that aren't loaded: failed / unread / idle.
    var savedStatus: SessionStatus {
        if let e = turns.last?.error, e != "Stopped" { return .failed }
        if updated > (lastViewed ?? .distantPast) { return .unread }
        return .idle
    }
}

extension ChatDocument {
    /// Waiting for the user (a permission, question or plan card), working (a reply or a summary in progress), failed,
    /// unread (changed since it was last in front) or idle.
    var status: SessionStatus {
        if permission != nil || question != nil || plan != nil { return .waiting }
        if isStreaming || compacting { return .working }
        return transcript.savedStatus
    }

    /// A reply is streaming, the user is asked something, or background commands are running.
    var isBusy: Bool {
        isStreaming || permission != nil || question != nil || plan != nil || !runningJobs.isEmpty
    }

    /// Short text of what the session is waiting for.
    var pendingText: String? {
        if let p = permission { return [p.preview.title, p.preview.detail].compactMap { $0 }.joined(separator: ": ") }
        if let q = question { return q.questions.first?.question ?? "A question" }
        if plan != nil { return "A plan to approve" }
        return nil
    }
}

extension AppModel {
    /// A session's status as shown: the live one when it's loaded, else the saved one. The one in front is never unread.
    func status(of doc: ChatDocument) -> SessionStatus {
        let s = doc.status
        return s == .unread && isInFront(doc) ? .idle : s
    }

    func status(of summary: ChatSummary) -> SessionStatus {
        if let doc = (chats + helperChats).first(where: { $0.id == summary.id }) { return status(of: doc) }
        return summary.status
    }

    /// The session is the chat tab in front, or the helper docked next to the shown workflow.
    func isInFront(_ doc: ChatDocument) -> Bool {
        if activeChatID == doc.id { return true }
        return activeChatID == nil && showHelper && existingHelper(active) === doc
    }

    /// Marks the session in front as viewed.
    func markFrontViewed() {
        if let doc = activeChat { doc.transcript.lastViewed = Date() }
        else if showHelper, let h = existingHelper(active) { h.transcript.lastViewed = Date() }
    }

    var notifySessions: Bool { UserDefaults.standard.object(forKey: "notifySessions") as? Bool ?? true }

    /// Models of every session that is working or waiting: kept warm.
    var workingModels: Set<LLMKey> {
        Set((chats + helperChats).filter { $0.status == .working || $0.status == .waiting }.compactMap(\.modelKey))
    }

    /// A session's pending card or streaming state changed: the dock badge, and a notification when it's not in front
    /// and now needs the user (or, with `finished`, has finished).
    func sessionChanged(_ doc: ChatDocument, finished: Bool = false) {
        let front = isInFront(doc)
        if front { doc.transcript.lastViewed = Date() }
        if !Self.isAutomatedRun {
            let n = (chats + helperChats).filter { $0.status == .waiting }.count
            NSApp?.dockTile.badgeLabel = n > 0 ? "\(n)" : nil
        }
        guard notifySessions, !Self.isAutomatedRun, !(front && NSApp?.isActive == true) else { return }
        if doc.status == .waiting, let text = doc.pendingText {
            SessionNotifier.post(id: doc.id, title: "\u{201C}\(doc.title)\u{201D} needs you", body: text)
        } else if finished {
            let last = doc.transcript.turns.last
            if last?.error == "Stopped" { return }
            let reply = doc.transcript.turns.last { $0.role == .assistant && !$0.text.isEmpty }?.text
            let body = last?.error ?? String((reply ?? "Done.").prefix(120))
            SessionNotifier.post(id: doc.id, title: "\u{201C}\(doc.title)\u{201D} finished", body: body)
        }
    }
}

/// Posts the session notifications; clicking one opens the session.
@MainActor
enum SessionNotifier {
    private static let delegate = Delegate()
    private static var configured = false

    static func post(id: UUID, title: String, body: String) {
        guard Bundle.main.bundleIdentifier != nil else { return }
        let center = UNUserNotificationCenter.current()
        if !configured {
            configured = true
            center.delegate = delegate
        }
        let sessionID = id.uuidString
        Task {
            // Asks the first time only.
            guard (try? await center.requestAuthorization(options: [.alert, .sound])) == true else { return }
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            content.userInfo = ["session": sessionID]
            try? await center.add(UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil))
        }
    }

    private final class Delegate: NSObject, UNUserNotificationCenterDelegate {
        nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification,
                                                withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
            completionHandler([.banner, .sound])
        }

        nonisolated func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse,
                                                withCompletionHandler completionHandler: @escaping () -> Void) {
            let s = response.notification.request.content.userInfo["session"] as? String
            Task { @MainActor in
                if let s, let id = UUID(uuidString: s) {
                    NSApp.activate(ignoringOtherApps: true)
                    AppDelegate.model?.openChat(id)
                }
            }
            completionHandler()
        }
    }
}
