import Foundation

/// What can be in front of the main area: an open workflow or a recent chat.
enum FrontItem: Hashable {
    case workflow(UUID)
    case chat(UUID)
}

/// Switching between open workflows and recent chats without tabs: ⌃Tab to the last one used, ⇧⌘[ / ⇧⌘] and
/// ⌘1–⌘9 through the Window menu's list. Only the most recent chats stay loaded.
extension AppModel {
    /// How many chats stay loaded after the user leaves them (busy sessions stay loaded until they finish).
    static let recentChatLimit = 6

    /// Chat views kept built (the one in front and the last ones used), so switching back to one is instant.
    static let keptChatViews = 3

    /// The chats whose views stay built, in a stable order (a reorder would rebuild them).
    var keptChats: [ChatDocument] {
        var ids: [UUID] = activeChatID.map { [$0] } ?? []
        for case .chat(let id) in frontHistory where ids.count < Self.keptChatViews && !ids.contains(id) {
            if chats.contains(where: { $0.id == id && $0.isRecent }) { ids.append(id) }
        }
        return chats.filter { ids.contains($0.id) }
    }

    var frontItem: FrontItem { activeChatID.map { .chat($0) } ?? .workflow(activeTabID) }

    /// The Window menu's list: open workflows in order, then recent chats in the order they were opened.
    var openItems: [FrontItem] {
        tabs.map { .workflow($0.id) } + chats.filter(\.isRecent).map { .chat($0.id) }
    }

    func isOpen(_ item: FrontItem) -> Bool {
        switch item {
        case .workflow(let id): return tabs.contains { $0.id == id }
        case .chat(let id): return chats.contains { $0.id == id && $0.isRecent }
        }
    }

    func title(of item: FrontItem) -> String {
        switch item {
        case .workflow(let id): return tabs.first { $0.id == id }?.name ?? "Workflow"
        case .chat(let id): return chats.first { $0.id == id }?.title ?? "Chat"
        }
    }

    func show(_ item: FrontItem) {
        switch item {
        case .workflow(let id): activate(id)
        case .chat(let id): activateChat(id)
        }
    }

    /// ⌃Tab: back to what was in front before this.
    func switchToPrevious() {
        let current = frontItem
        if let item = frontHistory.first(where: { $0 != current && isOpen($0) }) { show(item) }
    }

    /// ⇧⌘] / ⇧⌘[: the next or previous item of `openItems`.
    func selectOpenItem(offset: Int) {
        let items = openItems
        guard items.count > 1, let i = items.firstIndex(of: frontItem) else { return }
        show(items[(i + offset + items.count) % items.count])
    }

    /// Records the item now in front, and lets chats that fell out of the recent few unload.
    func noteFront() {
        let item = frontItem
        guard frontHistory.first != item else { return }
        frontHistory.removeAll { $0 == item }
        frontHistory.insert(item, at: 0)
        if frontHistory.count > 40 { frontHistory.removeLast(frontHistory.count - 40) }
        if case .chat = item {
            // After the current change finishes: callers may still be setting the chat up.
            Task { @MainActor [weak self] in self?.trimRecentChats() }
        }
    }

    /// Keeps `recentChatLimit` recent chats (by last use); older ones leave the list and unload unless still busy.
    func trimRecentChats() {
        let order = frontHistory.compactMap { item -> UUID? in
            if case .chat(let id) = item { return id }
            return nil
        }
        let recent = chats.filter(\.isRecent)
        guard recent.count > Self.recentChatLimit else { return }
        let ranked = recent.sorted { (order.firstIndex(of: $0.id) ?? .max) < (order.firstIndex(of: $1.id) ?? .max) }
        for doc in ranked.dropFirst(Self.recentChatLimit) where doc.id != activeChatID {
            doc.isRecent = false
            unloadIfHidden(doc)
        }
    }

    /// Takes `item` out of the front, showing what was used before it (or the active workflow).
    func leaveFront(_ item: FrontItem) {
        if frontItem == item {
            if let before = frontHistory.first(where: { $0 != item && isOpen($0) }) { show(before) } else { activeChatID = nil }
        }
        frontHistory.removeAll { $0 == item }
    }
}
