import LLMClient
import SwiftUI

/// Rail panel: saved chats, newest first. Click to open one in a tab; + for a new chat.
struct ChatsPanel: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Chats") {
                Button { model.newChat() } label: { Image(systemName: "square.and.pencil") }
                    .mgHelp("New chat (⌥⌘N)").accessibilityLabel("New chat")
            }
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 2) {
                    if model.chatSummaries.isEmpty && model.chats.isEmpty {
                        Text("Chats with language models on your GPU server. Add models in Servers › Language Models, from any Hugging Face link.")
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(12)
                    }
                    // Open chats that aren't saved yet (no messages).
                    ForEach(model.chats.filter { c in !model.chatSummaries.contains { $0.id == c.id } }) { c in
                        ChatRow(title: c.title, detail: c.transcript.modelID ?? "", date: nil,
                                open: true, active: c.id == model.activeChatID)
                            .onTapGesture { model.activateChat(c.id) }
                    }
                    ForEach(model.chatSummaries) { s in
                        let open = model.chats.contains { $0.id == s.id }
                        ChatRow(title: model.chats.first { $0.id == s.id }?.title ?? s.title, detail: s.modelID ?? "",
                                date: s.updated, open: open, active: s.id == model.activeChatID)
                            .onTapGesture { model.openChat(s.id) }
                            .mgContextMenu {
                                MGMenuItem.button("Open") { model.openChat(s.id) }
                                MGMenuItem.divider
                                MGMenuItem.button("Delete…", destructive: true) { confirmDelete(s) }
                            }
                    }
                }
                .padding(.vertical, 6)
            }
            .mgScrollIndicators()
        }
        .task { model.reloadChats() }
    }

    func confirmDelete(_ s: ChatSummary) {
        model.presenter.alert("Delete “\(s.title)”?", message: "The chat moves to the Trash.", actions: [
            .destructive("Delete") { model.deleteChat(s.id) },
            .cancel(),
        ])
    }
}

struct ChatRow: View {
    let title: String
    let detail: String
    let date: Date?
    let open: Bool
    let active: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "bubble.left").font(.system(size: 11))
                .foregroundStyle(active ? MG.accent : MG.secondaryLabel)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.mgCallout).lineLimit(1)
                HStack(spacing: 6) {
                    if !detail.isEmpty { Text(detail).lineLimit(1) }
                    if let date { Text(date, format: .relative(presentation: .named)) }
                }
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            Spacer(minLength: 0)
            if open { Circle().fill(active ? MG.accent : MG.tertiaryLabel).frame(width: 6, height: 6) }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
            .fill(active ? MG.selection : .clear))
        .padding(.horizontal, 6)
        .contentShape(Rectangle())
    }
}
