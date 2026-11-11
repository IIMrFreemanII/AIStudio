import SwiftUI

/// The graph helper docked next to the graph: the active tab's helper chat, with a compact header.
struct HelperPanel: View {
    @Environment(AppModel.self) private var model
    let doc: WorkflowDocument

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 6) {
                Image(systemName: "wand.and.sparkles").foregroundStyle(MG.accent)
                Text("Graph Helper").font(.mgHeadline)
                Spacer()
                if let chat = model.existingHelper(doc), !chat.transcript.turns.isEmpty, !chat.isStreaming {
                    Button { model.startNewHelper(doc) } label: { Image(systemName: "square.and.pencil") }
                        .buttonStyle(.mg(.icon))
                        .mgHelp("Start over with a new helper conversation (this one stays in Chats)")
                }
                Button { model.showHelper = false } label: { Image(systemName: "xmark") }
                    .buttonStyle(.mg(.icon))
                    .mgHelp("Hide the graph helper (⇧⌘H)")
            }
            .padding(.horizontal, 10)
            .frame(height: 34)
            MGDivider()
            if let chat = model.existingHelper(doc) {
                ChatView(doc: chat, compact: true).id(chat.id)
            } else {
                Spacer()
            }
        }
        .background(MG.windowBackground)
        .task(id: doc.id) { model.ensureHelper(doc) }
    }
}
