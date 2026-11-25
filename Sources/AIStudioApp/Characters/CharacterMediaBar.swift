import CharacterCore
import LLMClient
import MovieCore
import SwiftUI

/// Above the composer of a character chat: the media waiting for the user's OK (videos and video messages always,
/// photos and voice messages over the automatic limit), one row each with Make It (Take It for a photo) and Skip.
struct CharacterMediaBar: View {
    let doc: ChatDocument
    @Environment(AppModel.self) private var model

    private struct Item: Identifiable {
        var turn: UUID
        var media: ChatMedia
        var name: String
        var id: UUID { media.id }
    }

    private var items: [Item] {
        doc.transcript.turns.flatMap { t in
            (t.media ?? []).filter { $0.status == .waiting }.map { m in
                Item(turn: t.id, media: m, name: t.speaker.flatMap { model.character($0)?.name } ?? "Your character")
            }
        }
    }

    var body: some View {
        let waiting = items
        if !waiting.isEmpty {
            VStack(spacing: 0) {
                MGDivider()
                ForEach(waiting) { item in
                    HStack(spacing: 8) {
                        let kind = CharacterCore.MediaKind(rawValue: item.media.kind) ?? .photo
                        let l = CharacterDisplay.label(kind)
                        Text("\(l.icon) \(item.name) wants to send \(kind == .photo ? "a photo" : "a \(l.name.lowercased())")"
                             + (item.media.estimateUSD.map { " · about \(CostEstimate.format($0))" } ?? ""))
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            .lineLimit(1)
                        Spacer(minLength: 8)
                        Button("Skip") { model.skipMedia(doc, turnID: item.turn, mediaID: item.id) }
                            .buttonStyle(.mg(.bordered))
                        Button(kind == .photo ? "Take It" : "Make It") {
                            model.approveMedia(doc, turnID: item.turn, mediaID: item.id)
                        }
                        .buttonStyle(.mg(.prominent))
                    }
                    .padding(.horizontal, 14).padding(.vertical, 6)
                }
            }
            .background(MG.accent.opacity(0.06))
            .mgTestAnchor("character-media-bar")
        }
    }
}
