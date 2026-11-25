import CharacterCore
import SwiftUI

/// A character chat's items in the chat header: who, the scene, and how they get along.
struct CharacterHeaderItems: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument

    var body: some View {
        let people = model.characters(of: doc)
        let store = model.characterStore
        let scene = model.scene(of: doc)
        MGMenuButton(kind: .bordered) {
            if people.count == 1, let first = people.first {
                MGMenuItem.button("Edit Character…", icon: "pencil") {
                    model.editingCharacter = CharacterEdit(character: first, isNew: false)
                }
                MGMenuItem.button("Memory…", icon: "brain") { model.memoryCharacter = CharacterRef(id: first.id) }
            } else if !people.isEmpty {
                for c in people {
                    MGMenuItem.header(c.name)
                    MGMenuItem.button("Edit \(c.name)…", icon: "pencil") {
                        model.editingCharacter = CharacterEdit(character: c, isNew: false)
                    }
                    MGMenuItem.button("\(c.name)'s Memory…", icon: "brain") { model.memoryCharacter = CharacterRef(id: c.id) }
                }
            } else {
                MGMenuItem.header("This character was deleted")
            }
        } label: {
            HStack(spacing: 6) {
                ForEach(people.prefix(3)) { CharacterAvatar($0, model: model, size: 18) }
                Text(people.isEmpty ? "Deleted character" : ListFormatter.localizedString(byJoining: people.map(\.name))).lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
            }
        }
        .mgHelp("Who you're talking to: edit them, or see what they remember")
        MGMenuButton(kind: .bordered) {
            for s in store.scenarios() {
                MGMenuItem.button(s.title, checked: s.id == doc.transcript.sceneID) { model.setScene(s.id, for: doc) }
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "sparkles").foregroundStyle(MG.secondaryLabel)
                Text(scene?.title ?? "No scene").lineLimit(1)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .bold)).foregroundStyle(MG.secondaryLabel)
            }
        }
        .mgHelp("Where and when this chat happens; it shapes their replies")
        if people.count == 1, let first = people.first {
            let r = store.memory(first.id).relationship
            Text("\(r.stage.label) · \(r.affection)♥")
                .font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                .mgHelp("How close you two are (change it in Memory…)")
        }
    }
}
