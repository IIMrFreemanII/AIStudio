import AppKit
import CharacterCore
import MovieCore
import NodesRemote
import NodesStandard
import SwiftUI
import UniformTypeIdentifiers

private func describe(_ error: Error) -> String {
    (error as? CustomStringConvertible)?.description ?? error.localizedDescription
}

/// The character's portraits: make or import them, and choose which one is the face in photos. The editor edits a
/// copy of the character while these functions write the stored one, so only the portrait fields are refreshed here.
struct CharacterLookSection: View {
    @Environment(AppModel.self) private var model
    @Binding var c: AppModel.Companion
    let isNew: Bool
    @State private var running = false
    @State private var progress = ""
    @State private var task: Task<Void, Never>?
    @State private var problem: String?

    private var server: String? { model.connections.gpuServers.first?.id }

    var body: some View {
        if isNew {
            VStack(alignment: .leading, spacing: 6) {
                Text("Look").font(.mgCallout).foregroundStyle(MG.secondaryLabel).padding(.horizontal, 10)
                Text("Save the character first, then make or import portraits.")
                    .font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).padding(.horizontal, 10)
            }
        } else {
            MGSection("Look") {
                thumbnails
                HStack(spacing: 8) {
                    Button("Make 4 Portraits") { generate() }
                        .buttonStyle(.mg(.bordered)).disabled(server == nil || running)
                        .mgHelp(server == nil ? "Add a GPU server in Servers first" : "Draw 4 portrait candidates on your GPU server")
                    Button("Import Artwork…") { chooseArtwork() }
                        .buttonStyle(.mg(.bordered)).disabled(running)
                        .mgHelp("Use a picture of a fictional adult as a portrait")
                    if running {
                        MGSpinner().controlSize(.small)
                        Text(progress).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                        Spacer(minLength: 0)
                        Button("Cancel") { task?.cancel() }
                            .buttonStyle(.mg(.bordered)).controlSize(.small)
                            .mgHelp("Stop making portraits (finished ones are kept)")
                    }
                }
            } footer: {
                VStack(alignment: .leading, spacing: 2) {
                    Text(footer)
                    if let problem { Text(problem).foregroundStyle(MG.destructive) }
                }
            }
            .task(id: running) {
                // Portraits are stored as each one finishes: show them as they arrive.
                while running, !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                    refresh()
                }
            }
        }
    }

    private var footer: String {
        var s = "Their face in photos comes from the chosen portrait."
        if let server, let each = model.photoEstimate(server: server, family: c.artModel.family) {
            s += " Making 4 costs about \(CostEstimate.format(4 * each))."
        }
        return s
    }

    @ViewBuilder private var thumbnails: some View {
        if c.portraits.isEmpty {
            Text("No portraits yet.").foregroundStyle(MG.secondaryLabel)
        } else {
            ScrollView(.horizontal) {
                HStack(spacing: 10) {
                    ForEach(c.portraits, id: \.self) { name in
                        PortraitThumb(url: model.characterStore.portraitsFolder(c.id).appendingPathComponent(name),
                                      selected: c.selectedPortrait == name,
                                      select: { model.selectPortrait(name, for: c.id); refresh() },
                                      delete: { confirmDelete(name) })
                    }
                }
                .padding(.vertical, 4)
            }
            .mgScrollIndicators()
        }
    }

    private func refresh() {
        guard let fresh = model.characterStore.load(c.id) else { return }
        if c.portraits != fresh.portraits { c.portraits = fresh.portraits }
        if c.selectedPortrait != fresh.selectedPortrait { c.selectedPortrait = fresh.selectedPortrait }
    }

    private func generate() {
        guard !running else { return }
        running = true
        problem = nil
        progress = "Starting…"
        let id = c.id
        task = Task { @MainActor in
            do {
                _ = try await model.generatePortraits(for: id, count: 4) { progress = $0 }
            } catch {
                if !Task.isCancelled, !(error is CancellationError) { problem = describe(error) }
            }
            refresh()
            running = false
        }
    }

    private func chooseArtwork() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.png, .jpeg, .heic, .webP]
        panel.message = "Choose a picture of your character"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let id = c.id
        model.presenter.alert("Import this picture?",
                              message: "Only use artwork of a fictional adult — not a photo of a real person.", actions: [
            MGAlertAction(label: "Import") {
                problem = nil
                do { _ = try model.importPortrait(url, for: id) } catch { problem = describe(error) }
                refresh()
            },
            .cancel(),
        ])
    }

    private func confirmDelete(_ name: String) {
        let id = c.id
        model.presenter.alert("Delete this portrait?", message: "The picture moves to the Trash.", actions: [
            .destructive("Delete") {
                model.deletePortrait(name, for: id)
                refresh()
            },
            .cancel(),
        ])
    }
}

/// One portrait candidate: the chosen one has an accent ring and a "Face" badge; a trash button shows on hover.
private struct PortraitThumb: View {
    let url: URL
    let selected: Bool
    let select: () -> Void
    let delete: () -> Void
    @State private var hover = false

    var body: some View {
        let shape = RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
        MGFileThumb(url: url)
            .frame(width: 96, height: 96)
            .background(MG.fill)
            .clipShape(shape)
            .overlay { if selected { shape.strokeBorder(MG.accent, lineWidth: 3) } }
            .overlay(alignment: .bottomLeading) {
                if selected {
                    Text("Face").font(.mgFootnote).foregroundStyle(MG.accentForeground)
                        .padding(.horizontal, 6).padding(.vertical, 2)
                        .background(Capsule().fill(MG.accent))
                        .padding(6)
                }
            }
            .overlay(alignment: .topTrailing) {
                if hover {
                    Button(action: delete) { Image(systemName: "trash") }
                        .accessibilityLabel("Delete Portrait")
                        .buttonStyle(.mg(.icon)).controlSize(.small)
                        .background(Circle().fill(.regularMaterial))
                        .padding(4)
                        .mgHelp("Delete this portrait")
                }
            }
            .contentShape(shape)
            .onTapGesture(perform: select)
            .onHover { hover = $0 }
            .mgHelp(selected ? "Their face in photos" : "Use this portrait for their face")
            .accessibilityAddTraits(.isButton)
            .accessibilityLabel(selected ? "Portrait, chosen as the face" : "Portrait")
    }
}

/// The image model (SDXL or Chroma, and a fine-tune of it), art style and LoRAs their photos use, and what one photo
/// may cost without asking.
struct CharacterPhotosSection: View {
    @Environment(AppModel.self) private var model
    @Binding var c: AppModel.Companion
    @State private var autoUSD = 0.05

    private var library: LinkLibrary? { model.store?.library }
    /// The library family of the character's image model.
    private var family: String { c.artModel.family.rawValue }
    private var familyName: String { LoRASources.familyName(family) }
    private var fineTunes: [LibraryEntry] { library?.models(for: family) ?? [] }
    private var loras: [LibraryEntry] { library?.loras(for: family) ?? [] }

    var body: some View {
        MGSection("Photos") {
            MGFormRow("Image model") {
                MGSegmentedPicker(selection: modelFamily,
                                  options: ArtModel.Family.allCases.map { ($0, $0 == .chroma ? "Chroma" : "SDXL") })
                    .frame(width: 200)
                    .mgHelp("SDXL is quick, keeps the portrait's face closely and has many fine-tunes; Chroma (Flux-based) follows descriptions and outfits more closely, but holds the face more loosely and costs about 10× more per photo")
            }
            MGFormRow("Style") {
                MGSegmentedPicker(selection: $c.artModel.style,
                                  options: ArtModel.Style.allCases.map { ($0, $0 == .anime ? "Anime" : "Photoreal") })
                    .frame(width: 200)
                    .mgHelp("Photoreal looks like a camera picture; anime like a drawing")
            }
            MGFormRow("Model") {
                MGPicker(selection: variant, options: variantOptions, maxWidth: 260)
                    .mgHelp("The \(familyName) fine-tune that draws their photos (from the Models library)")
            }
            ForEach(Array(c.loras.enumerated()), id: \.offset) { i, _ in loraRow(i) }
            if c.loras.count < CharacterArt.maxLoRAs {
                HStack {
                    Button("Add LoRA") { c.loras.append(CharacterLoRA(lora: loras.first?.id ?? "")) }
                        .buttonStyle(.mg(.bordered)).controlSize(.small).disabled(loras.isEmpty)
                        .mgHelp(loras.isEmpty ? "Add a \(familyName) LoRA in the Models library first"
                                : "Add a LoRA that shapes their photos (up to \(CharacterArt.maxLoRAs))")
                    Spacer(minLength: 0)
                }
            }
            MGFormRow("Ask before a photo over") {
                HStack(spacing: 10) {
                    MGSlider(value: Binding(get: { autoUSD }, set: { autoUSD = $0; model.characterAutoPhotoUSD = $0 }),
                             in: 0.01...0.50, step: 0.01).frame(width: 160)
                        .mgHelp("Photos that cost more than this wait for your OK in the chat")
                    Text(CostEstimate.format(autoUSD)).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                        .frame(width: 44, alignment: .trailing)
                }
            }
        } footer: {
            Text("LoRAs shape the look of their photos. The cost limit applies to all characters.")
        }
        .onAppear { autoUSD = min(max(model.characterAutoPhotoUSD, 0.01), 0.50) }
    }

    /// Switching families drops the fine-tune and LoRAs: they only fit the model they were made for.
    private var modelFamily: Binding<ArtModel.Family> {
        Binding(get: { c.artModel.family }, set: { f in
            guard f != c.artModel.family else { return }
            c.artModel.family = f
            c.artModel.variant = nil
            c.loras = []
        })
    }

    private var variant: Binding<String> {
        Binding(get: { c.artModel.variant ?? "" }, set: { c.artModel.variant = $0.isEmpty ? nil : $0 })
    }

    private var variantOptions: [(value: String, title: String)] {
        var o: [(value: String, title: String)] = [("", RemoteNodes.baseTitle(family: family))]
        o += fineTunes.map { ($0.id, $0.title) }
        if let v = c.artModel.variant, !o.contains(where: { $0.value == v }) { o.append((v, "Missing: \(v)")) }
        return o
    }

    private func loraOptions(current: String) -> [(value: String, title: String)] {
        var o = loras.map { (value: $0.id, title: $0.title) }
        if !o.contains(where: { $0.value == current }) {
            o.append((current, current.isEmpty ? "Pick a LoRA" : "Missing: \(current)"))
        }
        return o
    }

    @ViewBuilder private func loraRow(_ i: Int) -> some View {
        if c.loras.indices.contains(i) {
            HStack(spacing: 8) {
                MGPicker(selection: Binding(get: { c.loras.indices.contains(i) ? c.loras[i].lora : "" },
                                            set: { if c.loras.indices.contains(i) { c.loras[i].lora = $0 } }),
                         options: loraOptions(current: c.loras[i].lora), placeholder: "Pick a LoRA", maxWidth: 170)
                    .mgHelp("A \(familyName) LoRA from the Models library")
                MGSlider(value: Binding(get: { c.loras.indices.contains(i) ? c.loras[i].strength : 0 },
                                        set: { if c.loras.indices.contains(i) { c.loras[i].strength = $0 } }),
                         in: -1...2, step: 0.05).frame(width: 100)
                    .mgHelp("How strongly it shapes the picture, −1 to 2")
                Text(String(format: "%.2f", c.loras[i].strength)).monospacedDigit().foregroundStyle(MG.secondaryLabel)
                    .frame(width: 38, alignment: .trailing)
                Button { if c.loras.indices.contains(i) { c.loras.remove(at: i) } } label: { Image(systemName: "trash") }
                    .accessibilityLabel("Remove LoRA")
                    .buttonStyle(.mg(.icon)).controlSize(.small)
                    .mgHelp("Remove this LoRA")
            }
        }
    }
}

/// The photos a character has sent, newest first; a click opens one.
struct CharacterAlbumGrid: View {
    @Environment(AppModel.self) private var model
    let characterID: UUID
    let items: [AlbumItem]

    var body: some View {
        if items.isEmpty {
            Text("Photos they send appear here.").foregroundStyle(MG.secondaryLabel)
        } else {
            let folder = model.characterStore.albumFolder(characterID)
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 130), spacing: 10)], alignment: .leading, spacing: 12) {
                ForEach(items.sorted { $0.date > $1.date }) { item in
                    let url = folder.appendingPathComponent(item.file)
                    VStack(spacing: 4) {
                        Group {
                            if item.kind == .photo { MGFileThumb(url: url) }
                            else { Image(systemName: "film").foregroundStyle(MG.tertiaryLabel) }
                        }
                        .frame(width: 110, height: 110)
                        .background(MG.fill)
                        .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
                        Text(item.date.formatted(date: .abbreviated, time: .omitted))
                            .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { NSWorkspace.shared.open(url) }
                    .mgHelp("Open this photo")
                    .accessibilityAddTraits(.isButton)
                    .accessibilityLabel("Photo from \(item.date.formatted(date: .abbreviated, time: .omitted))")
                }
            }
            .padding(.vertical, 4)
        }
    }
}
