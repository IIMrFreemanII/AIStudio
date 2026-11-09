import AppKit
import GraphRuntime
import MLXComfyLibrary
import SwiftUI

/// ⌘N: every preset as a card, grouped by what it makes. Presets whose model isn't installed are dimmed, with a link
/// to the Models window.
struct PresetGallery: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    @Environment(\.openWindow) private var openWindow
    @State private var query = ""

    var body: some View {
        let installed = model.installedModelIDs
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        let matches = model.presets.filter {
            q.isEmpty || $0.name.lowercased().contains(q) || $0.modelName.lowercased().contains(q)
                || $0.summary.lowercased().contains(q)
        }
        VStack(spacing: 0) {
            HStack {
                Text("New Workflow").font(.mgTitle2)
                Spacer()
                MGSearchField(prompt: "Search presets", text: $query).frame(width: 240)
            }
            .padding(18)
            MGDivider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    ForEach(Preset.Category.allCases) { cat in
                        let items = matches.filter { $0.category == cat }
                        if !items.isEmpty {
                            VStack(alignment: .leading, spacing: 10) {
                                HStack(spacing: 8) {
                                    MGKindBadge(color: cat.hue, symbol: cat.symbol, size: 18)
                                    Text(cat.title).font(.mgHeadline)
                                }
                                LazyVGrid(columns: [GridItem(.adaptive(minimum: 190, maximum: 230), spacing: 12)],
                                          alignment: .leading, spacing: 12) {
                                    ForEach(items) { p in
                                        PresetCard(preset: p, available: p.isAvailable(installed: installed)) {
                                            model.openPreset(p)
                                            dismiss()
                                        } getModel: {
                                            openWindow(id: "models")
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if matches.isEmpty {
                        Text("No presets match.").foregroundStyle(MG.secondaryLabel).frame(maxWidth: .infinity)
                    }
                }
                .padding(18)
            }
            .mgScrollIndicators()
            .background(MG.groupedBackground)
            MGDivider()
            HStack {
                Button("Blank Text to Image (⌘T)") { model.newTab(); dismiss() }
                Button("Open Workflow…") {
                    dismiss()
                    AppCommands.openPanel(model: model)
                }
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }.keyboardShortcut(.cancelAction)
            }
            .padding(14)
        }
        .frame(width: 820, height: 640)
    }
}

struct PresetCard: View {
    let preset: Preset
    let available: Bool
    let open: () -> Void
    let getModel: () -> Void
    @State private var hovering = false
    @State private var thumbnail: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ZStack {
                RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(
                    (available ? preset.category.hue : MG.hueGray).opacity(0.14))
                if let thumbnail {
                    Image(nsImage: thumbnail).resizable().aspectRatio(contentMode: .fill)
                        .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
                } else {
                    Image(systemName: preset.symbol).font(.system(size: 34, weight: .light))
                        .foregroundStyle(available ? preset.category.hue : MG.secondaryLabel)
                }
            }
            .frame(height: 104)
            .clipped()
            VStack(alignment: .leading, spacing: 2) {
                Text(preset.name).font(.mgCallout.weight(.semibold)).lineLimit(1)
                Text(preset.modelName).font(.mgSubheadline).foregroundStyle(MG.secondaryLabel).lineLimit(1)
                Text(preset.summary).font(.mgFootnote).foregroundStyle(MG.secondaryLabel).lineLimit(2)
                    .frame(height: 28, alignment: .top)
            }
            if !available {
                Button("Get Model…", action: getModel).controlSize(.small)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(hovering && available ? MG.card : MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous)
            .strokeBorder(hovering && available ? MG.accent : MG.separator, lineWidth: hovering && available ? 2 : 0.5))
        .mgShadow(hovering && available ? MGShadow.control : MGShadowSpec(x: 0, y: 0, blur: 0, color: .clear))
        .opacity(available ? 1 : 0.6)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { if available { open() } else { NSSound.beep() } }
        .mgHelp(available ? preset.summary : "Needs \(preset.modelName) — get it in the Models window")
        .task(id: preset.id) {
            if let url = PresetCatalog.thumbnailURL(preset) { thumbnail = NSImage(contentsOf: url) }
        }
    }
}

/// File › Save as Preset…: name, category and a short description; the latest result becomes the card's picture.
struct SavePresetSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.mgDismiss) private var dismiss
    @State private var name = ""
    @State private var category: Preset.Category = .image
    @State private var summary = ""

    var body: some View {
        MGForm {
            Text("Save as Preset").font(.mgTitle3)
            MGSection {
                MGFormRow("Name") { TextField("Name", text: $name).frame(maxWidth: 240) }
                MGFormRow("Category") {
                    MGPicker(selection: $category, options: Preset.Category.allCases.map { ($0, $0.title) })
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Description").foregroundStyle(MG.label)
                    TextField("What it makes", text: $summary, axis: .vertical).lineLimit(2...4)
                }
            } footer: {
                Text("Saved to \(model.library.presets.path). Model folders and the output folder are stored by name, so the preset works on another Mac with the same models.")
            }
        }
        .frame(width: 460, height: 360)
        .safeAreaInset(edge: .bottom) {
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save Preset") {
                    model.saveAsPreset(name: name, category: category, summary: summary)
                    dismiss()
                }
                .buttonStyle(.mg(.prominent))
                .keyboardShortcut(.defaultAction)
                .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding(14)
        }
        .onAppear {
            name = model.active.name
            category = Self.category(of: model.graph)
        }
    }

    static func category(of g: Graph) -> Preset.Category {
        if g.nodes.contains(where: { $0.type == "VideoSampler" }) { return .video }
        if g.nodes.contains(where: { $0.type == "TextToAudio" }) { return .sound }
        if g.nodes.contains(where: { $0.type == "TextToMusic" }) { return .music }
        if g.nodes.contains(where: { $0.type == "TextToSpeech" }) { return .speech }
        return .image
    }
}

/// The mask editor for one Load Image node; the painted mask is wired in as inpainting.
struct MaskEditorSheet: View {
    @Environment(AppModel.self) private var model
    let target: MaskEditorTarget

    var body: some View {
        MaskEditorView(imageURL: URL(fileURLWithPath: target.imagePath),
                       existingMask: model.maskPath.map { URL(fileURLWithPath: $0) },
                       saveDirectory: model.maskDirectory) { url in
            model.setMask(url, undoManager: model.activeUndoManager)
            model.groupAddedChain(.inpaint)
            model.reloadAssets()
        }
    }
}
