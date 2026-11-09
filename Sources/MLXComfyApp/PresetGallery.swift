import AppKit
import GraphRuntime
import MLXComfyLibrary
import SwiftUI

/// ⌘N: every preset as a card, grouped by what it makes. Presets whose model isn't installed are dimmed, with a link
/// to the Models window.
struct PresetGallery: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
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
                Text("New Workflow").font(.title2.weight(.semibold))
                Spacer()
                TextField("Search presets", text: $query).textFieldStyle(.roundedBorder).frame(width: 240)
            }
            .padding(18)
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    ForEach(Preset.Category.allCases) { cat in
                        let items = matches.filter { $0.category == cat }
                        if !items.isEmpty {
                            VStack(alignment: .leading, spacing: 10) {
                                Label(cat.title, systemImage: cat.symbol).font(.headline)
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
                        Text("No presets match.").foregroundStyle(.secondary).frame(maxWidth: .infinity)
                    }
                }
                .padding(18)
            }
            Divider()
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
                RoundedRectangle(cornerRadius: 8).fill(LinearGradient(
                    colors: [Color.accentColor.opacity(available ? 0.35 : 0.12), Color.accentColor.opacity(0.06)],
                    startPoint: .topLeading, endPoint: .bottomTrailing))
                if let thumbnail {
                    Image(nsImage: thumbnail).resizable().aspectRatio(contentMode: .fill)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Image(systemName: preset.symbol).font(.system(size: 34, weight: .light))
                        .foregroundStyle(available ? Color.accentColor : .secondary)
                }
            }
            .frame(height: 104)
            .clipped()
            VStack(alignment: .leading, spacing: 2) {
                Text(preset.name).font(.callout.weight(.semibold)).lineLimit(1)
                Text(preset.modelName).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                Text(preset.summary).font(.caption2).foregroundStyle(.secondary).lineLimit(2)
                    .frame(height: 28, alignment: .top)
            }
            if !available {
                Button("Get Model…", action: getModel).controlSize(.small)
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(hovering && available ? Color.accentColor : Color.primary.opacity(0.1),
                                                                 lineWidth: hovering && available ? 2 : 1))
        .opacity(available ? 1 : 0.6)
        .contentShape(Rectangle())
        .onHover { hovering = $0 }
        .onTapGesture { if available { open() } else { NSSound.beep() } }
        .help(available ? preset.summary : "Needs \(preset.modelName) — get it in the Models window")
        .task(id: preset.id) {
            if let url = PresetCatalog.thumbnailURL(preset) { thumbnail = NSImage(contentsOf: url) }
        }
    }
}

/// File › Save as Preset…: name, category and a short description; the latest result becomes the card's picture.
struct SavePresetSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var name = ""
    @State private var category: Preset.Category = .image
    @State private var summary = ""

    var body: some View {
        Form {
            TextField("Name", text: $name)
            Picker("Category", selection: $category) {
                ForEach(Preset.Category.allCases) { Label($0.title, systemImage: $0.symbol).tag($0) }
            }
            TextField("Description", text: $summary, axis: .vertical).lineLimit(2...4)
            Text("Saved to \(model.library.presets.path). Model folders and the output folder are stored by name, so the preset works on another Mac with the same models.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .frame(width: 440)
        .safeAreaInset(edge: .bottom) {
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save Preset") {
                    model.saveAsPreset(name: name, category: category, summary: summary)
                    dismiss()
                }
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
        }
    }
}
