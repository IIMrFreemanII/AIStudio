import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

// Open and save inside the app's own folders (the workflow library, Assets › Input and Output), drawn by the design
// system. "Browse in Finder…" hands over to the system's panel for anywhere else on disk.

struct MGFileRoot: Identifiable {
    var id: String { url.path }
    let title: String
    let url: URL
}

enum MGFilePickerMode {
    case open(types: [UTType], multiple: Bool)
    case save(name: String, type: UTType)

    var isSave: Bool { if case .save = self { return true } else { return false } }
}

struct MGFileEntry: Identifiable, Hashable {
    var id: URL { url }
    let url: URL
    let isFolder: Bool
    let modified: Date
    var name: String { url.lastPathComponent }
}

/// Lists a folder for the picker: subfolders first, then the files of the wanted types, by name.
enum MGFileListing {
    static func list(_ dir: URL, types: [UTType]) -> [MGFileEntry] {
        let keys: [URLResourceKey] = [.isDirectoryKey, .contentModificationDateKey, .isPackageKey]
        let urls = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: keys,
                                                                 options: [.skipsHiddenFiles])) ?? []
        var out: [MGFileEntry] = []
        for u in urls {
            let v = try? u.resourceValues(forKeys: Set(keys))
            let folder = v?.isDirectory == true && v?.isPackage != true
            if !folder, !matches(u, types) { continue }
            out.append(MGFileEntry(url: u, isFolder: folder, modified: v?.contentModificationDate ?? .distantPast))
        }
        return out.sorted { a, b in
            a.isFolder != b.isFolder ? a.isFolder : a.name.localizedStandardCompare(b.name) == .orderedAscending
        }
    }

    static func matches(_ url: URL, _ types: [UTType]) -> Bool {
        guard !types.isEmpty else { return true }
        guard let t = UTType(filenameExtension: url.pathExtension) else { return false }
        return types.contains { t.conforms(to: $0) }
    }
}

struct MGFilePicker: View {
    let title: String
    var message: String? = nil
    let roots: [MGFileRoot]
    let mode: MGFilePickerMode
    /// A folder inside one of the roots to start in.
    var startIn: URL? = nil
    /// The system's panel, for anywhere else on disk (its result, or nil when cancelled).
    let finder: () -> [URL]?
    let onChoose: ([URL]) -> Void

    @Environment(\.mgDismiss) private var dismiss
    @Environment(MGPresenter.self) private var presenter: MGPresenter?
    @State private var rootIndex = 0
    @State private var path: [String] = []
    @State private var entries: [MGFileEntry] = []
    @State private var selection: Set<URL> = []
    @State private var name = ""
    @State private var newFolder: String?
    @FocusState private var nameFocused: Bool

    var types: [UTType] {
        switch mode {
        case .open(let t, _): return t
        case .save(_, let t): return [t]
        }
    }

    var root: MGFileRoot { roots[min(rootIndex, roots.count - 1)] }
    var folder: URL { path.reduce(root.url) { $0.appendingPathComponent($1, isDirectory: true) } }
    var imagesOnly: Bool {
        !types.isEmpty && types.allSatisfy { $0.conforms(to: .image) } && entries.contains { !$0.isFolder }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.mgHeadline)
                    if let message { Text(message).font(.mgCallout).foregroundStyle(MG.secondaryLabel) }
                }
                Spacer()
                if roots.count > 1 {
                    MGSegmentedPicker(selection: $rootIndex, options: roots.indices.map { ($0, roots[$0].title) })
                        .frame(width: 70 * CGFloat(roots.count))
                }
            }
            breadcrumbs
            listing
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.contentBackground))
                .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
            if case .save = mode {
                MGFormRow("Save As") {
                    TextField("Name", text: $name).focused($nameFocused).onSubmit { commit() }.frame(maxWidth: 300)
                }
            }
            HStack(spacing: 8) {
                Button("Browse in Finder…") { browse() }.buttonStyle(.mg(.borderless))
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button(mode.isSave ? "Save" : "Open") { commit() }
                    .buttonStyle(.mg(.prominent))
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canCommit)
            }
        }
        .padding(20)
        .frame(width: 600, height: 460)
        .onAppear {
            if case .save(let n, _) = mode { name = n; nameFocused = true }
            if let start = startIn?.standardizedFileURL.path {
                for (i, r) in roots.enumerated() {
                    let base = r.url.standardizedFileURL.path
                    guard start == base || start.hasPrefix(base + "/") else { continue }
                    rootIndex = i
                    path = start.dropFirst(base.count).split(separator: "/").map(String.init)
                    break
                }
            }
            reload()
        }
        .onChange(of: rootIndex) { path = []; reload() }
        .onChange(of: path) { reload() }
    }

    // MARK: parts

    var breadcrumbs: some View {
        HStack(spacing: 2) {
            crumb(root.title, depth: 0)
            ForEach(path.indices, id: \.self) { i in
                Image(systemName: "chevron.right").font(.system(size: 8, weight: .semibold)).foregroundStyle(MG.tertiaryLabel)
                crumb(path[i], depth: i + 1)
            }
            Spacer()
            if mode.isSave {
                if let text = newFolder {
                    TextField("Folder name", text: Binding(get: { text }, set: { newFolder = $0 }))
                        .frame(width: 160)
                        .onSubmit { makeFolder() }
                    Button("Create") { makeFolder() }.controlSize(.small)
                    Button("Cancel") { newFolder = nil }.controlSize(.small)
                } else {
                    Button { newFolder = "" } label: { Label("New Folder", systemImage: "folder.badge.plus") }
                        .buttonStyle(.mg(.borderless))
                        .controlSize(.small)
                }
            }
        }
        .frame(height: 22)
    }

    func crumb(_ title: String, depth: Int) -> some View {
        Button(title) { path = Array(path.prefix(depth)) }
            .buttonStyle(.mg(.borderless))
            .controlSize(.small)
            .fontWeight(depth == path.count ? .medium : .regular)
    }

    @ViewBuilder var listing: some View {
        if entries.isEmpty {
            MGEmptyState(symbol: "folder", title: "Nothing here",
                         message: "Use Browse in Finder… for files elsewhere on your Mac.")
        } else if imagesOnly {
            ScrollView {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 104), spacing: 10)], spacing: 10) {
                    ForEach(entries) { e in
                        VStack(spacing: 4) {
                            ZStack {
                                RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.fill)
                                if e.isFolder {
                                    Image(systemName: "folder.fill").font(.system(size: 30)).foregroundStyle(MG.hueFolder)
                                } else {
                                    MGFileThumb(url: e.url)
                                }
                            }
                            .frame(width: 96, height: 96)
                            .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
                            .overlay(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous)
                                .strokeBorder(selection.contains(e.url) ? MG.accent : .clear, lineWidth: 2))
                            Text(e.name).font(.mgFootnote).lineLimit(1).truncationMode(.middle).frame(width: 100)
                        }
                        .contentShape(Rectangle())
                        .onTapGesture(count: 2) { open(e) }
                        .onTapGesture { select(e) }
                    }
                }
                .padding(10)
            }
            .mgScrollIndicators()
        } else {
            ScrollView {
                LazyVStack(spacing: 0) {
                    ForEach(entries) { e in
                        HStack(spacing: 8) {
                            Image(systemName: e.isFolder ? "folder.fill" : icon(e.url))
                                .foregroundStyle(e.isFolder ? MG.hueFolder : MG.secondaryLabel)
                                .frame(width: 16)
                            Text(e.isFolder ? e.name : e.url.deletingPathExtension().lastPathComponent).lineLimit(1)
                            Spacer()
                            if e.isFolder {
                                Image(systemName: "chevron.right").font(.system(size: 9)).foregroundStyle(MG.tertiaryLabel)
                            } else {
                                Text(e.modified, format: .dateTime.day().month(.abbreviated).hour().minute())
                                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                            }
                        }
                        .mgRow(selected: selection.contains(e.url))
                        .onTapGesture(count: 2) { open(e) }
                        .onTapGesture { select(e) }
                    }
                }
                .padding(.vertical, 6)
            }
            .mgScrollIndicators()
        }
    }

    func icon(_ url: URL) -> String {
        guard let t = UTType(filenameExtension: url.pathExtension) else { return "doc" }
        if t.conforms(to: .image) { return "photo" }
        if t.conforms(to: .movie) { return "film" }
        if t.conforms(to: .audio) { return "waveform" }
        return "doc.text"
    }

    // MARK: actions

    func reload() {
        entries = MGFileListing.list(folder, types: types)
        selection = selection.filter { s in entries.contains { $0.url == s } }
    }

    func select(_ e: MGFileEntry) {
        if case .open(_, true) = mode, NSEvent.modifierFlags.contains(.command) {
            if selection.contains(e.url) { selection.remove(e.url) } else { selection.insert(e.url) }
        } else {
            selection = [e.url]
        }
        if case .save = mode, !e.isFolder { name = e.url.deletingPathExtension().lastPathComponent }
    }

    func open(_ e: MGFileEntry) {
        if e.isFolder { path.append(e.name); return }
        selection = [e.url]
        commit()
    }

    var canCommit: Bool {
        switch mode {
        case .open: return !selection.isEmpty
        case .save: return !name.trimmingCharacters(in: .whitespaces).isEmpty
        }
    }

    func commit() {
        switch mode {
        case .open:
            let picked = entries.filter { selection.contains($0.url) }
            if picked.count == 1, picked[0].isFolder { path.append(picked[0].name); return }
            let files = picked.filter { !$0.isFolder }.map(\.url)
            guard !files.isEmpty else { return }
            dismiss()
            onChoose(files)
        case .save(_, let type):
            var n = name.trimmingCharacters(in: .whitespaces)
            guard !n.isEmpty else { return }
            if let ext = type.preferredFilenameExtension, (n as NSString).pathExtension.lowercased() != ext { n += ".\(ext)" }
            let url = folder.appendingPathComponent(n)
            if FileManager.default.fileExists(atPath: url.path), let presenter {
                presenter.alert("“\(n)” already exists. Replace it?",
                                message: "A file with that name is already in “\(path.last ?? root.title)”.",
                                actions: [.cancel(), .destructive("Replace") { dismiss(); onChoose([url]) }])
                return
            }
            dismiss()
            onChoose([url])
        }
    }

    func makeFolder() {
        guard let n = newFolder?.trimmingCharacters(in: .whitespaces), !n.isEmpty, !n.contains("/") else { return }
        try? FileManager.default.createDirectory(at: folder.appendingPathComponent(n, isDirectory: true),
                                                 withIntermediateDirectories: true)
        newFolder = nil
        path.append(n)
    }

    func browse() {
        dismiss()
        DispatchQueue.main.async { if let urls = finder(), !urls.isEmpty { onChoose(urls) } }
    }
}

/// A small image preview, made off the main thread.
struct MGFileThumb: View {
    let url: URL
    @State private var image: CGImage?

    var body: some View {
        Group {
            if let image {
                Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "photo").foregroundStyle(MG.tertiaryLabel)
            }
        }
        .task(id: url) {
            image = await Task.detached(priority: .utility) { () -> CGImage? in
                guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
                return CGImageSourceCreateThumbnailAtIndex(src, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true,
                    kCGImageSourceThumbnailMaxPixelSize: 192,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                ] as CFDictionary)
            }.value
        }
    }
}

extension MGPresenter {
    /// Opens the in-app picker.
    func chooseFile(title: String, message: String? = nil, roots: [MGFileRoot], mode: MGFilePickerMode,
                    startIn: URL? = nil, finder: @escaping () -> [URL]?, onChoose: @escaping ([URL]) -> Void) {
        present(id: "file-picker") {
            MGFilePicker(title: title, message: message, roots: roots, mode: mode, startIn: startIn, finder: finder,
                         onChoose: onChoose)
        }
    }
}
