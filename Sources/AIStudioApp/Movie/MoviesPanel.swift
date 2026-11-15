import AppKit
import MovieCore
import SwiftUI

/// The sidebar's Movies panel: every saved movie, newest change first; open ones are marked.
struct MoviesPanel: View {
    @Environment(AppModel.self) private var model
    @State private var confirmDelete: MovieSummary?

    var body: some View {
        VStack(spacing: 0) {
            PanelHeader(title: "Movies") {
                Button { model.newMovie() } label: { Image(systemName: "plus") }
                    .mgHelp("New movie (⌃⌘N)")
                    .mgTestAnchor("movies-new")
                Button { NSWorkspace.shared.open(model.library.movies) } label: { Image(systemName: "arrow.up.forward.app") }
                    .mgHelp("Open the movies folder in Finder")
            }
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    if model.movieSummaries.isEmpty {
                        Text("Make a short film from an idea: the model writes the script and shot list, then the cloud GPU renders it shot by shot. Press + to start.")
                            .font(.mgCallout).foregroundStyle(MG.secondaryLabel)
                            .padding(.horizontal, 16).padding(.vertical, 6)
                    }
                    ForEach(model.movieSummaries) { m in row(m) }
                }
                .padding(.bottom, 8)
            }
            .mgScrollIndicators()
        }
        .onAppear { model.reloadMovies() }
        .mgAlert("Move “\(confirmDelete?.title ?? "")” to the Trash?",
                 isPresented: Binding(get: { confirmDelete != nil }, set: { if !$0 { confirmDelete = nil } }),
                 message: "Its script, shots and everything generated for it go with it.") {
            [.cancel(), .destructive("Move to Trash") { if let m = confirmDelete { model.deleteMovie(m.id) } }]
        }
    }

    private func row(_ m: MovieSummary) -> some View {
        let open = model.movies.contains { $0.id == m.id }
        let active = model.activeMovieID == m.id
        return HStack(spacing: 6) {
            Image(systemName: "film").foregroundStyle(active ? MG.accent : MG.secondaryLabel)
            VStack(alignment: .leading, spacing: 1) {
                Text(m.title).lineLimit(1)
                Text("\(m.stage.title) · \(m.shots) shots").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
            }
            Spacer()
            if open {
                Circle().fill(active ? MG.accent : MG.tertiaryLabel).frame(width: 6, height: 6).mgHelp("Open")
            }
        }
        .mgRow(selected: active, height: 38, indent: 16)
        .contentShape(Rectangle())
        .onTapGesture { model.openMovie(m.id) }
        .mgTestAnchor("movie-row:\(m.title)")
        .mgContextMenu {
            MGMenuItem.button("Open") { model.openMovie(m.id) }
            if open { MGMenuItem.button("Close") { model.closeMovie(m.id) } }
            MGMenuItem.divider
            MGMenuItem.button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([model.movieStore.file(m.id)])
            }
            MGMenuItem.button("Move to Trash…", destructive: true) { confirmDelete = m }
        }
    }
}
