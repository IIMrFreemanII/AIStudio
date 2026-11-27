import AppKit
import MovieCore
import SwiftUI

// MARK: cast

/// Cast: each character's reference sheet (a few takes, one chosen), the picture every keyframe of them is made from.
struct MovieCastPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.isApproved(.cast)
        let seen = doc.movie.characters.filter { !$0.isNarrator }
        let missing = seen.filter { $0.sheets.isEmpty }.count + doc.movie.locations.filter { $0.plates.isEmpty }.count
        let total = seen.count + doc.movie.locations.count
        MoviePage(title: doc.movie.locations.isEmpty ? "Cast" : "Cast & Places",
                  subtitle: "One reference sheet per character, made from their look, and an empty plate of each place. Keyframes copy faces, clothes and places from the pictures you choose.") {
            Button(missing > 0 ? (missing == total ? "Make Sheets…" : "Make Missing…") : "Retake All…") {
                model.requestSheets(doc)
            }
            .buttonStyle(.mg(missing > 0 ? .prominent : .bordered))
            .disabled(doc.isBusy || total == 0)
            .mgHelp(missing > 0 ? "A sheet for each character and a plate for each place without one (you see the cost first)"
                    : "A new take of every sheet and plate; the chosen ones stay until you pick another")
            .mgTestAnchor("movie-make-sheets")
            MovieFastPicturesToggle(doc: doc)
            Button("Edit Template") {
                model.editMovieTemplate(MovieArt.TemplateName.sheet, for: doc) {
                    MovieArt.textToImageTemplate(name: MovieArt.TemplateName.sheet)
                }
            }
            .buttonStyle(.mg(.borderless))
            .mgHelp("Opens the graph sheets are made with as a workflow; this movie uses your edited copy")
            Button(approved ? "Approved" : "Approve Cast") {
                doc.approve(.cast)
                doc.stage = .keyframes
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || !doc.movie.castIsReady)
            .mgHelp(doc.movie.castIsReady ? "Locks the sheets in; keyframes are made from them next"
                    : "Every character needs a chosen sheet first")
            .mgTestAnchor("movie-approve-cast")
        } content: {
            VStack(alignment: .leading, spacing: 16) {
                if doc.movie.characters.isEmpty {
                    MGEmptyState(symbol: "person.2", title: "No characters",
                                 message: "The script has no cast. Add characters on the Script page.")
                        .frame(maxWidth: .infinity, minHeight: 240)
                }
                ForEach($doc.movie.characters) { $c in
                    if !c.isNarrator { CastCard(doc: doc, character: $c) }
                }
                if !doc.movie.locations.isEmpty {
                    Text("PLACES").font(.mg(size: 10, weight: .semibold)).foregroundStyle(MG.tertiaryLabel).padding(.top, 6)
                    ForEach($doc.movie.locations) { $l in
                        PlaceCard(doc: doc, location: $l)
                    }
                }
            }
        }
    }
}

private struct CastCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var character: MovieCharacter

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(character.name).font(.mgHeadline)
                Text(shotCount).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer()
                Button(character.sheets.isEmpty ? "Make Sheet…" : "Another Take…") {
                    model.requestSheets(doc, characters: [character.id])
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small).disabled(doc.isBusy)
                .mgHelp("One more reference sheet for \(character.name) (you see the cost first)")
            }
            MovieField(label: "Look", hint: "edit, then make another take") {
                TextField("Age, face, hair, build, clothes", text: $character.look, axis: .vertical)
                    .lineLimit(1...4).font(.mgCallout)
            }
            if character.sheets.isEmpty {
                RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous).fill(MG.fill)
                    .frame(height: 120)
                    .overlay {
                        Label("No sheet yet", systemImage: "person.crop.rectangle").font(.mgFootnote)
                            .foregroundStyle(MG.tertiaryLabel)
                    }
            } else {
                TakeStrip(doc: doc, takes: character.sheets, selected: character.selectedSheet, height: 150,
                          select: { model.selectSheet(doc, character: character.id, take: $0) },
                          delete: { model.deleteTake(doc, character: character.id, take: $0) },
                          open: { model.viewSheets(doc, character: character.id, take: $0) })
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: character.look) { _, _ in
            if doc.isApproved(.cast), !doc.isBusy { doc.reopen(.cast) }
        }
        .mgTestAnchor("movie-cast-card")
        .mgScrollTarget("movie-cast:\(character.id)")
    }

    private var shotCount: String {
        let n = doc.movie.shots.filter { $0.characters.contains(character.id) }.count
        return n == 1 ? "in 1 shot" : "in \(n) shots"
    }
}

/// A place on the Cast page: its look and its plates (the place itself, empty), one chosen.
private struct PlaceCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var location: MovieLocation

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Text(location.name).font(.mgHeadline)
                Text(sceneCount).font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer()
                Button(location.plates.isEmpty ? "Make Plate…" : "Another Take…") {
                    model.requestSheets(doc, locations: [location.id])
                }
                .buttonStyle(.mg(.bordered)).controlSize(.small).disabled(doc.isBusy)
                .mgHelp("An empty picture of \(location.name) every keyframe there is made from (you see the cost first)")
            }
            MovieField(label: "Look", hint: "edit, then make another take") {
                TextField("Architecture, materials, light, colours, weather", text: $location.look, axis: .vertical)
                    .lineLimit(1...4).font(.mgCallout)
            }
            if location.plates.isEmpty {
                RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous).fill(MG.fill)
                    .frame(height: 120)
                    .overlay {
                        Label("No plate yet", systemImage: "photo").font(.mgFootnote).foregroundStyle(MG.tertiaryLabel)
                    }
            } else {
                TakeStrip(doc: doc, takes: location.plates, selected: location.selectedPlate, height: 120,
                          select: { model.selectPlate(doc, location: location.id, take: $0) },
                          delete: { model.deletePlate(doc, location: location.id, take: $0) },
                          open: { id in
                              model.openMovieViewer(doc, takes: location.plates, name: location.name, take: id)
                          })
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: location.look) { _, _ in
            if doc.isApproved(.cast), !doc.isBusy { doc.reopen(.cast) }
        }
        .mgTestAnchor("movie-place-card")
        .mgScrollTarget("movie-place:\(location.id)")
    }

    private var sceneCount: String {
        let n = doc.movie.scenes.filter { $0.location == location.id }.count
        return n == 1 ? "in 1 scene" : "in \(n) scenes"
    }
}

// MARK: keyframes

/// Keyframes: the first frame of every shot, made from the cast's sheets; a few takes each, one chosen.
struct MovieKeyframesPage: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: MovieDocument

    var body: some View {
        let approved = doc.isApproved(.keyframes)
        let shots = doc.scopeShots
        let missing = shots.filter { $0.keyframes.isEmpty }.count
        let done = shots.count - missing
        let ready = doc.movie.keyframesAreReady(in: doc.sequence)
        let index = doc.movie.shotIndex
        MoviePage(title: "Keyframes" + (doc.scopeTitle.map { " · " + $0 } ?? ""),
                  subtitle: "\(done) of \(shots.count) shots have a keyframe. Each is the still its clip starts from.") {
            Button(missing > 0 ? (done == 0 ? "Make Keyframes…" : "Make Missing Keyframes…") : "Retake All…") {
                model.requestKeyframes(doc)
            }
            .buttonStyle(.mg(missing > 0 ? .prominent : .bordered))
            .disabled(doc.isBusy || shots.isEmpty)
            .mgHelp(missing > 0 ? "One keyframe for each shot without one (you see the cost first)"
                    : "A new take for every shot; the chosen ones stay until you pick another")
            .mgTestAnchor("movie-make-keyframes")
            MovieFastPicturesToggle(doc: doc)
            Button("Edit Template") {
                model.editMovieTemplate(MovieArt.TemplateName.keyframeEdit, for: doc) { MovieArt.editTemplate() }
            }
            .buttonStyle(.mg(.borderless))
            .mgHelp("Opens the graph keyframes with characters are made with as a workflow; this movie uses your edited copy")
            Button(approved ? "Approved" : "Approve Keyframes") {
                doc.approve(.keyframes)
            }
            .buttonStyle(.mg(.prominent))
            .disabled(approved || doc.isBusy || !ready)
            .mgHelp(ready ? "Locks the keyframes in; the shots are animated from them next"
                    : "Every shot needs a chosen keyframe first")
            .mgTestAnchor("movie-approve-keyframes")
        } content: {
            LazyVStack(alignment: .leading, spacing: 26) {
                ForEach(doc.scopeScenes) { scene in
                    let ids = shots.filter { $0.scene == scene.id }.map(\.id)
                    if !ids.isEmpty {
                        VStack(alignment: .leading, spacing: 10) {
                            Text(scene.heading).font(.mgHeadline)
                            LazyVGrid(columns: [GridItem(.adaptive(minimum: 300, maximum: 420), spacing: 12, alignment: .top)],
                                      alignment: .leading, spacing: 12) {
                                ForEach(ids, id: \.self) { id in
                                    if let i = index[id] {
                                        KeyframeCard(doc: doc, shot: $doc.movie.shots[i], number: i + 1)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

private struct KeyframeCard: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    @Binding var shot: Shot
    let number: Int

    var body: some View {
        let refs = MovieArt.references(for: shot, in: doc.movie)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("\(number)").font(.mg(size: 12, weight: .bold)).monospacedDigit()
                    .frame(minWidth: 22, minHeight: 22)
                    .background(Circle().fill(MG.accent.opacity(0.15)))
                    .foregroundStyle(MG.accent)
                Text([shot.framing, MovieFormat.seconds(shot.duration)].filter { !$0.isEmpty }.joined(separator: " · "))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                Spacer(minLength: 2)
                Button { model.requestKeyframes(doc, shots: [shot.id]) } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.mg(.icon)).disabled(doc.isBusy)
                    .mgHelp(shot.keyframes.isEmpty ? "Make this shot's keyframe (you see the cost first)"
                            : "Another take of this keyframe (you see the cost first)")
            }
            if let file = shot.selectedKeyframeFile {
                MovieStill(url: model.movieStore.url(doc.id, relative: file), maxPixels: 840)
                    .aspectRatio(doc.movie.aspect.ratio, contentMode: .fit)
                    .contentShape(Rectangle())
                    .onTapGesture(count: 2) { model.viewKeyframes(doc, shot: shot.id) }
                    .overlay(alignment: .topTrailing) {
                        MovieExpandButton { model.viewKeyframes(doc, shot: shot.id) }.padding(6)
                    }
            } else {
                RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous).fill(MG.fill)
                    .aspectRatio(doc.movie.aspect.ratio, contentMode: .fit)
                    .overlay {
                        Image(systemName: "photo").font(.mg(size: 18)).foregroundStyle(MG.tertiaryLabel)
                    }
            }
            if shot.keyframes.count > 1 {
                TakeStrip(doc: doc, takes: shot.keyframes, selected: shot.selectedKeyframe, height: 54,
                          select: { model.selectKeyframe(doc, shot: shot.id, take: $0) },
                          delete: { model.deleteTake(doc, shot: shot.id, take: $0) },
                          open: { model.viewKeyframes(doc, shot: shot.id, take: $0) })
            }
            TextField("The first frame: who, where, light, composition", text: $shot.keyframePrompt, axis: .vertical)
                .lineLimit(2...6).font(.mgCallout)
            HStack(spacing: 6) {
                Image(systemName: refs.isEmpty ? "text.below.photo" : "person.crop.rectangle.stack")
                    .foregroundStyle(MG.tertiaryLabel)
                Text(refs.isEmpty ? "From the prompt alone" : "From " + refs.map(\.label).joined(separator: ", "))
                    .font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            .mgHelp(refs.isEmpty ? "No one is in this shot: a plain picture from the prompt"
                    : "Made from these pictures, so faces, clothes and the place carry over")
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).fill(MG.card))
        .overlay(RoundedRectangle(cornerRadius: MGRadius.lg, style: .continuous).strokeBorder(MG.separator))
        .onChange(of: shot.keyframePrompt) { _, _ in
            if doc.movie.isApproved(.keyframes, in: doc.movie.sequence(of: shot)), !doc.isBusy {
                doc.reopen(.keyframes, shot: shot.id)
            }
        }
        .mgTestAnchor("movie-keyframe-card")
        .mgScrollTarget("movie-keyframe:\(shot.id.uuidString)")
    }
}

// MARK: shared

/// A row of takes (sheets, keyframes, clips): click one to choose it; double-click (or its hover button) opens it in
/// the viewer; its menu opens, reveals or deletes it.
struct TakeStrip: View {
    @Environment(AppModel.self) private var model
    let doc: MovieDocument
    let takes: [Take]
    let selected: UUID?
    let height: CGFloat
    let select: (UUID) -> Void
    let delete: (UUID) -> Void
    /// Opens the take in the viewer (nil: no viewer).
    var open: ((UUID) -> Void)? = nil
    @State private var hovered: UUID?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(takes) { take in
                    let url = model.movieStore.url(doc.id, relative: take.file)
                    Button { select(take.id) } label: {
                        MovieStill(url: model.movieStore.url(doc.id, relative: take.poster ?? take.file),
                                   maxPixels: Int(height * 4))
                            .frame(height: height)
                            .overlay(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous)
                                .strokeBorder(take.id == selected ? MG.accent : MG.separator,
                                              lineWidth: take.id == selected ? 2.5 : 1))
                            .overlay(alignment: .topTrailing) {
                                if take.id == selected {
                                    Image(systemName: "checkmark.circle.fill").foregroundStyle(MG.accent)
                                        .padding(4)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                    .disabled(doc.isBusy)
                    .mgHelp((take.id == selected ? "Chosen" : "Choose this take") + costText(take)
                            + (open == nil ? "" : " · double-click to open it"))
                    .simultaneousGesture(TapGesture(count: 2).onEnded { open?(take.id) })
                    .overlay(alignment: .bottomTrailing) {
                        if let open, hovered == take.id {
                            MovieExpandButton(small: height < 100) { open(take.id) }.padding(4)
                        }
                    }
                    .onHover { inside in
                        if inside { hovered = take.id } else if hovered == take.id { hovered = nil }
                    }
                    .mgContextMenu {
                        MGMenuItem.button("Choose") { select(take.id) }
                        if let open { MGMenuItem.button("Open in Viewer") { open(take.id) } }
                        MGMenuItem.button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                        MGMenuItem.divider
                        MGMenuItem.button("Move to Trash", destructive: true) { delete(take.id) }
                    }
                }
            }
            .padding(2)
        }
    }

    private func costText(_ take: Take) -> String {
        take.costUSD.map { " · " + CostEstimate.format($0) } ?? ""
    }
}

/// A generated still from the movie's folder, decoded at a small size off the main thread.
struct MovieStill: View {
    let url: URL
    var maxPixels: Int = 600
    @State private var image: CGImage?

    var body: some View {
        ZStack {
            if let image {
                Image(decorative: image, scale: 1).resizable().aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(MG.fill)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: MGRadius.sm, style: .continuous))
        .task(id: url) {
            let url = url, size = maxPixels
            image = await Task.detached(priority: .userInitiated) {
                guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil as CGImage? }
                return CGImageSourceCreateThumbnailAtIndex(src, 0, [
                    kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceThumbnailMaxPixelSize: size,
                    kCGImageSourceCreateThumbnailWithTransform: true,
                ] as CFDictionary)
            }.value
        }
    }
}

extension MovieAspect {
    var ratio: CGFloat {
        let s = keyframeSize
        return CGFloat(s.width) / CGFloat(s.height)
    }
}

/// The button that opens a still, clip or sound in the full-window viewer.
struct MovieExpandButton: View {
    var small = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: "arrow.up.left.and.arrow.down.right")
                .font(.system(size: small ? 9 : 11, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: small ? 18 : 24, height: small ? 18 : 24)
                .background(Circle().fill(.black.opacity(0.55)))
        }
        .buttonStyle(.plain)
        .mgHelp("Open in the viewer")
    }
}

/// The movie's Fast switch for its pictures (sheets, plates and keyframes): Qwen's Lightning modes, much cheaper but
/// without the negative prompt. Shown on the Cast and Keyframes pages; one setting for both.
struct MovieFastPicturesToggle: View {
    @Bindable var doc: MovieDocument

    var body: some View {
        Toggle("Fast", isOn: $doc.movie.fastPictures)
            .toggleStyle(.switch).controlSize(.small)
            .disabled(doc.isBusy)
            .mgHelp("Qwen's Lightning mode for sheets, plates and keyframes: 8 steps without guidance, about 6× cheaper "
                    + "(10× for keyframes with the cast), but the negative prompt is ignored, so text or extra people "
                    + "can slip in. Off: the full models (50 and 40 steps with guidance)")
            .mgTestAnchor("movie-fast-pictures")
    }
}
