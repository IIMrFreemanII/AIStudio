import Foundation
@testable import MovieCore
import Testing

// MARK: places

private func withSheet(_ id: String, _ name: String) -> MovieCharacter {
    let t = Take(file: "cast/\(id).png")
    return MovieCharacter(id: id, name: name, look: "", voice: "", sheets: [t], selectedSheet: t.id)
}

private func placed() -> MovieProject {
    var m = MovieProject()
    m.characters = [withSheet("mara", "Mara"), withSheet("tomas", "Tomas"), withSheet("ana", "Ana"),
                    MovieCharacter(id: "narrator", name: "Narrator", look: "", voice: "", isNarrator: true)]
    let plate = Take(file: "places/lighthouse.png")
    m.locations = [MovieLocation(id: "lighthouse", name: "Lamp room", look: "brass and salt glass", plates: [plate],
                                 selectedPlate: plate.id),
                   MovieLocation(id: "cliff", name: "Cliff path")]
    m.scenes = [MovieScene(id: "s1", heading: "INT. LAMP ROOM", location: "lighthouse"),
                MovieScene(id: "s2", heading: "EXT. CLIFF", location: "cliff")]
    return m
}

@Test func keyframesAreMadeFromTheSheetsThenThePlaceThenTheShotBefore() {
    var m = placed()
    let key = Take(file: "keyframes/1.png")
    m.shots = [Shot(scene: "s1", characters: ["mara"], keyframes: [key], selectedKeyframe: key.id),
               Shot(scene: "s1", characters: ["mara"], continuesPrevious: true),
               Shot(scene: "s1", characters: ["mara", "tomas"], continuesPrevious: true),
               Shot(scene: "s1", characters: ["mara", "tomas", "ana"]),
               Shot(scene: "s1", characters: ["narrator"]),
               Shot(scene: "s2", characters: ["mara"])]
    func labels(_ i: Int) -> [String] { MovieArt.references(for: m.shots[i], in: m).map(\.label) }
    #expect(labels(1) == ["Mara", "Lamp room", "the previous shot"])
    #expect(labels(2) == ["Mara", "Tomas", "Lamp room"], "two people and the place: no room for the shot before")
    #expect(labels(3) == ["Mara", "Tomas", "Ana"], "three people: the plate is dropped")
    #expect(labels(4) == ["Lamp room"], "a narrator has no sheet and is never in the picture")
    #expect(labels(5) == ["Mara"], "a place without a plate adds nothing")
    #expect(MovieArt.missingSheets(for: m.shots[4], in: m).isEmpty)

    let refs = MovieArt.references(for: m.shots[2], in: m)
    let prompt = MovieArt.keyframePrompt(for: m.shots[2], in: m, references: refs)
    #expect(prompt.contains("Picture 3 is the place, Lamp room: keep its architecture"))
    let render = MovieArt.keyframeRender(m.shots[2], movie: m, folder: URL(fileURLWithPath: "/m"), seed: 1, server: "s")
    #expect(render.render.values["image3"] == .string("/m/places/lighthouse.png"))
}

@Test func aPlatesPromptIsThePlaceEmptyAtTheFilmsShape() {
    var m = placed()
    m.aspect = .vertical
    let r = MovieArt.plateRender(m.locations[0], movie: m, seed: 7, server: "gpu")
    guard case .string(let prompt)? = r.values["prompt"], case .string(let negative)? = r.values["negative"] else {
        Issue.record("no prompt")
        return
    }
    #expect(prompt.contains("empty establishing shot of Lamp room: brass and salt glass") && prompt.contains("No people"))
    #expect(negative.contains("people") && r.values["width"] == .int(768) && r.values["height"] == .int(1344))
}

@Test func theCastIsReadyWithoutANarratorsSheetOrPlates() {
    var m = placed()
    #expect(m.castIsReady)
    m.characters[1].selectedSheet = nil
    #expect(!m.castIsReady)
    // Old places decode without plates.
    let old = try? JSONDecoder().decode(MovieLocation.self, from: Data(#"{"id": "a", "name": "A"}"#.utf8))
    #expect(old?.plates.isEmpty == true && old?.look == "")
}

@Test func h3TalkingShotsGetThePlateWhenThereIsRoom() {
    var m = placed()
    m.shots = [Shot(scene: "s1", dialogue: [DialogueLine(character: "mara", text: "Hi.")], talking: true),
               Shot(scene: "s1", dialogue: [DialogueLine(character: "mara", text: "Hi."),
                                            DialogueLine(character: "tomas", text: "Ho."),
                                            DialogueLine(character: "ana", text: "Hey.")], talking: true)]
    let folder = URL(fileURLWithPath: "/nowhere")
    // The sheets can't be cut here (no files): each goes whole, and the plate follows when there's room.
    let one = MovieH3.subjectPictures(for: m.shots[0], in: m, folder: folder)
    #expect(one.pictures.map(\.view) == [nil, .place] && one.paths.last == "/nowhere/places/lighthouse.png")
    let prompt = MovieH3.referencePrompt(for: m.shots[0], in: m, pictures: one.pictures, voices: [])
    #expect(prompt.contains("<Picture 3> is the place, Lamp room, empty") && prompt.contains("<Subject 1> is Mara"))
    #expect(!prompt.contains("<Subject 2>"), "the place isn't a subject")
    let three = MovieH3.subjectPictures(for: m.shots[1], in: m, folder: folder)
    #expect(three.pictures.count == 3 && !three.pictures.contains { $0.view == .place })
}

// MARK: score

private func scene(_ id: String, _ shots: [Double], into m: inout MovieProject) {
    m.scenes.append(MovieScene(id: id, heading: id.uppercased(), mood: "tense strings", characters: ["mara"]))
    m.shots += shots.map { Shot(scene: id, duration: $0) }
}

@Test func aLongScenesScoreComesInPartsCutAtShots() {
    #expect(MovieSound.parts((0..<10).map { _ in Shot(scene: "s", duration: 10) }).count == 1)
    let long = (0..<40).map { _ in Shot(scene: "s", duration: 10) }  // 400 s → 3 parts of about 133 s
    let p = MovieSound.parts(long)
    #expect(p.count == 3 && p.flatMap { $0 }.map(\.id) == long.map(\.id))
    #expect(p.allSatisfy { $0.reduce(0) { $0 + $1.duration } <= MovieSound.maxCue })

    var m = MovieProject(style: "noir", theme: "a music-box melody")
    m.characters = [MovieCharacter(id: "mara", name: "Mara", look: "", voice: "", motif: "a lone flute")]
    scene("s1", Array(repeating: 10, count: 40), into: &m)
    scene("s2", [5, 5], into: &m)
    let cues = MovieSound.cues(for: m)
    #expect(cues.map(\.id) == ["s1", "s1.2", "s1.3", "s2"] && cues.map(\.part) == [1, 2, 3, 1])
    #expect(cues.allSatisfy { $0.prompt.contains("main theme: a music-box melody") && $0.prompt.contains("Mara's motif: a lone flute") })
    #expect(MovieSound.cueSeconds(cues[1], in: m) > 100 && MovieSound.cueSeconds(cues[3], in: m) == 10)
    // Existing cues (their takes) are kept; old ones decode as their scene's only part.
    m.music = [MusicCue(id: "s2", prompt: "mine")]
    #expect(MovieSound.cues(for: m).last?.prompt == "mine")
    let old = try? JSONDecoder().decode(MusicCue.self, from: Data(#"{"id": "s4", "prompt": "p", "takes": []}"#.utf8))
    #expect(old?.scene == "s4" && old?.part == 1)
}

@Test func cuesThatMeetCrossfadeAndEachPlaysUnderItsOwnShots() {
    var m = MovieProject()
    scene("s1", Array(repeating: 10, count: 20), into: &m)  // 200 s → 2 parts of 100 s
    scene("s2", [10, 10], into: &m)
    var durations: [String: Double] = [:]
    for i in m.shots.indices {
        let clip = Take(file: "shots/\(i).mp4")
        m.shots[i].videos = [clip]
        m.shots[i].selectedVideo = clip.id
        durations[clip.file] = 10
    }
    m.music = MovieSound.cues(for: m).map { cue in
        var c = cue
        let t = Take(file: "music/\(cue.id).wav", seconds: MovieSound.cueSeconds(cue, in: m) + 2)
        c.takes = [t]
        c.selected = t.id
        return c
    }
    let t = MovieTimeline(movie: m, durations: durations)
    let music = t.sounds.filter { $0.kind == .music }.sorted { $0.start < $1.start }
    #expect(music.map(\.id) == ["music-s1", "music-s1.2", "music-s2"])
    #expect(music[1].label == "S1 (2)")
    // Part 2 comes in a second early, part 1 plays a second on: two seconds crossfaded, across the scene cut too.
    #expect(abs(music[1].start - 99) < 1e-6 && abs(music[0].end - 101) < 1e-6)
    #expect(abs(music[1].fadeIn - 2) < 1e-6 && abs(music[0].fadeOut - 2) < 1e-6)
    #expect(abs(music[2].start - 199) < 1e-6 && abs(music[1].end - 201) < 1e-6)
    #expect(music[0].track != music[1].track, "overlapping cues are on two lanes")
}
