import Foundation
import GraphRuntime

/// The cast's reference sheets and the shots' keyframes: their prompts, which pictures a keyframe is made from, and
/// the built-in graphs that render them (Qwen-Image for sheets and shots without characters, Qwen-Image-Edit from
/// the sheets for the rest).
public enum MovieArt {
    /// Qwen-Image's wide training size: room for a full figure, a three-quarter view and a face side by side.
    public static let sheetSize = (width: 1664, height: 928)
    /// Qwen-Image-Edit reads at most three pictures.
    public static let maxReferences = 3

    public static func sheetPrompt(_ c: MovieCharacter, style: MovieStyle) -> String {
        let look = c.look.trimmingCharacters(in: .whitespacesAndNewlines)
        return "Character reference sheet of \(c.name)\(look.isEmpty ? "" : ", \(look)"). Three views side by side "
            + "of the same person: full body from the front, full body three-quarter view, and a close-up portrait "
            + "of the face. Neutral pose, plain light grey background, even soft studio light, the same face, hair "
            + "and clothes in every view. \(style.keyframeSuffix)"
    }

    public static func sheetNegative(style: MovieStyle) -> String {
        style.negative + ", text, labels, extra people, different outfits, cropped"
    }

    /// A picture a keyframe is made from: a character's chosen sheet, the place's plate, or the shot before's
    /// keyframe.
    public struct Reference: Equatable, Sendable {
        /// "Mara", "Lamp room", or "the previous shot".
        public var label: String
        /// Relative to the movie's folder.
        public var file: String
        public var character: String?
        /// The place's plate (a `MovieLocation` id).
        public var location: String? = nil
    }

    /// The place a shot is in, when its scene names one.
    public static func location(of shot: Shot, in movie: MovieProject) -> MovieLocation? {
        guard let id = movie.scenes.first(where: { $0.id == shot.scene })?.location, !id.isEmpty else { return nil }
        return movie.locations.first { $0.id == id }
    }

    /// The pictures a shot's keyframe is made from, at most three: the sheets of its characters (in the order the
    /// shot lists them), then its place's plate, then, for a shot that continues the one before, that shot's keyframe
    /// (same place and light). With three characters there's no room for the plate. Characters without a chosen
    /// sheet are left out (`missingSheets` names them).
    public static func references(for shot: Shot, in movie: MovieProject) -> [Reference] {
        var refs: [Reference] = []
        for id in shot.characters {
            guard refs.count < maxReferences, let c = movie.character(id), !c.isNarrator, let file = c.selectedSheetFile,
                  !refs.contains(where: { $0.character == id }) else { continue }
            refs.append(Reference(label: c.name, file: file, character: id))
        }
        if refs.count < maxReferences, let place = location(of: shot, in: movie), let file = place.selectedPlateFile {
            refs.append(Reference(label: place.name, file: file, character: nil, location: place.id))
        }
        if shot.continuesPrevious, refs.count < maxReferences, let prev = movie.shot(before: shot),
           prev.scene == shot.scene, let file = prev.selectedKeyframeFile {
            refs.append(Reference(label: "the previous shot", file: file, character: nil))
        }
        return refs
    }

    /// The names of the shot's characters that have no chosen reference sheet yet.
    public static func missingSheets(for shot: Shot, in movie: MovieProject) -> [String] {
        shot.characters.compactMap { id in
            guard let c = movie.character(id), !c.isNarrator else { return nil }
            return c.selectedSheetFile == nil ? c.name : nil
        }
    }

    /// The keyframe's instruction. With pictures: says which picture is who ("Picture 1 is Mara…") and to keep
    /// their faces and clothes; without: a plain text-to-image prompt.
    public static func keyframePrompt(for shot: Shot, in movie: MovieProject, references: [Reference]) -> String {
        let style = MovieStyle.named(movie.style)
        // A still: the framing matters, the camera move doesn't (that's the video stage's).
        let framing = shot.framing.trimmingCharacters(in: .whitespacesAndNewlines)
        var parts: [String] = []
        if !references.isEmpty {
            let who = references.enumerated().map { i, r in
                r.location != nil ? "Picture \(i + 1) is the place, \(r.label): keep its architecture, materials, light and colours"
                    : r.character == nil ? "Picture \(i + 1) is \(r.label): keep its place, light and colours"
                    : "Picture \(i + 1) is \(r.label)"
            }
            parts.append(who.joined(separator: ". ") + ".")
        }
        var scene = shot.keyframePrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if scene.isEmpty, let s = movie.scenes.first(where: { $0.id == shot.scene }) { scene = s.heading }
        parts.append(scene.hasSuffix(".") ? scene : scene + ".")
        if !framing.isEmpty { parts.append("\(framing.prefix(1).uppercased() + framing.dropFirst()) shot.") }
        if references.contains(where: { $0.character != nil }) {
            parts.append("Keep each character's face, hair, build and clothes exactly as in their picture; "
                + "a single frame of a film, not a reference sheet, no plain background.")
        }
        parts.append(style.keyframeSuffix)
        return parts.joined(separator: " ")
    }

    // MARK: templates

    /// Names of the built-in templates, also the files of a movie's edited copies (`templates/<name>.json`).
    public enum TemplateName {
        public static let sheet = "character-sheet"
        public static let keyframe = "keyframe"
        public static let keyframeEdit = "keyframe-edit"
        public static let plate = "location-plate"
    }

    /// A place's plate: the place itself, empty, as a wide film frame.
    public static func platePrompt(_ l: MovieLocation, style: MovieStyle) -> String {
        let look = l.look.trimmingCharacters(in: .whitespacesAndNewlines)
        return "An empty establishing shot of \(l.name)\(look.isEmpty ? "" : ": \(look)"). No people, no figures, "
            + "nobody in the frame. A wide film frame that shows the whole place, its light and atmosphere. "
            + style.keyframeSuffix
    }

    public static func plateRender(_ l: MovieLocation, movie: MovieProject, seed: Int, server: String) -> Render {
        let style = MovieStyle.named(movie.style)
        let size = movie.aspect.keyframeSize
        return Render(values: [
            "prompt": .string(platePrompt(l, style: style)),
            "negative": .string(style.negative + ", people, person, figure, crowd, text"),
            "seed": .int(seed), "width": .int(size.width), "height": .int(size.height), "server": .string(server),
        ], unusedPictures: [])
    }

    /// The slots every image template has. Edit templates have `image1`…`image3` too (a Load Image's path each).
    public static let imageSlots = ["prompt", "negative", "seed", "width", "height", "server"]
    public static let pictureSlots = ["image1", "image2", "image3"]

    /// Qwen-Image (Cloud GPU) into a Preview Image: character sheets and shots with no one in them.
    public static func textToImageTemplate(name: String) -> MovieTemplate {
        let gen = GraphNode(id: "image", type: "RemoteQwenImage", inputs: [
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(1328)), "height": .value(.int(1328)), "fast": .value(.bool(true)),
            "server": .value(.string("")),
        ], position: GraphPoint(x: 40, y: 60))
        let out = GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "image", output: "image")],
                            position: GraphPoint(x: 420, y: 80))
        return MovieTemplate(name: name, graph: Graph(nodes: [gen, out]),
                             slots: Dictionary(uniqueKeysWithValues: imageSlots.map { ($0, .init(node: "image", input: $0)) }))
    }

    /// Three Load Images into Qwen-Image-Edit (Cloud GPU) into a Preview Image: keyframes from the cast's sheets.
    public static func editTemplate(name: String = TemplateName.keyframeEdit) -> MovieTemplate {
        var nodes: [GraphNode] = []
        var inputs: [String: NodeInput] = [
            "prompt": .value(.string("")), "negative": .value(.string("")), "seed": .value(.int(0)),
            "width": .value(.int(0)), "height": .value(.int(0)), "fast": .value(.bool(true)),
            "server": .value(.string("")),
        ]
        var slots = Dictionary(uniqueKeysWithValues: imageSlots.map { ($0, MovieTemplate.Slot(node: "edit", input: $0)) })
        for (i, slot) in pictureSlots.enumerated() {
            let id = "picture\(i + 1)"
            nodes.append(GraphNode(id: id, type: "LoadImage", inputs: ["path": .value(.string(""))],
                                   position: GraphPoint(x: 40, y: 40 + Double(i) * 140)))
            inputs[slot] = .link(node: id, output: "image")
            slots[slot] = .init(node: id, input: "path")
        }
        nodes.append(GraphNode(id: "edit", type: "RemoteQwenImageEdit", inputs: inputs, position: GraphPoint(x: 340, y: 60)))
        nodes.append(GraphNode(id: "out", type: "PreviewImage", inputs: ["image": .link(node: "edit", output: "image")],
                               position: GraphPoint(x: 720, y: 80)))
        return MovieTemplate(name: name, graph: Graph(nodes: nodes), slots: slots)
    }

    /// The values a sheet or keyframe render binds, and which picture slots it leaves out (their Load Image nodes
    /// are removed, so the edit node runs with fewer pictures).
    public struct Render: Sendable, Equatable {
        public var values: [String: JSONValue]
        public var unusedPictures: Set<String>
        public init(values: [String: JSONValue], unusedPictures: Set<String>) {
            self.values = values
            self.unusedPictures = unusedPictures
        }
    }

    public static func sheetRender(_ c: MovieCharacter, movie: MovieProject, seed: Int, server: String) -> Render {
        let style = MovieStyle.named(movie.style)
        return Render(values: [
            "prompt": .string(sheetPrompt(c, style: style)), "negative": .string(sheetNegative(style: style)),
            "seed": .int(seed), "width": .int(sheetSize.width), "height": .int(sheetSize.height),
            "server": .string(server),
        ], unusedPictures: [])
    }

    /// A keyframe: with pictures, for the edit template (absolute paths under `folder`); without, for the
    /// text-to-image one.
    public static func keyframeRender(_ shot: Shot, movie: MovieProject, folder: URL, seed: Int, server: String)
        -> (render: Render, references: [Reference])
    {
        let refs = references(for: shot, in: movie)
        let size = movie.aspect.keyframeSize
        var values: [String: JSONValue] = [
            "prompt": .string(keyframePrompt(for: shot, in: movie, references: refs)),
            "negative": .string(MovieStyle.named(movie.style).negative), "seed": .int(seed),
            "width": .int(size.width), "height": .int(size.height), "server": .string(server),
        ]
        for (i, r) in refs.enumerated() {
            values[pictureSlots[i]] = .string(folder.appendingPathComponent(r.file).path)
        }
        let unused = refs.isEmpty ? [] : Set(pictureSlots.dropFirst(refs.count))
        return (Render(values: values, unusedPictures: unused), refs)
    }
}

extension MovieCharacter {
    public var selectedSheetFile: String? {
        sheets.first { $0.id == selectedSheet }?.file
    }

    /// What the script says about them: name, look and voice (not their sheets).
    public var scriptPart: [String] { [id, name, look, voice] }
}

extension Shot {
    public var selectedKeyframeFile: String? {
        keyframes.first { $0.id == selectedKeyframe }?.file
    }

    /// The shot as planned, without what was generated for it (what a shot list edit changes).
    public var planPart: Shot {
        var s = self
        s.keyframes = []
        s.selectedKeyframe = nil
        s.videos = []
        s.selectedVideo = nil
        s.trimStart = 0
        s.trimEnd = 0
        s.sounds = []
        s.selectedSound = nil
        for i in s.dialogue.indices {
            s.dialogue[i].takes = []
            s.dialogue[i].selectedTake = nil
        }
        return s
    }
}

extension MovieProject {
    /// The shot just before this one in the film (any scene).
    public func shot(before shot: Shot, index: [UUID: Int]? = nil) -> Shot? {
        guard let i = index?[shot.id] ?? shots.firstIndex(where: { $0.id == shot.id }), i > 0 else { return nil }
        return shots[i - 1]
    }

    /// Every character seen on screen has a chosen reference sheet (a narrator has none). Places' plates are
    /// optional.
    public var castIsReady: Bool {
        let seen = characters.filter { !$0.isNarrator }
        return !seen.isEmpty && seen.allSatisfy { $0.selectedSheetFile != nil }
    }

    /// Every shot has a chosen keyframe.
    public var keyframesAreReady: Bool { keyframesAreReady(in: nil) }

    /// Every shot of a sequence (nil: the film) has a chosen keyframe.
    public func keyframesAreReady(in sequence: String?) -> Bool {
        let s = shots(inSequence: sequence)
        return !s.isEmpty && s.allSatisfy { $0.selectedKeyframeFile != nil }
    }
}
