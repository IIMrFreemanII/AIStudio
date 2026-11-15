import Foundation

/// A movie's look: what every keyframe prompt ends with, what to avoid, and which image model makes the keyframes.
public struct MovieStyle: Identifiable, Sendable, Equatable {
    public var id: String
    public var name: String
    /// One line for the LLM: how to describe shots in this style.
    public var brief: String
    /// Appended to every keyframe prompt.
    public var keyframeSuffix: String
    public var negative: String
    /// Appended to every motion prompt.
    public var motionSuffix: String

    public static let cinematic = MovieStyle(
        id: "cinematic", name: "Cinematic",
        brief: "photoreal live-action film, natural light, shallow depth of field, anamorphic framing",
        keyframeSuffix: "cinematic film still, 35mm, natural lighting, shallow depth of field, film grain, highly detailed",
        negative: "cartoon, illustration, text, watermark, deformed hands, extra fingers, blurry",
        motionSuffix: "smooth cinematic camera motion, natural movement")

    public static let anime = MovieStyle(
        id: "anime", name: "Anime",
        brief: "hand-drawn Japanese animation, clean line art, painted backgrounds, expressive faces",
        keyframeSuffix: "anime key visual, clean line art, cel shading, painted background, vibrant colors",
        negative: "photo, realistic, 3d render, text, watermark, deformed",
        motionSuffix: "anime animation, fluid character motion")

    public static let animated3D = MovieStyle(
        id: "3d", name: "3D Animation",
        brief: "feature-film 3D animation, stylised characters, soft global illumination, warm palette",
        keyframeSuffix: "3d animated film still, stylized characters, soft global illumination, subsurface scattering, octane render",
        negative: "photo, 2d, sketch, text, watermark, deformed",
        motionSuffix: "3d animation, smooth character animation")

    public static let pixel = MovieStyle(
        id: "pixel", name: "Pixel Art",
        brief: "retro 16-bit pixel art game cutscene, limited palette, side or three-quarter views",
        keyframeSuffix: "pixel art, 16-bit, limited palette, crisp pixels",
        negative: "photo, realistic, blurry, smooth gradients, text, watermark",
        motionSuffix: "pixel art animation")

    public static let noir = MovieStyle(
        id: "noir", name: "Film Noir",
        brief: "black-and-white 1940s noir, hard shadows, venetian-blind light, rain-slick streets",
        keyframeSuffix: "black and white film noir still, high contrast, hard shadows, 1940s, film grain",
        negative: "color, cartoon, text, watermark, deformed",
        motionSuffix: "moody noir camera motion")

    public static let all: [MovieStyle] = [.cinematic, .anime, .animated3D, .pixel, .noir]

    public static func named(_ id: String) -> MovieStyle { all.first { $0.id == id } ?? .cinematic }
}
