import CharacterCore
import SwiftUI

/// How a character's reply shows in the transcript. Media are generated in a later update; for now a tag shows as a
/// placeholder line.
enum CharacterDisplay {
    static func markdown(_ text: String) -> String {
        func label(_ k: CharacterCore.MediaKind) -> (icon: String, name: String) {
            switch k {
            case .photo: ("📷", "Photo")
            case .video: ("🎬", "Video")
            case .voice: ("🎙", "Voice message")
            case .talk: ("🗣", "Video message")
            }
        }
        return MediaTags.segments(text).map { seg -> String in
            switch seg {
            case .text(let s): return s
            case .media(let r):
                let l = label(r.kind)
                return "> \(l.icon) *\(l.name):* \(r.text)"
            case .pending(let k):
                let l = label(k)
                return "> \(l.icon) *Sending a \(l.name.lowercased())…*"
            }
        }.joined(separator: "\n\n")
    }

    /// A stable colour for a name, from the palette hues (Swift's own hash changes on every launch).
    static func hue(_ name: String) -> Color {
        let hues = [MG.hueBlue, MG.huePurple, MG.huePink, MG.hueOrange, MG.hueGreen, MG.hueTeal, MG.hueIndigo, MG.hueMint]
        let sum = name.unicodeScalars.reduce(0) { ($0 &* 31 &+ Int($1.value)) % 100_003 }
        return hues[sum % hues.count]
    }
}

/// A circle with the name's first letter on a colour from the name.
struct CharacterAvatar: View {
    let name: String
    var size: CGFloat = 32

    var body: some View {
        let hue = CharacterDisplay.hue(name)
        Text(String(name.trimmingCharacters(in: .whitespaces).first ?? "?").uppercased())
            .font(.system(size: size * 0.46, weight: .semibold))
            .foregroundStyle(hue)
            .frame(width: size, height: size)
            .background(Circle().fill(hue.opacity(0.18)))
            .overlay(Circle().strokeBorder(hue.opacity(0.5), lineWidth: 1))
    }
}
