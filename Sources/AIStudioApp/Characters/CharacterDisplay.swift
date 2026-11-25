import CharacterCore
import LLMClient
import MovieCore
import SwiftUI

/// How a character's reply shows in the transcript: a media tag is its photo, video or voice message once made,
/// else a line about where it is.
enum CharacterDisplay {
    static func markdown(_ text: String) -> String {
        markdown(ChatTurn(role: .assistant, text: text))
    }

    static func label(_ k: CharacterCore.MediaKind) -> (icon: String, name: String, making: String) {
        switch k {
        case .photo: ("📷", "Photo", "Taking a photo…")
        case .video: ("🎬", "Video", "Filming a video…")
        case .voice: ("🎙", "Voice message", "Recording a voice message…")
        case .talk: ("🗣", "Video message", "Filming a video message…")
        }
    }

    /// Text that can stand as a picture's alt text: no brackets, one line.
    private static func alt(_ s: String) -> String {
        s.replacingOccurrences(of: "[\\[\\]\\n]+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
    }

    /// `phases`: what each media being made is doing (`ChatDocument.mediaPhase`), shown on its line.
    static func markdown(_ turn: ChatTurn, phases: [UUID: String] = [:]) -> String {
        var index = 0
        return MediaTags.segments(turn.text).map { seg -> String in
            switch seg {
            case .text(let s): return s
            case .media(let r):
                let l = label(r.kind)
                defer { index += 1 }
                if let media = turn.media, media.indices.contains(index) {
                    let m = media[index]
                    switch m.status {
                    case .done:
                        // A voice message shows its words under the player; a video message its words too.
                        if let file = m.file {
                            return r.kind == .voice || r.kind == .talk ? "![\(alt(r.text))](\(file))" : "![\(l.name)](\(file))"
                        }
                    case .running:
                        return "> \(l.icon) *\(l.making)*" + (phases[m.id].map { " \($0)" } ?? "")
                    case .waiting:
                        return "> \(l.icon) *\(l.name) · about \(m.estimateUSD.map { CostEstimate.format($0) } ?? "?") — waiting for your OK*"
                    case .failed:
                        let what = r.kind == .photo ? "take that photo" : "make that \(l.name.lowercased())"
                        return "> \(l.icon) *Couldn't \(what): \(m.error ?? "unknown error")*"
                    case .skipped:
                        return "> \(l.icon) *\(l.name) skipped\(m.error.map { ": \($0)" } ?? "")*"
                    }
                }
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
