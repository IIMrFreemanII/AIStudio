import CharacterCore
import LLMClient
import MovieCore
import SwiftUI

/// How a character's reply shows in the transcript: a photo tag is its picture once taken, else a line about where it
/// is; the other kinds are placeholder lines for now.
enum CharacterDisplay {
    static func markdown(_ text: String) -> String {
        markdown(ChatTurn(role: .assistant, text: text))
    }

    static func markdown(_ turn: ChatTurn) -> String {
        func label(_ k: CharacterCore.MediaKind) -> (icon: String, name: String) {
            switch k {
            case .photo: ("📷", "Photo")
            case .video: ("🎬", "Video")
            case .voice: ("🎙", "Voice message")
            case .talk: ("🗣", "Video message")
            }
        }
        var index = 0
        return MediaTags.segments(turn.text).map { seg -> String in
            switch seg {
            case .text(let s): return s
            case .media(let r):
                let l = label(r.kind)
                defer { index += 1 }
                if r.kind == .photo, let media = turn.media, media.indices.contains(index) {
                    let m = media[index]
                    switch m.status {
                    case .done:
                        if let file = m.file { return "![Photo](\(file))" }
                    case .running: return "> 📷 *Taking a photo…*"
                    case .waiting:
                        return "> 📷 *Photo · about \(m.estimateUSD.map { CostEstimate.format($0) } ?? "?") — waiting for your OK*"
                    case .failed: return "> 📷 *Couldn't take that photo: \(m.error ?? "unknown error")*"
                    case .skipped: return "> 📷 *Photo skipped*"
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
