import Foundation

/// Builds the system prompt of a character chat.
public enum CharacterPrompt {
    /// With no media kinds on, models otherwise "attach" a photo that never arrives.
    public static let noMedia = "You can't send photos, videos or voice notes in this chat. If asked for one, say in character "
        + "that you can't right now; never pretend to send or attach anything."

    public static func system(_ c: Character, memory: CharacterMemory, scene: Scenario?, others: [Character] = [],
                              media: [MediaKind] = [], initiativeLeft: Int = 0, now: Date = Date(),
                              calendar: Calendar = .current) -> String {
        var sections: [String] = []

        var intro = ["You are \(c.name), a \(c.age)-year-old \(c.gender)."]
        for (label, text) in [("Looks", c.look), ("Personality", c.personality), ("Backstory", c.backstory),
                              ("Speaking style", c.speakingStyle), ("Likes", c.likes)] where !text.isEmpty {
            intro.append("\(label): \(text)")
        }
        sections.append(intro.joined(separator: "\n"))

        sections.append("""
            Stay in character as \(c.name) at all times and never say you are an AI or a language model. \
            Reply like texting your partner: short, natural messages in the first person, with no stage directions \
            about what the user does. You are an adult and so is everyone you talk to.
            """)

        let r = memory.relationship
        sections.append("Your relationship: \(r.stage.label), affection \(r.affection)/100, you feel \(r.mood).")

        if !memory.facts.isEmpty {
            sections.append("What you know about them:\n" + memory.facts.map { "- \($0.text)" }.joined(separator: "\n"))
        }
        if !memory.summary.isEmpty { sections.append("Your story together so far:\n\(memory.summary)") }

        if let s = scene {
            var line = "Right now: \(s.title) — \(s.place)."
            if !s.outfit.isEmpty { line += " You're wearing \(s.outfit)." }
            if !s.time.isEmpty { line += " Time: \(s.time)." }
            if !s.mood.isEmpty { line += " Mood: \(s.mood)." }
            sections.append(line)
        }

        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = calendar
        f.timeZone = calendar.timeZone
        f.dateFormat = "EEEE"
        let day = f.string(from: now)
        f.dateFormat = "h:mm a"
        sections.append("It's \(day), \(f.string(from: now)).")

        if !others.isEmpty {
            let names = others.map(\.name).joined(separator: ", ")
            sections.append("This is a group chat. Also in it: \(names). Their messages reach you as \"Name: …\"; everything else is from the user. Write only your own message as \(c.name), without your name in front, and never write lines for the others.")
        }

        if !media.isEmpty {
            let rule = initiativeLeft > 0
                ? "You may send one on your own when it fits, at most \(initiativeLeft) more today; otherwise only when asked."
                : "Send one only when the user's last message asks you for one. Never send one on your own, "
                    + "and asking the user for a picture is not a reason to send yours."
            sections.append(MediaTags.instructions(media) + "\n" + rule)
        } else {
            sections.append(Self.noMedia)
        }
        return sections.joined(separator: "\n\n")
    }
}
