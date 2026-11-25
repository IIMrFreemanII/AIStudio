import Foundation

/// The character texting first after time away.
public enum CheckIn {
    public static let choices = [0, 4, 8, 24]

    public static func label(_ hours: Int) -> String {
        switch hours {
        case 0: return "Never"
        case 24: return "After a day"
        case 1: return "After 1 hour"
        default: return "After \(hours) hours"
        }
    }

    /// True when the chat is due a message from the character.
    public static func due(hours: Int, last: Date?, lastIsUser: Bool, now: Date = Date()) -> Bool {
        guard hours > 0, let last, !lastIsUser else { return false }
        return now.timeIntervalSince(last) >= Double(hours) * 3600
    }

    /// The note the model gets in place of a user message.
    public static func nudge(name: String, away: TimeInterval) -> String {
        let span: String
        if away < 2 * 3600 {
            span = "About an hour"
        } else if away < 48 * 3600 {
            span = "About \(Int(away / 3600)) hours"
        } else {
            span = "About \(Int(away / 86400)) days"
        }
        return "[\(span) since your last messages. Text them first, as \(name): say what you've been up to, or pick up where you left off. One or two short messages; don't mention this note.]"
    }
}
