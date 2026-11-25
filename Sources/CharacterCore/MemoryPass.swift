import Agent
import Foundation
import LLMClient
import MovieCore

/// The memory keeper: every few user turns a model reads the recent chat and updates what the character remembers.
public enum MemoryPass {
    /// User turns between passes.
    public static let every = 6

    /// The memory keeper's instructions.
    public static func system(_ c: Character) -> String {
        """
        You are the memory keeper for \(c.name), a fictional adult companion character in a chat with the user. \
        You read a stretch of their chat and update \(c.name)'s memory. Reply with only JSON, no other text, in \
        this shape:
        {"new_facts": [...], "forget": [...], "affection_delta": 0, "stage": "strangers", "mood": "curious", "summary": "..."}
        - new_facts: short third-person sentences about the user worth remembering ("The user has a cat called Miso"). \
        Only things the user said or clearly showed; nothing invented, nothing already known.
        - forget: the exact text of known facts that are now wrong or were taken back.
        - affection_delta: how much \(c.name)'s affection for the user changed, an integer from -10 to 10 (0 when nothing changed).
        - stage: the relationship now, one of strangers, flirting, dating, partners.
        - mood: one or two words for how \(c.name) feels now.
        - summary: the whole updated story so far in at most 200 words, merging the previous summary with what happened.
        """
    }

    /// The request. Its first line is "Task: character memory".
    public static func user(_ c: Character, memory: CharacterMemory,
                            recent: [(speaker: String, text: String)]) -> String {
        var out = "Task: character memory\nCharacter: \(c.name)\n\n"
        out += "Known facts:\n"
        out += memory.facts.isEmpty ? "(none)\n" : memory.facts.map { "- \($0.text)" }.joined(separator: "\n") + "\n"
        let r = memory.relationship
        out += "\nRelationship: stage \(r.stage.rawValue), affection \(r.affection)/100, mood \(r.mood)\n"
        out += "\nPrevious summary:\n\(memory.summary.isEmpty ? "(none)" : memory.summary)\n"
        out += "\nRecent chat:\n"
        out += recent.map { "\($0.speaker): \($0.text)" }.joined(separator: "\n")
        return out
    }

    /// Parses a reply: strips ``` fences, takes the first `{` to the last `}`, decodes it.
    public static func parse(_ text: String) throws -> MemoryUpdate {
        var s = text
        s = s.replacingOccurrences(of: "```json", with: "").replacingOccurrences(of: "```", with: "")
        guard let open = s.firstIndex(of: "{"), let close = s.lastIndex(of: "}"), open < close else {
            throw MoviePlannerError.noJSON
        }
        let json = String(s[open...close])
        do {
            return try JSONDecoder().decode(MemoryUpdate.self, from: Data(json.utf8))
        } catch {
            throw MoviePlannerError.invalid("\(error)")
        }
    }

    /// Asks the model (one retry on a parse error) and returns the update with the summed cost.
    public static func run(client: any ChatStreaming, model: String, character: Character, memory: CharacterMemory,
                           recent: [(speaker: String, text: String)]) async throws -> (update: MemoryUpdate, cost: Double) {
        let box = CostBox()
        let update: MemoryUpdate = try await MoviePlanner.ask(
            client: client, model: model, system: system(character),
            user: user(character, memory: memory, recent: recent), maxTokens: 800,
            onCost: { box.add($0) }, parse: { try parse($0) })
        return (update, box.total)
    }

    /// A system prompt for `Compactor.summarize` in character chats.
    public static let summarySystem = """
        Summarize this chat between the user and a companion character. Keep the relationship and how it has \
        changed, what happened, promises made, inside jokes, facts about the user and the current situation. Write \
        prose plus short bullets. Invent nothing.
        """
}

/// Sums costs reported from `@Sendable` closures.
final class CostBox: @unchecked Sendable {
    private let lock = NSLock()
    private var sum = 0.0
    func add(_ x: Double) { lock.lock(); sum += x; lock.unlock() }
    var total: Double { lock.lock(); defer { lock.unlock() }; return sum }
}
