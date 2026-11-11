import Foundation
import LLMClient

/// A skill: a folder with a `SKILL.md` (frontmatter, then instructions) and maybe other files beside it.
public struct Skill: Equatable, Sendable {
    public enum Source: String, Sendable { case personal, claude, project }
    public var name: String
    public var description: String
    public var folder: URL
    /// The instructions: everything after the frontmatter.
    public var body: String
    /// The user can run it by typing `/name`.
    public var userInvocable: Bool
    /// The model may load it with `use_skill`, and it is listed in the system prompt.
    public var modelInvocable: Bool
    public var source: Source

    public init(name: String, description: String, folder: URL, body: String, userInvocable: Bool = true,
                modelInvocable: Bool = true, source: Source) {
        self.name = name
        self.description = description
        self.folder = folder
        self.body = body
        self.userInvocable = userInvocable
        self.modelInvocable = modelInvocable
        self.source = source
    }
}

public enum SkillStore {
    /// ~/Library/Application Support/AIStudio/skills
    public static var personalFolder: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        return base.appendingPathComponent("AIStudio/skills", isDirectory: true)
    }

    /// Where skills are looked for, in order: personal, then the project's `.claude/skills`, then `.aistudio/skills`.
    public static func folders(root: URL?, personal: URL = personalFolder) -> [(url: URL, source: Skill.Source)] {
        var out: [(url: URL, source: Skill.Source)] = [(personal, .personal)]
        if let root {
            out.append((root.appendingPathComponent(".claude/skills", isDirectory: true), .claude))
            out.append((root.appendingPathComponent(".aistudio/skills", isDirectory: true), .project))
        }
        return out
    }

    /// Every skill found, a later source replacing an earlier one of the same name; sorted by name.
    public static func load(root: URL?, personal: URL = personalFolder, disabled: Set<String> = []) -> [Skill] {
        let fm = FileManager.default
        var byName: [String: Skill] = [:]
        for (folder, source) in folders(root: root, personal: personal) {
            let subs = ((try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil,
                                                     options: [.skipsHiddenFiles])) ?? [])
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
            for sub in subs {
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: sub.path, isDirectory: &isDir), isDir.boolValue else { continue }
                guard let data = fm.contents(atPath: sub.appendingPathComponent("SKILL.md").path),
                      let text = String(data: data, encoding: .utf8),
                      let skill = parse(text, folder: sub, source: source) else { continue }
                byName[skill.name] = skill
            }
        }
        return byName.values.filter { !disabled.contains($0.name) }.sorted { $0.name < $1.name }
    }

    /// A `SKILL.md`: frontmatter between two `---` lines, then the instructions. Nil when it has no frontmatter or
    /// no description.
    public static func parse(_ text: String, folder: URL, source: Skill.Source) -> Skill? {
        let lines = text.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        guard lines.first == "---", let end = lines.dropFirst().firstIndex(of: "---") else { return nil }
        var fields: [String: String] = [:]
        var i = 1
        while i < end {
            let line = lines[i]
            i += 1
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") || line.first == " " || line.first == "\t" { continue }
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces)
            var value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            if [">", ">-", "|", "|-"].contains(value) {
                var block: [String] = []
                while i < end, lines[i].first == " " || lines[i].first == "\t" || lines[i].trimmingCharacters(in: .whitespaces).isEmpty {
                    let t = lines[i].trimmingCharacters(in: .whitespaces)
                    if !t.isEmpty { block.append(t) }
                    i += 1
                }
                value = block.joined(separator: value.hasPrefix(">") ? " " : "\n")
            } else if value.count >= 2, let f = value.first, f == "\"" || f == "'", value.last == f {
                value = String(value.dropFirst().dropLast())
            }
            fields[key] = value
        }
        guard let description = fields["description"], !description.isEmpty else { return nil }
        let name = fields["name"].flatMap { $0.isEmpty ? nil : $0 } ?? folder.lastPathComponent
        let body = lines[(end + 1)...].joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return Skill(name: name, description: description, folder: folder, body: body,
                     userInvocable: fields["user-invocable"] != "false",
                     modelInvocable: fields["disable-model-invocation"] != "true", source: source)
    }

    /// The system prompt's list of skills ("" when the model can't use any).
    public static func promptSection(_ skills: [Skill]) -> String {
        let usable = skills.filter(\.modelInvocable)
        if usable.isEmpty { return "" }
        let header = "# Skills\n\nSkills are instructions for particular kinds of tasks. When a task matches one below, "
            + "call use_skill with its name first and follow it.\n"
        let lines = usable.map { s -> String in
            var d = s.description.replacingOccurrences(of: "\n", with: " ")
            if d.count > 250 { d = String(d.prefix(249)) + "…" }
            return "\n- \(s.name): \(d)"
        }
        let limit = 8_000
        if header.count + lines.reduce(0, { $0 + $1.count }) <= limit { return header + lines.joined() }
        func tail(_ n: Int) -> String { "\n- … and \(n) more (call use_skill with a name to load one)" }
        var out = header
        var kept = 0
        while kept < lines.count, out.count + lines[kept].count + tail(lines.count - kept - 1).count <= limit {
            out += lines[kept]
            kept += 1
        }
        return out + tail(lines.count - kept)
    }

    /// `/name args` typed by the user: the skill's instructions, then the arguments. Nil when it isn't a skill call.
    public static func expand(_ text: String, skills: [Skill]) -> String? {
        let t = String(text.drop(while: { $0.isWhitespace }))
        guard t.hasPrefix("/") else { return nil }
        let matches = skills.filter { s in
            guard s.userInvocable, t.hasPrefix("/" + s.name) else { return false }
            let next = t.dropFirst(1 + s.name.count).first
            return next == nil || next!.isWhitespace
        }
        guard let skill = matches.max(by: { $0.name.count < $1.name.count }) else { return nil }
        let rest = t.dropFirst(1 + skill.name.count).trimmingCharacters(in: .whitespacesAndNewlines)
        return "<skill name=\"\(skill.name)\" folder=\"\(skill.folder.path)\">\n\(skill.body)\n</skill>\n\n"
            + (rest.isEmpty ? "Follow the skill above." : rest)
    }
}

/// `use_skill`: loads a skill's instructions for the model.
public struct UseSkillTool: AgentTool {
    let skills: [Skill]
    public init(skills: [Skill]) { self.skills = skills }
    public let name = "use_skill"
    public let description = "Load a skill's instructions by its name (the skills are listed in the system prompt) and follow them. Paths in a skill are relative to its folder."
    public var parameters: JSONValue {
        schema(["name": ("string", "The skill's name"), "args": ("string", "What the skill should work on")], required: ["name"])
    }
    public let effect = ToolEffect.read

    public func preview(_ args: ToolArgs, _ ctx: ToolContext) throws -> ToolPreview {
        let n = try args.required("name")
        let skill = skills.first { $0.name == n }
        return ToolPreview(kind: .read, title: "Use skill \(n)", paths: skill.map { [$0.folder] } ?? [])
    }

    public func run(_ args: ToolArgs, _ ctx: ToolContext) async throws -> ToolOutput {
        let n = try args.required("name")
        guard let skill = skills.first(where: { $0.name == n }) else {
            let names = skills.map(\.name).sorted().joined(separator: ", ")
            return ToolOutput(skills.isEmpty ? "There are no skills." : "There is no skill named \(n). The skills are: \(names).",
                              isError: true)
        }
        var out = "Skill folder: \(skill.folder.path)\n(Paths in the skill are relative to this folder: read its files with read_file, run its scripts with run_command.)\n\n\(skill.body)"
        if let a = args.string("args"), !a.isEmpty { out += "\n\nArguments: \(a)" }
        return ToolOutput(out)
    }
}
