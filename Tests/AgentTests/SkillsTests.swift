import Foundation
import Testing
@testable import Agent
@testable import LLMClient

private func tempDir() throws -> URL {
    let d = FileManager.default.temporaryDirectory.appendingPathComponent("skills-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    return d
}

@discardableResult
private func writeSkill(_ base: URL, _ folder: String, _ text: String) throws -> URL {
    let d = base.appendingPathComponent(folder)
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    try text.write(to: d.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)
    return d
}

private func skill(_ name: String, _ desc: String = "d", body: String = "BODY", user: Bool = true, model: Bool = true) -> Skill {
    Skill(name: name, description: desc, folder: URL(fileURLWithPath: "/skills/\(name)"), body: body,
          userInvocable: user, modelInvocable: model, source: .personal)
}

struct SkillParseTests {
    let folder = URL(fileURLWithPath: "/x/my-skill")

    @Test func quotesAndBody() {
        let s = SkillStore.parse("---\nname: \"fancy\"\n# comment\ndescription: 'Does things'\nextra: 1\n---\n\n# Title\nText\n\n",
                                 folder: folder, source: .claude)
        #expect(s?.name == "fancy" && s?.description == "Does things")
        #expect(s?.body == "# Title\nText" && s?.source == .claude && s?.folder == folder)
        #expect(s?.userInvocable == true && s?.modelInvocable == true)
    }

    @Test func foldedAndLiteralBlocks() {
        let s = SkillStore.parse("---\ndescription: >\n  first line\n  second line\nname: n\n---\nb", folder: folder, source: .personal)
        #expect(s?.description == "first line second line" && s?.name == "n")
        let l = SkillStore.parse("---\ndescription: |-\n  one\n  two\n---\nb", folder: folder, source: .personal)
        #expect(l?.description == "one\ntwo")
    }

    @Test func missingPartsAndFlags() {
        #expect(SkillStore.parse("---\ndescription: d\n---\nb", folder: folder, source: .personal)?.name == "my-skill")
        #expect(SkillStore.parse("---\nname: x\n---\nb", folder: folder, source: .personal) == nil)
        #expect(SkillStore.parse("---\nname: x\ndescription:\n---\nb", folder: folder, source: .personal) == nil)
        #expect(SkillStore.parse("# no frontmatter\ndescription: d", folder: folder, source: .personal) == nil)
        #expect(SkillStore.parse("---\ndescription: d\nno end", folder: folder, source: .personal) == nil)
        let f = SkillStore.parse("---\ndescription: d\nuser-invocable: false\ndisable-model-invocation: true\n---\n", folder: folder, source: .personal)
        #expect(f?.userInvocable == false && f?.modelInvocable == false && f?.body == "")
    }
}

struct SkillLoadTests {
    @Test func laterSourcesReplaceEarlierOnes() throws {
        let base = try tempDir()
        let personal = base.appendingPathComponent("personal"), root = base.appendingPathComponent("proj")
        try writeSkill(personal, "a", "---\ndescription: personal a\n---\npa")
        try writeSkill(personal, "b", "---\ndescription: personal b\n---\npb")
        try writeSkill(root.appendingPathComponent(".claude/skills"), "a", "---\ndescription: claude a\n---\nca")
        try writeSkill(root.appendingPathComponent(".claude/skills"), "c", "---\nname: cc\ndescription: claude c\n---\ncc")
        try writeSkill(root.appendingPathComponent(".aistudio/skills"), "a", "---\ndescription: project a\n---\nxa")
        try writeSkill(root.appendingPathComponent(".aistudio/skills"), "bad", "no frontmatter")
        let all = SkillStore.load(root: root, personal: personal)
        #expect(all.map(\.name) == ["a", "b", "cc"])
        #expect(all[0].description == "project a" && all[0].source == .project && all[0].body == "xa")
        #expect(all[1].source == .personal && all[2].source == .claude)
        #expect(SkillStore.load(root: root, personal: personal, disabled: ["a", "cc"]).map(\.name) == ["b"])
        #expect(SkillStore.load(root: nil, personal: personal).map(\.name) == ["a", "b"])
        #expect(SkillStore.load(root: nil, personal: base.appendingPathComponent("missing")).isEmpty)
    }
}

struct SkillPromptTests {
    @Test func emptyAndFiltered() {
        #expect(SkillStore.promptSection([]) == "")
        #expect(SkillStore.promptSection([skill("a", model: false)]) == "")
        let s = SkillStore.promptSection([skill("a", "does a"), skill("hidden", model: false), skill("b", "line1\nline2")])
        #expect(s == "# Skills\n\nSkills are instructions for particular kinds of tasks. When a task matches one below, call use_skill with its name first and follow it.\n\n- a: does a\n- b: line1 line2")
    }

    @Test func longDescriptionsAreCut() {
        let s = SkillStore.promptSection([skill("a", String(repeating: "x", count: 400))])
        let line = s.components(separatedBy: "\n").last!
        #expect(line == "- a: " + String(repeating: "x", count: 249) + "…")
    }

    @Test func theSectionIsCapped() {
        let many = (0..<100).map { skill("skill-\($0)", String(repeating: "y", count: 240)) }
        let s = SkillStore.promptSection(many)
        #expect(s.count <= 8_000)
        let last = s.components(separatedBy: "\n").last!
        #expect(last.hasPrefix("- … and ") && last.hasSuffix(" more (call use_skill with a name to load one)"))
        let listed = s.components(separatedBy: "\n").filter { $0.hasPrefix("- skill-") }.count
        #expect(last == "- … and \(100 - listed) more (call use_skill with a name to load one)" && listed > 20)
    }
}

struct SkillExpandTests {
    @Test func expansion() {
        let skills = [skill("foo", body: "Do foo"), skill("foobar", body: "Do foobar"), skill("hid", user: false)]
        #expect(SkillStore.expand("/foo", skills: skills) == "<skill name=\"foo\" folder=\"/skills/foo\">\nDo foo\n</skill>\n\nFollow the skill above.")
        #expect(SkillStore.expand("  /foo  fix the bug\nnow ", skills: skills)?.hasSuffix("</skill>\n\nfix the bug\nnow") == true)
        #expect(SkillStore.expand("/foobar x", skills: skills)?.contains("Do foobar") == true)
        #expect(SkillStore.expand("/foobaz", skills: skills) == nil)
        #expect(SkillStore.expand("/fo", skills: skills) == nil)
        #expect(SkillStore.expand("/hid", skills: skills) == nil)
        #expect(SkillStore.expand("foo", skills: skills) == nil)
        #expect(SkillStore.expand("/nope", skills: skills) == nil)
    }
}

struct UseSkillToolTests {
    @Test func outputAndUnknown() async throws {
        let tool = UseSkillTool(skills: [skill("b"), skill("a", body: "Steps")])
        let ctx = ToolContext(root: try tempDir())
        let ok = try await tool.run(ToolArgs(["name": "a"]), ctx)
        #expect(!ok.isError)
        #expect(ok.text == "Skill folder: /skills/a\n(Paths in the skill are relative to this folder: read its files with read_file, run its scripts with run_command.)\n\nSteps")
        let withArgs = try await tool.run(ToolArgs(["name": "a", "args": "the parser"]), ctx)
        #expect(withArgs.text.hasSuffix("Steps\n\nArguments: the parser"))
        let bad = try await tool.run(ToolArgs(["name": "zzz"]), ctx)
        #expect(bad.isError && bad.text == "There is no skill named zzz. The skills are: a, b.")
        let none = try await UseSkillTool(skills: []).run(ToolArgs(["name": "zzz"]), ctx)
        #expect(none.isError && none.text == "There are no skills.")
        let p = try tool.preview(ToolArgs(["name": "a"]), ctx)
        #expect(p.title == "Use skill a" && p.kind == .read && p.paths == [URL(fileURLWithPath: "/skills/a")])
        #expect(try tool.preview(ToolArgs(["name": "zzz"]), ctx).paths.isEmpty)
        #expect(tool.name == "use_skill" && tool.effect == .read)
    }

    @Test func skillFolderReadsDontAskButOtherOutsideReadsDo() async throws {
        let root = try tempProject()
        let base = try tempDir()
        let folder = try writeSkill(base, "s", "---\ndescription: d\n---\nbody")
        try "helper".write(to: folder.appendingPathComponent("helper.txt"), atomically: true, encoding: .utf8)
        let other = base.appendingPathComponent("other.txt")
        try "secret".write(to: other, atomically: true, encoding: .utf8)
        let ctx = ToolContext(root: root)
        ctx.readRoots = [folder]
        let model = ScriptedModel([
            [call("read_file", ["path": folder.appendingPathComponent("helper.txt").path])],
            [call("read_file", ["path": other.path], id: "c2")],
            [.content("done")],
        ])
        let rec = Recorder()
        _ = try await AgentLoop.run(client: model, messages: [.user("x")], config: AgentConfig(model: "m"), context: ctx,
                                    ask: { r in rec.ask(r); return .allow }, events: { _ in })
        #expect(rec.asks.count == 1 && rec.asks[0].preview.paths.map(\.lastPathComponent) == ["other.txt"])
        #expect(model.requests[1].messages.last?.text.contains("helper") == true)
    }
}
