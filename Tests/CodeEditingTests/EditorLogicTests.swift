import Testing
import Foundation
@testable import CodeEditing

// MARK: - Language detection

@Test func detectByExtension() {
    #expect(CodeLanguage.detect(path: "a/b/Main.swift") == .swift)
    #expect(CodeLanguage.detect(path: "x.PY") == .python)
    #expect(CodeLanguage.detect(path: "x.pyi") == .python)
    #expect(CodeLanguage.detect(path: "x.jsx") == .javascript)
    #expect(CodeLanguage.detect(path: "x.mts") == .typescript)
    #expect(CodeLanguage.detect(path: "x.tsx") == .tsx)
    #expect(CodeLanguage.detect(path: "Package.resolved") == .json)
    #expect(CodeLanguage.detect(path: "p.code-workspace") == .json)
    #expect(CodeLanguage.detect(path: "run.command") == .bash)
    #expect(CodeLanguage.detect(path: "x.h") == .c)
    #expect(CodeLanguage.detect(path: "x.mm") == .cpp)
    #expect(CodeLanguage.detect(path: "x.rs") == .rust)
    #expect(CodeLanguage.detect(path: "x.HTM") == .html)
    #expect(CodeLanguage.detect(path: "x.yml") == .yaml)
    #expect(CodeLanguage.detect(path: "README.mdx") == .markdown)
}

@Test func detectByFilename() {
    #expect(CodeLanguage.detect(path: "/home/u/.zshrc") == .bash)
    #expect(CodeLanguage.detect(path: ".envrc") == .bash)
    #expect(CodeLanguage.detect(path: "sub/Cargo.lock") == .toml)
    #expect(CodeLanguage.detect(path: "Pipfile") == .toml)
    #expect(CodeLanguage.detect(path: "Dockerfile") == nil)
    #expect(CodeLanguage.detect(path: "Makefile", firstLine: "#!/bin/bash") == nil)
    #expect(CodeLanguage.detect(path: ".prettierrc") == .json)
    #expect(CodeLanguage.detect(path: ".zshrc.bak") == nil)
}

@Test func detectByShebang() {
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/bin/sh") == .bash)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/env bash") == .bash)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/env -S zsh -f") == .bash)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/python3") == .python)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/env python") == .python)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/env node") == .javascript)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/swift") == .swift)
    #expect(CodeLanguage.detect(path: "run", firstLine: "#!/usr/bin/perl") == nil)
    #expect(CodeLanguage.detect(path: "run", firstLine: "echo hi") == nil)
    #expect(CodeLanguage.detect(path: "x.swift", firstLine: "#!/bin/sh") == .swift)
}

@Test func detectUnknown() {
    #expect(CodeLanguage.detect(path: "x.xyz") == nil)
    #expect(CodeLanguage.detect(path: "noext") == nil)
    #expect(CodeLanguage.detect(path: "") == nil)
}

@Test func fromFence() {
    #expect(CodeLanguage.fromFence("swift") == .swift)
    #expect(CodeLanguage.fromFence("Python title=\"x\"") == .python)
    #expect(CodeLanguage.fromFence("  JS") == .javascript)
    #expect(CodeLanguage.fromFence("console") == .bash)
    #expect(CodeLanguage.fromFence("objective-c") == .cpp)
    #expect(CodeLanguage.fromFence("golang") == .go)
    #expect(CodeLanguage.fromFence("xml") == nil)
    #expect(CodeLanguage.fromFence("") == nil)
    #expect(CodeLanguage.fromFence("klingon") == nil)
}

// MARK: - TextLines

@Test func textLinesCounts() {
    #expect(TextLines("").count == 1)
    #expect(TextLines("a").count == 1)
    #expect(TextLines("a\n").count == 2)
    #expect(TextLines("a\r\nb").count == 2)
    #expect(TextLines("a\rb\rc").count == 3)
    #expect(TextLines("a\r\n\r\n").count == 3)
    #expect(TextLines(NSString(string: "x\ny")).count == 2)
}

@Test func textLinesLineAt() {
    let t = TextLines("ab\ncd\r\nef\rgh")
    #expect(t.line(at: 0) == 1)
    #expect(t.line(at: 2) == 1)   // before the "\n"
    #expect(t.line(at: 3) == 2)   // right after "\n"
    #expect(t.line(at: 6) == 2)   // between \r and \n
    #expect(t.line(at: 7) == 3)
    #expect(t.line(at: 10) == 4)  // right after the lone \r
    #expect(t.line(at: 999) == 4)
    #expect(t.line(at: -5) == 1)
    #expect(TextLines("").line(at: 0) == 1)
}

@Test func textLinesRanges() {
    let t = TextLines("ab\ncd\r\nef\n")
    #expect(t.range(ofLine: 1) == NSRange(location: 0, length: 2))
    #expect(t.range(ofLine: 2) == NSRange(location: 3, length: 2))
    #expect(t.range(ofLine: 3) == NSRange(location: 7, length: 2))
    #expect(t.range(ofLine: 4) == NSRange(location: 10, length: 0))
    #expect(t.range(ofLine: 99) == NSRange(location: 10, length: 0))
    #expect(t.range(ofLine: 0) == NSRange(location: 0, length: 2))
    #expect(t.range(ofLines: 1...2) == NSRange(location: 0, length: 5))
    #expect(t.range(ofLines: 2...3) == NSRange(location: 3, length: 6))
}

@Test func textLinesSpan() {
    let t = TextLines("one\ntwo\nthree\nfour")
    #expect(t.lineSpan(of: NSRange(location: 5, length: 0)) == 2...2)
    #expect(t.lineSpan(of: NSRange(location: 5, length: 1)) == 2...2)      // within one line
    #expect(t.lineSpan(of: NSRange(location: 4, length: 10)) == 2...3)     // 4..14 ends at column 0 of line 4
    #expect(t.lineSpan(of: NSRange(location: 4, length: 11)) == 2...4)
    #expect(t.lineSpan(of: NSRange(location: 4, length: 4)) == 2...2)      // ends at column 0 of line 3
    #expect(t.lineSpan(of: NSRange(location: 0, length: 0)) == 1...1)
}

@Test func textLinesPositionAndOffset() {
    let t = TextLines("ab\nc😀d\n")
    #expect(t.position(of: 0) == (1, 0))
    #expect(t.position(of: 4) == (2, 1))
    #expect(t.position(of: 6) == (2, 3))   // after the emoji (2 UTF-16 units)
    #expect(t.position(of: 99) == (3, 0))
    #expect(t.offset(line: 2, column: 1) == 4)
    #expect(t.offset(line: 2, column: 99) == 7)
    #expect(t.offset(line: 1, column: 99) == 2)
    #expect(t.offset(line: 99, column: 5) == t.length)
    #expect(t.offset(line: 1, column: -3) == 0)
}

@Test func textLinesRemap() {
    let old = TextLines("aaa\nbbb\nccc")
    let new = TextLines("x\ny\naaa\nbbb\nccc")
    // Same (line, column): line 3 col 1..2 stays line 3 col 1..2, even though lines were inserted above.
    #expect(TextLines.remap(NSRange(location: 9, length: 1), from: old, to: new) == NSRange(location: 5, length: 1))
    // Columns past the end of the shorter new line are clamped (line 2 is "y" now).
    #expect(TextLines.remap(NSRange(location: 5, length: 2), from: old, to: new) == NSRange(location: 3, length: 0))
    // Clamped when the new text is shorter.
    let short = TextLines("a\nb")
    #expect(TextLines.remap(NSRange(location: 9, length: 2), from: old, to: short) == NSRange(location: 3, length: 0))
    // A caret stays a caret.
    #expect(TextLines.remap(NSRange(location: 1, length: 0), from: old, to: new) == NSRange(location: 1, length: 0))
}

// MARK: - EditorLimits

@Test func limitsBinary() {
    #expect(EditorLimits.mode(head: Data([0x61, 0, 0x62]), size: 3) == .binary)
    // A NUL past 8 KB doesn't count.
    var late = Data(repeating: 0x61, count: 9000); late.append(0)
    #expect(EditorLimits.mode(head: late, size: late.count) == .editable)
}

@Test func limitsSizes() {
    let head = Data("hello".utf8)
    #expect(EditorLimits.mode(head: head, size: 5) == .editable)
    #expect(EditorLimits.mode(head: head, size: EditorLimits.editableBytes) == .editable)
    guard case .readOnly(let big) = EditorLimits.mode(head: head, size: 5 << 20) else { Issue.record("expected readOnly"); return }
    #expect(big.hasPrefix("This file is ") && big.hasSuffix(", too big to edit here.") && big.contains("MB"))
    guard case .partial(let note) = EditorLimits.mode(head: head, size: 32 << 20) else { Issue.record("expected partial"); return }
    #expect(note.hasPrefix("Showing the first 2 MB of ") && note.hasSuffix("."))
    #expect(EditorLimits.mode(head: head, size: EditorLimits.readableBytes + 1) != EditorLimits.mode(head: head, size: 5 << 20))
}

@Test func limitsEncodingAndLongLines() {
    #expect(EditorLimits.mode(head: Data([0xFF, 0xFE, 0x41]), size: 3) == .readOnly("This file isn't UTF-8 text: read-only."))
    let long = Data((String(repeating: "x", count: 20_001)).utf8)
    #expect(EditorLimits.mode(head: long, size: long.count) == .plainEditable)
    let exact = Data((String(repeating: "x", count: 20_000) + "\n" + String(repeating: "y", count: 20_000)).utf8)
    #expect(EditorLimits.mode(head: exact, size: exact.count) == .editable)
}

// MARK: - Indentation

@Test func indentationDetect() {
    #expect(Indentation.detect("") == .spaces(4))
    #expect(Indentation.detect("a\nb\nc\n") == .spaces(4))
    #expect(Indentation.detect("func f() {\n\tx()\n\tif y {\n\t\tz()\n\t}\n}\n") == .tabs)
    #expect(Indentation.detect("a:\n  b:\n    c: 1\n  d: 2\ne:\n  f: 3\n") == .spaces(2))
    #expect(Indentation.detect("class A {\n    func f() {\n        x()\n    }\n}\n") == .spaces(4))
    #expect(Indentation.detect("if x:\n        y()\nif z:\n        w()\n") == .spaces(8))
    #expect(Indentation.detect("a\n    \n\n").unit == "    ")  // blank lines are ignored
    #expect(Indentation.tabs.unit == "\t")
    #expect(Indentation.spaces(2).unit == "  ")
}

// MARK: - FileTabList

@Test func tabsTransientReplacedInPlace() {
    var l = FileTabList()
    l.open("a", pin: true)
    l.open("b", pin: false)
    l.open("c", pin: true)
    #expect(l.tabs == ["a", "b", "c"])  // b opened after a; c after active b
    #expect(l.isTransient("b"))
    l.open("d", pin: false)
    #expect(l.tabs == ["a", "d", "c"])
    #expect(l.transient == "d" && l.active == "d")
    #expect(!l.isTransient("b"))
}

@Test func tabsInsertAfterActiveAndPin() {
    var l = FileTabList(tabs: ["a", "b", "c"], active: "a")
    l.open("x", pin: false)
    #expect(l.tabs == ["a", "x", "b", "c"])
    l.pin("x")
    #expect(l.transient == nil)
    l.active = "c"
    l.open("y", pin: true)
    #expect(l.tabs == ["a", "x", "b", "c", "y"])
    #expect(l.transient == nil)
    var e = FileTabList()
    e.open("only", pin: false)
    #expect(e.tabs == ["only"] && e.active == "only")
}

@Test func tabsOpenExistingActivates() {
    var l = FileTabList()
    l.open("a", pin: false)
    l.open("b", pin: true)
    l.open("a", pin: false)
    #expect(l.active == "a" && l.tabs == ["a", "b"] && l.isTransient("a"))
    l.open("a", pin: true)
    #expect(!l.isTransient("a"))
    l.open("b", pin: false)
    #expect(l.tabs == ["a", "b"] && l.active == "b" && l.transient == nil)
}

@Test func tabsClose() {
    var l = FileTabList(tabs: ["a", "b", "c"], active: "b")
    l.close("b"); #expect(l.active == "c" && l.tabs == ["a", "c"])
    l.close("c"); #expect(l.active == "a")
    l.close("a"); #expect(l.active == nil && l.tabs.isEmpty)
    var m = FileTabList(tabs: ["a", "b", "c"], active: "a")
    m.close("c"); #expect(m.active == "a")
    m.close("zzz"); #expect(m.tabs == ["a", "b"])
    var t = FileTabList()
    t.open("a", pin: false)
    t.close("a")
    #expect(t.transient == nil)
}

@Test func tabsRename() {
    var l = FileTabList()
    l.open("a", pin: true); l.open("b", pin: false); l.open("c", pin: true)
    l.rename("b", to: "b2")
    #expect(l.tabs == ["a", "b2", "c"] && l.transient == "b2")
    l.rename("c", to: "d")
    #expect(l.active == "d")
    l.rename("a", to: "d")  // target already open: old tab dropped
    #expect(l.tabs == ["b2", "d"])
}

@Test func tabsKeepOnly() {
    var l = FileTabList(tabs: ["a", "b", "c", "d"], active: "b")
    l.keepOnly(["a", "c", "d"])
    #expect(l.tabs == ["a", "c", "d"] && l.active == "c")
    l.active = "d"
    l.keepOnly(["a", "c"])
    #expect(l.active == "c")
    l.keepOnly([])
    #expect(l.tabs.isEmpty && l.active == nil)
    var t = FileTabList()
    t.open("a", pin: false)
    t.keepOnly([])
    #expect(t.transient == nil)
}

@Test func tabsCodable() throws {
    var l = FileTabList()
    l.open("a", pin: true); l.open("b", pin: false)
    let data = try JSONEncoder().encode(l)
    let back = try JSONDecoder().decode(FileTabList.self, from: data)
    #expect(back == l && back.transient == "b" && back.active == "b")
}

// MARK: - FileSync

private func stamp(_ n: Double, size: Int = 1) -> FileStamp {
    FileStamp(mtime: Date(timeIntervalSince1970: n), size: size, inode: 7)
}

@Test func syncDecide() {
    let base = FileSnapshot(stamp: stamp(1), text: "a\nb\nc\n")
    #expect(base.hash == FileSnapshot.hash(of: "a\nb\nc\n") && base.hash.count == 64)
    // deleted
    #expect(FileSync.decide(base: base, diskStamp: nil, diskText: nil, buffer: "x", dirty: true) == .deleted)
    // none
    #expect(FileSync.decide(base: base, diskStamp: stamp(1), diskText: nil, buffer: "x", dirty: true) == .none)
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: nil, buffer: "x", dirty: true) == .none)
    // adoptStamp: touched, same content
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: "a\nb\nc\n", buffer: "x", dirty: true) == .adoptStamp(stamp(2)))
    // reload: clean
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: "a\nB\nc\n", buffer: "a\nb\nc\n", dirty: false) == .reload("a\nB\nc\n"))
    // reload: disk equals buffer
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: "a\nb\nC\n", buffer: "a\nb\nC\n", dirty: true) == .reload("a\nb\nC\n"))
    // merged: disjoint
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: "a\nb\nC\n", buffer: "A\nb\nc\n", dirty: true) == .merged("A\nb\nC\n"))
    // conflict: same line
    #expect(FileSync.decide(base: base, diskStamp: stamp(2), diskText: "a\nX\nc\n", buffer: "a\nY\nc\n", dirty: true) == .conflict("a\nX\nc\n"))
}

@Test func syncCanSave() {
    let base = FileSnapshot(stamp: stamp(1), text: "a")
    #expect(FileSync.canSave(base: base, diskStamp: stamp(1)))
    #expect(!FileSync.canSave(base: base, diskStamp: stamp(2)))
    #expect(!FileSync.canSave(base: base, diskStamp: stamp(1, size: 2)))
    #expect(FileSync.canSave(base: base, diskStamp: nil))
}

@Test func fileStampOfRealFile() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("stamp-\(UUID().uuidString).txt")
    #expect(FileStamp.of(url) == nil)
    try "hello".write(to: url, atomically: true, encoding: .utf8)
    defer { try? FileManager.default.removeItem(at: url) }
    let s = try #require(FileStamp.of(url))
    #expect(s.size == 5 && s.inode != 0)
}

// MARK: - LineMerge

@Test func mergeShortcuts() {
    #expect(LineMerge.merge(base: "a\n", mine: "b\n", theirs: "b\n") == "b\n")
    #expect(LineMerge.merge(base: "a\n", mine: "a\n", theirs: "b\n") == "b\n")
    #expect(LineMerge.merge(base: "a\n", mine: "b\n", theirs: "a\n") == "b\n")
}

@Test func mergeDisjoint() {
    let base = "1\n2\n3\n4\n5\n"
    #expect(LineMerge.merge(base: base, mine: "1\nTWO\n3\n4\n5\n", theirs: "1\n2\n3\n4\nFIVE\n") == "1\nTWO\n3\n4\nFIVE\n")
    // theirs inserts, mine edits elsewhere
    #expect(LineMerge.merge(base: base, mine: "ONE\n2\n3\n4\n5\n", theirs: "1\n2\n3\nnew\n4\n5\n") == "ONE\n2\n3\nnew\n4\n5\n")
    // adjacent non-empty replacements merge
    #expect(LineMerge.merge(base: base, mine: "1\nA\n3\n4\n5\n", theirs: "1\n2\nB\n4\n5\n") == "1\nA\nB\n4\n5\n")
}

@Test func mergeIdenticalEdits() {
    let base = "1\n2\n3\n4\n"
    #expect(LineMerge.merge(base: base, mine: "1\nX\n3\n4\n", theirs: "1\nX\n3\nY\n") == "1\nX\n3\nY\n")
    #expect(LineMerge.merge(base: base, mine: "1\nX\n3\nA\n", theirs: "1\nX\n3\nB\n") == nil)
}

@Test func mergeOverlapNil() {
    let base = "1\n2\n3\n"
    #expect(LineMerge.merge(base: base, mine: "1\nA\n3\n", theirs: "1\nB\n3\n") == nil)
    // theirs deletes the line mine edits
    #expect(LineMerge.merge(base: base, mine: "1\nA\n3\n", theirs: "1\n3\n") == nil)
    // a range covering a line the other edited
    #expect(LineMerge.merge(base: base, mine: "X\n3\n", theirs: "1\nB\n3\n") == nil)
}

@Test func mergeInsertionsAtSamePoint() {
    let base = "1\n2\n"
    #expect(LineMerge.merge(base: base, mine: "1\nA\n2\n", theirs: "1\nB\n2\n") == nil)
    #expect(LineMerge.merge(base: base, mine: "1\nA\n2\n", theirs: "1\nA\n2\n") == "1\nA\n2\n")
    // insertion at a different point than the other's insertion
    #expect(LineMerge.merge(base: base, mine: "A\n1\n2\n", theirs: "1\n2\nB\n") == "A\n1\n2\nB\n")
}

@Test func mergeStartAndEnd() {
    let base = "1\n2\n3\n"
    #expect(LineMerge.merge(base: base, mine: "0\n1\n2\n3\n", theirs: "1\n2\n3\n4\n") == "0\n1\n2\n3\n4\n")
    #expect(LineMerge.merge(base: base, mine: "ONE\n2\n3\n", theirs: "1\n2\nTHREE\n") == "ONE\n2\nTHREE\n")
}

@Test func mergeMissingFinalNewline() {
    // base has no trailing newline; theirs edits the first line, mine appends a line.
    let base = "a\nb"
    let merged = LineMerge.merge(base: base, mine: "a\nb\nc", theirs: "A\nb")
    #expect(merged == "A\nb\nc")
    // neither touches the end: the lack of a final newline survives.
    #expect(LineMerge.merge(base: "a\nb\nc", mine: "A\nb\nc", theirs: "a\nb\nC") == "A\nb\nC")
    #expect(LineMerge.merge(base: "a\nb\nc", mine: "A\nb\nc", theirs: "a\nB\nc") == "A\nB\nc")
}

@Test func mergeDeletionPlusEdit() {
    let base = "1\n2\n3\n4\n"
    #expect(LineMerge.merge(base: base, mine: "1\n3\n4\n", theirs: "1\n2\n3\nFOUR\n") == "1\n3\nFOUR\n")
    #expect(LineMerge.merge(base: base, mine: "1\n2\n3\n", theirs: "ONE\n2\n3\n4\n") == "ONE\n2\n3\n")
}

@Test func mergeCRLF() {
    let base = "a\r\nb\r\nc\r\n"
    #expect(LineMerge.merge(base: base, mine: "A\r\nb\r\nc\r\n", theirs: "a\r\nb\r\nC\r\n") == "A\r\nb\r\nC\r\n")
}
