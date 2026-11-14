import CoreGraphics
import Foundation
import Agent

/// A key press: an ANSI virtual key code and modifier flags. Parsed from "cmd+shift+z", "return", "f5"…
public struct KeyCombo: Equatable {
    public var keyCode: CGKeyCode
    public var flags: CGEventFlags
    public init(keyCode: CGKeyCode, flags: CGEventFlags = []) { self.keyCode = keyCode; self.flags = flags }

    static func modifier(_ name: String) -> CGEventFlags? {
        switch name.lowercased() {
        case "cmd", "command": return .maskCommand
        case "shift": return .maskShift
        case "opt", "option", "alt": return .maskAlternate
        case "ctrl", "control": return .maskControl
        case "fn": return .maskSecondaryFn
        default: return nil
        }
    }

    /// Pure and case-insensitive; parts are separated by "+". Exactly one non-modifier key.
    public static func parse(_ s: String) throws -> KeyCombo {
        var flags: CGEventFlags = []
        var code: CGKeyCode?
        for part in s.split(separator: "+", omittingEmptySubsequences: false).map(String.init) {
            let p = part.trimmingCharacters(in: .whitespaces).lowercased()
            if let f = modifier(p) { flags.insert(f); continue }
            if code == nil, let c = keyCodes[p] { code = c; continue }
            throw ToolError("unknown key “\(part)” in “\(s)”; use names like cmd+s, return, tab, esc, left")
        }
        guard let code else {
            throw ToolError("no key in “\(s)”; use names like cmd+s, return, tab, esc, left")
        }
        return KeyCombo(keyCode: code, flags: flags)
    }

    // kVK_* values from Carbon's Events.h (ANSI layout), hard-coded so nothing imports Carbon.
    static let keyCodes: [String: CGKeyCode] = [
        "a": 0x00, "s": 0x01, "d": 0x02, "f": 0x03, "h": 0x04, "g": 0x05, "z": 0x06, "x": 0x07, "c": 0x08, "v": 0x09,
        "b": 0x0B, "q": 0x0C, "w": 0x0D, "e": 0x0E, "r": 0x0F, "y": 0x10, "t": 0x11,
        "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "6": 0x16, "5": 0x17, "=": 0x18, "9": 0x19, "7": 0x1A, "-": 0x1B,
        "8": 0x1C, "0": 0x1D, "]": 0x1E, "o": 0x1F, "u": 0x20, "[": 0x21, "i": 0x22, "p": 0x23,
        "return": 0x24, "enter": 0x24, "l": 0x25, "j": 0x26, "'": 0x27, "k": 0x28, ";": 0x29, "\\": 0x2A, ",": 0x2B,
        "/": 0x2C, "n": 0x2D, "m": 0x2E, ".": 0x2F, "tab": 0x30, "space": 0x31, "`": 0x32,
        "delete": 0x33, "backspace": 0x33, "esc": 0x35, "escape": 0x35,
        "f1": 0x7A, "f2": 0x78, "f3": 0x63, "f4": 0x76, "f5": 0x60, "f6": 0x61, "f7": 0x62, "f8": 0x64, "f9": 0x65,
        "f10": 0x6D, "f11": 0x67, "f12": 0x6F,
        "home": 0x73, "pageup": 0x74, "forwarddelete": 0x75, "end": 0x77, "pagedown": 0x79,
        "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
    ]
}

/// Posts mouse and keyboard events to the HID tap. Every call takes `check`, run right before an event batch, which
/// throws when the target app is no longer frontmost.
enum Input {
    typealias Check = () throws -> Void

    static func sleep(_ ms: Int) async { try? await Task.sleep(nanoseconds: UInt64(ms) * 1_000_000) }
    private static func post(_ e: CGEvent?) { e?.post(tap: .cghidEventTap) }
    private static var source: CGEventSource? { CGEventSource(stateID: .hidSystemState) }

    static func flags(_ modifiers: [String]) throws -> CGEventFlags {
        var f: CGEventFlags = []
        for m in modifiers {
            guard let one = KeyCombo.modifier(m), m.lowercased() != "fn" else {
                throw ToolError("unknown modifier “\(m)”; use cmd, shift, option or ctrl")
            }
            f.insert(one)
        }
        return f
    }

    static func move(to p: CGPoint, check: Check) async throws {
        try check()
        post(CGEvent(mouseEventSource: source, mouseType: .mouseMoved, mouseCursorPosition: p, mouseButton: .left))
        await sleep(30)
    }

    static func click(at p: CGPoint, button: UIMouseButton, count: Int, flags: CGEventFlags, check: Check) async throws {
        try await move(to: p, check: check)
        let (downType, upType, cgButton): (CGEventType, CGEventType, CGMouseButton) =
            button == .right ? (.rightMouseDown, .rightMouseUp, .right) : (.leftMouseDown, .leftMouseUp, .left)
        try check()
        for n in 1...max(1, min(count, 2)) {
            let down = CGEvent(mouseEventSource: source, mouseType: downType, mouseCursorPosition: p, mouseButton: cgButton)
            let up = CGEvent(mouseEventSource: source, mouseType: upType, mouseCursorPosition: p, mouseButton: cgButton)
            for e in [down, up] {
                e?.setIntegerValueField(.mouseEventClickState, value: Int64(n))
                e?.flags = flags
            }
            post(down)
            await sleep(20)
            post(up)
            await sleep(20)
        }
    }

    /// Typed as unicode strings in chunks (so any character works); "\n" is a Return key press.
    static func type(_ text: String, check: Check) async throws {
        var buffer: [UInt16] = []
        func flush() async throws {
            var i = 0
            while i < buffer.count {
                var end = min(i + 16, buffer.count)
                // Don't split a surrogate pair across chunks.
                if end < buffer.count, UTF16.isLeadSurrogate(buffer[end - 1]), end - i > 1 { end -= 1 }
                let chunk = Array(buffer[i..<end])
                try check()
                for down in [true, false] {
                    let e = CGEvent(keyboardEventSource: source, virtualKey: 0, keyDown: down)
                    // No modifiers: a ⌘ left over from the previous key press would turn the text into a shortcut
                    // (virtual key 0 is "a", so ⌘A).
                    e?.flags = []
                    e?.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: chunk)
                    post(e)
                }
                await sleep(8)
                i = end
            }
            buffer.removeAll()
        }
        var previousWasCR = false
        for scalar in text.unicodeScalars {
            if scalar == "\n" || scalar == "\r" {
                if scalar == "\n", previousWasCR { previousWasCR = false; continue }
                previousWasCR = scalar == "\r"
                try await flush()
                try await press(KeyCombo(keyCode: 0x24), check: check)
                await sleep(8)
            } else {
                previousWasCR = false
                buffer.append(contentsOf: Array(String(scalar).utf16))
            }
        }
        try await flush()
    }

    static func press(_ combo: KeyCombo, check: Check) async throws {
        try check()
        for down in [true, false] {
            let e = CGEvent(keyboardEventSource: source, virtualKey: combo.keyCode, keyDown: down)
            e?.flags = combo.flags
            post(e)
            if down { await sleep(8) }
        }
    }

    static func key(_ combo: KeyCombo, repeat n: Int, check: Check) async throws {
        for i in 0..<max(1, n) {
            try await press(combo, check: check)
            if i < n - 1 { await sleep(30) }
        }
    }

    /// Positive `dy` scrolls down (the content moves up), positive `dx` right. CGEvent's wheel values go the other way
    /// (a positive wheel1 scrolls up, i.e. the content moves down), hence the minus signs. Checked by `--smoke-ui` on a
    /// real list, with natural scrolling on (it doesn't flip synthetic events).
    static func scroll(at p: CGPoint, dx: Int, dy: Int, check: Check) async throws {
        try await move(to: p, check: check)
        try check()
        post(CGEvent(scrollWheelEvent2Source: source, units: .line, wheelCount: 2,
                     wheel1: Int32(-dy), wheel2: Int32(-dx), wheel3: 0))
    }

    static func drag(from a: CGPoint, to b: CGPoint, check: Check) async throws {
        try await move(to: a, check: check)
        try check()
        post(CGEvent(mouseEventSource: source, mouseType: .leftMouseDown, mouseCursorPosition: a, mouseButton: .left))
        await sleep(15)
        let steps = 12
        for i in 1...steps {
            let t = Double(i) / Double(steps)
            let p = CGPoint(x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t)
            post(CGEvent(mouseEventSource: source, mouseType: .leftMouseDragged, mouseCursorPosition: p, mouseButton: .left))
            await sleep(15)
        }
        post(CGEvent(mouseEventSource: source, mouseType: .leftMouseUp, mouseCursorPosition: b, mouseButton: .left))
    }
}
