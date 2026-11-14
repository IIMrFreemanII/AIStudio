import CoreGraphics
import Foundation
import Testing
import Agent
@testable import UIControl

struct KeyComboTests {
    @Test func parsesKeys() throws {
        #expect(try KeyCombo.parse("cmd+s") == KeyCombo(keyCode: 0x01, flags: .maskCommand))
        #expect(try KeyCombo.parse("cmd+shift+z") == KeyCombo(keyCode: 0x06, flags: [.maskCommand, .maskShift]))
        #expect(try KeyCombo.parse("Return") == KeyCombo(keyCode: 0x24))
        #expect(try KeyCombo.parse("esc").keyCode == 0x35)
        #expect(try KeyCombo.parse("left").keyCode == 0x7B)
        #expect(try KeyCombo.parse("f5").keyCode == 0x60)
        #expect(try KeyCombo.parse("Ctrl+Opt+Tab") == KeyCombo(keyCode: 0x30, flags: [.maskControl, .maskAlternate]))
    }

    @Test func rejectsUnknown() {
        #expect(throws: ToolError.self) { try KeyCombo.parse("cmd+") }
        #expect(throws: ToolError.self) { try KeyCombo.parse("hyper+k") }
        #expect(throws: ToolError.self) { try KeyCombo.parse("cmd") }
    }
}

struct CoordMapTests {
    @Test func mapsPixels() {
        let frame = UIRect(x: 100, y: 50, width: 640, height: 400)
        let size = CGSize(width: 1280, height: 800)
        #expect(CoordMap.screenPoint(pixel: CGPoint(x: 640, y: 400), imageSize: size, windowFrame: frame) == CGPoint(x: 420, y: 250))
        #expect(CoordMap.screenPoint(pixel: .zero, imageSize: size, windowFrame: frame) == CGPoint(x: 100, y: 50))
    }
}

struct RoleNameTests {
    @Test func names() {
        #expect(AXFormat.roleName("AXButton", subrole: nil) == "button")
        #expect(AXFormat.roleName("AXTextField", subrole: nil) == "text field")
        #expect(AXFormat.roleName("AXPopUpButton", subrole: nil) == "pop up button")
        #expect(AXFormat.roleName("AXStaticText", subrole: nil) == "text")
        #expect(AXFormat.roleName("AXTextField", subrole: "AXSecureTextField") == "secure text field")
        #expect(AXFormat.roleName("AXCheckBox", subrole: nil) == "checkbox")
        #expect(AXFormat.roleName("AXMenuBarItem", subrole: nil) == "menu bar item")
        #expect(AXFormat.roleName("AXButton", subrole: "AXCloseButton") == "close button")
        #expect(AXFormat.roleName("AXButton", subrole: "AXZoomButton") == "zoom button")
    }
}

struct ElementFilterTests {
    // window > group > (button Save, text field) ; window2 > button Cancel
    private let nodes: [UIElement] = [
        UIElement(id: "e1", role: "window", title: "Doc", depth: 0),
        UIElement(id: "e2", role: "group", depth: 1),
        UIElement(id: "e3", role: "button", title: "Save", depth: 2),
        UIElement(id: "e4", role: "text field", value: "hello", depth: 2),
        UIElement(id: "e5", role: "window", title: "Other", depth: 0),
        UIElement(id: "e6", role: "button", title: "Cancel", depth: 1),
    ]

    @Test func noQueryCaps() {
        let r = ElementFilter.apply(nodes, query: nil, max: 3)
        #expect(r.elements.map(\.id) == ["e1", "e2", "e3"])
        #expect(r.total == 6)
    }

    @Test func queryKeepsAncestors() {
        let r = ElementFilter.apply(nodes, query: "SAVE", max: 10)
        #expect(r.elements.map(\.id) == ["e1", "e2", "e3"])
        #expect(r.total == 1)
        let b = ElementFilter.apply(nodes, query: "button", max: 10)
        #expect(b.elements.map(\.id) == ["e1", "e2", "e3", "e5", "e6"])
        #expect(b.total == 2)
        #expect(ElementFilter.apply(nodes, query: "button", max: 2).elements.map(\.id) == ["e1", "e2"])
        #expect(ElementFilter.apply(nodes, query: "zzz", max: 5).total == 0)
    }
}

struct WindowBadgeTests {
    @Test func theCaptureBadgeIsNotAWindow() {
        #expect(AppList.isBadge(CGRect(x: 666, y: 205, width: 66, height: 20)))
        #expect(!AppList.isBadge(CGRect(x: 654, y: 201, width: 420, height: 328)))
    }
}
