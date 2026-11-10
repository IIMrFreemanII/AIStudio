import CoreGraphics
import Foundation
import Testing
@testable import AIStudioLibrary

@Suite struct SurfacePlacementTests {
    let screen = CGRect(x: 0, y: 0, width: 1000, height: 800)
    let size = CGSize(width: 200, height: 300)

    @Test func contextMenuOpensDownAndRightOfTheMouse() {
        let r = SurfacePlacement.atPoint(CGPoint(x: 100, y: 600), size: size, screen: screen)
        #expect(r == CGRect(x: 100, y: 300, width: 200, height: 300))
    }

    @Test func contextMenuFlipsAtTheRightAndBottomEdges() {
        let r = SurfacePlacement.atPoint(CGPoint(x: 900, y: 100), size: size, screen: screen)
        #expect(r.maxX == 900, "opens to the left")
        #expect(r.minY == 100, "opens upwards")
        #expect(screen.contains(r))
    }

    @Test func belowAnchorWithGapOrAboveWhenNoRoom() {
        let anchor = CGRect(x: 50, y: 500, width: 120, height: 22)
        let r = SurfacePlacement.below(anchor, size: size, screen: screen)
        #expect(r.maxY == 496 && r.minX == 50)
        let low = CGRect(x: 50, y: 100, width: 120, height: 22)
        let up = SurfacePlacement.below(low, size: size, screen: screen)
        #expect(up.minY == 126)
    }

    @Test func belowAnchorAtTheRightEdgeShiftsLeft() {
        let r = SurfacePlacement.below(CGRect(x: 950, y: 500, width: 40, height: 22), size: size, screen: screen)
        #expect(r.maxX == screen.maxX)
    }

    @Test func submenuOpensRightThenFlipsLeft() {
        let parent = CGRect(x: 100, y: 300, width: 200, height: 300)
        let row = CGRect(x: 105, y: 500, width: 190, height: 22)
        let r = SurfacePlacement.submenu(row: row, parent: parent, size: size, screen: screen)
        #expect(r.minX == 298 && r.maxY == 527)
        let rightParent = CGRect(x: 750, y: 300, width: 200, height: 300)
        let l = SurfacePlacement.submenu(row: row.offsetBy(dx: 650, dy: 0), parent: rightParent, size: size, screen: screen)
        #expect(l.maxX == 752)
    }

    @Test func tooltipBelowPointerOrAboveAtTheBottom() {
        let s = CGSize(width: 120, height: 24)
        let r = SurfacePlacement.tooltip(at: CGPoint(x: 300, y: 400), size: s, screen: screen)
        #expect(r.maxY == 378)
        let b = SurfacePlacement.tooltip(at: CGPoint(x: 300, y: 20), size: s, screen: screen)
        #expect(b.minY == 28)
    }

    @Test func oversizedSurfaceIsPinnedTopLeft() {
        let r = SurfacePlacement.clamp(CGRect(x: 500, y: -50, width: 1200, height: 900), to: screen)
        #expect(r.minX == 0 && r.maxY == 800)
    }

    @Test func everyPlacementStaysOnScreen() {
        for x in stride(from: 0.0, through: 1000, by: 125) {
            for y in stride(from: 0.0, through: 800, by: 100) {
                let p = CGPoint(x: x, y: y)
                #expect(screen.contains(SurfacePlacement.atPoint(p, size: size, screen: screen)))
                #expect(screen.contains(SurfacePlacement.tooltip(at: p, size: CGSize(width: 100, height: 20), screen: screen)))
            }
        }
    }
}

@Suite struct ScrollThumbTests {
    @Test func noThumbWhenEverythingFits() {
        #expect(ScrollThumb.make(content: 300, viewport: 300, offset: 0, track: 296) == nil)
    }

    @Test func lengthIsProportionalAndPositionFollowsOffset() {
        let top = ScrollThumb.make(content: 1000, viewport: 250, offset: 0, track: 246)!
        #expect(abs(top.length - 61.5) < 0.01 && top.start == 0)
        let bottom = ScrollThumb.make(content: 1000, viewport: 250, offset: 750, track: 246)!
        #expect(abs(bottom.start + bottom.length - 246) < 0.01)
    }

    @Test func thumbHasAMinimumLength() {
        let t = ScrollThumb.make(content: 100_000, viewport: 200, offset: 0, track: 196)!
        #expect(t.length == ScrollThumb.minimumLength)
    }

    @Test func rubberBandingShrinksTheThumbAndClamps() {
        let t = ScrollThumb.make(content: 1000, viewport: 250, offset: -30, track: 246)!
        #expect(t.start == 0 && abs(t.length - 31.5) < 0.01)
        let b = ScrollThumb.make(content: 1000, viewport: 250, offset: 800, track: 246)!
        #expect(abs(b.start + b.length - 246) < 0.01 && b.length == 20)
    }

    @Test func draggingRoundTrips() {
        for offset in [0.0, 120, 375, 750] {
            let t = ScrollThumb.make(content: 1000, viewport: 250, offset: offset, track: 246)!
            let back = ScrollThumb.offset(forStart: t.start, content: 1000, viewport: 250, track: 246)
            #expect(abs(back - offset) < 0.01)
        }
        #expect(ScrollThumb.offset(forStart: 9999, content: 1000, viewport: 250, track: 246) == 750)
        #expect(ScrollThumb.offset(forStart: -5, content: 1000, viewport: 250, track: 246) == 0)
    }
}

@Suite struct MenuNavigationTests {
    // Open, —, Save (disabled), Save As, —, Close
    let selectable = [true, false, false, true, false, true]
    let titles = ["Open", "", "Save", "Save As…", "", "Close"]

    @Test func arrowsSkipSeparatorsAndDisabledItemsAndWrap() {
        #expect(MenuNavigation.move(from: nil, step: 1, selectable: selectable) == 0)
        #expect(MenuNavigation.move(from: nil, step: -1, selectable: selectable) == 5)
        #expect(MenuNavigation.move(from: 0, step: 1, selectable: selectable) == 3)
        #expect(MenuNavigation.move(from: 5, step: 1, selectable: selectable) == 0)
        #expect(MenuNavigation.move(from: 0, step: -1, selectable: selectable) == 5)
        #expect(MenuNavigation.move(from: nil, step: 1, selectable: [false, false]) == nil)
    }

    @Test func typeAheadMatchesSelectableItemsIgnoringCase() {
        #expect(MenuNavigation.match("s", titles: titles, selectable: selectable, from: nil) == 3)
        #expect(MenuNavigation.match("CL", titles: titles, selectable: selectable, from: nil) == 5)
        #expect(MenuNavigation.match("x", titles: titles, selectable: selectable, from: nil) == nil)
    }

    @Test func repeatedLetterCyclesThroughMatches() {
        let t = ["Copy", "Cut", "Paste", "Crop"]
        let s = [true, true, true, true]
        #expect(MenuNavigation.match("c", titles: t, selectable: s, from: 0) == 1)
        #expect(MenuNavigation.match("c", titles: t, selectable: s, from: 1) == 3)
        #expect(MenuNavigation.match("c", titles: t, selectable: s, from: 3) == 0)
        #expect(MenuNavigation.match("cr", titles: t, selectable: s, from: 3) == 3)
        #expect(MenuNavigation.match("é", titles: ["Edit", "Échelle"], selectable: [true, true], from: nil) == 0)
    }
}
