import CoreGraphics
import Foundation
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers
import Agent

/// Pixel of a screenshot → screen point. Pure.
public enum CoordMap {
    public static func screenPoint(pixel: CGPoint, imageSize: CGSize, windowFrame: UIRect) -> CGPoint {
        CGPoint(x: windowFrame.x + pixel.x * windowFrame.width / imageSize.width,
                y: windowFrame.y + pixel.y * windowFrame.height / imageSize.height)
    }
}

enum Capture {
    static let maxSide = 1280

    /// The app's window (front-most, or the front-most whose title contains `title`) with its capture filter.
    static func window(of app: UIApp, titleContains title: String?) async throws -> (SCWindow, SCContentFilter) {
        guard CGPreflightScreenCaptureAccess() else {
            throw ToolError("the Screen Recording permission is missing: ask the user to allow AIStudio in System Settings > Privacy & Security > Screen & System Audio Recording.")
        }
        let content: SCShareableContent
        do {
            content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        } catch {
            throw ToolError("couldn't list windows for capture: \(error.localizedDescription)")
        }
        var candidates = content.windows.filter {
            $0.owningApplication?.processID == app.pid && $0.windowLayer == 0 && $0.isOnScreen && !AppList.isBadge($0.frame)
        }
        if let t = title?.lowercased(), !t.isEmpty {
            candidates = candidates.filter { ($0.title ?? "").lowercased().contains(t) }
        }
        let order = Dictionary(AppList.zOrder().enumerated().map { ($1, $0) }, uniquingKeysWith: { a, _ in a })
        guard let win = candidates.min(by: { (order[$0.windowID] ?? .max) < (order[$1.windowID] ?? .max) }) else {
            throw ToolError("\(app.name) has no window on screen")
        }
        return (win, SCContentFilter(desktopIndependentWindow: win))
    }

    /// A stream configuration sized to the window in pixels, long side at most 1280 (even numbers, for H.264).
    static func configuration(for window: SCWindow, filter: SCContentFilter) -> SCStreamConfiguration {
        let scale = Double(filter.pointPixelScale)   // the display's backing scale
        let w = max(window.frame.width * scale, 2), h = max(window.frame.height * scale, 2)
        let k = min(1, Double(maxSide) / max(w, h))
        let c = SCStreamConfiguration()
        c.width = max(2, Int((w * k).rounded()) / 2 * 2)
        c.height = max(2, Int((h * k).rounded()) / 2 * 2)
        c.showsCursor = false
        c.pixelFormat = kCVPixelFormatType_32BGRA
        return c
    }

    static func screenshot(of app: UIApp, window title: String?) async throws -> UIShot {
        let (win, filter) = try await window(of: app, titleContains: title)
        let config = configuration(for: win, filter: filter)
        let image: CGImage
        do {
            image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
        } catch {
            throw ToolError("couldn't capture \(app.name)'s window: \(error.localizedDescription)")
        }
        guard let url = jpegDataURL(image, maxSide: maxSide) else { throw ToolError("couldn't encode the screenshot") }
        let frame = win.frame
        return UIShot(dataURL: url, width: image.width, height: image.height,
                      window: UIWindowInfo(title: win.title ?? "",
                                           frame: UIRect(x: frame.origin.x, y: frame.origin.y, width: frame.width, height: frame.height)))
    }

    /// An image as a JPEG (quality 0.8) `data:` URL, scaled down so its long side is at most `maxSide` pixels.
    static func jpegDataURL(_ image: CGImage, maxSide: Int) -> String? {
        let k = min(1, Double(maxSide) / Double(max(image.width, image.height, 1)))
        let w = max(1, Int((Double(image.width) * k).rounded())), h = max(1, Int((Double(image.height) * k).rounded()))
        // Redraw opaque: JPEG has no alpha.
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        guard let flat = ctx.makeImage() else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, flat, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return "data:image/jpeg;base64," + (data as Data).base64EncodedString()
    }
}
