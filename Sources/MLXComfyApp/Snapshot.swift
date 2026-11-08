import AppKit
import GraphRuntime
import ImageIO
import SwiftUI

/// `MLXComfyApp --snapshot <dir>` renders the main window offscreen in a few representative states and exits.
/// Used to check layouts without driving the GUI by hand.
@MainActor
enum Snapshot {
    static var requestedDirectory: URL? {
        let args = CommandLine.arguments
        guard let i = args.firstIndex(of: "--snapshot"), i + 1 < args.count else { return nil }
        return URL(fileURLWithPath: args[i + 1])
    }

    static func renderAll(to dir: URL) async {
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let projectOutputs = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent("outputs")

        let model = AppModel(persistSettings: false, autoloadModel: false)
        model.outputDirectory = projectOutputs.path
        model.modelStatus = "Qwen-Image 2.1 (qwen-turbo-mlx)"
        if let first = model.history.first { model.show(first) }

        // 1. Image workspace, idle with a result.
        model.workspace = .image
        await render(model, to: dir.appendingPathComponent("1-image.png"))

        // 2. Image workspace mid-run: live preview, queue, node states.
        let g = model.graph
        model.queue = [
            QueueItem(graph: g, title: "A red fox standing in fresh snow at sunrise", seed: 1,
                      status: .done(seconds: 52.6)),
            QueueItem(graph: g, title: "A cozy coffee shop storefront with a wooden sign", seed: 7,
                      status: .running(step: 3, total: 8)),
            QueueItem(graph: g, title: "A lighthouse at dusk, long exposure", seed: 42),
        ]
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .cached,
                            "latent": .cached, "sampler": .running(step: 3, total: 8)]
        await render(model, to: dir.appendingPathComponent("2-running.png"))

        // 3. Graph workspace with a selected node.
        model.workspace = .graph
        model.selection = ["sampler"]
        model.nodeStates = ["model": .done(seconds: 0.1, peakMemory: nil), "prompt": .done(seconds: 3.3, peakMemory: 22_600_000_000),
                            "latent": .cached, "sampler": .running(step: 5, total: 8)]
        await render(model, to: dir.appendingPathComponent("3-graph.png"))

        // 4. Graph workspace, nothing selected (Simple panel in the inspector), idle.
        model.queue = []
        model.nodeStates = [:]
        model.selection = []
        await render(model, to: dir.appendingPathComponent("4-graph-idle.png"))

        // 5. Inpainting set up from the Simple panel: reference image + mask, graph rewired.
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let ref = cwd.appendingPathComponent("refs/girl.png"), mask = cwd.appendingPathComponent("examples/inpaint-mask.png")
        if FileManager.default.fileExists(atPath: ref.path) {
            model.setReferenceImage(ref, undoManager: nil)
            model.setMask(mask, undoManager: nil)
            model.workspace = .graph
            await render(model, to: dir.appendingPathComponent("5-inpaint-graph.png"))
            model.workspace = .image
            await render(model, to: dir.appendingPathComponent("6-inpaint-panel.png"))
            // 7. The mask editor on its own.
            await render(MaskEditorView(imageURL: ref, existingMask: mask, saveDirectory: dir) { _ in },
                         to: dir.appendingPathComponent("7-mask-editor.png"), size: CGSize(width: 900, height: 760))
            // 8. Inpainting with the hi-res fix on: feather and hi-res controls in the panel.
            model.setHiRes(true, undoManager: nil)
            await render(model, to: dir.appendingPathComponent("8-inpaint-hires-panel.png"))
            // 9. Outpainting instead of inpainting, in the graph and the panel.
            model.setOutpaint(left: 256, top: 0, right: 256, bottom: 128, undoManager: nil)
            model.workspace = .graph
            await render(model, to: dir.appendingPathComponent("9-outpaint-graph.png"))
            model.workspace = .image
            await render(model, to: dir.appendingPathComponent("10-outpaint-panel.png"))
            print("graph: " + model.graph.jsonString())
            // 11. The prompt assistant mid-run (progress + Stop in place of Enhance / Describe).
            model.clearOutpaint(undoManager: nil)
            model.setHiRes(false, undoManager: nil)
            model.assistantMode = .describe
            await render(model, to: dir.appendingPathComponent("11-assistant-running.png"))
            model.assistantMode = nil
            await render(model, to: dir.appendingPathComponent("12-assistant-idle.png"))
            // 13. Settings with the assistant's instructions.
            await render(SettingsView().environment(model), to: dir.appendingPathComponent("13-settings.png"),
                         size: CGSize(width: 600, height: 640))
        }
    }

    static func render<V: View>(_ view: V, to url: URL, size: CGSize) async {
        let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                              styleMask: [.titled], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: view.frame(width: size.width, height: size.height))
        window.contentView = host
        window.orderFrontRegardless()
        try? await Task.sleep(nanoseconds: 800_000_000)
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return }
        host.cacheDisplay(in: host.bounds, to: rep)
        if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: url) }
        window.close()
        print("wrote \(url.path)")
    }

    static func render(_ model: AppModel, to url: URL, size: CGSize = CGSize(width: 1440, height: 900)) async {
        let window = NSWindow(contentRect: CGRect(origin: CGPoint(x: -4000, y: -4000), size: size),
                              styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
                              backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: ContentView().environment(model).frame(width: size.width, height: size.height))
        window.contentView = host
        window.orderFrontRegardless()
        try? await Task.sleep(nanoseconds: 1_200_000_000)  // let layout, tasks and thumbnails settle
        host.layoutSubtreeIfNeeded()
        let view = window.contentView!.superview ?? host
        guard let rep = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: rep)
        if let data = rep.representation(using: .png, properties: [:]) { try? data.write(to: url) }
        window.close()
        print("wrote \(url.path)")
    }
}
