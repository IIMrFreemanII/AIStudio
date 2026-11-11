import Agent
import AppKit
import LLMClient
import SwiftUI
import UniformTypeIdentifiers
import WebKit

// HTML widgets the model draws with `show_widget`: a sandboxed web view per widget. What keeps them in:
// - the page's Content-Security-Policy (`WidgetPage`, parsed before the model's HTML): scripts and styles only
//   inline or from three CDNs, no fetch/XHR/WebSocket, no frames, no forms;
// - a content rule list that blocks every load but those CDNs (a second wall, in case the policy is got around);
// - a non-persistent data store per widget (no cookies or storage kept or shared);
// - navigation: only the first load; links open in the browser after a question; no new windows.
// The bridge (`aistudio.sendPrompt`, `aistudio.reportState`, the height) is one message handler.

/// The app's look for widgets: CSS variables for light and dark (the web view follows the app's appearance), and
/// base styles for text and controls.
@MainActor
enum WidgetTheme {
    static var css: String {
        func rgba(_ c: NSColor, _ name: NSAppearance.Name) -> String {
            var out = "transparent"
            NSAppearance(named: name)?.performAsCurrentDrawingAppearance {
                guard let s = c.usingColorSpace(.sRGB) else { return }
                out = String(format: "rgba(%d, %d, %d, %.3f)", Int(s.redComponent * 255), Int(s.greenComponent * 255),
                             Int(s.blueComponent * 255), s.alphaComponent)
            }
            return out
        }
        let vars: [(String, NSColor)] = [
            ("bg", MGColor.contentBackground), ("surface", MGColor.controlBackground), ("text", MGColor.label),
            ("text-2", MGColor.secondaryLabel), ("text-3", MGColor.tertiaryLabel), ("accent", MGColor.accent),
            ("border", MGColor.border), ("fill", MGColor.fill), ("success", MGColor.success),
            ("warning", MGColor.warning), ("danger", MGColor.destructive),
            ("series-1", MGColor.hueBlue), ("series-2", MGColor.hueOrange), ("series-3", MGColor.hueGreen),
            ("series-4", MGColor.huePurple), ("series-5", MGColor.huePink), ("series-6", MGColor.hueTeal),
        ]
        func block(_ name: NSAppearance.Name) -> String {
            vars.map { "--mg-\($0.0): \(rgba($0.1, name));" }.joined(separator: " ")
        }
        return """
            :root { color-scheme: light dark; \(block(.aqua)) --mg-radius: \(Int(MGRadius.md))px; \
            --mg-font: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; \
            --mg-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
            @media (prefers-color-scheme: dark) { :root { \(block(.darkAqua)) } }
            html, body { margin: 0; padding: 0; background: transparent; color: var(--mg-text);
              font: 13px/1.45 var(--mg-font); -webkit-font-smoothing: antialiased; }
            body { padding: 10px 12px; overflow-wrap: anywhere; }
            h1, h2, h3, h4 { margin: 0.4em 0; font-weight: 600; }
            a { color: var(--mg-accent); }
            button, select, input, textarea { font: inherit; color: var(--mg-text); background: var(--mg-surface);
              border: 1px solid var(--mg-border); border-radius: var(--mg-radius); padding: 4px 10px; }
            button { cursor: pointer; } button:hover { background: var(--mg-fill); }
            input[type=range] { accent-color: var(--mg-accent); padding: 0; border: 0; background: transparent; }
            input[type=checkbox], input[type=radio] { accent-color: var(--mg-accent); }
            table { border-collapse: collapse; } th, td { border-bottom: 1px solid var(--mg-border); padding: 4px 8px; text-align: left; }
            code, pre { font-family: var(--mg-mono); }
            """
    }

    static func page(_ html: String, title: String, bridge: Bool = true) -> String {
        WidgetPage.html(body: html, title: title, theme: css, bridge: bridge)
    }
}

/// The content rule list every widget's web view gets: block everything but the CDNs.
@MainActor
enum WidgetRules {
    private static var list: WKContentRuleList?
    /// Why the rules didn't compile (shown instead of the widget).
    private(set) static var failure: String?
    private static var waiting: [CheckedContinuation<WKContentRuleList?, Never>]?

    static var json: String {
        let rules: [[String: Any]] = [
            ["trigger": ["url-filter": ".*"], "action": ["type": "block"]],
        ] + WidgetPage.cdnHosts.map { host in
            ["trigger": ["url-filter": "^https://" + NSRegularExpression.escapedPattern(for: host) + "/"],
             "action": ["type": "ignore-previous-rules"]]
        } + ["about", "data", "blob"].map { scheme in  // (content rules take no `a|b` alternation)
            ["trigger": ["url-filter": "^\(scheme):"], "action": ["type": "ignore-previous-rules"]]
        }
        let data = (try? JSONSerialization.data(withJSONObject: rules)) ?? Data("[]".utf8)
        return String(decoding: data, as: UTF8.self)
    }

    static func get() async -> WKContentRuleList? {
        if let list { return list }
        if waiting != nil { return await withCheckedContinuation { waiting?.append($0) } }
        waiting = []
        var compiled: WKContentRuleList?
        do {
            compiled = try await WKContentRuleListStore.default().compileContentRuleList(
                forIdentifier: "aistudio-widgets-2", encodedContentRuleList: json)
        } catch {
            failure = error.localizedDescription
        }
        list = compiled
        for c in waiting ?? [] { c.resume(returning: compiled) }
        waiting = nil
        return compiled
    }
}

/// The web view: passes wheel scrolling on to the chat while the page itself has nothing to scroll.
final class WidgetWKWebView: WKWebView {
    var scrollsItself = false
    override func scrollWheel(with event: NSEvent) {
        if scrollsItself { super.scrollWheel(with: event) } else { nextResponder?.scrollWheel(with: event) }
    }
}

/// One widget's page in a web view, reporting its height, prompts and state.
struct WidgetWebView: NSViewRepresentable {
    let html: String
    let title: String
    /// Changing it loads the page again (Reload).
    var reload = 0
    var maxHeight: CGFloat = 600
    var onHeight: (CGFloat) -> Void = { _ in }
    var onPrompt: (String) -> Void = { _ in }
    var onState: (String) -> Void = { _ in }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WidgetWKWebView {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.userContentController.add(WeakHandler(context.coordinator), name: "aistudio")
        config.preferences.javaScriptCanOpenWindowsAutomatically = false
        let view = WidgetWKWebView(frame: .zero, configuration: config)
        view.setValue(false, forKey: "drawsBackground")
        view.navigationDelegate = context.coordinator
        view.uiDelegate = context.coordinator
        context.coordinator.view = view
        context.coordinator.parent = self
        context.coordinator.load(html: html, title: title, reload: reload)
        return view
    }

    func updateNSView(_ view: WidgetWKWebView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.load(html: html, title: title, reload: reload)
    }

    static func dismantleNSView(_ view: WidgetWKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeScriptMessageHandler(forName: "aistudio")
        view.stopLoading()
    }

    @MainActor
    final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate, WKUIDelegate {
        weak var view: WidgetWKWebView?
        var parent: WidgetWebView?
        private var loaded: (String, Int)?
        private var interacted = false
        private var lastPrompt = Date.distantPast
        private var firstLoadDone = false

        func load(html: String, title: String, reload: Int) {
            guard loaded?.0 != html || loaded?.1 != reload else { return }
            loaded = (html, reload)
            interacted = false
            firstLoadDone = false
            let page = WidgetTheme.page(html, title: title)
            Task { @MainActor [weak self] in
                guard let view = self?.view else { return }
                let rules = await WidgetRules.get()
                view.configuration.userContentController.removeAllContentRuleLists()
                if let rules { view.configuration.userContentController.add(rules) }
                guard rules != nil else {
                    // No rule list, no widget: the page's policy alone isn't trusted to keep it in.
                    let why = WidgetPage.escape(WidgetRules.failure ?? "unknown error")
                    view.loadHTMLString("<p style='font:12px -apple-system;color:gray'>The widget can't be shown: its sandbox didn't load (\(why)).</p>", baseURL: nil)
                    return
                }
                view.loadHTMLString(page, baseURL: nil)
            }
        }

        func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
            guard let body = message.body as? [String: Any], let type = body["type"] as? String, let parent else { return }
            switch type {
            case "height":
                guard let h = (body["height"] as? NSNumber)?.doubleValue else { return }
                let clamped = min(max(CGFloat(h), 40), parent.maxHeight)
                view?.scrollsItself = CGFloat(h) > parent.maxHeight
                parent.onHeight(clamped)
            case "interacted":
                interacted = true
            case "sendPrompt":
                // Only after the user clicked or typed in it, and not more than once every 2 s.
                guard interacted, Date().timeIntervalSince(lastPrompt) > 2,
                      let text = (body["text"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
                      !text.isEmpty else { return }
                lastPrompt = Date()
                parent.onPrompt(String(text.prefix(4000)))
            case "state":
                guard let s = body["state"] as? String, s.utf8.count <= 4096 else { return }
                parent.onState(s)
            default:
                break
            }
        }

        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction) async -> WKNavigationActionPolicy {
            // The page itself (loadHTMLString: about:blank), and in-page anchors.
            if !firstLoadDone, action.navigationType == .other { return .allow }
            if let url = action.request.url, url.scheme == "about" { return .allow }  // in-page anchors
            if action.navigationType == .linkActivated, let url = action.request.url,
               ["http", "https"].contains(url.scheme ?? "") {
                openAfterAsking(url)
            }
            return .cancel
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { firstLoadDone = true }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                     for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
            if let url = action.request.url, ["http", "https"].contains(url.scheme ?? ""), interacted { openAfterAsking(url) }
            return nil
        }

        func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                     initiatedByFrame frame: WKFrameInfo) async {}

        private func openAfterAsking(_ url: URL) {
            let alert = NSAlert()
            alert.messageText = "Open this link in your browser?"
            alert.informativeText = url.absoluteString
            alert.addButton(withTitle: "Open")
            alert.addButton(withTitle: "Cancel")
            if alert.runModal() == .alertFirstButtonReturn { NSWorkspace.shared.open(url) }
        }
    }

    /// The content controller keeps its handlers strongly: this breaks the cycle with the coordinator.
    @MainActor
    final class WeakHandler: NSObject, WKScriptMessageHandler {
        weak var target: (any WKScriptMessageHandler)?
        init(_ target: any WKScriptMessageHandler) { self.target = target }
        func userContentController(_ c: WKUserContentController, didReceive message: WKScriptMessage) {
            target?.userContentController(c, didReceive: message)
        }
    }
}

/// A `show_widget` call in the transcript: its title and the live widget, with Expand, Reload, Copy HTML and Save.
struct WidgetCard: View {
    @Environment(AppModel.self) private var model
    let doc: ChatDocument
    let call: ToolCall
    let result: ChatTurn?
    var live = false
    @State private var height: CGFloat = 120
    @State private var reload = 0
    @State private var hover = false

    var args: (title: String, html: String)? {
        guard let a = try? ToolArgs(json: call.arguments), let html = a.string("html") else { return nil }
        return (a.string("title") ?? "Widget", html)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "square.on.square.dashed").foregroundStyle(MG.accent)
                Text(args?.title ?? "Widget").font(.mgCallout).lineLimit(1)
                Spacer(minLength: 4)
                if let a = args, hover || WidgetCard.snapshotting {
                    toolbar(a)
                }
            }
            .padding(.horizontal, 10).frame(height: 32)
            MGDivider()
            content
        }
        .background(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous).fill(MG.controlBackground))
        .clipShape(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous))
        .mgHairline(RoundedRectangle(cornerRadius: MGRadius.md, style: .continuous), color: MG.border)
        .onHover { hover = $0 }
    }

    @ViewBuilder
    var content: some View {
        if let a = args, !(live && result == nil && call.arguments.last != "}") {
            if result?.toolStatus == "failed" {
                Label(result?.text ?? "The widget couldn't be shown.", systemImage: "exclamationmark.triangle")
                    .font(.mgFootnote).foregroundStyle(MG.destructive).padding(10)
            } else if WidgetCard.snapshotting {
                // Offscreen renders don't draw web views: a stand-in.
                Text("Interactive widget").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                WidgetWebView(html: a.html, title: a.title, reload: reload,
                              onHeight: { h in height = h },
                              onPrompt: { text in model.sendText(doc, text) },
                              onState: { s in doc.widgetStates[call.id] = s })
                    .frame(height: height)
            }
        } else {
            HStack(spacing: 8) {
                MGSpinner().controlSize(.mini)
                Text("Building the widget…").font(.mgFootnote).foregroundStyle(MG.secondaryLabel)
            }
            .padding(10).frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    func toolbar(_ a: (title: String, html: String)) -> some View {
        HStack(spacing: 2) {
            Button { WidgetWindows.open(doc: doc, callID: call.id, title: a.title, html: a.html, model: model) } label: {
                Image(systemName: "arrow.up.left.and.arrow.down.right")
            }
            .mgHelp("Open in its own window")
            Button { reload += 1 } label: { Image(systemName: "arrow.clockwise") }.mgHelp("Reload")
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(a.html, forType: .string)
            } label: { Image(systemName: "doc.on.doc") }
            .mgHelp("Copy the HTML")
            Button { save(a) } label: { Image(systemName: "square.and.arrow.down") }.mgHelp("Save as a web page")
        }
        .buttonStyle(.mg(.icon)).controlSize(.small)
    }

    func save(_ a: (title: String, html: String)) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = a.title.replacingOccurrences(of: "/", with: "-") + ".html"
        panel.allowedContentTypes = [.html]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        try? WidgetTheme.page(a.html, title: a.title, bridge: false).write(to: url, atomically: true, encoding: .utf8)
    }

    static var snapshotting: Bool { CommandLine.arguments.contains("--snapshot") }
}

/// Widgets opened in their own windows (Expand), kept until closed.
@MainActor
enum WidgetWindows {
    private static var windows: [String: NSWindow] = [:]

    static func open(doc: ChatDocument, callID: String, title: String, html: String, model: AppModel) {
        if let w = windows[callID] { w.makeKeyAndOrderFront(nil); return }
        let view = WidgetWebView(html: html, title: title, maxHeight: 10_000,
                                 onPrompt: { [weak doc] text in if let doc { model.sendText(doc, text) } },
                                 onState: { [weak doc] s in doc?.widgetStates[callID] = s })
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 640),
                              styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = title
        window.isReleasedWhenClosed = false
        window.contentView = NSHostingView(rootView: view.background(MG.contentBackground))
        window.center()
        windows[callID] = window
        NotificationCenter.default.addObserver(forName: NSWindow.willCloseNotification, object: window, queue: .main) { _ in
            MainActor.assumeIsolated { windows[callID] = nil }
        }
        window.makeKeyAndOrderFront(nil)
    }
}
