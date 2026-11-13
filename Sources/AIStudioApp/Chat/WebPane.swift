import AIStudioLibrary
import Agent
import AppKit
import SwiftUI
import WebKit

/// The web page of a coding chat's Browser tab. One web view per chat, kept while the chat is open, so switching tabs
/// (or chats) keeps the page, its history and where it was scrolled.
@MainActor @Observable
final class ChatWebPage: NSObject, WKNavigationDelegate, WKUIDelegate {
    @ObservationIgnored let view: WKWebView
    private(set) var url: URL?
    private(set) var title = ""
    private(set) var loading = false
    private(set) var progress = 0.0
    private(set) var canGoBack = false
    private(set) var canGoForward = false
    /// Why the last page didn't load (nil once another starts).
    private(set) var error: String?
    /// Counts loads that finished or failed (the agent's `open_page` waits for it to move).
    private(set) var settled = 0
    /// Called when the page settles on an address (the chat remembers it).
    @ObservationIgnored var onNavigate: ((URL) -> Void)?
    @ObservationIgnored private var observers: [NSKeyValueObservation] = []

    override init() {
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
        config.preferences.isElementFullscreenEnabled = true
        view = WKWebView(frame: .zero, configuration: config)
        super.init()
        view.navigationDelegate = self
        view.uiDelegate = self
        view.allowsBackForwardNavigationGestures = true
        view.allowsMagnification = true
        view.isInspectable = true
        func watch<V>(_ key: KeyPath<WKWebView, V>, _ apply: @escaping (ChatWebPage, WKWebView) -> Void) -> NSKeyValueObservation {
            view.observe(key, options: [.initial, .new]) { [weak self] v, _ in
                MainActor.assumeIsolated { if let self { apply(self, v) } }
            }
        }
        observers = [
            watch(\.url) { p, v in if let u = v.url { p.url = u } },
            watch(\.title) { p, v in p.title = v.title ?? "" },
            watch(\.isLoading) { p, v in p.loading = v.isLoading },
            watch(\.estimatedProgress) { p, v in p.progress = v.estimatedProgress },
            watch(\.canGoBack) { p, v in p.canGoBack = v.canGoBack },
            watch(\.canGoForward) { p, v in p.canGoForward = v.canGoForward },
        ]
    }

    func load(_ url: URL) {
        error = nil
        self.url = url
        if url.isFileURL {
            view.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
        } else {
            view.load(URLRequest(url: url))
        }
    }

    func reload() {
        error = nil
        if view.url == nil, let url { load(url) } else { view.reload() }
    }

    // MARK: WKNavigationDelegate

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
        let scheme = action.request.url?.scheme?.lowercased() ?? ""
        if ["http", "https", "file", "about", "data", "blob"].contains(scheme) {
            decisionHandler(.allow)
        } else {
            // mailto:, tel:, app links: the system handles them.
            if let u = action.request.url, action.navigationType == .linkActivated { NSWorkspace.shared.open(u) }
            decisionHandler(.cancel)
        }
    }

    func webView(_ webView: WKWebView, didStartProvisionalNavigation navigation: WKNavigation!) { error = nil }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        settled += 1
        if let u = webView.url { onNavigate?(u) }
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) { failed(error) }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        failed(error)
    }

    private func failed(_ e: Error) {
        let ns = e as NSError
        // A load replaced by another, or a download the page started.
        if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled { return }
        if ns.domain == "WebKitErrorDomain" && ns.code == 102 { return }
        error = ns.localizedDescription
        // A page that never loaded leaves the web view on the one before: show the address that failed.
        if let failing = ns.userInfo[NSURLErrorFailingURLErrorKey] as? URL { url = failing }
        settled += 1
    }

    // MARK: WKUIDelegate

    /// Links that open a new window (target=_blank) open here.
    func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                 for action: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? {
        if let u = action.request.url { load(u) }
        return nil
    }
}

/// The Browser tab: back, forward, reload, the address, open in the default browser; the page under it.
struct WebPane: View {
    @Environment(AppModel.self) private var model
    @Bindable var doc: ChatDocument
    let page: ChatWebPage
    @State private var address = ""
    @State private var addressError = false
    @FocusState private var addressFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 2) {
                Button { page.view.goBack() } label: { Image(systemName: "chevron.left") }.accessibilityLabel("Back")
                    .disabled(!page.canGoBack).mgHelp("Back")
                Button { page.view.goForward() } label: { Image(systemName: "chevron.right") }.accessibilityLabel("Forward")
                    .disabled(!page.canGoForward).mgHelp("Forward")
                Button { if page.loading { page.view.stopLoading() } else { page.reload() } } label: {
                    Image(systemName: page.loading ? "xmark" : "arrow.clockwise")
                }
                .disabled(page.url == nil).mgHelp(page.loading ? "Stop" : "Reload")
                MGFieldChrome(icon: icon(page)) {
                    TextField("Enter a web address", text: $address)
                        .textFieldStyle(.plain)
                        .focused($addressFocused)
                        .onSubmit { go(page) }
                        .onExitCommand { address = shown(page); addressFocused = false }
                }
                .overlay {
                    if addressError {
                        RoundedRectangle(cornerRadius: MGRadius.md).strokeBorder(MG.destructive.opacity(0.7))
                    }
                }
                .padding(.horizontal, 4)
                .mgTestAnchor("web-address")
                Button { if let u = page.url { NSWorkspace.shared.open(u) } } label: { Image(systemName: "safari") }.accessibilityLabel("Open in Safari")
                    .disabled(page.url == nil).mgHelp("Open in your default browser")
                MGMenuButton(kind: .icon) {
                    MGMenuItem.button("Copy Link", icon: "link") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(page.url?.absoluteString ?? "", forType: .string)
                    }
                    MGMenuItem.button("Actual Size", icon: "1.magnifyingglass") { page.view.magnification = 1 }
                    MGMenuItem.divider
                    MGMenuItem.button("Close Page", icon: "xmark.square") { model.closeWebPage(doc) }
                } label: { Image(systemName: "ellipsis") }
                    .mgHelp("More")
            }
            .buttonStyle(.mg(.icon))
            .padding(.horizontal, 8)
            .frame(height: 38)
            ZStack(alignment: .top) {
                MGDivider()
                if page.loading {
                    GeometryReader { g in
                        Rectangle().fill(MG.accent).frame(width: g.size.width * max(0.08, page.progress), height: 2)
                            .animation(.easeOut(duration: 0.2), value: page.progress)
                    }
                    .frame(height: 2)
                }
            }
            ZStack {
                WebViewHost(view: page.view).opacity(page.url == nil ? 0 : 1)
                if page.url == nil {
                    MGEmptyState(symbol: "globe", title: "Open a web page",
                                 message: "Type an address above (localhost:3000 works too), or click a link in the chat. "
                                     + "⌘-click a link to open it in your default browser instead.")
                } else if let e = page.error {
                    VStack(spacing: 10) {
                        MGEmptyState(symbol: "exclamationmark.triangle", title: "Couldn't open the page", message: e)
                            .frame(maxHeight: 220)
                        Button("Try Again") { page.reload() }.buttonStyle(.mg(.bordered))
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(MG.windowBackground)
                }
            }
        }
        .onAppear {
            address = shown(page)
            if page.url == nil && address.isEmpty { addressFocused = true }
        }
        .onChange(of: page.url) { _, _ in if !addressFocused { address = shown(page) } }
        .onChange(of: addressFocused) { _, f in if !f { addressError = false } }
        .mgTestAnchor("web-pane")
    }

    func icon(_ page: ChatWebPage) -> String {
        guard let u = page.url else { return "globe" }
        if u.isFileURL { return "doc" }
        return u.scheme == "https" ? "lock.fill" : "globe"
    }

    func shown(_ page: ChatWebPage) -> String { page.url.map(WebAddress.display) ?? "" }

    func go(_ page: ChatWebPage) {
        guard let url = WebAddress.url(from: address) else { addressError = true; return }
        addressError = false
        addressFocused = false
        model.openWebPage(doc, url)
    }
}

/// Hosts the chat's web view (kept by `ChatWebPage`, so it moves between hosts as the tab comes and goes).
private struct WebViewHost: NSViewRepresentable {
    let view: WKWebView

    func makeNSView(context: Context) -> NSView {
        let host = NSView()
        attach(to: host)
        return host
    }

    func updateNSView(_ host: NSView, context: Context) {
        if view.superview !== host { attach(to: host) }
    }

    private func attach(to host: NSView) {
        view.removeFromSuperview()
        view.frame = host.bounds
        view.autoresizingMask = [.width, .height]
        host.addSubview(view)
    }
}

extension AppModel {
    /// The chat's web page (made the first time it's needed; the address it last showed is loaded again).
    func webPage(_ doc: ChatDocument) -> ChatWebPage {
        if let p = doc.webPage { return p }
        let p = ChatWebPage()
        p.onNavigate = { [weak self, weak doc] url in
            guard let self, let doc, doc.transcript.browserURL != url.absoluteString else { return }
            doc.transcript.browserURL = url.absoluteString
            self.saveChat(doc)
        }
        doc.webPage = p
        if let s = doc.transcript.browserURL, let u = URL(string: s) { p.load(u) }
        return p
    }

    /// Shows `url` in the chat's Browser tab.
    func openWebPage(_ doc: ChatDocument, _ url: URL) {
        guard doc.isCoding else { NSWorkspace.shared.open(url); return }
        let page = webPage(doc)
        page.load(url)
        doc.transcript.browserURL = url.absoluteString
        showSidePanel(doc, .web)
    }

    /// `open_page`: shows `url` in the Browser tab, waits for it to load (30 s at most), and reads its title, visible
    /// text and (if asked) a screenshot.
    func agentOpenPage(_ doc: ChatDocument, _ url: URL, screenshot: Bool) async -> PageView {
        guard doc.isCoding else { return PageView(url: url.absoluteString, error: "this chat has no browser") }
        let page = webPage(doc)
        let before = page.settled
        openWebPage(doc, url)
        let start = Date()
        while page.settled == before, !Task.isCancelled, Date().timeIntervalSince(start) < 30 {
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
        let shown = page.url?.absoluteString ?? url.absoluteString
        if let e = page.error { return PageView(url: shown, title: page.title, error: e) }
        if page.settled == before {
            return PageView(url: shown, title: page.title, error: "the page didn't finish loading in 30 seconds")
        }
        // Scripts often draw the page just after it loads.
        try? await Task.sleep(nanoseconds: 400_000_000)
        let text = (try? await page.view.evaluateJavaScript("document.body ? document.body.innerText : ''")) as? String
        var image: String?
        if screenshot, page.view.window != nil, let shot = try? await page.view.takeSnapshot(configuration: nil) {
            image = Self.jpegDataURL(shot, maxWidth: 1280)
        }
        return PageView(url: page.url?.absoluteString ?? shown, title: page.title, text: text ?? "", image: image)
    }

    /// An image as a JPEG `data:` URL, scaled down to `maxWidth` points at most.
    static func jpegDataURL(_ image: NSImage, maxWidth: CGFloat) -> String? {
        let scale = min(1, maxWidth / max(image.size.width, 1))
        let size = NSSize(width: (image.size.width * scale).rounded(), height: (image.size.height * scale).rounded())
        guard size.width > 0, size.height > 0,
              let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: Int(size.width), pixelsHigh: Int(size.height),
                                         bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                         colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        image.draw(in: NSRect(origin: .zero, size: size))
        NSGraphicsContext.restoreGraphicsState()
        guard let data = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.8]) else { return nil }
        return "data:image/jpeg;base64," + data.base64EncodedString()
    }

    /// Empties the Browser tab (and forgets its page).
    func closeWebPage(_ doc: ChatDocument) {
        doc.webPage?.view.stopLoading()
        doc.webPage?.view.removeFromSuperview()
        doc.webPage = nil
        doc.transcript.browserURL = nil
        saveChat(doc)
    }
}
