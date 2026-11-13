import Foundation
import Testing
@testable import AIStudioLibrary

struct WebAddressTests {
    @Test func typedAddresses() {
        #expect(WebAddress.url(from: "https://example.com/a?b=1")?.absoluteString == "https://example.com/a?b=1")
        #expect(WebAddress.url(from: " example.com/docs ")?.absoluteString == "https://example.com/docs")
        #expect(WebAddress.url(from: "localhost:3000")?.absoluteString == "http://localhost:3000")
        #expect(WebAddress.url(from: "127.0.0.1:8189/health")?.absoluteString == "http://127.0.0.1:8189/health")
        #expect(WebAddress.url(from: "app.localhost")?.absoluteString == "http://app.localhost")
        #expect(WebAddress.url(from: "192.168.1.20:8080")?.absoluteString == "http://192.168.1.20:8080")
        #expect(WebAddress.url(from: "file:///tmp/a.html")?.isFileURL == true)
        #expect(WebAddress.url(from: "about:blank")?.absoluteString == "about:blank")
    }

    @Test func notAddresses() {
        #expect(WebAddress.url(from: "") == nil)
        #expect(WebAddress.url(from: "how to sort") == nil)
        #expect(WebAddress.url(from: "hello") == nil)
        #expect(WebAddress.url(from: "ftp://example.com") == nil)
        #expect(WebAddress.url(from: "javascript://alert(1)") == nil)
        #expect(WebAddress.url(from: ".com") == nil)
    }

    @Test func shownAddresses() {
        #expect(WebAddress.display(URL(string: "https://example.com/")!) == "example.com")
        #expect(WebAddress.display(URL(string: "https://example.com/a/")!) == "example.com/a/")
        #expect(WebAddress.display(URL(string: "http://localhost:3000/")!) == "http://localhost:3000/")
    }
}
