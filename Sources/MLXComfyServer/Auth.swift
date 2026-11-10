import CryptoKit
import Foundation
import HTTPTypes
import Hummingbird
import MLXComfyProtocol

/// Who a request comes from.
enum Principal: Equatable {
    /// `Authorization: Bearer <token>`: the app, the CLI, scripts. Full access.
    case bearer
    /// The dashboard's session cookie. Limited to dashboard actions; changes need the CSRF header.
    case session(id: String, csrf: String)
}

/// The bearer token and the dashboard's cookie sessions.
final class Auth: @unchecked Sendable {
    static let sessionCookie = "mlxc_session"
    static let csrfCookie = "mlxc_csrf"
    static let csrfHeader = HTTPField.Name("X-MLXComfy-CSRF")!
    static let sessionLifetime: TimeInterval = 12 * 3600

    private let tokenDigest: SHA256.Digest
    private var sessions: [String: (csrf: String, expires: Date)] = [:]
    private var failures: [Date] = []
    private let lock = NSLock()

    init(token: String) { tokenDigest = SHA256.hash(data: Data(token.utf8)) }

    /// Constant-time token check (compares digests, so neither length nor content leaks through timing).
    func check(token: String) -> Bool {
        let d = SHA256.hash(data: Data(token.utf8))
        var diff: UInt8 = 0
        for (a, b) in zip(d, tokenDigest) { diff |= a ^ b }
        return diff == 0
    }

    func principal(for request: Request) -> Principal? {
        if let header = request.headers[.authorization], header.hasPrefix("Bearer ") {
            return check(token: String(header.dropFirst(7)).trimmingCharacters(in: .whitespaces)) ? .bearer : nil
        }
        if let id = request.cookies[Self.sessionCookie]?.value {
            return lock.withLock {
                guard let s = sessions[id] else { return nil }
                guard s.expires > Date() else { sessions.removeValue(forKey: id); return nil }
                return .session(id: id, csrf: s.csrf)
            }
        }
        return nil
    }

    /// Trades the token for a session; nil (after a short delay that grows with recent failures) when wrong.
    func login(token: String) async -> (id: String, csrf: String)? {
        guard check(token: token) else {
            let recent: Int = lock.withLock {
                failures.append(Date())
                failures.removeAll { $0 < Date().addingTimeInterval(-60) }
                return failures.count
            }
            try? await Task.sleep(nanoseconds: UInt64(min(recent, 20)) * 250_000_000)
            ServerLog.shared.warning("dashboard login with a wrong token")
            return nil
        }
        let id = ServerConfig.generateToken(), csrf = ServerConfig.generateToken()
        lock.withLock {
            sessions = sessions.filter { $0.value.expires > Date() }
            sessions[id] = (csrf, Date().addingTimeInterval(Self.sessionLifetime))
        }
        return (id, csrf)
    }

    func logout(_ id: String) { lock.withLock { _ = sessions.removeValue(forKey: id) } }
}

/// Requires a principal for every `/v1` route except login and health, checks the CSRF header on cookie-authenticated
/// changes, and adds the security headers to every response.
struct SecurityMiddleware<Context: RequestContext>: RouterMiddleware {
    let auth: Auth
    let tls: Bool

    static func isPublic(_ request: Request) -> Bool {
        let path = request.uri.path
        if path == "/v1/health" { return true }
        if path == "/v1/session" && request.method == .post { return true }
        return !path.hasPrefix("/v1/")
    }

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws
        -> Response
    {
        var response: Response
        if Self.isPublic(request) {
            response = try await next(request, context)
        } else {
            guard let who = auth.principal(for: request) else { throw HTTPError(.unauthorized) }
            if case .session(_, let csrf) = who, ![.get, .head].contains(request.method) {
                guard let sent = request.headers[Auth.csrfHeader], Auth.constantTimeEqual(sent, csrf) else {
                    throw HTTPError(.forbidden, message: "missing or wrong CSRF header")
                }
            }
            response = try await next(request, context)
        }
        response.headers[.xContentTypeOptions] = "nosniff"
        response.headers[HTTPField.Name("Referrer-Policy")!] = "no-referrer"
        response.headers[HTTPField.Name("X-Frame-Options")!] = "DENY"
        response.headers[.contentSecurityPolicy] = "default-src 'self'; img-src 'self' blob: data:; "
            + "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
            + "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        if tls { response.headers[.strictTransportSecurity] = "max-age=31536000" }
        return response
    }
}

extension Auth {
    static func constantTimeEqual(_ a: String, _ b: String) -> Bool {
        let x = Array(a.utf8), y = Array(b.utf8)
        guard x.count == y.count else { return false }
        var diff: UInt8 = 0
        for i in 0..<x.count { diff |= x[i] ^ y[i] }
        return diff == 0
    }
}
