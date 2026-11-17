package app

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/server/protocol"
)

// withAPIClient marks a request as a non-browser API client: no Origin, the
// explicit API header and, when given, its API token.
func withAPIClient(token string) requestOption {
	return func(request *http.Request) {
		request.Header.Set("X-Piik-Api", "1")
		if token != "" {
			request.Header.Set("X-Piik-Api-Token", token)
		}
	}
}

// remoteSession is the API client's login: the site password for an API token.
func (h *harness) remoteSession(password string) remoteSessionBody {
	h.t.Helper()
	response := h.do(http.MethodPost, "/api/remote/session", withBearer(password)).
		expectStatus(http.StatusOK)
	var session remoteSessionBody
	if err := json.Unmarshal([]byte(response.body), &session); err != nil {
		h.t.Fatalf("decode remote session from %s: %v", response.body, err)
	}
	return session
}

func TestRemoteInfoDescribesTheService(t *testing.T) {
	server := start(t, Options{RemoteCertificateSHA256: "ab12cd34"})
	response := server.do(http.MethodGet, "/api/remote/info").
		expectStatus(http.StatusOK).
		expectHeader("Cache-Control", "no-store").
		expectHeader("X-Content-Type-Options", "nosniff")

	var info remoteInfoBody
	if err := json.Unmarshal([]byte(response.body), &info); err != nil {
		t.Fatalf("decode remote info from %s: %v", response.body, err)
	}
	if info.Service != "piik" || info.Protocol != protocol.SignalingProtocol {
		t.Fatalf("info = %+v", info)
	}
	if !info.SiteAccessRequired || !info.RoomCreation {
		t.Fatalf("info = %+v", info)
	}
	if info.APIOrigin != "https://share.example.test" {
		t.Fatalf("apiOrigin = %q", info.APIOrigin)
	}
	if info.CertificateSHA256 != "ab12cd34" {
		t.Fatalf("certificateSha256 = %q", info.CertificateSHA256)
	}

	// Both routes advertise the one capability set.
	var advertised protocol.RuntimeCapabilities
	capabilities := server.do(http.MethodGet, "/api/capabilities").expectStatus(http.StatusOK)
	if err := json.Unmarshal([]byte(capabilities.body), &advertised); err != nil {
		t.Fatalf("decode capabilities from %s: %v", capabilities.body, err)
	}
	if !reflect.DeepEqual(info.Capabilities, advertised) {
		t.Fatalf("remote capabilities = %+v, want %+v", info.Capabilities, advertised)
	}
	if !info.Capabilities.RoomInteractions || !info.Capabilities.HostRoomSession {
		t.Fatalf("capabilities = %+v", info.Capabilities)
	}
}

func TestRemoteInfoOmitsAnUnsetCertificate(t *testing.T) {
	server := start(t, Options{})
	response := server.do(http.MethodGet, "/api/remote/info").expectStatus(http.StatusOK)
	if strings.Contains(response.body, "certificateSha256") {
		t.Fatalf("info = %s must omit an unset certificate", response.body)
	}
}

func TestRemoteInfoRejectsForeignOriginsAndOtherMethods(t *testing.T) {
	server := start(t, Options{})
	server.do(http.MethodGet, "/api/remote/info", withOrigin(allowedOrigin)).
		expectStatus(http.StatusOK)
	for _, origin := range []string{"https://foreign.test", allowedOrigin + "/path", "not-an-origin"} {
		server.do(http.MethodGet, "/api/remote/info", withOrigin(origin)).
			expect(http.StatusForbidden, `{"error":"Origin not allowed"}`)
	}
	server.do(http.MethodPost, "/api/remote/info").
		expect(http.StatusMethodNotAllowed, `{"error":"Method not allowed"}`).
		expectHeader("Allow", http.MethodGet)
}

func TestRemoteSessionIssuesAnAPI1Token(t *testing.T) {
	server := start(t, Options{})
	response := server.do(http.MethodPost, "/api/remote/session", withBearer(testAccessPassword)).
		expectStatus(http.StatusOK).
		expectHeader("Cache-Control", "no-store")

	var session remoteSessionBody
	if err := json.Unmarshal([]byte(response.body), &session); err != nil {
		t.Fatalf("decode remote session from %s: %v", response.body, err)
	}
	if !session.Required || !session.Authenticated {
		t.Fatalf("session = %+v", session)
	}
	if !strings.HasPrefix(session.Token, apiTokenVersion+".") {
		t.Fatalf("token = %q, want the %q version", session.Token, apiTokenVersion)
	}
	if session.TTLSeconds != defaultSessionTTLSeconds || session.ExpiresAtMs <= 0 {
		t.Fatalf("session = %+v", session)
	}
	if strings.Contains(session.Token, testAccessPassword) {
		t.Fatal("the token must never carry the access password")
	}

	// The token authorizes room creation without a cookie or an Origin.
	server.do(http.MethodPost, "/api/rooms", withAPIClient(session.Token),
		withJSON(`{"codeEntryPolicy":"open"}`)).expectStatus(http.StatusCreated)

	// A JSON body carries the same password, and a body without one falls back
	// to the Bearer credential.
	server.do(http.MethodPost, "/api/remote/session", withJSON(`{"password":"`+testAccessPassword+`"}`)).
		expectStatus(http.StatusOK)
	server.do(http.MethodPost, "/api/remote/session", withBearer(testAccessPassword), withJSON(`{}`)).
		expectStatus(http.StatusOK)
}

func TestRemoteSessionRejectsWrongCredentialsAndMalformedBodies(t *testing.T) {
	server := start(t, Options{})
	server.do(http.MethodPost, "/api/remote/session", withBearer("wrong-password")).
		expect(http.StatusUnauthorized, `{"error":"Unauthorized"}`).
		expectHeader("WWW-Authenticate", "Bearer")
	server.do(http.MethodPost, "/api/remote/session").
		expect(http.StatusUnauthorized, `{"error":"Unauthorized"}`)
	for _, body := range []string{
		`{"password":`,
		`{"password":null}`,
		`{"password":"a","other":"b"}`,
		`{"other":"b"}`,
		`[]`,
	} {
		server.do(http.MethodPost, "/api/remote/session", withJSON(body)).
			expect(http.StatusBadRequest, `{"error":"Invalid remote session request"}`)
	}
	server.do(http.MethodPost, "/api/remote/session", withBody("text/plain", "password")).
		expect(http.StatusBadRequest, `{"error":"Invalid remote session request"}`)
	server.do(http.MethodPost, "/api/remote/session", withOrigin("https://foreign.test"),
		withBearer(testAccessPassword)).
		expect(http.StatusForbidden, `{"error":"Origin not allowed"}`)
	server.do(http.MethodGet, "/api/remote/session").
		expect(http.StatusMethodNotAllowed, `{"error":"Method not allowed"}`).
		expectHeader("Allow", http.MethodPost)
}

func TestRemoteSessionWithoutSiteAccessNeedsNoPassword(t *testing.T) {
	configuration := testConfig(t)
	configuration.SiteAccessPassword = ""
	server := start(t, Options{Config: configuration})

	server.do(http.MethodPost, "/api/remote/session").
		expect(http.StatusOK, `{"required":false,"authenticated":true,"token":"","expiresAtMs":0,"ttlSeconds":0}`)
	server.do(http.MethodPost, "/api/remote/session", withJSON(`{}`)).
		expectStatus(http.StatusOK)

	response := server.do(http.MethodGet, "/api/remote/info").expectStatus(http.StatusOK)
	if !strings.Contains(response.body, `"siteAccessRequired":false`) {
		t.Fatalf("info = %s", response.body)
	}
}

func TestAPITokenIsNotInterchangeableWithTheSiteCookie(t *testing.T) {
	now := int64(1_000)
	access := newTestAccess(t, siteAccessOptions{
		Password:   testAccessPassword,
		Now:        func() int64 { return now },
		TTLSeconds: 60,
	})

	token, expiresAtMs := access.createAPIToken()
	if want := (floorSeconds(now) + 60) * 1_000; expiresAtMs != want {
		t.Fatalf("expiresAtMs = %d, want %d", expiresAtMs, want)
	}
	if !strings.HasPrefix(token, apiTokenVersion+".") || !access.isValidAPIToken(token) {
		t.Fatalf("token = %q", token)
	}
	if access.isAuthenticated("piik-site-access=" + token) {
		t.Fatal("an API token must not authenticate a site cookie")
	}
	cookieValue := strings.TrimPrefix(cookiePair(t, access.createCookie()), "piik-site-access=")
	if access.isValidAPIToken(cookieValue) {
		t.Fatal("a site cookie must not act as an API token")
	}
	for _, malformed := range []string{"", "api1", "api1.1", "api1.1.", "v1.1." + strings.Repeat("a", 43)} {
		if access.isValidAPIToken(malformed) {
			t.Fatalf("malformed token %q was accepted", malformed)
		}
	}

	// A token expires with the same rule as the cookie.
	now += 60_000
	if access.isValidAPIToken(token) {
		t.Fatal("an expired API token must be rejected")
	}

	// Only the issuing process key validates the signature.
	other := newTestAccess(t, siteAccessOptions{Password: testAccessPassword})
	if other.isValidAPIToken(token) {
		t.Fatal("an API token must authenticate only its issuing process")
	}
}

func TestAPITokenIsUnnecessaryWithoutSiteAccess(t *testing.T) {
	access := newTestAccess(t, siteAccessOptions{})
	token, expiresAtMs := access.createAPIToken()
	if token != "" || expiresAtMs != 0 {
		t.Fatalf("createAPIToken() = (%q, %d), want (\"\", 0)", token, expiresAtMs)
	}
	if !access.isValidAPIToken("") {
		t.Fatal("site access that requires no password must accept an absent token")
	}
}

func TestRoomRoutesAcceptAnAPIClient(t *testing.T) {
	server := start(t, Options{})
	token := server.remoteSession(testAccessPassword).Token

	created := server.do(http.MethodPost, "/api/rooms", withAPIClient(token),
		withJSON(`{"codeEntryPolicy":"open"}`)).expectStatus(http.StatusCreated).room()

	server.do(http.MethodPost, "/api/rooms/"+created.RoomID+"/access",
		withAPIClient(token), withBearer(created.HostToken),
		withJSON(`{"action":"rotate-viewer-grant"}`)).expectStatus(http.StatusOK)

	replacement := server.do(http.MethodPost, "/api/rooms/"+created.RoomID+"/replacement",
		withAPIClient(token), withBearer(created.HostToken),
		withJSON(`{"codeEntryPolicy":"open"}`)).expectStatus(http.StatusCreated).room()
	if replacement.RoomID == created.RoomID {
		t.Fatal("the replacement kept the old room code")
	}

	// /api/rooms has no Host token of its own, so the API token may travel as
	// the Bearer credential there.
	server.do(http.MethodPost, "/api/rooms", withAPIClient(""), withBearer(token),
		withJSON(`{"codeEntryPolicy":"open"}`)).expectStatus(http.StatusCreated)
}

func TestRoomRoutesRejectUntrustedOriginsAndMissingTokens(t *testing.T) {
	server := start(t, Options{})
	token := server.remoteSession(testAccessPassword).Token
	created := server.do(http.MethodPost, "/api/rooms", withAPIClient(token),
		withJSON(`{"codeEntryPolicy":"open"}`)).expectStatus(http.StatusCreated).room()
	cookie := server.cookie()
	body := `{"codeEntryPolicy":"open"}`

	for _, path := range []string{
		"/api/rooms",
		"/api/rooms/" + created.RoomID + "/replacement",
		"/api/rooms/" + created.RoomID + "/access",
	} {
		// A browser Origin that is not configured is refused even with a valid
		// API token, so a page cannot borrow the client's credential.
		server.do(http.MethodPost, path, withOrigin("https://foreign.test"),
			withAPIClient(token), withBearer(created.HostToken), withJSON(body)).
			expect(http.StatusForbidden, `{"error":"Origin not allowed"}`)
		// Neither a configured Origin nor the API declaration.
		server.do(http.MethodPost, path, withBearer(created.HostToken), withJSON(body)).
			expect(http.StatusForbidden, `{"error":"Origin not allowed"}`)
		// A declared client without a token, and a site cookie is not a token.
		server.do(http.MethodPost, path, withAPIClient(""), withBearer(created.HostToken), withJSON(body)).
			expect(http.StatusUnauthorized, `{"error":"Unauthorized"}`)
		server.do(http.MethodPost, path, withAPIClient(""), withCookie(cookie),
			withBearer(created.HostToken), withJSON(body)).
			expect(http.StatusUnauthorized, `{"error":"Unauthorized"}`)
	}

	// The room routes read the Host token from Bearer, so an API token there is
	// not a site credential.
	server.do(http.MethodPost, "/api/rooms/"+created.RoomID+"/replacement",
		withAPIClient(""), withBearer(token), withJSON(body)).
		expect(http.StatusUnauthorized, `{"error":"Unauthorized"}`)
}

func TestSignalAdmissionAcceptsAnAPIClient(t *testing.T) {
	server := start(t, Options{})
	token := server.remoteSession(testAccessPassword).Token
	upgrade := func() *http.Request {
		return httptest.NewRequest(http.MethodGet, "https://share.example.test/signal", nil)
	}

	browser := upgrade()
	browser.Header.Set("Origin", allowedOrigin)
	browser.Header.Set("Cookie", server.cookie())
	if !server.signalOptions.SiteAccessAtUpgrade(browser) {
		t.Fatal("a browser cookie must still admit a signaling upgrade")
	}
	if server.signalOptions.AllowAPIClient(browser) {
		t.Fatal("a browser request must not be admitted as an API client")
	}

	client := upgrade()
	client.Header.Set("X-Piik-Api", "1")
	client.Header.Set("X-Piik-Api-Token", token)
	if !server.signalOptions.AllowAPIClient(client) || !server.signalOptions.SiteAccessAtUpgrade(client) {
		t.Fatal("an API client with a valid token must be admitted")
	}

	bearer := upgrade()
	bearer.Header.Set("X-Piik-Api", "1")
	bearer.Header.Set("Authorization", "Bearer "+token)
	if !server.signalOptions.SiteAccessAtUpgrade(bearer) {
		t.Fatal("the Bearer fallback must admit an API client")
	}

	missing := upgrade()
	missing.Header.Set("X-Piik-Api", "1")
	if server.signalOptions.SiteAccessAtUpgrade(missing) {
		t.Fatal("an API client without a token must not be admitted")
	}

	cookieOnly := upgrade()
	cookieOnly.Header.Set("X-Piik-Api", "1")
	cookieOnly.Header.Set("Cookie", server.cookie())
	if server.signalOptions.SiteAccessAtUpgrade(cookieOnly) {
		t.Fatal("a site cookie must not act as an API token")
	}

	// A browser origin is never an API client, whatever else it sends.
	origin := upgrade()
	origin.Header.Set("Origin", allowedOrigin)
	origin.Header.Set("X-Piik-Api", "1")
	origin.Header.Set("X-Piik-Api-Token", token)
	if server.signalOptions.AllowAPIClient(origin) {
		t.Fatal("a request with an Origin must not be admitted as an API client")
	}
}
