package remoteclient

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/TNTcraftHIM/Piik/internal/server/protocol"
	"github.com/coder/websocket"
)

const (
	testPassword = "site-password"
	testToken    = "session-token"
	testService  = "piik"
)

func TestParseDescriptor(t *testing.T) {
	fingerprint := strings.Repeat("AB", 32)
	for _, test := range []struct {
		name  string
		raw   string
		want  Descriptor
		fails bool
	}{
		{
			name: "invitation",
			raw:  "piik://connect?origin=https%3A%2F%2Fdesk.example%3A8443&room=1234&grant=grant-1&fp=" + strings.ToLower(fingerprint),
			want: Descriptor{Origin: "https://desk.example:8443", RoomID: "1234", Grant: "grant-1", Fingerprint: fingerprint},
		},
		{
			name: "browser invite",
			raw:  "https://Desk.Example:8443/r/1234#v=grant-1",
			want: Descriptor{Origin: "https://desk.example:8443", RoomID: "1234", Grant: "grant-1"},
		},
		{
			name: "invite without room",
			raw:  "https://desk.example/",
			want: Descriptor{Origin: "https://desk.example"},
		},
		{
			name: "bare host and port",
			raw:  "desk.example:8443",
			want: Descriptor{Origin: "https://desk.example:8443"},
		},
		{
			name: "bare host",
			raw:  " desk.example ",
			want: Descriptor{Origin: "https://desk.example"},
		},
		{
			name: "plain http",
			raw:  "http://127.0.0.1:8787",
			want: Descriptor{Origin: "http://127.0.0.1:8787"},
		},
		{name: "empty", raw: "  ", fails: true},
		{name: "unsupported scheme", raw: "ftp://desk.example", fails: true},
		{name: "credentials", raw: "https://user:secret@desk.example", fails: true},
		{name: "short room", raw: "https://desk.example/r/123", fails: true},
		{name: "leading zero room", raw: "https://desk.example/r/0123", fails: true},
		{name: "invitation without origin", raw: "piik://connect?room=1234", fails: true},
		{name: "invitation with bad room", raw: "piik://connect?origin=desk.example&room=12", fails: true},
		{name: "invitation with bad fingerprint", raw: "piik://connect?origin=desk.example&fp=abcd", fails: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			descriptor, err := ParseDescriptor(test.raw)
			if test.fails {
				if err == nil {
					t.Fatalf("accepted %q: %+v", test.raw, descriptor)
				}
				return
			}
			if err != nil || descriptor != test.want {
				t.Fatalf("ParseDescriptor(%q) = %+v, %v", test.raw, descriptor, err)
			}
		})
	}
}

func TestProbeReportsTrustAndFirstUse(t *testing.T) {
	host := startTestHost(t)
	manager := NewManager()
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	pinned, err := manager.Probe(ctx, host.server.URL, host.fingerprint)
	if err != nil {
		t.Fatal(err)
	}
	if pinned.TLSStatus != TLSStatusTrusted || pinned.CertificateSHA256 != host.fingerprint ||
		pinned.Origin != host.server.URL || pinned.HostLabel != "127.0.0.1" ||
		pinned.Info.Service != testService || !pinned.Info.SiteAccessRequired || !pinned.Info.RoomCreation ||
		pinned.Info.Capabilities == nil || !pinned.Info.Capabilities.NatPrediction {
		t.Fatalf("pinned probe = %+v", pinned)
	}
	recorded := host.last(t, "/api/remote/info")
	if recorded.header.Get("X-Piik-Api") != "1" || recorded.header.Get("Accept") != "application/json" {
		t.Fatalf("probe headers = %v", recorded.header)
	}

	if _, err := manager.Probe(ctx, host.server.URL, strings.Repeat("C", 64)); err == nil ||
		!strings.Contains(err.Error(), "does not match the pinned fingerprint") {
		t.Fatalf("mismatched fingerprint = %v", err)
	}

	// Trust on first use: the certificate is reported, not accepted.
	untrusted, err := manager.Probe(ctx, host.server.URL, "")
	if err != nil {
		t.Fatal(err)
	}
	if untrusted.TLSStatus != TLSStatusUntrusted || untrusted.CertificateSHA256 != host.fingerprint ||
		untrusted.Info != (RemoteInfo{}) {
		t.Fatalf("first-use probe = %+v", untrusted)
	}
}

func TestConnectKeepsOneSessionAndForwardsAllowlistedCalls(t *testing.T) {
	host := startTestHost(t)
	manager := NewManager()
	t.Cleanup(manager.Disconnect)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()

	status, err := manager.Connect(ctx, ConnectRequest{
		Target: host.server.URL, AcceptedFingerprint: host.fingerprint, SitePassword: testPassword,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !status.Connected || status.Origin != host.server.URL || status.CertificateSHA256 != host.fingerprint ||
		!status.SiteAccessRequired || !status.RoomCreation || status.Protocol != protocol.SignalingProtocol ||
		status.Capabilities == nil || !status.Capabilities.NatPrediction {
		t.Fatalf("session = %+v", status)
	}
	if strings.Contains(fmt.Sprintf("%+v", status), testToken) {
		t.Fatal("session status leaked the API token")
	}
	opened := host.last(t, "/api/remote/session")
	if opened.header.Get("X-Piik-Api") != "1" || opened.body != `{"password":"`+testPassword+`"}` {
		t.Fatalf("session request = %+v", opened)
	}
	if host.last(t, "/api/capabilities").header.Get("X-Piik-Api-Token") != testToken {
		t.Fatal("capabilities request carried no session token")
	}
	if manager.Status() != status {
		t.Fatalf("status = %+v", manager.Status())
	}

	// The allowlisted calls travel with the session token and nothing else.
	response := forward(t, manager, http.MethodGet, "/api/capabilities", "", "")
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "natPrediction") {
		t.Fatalf("capabilities proxy = %d %s", response.Code, response.Body.String())
	}
	response = forward(t, manager, http.MethodPost, "/api/rooms", `{"roomId":"1234"}`, "Bearer host-token")
	if response.Code != http.StatusOK {
		t.Fatalf("room creation proxy = %d %s", response.Code, response.Body.String())
	}
	created := host.last(t, "/api/rooms")
	if created.method != http.MethodPost || created.body != `{"roomId":"1234"}` ||
		created.header.Get("X-Piik-Api") != "1" || created.header.Get("X-Piik-Api-Token") != testToken ||
		created.header.Get("Content-Type") != "application/json" ||
		created.header.Get("Origin") != "" || created.header.Get("Cookie") != "" ||
		created.header.Get("Authorization") != "" {
		t.Fatalf("proxied room creation = %+v", created)
	}
	for _, path := range []string{"/api/rooms/1234/replacement", "/api/rooms/1234/access"} {
		response = forward(t, manager, http.MethodPost, path, `{}`, "Bearer host-token")
		if response.Code != http.StatusOK {
			t.Fatalf("%s proxy = %d %s", path, response.Code, response.Body.String())
		}
		if host.last(t, path).header.Get("Authorization") != "Bearer host-token" {
			t.Fatalf("%s lost the host token", path)
		}
	}

	for _, path := range []string{
		"/", "/index.html", "/api/site-access", "/api/rooms/0123/access",
		"/api/rooms/1234", "/../../api/site-access", "/api/capabilities/extra",
	} {
		if response = forward(t, manager, http.MethodGet, path, "", ""); response.Code != http.StatusNotFound {
			t.Fatalf("proxy admitted %q: %d", path, response.Code)
		}
	}
	if response = forward(t, manager, http.MethodPost, "/api/capabilities", `{}`, ""); response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("wrong method = %d", response.Code)
	}
	oversized := `{"padding":"` + strings.Repeat("x", maxProxyBytes) + `"}`
	if response = forward(t, manager, http.MethodPost, "/api/rooms", oversized, ""); response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body = %d", response.Code)
	}

	manager.Disconnect()
	if manager.Status().Connected {
		t.Fatal("disconnect kept the session")
	}
	if response = forward(t, manager, http.MethodGet, "/api/capabilities", "", ""); response.Code != http.StatusServiceUnavailable {
		t.Fatalf("proxy without a session = %d", response.Code)
	}
}

// A host without site access issues no token, and the session still works.
func TestConnectWithoutSiteAccess(t *testing.T) {
	host := startTestHostWithAccess(t, false)
	manager := NewManager()
	t.Cleanup(manager.Disconnect)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	status, err := manager.Connect(ctx, ConnectRequest{
		Target: host.server.URL, AcceptedFingerprint: host.fingerprint,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !status.Connected || status.SiteAccessRequired || status.Capabilities == nil {
		t.Fatalf("open session = %+v", status)
	}
	if host.last(t, "/api/capabilities").header.Get("X-Piik-Api-Token") != "" {
		t.Fatal("open host was sent a token header")
	}
	if response := forward(t, manager, http.MethodGet, "/api/capabilities", "", ""); response.Code != http.StatusOK {
		t.Fatalf("open host proxy = %d", response.Code)
	}
}

func TestConnectRejectsAWrongSitePassword(t *testing.T) {
	host := startTestHost(t)
	manager := NewManager()
	t.Cleanup(manager.Disconnect)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	_, err := manager.Connect(ctx, ConnectRequest{
		Target: host.server.URL, AcceptedFingerprint: host.fingerprint, SitePassword: "wrong",
	})
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("wrong password = %v", err)
	}
	if manager.Status().Connected {
		t.Fatal("rejected password opened a session")
	}
}

func TestBridgeSignalRelaysFramesUntilDisconnect(t *testing.T) {
	host := startTestHost(t)
	manager := NewManager()
	t.Cleanup(manager.Disconnect)
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if _, err := manager.Connect(ctx, ConnectRequest{
		Target: host.server.URL, AcceptedFingerprint: host.fingerprint, SitePassword: testPassword,
	}); err != nil {
		t.Fatal(err)
	}

	local := httptest.NewServer(http.HandlerFunc(manager.BridgeSignal))
	defer local.Close()
	connection, _, err := websocket.Dial(ctx, "ws"+strings.TrimPrefix(local.URL, "http"), &websocket.DialOptions{
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer connection.CloseNow()
	connection.SetReadLimit(protocol.MaxSignalBytes)
	if err := connection.Write(ctx, websocket.MessageText, []byte("hello")); err != nil {
		t.Fatal(err)
	}
	messageType, payload, err := connection.Read(ctx)
	if err != nil || messageType != websocket.MessageText || string(payload) != "hello" {
		t.Fatalf("relayed frame = %v %q %v", messageType, payload, err)
	}
	if host.last(t, "/signal").header.Get("X-Piik-Api-Token") != testToken {
		t.Fatal("signaling bridge carried no session token")
	}

	// Disconnect ends the session and the bridge with it.
	manager.Disconnect()
	readCtx, readCancel := context.WithTimeout(ctx, 5*time.Second)
	defer readCancel()
	if _, _, err := connection.Read(readCtx); err == nil {
		t.Fatal("bridge outlived the session")
	}
}

type recordedRequest struct {
	method string
	path   string
	header http.Header
	body   string
}

type testHost struct {
	server        *httptest.Server
	fingerprint   string
	requireAccess bool
	mu            sync.Mutex
	requests      []recordedRequest
}

func startTestHost(t *testing.T) *testHost {
	return startTestHostWithAccess(t, true)
}

// startTestHostWithAccess serves the host contract; requireAccess mirrors a
// service that asks for a site password before it issues an API token.
func startTestHostWithAccess(t *testing.T, requireAccess bool) *testHost {
	t.Helper()
	host := &testHost{requireAccess: requireAccess}
	token := ""
	if requireAccess {
		token = testToken
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/remote/info", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		writeTestJSON(response, map[string]any{
			"service":            testService,
			"protocol":           protocol.SignalingProtocol,
			"siteAccessRequired": host.requireAccess,
			"roomCreation":       true,
			"capabilities":       map[string]any{"sfu": false, "natPrediction": true},
		})
	})
	mux.HandleFunc("/api/remote/session", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		var body struct {
			Password string `json:"password"`
		}
		decoded := json.NewDecoder(request.Body).Decode(&body) == nil
		if host.requireAccess && (!decoded || body.Password != testPassword) {
			response.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeTestJSON(response, map[string]any{
			"required":      host.requireAccess,
			"authenticated": true,
			"token":         token,
			"expiresAtMs":   0,
			"ttlSeconds":    86400,
		})
	})
	mux.HandleFunc("/api/capabilities", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		if request.Header.Get("X-Piik-Api-Token") != token {
			response.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeTestJSON(response, map[string]any{"sfu": false, "natPrediction": true, "roomInteractions": true})
	})
	mux.HandleFunc("/api/rooms", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		writeTestJSON(response, map[string]any{"roomId": "1234"})
	})
	mux.HandleFunc("/api/rooms/1234/replacement", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		writeTestJSON(response, map[string]any{"replaced": true})
	})
	mux.HandleFunc("/api/rooms/1234/access", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		writeTestJSON(response, map[string]any{"granted": true})
	})
	mux.HandleFunc("/signal", func(response http.ResponseWriter, request *http.Request) {
		host.record(request)
		if request.Header.Get("X-Piik-Api-Token") != testToken {
			response.WriteHeader(http.StatusUnauthorized)
			return
		}
		connection, err := websocket.Accept(response, request, &websocket.AcceptOptions{
			InsecureSkipVerify: true,
			CompressionMode:    websocket.CompressionDisabled,
		})
		if err != nil {
			return
		}
		defer connection.CloseNow()
		connection.SetReadLimit(protocol.MaxSignalBytes)
		for {
			messageType, payload, err := connection.Read(context.Background())
			if err != nil {
				return
			}
			if err := connection.Write(context.Background(), messageType, payload); err != nil {
				return
			}
		}
	})
	host.server = httptest.NewTLSServer(mux)
	t.Cleanup(host.server.Close)
	host.fingerprint = fmt.Sprintf("%X", sha256.Sum256(host.server.Certificate().Raw))
	return host
}

// record captures the request and restores its body so the handler below can
// still decode it.
func (h *testHost) record(request *http.Request) {
	payload, _ := io.ReadAll(request.Body)
	request.Body = io.NopCloser(bytes.NewReader(payload))
	h.mu.Lock()
	defer h.mu.Unlock()
	h.requests = append(h.requests, recordedRequest{
		method: request.Method, path: request.URL.Path, header: request.Header.Clone(), body: string(payload),
	})
}

func (h *testHost) last(t *testing.T, path string) recordedRequest {
	t.Helper()
	h.mu.Lock()
	defer h.mu.Unlock()
	for index := len(h.requests) - 1; index >= 0; index-- {
		if h.requests[index].path == path {
			return h.requests[index]
		}
	}
	t.Fatalf("no request recorded for %s", path)
	return recordedRequest{}
}

func forward(t *testing.T, manager *Manager, method, path, body, authorization string) *httptest.ResponseRecorder {
	t.Helper()
	request := httptest.NewRequest(method, ProxyPrefix+path, strings.NewReader(body))
	request.Header.Set("Origin", "http://127.0.0.1:1")
	request.Header.Set("Cookie", "piik-site-access=granted")
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	if authorization != "" {
		request.Header.Set("Authorization", authorization)
	}
	response := httptest.NewRecorder()
	manager.ForwardAPI(response, request)
	return response
}

func writeTestJSON(response http.ResponseWriter, value any) {
	response.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(response).Encode(value)
}
