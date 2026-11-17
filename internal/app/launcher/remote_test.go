package launcher

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	appconfig "github.com/TNTcraftHIM/Piik/internal/app/config"
	"github.com/TNTcraftHIM/Piik/internal/app/remoteclient"
	"github.com/TNTcraftHIM/Piik/internal/server/protocol"
	"github.com/coder/websocket"
)

func TestLauncherRemoteEndpoints(t *testing.T) {
	host := startRemoteFixture(t)
	var savedMu sync.Mutex
	var saved []appconfig.RemoteHost
	server, err := Start(t.Context(), appAssets(), Options{
		Version: "development",
		ChangeRemoteHost: func(change appconfig.RemoteHost, allowed bool) ([]appconfig.RemoteHost, error) {
			savedMu.Lock()
			defer savedMu.Unlock()
			saved = slices.DeleteFunc(saved, func(value appconfig.RemoteHost) bool { return value.Origin == change.Origin })
			if allowed {
				saved = append(saved, change)
			}
			return saved, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Close() })
	origin := strings.TrimSuffix(server.URL(), "/client")

	var status remoteclient.SessionStatus
	response := remoteDo(t, http.MethodGet, origin+"/api/client-remote/status", "")
	decodeRemoteResponse(t, response, &status)
	if response.StatusCode != http.StatusOK || status.Connected {
		t.Fatalf("initial status = %d, %+v", response.StatusCode, status)
	}

	// First contact reports the certificate instead of trusting it.
	var probe remoteclient.ProbeResult
	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/probe",
		`{"target":"`+host.server.URL+`"}`)
	decodeRemoteResponse(t, response, &probe)
	if response.StatusCode != http.StatusOK || probe.TLSStatus != remoteclient.TLSStatusUntrusted ||
		probe.CertificateSHA256 != host.fingerprint || probe.Info != (remoteclient.RemoteInfo{}) {
		t.Fatalf("probe = %d, %+v", response.StatusCode, probe)
	}

	// Pinning the reported fingerprint lets the probe read /api/remote/info.
	var pinnedProbe remoteclient.ProbeResult
	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/probe",
		`{"target":"`+host.server.URL+`","acceptedFingerprint":"`+probe.CertificateSHA256+`"}`)
	decodeRemoteResponse(t, response, &pinnedProbe)
	if response.StatusCode != http.StatusOK || pinnedProbe.TLSStatus != remoteclient.TLSStatusTrusted ||
		pinnedProbe.CertificateSHA256 != host.fingerprint || pinnedProbe.Info.Service != "piik" {
		t.Fatalf("pinned probe = %d, %+v", response.StatusCode, pinnedProbe)
	}
	if response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/probe", `{"target":"ftp://desk"}`); response.StatusCode != http.StatusBadRequest {
		response.Body.Close()
		t.Fatalf("invalid descriptor = %d", response.StatusCode)
	}
	response.Body.Close()

	// The launcher's own admission rule still guards every remote route.
	for _, path := range []string{"probe", "connect", "disconnect", "host"} {
		request, _ := http.NewRequest(http.MethodPost, origin+"/api/client-remote/"+path, strings.NewReader(`{}`))
		request.Header.Set("Origin", "https://unrelated.example")
		request.Header.Set("Content-Type", "application/json")
		rejected, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
		if err != nil {
			t.Fatal(err)
		}
		rejected.Body.Close()
		if rejected.StatusCode != http.StatusNotFound {
			t.Fatalf("%s admitted a cross-origin request: %d", path, rejected.StatusCode)
		}
	}

	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/connect",
		`{"target":"`+host.server.URL+`","acceptedFingerprint":"`+host.fingerprint+`","sitePassword":"wrong"}`)
	response.Body.Close()
	if response.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong password = %d", response.StatusCode)
	}

	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/connect",
		`{"target":"`+host.server.URL+`","acceptedFingerprint":"`+host.fingerprint+`","sitePassword":"site-password"}`)
	payload, err := io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	var connected remoteclient.SessionStatus
	if response.StatusCode != http.StatusOK || json.Unmarshal(payload, &connected) != nil ||
		!connected.Connected || connected.CertificateSHA256 != host.fingerprint ||
		connected.Capabilities == nil || !connected.Capabilities.NatPrediction {
		t.Fatalf("connect = %d, %s", response.StatusCode, payload)
	}
	if strings.Contains(string(payload), host.token) {
		t.Fatal("connect response leaked the API token")
	}
	if status = server.remote.Status(); !status.Connected {
		t.Fatalf("manager status = %+v", status)
	}

	// Only the allowlisted remote API calls pass through the proxy.
	response = remoteDo(t, http.MethodGet, origin+"/api/client-remote/proxy/api/capabilities", "")
	payload, err = io.ReadAll(response.Body)
	response.Body.Close()
	if err != nil || response.StatusCode != http.StatusOK || !strings.Contains(string(payload), "natPrediction") {
		t.Fatalf("proxied capabilities = %d, %s, %v", response.StatusCode, payload, err)
	}
	for _, path := range []string{"", "/", "/index.html", "/api/site-access", "/../../api/site-access"} {
		response = remoteDo(t, http.MethodGet, origin+"/api/client-remote/proxy"+path, "")
		response.Body.Close()
		if response.StatusCode != http.StatusNotFound {
			t.Fatalf("proxy admitted %q: %d", path, response.StatusCode)
		}
	}

	// The signaling bridge relays frames to the saved host and back.
	readCtx, readCancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer readCancel()
	bridge, _, err := websocket.Dial(readCtx, "ws"+strings.TrimPrefix(origin, "http")+"/api/client-remote/proxy/signal",
		&websocket.DialOptions{CompressionMode: websocket.CompressionDisabled})
	if err != nil {
		t.Fatal(err)
	}
	bridge.SetReadLimit(protocol.MaxSignalBytes)
	if err := bridge.Write(readCtx, websocket.MessageText, []byte("hello")); err != nil {
		t.Fatal(err)
	}
	messageType, message, err := bridge.Read(readCtx)
	if err != nil || messageType != websocket.MessageText || string(message) != "hello" {
		t.Fatalf("bridged frame = %v %q %v", messageType, message, err)
	}
	bridge.CloseNow()

	// Approving a host persists the origin and its fingerprint, never a secret.
	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/host",
		`{"origin":"`+host.server.URL+`","fingerprint":"`+strings.ToLower(host.fingerprint)+`","allowed":true}`)
	var approval struct {
		RemoteHosts []appconfig.RemoteHost `json:"remoteHosts"`
	}
	decodeRemoteResponse(t, response, &approval)
	if response.StatusCode != http.StatusOK || len(approval.RemoteHosts) != 1 ||
		approval.RemoteHosts[0].Origin != host.server.URL || approval.RemoteHosts[0].Fingerprint != host.fingerprint {
		t.Fatalf("host approval = %d, %+v", response.StatusCode, approval)
	}
	var state struct {
		RemoteHosts []appconfig.RemoteHost `json:"remoteHosts"`
	}
	response = remoteDo(t, http.MethodGet, origin+"/api/client-launcher", "")
	decodeRemoteResponse(t, response, &state)
	if len(state.RemoteHosts) != 1 || state.RemoteHosts[0].Origin != host.server.URL {
		t.Fatalf("launcher state hosts = %+v", state.RemoteHosts)
	}

	response = remoteDo(t, http.MethodPost, origin+"/api/client-remote/disconnect", `{}`)
	status = remoteclient.SessionStatus{}
	decodeRemoteResponse(t, response, &status)
	if response.StatusCode != http.StatusOK || status.Connected {
		t.Fatalf("disconnect = %d, %+v", response.StatusCode, status)
	}
	if err := server.Close(); err != nil {
		t.Fatal(err)
	}
	if server.remote.Status().Connected {
		t.Fatal("launcher close kept the remote session")
	}
}

func remoteDo(t *testing.T, method, target, body string) *http.Response {
	t.Helper()
	request, err := http.NewRequest(method, target, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	address, err := url.Parse(target)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Origin", address.Scheme+"://"+address.Host)
	request.Close = true
	if body != "" {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := (&http.Client{Timeout: 20 * time.Second}).Do(request)
	if err != nil {
		t.Fatal(err)
	}
	return response
}

func decodeRemoteResponse(t *testing.T, response *http.Response, target any) {
	t.Helper()
	defer response.Body.Close()
	if err := json.NewDecoder(response.Body).Decode(target); err != nil {
		t.Fatalf("decode %s: %v", response.Request.URL, err)
	}
}

// startRemoteFixture serves the remote host contract the App connects to.
type remoteFixture struct {
	server      *httptest.Server
	fingerprint string
	token       string
}

func startRemoteFixture(t *testing.T) *remoteFixture {
	t.Helper()
	fixture := &remoteFixture{token: "session-token"}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/remote/info", func(response http.ResponseWriter, request *http.Request) {
		writeFixtureJSON(response, map[string]any{
			"service":            "piik",
			"protocol":           protocol.SignalingProtocol,
			"siteAccessRequired": true,
			"roomCreation":       true,
			"capabilities":       map[string]any{"sfu": false, "natPrediction": true},
		})
	})
	mux.HandleFunc("/api/remote/session", func(response http.ResponseWriter, request *http.Request) {
		var body struct {
			Password string `json:"password"`
		}
		if json.NewDecoder(request.Body).Decode(&body) != nil || body.Password != "site-password" {
			response.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeFixtureJSON(response, map[string]any{
			"required":      true,
			"authenticated": true,
			"token":         fixture.token,
			"expiresAtMs":   0,
			"ttlSeconds":    86400,
		})
	})
	mux.HandleFunc("/api/capabilities", func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Piik-Api-Token") != fixture.token {
			response.WriteHeader(http.StatusUnauthorized)
			return
		}
		writeFixtureJSON(response, map[string]any{"sfu": false, "natPrediction": true, "roomInteractions": true})
	})
	mux.HandleFunc("/signal", func(response http.ResponseWriter, request *http.Request) {
		if request.Header.Get("X-Piik-Api-Token") != fixture.token {
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
	fixture.server = httptest.NewTLSServer(mux)
	t.Cleanup(fixture.server.Close)
	fixture.fingerprint = fmt.Sprintf("%X", sha256.Sum256(fixture.server.Certificate().Raw))
	return fixture
}

func writeFixtureJSON(response http.ResponseWriter, value any) {
	response.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(response).Encode(value)
}
