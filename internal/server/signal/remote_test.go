package signal

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// startRemoteHarness builds a signaling server whose only non-browser admission
// is the injected predicate, mirroring what app.New composes.
func startRemoteHarness(t *testing.T, allow func(*http.Request) bool) string {
	t.Helper()
	store := newStore(t, 8, nil)
	publicBaseURL, err := url.Parse("https://share.example.test")
	if err != nil {
		t.Fatal(err)
	}
	server, err := New(Options{
		Store:                     store,
		EndpointMediaCopyCapacity: 2,
		AllowedOrigins:            map[string]struct{}{allowedOrigin: {}},
		AllowAPIClient:            allow,
		PublicBaseURL:             publicBaseURL,
		AuthenticationTimeoutMs:   500,
		ViewerDisconnectGraceMs:   50,
		HeartbeatIntervalMs:       60_000,
		Logger:                    slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	if err != nil {
		t.Fatalf("signal.New: %v", err)
	}
	httpServer := httptest.NewServer(server)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Close(ctx); err != nil {
			t.Errorf("server.Close: %v", err)
		}
		httpServer.Close()
		if err := store.Close(); err != nil {
			t.Errorf("store.Close: %v", err)
		}
	})
	return "ws" + strings.TrimPrefix(httpServer.URL, "http") + "/signal"
}

// dialSignal upgrades with exactly the headers given, so a test controls whether
// an Origin is sent at all.
func dialSignal(t *testing.T, wsURL string, header http.Header) (*websocket.Conn, *http.Response, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return websocket.Dial(ctx, wsURL, &websocket.DialOptions{HTTPHeader: header})
}

func rejectedSignalUpgrade(t *testing.T, wsURL string, header http.Header) int {
	t.Helper()
	conn, response, err := dialSignal(t, wsURL, header)
	if err == nil {
		_ = conn.CloseNow()
		t.Fatal("WebSocket upgrade unexpectedly succeeded")
	}
	if response == nil {
		t.Fatalf("upgrade failed without an HTTP response: %v", err)
	}
	return response.StatusCode
}

func acceptedSignalUpgrade(t *testing.T, wsURL string, header http.Header) {
	t.Helper()
	conn, response, err := dialSignal(t, wsURL, header)
	if err != nil {
		t.Fatalf("WebSocket upgrade: %v (status %v)", err, response)
	}
	_ = conn.Close(websocket.StatusNormalClosure, "")
}

func TestUpgradeWithoutOriginIsRejectedWithoutAnAdmittedAPIClient(t *testing.T) {
	for name, allow := range map[string]func(*http.Request) bool{
		"no predicate": nil,
		"refused":      func(*http.Request) bool { return false },
	} {
		t.Run(name, func(t *testing.T) {
			wsURL := startRemoteHarness(t, allow)
			if status := rejectedSignalUpgrade(t, wsURL, http.Header{}); status != http.StatusForbidden {
				t.Fatalf("status = %d, want %d", status, http.StatusForbidden)
			}
		})
	}
}

func TestUpgradeWithoutOriginIsAdmittedForAnAPIClient(t *testing.T) {
	wsURL := startRemoteHarness(t, func(request *http.Request) bool {
		return request.Header.Get("X-Piik-Api") == "1"
	})
	acceptedSignalUpgrade(t, wsURL, http.Header{"X-Piik-Api": {"1"}})
	if status := rejectedSignalUpgrade(t, wsURL, http.Header{}); status != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", status, http.StatusForbidden)
	}
}

func TestUpgradeWithAForeignOriginIsRejectedEvenWhenThePredicateAdmits(t *testing.T) {
	wsURL := startRemoteHarness(t, func(*http.Request) bool { return true })
	for _, origin := range []string{"https://foreign.test", allowedOrigin + "/path", "not-an-origin"} {
		if status := rejectedSignalUpgrade(t, wsURL, http.Header{"Origin": {origin}}); status != http.StatusForbidden {
			t.Fatalf("origin %q: status = %d, want %d", origin, status, http.StatusForbidden)
		}
	}
	// The same predicate admits the request once the Origin is gone.
	acceptedSignalUpgrade(t, wsURL, http.Header{})
}

func TestUpgradeWithAnAllowedOriginKeepsTheBrowserRule(t *testing.T) {
	wsURL := startRemoteHarness(t, func(*http.Request) bool { return false })
	acceptedSignalUpgrade(t, wsURL, http.Header{"Origin": {allowedOrigin}})
}
