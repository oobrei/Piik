package remoteclient

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/TNTcraftHIM/Piik/internal/server/protocol"
	"github.com/coder/websocket"
)

// ProxyPrefix is the launcher route that carries remote API calls.
const ProxyPrefix = "/api/client-remote/proxy"

// maxProxyBytes bounds both directions of one proxied exchange.
const maxProxyBytes = 1 << 20

// proxyRoute is one allowlisted remote API call. hostToken forwards the
// browser's Authorization header, which the room replacement and access
// routes authenticate with.
type proxyRoute struct {
	path      *regexp.Regexp
	method    string
	hostToken bool
}

var proxyRoutes = [...]proxyRoute{
	{regexp.MustCompile(`^/api/capabilities$`), http.MethodGet, false},
	{regexp.MustCompile(`^/api/remote/info$`), http.MethodGet, false},
	{regexp.MustCompile(`^/api/rooms$`), http.MethodPost, false},
	{regexp.MustCompile(`^/api/rooms/[1-9][0-9]{3}/replacement$`), http.MethodPost, true},
	{regexp.MustCompile(`^/api/rooms/[1-9][0-9]{3}/access$`), http.MethodPost, true},
}

// ForwardAPI relays one allowlisted remote API call. The browser's Origin and
// Cookie never travel upstream; the session token does.
func (m *Manager) ForwardAPI(response http.ResponseWriter, request *http.Request) {
	active := m.current()
	if active == nil {
		writeJSONError(response, http.StatusServiceUnavailable, "no remote session")
		return
	}
	path := strings.TrimPrefix(request.URL.Path, ProxyPrefix)
	route, ok := matchProxyRoute(path)
	if !ok {
		writeJSONError(response, http.StatusNotFound, "not found")
		return
	}
	if request.Method != route.method {
		response.Header().Set("Allow", route.method)
		writeJSONError(response, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var body io.Reader
	if route.method == http.MethodPost {
		payload, err := io.ReadAll(io.LimitReader(request.Body, maxProxyBytes+1))
		if err != nil || len(payload) > maxProxyBytes {
			writeJSONError(response, http.StatusRequestEntityTooLarge, "request body is too large")
			return
		}
		body = bytes.NewReader(payload)
	}
	ctx, cancel := context.WithTimeout(request.Context(), requestTimeout)
	defer cancel()
	upstream, err := http.NewRequestWithContext(ctx, route.method, active.origin+path, body)
	if err != nil {
		writeJSONError(response, http.StatusBadGateway, "remote request is invalid")
		return
	}
	upstream.Header.Set("Accept", "application/json")
	upstream.Header.Set("X-Piik-Api", "1")
	if active.token != "" {
		upstream.Header.Set("X-Piik-Api-Token", active.token)
	}
	if route.method == http.MethodPost {
		upstream.Header.Set("Content-Type", "application/json")
	}
	if route.hostToken {
		if hostToken := request.Header.Get("Authorization"); hostToken != "" {
			upstream.Header.Set("Authorization", hostToken)
		}
	}
	result, err := active.client.Do(upstream)
	if err != nil {
		writeJSONError(response, http.StatusBadGateway, "remote host is unavailable")
		return
	}
	defer result.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(result.Body, maxProxyBytes+1))
	if err != nil || len(payload) > maxProxyBytes {
		writeJSONError(response, http.StatusBadGateway, "remote response is too large")
		return
	}
	response.Header().Set("Cache-Control", "no-store")
	if contentType := result.Header.Get("Content-Type"); contentType != "" {
		response.Header().Set("Content-Type", contentType)
	}
	response.WriteHeader(result.StatusCode)
	_, _ = response.Write(payload)
}

// BridgeSignal relays one WebView signaling socket to the remote host. The
// launcher's requestAllowed already proved the local Origin, so the local
// upgrade skips the library's own origin check.
func (m *Manager) BridgeSignal(response http.ResponseWriter, request *http.Request) {
	active := m.current()
	if active == nil {
		writeJSONError(response, http.StatusServiceUnavailable, "no remote session")
		return
	}
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		writeJSONError(response, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	local, err := websocket.Accept(response, request, &websocket.AcceptOptions{
		InsecureSkipVerify: true,
		CompressionMode:    websocket.CompressionDisabled,
	})
	if err != nil {
		return
	}
	defer local.CloseNow()
	local.SetReadLimit(protocol.MaxSignalBytes)

	ctx, cancel := context.WithCancel(active.ctx)
	defer cancel()
	upstream, _, err := websocket.Dial(ctx, signalURL(active.origin), &websocket.DialOptions{
		HTTPClient:      active.client,
		HTTPHeader:      signalHeaders(active.token),
		CompressionMode: websocket.CompressionDisabled,
	})
	if err != nil {
		_ = local.Close(websocket.StatusTryAgainLater, "remote host is unavailable")
		return
	}
	defer upstream.CloseNow()
	upstream.SetReadLimit(protocol.MaxSignalBytes)
	if !m.stillActive(active) {
		_ = local.Close(websocket.StatusNormalClosure, "remote session ended")
		return
	}
	relayed := make(chan error, 2)
	go func() { relayed <- relay(ctx, local, upstream) }()
	go func() { relayed <- relay(ctx, upstream, local) }()
	select {
	case <-ctx.Done():
	case <-relayed:
	}
}

// relay copies frames one way until either side closes. Each connection has
// one reader and one writer, which is the shape the library supports.
func relay(ctx context.Context, from, to *websocket.Conn) error {
	for {
		messageType, payload, err := from.Read(ctx)
		if err != nil {
			return err
		}
		if err := to.Write(ctx, messageType, payload); err != nil {
			return err
		}
	}
}

func signalURL(origin string) string {
	if strings.HasPrefix(origin, "https://") {
		return "wss://" + strings.TrimPrefix(origin, "https://") + "/signal"
	}
	return "ws://" + strings.TrimPrefix(origin, "http://") + "/signal"
}

func signalHeaders(token string) http.Header {
	header := http.Header{"X-Piik-Api": {"1"}}
	if token != "" {
		header.Set("X-Piik-Api-Token", token)
	}
	return header
}

func matchProxyRoute(path string) (proxyRoute, bool) {
	for _, route := range proxyRoutes {
		if route.path.MatchString(path) {
			return route, true
		}
	}
	return proxyRoute{}, false
}

func writeJSONError(response http.ResponseWriter, status int, message string) {
	payload, err := json.Marshal(map[string]string{"error": message})
	if err != nil {
		http.Error(response, "response unavailable", http.StatusInternalServerError)
		return
	}
	response.Header().Set("Cache-Control", "no-store")
	response.Header().Set("Content-Type", "application/json; charset=utf-8")
	response.WriteHeader(status)
	_, _ = response.Write(payload)
}
