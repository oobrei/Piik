package launcher

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"slices"
	"time"

	appconfig "github.com/TNTcraftHIM/Piik/internal/app/config"
	"github.com/TNTcraftHIM/Piik/internal/app/remoteclient"
)

// remoteRequestTimeout covers one probe or connect, which may cross the public
// network twice; remoteProxyPrefix carries every proxied call and the bridge.
const (
	remoteRequestTimeout = 20 * time.Second
	remoteProxyPrefix    = remoteclient.ProxyPrefix + "/"
)

func (server *Server) handleRemoteProbe(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var probe struct {
		Target              string `json:"target"`
		AcceptedFingerprint string `json:"acceptedFingerprint,omitempty"`
	}
	if !decodeRemoteRequest(response, request, &probe) {
		return
	}
	if _, err := remoteclient.ParseDescriptor(probe.Target); err != nil {
		writeRemoteFailure(response, http.StatusBadRequest, "invalid remote address", err)
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), remoteRequestTimeout)
	defer cancel()
	result, err := server.remote.Probe(ctx, probe.Target, probe.AcceptedFingerprint)
	if err != nil {
		writeRemoteFailure(response, http.StatusBadGateway, "remote host could not be reached", err)
		return
	}
	writeJSON(response, http.StatusOK, result)
}

func (server *Server) handleRemoteConnect(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var connect remoteclient.ConnectRequest
	if !decodeRemoteRequest(response, request, &connect) {
		return
	}
	if _, err := remoteclient.ParseDescriptor(connect.Target); err != nil {
		writeRemoteFailure(response, http.StatusBadRequest, "invalid remote address", err)
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), remoteRequestTimeout)
	defer cancel()
	status, err := server.remote.Connect(ctx, connect)
	if errors.Is(err, remoteclient.ErrUnauthorized) {
		writeRemoteFailure(response, http.StatusUnauthorized, "the remote host rejected the site password", err)
		return
	}
	if err != nil {
		writeRemoteFailure(response, http.StatusBadGateway, "the remote host could not be joined", err)
		return
	}
	writeJSON(response, http.StatusOK, status)
}

func (server *Server) handleRemoteStatus(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		response.Header().Set("Allow", http.MethodGet)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(response, http.StatusOK, server.remote.Status())
}

func (server *Server) handleRemoteDisconnect(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	server.remote.Disconnect()
	writeJSON(response, http.StatusOK, server.remote.Status())
}

func (server *Server) handleRemoteHost(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var change struct {
		Origin      string `json:"origin"`
		Fingerprint string `json:"fingerprint,omitempty"`
		Allowed     *bool  `json:"allowed"`
	}
	if !decodeRemoteRequest(response, request, &change) {
		return
	}
	host, err := appconfig.NormalizeRemoteHost(appconfig.RemoteHost{Origin: change.Origin, Fingerprint: change.Fingerprint})
	if err != nil || change.Allowed == nil {
		writeRemoteFailure(response, http.StatusBadRequest, "invalid remote host", err)
		return
	}
	server.mu.Lock()
	defer server.mu.Unlock()
	result := func() map[string]any {
		return map[string]any{"remoteHosts": slices.Clone(server.remoteHosts)}
	}
	if server.changeRemoteHost == nil {
		writeRemoteFailure(response, http.StatusServiceUnavailable, "remote host approval is unavailable", nil)
		return
	}
	current, err := server.changeRemoteHost(host, *change.Allowed)
	// The page always shows the list the App actually holds, even when the
	// save failed and the previous list is still current.
	server.remoteHosts = current
	if err != nil {
		failure := result()
		failure["detail"] = errorDetail(err)
		writeJSON(response, http.StatusServiceUnavailable, failure)
		return
	}
	writeJSON(response, http.StatusOK, result())
}

func (server *Server) handleRemoteProxy(response http.ResponseWriter, request *http.Request) {
	if request.URL.Path == remoteProxyPrefix+"signal" {
		server.remote.BridgeSignal(response, request)
		return
	}
	server.remote.ForwardAPI(response, request)
}

func decodeRemoteRequest(response http.ResponseWriter, request *http.Request, target any) bool {
	request.Body = http.MaxBytesReader(response, request.Body, maxRequestBytes)
	decoder := json.NewDecoder(request.Body)
	decoder.DisallowUnknownFields()
	if decoder.Decode(target) != nil || decoder.Decode(&struct{}{}) != io.EOF {
		http.Error(response, "invalid remote request", http.StatusBadRequest)
		return false
	}
	return true
}

func writeRemoteFailure(response http.ResponseWriter, status int, message string, err error) {
	writeJSON(response, status, map[string]string{"error": message, "detail": errorDetail(err)})
}
