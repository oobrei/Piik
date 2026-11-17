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

type remoteOpError struct {
	status  int
	message string
	err     error
}

func (e *remoteOpError) Error() string {
	if e.err == nil {
		return e.message
	}
	return e.message + ": " + errorDetail(e.err)
}

func (e *remoteOpError) Unwrap() error { return e.err }

// ProbeRemote parses target and inspects the remote host's certificate and info.
func (server *Server) ProbeRemote(ctx context.Context, target, acceptedFingerprint string) (remoteclient.ProbeResult, error) {
	if _, err := remoteclient.ParseDescriptor(target); err != nil {
		return remoteclient.ProbeResult{}, &remoteOpError{status: http.StatusBadRequest, message: "invalid remote address", err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, remoteRequestTimeout)
	defer cancel()
	result, err := server.remote.Probe(ctx, target, acceptedFingerprint)
	if err != nil {
		return remoteclient.ProbeResult{}, &remoteOpError{status: http.StatusBadGateway, message: "remote host could not be reached", err: err}
	}
	return result, nil
}

// ConnectRemote establishes the single active remote session with the host.
func (server *Server) ConnectRemote(ctx context.Context, connect remoteclient.ConnectRequest) (remoteclient.SessionStatus, error) {
	if _, err := remoteclient.ParseDescriptor(connect.Target); err != nil {
		return remoteclient.SessionStatus{}, &remoteOpError{status: http.StatusBadRequest, message: "invalid remote address", err: err}
	}
	ctx, cancel := context.WithTimeout(ctx, remoteRequestTimeout)
	defer cancel()
	status, err := server.remote.Connect(ctx, connect)
	if errors.Is(err, remoteclient.ErrUnauthorized) {
		return remoteclient.SessionStatus{}, &remoteOpError{status: http.StatusUnauthorized, message: "the remote host rejected the site password", err: err}
	}
	if err != nil {
		return remoteclient.SessionStatus{}, &remoteOpError{status: http.StatusBadGateway, message: "the remote host could not be joined", err: err}
	}
	return status, nil
}

// RemoteStatus returns the current remote session status without secrets.
func (server *Server) RemoteStatus() remoteclient.SessionStatus {
	return server.remote.Status()
}

// DisconnectRemote closes any active remote session and returns the idle status.
func (server *Server) DisconnectRemote() remoteclient.SessionStatus {
	server.remote.Disconnect()
	return server.remote.Status()
}

// SavedRemoteHosts returns a non-nil copy of the saved remote host list.
func (server *Server) SavedRemoteHosts() []appconfig.RemoteHost {
	server.mu.Lock()
	defer server.mu.Unlock()
	return cloneRemoteHosts(server.remoteHosts)
}

// SetRemoteHostSaved approves or revokes a saved remote host and returns the
// complete persisted list even when saving fails.
func (server *Server) SetRemoteHostSaved(origin, fingerprint string, allowed bool) ([]appconfig.RemoteHost, error) {
	host, err := appconfig.NormalizeRemoteHost(appconfig.RemoteHost{Origin: origin, Fingerprint: fingerprint})
	if err != nil {
		return nil, &remoteOpError{status: http.StatusBadRequest, message: "invalid remote host", err: err}
	}
	server.mu.Lock()
	defer server.mu.Unlock()
	if server.changeRemoteHost == nil {
		return nil, &remoteOpError{status: http.StatusServiceUnavailable, message: "remote host approval is unavailable"}
	}
	current, saveErr := server.changeRemoteHost(host, allowed)
	// The page always shows the list the App actually holds, even when the
	// save failed and the previous list is still current.
	server.remoteHosts = current
	snapshot := cloneRemoteHosts(server.remoteHosts)
	if saveErr != nil {
		return snapshot, &remoteOpError{status: http.StatusServiceUnavailable, message: "remote host could not be saved", err: saveErr}
	}
	return snapshot, nil
}

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
	result, err := server.ProbeRemote(request.Context(), probe.Target, probe.AcceptedFingerprint)
	if err != nil {
		writeRemoteOpFailure(response, err)
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
	status, err := server.ConnectRemote(request.Context(), connect)
	if err != nil {
		writeRemoteOpFailure(response, err)
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
	writeJSON(response, http.StatusOK, server.RemoteStatus())
}

func (server *Server) handleRemoteDisconnect(response http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		response.Header().Set("Allow", http.MethodPost)
		http.Error(response, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(response, http.StatusOK, server.DisconnectRemote())
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
	if change.Allowed == nil {
		writeRemoteFailure(response, http.StatusBadRequest, "invalid remote host", nil)
		return
	}
	hosts, err := server.SetRemoteHostSaved(change.Origin, change.Fingerprint, *change.Allowed)
	if err != nil {
		var opErr *remoteOpError
		if errors.As(err, &opErr) && opErr.status == http.StatusServiceUnavailable && hosts != nil {
			writeJSON(response, http.StatusServiceUnavailable, map[string]any{
				"remoteHosts": hosts,
				"detail":      errorDetail(opErr.Unwrap()),
			})
			return
		}
		writeRemoteOpFailure(response, err)
		return
	}
	writeJSON(response, http.StatusOK, map[string]any{"remoteHosts": hosts})
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

func cloneSites(sites []string) []string {
	if len(sites) == 0 {
		return []string{}
	}
	return slices.Clone(sites)
}

func cloneRemoteHosts(hosts []appconfig.RemoteHost) []appconfig.RemoteHost {
	if len(hosts) == 0 {
		return []appconfig.RemoteHost{}
	}
	return slices.Clone(hosts)
}

func writeRemoteOpFailure(response http.ResponseWriter, err error) {
	var opErr *remoteOpError
	if errors.As(err, &opErr) {
		writeRemoteFailure(response, opErr.status, opErr.message, opErr.Unwrap())
		return
	}
	writeRemoteFailure(response, http.StatusInternalServerError, "remote operation failed", err)
}

func writeRemoteFailure(response http.ResponseWriter, status int, message string, err error) {
	writeJSON(response, status, map[string]string{"error": message, "detail": errorDetail(err)})
}
