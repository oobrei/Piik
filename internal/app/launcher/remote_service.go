package launcher

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"

	appconfig "github.com/TNTcraftHIM/Piik/internal/app/config"
	"github.com/TNTcraftHIM/Piik/internal/app/remoteclient"
)

// RemoteProbeRequest names one remote host to inspect over mygo IPC.
type RemoteProbeRequest struct {
	Target              string `json:"target"`
	AcceptedFingerprint string `json:"acceptedFingerprint,omitempty"`
}

// RemoteHostSaveRequest approves or revokes one saved remote host over mygo IPC.
type RemoteHostSaveRequest struct {
	Origin      string `json:"origin"`
	Fingerprint string `json:"fingerprint,omitempty"`
	Allowed     bool   `json:"allowed"`
}

// RemoteService exposes the launcher's remote host operations through mygo.Bind
// while keeping the launcher Server as the single owner of remote state.
type RemoteService struct {
	mu     sync.RWMutex
	server *Server
}

// NewRemoteService creates an unbound RemoteService ready for mygo.BindAs.
func NewRemoteService() *RemoteService { return &RemoteService{} }

// Attach binds the running launcher Server to this IPC service.
func (s *RemoteService) Attach(server *Server) {
	s.mu.Lock()
	s.server = server
	s.mu.Unlock()
}

// Detach clears the launcher Server reference once it stops.
func (s *RemoteService) Detach() {
	s.mu.Lock()
	s.server = nil
	s.mu.Unlock()
}

func (s *RemoteService) currentServer() (*Server, error) {
	s.mu.RLock()
	server := s.server
	s.mu.RUnlock()
	if server == nil {
		return nil, ipcStatusError(http.StatusServiceUnavailable, "remote host service is unavailable", nil)
	}
	return server, nil
}

// Probe inspects a remote host descriptor over mygo IPC.
func (s *RemoteService) Probe(ctx context.Context, request RemoteProbeRequest) (remoteclient.ProbeResult, error) {
	server, err := s.currentServer()
	if err != nil {
		return remoteclient.ProbeResult{}, err
	}
	result, err := server.ProbeRemote(ctx, request.Target, request.AcceptedFingerprint)
	if err != nil {
		return remoteclient.ProbeResult{}, formatIPCRemoteError(err)
	}
	return result, nil
}

// Connect opens the single active remote session over mygo IPC.
func (s *RemoteService) Connect(ctx context.Context, request remoteclient.ConnectRequest) (remoteclient.SessionStatus, error) {
	server, err := s.currentServer()
	if err != nil {
		return remoteclient.SessionStatus{}, err
	}
	status, err := server.ConnectRemote(ctx, request)
	if err != nil {
		return remoteclient.SessionStatus{}, formatIPCRemoteError(err)
	}
	return status, nil
}

// Status reports the current remote session status over mygo IPC.
func (s *RemoteService) Status() (remoteclient.SessionStatus, error) {
	server, err := s.currentServer()
	if err != nil {
		return remoteclient.SessionStatus{}, err
	}
	return server.RemoteStatus(), nil
}

// Disconnect ends any active remote session over mygo IPC.
func (s *RemoteService) Disconnect() (remoteclient.SessionStatus, error) {
	server, err := s.currentServer()
	if err != nil {
		return remoteclient.SessionStatus{}, err
	}
	return server.DisconnectRemote(), nil
}

// SavedHosts returns the non-nil saved remote host list over mygo IPC.
func (s *RemoteService) SavedHosts() ([]appconfig.RemoteHost, error) {
	server, err := s.currentServer()
	if err != nil {
		return nil, err
	}
	return server.SavedRemoteHosts(), nil
}

// SetSavedHost approves or revokes a saved remote host over mygo IPC.
func (s *RemoteService) SetSavedHost(request RemoteHostSaveRequest) ([]appconfig.RemoteHost, error) {
	server, err := s.currentServer()
	if err != nil {
		return nil, err
	}
	hosts, err := server.SetRemoteHostSaved(request.Origin, request.Fingerprint, request.Allowed)
	if err != nil {
		return nil, formatIPCRemoteError(err)
	}
	return hosts, nil
}

func formatIPCRemoteError(err error) error {
	var opErr *remoteOpError
	if errors.As(err, &opErr) {
		return ipcStatusError(opErr.status, opErr.message, opErr.Unwrap())
	}
	return ipcStatusError(http.StatusInternalServerError, "remote operation failed", err)
}

func ipcStatusError(status int, message string, cause error) error {
	if cause == nil {
		return fmt.Errorf("[status=%d] %s", status, message)
	}
	return fmt.Errorf("[status=%d] %s: %s", status, message, errorDetail(cause))
}
