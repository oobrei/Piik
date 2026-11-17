package remoteclient

import (
	"bytes"
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	appconfig "github.com/TNTcraftHIM/Piik/internal/app/config"
	"github.com/TNTcraftHIM/Piik/internal/server/protocol"
)

// requestTimeout bounds one remote exchange, including its TLS setup;
// maxBodyBytes bounds every remote JSON document the App reads.
const (
	requestTimeout = 10 * time.Second
	maxBodyBytes   = 64 << 10
)

// TLSStatus values reported by Probe.
const (
	TLSStatusTrusted   = "trusted"
	TLSStatusUntrusted = "untrusted"
)

// ErrUnauthorized reports a host that rejected the site password.
var ErrUnauthorized = errors.New("remote host rejected the site password")

// RemoteInfo is what a host reports about itself before a session exists; it
// mirrors the /api/remote/info body and is never secret.
type RemoteInfo struct {
	Service            string                        `json:"service,omitempty"`
	Protocol           string                        `json:"protocol,omitempty"`
	SiteAccessRequired bool                          `json:"siteAccessRequired,omitempty"`
	RoomCreation       bool                          `json:"roomCreation,omitempty"`
	APIOrigin          string                        `json:"apiOrigin,omitempty"`
	CertificateSHA256  string                        `json:"certificateSha256,omitempty"`
	Capabilities       *protocol.RuntimeCapabilities `json:"capabilities,omitempty"`
}

// ProbeResult is a reachable host plus the trust decision still owed to it.
type ProbeResult struct {
	Origin            string     `json:"origin"`
	HostLabel         string     `json:"hostLabel"`
	TLSStatus         string     `json:"tlsStatus"`
	CertificateSHA256 string     `json:"certificateSha256,omitempty"`
	Info              RemoteInfo `json:"info"`
	RoomID            string     `json:"roomId,omitempty"`
	Grant             string     `json:"grant,omitempty"`
}

// SessionStatus is the App's view of the single remote session. It never
// carries the API token.
type SessionStatus struct {
	Connected          bool                          `json:"connected"`
	Origin             string                        `json:"origin,omitempty"`
	HostLabel          string                        `json:"hostLabel,omitempty"`
	CertificateSHA256  string                        `json:"certificateSha256,omitempty"`
	SiteAccessRequired bool                          `json:"siteAccessRequired"`
	RoomCreation       bool                          `json:"roomCreation"`
	Protocol           string                        `json:"protocol,omitempty"`
	Capabilities       *protocol.RuntimeCapabilities `json:"capabilities,omitempty"`
}

// ConnectRequest carries one attempt to open the remote session.
type ConnectRequest struct {
	Target              string `json:"target"`
	AcceptedFingerprint string `json:"acceptedFingerprint,omitempty"`
	SitePassword        string `json:"sitePassword,omitempty"`
}

// Manager owns at most one remote session. Every operation reads that session
// under mu, so a replacement or Disconnect retires the previous transports.
type Manager struct {
	mu         sync.Mutex
	generation uint64
	active     *session
}

// session is one committed remote session. Its context ends when the session
// is replaced or disconnected, which closes the signaling bridge.
type session struct {
	generation uint64
	origin     string
	token      string
	status     SessionStatus
	client     *http.Client
	transport  *http.Transport
	ctx        context.Context
	cancel     context.CancelFunc
}

// NewManager returns a Manager with no active session.
func NewManager() *Manager { return &Manager{} }

// Probe reports whether target answers and which certificate it presents. An
// untrusted certificate is a result, not an error: the App shows the
// fingerprint and the user decides whether to pin it.
func (m *Manager) Probe(ctx context.Context, target, acceptedFingerprint string) (ProbeResult, error) {
	descriptor, err := ParseDescriptor(target)
	if err != nil {
		return ProbeResult{}, err
	}
	fingerprint, err := effectiveFingerprint(descriptor, acceptedFingerprint)
	if err != nil {
		return ProbeResult{}, err
	}
	client, transport := newClient(fingerprint)
	defer transport.CloseIdleConnections()
	requestCtx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()

	result := ProbeResult{
		Origin:    descriptor.Origin,
		HostLabel: hostLabel(descriptor.Origin),
		RoomID:    descriptor.RoomID,
		Grant:     descriptor.Grant,
	}
	info, observed, err := fetchInfo(requestCtx, client, descriptor.Origin)
	if err != nil {
		if fingerprint != "" || !untrustedCertificate(err) {
			return ProbeResult{}, err
		}
		// Trust on first use: report the certificate the host actually
		// presents so the App can ask the user to confirm it.
		presented, inspectErr := inspectCertificate(requestCtx, descriptor.Origin)
		if inspectErr != nil {
			return ProbeResult{}, err
		}
		result.TLSStatus = TLSStatusUntrusted
		result.CertificateSHA256 = presented
		return result, nil
	}
	result.TLSStatus = TLSStatusTrusted
	result.CertificateSHA256 = observed
	result.Info = info
	return result, nil
}

// Connect opens the one remote session: it verifies the host certificate,
// exchanges the site password for an API token, and reads the capabilities.
func (m *Manager) Connect(ctx context.Context, request ConnectRequest) (SessionStatus, error) {
	descriptor, err := ParseDescriptor(request.Target)
	if err != nil {
		return SessionStatus{}, err
	}
	fingerprint, err := effectiveFingerprint(descriptor, request.AcceptedFingerprint)
	if err != nil {
		return SessionStatus{}, err
	}
	client, transport := newClient(fingerprint)
	requestCtx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	// The host description travels over the verified channel, so reading it
	// proves the certificate before any credential leaves the App: a pinned
	// transport only completes the request for the pinned certificate, and an
	// unpinned one only for a certificate the system already trusts.
	info, observed, err := fetchInfo(requestCtx, client, descriptor.Origin)
	if err != nil {
		transport.CloseIdleConnections()
		if fingerprint == "" && untrustedCertificate(err) {
			return SessionStatus{}, errors.New("remote host certificate is not trusted; probe the host to confirm its fingerprint")
		}
		return SessionStatus{}, err
	}
	status, token, err := openSession(requestCtx, client, descriptor, info, observed, request.SitePassword)
	if err != nil {
		transport.CloseIdleConnections()
		return SessionStatus{}, err
	}
	sessionCtx, sessionCancel := context.WithCancel(context.Background())
	active := &session{
		origin:    descriptor.Origin,
		token:     token,
		status:    status,
		client:    client,
		transport: transport,
		ctx:       sessionCtx,
		cancel:    sessionCancel,
	}
	m.mu.Lock()
	retired := m.active
	m.generation++
	active.generation = m.generation
	m.active = active
	m.mu.Unlock()
	if retired != nil {
		retired.cancel()
		retired.transport.CloseIdleConnections()
	}
	return status, nil
}

// Status reports the active session without its token.
func (m *Manager) Status() SessionStatus {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.active == nil {
		return SessionStatus{}
	}
	return m.active.status
}

// Disconnect retires the active session and its transports.
func (m *Manager) Disconnect() {
	m.mu.Lock()
	retired := m.active
	m.active = nil
	m.mu.Unlock()
	if retired == nil {
		return
	}
	retired.cancel()
	retired.transport.CloseIdleConnections()
}

func (m *Manager) current() *session {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.active
}

// stillActive reports whether active is still the committed session, so a
// bridge opened during a replacement closes instead of outliving it.
func (m *Manager) stillActive(active *session) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.active == active && m.generation == active.generation
}

// newTransport stays on HTTP/1.1: the signaling bridge upgrades one of its
// connections, and an HTTP/2 connection cannot carry a WebSocket handshake.
func newTransport(fingerprint string) *http.Transport {
	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: requestTimeout}).DialContext,
		TLSHandshakeTimeout:   requestTimeout,
		ResponseHeaderTimeout: requestTimeout,
		MaxIdleConnsPerHost:   2,
	}
	if fingerprint != "" {
		transport.TLSClientConfig = pinnedTLSConfig(fingerprint)
	}
	return transport
}

func newClient(fingerprint string) (*http.Client, *http.Transport) {
	transport := newTransport(fingerprint)
	return &http.Client{Transport: transport, Timeout: requestTimeout}, transport
}

// pinnedTLSConfig trusts exactly one certificate: the fingerprint the user
// confirmed. A mismatch fails the connection instead of warning about it.
func pinnedTLSConfig(fingerprint string) *tls.Config {
	return &tls.Config{
		MinVersion:         tls.VersionTLS12,
		InsecureSkipVerify: true,
		VerifyConnection: func(state tls.ConnectionState) error {
			if len(state.PeerCertificates) == 0 {
				return errors.New("remote host presented no certificate")
			}
			if actual := certificateFingerprint(state.PeerCertificates[0]); actual != fingerprint {
				return fmt.Errorf("remote host certificate %s does not match the pinned fingerprint %s", actual, fingerprint)
			}
			return nil
		},
	}
}

func certificateFingerprint(certificate *x509.Certificate) string {
	return fmt.Sprintf("%X", sha256.Sum256(certificate.Raw))
}

// inspectCertificate reads the leaf certificate without an HTTP request: trust
// on first use needs the fingerprint, not the host's answer.
func inspectCertificate(ctx context.Context, origin string) (string, error) {
	parsed, err := url.Parse(origin)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "https" {
		return "", errors.New("remote host does not use TLS")
	}
	connection, err := dialTLS(ctx, parsed, &tls.Config{
		MinVersion: tls.VersionTLS12, InsecureSkipVerify: true, ServerName: parsed.Hostname(),
	})
	if err != nil {
		return "", err
	}
	defer connection.Close()
	state := connection.ConnectionState()
	if len(state.PeerCertificates) == 0 {
		return "", errors.New("remote host presented no certificate")
	}
	return certificateFingerprint(state.PeerCertificates[0]), nil
}

func dialTLS(ctx context.Context, parsed *url.URL, configuration *tls.Config) (*tls.Conn, error) {
	port := parsed.Port()
	if port == "" {
		port = "443"
	}
	dialer := &tls.Dialer{NetDialer: &net.Dialer{Timeout: requestTimeout}, Config: configuration}
	connection, err := dialer.DialContext(ctx, "tcp", net.JoinHostPort(parsed.Hostname(), port))
	if err != nil {
		return nil, err
	}
	tlsConnection, ok := connection.(*tls.Conn)
	if !ok {
		_ = connection.Close()
		return nil, errors.New("remote host did not negotiate TLS")
	}
	return tlsConnection, nil
}

func openSession(
	ctx context.Context,
	client *http.Client,
	descriptor Descriptor,
	info RemoteInfo,
	fingerprint, password string,
) (SessionStatus, string, error) {
	payload, err := json.Marshal(map[string]string{"password": password})
	if err != nil {
		return SessionStatus{}, "", err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, descriptor.Origin+"/api/remote/session", bytes.NewReader(payload))
	if err != nil {
		return SessionStatus{}, "", err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("X-Piik-Api", "1")
	response, err := client.Do(request)
	if err != nil {
		return SessionStatus{}, "", err
	}
	defer response.Body.Close()
	if response.StatusCode == http.StatusUnauthorized {
		return SessionStatus{}, "", ErrUnauthorized
	}
	if response.StatusCode != http.StatusOK {
		return SessionStatus{}, "", fmt.Errorf("remote host returned %s", response.Status)
	}
	var opened struct {
		Required      bool   `json:"required"`
		Authenticated bool   `json:"authenticated"`
		Token         string `json:"token"`
		ExpiresAtMs   int64  `json:"expiresAtMs"`
		TTLSeconds    int    `json:"ttlSeconds"`
	}
	if err := decodeBody(response, &opened); err != nil {
		return SessionStatus{}, "", err
	}
	if !opened.Authenticated {
		return SessionStatus{}, "", errors.New("remote host did not authenticate the session")
	}
	// A host without site access returns no token, and the session still works.
	if opened.Required && opened.Token == "" {
		return SessionStatus{}, "", errors.New("remote host returned no session token")
	}
	capabilities, err := fetchCapabilities(ctx, client, descriptor.Origin, opened.Token)
	if err != nil {
		return SessionStatus{}, "", err
	}
	status := SessionStatus{
		Connected:          true,
		Origin:             descriptor.Origin,
		HostLabel:          hostLabel(descriptor.Origin),
		CertificateSHA256:  fingerprint,
		SiteAccessRequired: info.SiteAccessRequired,
		RoomCreation:       info.RoomCreation,
		Protocol:           info.Protocol,
		Capabilities:       &capabilities,
	}
	return status, opened.Token, nil
}

func fetchInfo(ctx context.Context, client *http.Client, origin string) (RemoteInfo, string, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, origin+"/api/remote/info", nil)
	if err != nil {
		return RemoteInfo{}, "", err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("X-Piik-Api", "1")
	response, err := client.Do(request)
	if err != nil {
		return RemoteInfo{}, "", err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return RemoteInfo{}, "", fmt.Errorf("remote host returned %s", response.Status)
	}
	var info RemoteInfo
	if err := decodeBody(response, &info); err != nil {
		return RemoteInfo{}, "", err
	}
	return info, peerFingerprint(response), nil
}

func fetchCapabilities(ctx context.Context, client *http.Client, origin, token string) (protocol.RuntimeCapabilities, error) {
	var capabilities protocol.RuntimeCapabilities
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, origin+"/api/capabilities", nil)
	if err != nil {
		return capabilities, err
	}
	request.Header.Set("Accept", "application/json")
	request.Header.Set("X-Piik-Api", "1")
	if token != "" {
		request.Header.Set("X-Piik-Api-Token", token)
	}
	response, err := client.Do(request)
	if err != nil {
		return capabilities, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return capabilities, fmt.Errorf("remote host returned %s", response.Status)
	}
	if err := decodeBody(response, &capabilities); err != nil {
		return capabilities, errors.New("remote host returned invalid capabilities")
	}
	return capabilities, nil
}

// decodeBody reads one bounded JSON document; an empty body leaves target
// untouched so a host may answer a probe with no description.
func decodeBody(response *http.Response, target any) error {
	payload, err := io.ReadAll(io.LimitReader(response.Body, maxBodyBytes))
	if err != nil {
		return err
	}
	if len(payload) == 0 {
		return nil
	}
	return json.Unmarshal(payload, target)
}

func peerFingerprint(response *http.Response) string {
	if response.TLS == nil || len(response.TLS.PeerCertificates) == 0 {
		return ""
	}
	return certificateFingerprint(response.TLS.PeerCertificates[0])
}

func untrustedCertificate(err error) bool {
	var unknownAuthority x509.UnknownAuthorityError
	var invalid x509.CertificateInvalidError
	var hostname x509.HostnameError
	return errors.As(err, &unknownAuthority) || errors.As(err, &invalid) || errors.As(err, &hostname)
}

// effectiveFingerprint settles the pin: an accepted fingerprint wins, and a
// different invitation fingerprint is a conflict rather than a silent choice.
func effectiveFingerprint(descriptor Descriptor, accepted string) (string, error) {
	fingerprint, err := appconfig.NormalizeFingerprint(accepted)
	if err != nil {
		return "", err
	}
	if descriptor.Fingerprint != "" && fingerprint != "" && descriptor.Fingerprint != fingerprint {
		return "", errors.New("the accepted fingerprint does not match the invitation fingerprint")
	}
	if fingerprint != "" {
		return fingerprint, nil
	}
	return descriptor.Fingerprint, nil
}

func hostLabel(origin string) string {
	parsed, err := url.Parse(origin)
	if err != nil || parsed.Hostname() == "" {
		return origin
	}
	return parsed.Hostname()
}
