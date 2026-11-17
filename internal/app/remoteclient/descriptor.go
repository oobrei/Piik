// Package remoteclient connects the App to a remote Piik host: it parses
// connection descriptors, pins that host's certificate, keeps the single
// remote API session, and bridges the launcher page to it.
package remoteclient

import (
	"errors"
	"net/url"
	"regexp"
	"strings"

	appconfig "github.com/TNTcraftHIM/Piik/internal/app/config"
)

// roomIDPattern is the published room code shape.
var roomIDPattern = regexp.MustCompile(`^[1-9][0-9]{3}$`)

// roomPathPrefix is the Browser invite path; maxGrantLength bounds an
// invitation grant, which is never persisted.
const (
	roomPathPrefix = "/r/"
	maxGrantLength = 512
)

// Descriptor is one parsed remote address: where to connect and what the
// invitation already settled.
type Descriptor struct {
	Origin      string
	RoomID      string
	Grant       string
	Fingerprint string
}

// ParseDescriptor accepts the three published forms: a piik://connect
// invitation, a full HTTP(S) URL including a Browser invite, and a bare host.
func ParseDescriptor(raw string) (Descriptor, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return Descriptor{}, errors.New("remote address is empty")
	}
	if strings.HasPrefix(strings.ToLower(value), "piik://") {
		return parseInvite(value)
	}
	if !strings.Contains(value, "://") {
		value = "https://" + value
	}
	parsed, err := url.Parse(value)
	if err != nil {
		return Descriptor{}, errors.New("remote address is invalid")
	}
	origin, err := originOf(parsed)
	if err != nil {
		return Descriptor{}, err
	}
	room, err := roomFromPath(parsed.Path)
	if err != nil {
		return Descriptor{}, err
	}
	grant, err := grantFromFragment(parsed.Fragment)
	if err != nil {
		return Descriptor{}, err
	}
	return Descriptor{Origin: origin, RoomID: room, Grant: grant}, nil
}

// parseInvite reads piik://connect?origin=&room=&grant=&fp=, the form the
// desktop App hands to another App.
func parseInvite(value string) (Descriptor, error) {
	parsed, err := url.Parse(value)
	if err != nil || parsed.User != nil || !strings.EqualFold(parsed.Host, "connect") {
		return Descriptor{}, errors.New("remote invitation is invalid")
	}
	query := parsed.Query()
	origin, err := originFromValue(query.Get("origin"))
	if err != nil {
		return Descriptor{}, err
	}
	room := strings.TrimSpace(query.Get("room"))
	if room != "" && !roomIDPattern.MatchString(room) {
		return Descriptor{}, errors.New("remote invitation room must be a four-digit code")
	}
	grant, err := validateGrant(query.Get("grant"))
	if err != nil {
		return Descriptor{}, err
	}
	fingerprint, err := appconfig.NormalizeFingerprint(query.Get("fp"))
	if err != nil {
		return Descriptor{}, err
	}
	return Descriptor{Origin: origin, RoomID: room, Grant: grant, Fingerprint: fingerprint}, nil
}

func originFromValue(raw string) (string, error) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return "", errors.New("remote invitation has no origin")
	}
	if !strings.Contains(value, "://") {
		value = "https://" + value
	}
	parsed, err := url.Parse(value)
	if err != nil {
		return "", errors.New("remote invitation origin is invalid")
	}
	return originOf(parsed)
}

// originOf reduces a parsed URL to the scheme and host the session uses.
func originOf(parsed *url.URL) (string, error) {
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil {
		return "", errors.New("remote origin must be an HTTP or HTTPS URL")
	}
	if parsed.Hostname() == "" {
		return "", errors.New("remote origin must have a host")
	}
	return parsed.Scheme + "://" + strings.ToLower(parsed.Host), nil
}

func roomFromPath(path string) (string, error) {
	if !strings.HasPrefix(path, roomPathPrefix) {
		return "", nil
	}
	room := strings.TrimPrefix(path, roomPathPrefix)
	if !roomIDPattern.MatchString(room) {
		return "", errors.New("remote invitation room must be a four-digit code")
	}
	return room, nil
}

func grantFromFragment(fragment string) (string, error) {
	if fragment == "" {
		return "", nil
	}
	values, err := url.ParseQuery(fragment)
	if err != nil {
		return "", errors.New("remote invitation grant is invalid")
	}
	grant := values.Get("v")
	if grant == "" {
		grant = values.Get("grant")
	}
	return validateGrant(grant)
}

func validateGrant(grant string) (string, error) {
	if grant == "" {
		return "", nil
	}
	if len(grant) > maxGrantLength || strings.ContainsAny(grant, " \t\r\n") {
		return "", errors.New("remote invitation grant is invalid")
	}
	return grant, nil
}
