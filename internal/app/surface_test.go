package app

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/TNTcraftHIM/Piik/internal/app/failure"
)

// testSurface records what the App asks its window to show, so tests can drive
// the launcher and room lifecycle without a webview.
type testSurface struct {
	mu          sync.Mutex
	opened      []string
	language    string
	launcher    string
	room        string
	entry       string
	invitation  string
	certificate string
	protected   bool
	logPath     string
	export      func() (string, error)
	failure     error
	exported    string
	finished    bool
	open        func(string)
}

func (surface *testSurface) Open(target string) {
	surface.mu.Lock()
	open := surface.open
	if open == nil {
		surface.opened = append(surface.opened, target)
	}
	surface.mu.Unlock()
	// Record before the hook runs, so a hook that blocks still leaves the page
	// visible to the test that is waiting on it.
	if open != nil {
		surface.mu.Lock()
		surface.opened = append(surface.opened, target)
		surface.mu.Unlock()
		open(target)
	}
}

func (surface *testSurface) SetLanguage(language string) {
	surface.mu.Lock()
	surface.language = language
	surface.mu.Unlock()
}

func (surface *testSurface) SetLauncher(target string) {
	surface.mu.Lock()
	surface.launcher = target
	surface.mu.Unlock()
}

func (surface *testSurface) SetRoom(mode, entry, invitation, certificate string, protected bool) {
	surface.mu.Lock()
	surface.room, surface.entry, surface.invitation = mode, entry, invitation
	surface.certificate, surface.protected = certificate, protected
	surface.mu.Unlock()
}

func (surface *testSurface) SetDiagnostics(logPath string, export func() (string, error)) {
	surface.mu.Lock()
	surface.logPath, surface.export = logPath, export
	surface.mu.Unlock()
}

func (surface *testSurface) ReportFailure(err error) {
	surface.mu.Lock()
	surface.failure = err
	surface.mu.Unlock()
}

func (surface *testSurface) ReportExport(path string, err error) {
	surface.mu.Lock()
	surface.exported = path
	surface.mu.Unlock()
}

func (surface *testSurface) Finish(err error, exportPath string) {
	surface.mu.Lock()
	surface.finished = true
	surface.mu.Unlock()
}

func (surface *testSurface) Run(ctx context.Context) error {
	<-ctx.Done()
	return nil
}

func (surface *testSurface) Headless() bool { return false }

// last reports the most recent page the App asked to show.
func (surface *testSurface) last() string {
	surface.mu.Lock()
	defer surface.mu.Unlock()
	if len(surface.opened) == 0 {
		return ""
	}
	return surface.opened[len(surface.opened)-1]
}

// waitFor polls until the App asks to show a page that satisfies match.
func (surface *testSurface) waitFor(t *testing.T, match func(string) bool) string {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for {
		surface.mu.Lock()
		for _, target := range surface.opened {
			if match(target) {
				surface.mu.Unlock()
				return target
			}
		}
		surface.mu.Unlock()
		if time.Now().After(deadline) {
			t.Fatal("App did not show the expected page")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// failingReason builds a failure the surface must present.
func failingReason(reason failure.Reason) error { return reason }
