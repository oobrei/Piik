package app

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"

	"github.com/TNTcraftHIM/Piik/internal/app/buildinfo"
	"github.com/TNTcraftHIM/Piik/internal/app/failure"
	"github.com/TNTcraftHIM/Piik/internal/app/window"
	"github.com/TNTcraftHIM/Piik/internal/diagnostics"
)

// surface is how the App presents itself to the person running it. The mygo
// window implements it for interactive runs; headless runs keep the same
// lifecycle without a window so packaging and gates can drive the App.
type surface interface {
	// Open shows an App-owned page. External addresses leave for the browser.
	Open(target string)
	// SetLanguage follows the language the page reports.
	SetLanguage(language string)
	// SetLauncher records the launcher address for the running App.
	SetLauncher(target string)
	// SetRoom records the mode, entry, invitation and certificate of a room.
	SetRoom(mode, entry, invitation, certificate string, protected bool)
	// SetDiagnostics records where a report is written and how to export one.
	SetDiagnostics(logPath string, export func() (string, error))
	// ReportFailure shows a failure the user must act on.
	ReportFailure(err error)
	// ReportExport reports the result of an export the App started itself.
	ReportExport(path string, err error)
	// Finish reports the final outcome once the App stops.
	Finish(err error, exportPath string)
	// Run owns the main goroutine until the surface stops or ctx is done.
	// A headless surface returns as soon as the App's work finishes.
	Run(ctx context.Context) error
	// Headless reports whether no window is showing.
	Headless() bool
}

// newSurface returns the window surface, or the headless one when the App runs
// for automation. cancel stops the App when the user closes the window.
func newSurface(headless bool, cancel context.CancelFunc) surface {
	if headless {
		return &headlessSurface{}
	}
	return &windowSurface{window: window.New(window.Options{Title: "Piik App"}, cancel)}
}

// windowSurface presents the App in its mygo window.
type windowSurface struct {
	window *window.Window

	mu          sync.Mutex
	language    string
	launcher    string
	room        string
	entry       string
	invitation  string
	certificate string
	protected   bool
	logPath     string
	export      func() (string, error)
	exported    string
	problem     error
	finished    bool
}

func (surface *windowSurface) Open(target string) { surface.window.Open(target) }

func (surface *windowSurface) SetLanguage(language string) {
	surface.mu.Lock()
	surface.language = language
	surface.mu.Unlock()
}

func (surface *windowSurface) SetLauncher(target string) {
	surface.mu.Lock()
	surface.launcher = target
	surface.mu.Unlock()
}

func (surface *windowSurface) SetRoom(mode, entry, invitation, certificate string, protected bool) {
	surface.mu.Lock()
	surface.room, surface.entry, surface.invitation = mode, entry, invitation
	surface.certificate, surface.protected = certificate, protected
	surface.mu.Unlock()
}

func (surface *windowSurface) SetDiagnostics(logPath string, export func() (string, error)) {
	surface.mu.Lock()
	surface.logPath, surface.export = logPath, export
	surface.mu.Unlock()
}

// ReportFailure puts the failure copy in front of the user. The window is
// already showing the launcher, which explains the same reason, so a dialog is
// only added when there is no page to read.
func (surface *windowSurface) ReportFailure(err error) {
	if err == nil {
		return
	}
	surface.mu.Lock()
	finished := surface.finished
	surface.mu.Unlock()
	if finished {
		return
	}
	surface.window.Report("Piik App", failureText(err))
}

func (surface *windowSurface) ReportExport(path string, err error) {
	if err != nil || path == "" {
		return
	}
	surface.mu.Lock()
	surface.exported = path
	surface.mu.Unlock()
	surface.window.Reveal(path)
}

func (surface *windowSurface) Finish(err error, exportPath string) {
	surface.mu.Lock()
	surface.finished = true
	logPath, exported := surface.logPath, surface.exported
	if exportPath != "" {
		exported = exportPath
	}
	surface.mu.Unlock()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Piik App:", err)
	}
	if exported != "" {
		fmt.Fprintln(os.Stderr, "Piik App: diagnostics saved to", exported)
	} else if logPath != "" {
		fmt.Fprintln(os.Stderr, "Piik App: diagnostics log at", logPath)
	}
	slog.Debug("piik-client", "event", "surface-finished", diagnostics.Error(err))
}

// Run owns the main goroutine: mygo's event loop must run on the thread that
// started the process.
func (surface *windowSurface) Run(ctx context.Context) error { return surface.window.Run(ctx) }

func (surface *windowSurface) Headless() bool { return false }

// headlessSurface keeps the App's lifecycle without a window: the packaging
// smoke and the gates read the readiness lines from stdout.
type headlessSurface struct{}

func (headlessSurface) Open(string)                                   {}
func (headlessSurface) SetLanguage(string)                            {}
func (headlessSurface) SetLauncher(string)                            {}
func (headlessSurface) SetRoom(string, string, string, string, bool)  {}
func (headlessSurface) SetDiagnostics(string, func() (string, error)) {}
func (headlessSurface) ReportFailure(error)                           {}
func (headlessSurface) ReportExport(string, error)                    {}
func (headlessSurface) Finish(error, string)                          {}
func (headlessSurface) Headless() bool                                { return true }

// Run waits for ctx: a headless App has no event loop to own.
func (headlessSurface) Run(ctx context.Context) error {
	<-ctx.Done()
	return nil
}

// failureText is the user-facing copy for a failure reason. It never carries
// the underlying cause, which can name a path or an address.
func failureText(err error) string {
	switch failure.Of(err) {
	case failure.TunnelExited:
		return "Public invite disconnected before it was ready. Try again or use an existing Piik site."
	case failure.TunnelTimeout:
		return "Public invite did not connect in time. Try again or use an existing Piik site."
	case failure.TunnelUnavailable:
		return "The public-invite helper is unavailable. Extract the complete App package again."
	case failure.TunnelStart:
		return "Could not start public invite. Check the details or use an existing Piik site."
	case failure.TunnelStopped:
		return "Public invite stopped. Reopen App for a new link or use an existing Piik site."
	case failure.LocalPort:
		return "The local port is unavailable. Close the other process or choose another port."
	case failure.LocalNetwork:
		return "No suitable local invitation address. Check your network and selected address."
	case failure.Configuration:
		return "Could not apply App settings. Check the details."
	case failure.NativeControl:
		return "App connection service is unavailable. Reopen App and try again."
	case failure.Launcher:
		return "App launcher is unavailable. Reopen App and try again."
	case failure.Diagnostics:
		return "Could not start diagnostics. Check the save location and permissions."
	case failure.LocalServer:
		return "The local room service is unavailable. Check the details."
	}
	return "Piik App could not continue. Check the details."
}

// version is the build identity the window title carries.
func version() string {
	revision := buildinfo.Revision
	if len(revision) == 40 {
		revision = revision[:7]
	}
	return buildinfo.Version() + " " + revision
}
