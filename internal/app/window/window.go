// Package window owns Piik App's desktop surface: one mygo window that shows
// the App's own loopback pages. The room service, the launcher and the native
// control service stay in-process; only presentation lives here.
package window

import (
	"context"
	"errors"
	"log/slog"
	"maps"
	"net/url"
	"slices"
	"strings"
	"sync"

	"github.com/TNTcraftHIM/Piik/internal/app/voice"
	"github.com/TNTcraftHIM/Piik/internal/diagnostics"
	"github.com/egoist/mygo"
)

// Options describes the App window. Zero values fall back to the defaults
// below, which keep the launcher's two-column layout readable.
type Options struct {
	Title           string
	Width, Height   int
	MinWidth        int
	MinHeight       int
	BackgroundColor string
	Services        map[string]any
}

const (
	defaultTitle  = "Piik App"
	defaultWidth  = 1180
	defaultHeight = 820
	defaultMinW   = 720
	defaultMinH   = 560
	// The page paints its own light or dark surface; this only avoids a white
	// flash before the first paint.
	defaultBackground = "light-dark(#f4f3ef, #171a20)"
	// StateKey persists the window's geometry across launches.
	stateKey = "piik-app"
)

// Window is the single App window. Open may be called from any goroutine,
// including before the mygo application is ready; Run owns the event loop and
// must be called from the main goroutine.
type Window struct {
	options Options

	mu       sync.Mutex
	win      *mygo.Window
	pending  string
	opened   bool
	closed   bool
	onClose  func()
	quitOnce sync.Once
	cancel   context.CancelFunc
}

// New returns the App window. cancel stops the App when the user closes the
// window, so the room service and launcher shut down with it.
func New(options Options, cancel context.CancelFunc) *Window {
	if options.Title == "" {
		options.Title = defaultTitle
	}
	if options.Width <= 0 {
		options.Width = defaultWidth
	}
	if options.Height <= 0 {
		options.Height = defaultHeight
	}
	if options.MinWidth <= 0 {
		options.MinWidth = defaultMinW
	}
	if options.MinHeight <= 0 {
		options.MinHeight = defaultMinH
	}
	if options.BackgroundColor == "" {
		options.BackgroundColor = defaultBackground
	}
	return &Window{options: options, cancel: cancel}
}

// Open shows target in the window, creating it on first use. An empty target
// is ignored: the launcher page is already showing. A target the App does not
// own is handed to the system browser, which keeps external Sites and
// documentation links on the platform's own surface.
func (w *Window) Open(target string) {
	target = strings.TrimSpace(target)
	if target == "" {
		return
	}
	if !Owned(target) {
		if err := mygo.Shell.OpenExternal(target); err != nil {
			slog.Debug("Could not open the system browser", diagnostics.Error(err))
		}
		return
	}
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return
	}
	if w.win == nil {
		w.pending = target
		w.mu.Unlock()
		return
	}
	win := w.win
	w.mu.Unlock()
	if err := win.Page().LoadURL(target); err != nil {
		slog.Debug("Could not load the App page", diagnostics.Error(err))
	}
	win.Show()
	win.Focus()
}

// Owned reports whether target is one of the App's own loopback pages. Those
// are the only pages the window shows; every other address belongs to the
// system browser.
func Owned(target string) bool {
	parsed, err := url.Parse(target)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return false
	}
	host := parsed.Hostname()
	return host == "127.0.0.1" || host == "localhost" || host == "::1"
}

// Run starts the mygo application and blocks on its event loop until the
// window closes or ctx is done. It must run on the main goroutine, because the
// UI toolkits of every platform must run on the thread that started the
// process.
func (w *Window) Run(ctx context.Context) error {
	mygo.App.SetName(w.options.Title)
	// The room-voice chain is native Go bound to the page: the App runs the
	// same AGC2 and DFSMN code the rest of Piik uses, with no second
	// implementation in the Browser.
	mygo.Bind(voice.NewService())
	for _, name := range slices.Sorted(maps.Keys(w.options.Services)) {
		if service := w.options.Services[name]; service != nil {
			mygo.BindAs(name, service)
		}
	}
	mygo.App.WhenReady(w.create)
	go func() {
		<-ctx.Done()
		w.Close()
	}()
	err := mygo.App.Run()
	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()
	if errors.Is(ctx.Err(), context.Canceled) {
		return nil
	}
	return err
}

// create builds the window once the application is ready and loads whatever
// page was requested before then.
func (w *Window) create() {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return
	}
	pending := w.pending
	w.pending = ""
	w.mu.Unlock()

	win := mygo.NewWindow(mygo.WindowOptions{
		Title:           w.options.Title,
		Width:           w.options.Width,
		Height:          w.options.Height,
		MinWidth:        w.options.MinWidth,
		MinHeight:       w.options.MinHeight,
		BackgroundColor: w.options.BackgroundColor,
		StateKey:        stateKey,
		URL:             pending,
	})
	page := win.Page()
	// The App's pages are its own loopback origins: they alone may use the
	// camera and microphone. Anything else is refused rather than prompted.
	page.SetPermissionHandler(func(request mygo.PermissionRequest) bool {
		return Owned(request.Origin)
	})
	// A loopback certificate is the App's own; never accept a bad one silently.
	page.SetCertificateErrorHandler(func(mygo.CertificateError) bool { return false })
	// window.open() and target=_blank keep working by leaving for the system
	// browser: the launcher's approval page, the guide and release downloads
	// are browser surfaces, not App windows.
	page.SetWindowOpenHandler(func(request mygo.WindowOpenRequest) *mygo.WindowOptions {
		if request.URL == "" {
			return nil
		}
		if err := mygo.Shell.OpenExternal(request.URL); err != nil {
			slog.Debug("Could not open the system browser", diagnostics.Error(err))
		}
		return nil
	})
	win.OnClosed(func() {
		w.mu.Lock()
		w.closed = true
		onClose := w.onClose
		w.mu.Unlock()
		if onClose != nil {
			onClose()
		}
		w.quit()
	})

	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		win.Close()
		return
	}
	w.win, w.opened = win, true
	w.mu.Unlock()
}

// Close closes the window, if any, and stops the App.
func (w *Window) Close() {
	w.mu.Lock()
	win := w.win
	w.mu.Unlock()
	if win != nil && !win.IsDestroyed() {
		win.Close()
	}
	w.quit()
}

// quit stops the App exactly once, whether the user closed the window or the
// App asked to exit.
func (w *Window) quit() {
	w.quitOnce.Do(func() {
		if w.cancel != nil {
			w.cancel()
		}
		mygo.App.Quit()
	})
}

// Report shows a failure the user must see. It never blocks the caller and
// never carries credentials: callers pass the failure copy, not raw URLs.
func (w *Window) Report(title, detail string) {
	go func() {
		mygo.Dialog.Error(title, detail)
	}()
}

// Reveal shows a saved file, such as an exported diagnostics report, in the
// platform's file manager.
func (w *Window) Reveal(path string) {
	if path == "" {
		return
	}
	mygo.Shell.ShowItemInFolder(path)
}
