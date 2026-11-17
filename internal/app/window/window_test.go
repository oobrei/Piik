package window

import (
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/app/launcher"
	"github.com/egoist/mygo"
)

func TestBindRemoteService(t *testing.T) {
	// mygo.BindAs validates every exported method and JSON-serializable type
	// via mygo's TypeScript generator checks.
	mygo.BindAs("Remote", launcher.NewRemoteService())
}
