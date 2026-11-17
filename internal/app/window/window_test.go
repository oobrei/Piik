package window

import (
	"slices"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/app/launcher"
	"github.com/egoist/mygo"
)

func TestBindRemoteService(t *testing.T) {
	// mygo.BindAs validates every exported method and JSON-serializable type
	// via mygo's TypeScript generator checks.
	mygo.BindAs("Remote", launcher.NewRemoteService())
}

func TestBindLocaleService(t *testing.T) {
	mygo.BindAs("Locale", LocaleService{})
}

func TestSetSupportedLanguages(t *testing.T) {
	t.Cleanup(func() { mygo.App.SetLanguages() })
	mygo.App.SetLanguages(supportedLanguages...)
	want := []string{"en", "zh-Hans", "zh-Hant", "ja", "ko"}
	if got := mygo.App.Languages(); !slices.Equal(got, want) {
		t.Fatalf("mygo.App.Languages() = %v, want %v", got, want)
	}
}

func TestMatchSupportedLanguage(t *testing.T) {
	cases := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{input: "", want: ""},
		{input: "en", want: "en"},
		{input: "zh-CN", want: "zh-Hans"},
		{input: "zh-Hans", want: "zh-Hans"},
		{input: "zh-Hant", want: "zh-Hant"},
		{input: "zh-TW", want: "zh-Hant"},
		{input: "ja", want: "ja"},
		{input: "ko", want: "ko"},
		{input: "fr", wantErr: true},
		{input: "not a tag!", wantErr: true},
	}
	for _, tc := range cases {
		got, err := matchSupportedLanguage(tc.input)
		if (err != nil) != tc.wantErr {
			t.Fatalf("matchSupportedLanguage(%q) err = %v, wantErr=%v", tc.input, err, tc.wantErr)
		}
		if got != tc.want {
			t.Fatalf("matchSupportedLanguage(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}
