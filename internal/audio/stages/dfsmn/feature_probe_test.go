package dfsmn

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDumpStageFeatures writes the feature rows the stage feeds the model, so
// they can be compared against the reference front-end for the same clip.
func TestDumpStageFeatures(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES to the reference fixture directory")
	}
	samples, err := readWAV(filepath.Join(dir, "rain_source.wav"))
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	input := make([]float32, len(samples))
	for index, value := range samples {
		input[index] = float32(value * sixteenBitScale)
	}
	stage, err := NewStage(loadModel(t))
	if err != nil {
		t.Fatalf("stage: %v", err)
	}
	rows := stage.FeatureRows(input, 200)
	if len(rows) != 200 {
		t.Fatalf("got %d rows, want 200", len(rows))
	}
	flat := make([]float32, 0, len(rows)*Features)
	for _, row := range rows {
		flat = append(flat, row...)
	}
	path := filepath.Join(dir, "go_stage_features.f32")
	if err := writeFloats(path, flat); err != nil {
		t.Fatalf("write: %v", err)
	}
	t.Logf("wrote %d rows to %s", len(rows), path)
}
