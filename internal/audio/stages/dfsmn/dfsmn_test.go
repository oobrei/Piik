package dfsmn

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/audio"
)

// graphPath is the model the App ships, committed once for the reader's test.
const graphPath = "../../onnx/testdata/dfsmn_ans.onnx"

// testFrames is the length of the reference prefix in testdata: the first 64
// frames of the Rain clip, small enough to commit and long enough for the nine
// causal convolutions to run on real history.
const testFrames = 64

func loadModel(t *testing.T) *Model {
	t.Helper()
	model, err := Load(graphPath)
	if err != nil {
		t.Fatalf("loading model: %v", err)
	}
	return model
}

func readFloats(t *testing.T, path string, count int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	if len(raw) != count*4 {
		t.Fatalf("%s holds %d values, want %d", path, len(raw)/4, count)
	}
	values := make([]float32, count)
	for index := range values {
		values[index] = math.Float32frombits(binary.LittleEndian.Uint32(raw[index*4:]))
	}
	return values
}

func rows(flat []float32, count, width int) [][]float32 {
	out := make([][]float32, count)
	for row := range out {
		out[row] = flat[row*width : (row+1)*width]
	}
	return out
}

// The reference masks come from the exported ONNX graph run over the same
// features in PyTorch, so agreement means the ops and the graph reader agree
// with the model the audio was trained for.
func TestForwardMatchesReference(t *testing.T) {
	features := rows(readFloats(t, "testdata/rain_fbank_64.f32", testFrames*Features), testFrames, Features)
	want := readFloats(t, "testdata/rain_mask_64.f32", testFrames*Bins)
	masks, err := loadModel(t).Forward(features)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(masks) != testFrames {
		t.Fatalf("got %d mask rows, want %d", len(masks), testFrames)
	}
	worst, worstFrame, worstBin := 0.0, 0, 0
	for frame := range masks {
		for bin, got := range masks[frame] {
			if diff := math.Abs(float64(got - want[frame*Bins+bin])); diff > worst {
				worst, worstFrame, worstBin = diff, frame, bin
			}
		}
	}
	if worst > 1e-3 {
		t.Fatalf("mask error %.3e at frame %d bin %d, want under 1e-3", worst, worstFrame, worstBin)
	}
	t.Logf("max mask error %.2e (frame %d bin %d)", worst, worstFrame, worstBin)
}

// A live stream feeds one frame at a time, so that path has to produce the same
// masks as one whole-sequence call.
func TestStreamingMatchesBatch(t *testing.T) {
	features := rows(readFloats(t, "testdata/rain_fbank_64.f32", testFrames*Features), testFrames, Features)
	batch, err := loadModel(t).Forward(features)
	if err != nil {
		t.Fatalf("batch forward: %v", err)
	}
	model := loadModel(t)
	for frame := range features {
		streamed, err := model.Forward(features[frame : frame+1])
		if err != nil {
			t.Fatalf("frame %d: %v", frame, err)
		}
		worst := 0.0
		for bin := range streamed[0] {
			worst = math.Max(worst, math.Abs(float64(streamed[0][bin]-batch[frame][bin])))
		}
		if worst > 1e-6 {
			t.Fatalf("frame %d differs from the batch run by %.3e", frame, worst)
		}
	}
}

func TestStageRegistersByName(t *testing.T) {
	found := false
	for _, name := range audio.Registered() {
		if name == Name {
			found = true
		}
	}
	if !found {
		t.Fatalf("%q is not registered, have %v", Name, audio.Registered())
	}
	if _, err := audio.Build(Name, audio.Config{}); err == nil {
		t.Fatal("building without a model directory should fail")
	}
}
