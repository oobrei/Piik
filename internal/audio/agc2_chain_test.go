package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestAGC2ChainMatchesWebRTC runs the whole Go chain over the clip and compares
// the processed samples with WebRTC's AGC2 output.
//
// The reference was driven with a fixed speech probability of zero, which keeps
// the speech level estimator from ever becoming confident. That pins the gain
// path - level estimation, headroom, the adaptive gain law and the limiter - to
// WebRTC's, but it does not exercise the VAD branch. TestAGC2UsesVAD covers
// that separately.
func TestAGC2ChainMatchesWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	payload, err := os.ReadFile(filepath.Join(dir, "chain_out.f32"))
	if err != nil {
		t.Skipf("no reference output: %v", err)
	}
	reference := make([]float32, len(payload)/4)
	for index := range reference {
		reference[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
	}
	samples := readWAV24kSamples(t, filepath.Join(dir, "poem_24k.wav"))

	chain := NewAGC2(DefaultAGC2Config())
	out, err := chain.Process(samples, nil)
	if err != nil {
		t.Fatalf("process: %v", err)
	}
	if len(out) > len(reference) {
		out = out[:len(reference)]
	}
	worst, worstIndex := 0.0, 0
	for index := range out {
		if diff := math.Abs(float64(out[index] - reference[index])); diff > worst {
			worst, worstIndex = diff, index
		}
	}
	t.Logf("chain vs WebRTC: %d samples, max error %.6f at index %d", len(out), worst, worstIndex)
	if worst > 1 {
		t.Fatalf("chain output diverges by %.4f at index %d", worst, worstIndex)
	}
}

// TestAGC2UsesVAD checks that the chain's speech probability comes from the RNN
// VAD rather than a constant, and that the decimator feeds it 10 ms frames.
func TestAGC2UsesVAD(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	samples := readWAV24kSamples(t, filepath.Join(dir, "poem_24k.wav"))
	chain := NewAGC2(DefaultAGC2Config())
	frame := make([]float32, 480)
	speech, total := 0, 0
	low, high := 1.0, 0.0
	for start := 0; start+480 <= len(samples); start += 480 {
		copy(frame, samples[start:start+480])
		probability := chain.speechProbability(frame)
		if probability < low {
			low = probability
		}
		if probability > high {
			high = probability
		}
		if probability > 0.5 {
			speech++
		}
		total++
	}
	t.Logf("VAD over %d frames: %.1f%% speech, probability range [%.3f, %.3f]",
		total, 100*float64(speech)/float64(total), low, high)
	if high-low < 0.2 {
		t.Fatalf("speech probability is nearly constant (%.3f..%.3f), the VAD is not driving it", low, high)
	}
	if speech == 0 || speech == total {
		t.Fatalf("VAD classified every frame the same way (%d/%d speech)", speech, total)
	}
}
