package dfsmn

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/audio"
)

// TestSingleFrameMatchesReference takes one frame, applies the model's mask and
// compares the synthesised samples with a direct reconstruction of the same
// frame. Whole-clip statistics hide which stage loses the attenuation; one
// frame cannot.
func TestSingleFrameMatchesReference(t *testing.T) {
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
	analysis, err := audio.NewAnalyzer(audio.STFTConfig{})
	if err != nil {
		t.Fatalf("analyzer: %v", err)
	}
	size := analysis.Size()
	bins := make([]float32, 2*analysis.Bins())
	mask := readFloats(t, filepath.Join(dir, "rain_mask_full.f32"), 1026*Bins)

	for _, frame := range []int{0, 83} {
		start := frame * analysis.Hop()
		analysis.Analyze(input[start:start+size], bins)
		var power float64
		for index := 0; index < len(bins); index += 2 {
			power += float64(bins[index])*float64(bins[index]) + float64(bins[index+1])*float64(bins[index+1])
		}
		var masked float64
		for bin := 0; bin < analysis.Bins(); bin++ {
			re := float64(bins[2*bin]) * float64(mask[frame*Bins+bin])
			im := float64(bins[2*bin+1]) * float64(mask[frame*Bins+bin])
			masked += re*re + im*im
		}
		synth, err := audio.NewSynthesizer(audio.STFTConfig{})
		if err != nil {
			t.Fatalf("synthesizer: %v", err)
		}
		for bin := 0; bin < analysis.Bins(); bin++ {
			bins[2*bin] *= mask[frame*Bins+bin]
			bins[2*bin+1] *= mask[frame*Bins+bin]
		}
		out := synth.Synthesize(bins, nil)
		var outPower, inPower float64
		for _, value := range out {
			outPower += float64(value) * float64(value)
		}
		for _, value := range input[start : start+len(out)] {
			inPower += float64(value) * float64(value)
		}
		spectrum := math.Sqrt(masked / power)
		synthesised := math.Sqrt(outPower / inPower)
		t.Logf("frame %d: spectrum ratio %.4f, synthesised ratio %.4f", frame, spectrum, synthesised)
		if math.Abs(synthesised-spectrum) > 0.05 {
			t.Fatalf("frame %d: synthesis lost the mask's attenuation (%.4f vs %.4f)",
				frame, synthesised, spectrum)
		}
	}
}

var _ = fmt.Sprintf
