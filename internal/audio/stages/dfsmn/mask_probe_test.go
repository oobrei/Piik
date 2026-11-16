package dfsmn

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestMaskMatchesFullClip runs the whole reference feature set through the
// model and compares every mask row with the reference. The committed test
// covers 64 frames; a causal model can agree there and diverge later, so the
// full clip is the only way to know the streaming state is right.
func TestMaskMatchesFullClip(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES to the reference fixture directory")
	}
	const frames = 1026
	flat := readFloats(t, filepath.Join(dir, "rain_fbank_full.f32"), frames*Features)
	want := readFloats(t, filepath.Join(dir, "rain_mask_full.f32"), frames*Bins)
	masks, err := loadModel(t).Forward(rows(flat, frames, Features))
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	worst, worstFrame, worstBin := 0.0, 0, 0
	for frame := range masks {
		for bin, got := range masks[frame] {
			if diff := math.Abs(float64(got - want[frame*Bins+bin])); diff > worst {
				worst, worstFrame, worstBin = diff, frame, bin
			}
		}
	}
	t.Logf("full clip: max mask error %.3e at frame %d bin %d", worst, worstFrame, worstBin)
	// Per-frame means are what show a mask that agrees on average but closes in
	// the wrong frames, which a global mean hides.
	worstFrameMean, worstFrameIndex := 0.0, 0
	for frame := range masks {
		var got, wantSum float64
		for bin, value := range masks[frame] {
			got += float64(value)
			wantSum += float64(want[frame*Bins+bin])
		}
		diff := math.Abs(got/float64(Bins) - wantSum/float64(Bins))
		if diff > worstFrameMean {
			worstFrameMean, worstFrameIndex = diff, frame
		}
	}
	t.Logf("worst per-frame mean difference %.3e at frame %d", worstFrameMean, worstFrameIndex)

	// Does the model actually respond to its input? Feed it features that are
	// clearly different and see whether the mask moves at all.
	shifted := make([][]float32, len(masks))
	for frame := range masks {
		shifted[frame] = make([]float32, Features)
		for bin := range shifted[frame] {
			shifted[frame][bin] = float32(flat[frame*Features+bin]) - 6
		}
	}
	other, err := loadModel(t).Forward(shifted)
	if err != nil {
		t.Fatalf("forward on shifted features: %v", err)
	}
	delta := 0.0
	for frame := range other {
		for bin := range other[frame] {
			delta = math.Max(delta, math.Abs(float64(other[frame][bin]-masks[frame][bin])))
		}
	}
	t.Logf("mask response to a 6-unit feature shift: %.3e", delta)

	// The stage feeds clean features; the reference mask came from dithered
	// ones. Compare the two masks directly, so a model that ignores its input
	// is distinguishable from one that does not.
	cleanFlat := readFloats(t, filepath.Join(dir, "rain_fbank_clean.f32"), frames*Features)
	cleanMasks, err := loadModel(t).Forward(rows(cleanFlat, frames, Features))
	if err != nil {
		t.Fatalf("forward on clean features: %v", err)
	}
	cleanDelta := 0.0
	for frame := range cleanMasks {
		for bin := range cleanMasks[frame] {
			cleanDelta = math.Max(cleanDelta, math.Abs(float64(cleanMasks[frame][bin]-masks[frame][bin])))
		}
	}
	t.Logf("mask difference between clean and reference features: %.3e", cleanDelta)
	// Report where the error first exceeds a tight bound, which is what a
	// broken causal state shows up as.
	for frame := range masks {
		worstRow := 0.0
		for bin, got := range masks[frame] {
			worstRow = math.Max(worstRow, math.Abs(float64(got-want[frame*Bins+bin])))
		}
		if worstRow > 1e-3 {
			t.Fatalf("frame %d already differs by %.3e", frame, worstRow)
		}
	}
}
