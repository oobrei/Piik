package audio

import (
	"math"
	"testing"
)

func TestCompositeTransformMatchesADirectDFT(t *testing.T) {
	const size = 24 // 8 x 3: exercise both stages at a size that is easy to check
	plan, err := newSpectral(size)
	if err != nil {
		t.Fatalf("newSpectral(%d): %v", size, err)
	}
	samples := make([]float32, size)
	for index := range samples {
		samples[index] = float32(0.4*math.Sin(2*math.Pi*3*float64(index)/size) +
			0.2*math.Cos(2*math.Pi*5*float64(index)/size))
	}
	bins := make([]float32, 2*(size/2+1))
	plan.forward(samples, bins)
	for bin := 0; bin < size/2+1; bin++ {
		var wantRe, wantIm float64
		for index, sample := range samples {
			angle := -2 * math.Pi * float64(bin*index) / size
			wantRe += float64(sample) * math.Cos(angle)
			wantIm += float64(sample) * math.Sin(angle)
		}
		if diff := math.Hypot(float64(bins[2*bin])-wantRe, float64(bins[2*bin+1])-wantIm); diff > 1e-4 {
			t.Fatalf("bin %d differs from a direct DFT by %v", bin, diff)
		}
	}
}

func TestCompositeTransformRoundTripsTheReferenceFrame(t *testing.T) {
	const size = 1920 // the frame length the denoiser masks
	plan, err := newSpectral(size)
	if err != nil {
		t.Fatalf("newSpectral(%d): %v", size, err)
	}
	samples := make([]float32, size)
	for index := range samples {
		samples[index] = float32(0.5*math.Sin(2*math.Pi*3*float64(index)/size) +
			0.25*math.Sin(2*math.Pi*440*float64(index)/SampleRate))
	}
	bins := make([]float32, 2*(size/2+1))
	plan.forward(samples, bins)
	if magnitude := math.Hypot(float64(bins[6]), float64(bins[7])); math.Abs(magnitude-size/4) > 1 {
		t.Fatalf("bin 3 magnitude = %v, want %v", magnitude, size/4)
	}
	recovered := make([]float32, size)
	plan.inverse(bins, recovered)
	for index := range samples {
		if diff := math.Abs(float64(samples[index] - recovered[index])); diff > 1e-4 {
			t.Fatalf("sample %d: diff %v", index, diff)
		}
	}
}

func TestNewSpectralRejectsASizeItCannotDecompose(t *testing.T) {
	if _, err := newSpectral(37); err == nil {
		t.Fatal("a size with no radix-2 stage and a long direct stage was accepted")
	}
}

func TestFrameStageReconstructsA1920Frame(t *testing.T) {
	cfg := STFTConfig{FFTSize: 1920, Hop: 960}
	stage, err := FrameStage("identity", cfg, func([]float32) error { return nil })
	if err != nil {
		t.Fatalf("FrameStage: %v", err)
	}
	const total = 1920 * 4
	signal := make([]float32, total)
	for index := range signal {
		signal[index] = float32(0.4 * math.Sin(2*math.Pi*440*float64(index)/SampleRate))
	}
	out, err := stage.Process(signal, nil)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	tail, err := stage.Flush(nil)
	if err != nil {
		t.Fatalf("Flush: %v", err)
	}
	out = append(out, tail...)
	if len(out) < total {
		t.Fatalf("emitted %d samples, want at least %d", len(out), total)
	}
	for index := cfg.FFTSize; index < total-cfg.FFTSize; index++ {
		if diff := math.Abs(float64(signal[index] - out[index])); diff > 2e-3 {
			t.Fatalf("sample %d: %v vs %v (diff %v)", index, signal[index], out[index], diff)
		}
	}
}
