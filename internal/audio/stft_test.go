package audio

import (
	"math"
	"testing"
)

func TestFFTRoundTripsASine(t *testing.T) {
	const size = 64
	plan := newFFTPlan(size)
	samples := make([]float32, size)
	for index := range samples {
		samples[index] = float32(math.Sin(2 * math.Pi * 3 * float64(index) / size))
	}
	bins := make([]float32, 2*(size/2+1))
	plan.forward(samples, bins)
	if magnitude := math.Hypot(float64(bins[6]), float64(bins[7])); magnitude < 20 {
		t.Fatalf("bin 3 magnitude = %v, want ~32", magnitude)
	}
	recovered := make([]float32, size)
	plan.inverse(bins, recovered)
	for index := range samples {
		if diff := math.Abs(float64(samples[index] - recovered[index])); diff > 1e-5 {
			t.Fatalf("sample %d: diff %v", index, diff)
		}
	}
}

func TestFrameStageReconstructsItsInput(t *testing.T) {
	// An identity frame transform must return the signal it was given, at
	// unity gain and without a seam at the frame boundary.
	cfg := STFTConfig{FFTSize: 256, Hop: 128}
	stage, err := FrameStage("identity", cfg, func([]float32) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	const total = 256 * 8
	signal := make([]float32, total)
	for index := range signal {
		signal[index] = float32(0.4*math.Sin(2*math.Pi*440*float64(index)/SampleRate) +
			0.2*math.Sin(2*math.Pi*97*float64(index)/SampleRate))
	}
	out, err := stage.Process(signal, nil)
	if err != nil {
		t.Fatal(err)
	}
	tail, err := stage.Flush(nil)
	if err != nil {
		t.Fatal(err)
	}
	out = append(out, tail...)
	if len(out) < total {
		t.Fatalf("emitted %d samples, want at least %d", len(out), total)
	}
	// The first frame is incomplete by construction, so compare after it.
	for index := cfg.FFTSize; index < total-cfg.FFTSize; index++ {
		if diff := math.Abs(float64(signal[index] - out[index])); diff > 2e-3 {
			t.Fatalf("sample %d: %v vs %v (diff %v)", index, signal[index], out[index], diff)
		}
	}
}

func TestPipelineRunsStagesInOrder(t *testing.T) {
	double := SampleStage("double", func(sample float32) float32 { return sample * 2 }, nil)
	offset := SampleStage("offset", func(sample float32) float32 { return sample + 1 }, nil)
	pipeline, err := New(double, offset)
	if err != nil {
		t.Fatal(err)
	}
	out, err := pipeline.Process([]float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3, 5, 7}
	for index := range want {
		if out[index] != want[index] {
			t.Fatalf("out[%d] = %v, want %v", index, out[index], want[index])
		}
	}
	if names := pipeline.Names(); len(names) != 2 || names[0] != "double" || names[1] != "offset" {
		t.Fatalf("names = %v", names)
	}
}

func TestNewRefusesANilStage(t *testing.T) {
	if _, err := New(SampleStage("ok", func(v float32) float32 { return v }, nil), nil); err == nil {
		t.Fatal("a nil stage was accepted")
	}
}

func TestRegistryBuildsByName(t *testing.T) {
	Register("test-stage", func(Config) (Stage, error) {
		return SampleStage("test-stage", func(v float32) float32 { return v }, nil), nil
	})
	stage, err := Build("test-stage", Config{})
	if err != nil || stage.Name() != "test-stage" {
		t.Fatalf("Build = %v, %v", stage, err)
	}
	if _, err := Build("absent", Config{}); err == nil {
		t.Fatal("an unknown stage was built")
	}
}
