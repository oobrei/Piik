package audio

import (
	"math"
	"testing"
)

// limiterFrame builds a frame of the size the limiter expects: 20 sub-frames of
// equal length.
func limiterFrame(samples int) []float32 { return make([]float32, samples) }

// A signal below the knee must pass through untouched.
func TestLimiterLeavesQuietSignalAlone(t *testing.T) {
	limiter := NewLimiter()
	samples := limiterFrame(480)
	for index := range samples {
		samples[index] = float32(100 * math.Sin(2*math.Pi*440*float64(index)/SampleRate))
	}
	want := append([]float32(nil), samples...)
	limiter.Process(samples)
	for index := range samples {
		if math.Abs(float64(samples[index]-want[index])) > 0.5 {
			t.Fatalf("sample %d changed from %.1f to %.1f", index, want[index], samples[index])
		}
	}
}

// A signal above the ceiling must come out below it.
func TestLimiterHoldsTheCeiling(t *testing.T) {
	limiter := NewLimiter()
	samples := limiterFrame(480)
	for index := range samples {
		samples[index] = float32(60000 * math.Sin(2*math.Pi*440*float64(index)/SampleRate))
	}
	limiter.Process(samples)
	peak := 0.0
	for _, value := range samples {
		peak = math.Max(peak, math.Abs(float64(value)))
	}
	// WebRTC clamps to the float S16 range, so the ceiling is 32768.
	if peak > 32768 {
		t.Fatalf("peak %.1f exceeds the float S16 ceiling", peak)
	}
	// The curve compresses towards +1 dBFS, so the peak should land near it.
	if peak < 30000 {
		t.Fatalf("peak %.1f is far below the ceiling, the gain over-corrected", peak)
	}
	t.Logf("peak after limiting: %.1f", peak)
}

// The gain must not step between samples: a jump is audible as a click.
func TestLimiterRampsGain(t *testing.T) {
	limiter := NewLimiter()
	samples := limiterFrame(480)
	for index := range samples {
		samples[index] = float32(60000 * math.Sin(2*math.Pi*440*float64(index)/SampleRate))
	}
	limiter.Process(samples)
	worst := 0.0
	for index := 1; index < len(samples); index++ {
		worst = math.Max(worst, math.Abs(float64(samples[index]-samples[index-1])))
	}
	// A 440 Hz sine at 60 dB above the ceiling would step by thousands if the
	// gain were applied without ramping.
	if worst > 8000 {
		t.Fatalf("gain stepped by %.0f between samples", worst)
	}
}

// The limiter reports the envelope the gain controller depends on.
func TestLimiterReportsAudioLevel(t *testing.T) {
	limiter := NewLimiter()
	if level := limiter.LastAudioLevel(); level != 0 {
		t.Fatalf("fresh limiter reports level %.1f, want 0", level)
	}
	samples := limiterFrame(480)
	for index := range samples {
		samples[index] = 1000
	}
	limiter.Process(samples)
	if level := limiter.LastAudioLevel(); level < 900 || level > 1000 {
		t.Fatalf("level %.1f does not track a constant 1000 envelope", level)
	}
	limiter.Reset()
	if level := limiter.LastAudioLevel(); level != 0 {
		t.Fatalf("reset limiter reports level %.1f, want 0", level)
	}
}
