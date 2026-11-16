package vad

import (
	"math"
	"testing"
)

// The approximated tanh must stay close to the real one and saturate at the
// ends, which is what the table plus correction term is for.
func TestTansigApproximation(t *testing.T) {
	for _, x := range []float32{-10, -4, -1, -0.5, 0, 0.5, 1, 4, 10} {
		got := tansig(x)
		want := float32(math.Tanh(float64(x)))
		if diff := math.Abs(float64(got - want)); diff > 0.01 {
			t.Fatalf("tansig(%v) = %v, want about %v", x, got, want)
		}
	}
	if tansig(8) != 1 || tansig(-8) != -1 {
		t.Fatalf("tansig must saturate outside [-8, 8]")
	}
}

// Sigmoid must be in (0, 1) and monotonic.
func TestSigmoidRange(t *testing.T) {
	previous := float32(-1)
	for _, x := range []float32{-8, -2, 0, 2, 8} {
		got := sigmoid(x)
		if got <= 0 || got >= 1 {
			t.Fatalf("sigmoid(%v) = %v, want within (0, 1)", x, got)
		}
		if got < previous {
			t.Fatalf("sigmoid is not monotonic at %v", x)
		}
		previous = got
	}
}

// The network must produce a probability in range and respond to its input.
func TestNetworkRespondsToFeatures(t *testing.T) {
	network := NewNetwork()
	quiet := make([]float32, FeatureSize)
	for index := range quiet {
		quiet[index] = -1
	}
	loud := make([]float32, FeatureSize)
	for index := range loud {
		loud[index] = 1
	}
	low := network.Probability(quiet)
	network.Reset()
	high := network.Probability(loud)
	if low <= 0 || low >= 1 || high <= 0 || high >= 1 {
		t.Fatalf("probabilities out of range: %v, %v", low, high)
	}
	if math.Abs(float64(high-low)) < 1e-3 {
		t.Fatalf("network ignored its input: %v vs %v", low, high)
	}
	t.Logf("quiet %.4f, loud %.4f", low, high)
}

// Reset must clear the recurrent state, so a stream restarts cleanly.
func TestResetClearsState(t *testing.T) {
	network := NewNetwork()
	features := make([]float32, FeatureSize)
	for index := range features {
		features[index] = 0.5
	}
	network.Probability(features)
	network.Reset()
	for index, value := range network.state {
		if value != 0 {
			t.Fatalf("state %d is %v after reset", index, value)
		}
	}
}
