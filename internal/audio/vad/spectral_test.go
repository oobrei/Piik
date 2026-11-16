package vad

import (
	"encoding/binary"
	"math"
	"os"
	"testing"
)

func readFloats(t *testing.T, path string, count int) []float64 {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture %s unavailable: %v", path, err)
	}
	if len(payload) < count*4 {
		t.Fatalf("%s holds %d bytes, want %d", path, len(payload), count*4)
	}
	values := make([]float64, count)
	for index := range values {
		values[index] = float64(math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:])))
	}
	return values
}

// The generated weight table must match the one WebRTC hard-codes.
func TestBandWeightsMatchReference(t *testing.T) {
	if len(opusBandWeights) != len(referenceBandWeights) {
		t.Fatalf("got %d weights, want %d", len(opusBandWeights), len(referenceBandWeights))
	}
	for index := range opusBandWeights {
		if diff := float64(abs32(opusBandWeights[index] - referenceBandWeights[index])); diff > 1e-9 {
			t.Fatalf("weight %d = %v, want %v", index, opusBandWeights[index], referenceBandWeights[index])
		}
	}
}

// The generated window and DCT table must match the reference translation.
func TestWindowAndDctMatchReference(t *testing.T) {
	// The fixture is written with %.9g, so it carries nine significant digits.
	for index := range halfWindow {
		if diff := float64(abs32(halfWindow[index] - referenceHalfWindow[index])); diff > 1e-9 {
			t.Fatalf("window %d = %v, want %v", index, halfWindow[index], referenceHalfWindow[index])
		}
	}
	for index := range dctTable {
		// The fixture carries nine significant digits, which is at the limit of
		// float32; compare with a tolerance instead of bit equality.
		if diff := float64(abs32(dctTable[index] - referenceDctTable[index])); diff > 1e-5 {
			t.Fatalf("dct %d = %v, want %v", index, dctTable[index], referenceDctTable[index])
		}
	}
}

func toF32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = float32(v)
	}
	return out
}
