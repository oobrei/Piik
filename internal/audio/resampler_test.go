package audio

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestResamplerMatchesWebRTC decimates the clip from 48 kHz to 24 kHz in 480
// sample frames and compares with WebRTC's PushSincResampler.
func TestResamplerMatchesWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	payload, err := os.ReadFile(filepath.Join(dir, "res_out.f32"))
	if err != nil {
		t.Skipf("no reference: %v", err)
	}
	reference := make([]float32, len(payload)/4)
	for index := range reference {
		reference[index] = math.Float32frombits(binary.LittleEndian.Uint32(payload[index*4:]))
	}
	samples := readWAV24kSamples(t, filepath.Join(dir, "poem_24k.wav"))

	const srcFrames, dstFrames = 480, 240
	resampler := newSincResampler(float64(srcFrames)/float64(dstFrames), srcFrames)
	// PushSincResampler primes the buffer with a discarded zero-input pass.
	resampler.prime()

	got := make([]float64, 0, len(reference))
	destination := make([]float64, dstFrames)
	// The resampler pulls 480 samples at a time from one continuous stream.
	consumed := 0
	read := func(dst []float64) {
		for index := range dst {
			if consumed < len(samples) {
				dst[index] = float64(samples[consumed])
				consumed++
			} else {
				dst[index] = 0
			}
		}
	}
	for start := 0; start+srcFrames <= len(samples); start += srcFrames {
		resampler.resample(destination, read)
		got = append(got, destination...)
	}
	limit := len(got)
	if len(reference) < limit {
		limit = len(reference)
	}
	worst, worstIndex := 0.0, 0
	for index := 0; index < limit; index++ {
		if diff := math.Abs(got[index] - float64(reference[index])); diff > worst {
			worst, worstIndex = diff, index
		}
	}
	t.Logf("resampler vs WebRTC: %d samples compared, max error %.6f at index %d", limit, worst, worstIndex)
	if worst > 1 {
		t.Fatalf("resampler diverges by %.4f at index %d", worst, worstIndex)
	}
}
