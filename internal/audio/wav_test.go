package audio

import (
	"encoding/binary"
	"os"
	"testing"
)

// readWAV24kSamples reads a mono 16-bit WAV as int16-scale float32 samples,
// the scale WebRTC's AGC2 works in.
func readWAV24kSamples(t *testing.T, path string) []float32 {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(payload) < 44 || string(payload[0:4]) != "RIFF" {
		t.Fatalf("%s is not a WAV", path)
	}
	var data []byte
	for offset := 12; offset+8 <= len(payload); {
		id := string(payload[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(payload[offset+4:]))
		body := payload[offset+8:]
		if size > len(body) {
			size = len(body)
		}
		if id == "data" {
			data = body[:size]
		}
		offset += 8 + size + size%2
	}
	samples := make([]float32, len(data)/2)
	for index := range samples {
		samples[index] = float32(int16(binary.LittleEndian.Uint16(data[index*2:])))
	}
	return samples
}
