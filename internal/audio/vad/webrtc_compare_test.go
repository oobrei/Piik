package vad

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func readWAV24k(t *testing.T, path string) []float32 {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(payload) < 44 || string(payload[0:4]) != "RIFF" {
		t.Fatalf("%s is not a WAV", path)
	}
	var data []byte
	channels := 1
	for offset := 12; offset+8 <= len(payload); {
		id := string(payload[offset : offset+4])
		size := int(binary.LittleEndian.Uint32(payload[offset+4:]))
		body := payload[offset+8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			channels = int(binary.LittleEndian.Uint16(body[2:]))
		case "data":
			data = body[:size]
		}
		offset += 8 + size + size%2
	}
	if channels != 1 {
		t.Fatalf("not mono: %d", channels)
	}
	samples := make([]float32, len(data)/2)
	for i := range samples {
		// WebRTC's VAD expects int16-scale floats, not unit-scale.
		samples[i] = float32(int16(binary.LittleEndian.Uint16(data[i*2:])))
	}
	return samples
}

func readFloat32File(t *testing.T, path string) []float32 {
	t.Helper()
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	values := make([]float32, len(payload)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(payload[i*4:]))
	}
	return values
}

// TestNetworkMatchesWebRTC feeds the exact feature vectors WebRTC produced into
// our network and compares the speech probabilities. This isolates the network
// arithmetic from feature extraction.
func TestNetworkMatchesWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	features := readFloat32File(t, filepath.Join(dir, "webrtc_poem_features.f32"))
	want := readFloat32File(t, filepath.Join(dir, "webrtc_poem_vad.f32"))
	frames := len(features) / FeatureSize
	if frames != len(want) {
		t.Fatalf("%d feature frames, %d probabilities", frames, len(want))
	}
	network := NewNetwork()
	worst, worstFrame := 0.0, 0
	for frame := 0; frame < frames; frame++ {
		row := features[frame*FeatureSize : (frame+1)*FeatureSize]
		got := network.Probability(row)
		if diff := math.Abs(float64(got - want[frame])); diff > worst {
			worst, worstFrame = diff, frame
		}
	}
	t.Logf("network vs WebRTC: max probability error %.6f at frame %d", worst, worstFrame)
	if worst > 0.05 {
		t.Fatalf("network diverges from WebRTC by %.4f at frame %d", worst, worstFrame)
	}
}

// TestFeaturesMatchWebRTC compares our full extractor against WebRTC's on the
// same 10 ms frames. Every one of the 42 features is compared, frame by frame.
func TestFeaturesMatchWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	samples := readWAV24k(t, filepath.Join(dir, "poem_24k.wav"))
	want := readFloat32File(t, filepath.Join(dir, "webrtc_poem_features.f32"))
	frames := len(want) / FeatureSize

	extractor := NewExtractor()
	worst, worstFrame, worstBin := 0.0, 0, 0
	worstPitch, worstPitchFrame := 0.0, 0
	checked := 0
	frame := make([]float32, Frame10ms24k)
	for f := 0; f < frames; f++ {
		start := f * Frame10ms24k
		for i := range frame {
			frame[i] = samples[start+i]
		}
		got, silent := extractor.Process(frame)
		wantRow := want[f*FeatureSize : (f+1)*FeatureSize]
		if silent {
			if f < 3 {
				t.Logf("frame %d silent; want[0]=%.3f pitch=%.3f", f, wantRow[0], wantRow[FeatureSize-2])
			}
			continue
		}
		checked++
		for bin, value := range got {
			diff := math.Abs(float64(value) - float64(wantRow[bin]))
			if bin == FeatureSize-2 {
				if diff > worstPitch {
					worstPitch, worstPitchFrame = diff, f
				}
				continue
			}
			if diff > worst {
				worst, worstFrame, worstBin = diff, f, bin
			}
		}
	}
	t.Logf("features vs WebRTC: %d frames checked, max error %.5f at frame %d bin %d",
		checked, worst, worstFrame, worstBin)
	perBin := make([]float32, FeatureSize)
	ex2 := NewExtractor()
	for f := 0; f < frames; f++ {
		start := f * Frame10ms24k
		fr := make([]float32, Frame10ms24k)
		for i := range fr {
			fr[i] = samples[start+i]
		}
		got, silent := ex2.Process(fr)
		if silent {
			continue
		}
		row := want[f*FeatureSize : (f+1)*FeatureSize]
		for bin, v := range got {
			if d := abs32(v - row[bin]); d > perBin[bin] {
				perBin[bin] = d
			}
		}
	}
	t.Logf("per-bin max: %v", perBin)
	// Count how many feature values agree within 0.1, to gauge overall fidelity.
	within, total := 0, 0
	ex := NewExtractor()
	for f := 0; f < frames; f++ {
		start := f * Frame10ms24k
		fr := make([]float32, Frame10ms24k)
		for i := range fr {
			fr[i] = samples[start+i]
		}
		got, silent := ex.Process(fr)
		if silent {
			continue
		}
		row := want[f*FeatureSize : (f+1)*FeatureSize]
		for bin, v := range got {
			total++
			if math.Abs(float64(v)-float64(row[bin])) < 0.1 {
				within++
			}
		}
	}
	t.Logf("feature values within 0.1: %d/%d (%.1f%%)", within, total, 100*float64(within)/float64(total))
	t.Logf("pitch feature vs WebRTC: max error %.5f at frame %d", worstPitch, worstPitchFrame)
	if worst > 0.5 {
		t.Fatalf("feature extraction diverges from WebRTC by %.3f at frame %d bin %d", worst, worstFrame, worstBin)
	}
}

// TestEndToEndMatchesWebRTC runs the whole Go pipeline - extractor and network -
// and compares the speech probabilities with WebRTC's own output.
func TestEndToEndMatchesWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	samples := readWAV24k(t, filepath.Join(dir, "poem_24k.wav"))
	want := readFloat32File(t, filepath.Join(dir, "webrtc_poem_vad.f32"))
	frames := len(want)

	extractor := NewExtractor()
	network := NewNetwork()
	frame := make([]float32, Frame10ms24k)
	worst, worstFrame := 0.0, 0
	for f := 0; f < frames; f++ {
		start := f * Frame10ms24k
		for i := range frame {
			frame[i] = samples[start+i]
		}
		features, silent := extractor.Process(frame)
		if silent {
			continue
		}
		got := network.Probability(features)
		if diff := math.Abs(float64(got) - float64(want[f])); diff > worst {
			worst, worstFrame = diff, f
		}
	}
	t.Logf("end-to-end vs WebRTC: max probability error %.6f at frame %d", worst, worstFrame)
	if worst > 0.05 {
		t.Fatalf("end-to-end output diverges by %.4f at frame %d", worst, worstFrame)
	}
}
