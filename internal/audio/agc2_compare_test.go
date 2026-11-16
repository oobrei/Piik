package audio

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func readAGC2Reference(t *testing.T, path string) [][5]float64 {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Skipf("no reference: %v", err)
	}
	defer file.Close()
	var rows [][5]float64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 5 {
			continue
		}
		var row [5]float64
		for index, field := range fields {
			row[index], _ = strconv.ParseFloat(field, 64)
		}
		rows = append(rows, row)
	}
	return rows
}

// TestAGC2StagesMatchWebRTC compares the noise floor, RMS, speech level,
// headroom and peak against WebRTC's AGC2 on the same frames.
func TestAGC2StagesMatchWebRTC(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES")
	}
	reference := readAGC2Reference(t, filepath.Join(dir, "agc2_ref.txt"))
	if len(reference) == 0 {
		t.Skip("empty reference")
	}
	samples := readWAV24kSamples(t, filepath.Join(dir, "poem_24k.wav"))
	const frameSize = 480

	noise := NewNoiseLevelEstimator()
	speech := NewSpeechLevelEstimator(DefaultAdaptiveDigitalConfig())
	saturation := NewSaturationProtector(agc2SaturationProtectorInitialHeadroomDB)

	worst := [5]float64{}
	worstFrame := [5]int{}
	frames := 0
	for index := 0; index < len(reference); index++ {
		start := index * frameSize
		if start+frameSize > len(samples) {
			break
		}
		frame := samples[start : start+frameSize]
		noiseDBFS := noise.Analyze(frame)
		levels := computeAudioLevels(frame)
		speech.Update(levels.rmsDBFS, 0)
		target := 0.0
		if speech.IsConfident() {
			target = 0
		}
		saturation.Analyze(target, levels.peakDBFS, speech.LevelDBFS())

		got := [5]float64{noiseDBFS, levels.rmsDBFS, speech.LevelDBFS(), saturation.HeadroomDB(), levels.peakDBFS}
		for field := range got {
			if diff := math.Abs(got[field] - reference[index][field]); diff > worst[field] {
				worst[field] = diff
				worstFrame[field] = index
			}
		}
		frames++
	}
	names := [5]string{"noise", "rms", "speechLevel", "headroom", "peak"}
	for field, name := range names {
		t.Logf("%s: max error %.6f at frame %d", name, worst[field], worstFrame[field])
	}
	for field, name := range names {
		if worst[field] > 0.01 {
			t.Fatalf("%s diverges from WebRTC by %.4f at frame %d", name, worst[field], worstFrame[field])
		}
	}
	_ = fmt.Sprintf
}
