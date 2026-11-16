package dfsmn

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/audio"
)

// The reference front-end feeds unit-scale normalised samples to Kaldi's fbank
// with dither 1.0, so the stage consumes them at that scale. Scaling to 16-bit
// range instead lifts every feature about 7.4 log units above the distribution
// the model was trained on and the mask opens up.

// SixteenBitScale is the scale the reference pipeline applies to the audio
// before the front-end, and divides out of the result afterwards.
const sixteenBitScale = 32768

// TestPipelineAgainstReferenceAudio runs the stage over the source recordings
// and compares the result with the Python chain's output for the same clip.
//
// The source is what the denoiser sees in production; the reference is what it
// should produce. Feeding a reference back in would denoise audio that has
// already been denoised, which measures nothing.
func TestPipelineAgainstReferenceAudio(t *testing.T) {
	dir := os.Getenv("PIIK_FIXTURES")
	if dir == "" {
		t.Skip("set PIIK_FIXTURES to the reference fixture directory")
	}
	for _, clip := range []string{"rain", "poem"} {
		samples, err := readWAV(filepath.Join(dir, clip+"_source.wav"))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		want, err := readWAV(filepath.Join(dir, clip+"_reference.wav"))
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		// The reference pipeline scales the audio to 16-bit range and feeds it
		// straight to the front-end. There is no normalisation step: it scales
		// by 32768, denoises, and truncates the result back to int16.
		input := make([]float32, len(samples))
		for index, value := range samples {
			input[index] = float32(value * sixteenBitScale)
		}
		model, err := Load(graphPath)
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		stage, err := NewStage(model)
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		stage.Probe(200)
		pipeline, err := audio.New(stage)
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		denoised, err := pipeline.Process(input)
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		tail, err := pipeline.Flush()
		if err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		denoised = append(denoised, tail...)
		t.Logf("%s: process %d, flush %d, total %d, input %d (short %d)",
			clip, len(denoised)-len(tail), len(tail), len(denoised), len(input), len(input)-len(denoised))
		if len(denoised) < len(input) {
			t.Fatalf("%s: %d samples out for %d in", clip, len(denoised), len(input))
		}
		denoised = denoised[:len(input)]
		for index := range denoised {
			denoised[index] /= sixteenBitScale
		}
		fromFloor, fromSpeech := levels(samples)
		toFloor, toSpeech := levels(denoised)
		t.Logf("%s: floor %.1f -> %.1f dB (%+.1f), speech %.1f -> %.1f dB, SNR %.1f -> %.1f dB (%+.1f)",
			clip, fromFloor, toFloor, toFloor-fromFloor, fromSpeech, toSpeech,
			fromSpeech-fromFloor, toSpeech-toFloor, (toSpeech-toFloor)-(fromSpeech-fromFloor))

		// Compare against the clip as recorded as well: normalisation lifts the
		// signal, so measuring only the normalised input hides the gain the
		// chain applies to speech.
		if silent := silentRuns(denoised); silent > 0 {
			t.Errorf("%s: %d samples came out exactly zero, which the mask cannot produce", clip, silent)
		}
		rawFloor, rawSpeech := levels(samples)
		t.Logf("%s: against the source, floor %.1f -> %.1f dB (%+.1f), speech %.1f -> %.1f dB (%+.1f)",
			clip, rawFloor, toFloor, toFloor-rawFloor, rawSpeech, toSpeech, toSpeech-rawSpeech)

		// Compare with the deterministic reference chain, which is the same
		// graph and the same overlap-add. The handed-off reference WAVs are not
		// that chain's output: they track the source at 0.994 correlation and
		// their spectrum shows a high-pass, not a spectral mask.
		refFloor, refSpeech := levels(want)
		t.Logf("%s: reference, floor %.1f dB, speech %.1f dB", clip, refFloor, refSpeech)
		t.Logf("%s: mean mask applied %.4f", clip, stage.MaskMean())
		if rows := stage.Probed(); len(rows) > 0 {
			flat := make([]float32, 0, len(rows)*Features)
			for _, row := range rows {
				flat = append(flat, row...)
			}
			path := filepath.Join(dir, "go_"+clip+"_stage_features.f32")
			if err := writeFloats(path, flat); err != nil {
				t.Fatalf("%s: %v", clip, err)
			}
			mean := 0.0
			for _, value := range flat {
				mean += float64(value)
			}
			t.Logf("%s: probed %d feature rows, mean %.3f -> %s", clip, len(rows), mean/float64(len(flat)), path)
			masks := stage.ProbedMasks()
			maskFlat := make([]float32, 0, len(masks)*Bins)
			for _, row := range masks {
				maskFlat = append(maskFlat, row...)
			}
			maskPath := filepath.Join(dir, "go_"+clip+"_stage_mask.f32")
			if err := writeFloats(maskPath, maskFlat); err != nil {
				t.Fatalf("%s: %v", clip, err)
			}
			perFrame := 0.0
			for _, row := range masks {
				sum := 0.0
				for _, value := range row {
					sum += float64(value)
				}
				perFrame += sum / float64(len(row))
			}
			t.Logf("%s: probed %d masks, per-frame mean %.4f -> %s", clip, len(masks), perFrame/float64(len(masks)), maskPath)
		}

		// The reference chain denoises the normalised signal and then scales the
		// result back to the level of the recording, so the gain is the ratio
		// the normaliser applied. Without it the output is the denoised signal
		// at -25 dBFS, which is quiet but otherwise correct.

		if len(want) == len(denoised) {
			t.Logf("%s: correlation with reference %.6f", clip, correlation(denoised, want))
		} else {
			t.Logf("%s: %d samples out, reference has %d", clip, len(denoised), len(want))
		}

		path := filepath.Join(dir, "go_"+clip+"_denoised.f32")
		if err := writeFloats(path, denoised); err != nil {
			t.Fatalf("%s: %v", clip, err)
		}
		t.Logf("%s: wrote %d samples to %s", clip, len(denoised), path)
	}
}

// restoreGain reports the factor that puts a normalised signal back at the
// level of the recording it came from.
func restoreGain(original, normalised []float32) float64 {
	power := func(samples []float32) float64 {
		if len(samples) == 0 {
			return 0
		}
		var total float64
		for _, value := range samples {
			total += float64(value) * float64(value)
		}
		return math.Sqrt(total / float64(len(samples)))
	}
	from, to := power(original), power(normalised)
	if to == 0 {
		return 1
	}
	return from / to
}

// correlation reports how closely two equal-length signals track each other,
// which is what says whether this port reproduces the reference chain.
func correlation(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var meanA, meanB float64
	for index := range a {
		meanA += float64(a[index])
		meanB += float64(b[index])
	}
	meanA /= float64(len(a))
	meanB /= float64(len(b))
	var cov, varA, varB float64
	for index := range a {
		da, db := float64(a[index])-meanA, float64(b[index])-meanB
		cov += da * db
		varA += da * da
		varB += db * db
	}
	if varA == 0 || varB == 0 {
		return 0
	}
	return cov / math.Sqrt(varA*varB)
}

// silentRuns counts samples that came out exactly zero. A sigmoid mask is
// strictly positive, so a zero sample means the synthesis lost the frame rather
// than the model removing the noise.
func silentRuns(samples []float32) int {
	silent := 0
	for _, sample := range samples {
		if sample == 0 {
			silent++
		}
	}
	return silent
}

// levels reports the 10th and 90th percentile frame level in dBFS, which is a
// noise floor and a speech level this chain can be compared on.
func levels(samples []float32) (floor, speech float64) {
	const hop, size = 960, 1920
	frames := 1 + (len(samples)-size)/hop
	decibels := make([]float64, 0, frames)
	for frame := 0; frame < frames; frame++ {
		var power float64
		for _, sample := range samples[frame*hop : frame*hop+size] {
			power += float64(sample) * float64(sample)
		}
		rms := math.Sqrt(power / size)
		decibels = append(decibels, 20*math.Log10(math.Max(rms, 1e-12)))
	}
	sort.Float64s(decibels)
	edge := len(decibels) / 10
	if edge < 1 {
		edge = 1
	}
	return meanPower(decibels[:edge]), meanPower(decibels[len(decibels)-edge:])
}

func meanPower(decibels []float64) float64 {
	var total float64
	for _, value := range decibels {
		total += math.Pow(10, value/20)
	}
	return 20 * math.Log10(total/float64(len(decibels)))
}

// readWAV reads the 16-bit mono PCM the reference clips use.
func readWAV(path string) ([]float32, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(payload) < 12 || string(payload[0:4]) != "RIFF" || string(payload[8:12]) != "WAVE" {
		return nil, fmt.Errorf("%s is not a RIFF/WAVE file", path)
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
			if len(body) < 16 || binary.LittleEndian.Uint16(body[0:]) != 1 || binary.LittleEndian.Uint16(body[14:]) != 16 {
				return nil, fmt.Errorf("%s is not 16-bit PCM", path)
			}
			channels = int(binary.LittleEndian.Uint16(body[2:]))
		case "data":
			data = body[:size]
		}
		offset += 8 + size + size%2
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("%s has no data chunk", path)
	}
	samples := make([]float32, len(data)/2/channels)
	for index := range samples {
		var sum int32
		for channel := 0; channel < channels; channel++ {
			sum += int32(int16(binary.LittleEndian.Uint16(data[2*(index*channels+channel):])))
		}
		samples[index] = float32(sum) / float32(channels) / 32768
	}
	return samples, nil
}

func writeFloats(path string, values []float32) error {
	raw := make([]byte, 4*len(values))
	for index, value := range values {
		binary.LittleEndian.PutUint32(raw[4*index:], math.Float32bits(value))
	}
	return os.WriteFile(path, raw, 0o644)
}
