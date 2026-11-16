package dfsmn

import (
	"math"
	"testing"

	"github.com/TNTcraftHIM/Piik/internal/audio"
)

func newSTFT(t *testing.T) (*audio.Analyzer, *audio.Synthesizer) {
	t.Helper()
	analysis, err := audio.NewAnalyzer(audio.STFTConfig{})
	if err != nil {
		t.Fatalf("analyzer: %v", err)
	}
	synth, err := audio.NewSynthesizer(audio.STFTConfig{})
	if err != nil {
		t.Fatalf("synthesizer: %v", err)
	}
	return analysis, synth
}

func sine(length int) []float32 {
	samples := make([]float32, length)
	for index := range samples {
		samples[index] = float32(0.5 * math.Sin(2*math.Pi*440*float64(index)/audio.SampleRate))
	}
	return samples
}

// An identity mask must return the signal it was given. Anything else means
// the analysis/synthesis pair is losing or reshaping audio, which no model can
// compensate for.
func TestSynthesisRoundTripsIdentityMask(t *testing.T) {
	analysis, synth := newSTFT(t)
	size, hop := analysis.Size(), analysis.Hop()
	samples := sine(20*hop + size)
	bins := make([]float32, 2*analysis.Bins())
	out := make([]float32, 0, len(samples))
	for start := 0; start+size <= len(samples); start += hop {
		analysis.Analyze(samples[start:start+size], bins)
		out = synth.Synthesize(bins, out)
	}
	out = synth.Drain(out)

	for index, value := range out {
		if value == 0 {
			t.Fatalf("sample %d came out exactly zero", index)
		}
	}
	worst := 0.0
	for index := size; index < len(out)-size; index++ {
		worst = math.Max(worst, math.Abs(float64(out[index]-samples[index])))
	}
	if worst > 1e-3 {
		t.Fatalf("identity mask changed the signal by %.3e", worst)
	}
	t.Logf("identity round trip: %d samples, worst deviation %.2e", len(out), worst)
}

// A mask of zero must silence the output. If a zeroed spectrum still produces
// audio, the synthesizer is holding state that leaks past the masking.
func TestZeroMaskSilencesOutput(t *testing.T) {
	analysis, synth := newSTFT(t)
	size, hop := analysis.Size(), analysis.Hop()
	samples := sine(20*hop + size)
	bins := make([]float32, 2*analysis.Bins())
	out := make([]float32, 0, len(samples))
	for start := 0; start+size <= len(samples); start += hop {
		analysis.Analyze(samples[start:start+size], bins)
		for index := range bins {
			bins[index] = 0
		}
		out = synth.Synthesize(bins, out)
	}
	out = synth.Drain(out)

	worst := 0.0
	for _, value := range out {
		worst = math.Max(worst, math.Abs(float64(value)))
	}
	if worst != 0 {
		t.Fatalf("zero mask left %.3e of signal", worst)
	}
}

// A low tone must survive the analysis/synthesis pair at the same level as a
// high one. The lowest bins are the ones a windowed overlap-add is most likely
// to distort, so this separates a DSP defect from a model one.
func TestSynthesisPreservesLowTone(t *testing.T) {
	for _, hz := range []float64{40, 80, 440, 4000} {
		analysis, synth := newSTFT(t)
		size, hop := analysis.Size(), analysis.Hop()
		samples := make([]float32, 20*hop+size)
		for index := range samples {
			samples[index] = float32(0.5 * math.Sin(2*math.Pi*hz*float64(index)/audio.SampleRate))
		}
		bins := make([]float32, 2*analysis.Bins())
		out := make([]float32, 0, len(samples))
		for start := 0; start+size <= len(samples); start += hop {
			analysis.Analyze(samples[start:start+size], bins)
			out = synth.Synthesize(bins, out)
		}
		out = synth.Drain(out)
		// Compare the steady-state region only.
		from, to := 4*size, len(out)-4*size
		var inPower, outPower float64
		for index := from; index < to; index++ {
			inPower += float64(samples[index]) * float64(samples[index])
			outPower += float64(out[index]) * float64(out[index])
		}
		ratio := math.Sqrt(outPower / inPower)
		if math.Abs(ratio-1) > 0.01 {
			t.Fatalf("%.0f Hz: output is %.4f of the input", hz, ratio)
		}
		t.Logf("%.0f Hz: ratio %.6f", hz, ratio)
	}
}

// A mask that removes a band must remove it from the output. This is the
// property the denoiser depends on: a mask of one everywhere else and zero
// across a frequency range has to silence that range, which is only true if
// mask index and spectrum bin line up.
func TestMaskRemovesTheBandItTargets(t *testing.T) {
	analysis, synth := newSTFT(t)
	size, hop := analysis.Size(), analysis.Hop()
	const tone = 3000.0
	samples := make([]float32, 20*hop+size)
	for index := range samples {
		samples[index] = float32(0.5 * math.Sin(2*math.Pi*tone*float64(index)/audio.SampleRate))
	}
	bins := make([]float32, 2*analysis.Bins())
	out := make([]float32, 0, len(samples))
	for start := 0; start+size <= len(samples); start += hop {
		analysis.Analyze(samples[start:start+size], bins)
		// Zero the bins around the tone, leave the rest untouched.
		for bin := 0; bin < analysis.Bins(); bin++ {
			hz := float64(bin) * audio.SampleRate / float64(size)
			if hz > tone-400 && hz < tone+400 {
				bins[2*bin], bins[2*bin+1] = 0, 0
			}
		}
		out = synth.Synthesize(bins, out)
	}
	out = synth.Drain(out)

	// Measure the energy left at the tone.
	var power float64
	for index := size; index < len(out)-size; index++ {
		power += float64(out[index]) * float64(out[index])
	}
	rms := math.Sqrt(power / float64(len(out)-2*size))
	if rms > 0.02 {
		t.Fatalf("masking the tone left %.4f RMS, want it silenced", rms)
	}
	t.Logf("masked tone left %.6f RMS", rms)
}

// The stage must not emit samples that are exactly zero while it still holds
// signal. A sigmoid mask is strictly positive, so a zero sample means the
// synthesis lost the frame rather than the model removing noise.
func TestStageEmitsNoSilentSamples(t *testing.T) {
	stage, err := NewStage(loadModel(t))
	if err != nil {
		t.Fatalf("stage: %v", err)
	}
	in := sine(50 * 960)
	out, err := stage.Process(in, nil)
	if err != nil {
		t.Fatalf("process: %v", err)
	}
	out, err = stage.Flush(out)
	if err != nil {
		t.Fatalf("flush: %v", err)
	}
	if len(out) < len(in) {
		t.Fatalf("%d samples out for %d in", len(out), len(in))
	}
	silent, first := 0, -1
	for index, value := range out[:len(in)] {
		if value == 0 {
			silent++
			if first < 0 {
				first = index
			}
		}
	}
	if silent > 0 {
		t.Fatalf("%d of %d samples came out exactly zero, first at %d", silent, len(in), first)
	}
}
