package audio

import "math"

// FbankConfig describes the mel filterbank the denoiser expects.
type FbankConfig struct {
	SampleRate  int
	NumMelBins  int
	FrameLength float64 // milliseconds
	FrameShift  float64 // milliseconds
	LowFreq     float64
	HighFreq    float64
	Preemphasis float64
}

// DefaultFbankConfig matches the reference pipeline: 48 kHz, 120 mel bins,
// 40 ms frames every 20 ms, Hamming window.
func DefaultFbankConfig() FbankConfig {
	return FbankConfig{
		SampleRate:  SampleRate,
		NumMelBins:  120,
		FrameLength: 40,
		FrameShift:  20,
		LowFreq:     20,
		HighFreq:    0, // 0 means Nyquist
		Preemphasis: 0.97,
	}
}

// silenceFloor is Kaldi's energy floor: float32 epsilon. Clamping before the
// log keeps a silent band finite and matches the reference front-end.
const silenceFloor float32 = 1.1920928955078125e-07

// Fbank turns samples into log-mel features, one row per frame. samples must
// already be normalised and scaled to 16-bit range, which is what Kaldi's
// fbank expects.
func Fbank(samples []float32, cfg FbankConfig) [][]float32 {
	if cfg.SampleRate == 0 {
		cfg = DefaultFbankConfig()
	}
	frameLength := int(math.Round(cfg.FrameLength * float64(cfg.SampleRate) / 1000))
	frameShift := int(math.Round(cfg.FrameShift * float64(cfg.SampleRate) / 1000))
	if frameLength <= 0 || frameShift <= 0 || len(samples) < frameLength {
		return nil
	}
	// Kaldi rounds the FFT size up to a power of two.
	fftSize := 1
	for fftSize < frameLength {
		fftSize <<= 1
	}
	window := hammingWindow(frameLength, frameLength)
	plan := newFFTPlan(fftSize)
	bins := make([]float32, 2*(fftSize/2+1))
	frame := make([]float32, fftSize)
	mel := melFilterbank(cfg, fftSize)
	preemphasis := float32(cfg.Preemphasis)

	// Kaldi's snip_edges framing keeps the leading frame: one frame at
	// offset zero, then one per shift while a whole frame still fits.
	frames := 1 + (len(samples)-frameLength)/frameShift
	if len(samples) < frameLength {
		return nil
	}
	out := make([][]float32, 0, frames)
	for index := 0; index < frames; index++ {
		start := index * frameShift
		// Remove the DC offset, as Kaldi does by default.
		var mean float32
		for offset := 0; offset < frameLength; offset++ {
			mean += samples[start+offset]
		}
		mean /= float32(frameLength)
		// Kaldi preemphasises after DC removal and before the window:
		// x[i] -= coeff*x[i-1], where x[-1] repeats the frame's own first
		// sample rather than the preceding signal sample.
		for offset := 0; offset < frameLength; offset++ {
			frame[offset] = samples[start+offset] - mean
		}
		for offset := frameLength - 1; offset > 0; offset-- {
			frame[offset] -= preemphasis * frame[offset-1]
		}
		frame[0] -= preemphasis * frame[0]
		for offset := 0; offset < frameLength; offset++ {
			frame[offset] *= window[offset]
		}
		for offset := frameLength; offset < fftSize; offset++ {
			frame[offset] = 0
		}
		plan.forward(frame, bins)
		// Power spectrum, then the mel filterbank and the log.
		power := make([]float32, fftSize/2+1)
		for bin := range power {
			re, im := bins[2*bin], bins[2*bin+1]
			power[bin] = re*re + im*im
		}
		features := make([]float32, cfg.NumMelBins)
		for band := 0; band < cfg.NumMelBins; band++ {
			var energy float32
			for bin, weight := range mel[band] {
				energy += weight * power[bin]
			}
			// Kaldi clamps the band energy at float32 epsilon before the log,
			// so digital silence lands at log(1.1920928955078125e-07) instead
			// of an arbitrary sentinel.
			if energy < silenceFloor {
				energy = silenceFloor
			}
			features[band] = float32(math.Log(float64(energy)))
		}
		out = append(out, features)
	}
	return out
}

// melFilterbank builds triangular filters over the FFT bins.
func melFilterbank(cfg FbankConfig, fftSize int) [][]float32 {
	nyquist := float64(cfg.SampleRate) / 2
	high := cfg.HighFreq
	if high <= 0 || high > nyquist {
		high = nyquist
	}
	lowMel := hzToMel(cfg.LowFreq)
	highMel := hzToMel(high)
	points := make([]float64, cfg.NumMelBins+2)
	for index := range points {
		fraction := float64(index) / float64(cfg.NumMelBins+1)
		points[index] = melToHz(lowMel + fraction*(highMel-lowMel))
	}
	binCount := fftSize/2 + 1
	bank := make([][]float32, cfg.NumMelBins)
	for band := 0; band < cfg.NumMelBins; band++ {
		left, centre, right := points[band], points[band+1], points[band+2]
		weights := make([]float32, binCount)
		for bin := 0; bin < binCount; bin++ {
			freq := float64(bin) * float64(cfg.SampleRate) / float64(fftSize)
			switch {
			case freq > left && freq <= centre && centre > left:
				weights[bin] = float32((freq - left) / (centre - left))
			case freq > centre && freq < right && right > centre:
				weights[bin] = float32((right - freq) / (right - centre))
			}
		}
		bank[band] = weights
	}
	return bank
}

func hzToMel(hz float64) float64  { return 1127 * math.Log(1+hz/700) }
func melToHz(mel float64) float64 { return 700 * (math.Exp(mel/1127) - 1) }
