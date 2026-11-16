package audio

import (
	"errors"
	"math"
)

// STFTConfig describes a short-time Fourier transform. The zero value is not
// usable; NewAnalyzer fills in the defaults below.
type STFTConfig struct {
	// FFTSize is the transform length. Defaults to 1920.
	FFTSize int
	// Hop is the step between frames. Defaults to half the FFT size.
	Hop int
	// WindowLength is the analysis window length. Defaults to FFTSize.
	WindowLength int
}

const (
	defaultFFTSize = 1920
	defaultHop     = 960
)

func (cfg STFTConfig) normalized() (STFTConfig, error) {
	if cfg.FFTSize == 0 {
		cfg.FFTSize = defaultFFTSize
	}
	if cfg.Hop == 0 {
		cfg.Hop = cfg.FFTSize / 2
	}
	if cfg.WindowLength == 0 {
		cfg.WindowLength = cfg.FFTSize
	}
	if cfg.FFTSize <= 0 {
		return cfg, errors.New("audio: FFT size must be positive")
	}
	if cfg.Hop <= 0 || cfg.Hop > cfg.FFTSize {
		return cfg, errors.New("audio: hop must be in 1..FFTSize")
	}
	if cfg.WindowLength <= 0 || cfg.WindowLength > cfg.FFTSize {
		return cfg, errors.New("audio: window length must be in 1..FFTSize")
	}
	return cfg, nil
}

// Analyzer turns samples into STFT frames. It keeps no history: the caller
// supplies exactly one hop per frame, which is what the reference pipeline
// does with center=false.
type Analyzer struct {
	cfg    STFTConfig
	bins   int
	window []float32
	plan   spectral
	buffer []float32
}

// NewAnalyzer prepares an analyzer. It allocates its FFT plan once.
func NewAnalyzer(cfg STFTConfig) (*Analyzer, error) {
	cfg, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	plan, err := newSpectral(cfg.FFTSize)
	if err != nil {
		return nil, err
	}
	return &Analyzer{
		cfg:    cfg,
		bins:   cfg.FFTSize/2 + 1,
		window: hammingWindow(cfg.WindowLength, cfg.FFTSize),
		plan:   plan,
		buffer: make([]float32, cfg.FFTSize),
	}, nil
}

// Bins reports the number of frequency bins, FFTSize/2+1.
func (analyzer *Analyzer) Bins() int { return analyzer.bins }

// Hop reports the frame step in samples.
func (analyzer *Analyzer) Hop() int { return analyzer.cfg.Hop }

// Size reports the FFT size.
func (analyzer *Analyzer) Size() int { return analyzer.cfg.FFTSize }

// Analyze transforms one frame into dst, which holds interleaved real and
// imaginary pairs and must have 2*Bins() entries. It returns dst.
//
// frame must hold Size() samples. A shorter slice is zero-padded, so passing a
// hop produces a spectrum of half the frame and the mask computed from the
// whole frame no longer describes it.
func (analyzer *Analyzer) Analyze(frame, dst []float32) []float32 {
	for index := range analyzer.buffer {
		analyzer.buffer[index] = 0
	}
	copy(analyzer.buffer, frame)
	for index, value := range analyzer.window {
		analyzer.buffer[index] *= value
	}
	analyzer.plan.forward(analyzer.buffer, dst)
	return dst
}

// Reset clears the analyzer. It holds no state today, so this exists for the
// Stage contract and future windowing that does.
func (analyzer *Analyzer) Reset() {}

// Synthesizer turns STFT frames back into samples with weighted overlap-add.
// It keeps the overlap tail and the window envelope between calls, so frames
// may arrive one at a time and the output stays at unity gain.
type Synthesizer struct {
	cfg      STFTConfig
	bins     int
	window   []float32
	plan     spectral
	overlap  []float32 // accumulated weighted frames
	envelope []float32 // accumulated squared windows
	frame    []float32
}

// NewSynthesizer prepares a synthesizer matching analyzer's configuration.
func NewSynthesizer(cfg STFTConfig) (*Synthesizer, error) {
	cfg, err := cfg.normalized()
	if err != nil {
		return nil, err
	}
	plan, err := newSpectral(cfg.FFTSize)
	if err != nil {
		return nil, err
	}
	return &Synthesizer{
		cfg:      cfg,
		bins:     cfg.FFTSize/2 + 1,
		window:   hammingWindow(cfg.WindowLength, cfg.FFTSize),
		plan:     plan,
		overlap:  make([]float32, cfg.FFTSize),
		envelope: make([]float32, cfg.FFTSize),
		frame:    make([]float32, cfg.FFTSize),
	}, nil
}

// Synthesize adds one frame's contribution and returns the samples that became
// final, appended to out. It emits exactly one hop per frame.
//
// Analysis and synthesis both window, so the sum of squared windows must divide
// the result; without it the output is amplitude-modulated at the hop rate.
func (synth *Synthesizer) Synthesize(bins, out []float32) []float32 {
	synth.plan.inverse(bins, synth.frame)
	for index := range synth.frame {
		weighted := synth.frame[index] * synth.window[index]
		synth.overlap[index] += weighted
		synth.envelope[index] += synth.window[index] * synth.window[index]
	}
	hop := synth.cfg.Hop
	for index := 0; index < hop; index++ {
		value := synth.overlap[index]
		if envelope := synth.envelope[index]; envelope > 1e-8 {
			value /= envelope
		}
		out = append(out, value)
	}
	copy(synth.overlap, synth.overlap[hop:])
	copy(synth.envelope, synth.envelope[hop:])
	for index := len(synth.overlap) - hop; index < len(synth.overlap); index++ {
		synth.overlap[index] = 0
		synth.envelope[index] = 0
	}
	return out
}

// Drain emits the overlap still held after the last frame.
//
// Only the first hop can still hold signal. The synthesizer emits a hop per
// frame, so after the last frame the tail that remains is the second half of
// that frame's window, which is exactly one hop long. Emitting the whole
// buffer would append a hop of silence past the end of the stream.
func (synth *Synthesizer) Drain(out []float32) []float32 {
	hop := synth.cfg.Hop
	for index := 0; index < hop; index++ {
		value := synth.overlap[index]
		if envelope := synth.envelope[index]; envelope > 1e-8 {
			value /= envelope
		}
		out = append(out, value)
	}
	synth.Reset()
	return out
}

// Reset clears the overlap tail and the envelope.
func (synth *Synthesizer) Reset() {
	for index := range synth.overlap {
		synth.overlap[index] = 0
		synth.envelope[index] = 0
	}
}

// hammingWindow builds a Hamming window of length windowLength centred in a
// frame of fftSize, matching torch.hamming_window(periodic=False).
func hammingWindow(windowLength, fftSize int) []float32 {
	window := make([]float32, fftSize)
	if windowLength == 1 {
		window[0] = 1
		return window
	}
	for index := 0; index < windowLength; index++ {
		window[index] = float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(index)/float64(windowLength-1)))
	}
	return window
}
