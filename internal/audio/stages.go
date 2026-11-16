package audio

import (
	"fmt"
)

// SampleFunc transforms one sample. It may read previous output through the
// closure it captured, which is how a limiter keeps its envelope.
type SampleFunc func(sample float32) float32

// sampleStage adapts a per-sample function to the Stage interface.
type sampleStage struct {
	name    string
	process SampleFunc
	reset   func()
}

// SampleStage adapts fn to a Stage. reset, when not nil, clears whatever state
// fn closes over.
func SampleStage(name string, fn SampleFunc, reset func()) Stage {
	if reset == nil {
		reset = func() {}
	}
	return &sampleStage{name: name, process: fn, reset: reset}
}

func (stage *sampleStage) Name() string { return stage.name }

func (stage *sampleStage) Process(in, out []float32) ([]float32, error) {
	for _, sample := range in {
		out = append(out, stage.process(sample))
	}
	return out, nil
}

// Flush emits nothing: a per-sample stage holds no samples back unless it
// keeps its own lookahead, which SampleStage does not model.
func (stage *sampleStage) Flush(out []float32) ([]float32, error) { return out, nil }

func (stage *sampleStage) Reset() { stage.reset() }

// FrameFunc transforms one STFT frame in place. bins holds the spectrum in
// interleaved real/imaginary pairs, so len(bins) == 2*(fftSize/2+1). A stage
// that needs frame history keeps it in the closure it captured.
type FrameFunc func(bins []float32) error

// frameStage adapts an STFT frame processor to the Stage interface. It owns
// the analysis and synthesis, the overlap-add and the framing state, so a
// frame-domain stage only writes the per-frame transform.
type frameStage struct {
	name     string
	analysis *Analyzer
	synth    *Synthesizer
	fn       FrameFunc
	pending  []float32 // samples not yet forming a full frame
	bins     []float32
}

// FrameStage adapts fn to a Stage over an STFT with cfg. The stage buffers
// input until it has a whole hop, so callers may pass any length.
func FrameStage(name string, cfg STFTConfig, fn FrameFunc) (Stage, error) {
	analysis, err := NewAnalyzer(cfg)
	if err != nil {
		return nil, err
	}
	synth, err := NewSynthesizer(cfg)
	if err != nil {
		return nil, err
	}
	return &frameStage{
		name:     name,
		analysis: analysis,
		synth:    synth,
		fn:       fn,
		bins:     make([]float32, 2*analysis.Bins()),
	}, nil
}

func (stage *frameStage) Name() string { return stage.name }

func (stage *frameStage) Process(in, out []float32) ([]float32, error) {
	stage.pending = append(stage.pending, in...)
	size, hop := stage.analysis.Size(), stage.analysis.Hop()
	// A frame spans the window, not the hop: consecutive frames overlap by
	// size-hop, so the buffer must hold a whole frame before it can advance.
	for len(stage.pending) >= size {
		spec := stage.analysis.Analyze(stage.pending[:size], stage.bins)
		if err := stage.fn(spec); err != nil {
			return nil, err
		}
		out = stage.synth.Synthesize(spec, out)
		stage.pending = stage.pending[hop:]
	}
	if len(stage.pending) == 0 {
		stage.pending = nil
	}
	return out, nil
}

// Flush pads the remaining samples to a whole hop, runs the last frames and
// drains the synthesis tail, so no audio is lost at end of stream.
func (stage *frameStage) Flush(out []float32) ([]float32, error) {
	if len(stage.pending) > 0 {
		padding := make([]float32, stage.analysis.Size()-len(stage.pending))
		if _, err := stage.Process(padding, out); err != nil {
			return nil, err
		}
		stage.pending = nil
	}
	return stage.synth.Drain(out), nil
}

func (stage *frameStage) Reset() {
	stage.pending = nil
	stage.analysis.Reset()
	stage.synth.Reset()
}

// Describe reports a stage's name for diagnostics.
func Describe(stage Stage) string {
	if stage == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%s", stage.Name())
}
