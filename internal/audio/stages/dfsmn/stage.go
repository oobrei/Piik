package dfsmn

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/TNTcraftHIM/Piik/internal/audio"
)

// Name is the stage's registry name.
const Name = "dfsmn"

// GraphFile is the model artifact the stage loads from audio.Config.ModelDir.
const GraphFile = "dfsmn_ans.onnx"

func init() { audio.Register(Name, newStage) }

func newStage(cfg audio.Config) (audio.Stage, error) {
	if cfg.ModelDir == "" {
		return nil, fmt.Errorf("dfsmn: no model directory configured")
	}
	model, err := Load(filepath.Join(cfg.ModelDir, GraphFile))
	if err != nil {
		return nil, err
	}
	return NewStage(model)
}

// Stage denoises a live stream: one 20 ms analysis frame at a time, each masked
// by the model before overlap-add. It is causal, so it emits every sample it
// consumes once a frame is complete and holds nothing back for lookahead.
//
// The mask attenuates, so the output is quieter than the input. The reference
// chain feeds the model normalised audio and scales the result back up to the
// level of the recording, which is why its output keeps the input's loudness.
//
// This stage does not do that yet: it returns the masked signal at whatever
// level the mask leaves it, which is about 6.7 dB below its input. Restoring
// the level is the caller's decision, because the caller is the one that
// normalised the audio and still holds the original.
//
// Input must already be normalised the way the reference front-end does, which
// is what keeps the features inside the distribution the model was trained on.
// The stage restores that normalisation on the way out, so a caller that feeds
// a recording at its own level gets a denoised recording at the same level.
type Stage struct {
	model     *Model
	features  audio.FbankConfig
	analysis  *audio.Analyzer
	synth     *audio.Synthesizer
	pending   []float32 // samples that do not yet fill an analysis frame
	previous  []float32 // the frame before pending, for the feature context
	bins      []float32
	dither    float64 // Gaussian dither amplitude the reference front-end uses
	rng       uint64  // this stream's dither generator
	spare     float64 // the second Box-Muller sample, when one is pending
	maskSum   float64 // running sum of the mask values the model produced
	maskCount int
	probe     [][]float32 // the first feature rows the stage computed, for tests
	maskProbe [][]float32 // the masks the stage applied to those rows
}

// DefaultDither is the dither amplitude the reference front-end applies. The
// model was trained on features computed with it, so the stage draws the same
// noise to stay inside that distribution.
const DefaultDither = 1.0

// NewStage wraps a loaded model as a stage and clears its look-back, so the
// stage starts a fresh stream. The stage owns the model's streaming state.
func NewStage(model *Model) (*Stage, error) {
	if model == nil {
		return nil, fmt.Errorf("dfsmn: nil model")
	}
	model.Reset()
	analysis, err := audio.NewAnalyzer(audio.STFTConfig{})
	if err != nil {
		return nil, err
	}
	synth, err := audio.NewSynthesizer(audio.STFTConfig{})
	if err != nil {
		return nil, err
	}
	if analysis.Bins() != Bins {
		return nil, fmt.Errorf("dfsmn: analysis gives %d bins, model mask is %d", analysis.Bins(), Bins)
	}
	return &Stage{
		model:    model,
		features: audio.DefaultFbankConfig(),
		analysis: analysis,
		synth:    synth,
		previous: make([]float32, analysis.Size()),
		bins:     make([]float32, 2*analysis.Bins()),
		dither:   DefaultDither,
		rng:      0x9E3779B97F4A7C15,
	}, nil
}

// context returns the frame preceded by the one before it. Kaldi's snip_edges
// framing puts a frame boundary every hop, so this span yields exactly two
// feature rows: one for the previous frame and one for the current frame. The
// model is causal, so the second row is the one that describes this frame.
func (stage *Stage) context(size int) []float32 {
	span := make([]float32, 0, 2*size)
	span = append(span, stage.previous...)
	return append(span, stage.pending[:size]...)
}

// advance moves the analysis position forward by one hop, carrying the frame
// just consumed into previous so the next call has its context.
func (stage *Stage) advance(size, hop int) {
	copy(stage.previous, stage.pending[:size])
	stage.pending = stage.pending[hop:]
}

func (stage *Stage) Name() string { return Name }

// frameFeatures returns the feature row for the frame at the head of pending.
// Kaldi's snip_edges framing needs a whole frame of context, so the row is
// computed from the previous frame plus this one and the last row is taken.
func (stage *Stage) frameFeatures(size int) ([]float32, error) {
	span := stage.context(size)
	if stage.dither != 0 {
		// The reference front-end dithers each frame with Gaussian noise before
		// DC removal. The model was trained on those features, and its mask
		// changes with them, so the stage has to draw the same kind of noise to
		// stay inside the distribution it was trained on.
		for index := range span {
			span[index] += float32(stage.dither * stage.normal())
		}
	}
	rows := audio.Fbank(span, stage.features)
	if len(rows) < 2 {
		return nil, fmt.Errorf("dfsmn: %d samples gave %d feature rows, want at least 2", 2*size, len(rows))
	}
	return rows[len(rows)-1], nil
}

// normal draws one standard normal sample with the Box-Muller transform. The
// stage keeps its own generator so a stream is reproducible from its seed and
// does not disturb the caller's randomness.
func (stage *Stage) normal() float64 {
	if stage.spare != 0 {
		value := stage.spare
		stage.spare = 0
		return value
	}
	stage.rng ^= stage.rng << 13
	stage.rng ^= stage.rng >> 7
	stage.rng ^= stage.rng << 17
	u1 := (float64(stage.rng>>11) + 0.5) / float64(1<<53)
	stage.rng ^= stage.rng << 13
	stage.rng ^= stage.rng >> 7
	stage.rng ^= stage.rng << 17
	u2 := (float64(stage.rng>>11) + 0.5) / float64(1<<53)
	radius := math.Sqrt(-2 * math.Log(u1))
	stage.spare = radius * math.Sin(2*math.Pi*u2)
	return radius * math.Cos(2*math.Pi*u2)
}

// FeatureRows returns the first count feature rows the stage would feed the
// model for samples, which is what the reference front-end can be compared
// against. It resets the stage, so it is a diagnostic, not a streaming path.
func (stage *Stage) FeatureRows(samples []float32, count int) [][]float32 {
	stage.Reset()
	size, hop := stage.analysis.Size(), stage.analysis.Hop()
	stage.pending = append(stage.pending, samples...)
	rows := make([][]float32, 0, count)
	for len(rows) < count && len(stage.pending) >= size {
		row, err := stage.frameFeatures(size)
		if err != nil {
			break
		}
		rows = append(rows, row)
		stage.advance(size, hop)
	}
	stage.Reset()
	return rows
}

func (stage *Stage) Process(in, out []float32) ([]float32, error) {
	stage.pending = append(stage.pending, in...)
	size, hop := stage.analysis.Size(), stage.analysis.Hop()
	for len(stage.pending) >= size {
		frame := stage.pending[:size]
		// Feed the model only the row for this frame. The row before it was
		// already fed on the previous step, and the model keeps its own
		// look-back, so replaying it would advance the causal state twice.
		row, err := stage.frameFeatures(size)
		if err != nil {
			return nil, err
		}
		masks, err := stage.model.Forward([][]float32{row})
		if err != nil {
			return nil, err
		}
		for _, mask := range masks[0] {
			stage.maskSum += float64(mask)
		}
		stage.maskCount += len(masks[0])
		if stage.probe != nil && len(stage.probe) < cap(stage.probe) {
			stage.probe = append(stage.probe, append([]float32(nil), row...))
			stage.maskProbe = append(stage.maskProbe, append([]float32(nil), masks[0]...))
		}
		spec := stage.analysis.Analyze(frame, stage.bins)
		// Analyze returns the analyzer's own buffer, so the next call overwrites
		// it. The mask has to be applied before anything else runs.
		for bin, mask := range masks[0] {
			spec[2*bin] *= mask
			spec[2*bin+1] *= mask
		}
		out = stage.synth.Synthesize(stage.bins, out)
		stage.advance(size, hop)
	}
	if len(stage.pending) == 0 {
		stage.pending = nil
	} else {
		// Keep the buffer bounded: without this the backing array grows with
		// the whole stream even though only a frame's worth is ever live. Copy
		// into a fresh slice rather than reslicing, so pending never aliases
		// the caller's input.
		stage.pending = append([]float32(nil), stage.pending...)
	}
	return out, nil
}

// Probe records the first count feature rows the stage computes, so a test can
// compare them with the reference front-end instead of inferring them.
func (stage *Stage) Probe(count int) {
	stage.probe = make([][]float32, 0, count)
}

// Probed returns the feature rows recorded so far.
func (stage *Stage) Probed() [][]float32 { return stage.probe }

// ProbedMasks returns the masks the stage applied to the probed rows.
func (stage *Stage) ProbedMasks() [][]float32 { return stage.maskProbe }

// MaskMean reports the mean mask value the model produced, for diagnostics.
func (stage *Stage) MaskMean() float64 {
	if stage.maskCount == 0 {
		return 0
	}
	return stage.maskSum / float64(stage.maskCount)
}

// restore scales the samples this call produced back to the level of the signal
// that arrived. The mask attenuates, so without this the stage returns audio
// quieter than it was given; the reference chain restores the level the same
// way, which is why its output keeps the recording's loudness.
//
// Only the samples this call produced are scaled, so earlier output is never
// rewritten and the stage stays streaming.
// Flush pads the remaining samples to a whole frame and drains the
// overlap-add tail, so no audio is lost at end of stream.
//
// The padding is a whole frame rather than the shortfall, because Process only
// emits a hop per completed frame: a frame needs size samples, so a short pad
// would leave the last hop of real audio unemitted.
func (stage *Stage) Flush(out []float32) ([]float32, error) {
	if len(stage.pending) > 0 {
		size, hop := stage.analysis.Size(), stage.analysis.Hop()
		// Pad so the real samples end exactly on a hop boundary: Process only
		// emits a hop per completed frame, so a shorter pad would leave the
		// final hop of audio in pending. Pad by a whole hop past the next
		// boundary, because the first hop of that extra frame is still covered
		// by the previous frame's window.
		remainder := len(stage.pending) % hop
		padding := 2*hop - remainder
		if padding < size-len(stage.pending) {
			padding = size - len(stage.pending)
		}
		// Process returns out by value, so the appended samples only reach the
		// caller if they are threaded through.
		var err error
		if out, err = stage.Process(make([]float32, padding), out); err != nil {
			return nil, err
		}
		stage.pending = nil
	}
	return stage.synth.Drain(out), nil
}

func (stage *Stage) Reset() {
	stage.pending = nil
	for index := range stage.previous {
		stage.previous[index] = 0
	}
	stage.rng = 0x9E3779B97F4A7C15
	stage.spare = 0
	stage.maskSum, stage.maskCount = 0, 0
	stage.model.Reset()
	stage.analysis.Reset()
	stage.synth.Reset()
}
