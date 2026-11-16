// Package audio is Piik's room-voice processing chain: a linear pipeline of
// stages that transform 48 kHz mono float32 audio in place.
//
// It serves room voice only. Screen-share audio stays with the picture and is
// never routed through this package: it keeps its own capture path and no
// speech processing, so a denoiser or gain controller here cannot colour the
// sound of a shared film or game.
//
// A stage may work on samples (a limiter) or on STFT frames (a denoiser). The
// pipeline does not care which: FrameStage and SampleStage adapt either shape
// to the same interface, so a model-backed stage and an algorithm-backed one
// are peers.
package audio

import (
	"errors"
	"fmt"
	"sync"
)

// SampleRate is the rate every stage works at. It matches the App's Opus
// encoder, so a stage never resamples.
const SampleRate = 48_000

// Stage transforms audio. Process appends its output to out and returns the
// extended slice; a stage that holds samples back for lookahead returns fewer
// than it consumed and emits the rest from Flush.
//
// A stage keeps streaming state between calls. Reset clears it, so one
// session's noise estimate never leaks into the next.
type Stage interface {
	// Name identifies the stage in diagnostics and configuration.
	Name() string
	// Process consumes in and appends its output to out.
	Process(in, out []float32) ([]float32, error)
	// Flush appends whatever the stage held back, ending the stream.
	Flush(out []float32) ([]float32, error)
	// Reset clears streaming state.
	Reset()
}

// Pipeline runs stages in order. It is not safe for concurrent use; one
// pipeline belongs to one audio stream.
type Pipeline struct {
	stages []Stage
}

// New builds a pipeline from stages. A nil stage is refused rather than
// ignored, because a silently missing denoiser is worse than a startup error.
func New(stages ...Stage) (*Pipeline, error) {
	for index, stage := range stages {
		if stage == nil {
			return nil, fmt.Errorf("audio: stage %d is nil", index)
		}
	}
	return &Pipeline{stages: append([]Stage(nil), stages...)}, nil
}

// Process runs in through every stage.
func (pipeline *Pipeline) Process(in []float32) ([]float32, error) {
	out := make([]float32, 0, len(in))
	var err error
	for _, stage := range pipeline.stages {
		if out, err = stage.Process(in, out[:0]); err != nil {
			return nil, fmt.Errorf("audio: %s: %w", stage.Name(), err)
		}
		in = out
	}
	return in, nil
}

// Flush drains every stage in order, so a stage's tail passes through the
// stages after it.
func (pipeline *Pipeline) Flush() ([]float32, error) {
	var tail []float32
	for _, stage := range pipeline.stages {
		var err error
		if tail, err = stage.Flush(tail); err != nil {
			return nil, fmt.Errorf("audio: %s: %w", stage.Name(), err)
		}
	}
	return tail, nil
}

// Reset clears every stage.
func (pipeline *Pipeline) Reset() {
	for _, stage := range pipeline.stages {
		stage.Reset()
	}
}

// Names reports the stages in order, for diagnostics.
func (pipeline *Pipeline) Names() []string {
	names := make([]string, len(pipeline.stages))
	for index, stage := range pipeline.stages {
		names[index] = stage.Name()
	}
	return names
}

// ErrUnknownStage reports a stage name no package registered.
var ErrUnknownStage = errors.New("audio: unknown stage")

// Factory builds a stage from its configuration.
type Factory func(cfg Config) (Stage, error)

// Config is the shared stage configuration. A stage reads only the fields it
// needs, so one config type serves every stage without per-stage plumbing.
type Config struct {
	// ModelDir holds model artifacts, such as the denoiser's graph and
	// weights. Empty means the stage's embedded default.
	ModelDir string
	// Enabled turns a stage on. A disabled stage is not built at all.
	Enabled bool
}

var (
	registryMu sync.RWMutex
	registry   = map[string]Factory{}
)

// Register makes a stage available by name. Packages call it from init, so a
// build that imports a stage can configure it by name.
func Register(name string, factory Factory) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if name == "" || factory == nil {
		panic("audio: Register needs a name and a factory")
	}
	if _, exists := registry[name]; exists {
		panic("audio: stage registered twice: " + name)
	}
	registry[name] = factory
}

// Build creates a registered stage.
func Build(name string, cfg Config) (Stage, error) {
	registryMu.RLock()
	factory, ok := registry[name]
	registryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrUnknownStage, name)
	}
	return factory(cfg)
}

// Registered reports the stage names this build carries, sorted.
func Registered() []string {
	registryMu.RLock()
	defer registryMu.RUnlock()
	names := make([]string, 0, len(registry))
	for name := range registry {
		names = append(names, name)
	}
	sortStrings(names)
	return names
}

func sortStrings(values []string) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}
