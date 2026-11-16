// Package voice exposes Piik's room-voice processing chain to the App's page.
//
// The chain itself is the Go AGC2 and DFSMN code in internal/audio, so the App
// runs the same algorithms the native path uses. The page owns microphone
// capture (a Browser getUserMedia stream) and hands each block of mono samples
// here; this package owns the streaming state of one microphone.
//
// Screen-share audio never reaches this package: it stays with the picture and
// keeps its own capture path.
package voice

import (
	"errors"
	"sync"

	"github.com/TNTcraftHIM/Piik/internal/audio"
	"github.com/TNTcraftHIM/Piik/internal/audio/onnx"
	"github.com/TNTcraftHIM/Piik/internal/audio/stages/dfsmn"
)

// BlockSamples is the block size the page sends: one 128-frame render quantum
// at 48 kHz, the unit an AudioWorklet produces.
const BlockSamples = 128

// Service processes one room-voice microphone. The page creates it when the
// microphone opens and closes it when the microphone stops, so one voice
// session never inherits another's gain or noise estimate.
type Service struct {
	mu    sync.Mutex
	chain *chain
}

// NewService returns a service with no open microphone session.
func NewService() *Service { return &Service{} }

// chain is the per-microphone streaming state.
type chain struct {
	denoiser audio.Stage
	agc      *audio.AGC2
	input    []float32 // samples waiting for a whole denoise frame
	agcInput []float32 // denoised samples waiting for AGC2
	output   []float32 // processed samples not yet returned
}

// denoiseFrame is the DFSMN analysis frame: 20 ms at 48 kHz.
const denoiseFrame = 960

// Open starts a voice session and returns its identifier. The identifier is
// opaque to the page; it exists so a later Close can name the session.
func (s *Service) Open() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chain != nil {
		return "", errors.New("voice: a microphone session is already open")
	}
	s.chain = newChain()
	return "microphone", nil
}

// Close ends the session and drops its streaming state.
func (s *Service) Close(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chain == nil {
		return nil
	}
	s.chain = nil
	return nil
}

// Process runs one block of mono float32 samples through denoise then AGC2 and
// returns the processed block. The result always has the same length as the
// input, so the page can write it straight into its output buffer.
func (s *Service) Process(id string, samples []float32) ([]float32, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chain == nil {
		return nil, errors.New("voice: no microphone session is open")
	}
	if len(samples) == 0 {
		return nil, nil
	}
	return s.chain.process(samples), nil
}

// newChain builds the processing chain. The DFSMN model is embedded in the
// binary; if it cannot be read the chain still runs AGC2 alone rather than
// failing the microphone.
func newChain() *chain {
	var denoiser audio.Stage
	if model, err := dfsmn.LoadBytes(onnx.ModelBytes); err == nil {
		if stage, stageErr := dfsmn.NewStage(model); stageErr == nil {
			denoiser = stage
		}
	}
	return &chain{
		denoiser: denoiser,
		agc:      audio.NewAGC2(audio.DefaultAGC2Config()),
	}
}

// process buffers input into the stages' frames and returns one block of
// output. Both stages work in fixed frames, so the chain keeps an input
// backlog and an output backlog; that decouples the page's 128-sample blocks
// from the 10 ms and 20 ms stage frames.
func (c *chain) process(input []float32) []float32 {
	c.input = append(c.input, input...)

	if c.denoiser != nil {
		for len(c.input) >= denoiseFrame {
			frame := c.input[:denoiseFrame]
			denoised, err := c.denoiser.Process(frame, nil)
			if err != nil || len(denoised) < denoiseFrame {
				c.agcInput = append(c.agcInput, frame...)
			} else {
				c.agcInput = append(c.agcInput, denoised[:denoiseFrame]...)
			}
			c.input = c.input[denoiseFrame:]
		}
	} else {
		c.agcInput = append(c.agcInput, c.input...)
		c.input = c.input[:0]
	}

	if len(c.agcInput) > 0 {
		if out, err := c.agc.Process(c.agcInput, nil); err == nil {
			c.output = append(c.output, out...)
		}
		c.agcInput = c.agcInput[:0]
	}

	result := make([]float32, len(input))
	if len(c.output) >= len(input) {
		copy(result, c.output[:len(input)])
		c.output = append(c.output[:0], c.output[len(input):]...)
	} else {
		copy(result, c.output)
		c.output = c.output[:0]
	}
	return result
}

// Reset clears the chain's streaming state, for a microphone change that keeps
// the same session identifier.
func (s *Service) Reset(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chain == nil {
		return nil
	}
	if s.chain.denoiser != nil {
		s.chain.denoiser.Reset()
	}
	s.chain.agc.Reset()
	s.chain.input = s.chain.input[:0]
	s.chain.agcInput = s.chain.agcInput[:0]
	s.chain.output = s.chain.output[:0]
	return nil
}

// Describe reports what the chain is running, for the App's diagnostics.
func (s *Service) Describe() (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chain == nil {
		return "closed", nil
	}
	if s.chain.denoiser != nil {
		return "dfsmn+agc2", nil
	}
	return "agc2", nil
}
