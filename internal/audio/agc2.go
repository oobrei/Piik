// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of the AGC2 chain from gain_controller2.cc. The stages run in
// WebRTC's order: speech detection, level analysis, then the gain stages.

package audio

import (
	"math"

	"github.com/TNTcraftHIM/Piik/internal/audio/vad"
)

// AGC2Config configures the chain. The zero value is usable and matches
// WebRTC's defaults.
type AGC2Config struct {
	// AdaptiveDigital holds the gain controller tuning.
	AdaptiveDigital AdaptiveDigitalConfig
	// FixedGainDB is a constant gain applied after the adaptive controller.
	FixedGainDB float64
	// FrameSize is the number of samples per processed frame. WebRTC uses
	// 10 ms frames, which is 480 samples at 48 kHz.
	FrameSize int
}

// DefaultAGC2Config returns WebRTC's defaults.
func DefaultAGC2Config() AGC2Config {
	return AGC2Config{
		AdaptiveDigital: DefaultAdaptiveDigitalConfig(),
		FixedGainDB:     0,
		FrameSize:       SampleRate / 100,
	}
}

// AGC2 is the automatic gain control chain: it estimates the speech and noise
// levels, derives a headroom from the recent peaks, applies an adaptive gain
// bounded by the noise floor, then limits the result.
//
// It buffers input into fixed frames, so callers may pass any number of samples.
type AGC2 struct {
	config     AGC2Config
	noise      *NoiseLevelEstimator
	speech     *SpeechLevelEstimator
	saturation *SaturationProtector
	adaptive   *AdaptiveDigitalGainController
	limiter    *Limiter
	fixedGain  float64

	buffer   []float32
	analysis []float32

	// VAD state: the network, its feature extractor, and the 48->24 kHz
	// decimator that feeds them.
	vad          *vad.Extractor
	network      *vad.Network
	resampler    *sincResampler
	decimated    []float32
	decimatedOut []float64
}

// NewAGC2 prepares the chain.
func NewAGC2(config AGC2Config) *AGC2 {
	if config.FrameSize <= 0 {
		config.FrameSize = SampleRate / 100
	}
	return &AGC2{
		config:     config,
		noise:      NewNoiseLevelEstimator(),
		speech:     NewSpeechLevelEstimator(config.AdaptiveDigital),
		saturation: NewSaturationProtector(agc2SaturationProtectorInitialHeadroomDB),
		adaptive:   NewAdaptiveDigitalGainController(config.AdaptiveDigital),
		limiter:    NewLimiter(),
		fixedGain:  dbToRatio(config.FixedGainDB),
		buffer:     make([]float32, 0, config.FrameSize),
		analysis:   make([]float32, config.FrameSize),

		vad:          vad.NewExtractor(),
		network:      vad.NewNetwork(),
		resampler:    newSincResampler(2, config.FrameSize),
		decimatedOut: make([]float64, config.FrameSize/2),
	}
}

// Name identifies the stage.
func (agc *AGC2) Name() string { return "agc2" }

// Reset clears every component's state.
func (agc *AGC2) Reset() {
	agc.noise.Reset()
	agc.speech.Reset()
	agc.saturation.Reset()
	agc.adaptive.Reset()
	agc.limiter.Reset()
	agc.vad.Reset()
	agc.resampler = newSincResampler(2, agc.config.FrameSize)
	agc.buffer = agc.buffer[:0]
}

// Process buffers in into frames and returns the processed output.
func (agc *AGC2) Process(in, out []float32) ([]float32, error) {
	for _, sample := range in {
		agc.buffer = append(agc.buffer, sample)
		if len(agc.buffer) == agc.config.FrameSize {
			agc.processFrame(agc.buffer)
			out = append(out, agc.buffer...)
			agc.buffer = agc.buffer[:0]
		}
	}
	return out, nil
}

// Flush emits the remainder of a partial frame, processed as its own frame.
func (agc *AGC2) Flush(out []float32) ([]float32, error) {
	if len(agc.buffer) == 0 {
		return out, nil
	}
	frame := make([]float32, len(agc.buffer))
	copy(frame, agc.buffer)
	agc.buffer = agc.buffer[:0]
	agc.processFrame(frame)
	return append(out, frame...), nil
}

// processFrame runs one frame through the chain, in WebRTC's order.
func (agc *AGC2) processFrame(frame []float32) {
	// WebRTC's AGC2 takes the speech probability from the RNN VAD. This port
	// runs the same network, so the level estimators see the same input.
	copy(agc.analysis, frame)
	probability := agc.speechProbability(agc.analysis)

	noiseRMSDBFS := agc.noise.Analyze(frame)

	levels := computeAudioLevels(frame)
	agc.speech.Update(levels.rmsDBFS, probability)
	targetProbability := probability
	if !agc.speech.IsConfident() {
		targetProbability = 0
	}

	agc.saturation.Analyze(targetProbability, levels.peakDBFS, agc.speech.LevelDBFS())
	headroomDB := agc.saturation.HeadroomDB()
	limiterEnvelopeDBFS := floatS16ToDBFS(agc.limiter.LastAudioLevel())

	agc.adaptive.Process(frameInfo{
		speechProbability:   targetProbability,
		speechLevelDBFS:     agc.speech.LevelDBFS(),
		speechLevelReliable: agc.speech.IsConfident(),
		noiseRMSDBFS:        noiseRMSDBFS,
		headroomDB:          headroomDB,
		limiterEnvelopeDBFS: limiterEnvelopeDBFS,
	}, frame)

	if agc.fixedGain != 1 {
		for index := range frame {
			frame[index] = float32(float64(frame[index]) * agc.fixedGain)
		}
	}
	agc.limiter.Process(frame)
}

// speechProbability runs the RNN VAD over the frame. The VAD works at 24 kHz
// with 10 ms frames, so the 48 kHz input is decimated first, exactly as
// WebRTC's rnn_vad_tool does with a PushSincResampler.
func (agc *AGC2) speechProbability(frame []float32) float64 {
	if agc.resampler == nil {
		return 0
	}
	// Decimate 48 kHz to 24 kHz: 480 samples become 240.
	decimated := agc.decimated[:0]
	consumed := 0
	read := func(dst []float64) {
		for index := range dst {
			if consumed < len(frame) {
				dst[index] = float64(frame[consumed])
				consumed++
			} else {
				dst[index] = 0
			}
		}
	}
	agc.resampler.resample(agc.decimatedOut, read)
	for _, value := range agc.decimatedOut {
		decimated = append(decimated, float32(value))
	}
	features, silent := agc.vad.Process(decimated)
	if silent {
		return 0
	}
	return float64(agc.network.Probability(features))
}

// audioLevels holds the per-frame measurements the chain needs.
type audioLevels struct {
	rmsDBFS  float64
	peakDBFS float64
}

// computeAudioLevels returns the RMS and peak of a frame in dBFS.
func computeAudioLevels(frame []float32) audioLevels {
	if len(frame) == 0 {
		return audioLevels{rmsDBFS: agc2MinLevelDBFS, peakDBFS: agc2MinLevelDBFS}
	}
	var sumSquares, peak float64
	for _, sample := range frame {
		value := float64(sample)
		sumSquares += value * value
		if magnitude := math.Abs(value); magnitude > peak {
			peak = magnitude
		}
	}
	rms := math.Sqrt(sumSquares / float64(len(frame)))
	return audioLevels{
		rmsDBFS:  linearToDBFS(rms),
		peakDBFS: linearToDBFS(peak),
	}
}

// linearToDBFS converts a float S16 magnitude to dBFS, floored at the minimum.
func linearToDBFS(value float64) float64 {
	if value <= 0 {
		return agc2MinLevelDBFS
	}
	level := 20 * math.Log10(value/32768)
	if level < agc2MinLevelDBFS {
		return agc2MinLevelDBFS
	}
	return level
}

// floatS16ToDBFS converts a float S16 magnitude to dBFS.
func floatS16ToDBFS(value float64) float64 {
	if value <= 0 {
		return agc2MinLevelDBFS
	}
	return 20 * math.Log10(value/32768)
}
