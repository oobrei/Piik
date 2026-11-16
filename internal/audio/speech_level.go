// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of modules/audio_processing/agc2/speech_level_estimator_impl.cc.

package audio

// Shared AGC2 constants, from agc2_common.h.
const (
	// agc2FrameDurationMs is the duration of one processed frame.
	agc2FrameDurationMs = 10
	// agc2VadConfidenceThreshold is the speech probability above which a frame
	// counts as speech.
	agc2VadConfidenceThreshold = 0.95
	// agc2AdjacentSpeechFramesThreshold is how many adjacent speech frames are
	// needed before a level update is trusted.
	agc2AdjacentSpeechFramesThreshold = 12
	// agc2LevelEstimatorTimeToConfidenceMs is how long the estimator observes
	// speech before it stops treating the average as preliminary.
	agc2LevelEstimatorTimeToConfidenceMs = 400.0
	// agc2LevelEstimatorLeakFactor decays the running average once the buffer
	// is full, so the estimate tracks slow changes.
	agc2LevelEstimatorLeakFactor = 1.0 - 1.0/agc2LevelEstimatorTimeToConfidenceMs
	// agc2SaturationProtectorInitialHeadroomDB is the headroom assumed before
	// the saturation protector has seen anything.
	agc2SaturationProtectorInitialHeadroomDB = 20.0
)

// AdaptiveDigitalConfig holds the gain controller tuning, matching
// AudioProcessing::Config::GainController2::AdaptiveDigital.
type AdaptiveDigitalConfig struct {
	HeadroomDB               float64
	MaxGainDB                float64
	InitialGainDB            float64
	MaxGainChangeDBPerSecond float64
	MaxOutputNoiseLevelDBFS  float64
}

// DefaultAdaptiveDigitalConfig returns WebRTC's defaults.
func DefaultAdaptiveDigitalConfig() AdaptiveDigitalConfig {
	return AdaptiveDigitalConfig{
		HeadroomDB:               5,
		MaxGainDB:                50,
		InitialGainDB:            15,
		MaxGainChangeDBPerSecond: 6,
		MaxOutputNoiseLevelDBFS:  -50,
	}
}

// clampLevelEstimateDBFS bounds a level estimate to the range WebRTC allows.
func clampLevelEstimateDBFS(levelDBFS float64) float64 {
	if levelDBFS < -90 {
		return -90
	}
	if levelDBFS > 30 {
		return 30
	}
	return levelDBFS
}

// levelRatio is a weighted average kept as a numerator and denominator so new
// observations can be folded in without storing a history.
type levelRatio struct {
	numerator   float64
	denominator float64
}

func (ratio levelRatio) value() float64 { return ratio.numerator / ratio.denominator }

// levelEstimatorState is one averaging window: a running level plus how long
// until it stops being preliminary.
type levelEstimatorState struct {
	timeToConfidenceMs float64
	levelDBFS          levelRatio
}

// SpeechLevelEstimator tracks the speech level in dBFS. It averages frames
// weighted by speech probability and only trusts the average once enough
// adjacent speech frames have been seen.
type SpeechLevelEstimator struct {
	initialSpeechLevelDBFS        float64
	adjacentSpeechFramesThreshold int
	levelDBFS                     float64
	isConfident                   bool
	numAdjacentSpeechFrames       int
	preliminaryState              levelEstimatorState
	reliableState                 levelEstimatorState
}

// NewSpeechLevelEstimator prepares an estimator with WebRTC's initial level.
func NewSpeechLevelEstimator(config AdaptiveDigitalConfig) *SpeechLevelEstimator {
	estimator := &SpeechLevelEstimator{
		initialSpeechLevelDBFS: clampLevelEstimateDBFS(
			-agc2SaturationProtectorInitialHeadroomDB - config.InitialGainDB - config.HeadroomDB),
		adjacentSpeechFramesThreshold: agc2AdjacentSpeechFramesThreshold,
		isConfident:                   false,
	}
	estimator.Reset()
	return estimator
}

// Reset clears the running estimate.
func (estimator *SpeechLevelEstimator) Reset() {
	estimator.resetState(&estimator.preliminaryState)
	estimator.resetState(&estimator.reliableState)
	estimator.levelDBFS = estimator.initialSpeechLevelDBFS
	estimator.numAdjacentSpeechFrames = 0
}

func (estimator *SpeechLevelEstimator) resetState(state *levelEstimatorState) {
	state.timeToConfidenceMs = agc2LevelEstimatorTimeToConfidenceMs
	state.levelDBFS.numerator = estimator.initialSpeechLevelDBFS
	state.levelDBFS.denominator = 1
}

// Update folds one frame into the estimate.
func (estimator *SpeechLevelEstimator) Update(rmsDBFS, speechProbability float64) {
	if speechProbability < agc2VadConfidenceThreshold {
		if estimator.adjacentSpeechFramesThreshold > 1 {
			switch {
			case estimator.numAdjacentSpeechFrames >= estimator.adjacentSpeechFramesThreshold:
				// A long enough run of speech ended: promote it.
				estimator.reliableState = estimator.preliminaryState
			case estimator.numAdjacentSpeechFrames > 0:
				// Too short a run: discard it and fall back to the last
				// reliable state.
				estimator.preliminaryState = estimator.reliableState
			}
		}
		estimator.numAdjacentSpeechFrames = 0
	} else {
		estimator.numAdjacentSpeechFrames++
		bufferFull := estimator.preliminaryState.timeToConfidenceMs == 0
		if !bufferFull {
			estimator.preliminaryState.timeToConfidenceMs -= agc2FrameDurationMs
		}
		leak := 1.0
		if bufferFull {
			leak = agc2LevelEstimatorLeakFactor
		}
		estimator.preliminaryState.levelDBFS.numerator =
			estimator.preliminaryState.levelDBFS.numerator*leak + rmsDBFS*speechProbability
		estimator.preliminaryState.levelDBFS.denominator =
			estimator.preliminaryState.levelDBFS.denominator*leak + speechProbability
		level := estimator.preliminaryState.levelDBFS.value()
		if estimator.numAdjacentSpeechFrames >= estimator.adjacentSpeechFramesThreshold {
			estimator.levelDBFS = clampLevelEstimateDBFS(level)
		}
	}
	estimator.isConfident =
		estimator.numAdjacentSpeechFrames >= estimator.adjacentSpeechFramesThreshold
}

// LevelDBFS returns the current speech level estimate.
func (estimator *SpeechLevelEstimator) LevelDBFS() float64 { return estimator.levelDBFS }

// IsConfident reports whether enough adjacent speech has been observed for the
// estimate to be trusted.
func (estimator *SpeechLevelEstimator) IsConfident() bool { return estimator.isConfident }
