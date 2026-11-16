// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of modules/audio_processing/agc2/adaptive_digital_gain_controller.cc
// and gain_applier.cc.

package audio

import "math"

// agc2LimiterThresholdForAGCGainDBFS is the limiter level above which the
// adaptive gain stops increasing.
const agc2LimiterThresholdForAGCGainDBFS = -1.0

// dbToRatio converts a gain in dB to a linear factor.
func dbToRatio(db float64) float64 { return math.Pow(10, db/20) }

// computeGainDB returns the gain that brings inputLevelDBFS to -headroomDB,
// clamped to [0, maxGainDB].
func computeGainDB(inputLevelDBFS float64, config AdaptiveDigitalConfig) float64 {
	if inputLevelDBFS < -(config.HeadroomDB + config.MaxGainDB) {
		return config.MaxGainDB
	}
	if inputLevelDBFS < -config.HeadroomDB {
		return -config.HeadroomDB - inputLevelDBFS
	}
	return 0
}

// limitGainByNoise lowers the gain so the output noise stays at or below the
// configured maximum.
func limitGainByNoise(targetGainDB, inputNoiseLevelDBFS, maxOutputNoiseLevelDBFS float64) float64 {
	maxAllowedGainDB := maxOutputNoiseLevelDBFS - inputNoiseLevelDBFS
	return math.Min(targetGainDB, math.Max(maxAllowedGainDB, 0))
}

// limitGainByLowConfidence stops the gain rising while the limiter is active
// and the speech level estimate is not yet trusted.
func limitGainByLowConfidence(targetGainDB, lastGainDB, limiterAudioLevelDBFS float64, estimateIsConfident bool) float64 {
	if estimateIsConfident || limiterAudioLevelDBFS <= agc2LimiterThresholdForAGCGainDBFS {
		return targetGainDB
	}
	limiterLevelBeforeGain := limiterAudioLevelDBFS - lastGainDB
	newTargetGainDB := math.Max(agc2LimiterThresholdForAGCGainDBFS-limiterLevelBeforeGain, 0)
	return math.Min(newTargetGainDB, targetGainDB)
}

// computeGainChangeThisFrameDB bounds how far the gain may move in one frame.
func computeGainChangeThisFrameDB(targetGainDB, lastGainDB float64, gainIncreaseAllowed bool, maxGainDecreaseDB, maxGainIncreaseDB float64) float64 {
	difference := targetGainDB - lastGainDB
	if !gainIncreaseAllowed {
		difference = math.Min(difference, 0)
	}
	return clampFloat(difference, -maxGainDecreaseDB, maxGainIncreaseDB)
}

// gainApplier applies a linear gain to a frame, ramping from the previous
// frame's gain so a change never produces a step discontinuity.
type gainApplier struct {
	lastGainFactor    float64
	currentGainFactor float64
}

func newGainApplier(initialGainFactor float64) *gainApplier {
	return &gainApplier{lastGainFactor: initialGainFactor, currentGainFactor: initialGainFactor}
}

func (applier *gainApplier) reset() {
	applier.lastGainFactor = 1
	applier.currentGainFactor = 1
}

// setGainFactor sets the gain the frame will ramp towards.
func (applier *gainApplier) setGainFactor(factor float64) {
	applier.currentGainFactor = factor
}

// apply ramps the gain across the frame and advances the state.
func (applier *gainApplier) apply(frame []float32) {
	last := applier.lastGainFactor
	end := applier.currentGainFactor
	switch {
	case last == end && math.Abs(end-1) < 1e-6:
		// Unity gain: nothing to do.
	case last == end:
		for index := range frame {
			frame[index] = float32(float64(frame[index]) * end)
		}
	default:
		increment := (end - last) / float64(len(frame))
		gain := last
		for index := range frame {
			frame[index] = float32(float64(frame[index]) * gain)
			gain += increment
		}
	}
	applier.lastGainFactor = end
}

// AdaptiveDigitalGainController applies the adaptive gain law: it targets a
// speech level below full scale, bounded by the noise floor, the limiter state
// and how fast the gain may change.
type AdaptiveDigitalGainController struct {
	config                        AdaptiveDigitalConfig
	adjacentSpeechFramesThreshold int
	maxGainChangeDBPer10ms        float64
	framesToGainIncreaseAllowed   int
	lastGainDB                    float64
	applier                       *gainApplier
}

// NewAdaptiveDigitalGainController prepares a controller with the initial gain.
func NewAdaptiveDigitalGainController(config AdaptiveDigitalConfig) *AdaptiveDigitalGainController {
	return &AdaptiveDigitalGainController{
		config:                        config,
		adjacentSpeechFramesThreshold: agc2AdjacentSpeechFramesThreshold,
		maxGainChangeDBPer10ms:        config.MaxGainChangeDBPerSecond * agc2FrameDurationMs / 1000,
		framesToGainIncreaseAllowed:   agc2AdjacentSpeechFramesThreshold,
		lastGainDB:                    config.InitialGainDB,
		applier:                       newGainApplier(dbToRatio(config.InitialGainDB)),
	}
}

// Reset clears the running gain.
func (controller *AdaptiveDigitalGainController) Reset() {
	controller.framesToGainIncreaseAllowed = controller.adjacentSpeechFramesThreshold
	controller.lastGainDB = controller.config.InitialGainDB
	controller.applier = newGainApplier(dbToRatio(controller.config.InitialGainDB))
}

// frameInfo carries the per-frame measurements the gain law needs.
type frameInfo struct {
	speechProbability   float64
	speechLevelDBFS     float64
	speechLevelReliable bool
	noiseRMSDBFS        float64
	headroomDB          float64
	limiterEnvelopeDBFS float64
}

// Process updates the gain from the frame's measurements and applies it.
func (controller *AdaptiveDigitalGainController) Process(info frameInfo, frame []float32) {
	inputLevelDBFS := info.speechLevelDBFS + info.headroomDB
	targetGainDB := limitGainByLowConfidence(
		limitGainByNoise(
			computeGainDB(inputLevelDBFS, controller.config),
			info.noiseRMSDBFS, controller.config.MaxOutputNoiseLevelDBFS),
		controller.lastGainDB, info.limiterEnvelopeDBFS, info.speechLevelReliable)

	firstConfidentSpeechFrame := false
	if info.speechProbability < agc2VadConfidenceThreshold {
		controller.framesToGainIncreaseAllowed = controller.adjacentSpeechFramesThreshold
	} else if controller.framesToGainIncreaseAllowed > 0 {
		controller.framesToGainIncreaseAllowed--
		firstConfidentSpeechFrame = controller.framesToGainIncreaseAllowed == 0
	}
	gainIncreaseAllowed := controller.framesToGainIncreaseAllowed == 0

	maxGainIncreaseDB := controller.maxGainChangeDBPer10ms
	if firstConfidentSpeechFrame {
		// No increase happened while waiting for a long enough speech run, so
		// allow a proportionally faster catch-up.
		maxGainIncreaseDB *= float64(controller.adjacentSpeechFramesThreshold)
	}

	changeDB := computeGainChangeThisFrameDB(
		targetGainDB, controller.lastGainDB, gainIncreaseAllowed,
		controller.maxGainChangeDBPer10ms, maxGainIncreaseDB)

	if changeDB != 0 {
		controller.applier.setGainFactor(dbToRatio(controller.lastGainDB + changeDB))
	}
	controller.applier.apply(frame)
	controller.lastGainDB += changeDB
}

// LastGainDB returns the gain currently applied.
func (controller *AdaptiveDigitalGainController) LastGainDB() float64 { return controller.lastGainDB }
