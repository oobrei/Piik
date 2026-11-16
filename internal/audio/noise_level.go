// Copyright (c) 2016 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of modules/audio_processing/agc2/noise_level_estimator.cc.

package audio

import "math"

const (
	// agc2FramesPerSecond is how many frames the estimator sees per second.
	agc2FramesPerSecond = 100
	// noiseFloorUpdatePeriodNumFrames is how often the floor is updated.
	noiseFloorUpdatePeriodNumFrames = 500
	// noiseFloorMinDbfs is the floor of the dBFS scale the estimator reports.
	noiseFloorMinDbfs = -90.30899869919436
)

// NoiseLevelEstimator tracks the background noise floor in dBFS. It observes
// the minimum frame energy over a period and smooths it: instant decay when the
// floor drops, slow attack when it rises.
type NoiseLevelEstimator struct {
	sampleRateHz   int
	minNoiseEnergy float64
	firstPeriod    bool
	preliminarySet bool
	preliminary    float64
	noiseEnergy    float64
	counter        int
}

// NewNoiseLevelEstimator prepares an estimator for 48 kHz audio.
func NewNoiseLevelEstimator() *NoiseLevelEstimator {
	estimator := &NoiseLevelEstimator{}
	estimator.initialize(SampleRate)
	return estimator
}

func (estimator *NoiseLevelEstimator) initialize(sampleRateHz int) {
	estimator.sampleRateHz = sampleRateHz
	estimator.firstPeriod = true
	estimator.preliminarySet = false
	// The minimum measurable energy corresponds to -84 dBFS.
	estimator.minNoiseEnergy = float64(sampleRateHz) * 2 * 2 / agc2FramesPerSecond
	estimator.preliminary = estimator.minNoiseEnergy
	estimator.noiseEnergy = estimator.minNoiseEnergy
	estimator.counter = noiseFloorUpdatePeriodNumFrames
}

// Reset clears the running estimate.
func (estimator *NoiseLevelEstimator) Reset() {
	estimator.initialize(estimator.sampleRateHz)
}

// Analyze updates the noise floor from one frame and returns it in dBFS.
func (estimator *NoiseLevelEstimator) Analyze(frame []float32) float64 {
	sampleRateHz := len(frame) * agc2FramesPerSecond
	if sampleRateHz != estimator.sampleRateHz {
		estimator.initialize(sampleRateHz)
	}
	frameEnergy := frameEnergyOf(frame)
	if frameEnergy <= estimator.minNoiseEnergy {
		return energyToDbfs(estimator.noiseEnergy, len(frame))
	}
	if estimator.preliminarySet {
		estimator.preliminary = math.Min(estimator.preliminary, frameEnergy)
	} else {
		estimator.preliminary = frameEnergy
		estimator.preliminarySet = true
	}
	switch {
	case estimator.counter == 0:
		estimator.firstPeriod = false
		estimator.noiseEnergy = smoothNoiseFloorEstimate(estimator.noiseEnergy, estimator.preliminary)
		estimator.counter = noiseFloorUpdatePeriodNumFrames
		estimator.preliminarySet = false
	case estimator.firstPeriod:
		// During the initial period the estimate is monotonic.
		estimator.noiseEnergy = estimator.preliminary
		estimator.counter--
	default:
		// Afterwards the estimate may only fall within an observation period.
		estimator.noiseEnergy = math.Min(estimator.noiseEnergy, estimator.preliminary)
		estimator.counter--
	}
	return energyToDbfs(estimator.noiseEnergy, len(frame))
}

// frameEnergyOf returns the largest per-channel sum of squares.
func frameEnergyOf(frame []float32) float64 {
	var energy float64
	for _, value := range frame {
		energy += float64(value) * float64(value)
	}
	return energy
}

// energyToDbfs converts a summed frame energy to dBFS.
func energyToDbfs(signalEnergy float64, numSamples int) float64 {
	rmsSquare := signalEnergy / float64(numSamples)
	if rmsSquare <= 1 {
		return noiseFloorMinDbfs
	}
	return 10*math.Log10(rmsSquare) + noiseFloorMinDbfs
}

// smoothNoiseFloorEstimate applies instant decay and slow attack: the floor
// drops immediately but rises gradually, so music or fast speech cannot
// overestimate it quickly.
func smoothNoiseFloorEstimate(current, next float64) float64 {
	const attack = 0.5
	if current < next {
		return attack*next + (1-attack)*current
	}
	return next
}
