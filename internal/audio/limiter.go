// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of the WebRTC AGC2 limiter, from limiter.cc,
// interpolated_gain_curve.cc and fixed_digital_level_estimator.cc. The
// algorithm and its constants are WebRTC's; the notice above keeps the origin
// visible.

package audio

import "math"

// Limiter parameters, from agc2_testing_common.h.
const (
	// limiterMaxInputLevelDBFS is the ceiling the curve compresses towards.
	limiterMaxInputLevelDBFS = 1.0
	// limiterKneeSmoothnessDB is the width of the quadratic knee.
	limiterKneeSmoothnessDB = 1.0
	// limiterCompressionRatio is how hard the limiter region compresses.
	limiterCompressionRatio = 5.0
)

// Sample bounds on the float S16 scale the limiter works in.
const (
	minFloatS16 = -32768.0
	maxFloatS16 = 32767.0
	// kMaxInputLevelLinear is where the curve stops interpolating and saturates.
	maxInputLevelLinear = 36706.0
)

// Limiter frame structure, from agc2_common.h and limiter.cc.
const (
	// limiterSubFramesInFrame is how many sub-frames each frame is split into.
	limiterSubFramesInFrame = 20
	// levelDecayFilterConstant smooths the envelope when it falls.
	levelDecayFilterConstant = 0.9971259
	// levelAttackFilterConstant smooths the envelope when it rises.
	levelAttackFilterConstant = 0.0
	// attackFirstSubframeInterpolationPower shapes the attack ramp.
	attackFirstSubframeInterpolationPower = 0.25
)

// Limiter is a soft-knee limiter followed by hard clipping. It splits each
// frame into sub-frames, looks up the gain curve for each sub-frame envelope
// and interpolates the gain per sample, so a change in gain never produces a
// step discontinuity.
type Limiter struct {
	lastScalingFactor float64
	filterStateLevel  float64
	lastAudioLevel    float64
	scalingFactors    [limiterSubFramesInFrame + 1]float64
	perSampleFactors  []float64
}

// NewLimiter prepares a limiter with its gain at unity.
func NewLimiter() *Limiter {
	return &Limiter{lastScalingFactor: 1}
}

// Reset returns the limiter to its initial state.
func (limiter *Limiter) Reset() {
	limiter.lastScalingFactor = 1
	limiter.filterStateLevel = 0
	limiter.lastAudioLevel = 0
}

// LastAudioLevel returns the most recent smoothed envelope, in float S16 units.
func (limiter *Limiter) LastAudioLevel() float64 { return limiter.lastAudioLevel }

// computeLevel returns the smoothed per-sub-frame envelope of a frame.
func (limiter *Limiter) computeLevel(samples []float32) [limiterSubFramesInFrame]float64 {
	var envelope [limiterSubFramesInFrame]float64
	subFrameSize := len(samples) / limiterSubFramesInFrame
	for subFrame := 0; subFrame < limiterSubFramesInFrame; subFrame++ {
		for offset := 0; offset < subFrameSize; offset++ {
			value := math.Abs(float64(samples[subFrame*subFrameSize+offset]))
			if value > envelope[subFrame] {
				envelope[subFrame] = value
			}
		}
	}
	// Make envelope increases happen one step earlier so the corresponding
	// gain decrease is not missed by the interpolation.
	for subFrame := 0; subFrame < limiterSubFramesInFrame-1; subFrame++ {
		if envelope[subFrame] < envelope[subFrame+1] {
			envelope[subFrame] = envelope[subFrame+1]
		}
	}
	// Attack and decay smoothing.
	for subFrame := 0; subFrame < limiterSubFramesInFrame; subFrame++ {
		value := envelope[subFrame]
		if value > limiter.filterStateLevel {
			envelope[subFrame] = value*(1-levelAttackFilterConstant) +
				limiter.filterStateLevel*levelAttackFilterConstant
		} else {
			envelope[subFrame] = value*(1-levelDecayFilterConstant) +
				limiter.filterStateLevel*levelDecayFilterConstant
		}
		limiter.filterStateLevel = envelope[subFrame]
	}
	return envelope
}

// lookUpGainToApply returns the linear gain the curve applies at inputLevel.
func lookUpGainToApply(inputLevel float64) float64 {
	if inputLevel <= gainCurveX[0] {
		return 1
	}
	if inputLevel >= maxInputLevelLinear {
		return 32768 / inputLevel
	}
	// Find the linear piece containing inputLevel.
	index := 0
	for index < len(gainCurveX)-1 && gainCurveX[index+1] <= inputLevel {
		index++
	}
	return gainCurveM[index]*inputLevel + gainCurveQ[index]
}

// Process applies the limiter to samples in place.
func (limiter *Limiter) Process(samples []float32) {
	if len(samples) == 0 {
		return
	}
	envelope := limiter.computeLevel(samples)
	limiter.lastAudioLevel = envelope[limiterSubFramesInFrame-1]

	limiter.scalingFactors[0] = limiter.lastScalingFactor
	for subFrame, level := range envelope {
		limiter.scalingFactors[subFrame+1] = lookUpGainToApply(level)
	}

	if cap(limiter.perSampleFactors) < len(samples) {
		limiter.perSampleFactors = make([]float64, len(samples))
	}
	limiter.perSampleFactors = limiter.perSampleFactors[:len(samples)]
	limiter.computePerSampleFactors()

	for index, value := range samples {
		scaled := float64(value) * limiter.perSampleFactors[index]
		samples[index] = clampFloatS16(scaled)
	}
	limiter.lastScalingFactor = limiter.scalingFactors[limiterSubFramesInFrame]
}

// computePerSampleFactors interpolates the sub-frame gains into one factor per
// sample.
func (limiter *Limiter) computePerSampleFactors() {
	factors := limiter.perSampleFactors
	subFrameSize := len(factors) / limiterSubFramesInFrame

	// The first sub-frame handles an attack differently so the ramp starts at
	// the previous frame's gain.
	isAttack := limiter.scalingFactors[0] > limiter.scalingFactors[1]
	if isAttack {
		last := limiter.scalingFactors[0]
		current := limiter.scalingFactors[1]
		for index := 0; index < subFrameSize; index++ {
			t := float64(index) / float64(subFrameSize)
			factors[index] = math.Pow(1-t, attackFirstSubframeInterpolationPower)*(last-current) + current
		}
	}
	start := 0
	if isAttack {
		start = 1
	}
	for subFrame := start; subFrame < limiterSubFramesInFrame; subFrame++ {
		begin := subFrame * subFrameSize
		from := limiter.scalingFactors[subFrame]
		step := (limiter.scalingFactors[subFrame+1] - from) / float64(subFrameSize)
		for offset := 0; offset < subFrameSize; offset++ {
			factors[begin+offset] = from + step*float64(offset)
		}
	}
}

func clampFloatS16(value float64) float32 {
	if value < minFloatS16 {
		return minFloatS16
	}
	if value > maxFloatS16 {
		return maxFloatS16
	}
	return float32(value)
}
