// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of modules/audio_processing/agc2/saturation_protector.cc and
// saturation_protector_buffer.cc.

package audio

const (
	// saturationProtectorBufferSize is how many super-frame peaks are delayed.
	saturationProtectorBufferSize = 4
	// peakEnveloperSuperFrameLengthMs is how long peaks are accumulated before
	// one is pushed into the delay buffer.
	peakEnveloperSuperFrameLengthMs = 400
	// saturationMinMarginDB and saturationMaxMarginDB bound the headroom.
	saturationMinMarginDB = 12.0
	saturationMaxMarginDB = 25.0
	// saturationAttack and saturationDecay smooth the headroom.
	saturationAttack = 0.9988493699365052
	saturationDecay  = 0.9997697679981565
	// agc2MinLevelDBFS is the lowest level the AGC2 level math considers.
	agc2MinLevelDBFS = -90.31
)

// peakDelayBuffer is WebRTC's SaturationProtectorBuffer: a fixed-capacity ring
// that reports its oldest value as the front.
type peakDelayBuffer struct {
	buffer []float64
	next   int
	size   int
}

func newPeakDelayBuffer() *peakDelayBuffer {
	return &peakDelayBuffer{buffer: make([]float64, saturationProtectorBufferSize)}
}

func (ring *peakDelayBuffer) reset() {
	ring.next = 0
	ring.size = 0
}

func (ring *peakDelayBuffer) pushBack(value float64) {
	ring.buffer[ring.next] = value
	ring.next++
	if ring.next == len(ring.buffer) {
		ring.next = 0
	}
	if ring.size < len(ring.buffer) {
		ring.size++
	}
}

// front returns the oldest value and whether one exists.
func (ring *peakDelayBuffer) front() (float64, bool) {
	if ring.size == 0 {
		return 0, false
	}
	index := 0
	if ring.size == len(ring.buffer) {
		index = ring.next
	}
	return ring.buffer[index], true
}

// saturationState is one averaging window of the protector.
type saturationState struct {
	headroomDB      float64
	peaks           *peakDelayBuffer
	maxPeaksDBFS    float64
	timeSincePushMs int
}

func newSaturationState(initialHeadroomDB float64) saturationState {
	state := saturationState{peaks: newPeakDelayBuffer()}
	state.reset(initialHeadroomDB)
	return state
}

func (state *saturationState) reset(initialHeadroomDB float64) {
	state.headroomDB = initialHeadroomDB
	state.peaks.reset()
	state.maxPeaksDBFS = agc2MinLevelDBFS
	state.timeSincePushMs = 0
}

// update folds one frame's peak and speech level into the state.
func (state *saturationState) update(peakDBFS, speechLevelDBFS float64) {
	if peakDBFS > state.maxPeaksDBFS {
		state.maxPeaksDBFS = peakDBFS
	}
	state.timeSincePushMs += agc2FrameDurationMs
	if state.timeSincePushMs > peakEnveloperSuperFrameLengthMs {
		state.peaks.pushBack(state.maxPeaksDBFS)
		state.maxPeaksDBFS = agc2MinLevelDBFS
		state.timeSincePushMs = 0
	}
	delayedPeakDBFS := state.maxPeaksDBFS
	if front, ok := state.peaks.front(); ok {
		delayedPeakDBFS = front
	}
	differenceDB := delayedPeakDBFS - speechLevelDBFS
	if differenceDB > state.headroomDB {
		state.headroomDB = state.headroomDB*saturationAttack + differenceDB*(1-saturationAttack)
	} else {
		state.headroomDB = state.headroomDB*saturationDecay + differenceDB*(1-saturationDecay)
	}
	state.headroomDB = clampFloat(state.headroomDB, saturationMinMarginDB, saturationMaxMarginDB)
}

// SaturationProtector recommends headroom from the recent speech peaks so the
// adaptive gain does not push the signal into the limiter.
type SaturationProtector struct {
	initialHeadroomDB             float64
	adjacentSpeechFramesThreshold int
	numAdjacentSpeechFrames       int
	headroomDB                    float64
	preliminaryState              saturationState
	reliableState                 saturationState
}

// NewSaturationProtector prepares a protector with WebRTC's initial headroom.
func NewSaturationProtector(initialHeadroomDB float64) *SaturationProtector {
	protector := &SaturationProtector{
		initialHeadroomDB:             initialHeadroomDB,
		adjacentSpeechFramesThreshold: agc2AdjacentSpeechFramesThreshold,
	}
	protector.preliminaryState = newSaturationState(initialHeadroomDB)
	protector.reliableState = newSaturationState(initialHeadroomDB)
	protector.Reset()
	return protector
}

// Reset clears the running estimate.
func (protector *SaturationProtector) Reset() {
	protector.preliminaryState.reset(protector.initialHeadroomDB)
	protector.reliableState.reset(protector.initialHeadroomDB)
	protector.headroomDB = protector.initialHeadroomDB
	protector.numAdjacentSpeechFrames = 0
}

// Analyze folds one frame into the headroom estimate.
func (protector *SaturationProtector) Analyze(speechProbability, peakDBFS, speechLevelDBFS float64) {
	if speechProbability < agc2VadConfidenceThreshold {
		if protector.adjacentSpeechFramesThreshold > 1 {
			switch {
			case protector.numAdjacentSpeechFrames >= protector.adjacentSpeechFramesThreshold:
				protector.reliableState = protector.preliminaryState
			case protector.numAdjacentSpeechFrames > 0:
				protector.preliminaryState = protector.reliableState
			}
		}
		protector.numAdjacentSpeechFrames = 0
	} else {
		protector.numAdjacentSpeechFrames++
		protector.preliminaryState.update(peakDBFS, speechLevelDBFS)
		if protector.numAdjacentSpeechFrames >= protector.adjacentSpeechFramesThreshold {
			protector.headroomDB = protector.preliminaryState.headroomDB
		}
	}
}

// HeadroomDB returns the recommended headroom.
func (protector *SaturationProtector) HeadroomDB() float64 { return protector.headroomDB }

// clampFloat bounds value to [low, high].
func clampFloat(value, low, high float64) float64 {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}
