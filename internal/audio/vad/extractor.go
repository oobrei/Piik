// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file.
//
// Ported from modules/audio_processing/agc2/rnn_vad/features_extraction.cc.

package vad

// Extractor is the full VAD front end: it consumes one 10 ms frame at a time,
// maintains the pitch buffer, and produces the 42-element feature vector.
type Extractor struct {
	pitchBuf   []float32 // BufSize24k samples, oldest first
	lpResidual []float32
	spectral   *SpectralFeatures
	pitch      *PitchEstimator
	plan       *dftPlan
}

// NewExtractor prepares an extractor with a cleared pitch buffer.
func NewExtractor() *Extractor {
	return &Extractor{
		pitchBuf:   make([]float32, BufSize24k),
		lpResidual: make([]float32, BufSize24k),
		spectral:   NewSpectralFeatures(),
		pitch:      NewPitchEstimator(),
		plan:       newDFTPlan(Frame20ms24k),
	}
}

// Reset clears the pitch buffer and spectral history.
func (e *Extractor) Reset() {
	for i := range e.pitchBuf {
		e.pitchBuf[i] = 0
	}
	e.spectral.reset()
	e.pitch.Reset()
}

// Process consumes one 10 ms frame of 24 kHz samples and returns the feature
// vector plus whether silence was detected. On silence the vector is unusable.
func (e *Extractor) Process(frame10ms []float32) ([]float32, bool) {
	// Push: shift left by Frame10ms24k and append the new chunk.
	copy(e.pitchBuf, e.pitchBuf[Frame10ms24k:])
	copy(e.pitchBuf[BufSize24k-Frame10ms24k:], frame10ms)

	// LPC over the whole pitch buffer and the LP residual.
	lpc := lpcCoefficients(e.pitchBuf)
	copy(e.lpResidual, lpResidual(lpc, e.pitchBuf))

	// Pitch (48 kHz) from the LP residual.
	period48k := e.pitch.Estimate(e.lpResidual)

	// Feature 40 is the normalised pitch period.
	vector := make([]float32, FeatureSize)
	vector[FeatureSize-2] = 0.01 * (float32(period48k) - 300)

	// Reference = most recent 480 samples; lagged frame starts at
	// kMaxPitch24kHz - period48k/2.
	reference := e.pitchBuf[BufSize24k-Frame20ms24k:]
	laggedStart := MaxPitch24k - period48k/2
	if laggedStart < 0 {
		laggedStart = 0
	}
	if laggedStart+Frame20ms24k > BufSize24k {
		laggedStart = BufSize24k - Frame20ms24k
	}
	lagged := e.pitchBuf[laggedStart : laggedStart+Frame20ms24k]

	// Spectral features fill bins 0..39 and bin 41 (variability). Only bin 40
	// (the pitch period) is ours to set, so copy everything else.
	spectral, silent := e.spectral.Compute(reference, lagged, e.plan)
	if silent {
		return vector, true
	}
	copy(vector, spectral)
	vector[FeatureSize-2] = 0.01 * (float32(period48k) - 300)
	return vector, false
}
