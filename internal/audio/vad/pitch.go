// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Ported from modules/audio_processing/agc2/rnn_vad/: lp_residual.cc,
// pitch_search.cc, pitch_search_internal.cc and auto_correlation.cc.

package vad

// Pitch buffer geometry, from common.h.
const (
	// SampleRate24k is the rate the pitch buffer holds.
	SampleRate24k = 24000
	// MinPitch24k is the shortest pitch period the search allows.
	MinPitch24k = SampleRate24k / 800 // 30
	// MaxPitch24k is the longest pitch period the search allows.
	MaxPitch24k = 384
	// BufSize24k is the pitch buffer length.
	BufSize24k = MaxPitch24k + Frame20ms24k // 864
	// initialMinPitch24k skips very short periods in the initial search.
	initialMinPitch24k = 3 * MinPitch24k // 90
	// InitialNumLags24k is the number of lags in the initial search.
	InitialNumLags24k = MaxPitch24k - initialMinPitch24k // 294
	// RefineNumLags24k is the number of lags in the refinement.
	RefineNumLags24k = MaxPitch24k + 1 // 385
	// BufSize12k is the decimated pitch buffer length.
	BufSize12k = BufSize24k / 2
	// MaxPitch12k is the longest pitch period at 12 kHz.
	MaxPitch12k = MaxPitch24k / 2
	// initialMinPitch12k skips very short periods at 12 kHz.
	initialMinPitch12k = initialMinPitch24k / 2
	// NumLags12k is the number of lags in the 12 kHz search.
	NumLags12k = MaxPitch12k - initialMinPitch12k
	// MaxPitch48k is the longest pitch period the estimator reports.
	MaxPitch48k = MaxPitch24k * 2
	// MinPitch48k is the shortest pitch period the estimator reports.
	MinPitch48k = MinPitch24k * 2
	// numLpcCoefficients is the LPC order.
	numLpcCoefficients = 5
)

// subHarmonicMultipliers tells the extended search where to look for a
// sub-harmonic of a candidate period.
var subHarmonicMultipliers = [14]int{3, 2, 3, 2, 5, 2, 3, 2, 3, 2, 5, 2, 3, 2}

// initialPitchPeriodThresholds widens the acceptance window for candidates
// near the previously estimated period.
var initialPitchPeriodThresholds = [14]int{20, 45, 80, 125, 180, 245, 320, 405, 500, 605, 720, 845, 980, 1125}

// lpcCoefficients computes the post-processed LPC coefficients of x.
func lpcCoefficients(x []float32) []float32 {
	autoCorr := make([]float32, numLpcCoefficients)
	for lag := 0; lag < numLpcCoefficients; lag++ {
		var sum float32
		for index := 0; index+lag < len(x); index++ {
			sum += x[index] * x[index+lag]
		}
		autoCorr[lag] = sum
	}
	out := make([]float32, numLpcCoefficients)
	if autoCorr[0] == 0 {
		return out
	}
	// DenoiseAutoCorrelation: assume a -40 dB white noise floor.
	autoCorr[0] *= 1.0001
	autoCorr[1] -= autoCorr[1] * 0.000064
	autoCorr[2] -= autoCorr[2] * 0.000256
	autoCorr[3] -= autoCorr[3] * 0.000576
	autoCorr[4] -= autoCorr[4] * 0.001024

	pre := make([]float32, numLpcCoefficients-1)
	errorValue := autoCorr[0]
	for i := 0; i < numLpcCoefficients-1; i++ {
		var reflection float32
		for j := 0; j < i; j++ {
			reflection += pre[j] * autoCorr[i-j]
		}
		reflection += autoCorr[i+1]
		if abs32(errorValue) < 1e-6 {
			errorValue = copysign32(1e-6, errorValue)
		}
		reflection /= -errorValue
		pre[i] = reflection
		for j := 0; j < (i+1)>>1; j++ {
			tmp1, tmp2 := pre[j], pre[i-1-j]
			pre[j] = tmp1 + reflection*tmp2
			pre[i-1-j] = tmp2 + reflection*tmp1
		}
		errorValue -= reflection * reflection * errorValue
		if errorValue < 0.001*autoCorr[0] {
			break
		}
	}
	pre[0] *= 0.9
	pre[1] *= 0.9 * 0.9
	pre[2] *= 0.9 * 0.9 * 0.9
	pre[3] *= 0.9 * 0.9 * 0.9 * 0.9
	const c = 0.8
	out[0] = pre[0] + c
	out[1] = pre[1] + c*pre[0]
	out[2] = pre[2] + c*pre[1]
	out[3] = pre[3] + c*pre[2]
	out[4] = c * pre[3]
	return out
}

// lpResidual applies the LPC inverse filter to x.
func lpResidual(lpc, x []float32) []float32 {
	y := make([]float32, len(x))
	y[0] = x[0]
	for i := 1; i < numLpcCoefficients && i < len(y); i++ {
		sum := x[i]
		for j := 0; j < i; j++ {
			sum += x[i-1-j] * lpc[j]
		}
		y[i] = sum
	}
	for i := numLpcCoefficients; i < len(y); i++ {
		sum := x[i]
		for j := 0; j < numLpcCoefficients; j++ {
			sum += x[i-1-j] * lpc[j]
		}
		y[i] = sum
	}
	return y
}

// decimate2x halves the sample rate by taking every other sample.
func decimate2x(src []float32) []float32 {
	out := make([]float32, BufSize12k)
	for index := range out {
		out[index] = src[2*index]
	}
	return out
}

// autoCorrelation12k correlates the decimated pitch buffer against its last
// frame, matching AutoCorrelationCalculator::ComputeOnPitchBuffer.
func autoCorrelation12k(buffer []float32) []float32 {
	const convolutionLength = BufSize12k - MaxPitch12k
	reference := buffer[len(buffer)-convolutionLength:]
	out := make([]float32, NumLags12k)
	for invertedLag := 0; invertedLag < NumLags12k; invertedLag++ {
		var sum float32
		for index := 0; index < convolutionLength; index++ {
			sum += buffer[invertedLag+index] * reference[index]
		}
		out[invertedLag] = sum
	}
	return out
}

// autoCorrelation24k correlates a fixed reference frame against a window
// sliding from the inverted lag, so the peak sits at the maximum pitch period.
func autoCorrelation24k(invertedLag int, buffer []float32) float32 {
	var sum float32
	for index := 0; index < Frame20ms24k; index++ {
		sum += buffer[invertedLag+index] * buffer[MaxPitch24k+index]
	}
	return sum
}

// pitchCandidate carries a period encoded as an inverted lag plus its strength
// as a numerator and denominator, so comparisons avoid division.
type pitchCandidate struct {
	invertedLag int
	numerator   float32
	denominator float32
}

func (candidate pitchCandidate) strongerThan(other pitchCandidate) bool {
	return candidate.numerator*other.denominator > other.numerator*candidate.denominator
}

// pitchPeriod12k returns the best and second best inverted lags at 12 kHz.
func pitchPeriod12k(buffer, autoCorrelation []float32) (int, int) {
	denominator := float32(1.0)
	for index := 0; index < Frame20ms24k/2+1; index++ {
		denominator += buffer[index] * buffer[index]
	}
	best := pitchCandidate{invertedLag: 0, numerator: -1}
	second := pitchCandidate{invertedLag: 1, numerator: -1}
	for invertedLag := 0; invertedLag < NumLags12k; invertedLag++ {
		if autoCorrelation[invertedLag] > 0 {
			candidate := pitchCandidate{
				invertedLag: invertedLag,
				numerator:   autoCorrelation[invertedLag] * autoCorrelation[invertedLag],
				denominator: denominator,
			}
			if candidate.strongerThan(second) {
				if candidate.strongerThan(best) {
					second = best
					best = candidate
				} else {
					second = candidate
				}
			}
		}
		yOld := buffer[invertedLag]
		yNew := buffer[invertedLag+Frame20ms24k/2]
		denominator -= yOld * yOld
		denominator += yNew * yNew
		if denominator < 0 {
			denominator = 0
		}
	}
	return best.invertedLag, second.invertedLag
}

// slidingFrameEnergies24k pre-computes the energy of every sliding frame.
func slidingFrameEnergies24k(buffer []float32) []float32 {
	energies := make([]float32, RefineNumLags24k)
	var energy float32
	for index := 0; index < Frame20ms24k; index++ {
		energy += buffer[index] * buffer[index]
	}
	energies[0] = energy
	for invertedLag := 0; invertedLag < MaxPitch24k; invertedLag++ {
		energy -= buffer[invertedLag] * buffer[invertedLag]
		energy += buffer[invertedLag+Frame20ms24k] * buffer[invertedLag+Frame20ms24k]
		if energy < 1 {
			energy = 1
		}
		energies[invertedLag+1] = energy
	}
	return energies
}

// pseudoInterpolationOffset returns a lag correction of -1, 0 or 1.
func pseudoInterpolationOffset(prev, curr, next float32) int {
	if next-prev > 0.7*(curr-prev) {
		return 1
	}
	if prev-next > 0.7*(curr-next) {
		return -1
	}
	return 0
}

// createInvertedLagRange returns the clipped +/-2 neighbourhood around a
// candidate, matching WebRTC's CreateInvertedLagRange.
func createInvertedLagRange(invertedLag int) (int, int) {
	min := invertedLag - 2
	if min < 0 {
		min = 0
	}
	max := invertedLag + 2
	if max > InitialNumLags24k-1 {
		max = InitialNumLags24k - 1
	}
	return min, max
}

// pitchPeriod48k refines the 12 kHz candidates at 24 kHz, searching the
// neighbourhood of each candidate as WebRTC does.
func pitchPeriod48k(buffer []float32, energies []float32, candidates []int) int {
	invertedLags := make([]int, 0, 10)
	for _, c := range candidates {
		lo, hi := createInvertedLagRange(c)
		for lag := lo; lag <= hi; lag++ {
			invertedLags = append(invertedLags, lag)
		}
	}
	autoCorrelation := make([]float32, InitialNumLags24k)
	for invertedLag := 0; invertedLag < InitialNumLags24k; invertedLag++ {
		autoCorrelation[invertedLag] = autoCorrelation24k(invertedLag, buffer)
	}
	bestLag := 0
	bestNumerator := float32(-1.0)
	bestDenominator := float32(0.0)
	for _, invertedLag := range invertedLags {
		if autoCorrelation[invertedLag] > 0 {
			numerator := autoCorrelation[invertedLag] * autoCorrelation[invertedLag]
			denominator := energies[invertedLag]
			if numerator*bestDenominator > bestNumerator*denominator {
				bestLag = invertedLag
				bestNumerator = numerator
				bestDenominator = denominator
			}
		}
	}
	if bestLag == 0 || bestLag >= InitialNumLags24k-1 {
		return bestLag * 2
	}
	offset := pseudoInterpolationOffset(
		autoCorrelation[bestLag+1], autoCorrelation[bestLag], autoCorrelation[bestLag-1])
	return 2*bestLag + offset
}

// alternativePitchPeriod rounds multiplier*period/divisor.
func alternativePitchPeriod(period, multiplier, divisor int) int {
	return (2*multiplier*period + divisor) / (2 * divisor)
}

// alternativeStronger reports whether an alternative period beats the initial
// estimate, allowing for pitch tracking against the last period.
func alternativeStronger(last, initial, alternative [2]float32, periodDivisor int) bool {
	lowerThreshold := float32(0.0)
	distance := abs32(alternative[0] - last[0])
	if distance <= 1 {
		lowerThreshold = last[1]
	} else if distance == 2 && initial[0] > float32(initialPitchPeriodThresholds[periodDivisor-2]) {
		lowerThreshold = 0.5 * last[1]
	}
	threshold := max32(float32(0.3), 0.7*initial[1]-lowerThreshold)
	if alternative[0] < 3*MinPitch24k {
		threshold = max32(float32(0.4), 0.85*initial[1]-lowerThreshold)
	} else if alternative[0] < 2*MinPitch24k {
		threshold = max32(float32(0.5), 0.9*initial[1]-lowerThreshold)
	}
	return alternative[1] > threshold
}

// extendedPitchPeriod48k searches for a stronger period around the initial one.
func extendedPitchPeriod48k(buffer, energies []float32, initialPeriod48k int, lastPeriod, lastStrength float32) (int, float32) {
	xEnergy := energies[MaxPitch24k]
	strength := func(xy, yEnergy float32) float32 {
		return xy / sqrt32(1+xEnergy*yEnergy)
	}
	bestPeriod := initialPeriod48k / 2
	if bestPeriod > MaxPitch24k-1 {
		bestPeriod = MaxPitch24k - 1
	}
	bestXY := autoCorrelation24k(MaxPitch24k-bestPeriod, buffer)
	bestYEnergy := energies[MaxPitch24k-bestPeriod]
	bestStrength := strength(bestXY, bestYEnergy)
	initial := [2]float32{float32(bestPeriod), bestStrength}
	last := [2]float32{float32(lastPeriod / 2), lastStrength}

	// WebRTC derives every alternative from the FIXED initial period, not from
	// the best candidate found so far. Using the mutable best makes the search
	// cascade into sub-harmonics and produces octave errors.
	initialPeriod := bestPeriod
	maxDivisor := (2 * initialPeriod) / (2*MinPitch24k - 1)
	for divisor := 2; divisor <= maxDivisor; divisor++ {
		period := alternativePitchPeriod(initialPeriod, 1, divisor)
		dual := alternativePitchPeriod(initialPeriod, subHarmonicMultipliers[divisor-2], divisor)
		if divisor == 2 && dual > MaxPitch24k {
			dual = initialPeriod
		}
		xy := 0.5 * (autoCorrelation24k(MaxPitch24k-period, buffer) +
			autoCorrelation24k(MaxPitch24k-dual, buffer))
		yy := 0.5 * (energies[MaxPitch24k-period] + energies[MaxPitch24k-dual])
		alternative := [2]float32{float32(period), strength(xy, yy)}
		if alternativeStronger(last, initial, alternative, divisor) {
			bestPeriod = period
			bestStrength = alternative[1]
			bestXY = xy
			bestYEnergy = yy
		}
	}

	// Final strength and period, from the tail of the WebRTC function.
	if bestXY < 0 {
		bestXY = 0
	}
	finalStrength := bestStrength
	if bestYEnergy <= bestXY {
		finalStrength = 1
	} else {
		finalStrength = bestXY / (bestYEnergy + 1)
	}
	// WebRTC takes the MINIMUM of the candidate strength and the energy-based
	// strength, not the maximum.
	if bestStrength < finalStrength {
		finalStrength = bestStrength
	}
	// Pseudo-interpolation refines the period at twice the sample rate.
	finalPeriod := pseudoInterpolateLagPitchBuffer(bestPeriod, buffer)
	if finalPeriod < MinPitch48k {
		finalPeriod = MinPitch48k
	}
	return finalPeriod, finalStrength
}

// pseudoInterpolateLagPitchBuffer refines a 24 kHz lag to 48 kHz using the
// neighbouring auto-correlations, matching WebRTC's
// PitchPseudoInterpolationLagPitchBuf.
func pseudoInterpolateLagPitchBuffer(lag int, buffer []float32) int {
	period := 2 * lag
	if lag > 0 && lag < MaxPitch24k {
		inv := MaxPitch24k - lag
		offset := pseudoInterpolationOffset(
			autoCorrelation24k(inv+1, buffer),
			autoCorrelation24k(inv, buffer),
			autoCorrelation24k(inv-1, buffer))
		period += offset
	}
	return period
}

// PitchEstimator estimates pitch across a stream, carrying the last period so
// the extended search can track it and avoid octave errors.
type PitchEstimator struct {
	lastPeriod   int
	lastStrength float32
}

// NewPitchEstimator prepares an estimator with no prior pitch.
func NewPitchEstimator() *PitchEstimator { return &PitchEstimator{} }

// Reset clears the pitch tracking state.
func (estimator *PitchEstimator) Reset() {
	estimator.lastPeriod = 0
	estimator.lastStrength = 0
}

// Estimate returns the pitch period at 48 kHz for a 24 kHz pitch buffer.
func (estimator *PitchEstimator) Estimate(buffer []float32) int {
	decimated := decimate2x(buffer)
	autoCorrelation := autoCorrelation12k(decimated)
	best, second := pitchPeriod12k(decimated, autoCorrelation)
	best *= 2
	second *= 2
	energies := slidingFrameEnergies24k(buffer)
	period48k := pitchPeriod48k(buffer, energies, []int{best, second})
	initialPeriod := MaxPitch48k - period48k
	extended, strength := extendedPitchPeriod48k(
		buffer, energies, initialPeriod,
		float32(estimator.lastPeriod), estimator.lastStrength)
	estimator.lastPeriod = extended
	estimator.lastStrength = strength
	return extended
}

// EstimatePitch returns the pitch for a one-off buffer with no tracking state.
func EstimatePitch(buffer []float32) int { return NewPitchEstimator().Estimate(buffer) }
