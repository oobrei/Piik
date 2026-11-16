// Copyright (c) 2018 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Ported from modules/audio_processing/agc2/rnn_vad/spectral_features.cc and
// spectral_features_internal.cc.

package vad

import "math"

// Frame sizes and band counts, from common.h.
const (
	// Frame10ms24k is the analysis frame the extractor consumes.
	Frame10ms24k = 240
	// Frame20ms24k is the window the spectral features are computed over.
	Frame20ms24k = 480
	// NumBands is the number of cepstral bands.
	NumBands = 22
	// NumLowerBands is the number of bands kept in the derivative features.
	NumLowerBands = 6
	// CepstralHistory is how many cepstra the derivative features look back.
	CepstralHistory = 8
	// OpusBands24k is the number of Opus bands at 24 kHz.
	OpusBands24k = 20
	// silenceThreshold gates the extractor before the network runs.
	silenceThreshold = 0.04
	// dctScaling is WebRTC's precomputed sqrt(2/NumBands).
	dctScaling = 0.301511345
)

// opusScaleNumBins is how many FFT coefficients each Opus band covers.
var opusScaleNumBins = [OpusBands24k - 1]int{4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 24, 24, 32, 48}

// opusBandWeights is the triangular filter weight for each FFT coefficient.
// WebRTC hard-codes the table; it is generated here from the same rule, which
// is band j/n for the jth of n coefficients.
var opusBandWeights = buildBandWeights()

func buildBandWeights() []float32 {
	weights := make([]float32, 0, Frame20ms24k/2)
	for _, size := range opusScaleNumBins {
		for index := 0; index < size; index++ {
			weights = append(weights, float32(index)/float32(size))
		}
	}
	return weights
}

// halfWindow is the first half of WebRTC's Vorbis window, scaled by 1/480.
var halfWindow = buildHalfWindow()

func buildHalfWindow() []float32 {
	const half = Frame20ms24k / 2
	const scaling = 1.0 / Frame20ms24k
	window := make([]float32, half)
	for index := 0; index < half; index++ {
		sine := sin32(0.5 * pi32 * (float32(index) + 0.5) / half)
		window[index] = scaling * sin32(0.5*pi32*sine*sine)
	}
	return window
}

// dctTable is WebRTC's 22x22 DCT table.
var dctTable = buildDctTable()

func buildDctTable() []float32 {
	table := make([]float32, NumBands*NumBands)
	const scale = math.Sqrt2 / 2
	for i := 0; i < NumBands; i++ {
		for j := 0; j < NumBands; j++ {
			table[i*NumBands+j] = cos32((float32(i) + 0.5) * float32(j) * pi32 / NumBands)
		}
		table[i*NumBands] *= scale
	}
	return table
}

// windowedSpectrum applies the mirrored Vorbis window and returns the
// interleaved real/imaginary spectrum with the Nyquist coefficient zeroed.
func windowedSpectrum(frame []float32, plan *dftPlan) []float32 {
	windowed := make([]float32, Frame20ms24k)
	const half = Frame20ms24k / 2
	for index := 0; index < half; index++ {
		mirror := Frame20ms24k - 1 - index
		windowed[index] = frame[index] * halfWindow[index]
		windowed[mirror] = frame[mirror] * halfWindow[index]
	}
	spectrum := plan.forward(windowed)
	// WebRTC stores 240 coefficient pairs and zeroes the Nyquist term.
	spectrum[1] = 0
	return spectrum
}

// crossCorrelation accumulates the Opus band energies of two spectra.
func crossCorrelation(x, y []float32) []float32 {
	out := make([]float32, OpusBands24k)
	coefficient := 0
	for band := 0; band < OpusBands24k-1; band++ {
		for index := 0; index < opusScaleNumBins[band]; index++ {
			value := x[2*coefficient]*y[2*coefficient] + x[2*coefficient+1]*y[2*coefficient+1]
			part := opusBandWeights[coefficient] * value
			out[band] += value - part
			out[band+1] += part
			coefficient++
		}
	}
	out[0] *= 2
	return out
}

// smoothedLogMagnitude applies WebRTC's band smoothing.
func smoothedLogMagnitude(bandsEnergy []float32) []float32 {
	logBands := make([]float32, NumBands)
	logMax, follow := float32(-2.0), float32(-2.0)
	smooth := func(x float32) float32 {
		x = max32(logMax-7, max32(follow-1.5, x))
		logMax = max32(logMax, x)
		follow = max32(follow-1.5, x)
		return x
	}
	for index := 0; index < len(bandsEnergy); index++ {
		logBands[index] = smooth(log10_32(1e-2 + bandsEnergy[index]))
	}
	for index := len(bandsEnergy); index < NumBands; index++ {
		logBands[index] = smooth(-2)
	}
	return logBands
}

// dct applies WebRTC's DCT to the log band energies.
// dct is WebRTC's ComputeDct. It writes as many coefficients as the caller asks
// for, so the same table serves both the 22-band cepstrum and the 6-band
// cross-correlation cepstrum.
func dct(values []float32, outSize int) []float32 {
	out := make([]float32, outSize)
	for i := 0; i < outSize; i++ {
		var total float32
		for j := 0; j < len(values); j++ {
			total += values[j] * dctTable[j*NumBands+i]
		}
		out[i] = total * dctScaling
	}
	return out
}

// SpectralFeatures holds the cepstral history the derivative features need.
type SpectralFeatures struct {
	ring [][]float32
	// distances is WebRTC's SymmetricMatrixBuffer: the upper triangle of the
	// (CepstralHistory-1) x CepstralHistory comparison matrix, packed by rows.
	distances []float32
}

// pushDistances mirrors SymmetricMatrixBuffer::Push: shift the lower-right
// (S-2)x(S-2) sub-matrix up and left, then write the newest comparisons into
// the last column.
func (features *SpectralFeatures) pushDistances(values []float32) {
	size := CepstralHistory
	copy(features.distances, features.distances[size:])
	for index := 0; index < len(values); index++ {
		position := (size-1-index)*(size-1) - 1
		features.distances[position] = values[index]
	}
}

// distanceAt mirrors SymmetricMatrixBuffer::GetValue.
func (features *SpectralFeatures) distanceAt(delay1, delay2 int) float32 {
	size := CepstralHistory
	row := size - 1 - delay1
	col := size - 1 - delay2
	if row > col {
		row, col = col, row
	}
	return features.distances[row*(size-1)+(col-1)]
}

// NewSpectralFeatures prepares an extractor with an empty history.
func NewSpectralFeatures() *SpectralFeatures {
	features := &SpectralFeatures{}
	features.reset()
	return features
}

func (features *SpectralFeatures) reset() {
	features.ring = make([][]float32, CepstralHistory)
	for index := range features.ring {
		features.ring[index] = make([]float32, NumBands)
	}
	features.distances = make([]float32, CepstralHistory*CepstralHistory)
}

// Compute returns the 42-element feature vector and whether silence was
// detected. A silent frame leaves the vector zeroed, as WebRTC does.
func (features *SpectralFeatures) Compute(reference, lagged []float32, plan *dftPlan) ([]float32, bool) {
	vector := make([]float32, FeatureSize)
	referenceFFT := windowedSpectrum(reference, plan)
	referenceEnergy := crossCorrelation(referenceFFT, referenceFFT)
	var total float32
	for _, value := range referenceEnergy {
		total += value
	}
	if total < silenceThreshold {
		return vector, true
	}
	laggedFFT := windowedSpectrum(lagged, plan)
	laggedEnergy := crossCorrelation(laggedFFT, laggedFFT)

	cepstrum := dct(smoothedLogMagnitude(referenceEnergy), NumBands)
	cepstrum[0] -= 12
	cepstrum[1] -= 4

	// Push the new cepstrum and update the difference statistics. The shift
	// must run backwards: copying forwards overwrites the entries it is about
	// to read, which duplicates the newest cepstrum into every older slot.
	for index := CepstralHistory - 1; index > 0; index-- {
		copy(features.ring[index], features.ring[index-1])
	}
	copy(features.ring[0], cepstrum)
	// Push the new distances into WebRTC's packed symmetric matrix buffer.
	var distances [CepstralHistory - 1]float32
	for index := 0; index < CepstralHistory-1; index++ {
		var distance float32
		for band := 0; band < NumBands; band++ {
			diff := cepstrum[band] - features.ring[index+1][band]
			distance += diff * diff
		}
		distances[index] = distance
	}
	features.pushDistances(distances[:])

	copy(vector[NumLowerBands:NumBands], cepstrum[NumLowerBands:])
	for index := 0; index < NumLowerBands; index++ {
		curr, prev1, prev2 := features.ring[0][index], features.ring[1][index], features.ring[2][index]
		vector[index] = curr + prev1 + prev2
		vector[NumBands+index] = curr - prev2
		vector[NumBands+NumLowerBands+index] = curr - 2*prev1 + prev2
	}
	cross := crossCorrelation(referenceFFT, laggedFFT)
	// WebRTC normalises every band, not only the ones it keeps as features.
	for index := range cross {
		cross[index] = cross[index] /
			sqrt32(0.001+referenceEnergy[index]*laggedEnergy[index])
	}
	// WebRTC then takes the cepstrum of the normalised cross-correlation and
	// applies two ad-hoc correction terms.
	// WebRTC passes the whole 20-band correlation to ComputeDct and asks for
	// only the first 6 coefficients, so every band contributes to each output.
	correlationCepstrum := dct(cross, NumLowerBands)
	correlationCepstrum[0] -= 1.3
	correlationCepstrum[1] -= 0.9
	for index := 0; index < NumLowerBands; index++ {
		vector[NumBands+2*NumLowerBands+index] = correlationCepstrum[index]
	}
	vector[FeatureSize-1] = features.variability()
	return vector, false
}

// variability scores how much the cepstrum moves: for every history entry it
// takes the smallest distance to any other entry, sums those, and normalises
// with the training-set statistics. Ported from ComputeVariability.
func (features *SpectralFeatures) variability() float32 {
	var variability float32
	for delay1 := 0; delay1 < CepstralHistory; delay1++ {
		minDist := float32(math.MaxFloat32)
		for delay2 := 0; delay2 < CepstralHistory; delay2++ {
			if delay1 == delay2 {
				continue
			}
			minDist = min32(minDist, features.distanceAt(delay1, delay2))
		}
		variability += minDist
	}
	return variability/float32(CepstralHistory) - 2.1
}
