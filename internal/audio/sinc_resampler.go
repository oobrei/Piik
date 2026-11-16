// Copyright (c) 2012 The WebRTC project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file in the root of the source tree. An additional
// intellectual property rights grant can be found in the PATENTS file. All
// contributing project authors may be found in the AUTHORS file in the source
// tree.
//
// Go port of common_audio/resampler/sinc_resampler.cc. It is a windowed-sinc
// sample-rate converter: a Blackman-windowed sinc kernel is precomputed at
// several sub-sample offsets and two neighbouring kernels are interpolated per
// output sample.

package audio

import "math"

// Sinc resampler geometry, from sinc_resampler.h.
const (
	// sincKernelSize is the number of taps in the filter.
	sincKernelSize = 32
	// sincKernelOffsetCount is how many sub-sample kernel shifts are stored.
	sincKernelOffsetCount = 32
	// sincKernelStorageSize is the total number of stored kernel taps.
	sincKernelStorageSize = sincKernelSize * (sincKernelOffsetCount + 1)
	// sincDefaultRequestSize is how many input frames are buffered at a time.
	sincDefaultRequestSize = 512
)

// sincScaleFactor returns the normalised cutoff frequency of the low-pass
// filter for the given input/output ratio.
func sincScaleFactor(ioRatio float64) float64 {
	factor := 1.0
	if ioRatio > 1 {
		factor = 1 / ioRatio
	}
	// The windowed sinc does not transition instantly, so pull the cutoff down
	// slightly to avoid aliasing at the very high end.
	return factor * 0.9
}

// sincResampler converts a signal between two sample rates.
type sincResampler struct {
	ioRatio      float64
	requestSize  int
	kernel       []float64
	input        []float64
	r0, r1, r2   int
	r3, r4       int
	blockSize    int
	virtualIndex float64
	primed       bool
}

// newSincResampler prepares a converter for the given input/output ratio.
func newSincResampler(ioRatio float64, requestSize int) *sincResampler {
	resampler := &sincResampler{
		ioRatio:     ioRatio,
		requestSize: requestSize,
		kernel:      make([]float64, sincKernelStorageSize),
		// The regions slide by up to kKernelSize, so the buffer carries that
		// much slack beyond the requested frames.
		input: make([]float64, requestSize+2*sincKernelSize),
	}
	resampler.initializeKernel()
	resampler.flush()
	return resampler
}

// initializeKernel builds the Blackman-windowed sinc kernels.
func (resampler *sincResampler) initializeKernel() {
	const (
		alpha = 0.16
		a0    = 0.5 * (1 - alpha)
		a1    = 0.5
		a2    = 0.5 * alpha
	)
	scale := sincScaleFactor(resampler.ioRatio)
	for offsetIndex := 0; offsetIndex <= sincKernelOffsetCount; offsetIndex++ {
		subsampleOffset := float64(offsetIndex) / sincKernelOffsetCount
		for tap := 0; tap < sincKernelSize; tap++ {
			index := tap + offsetIndex*sincKernelSize
			preSinc := math.Pi * (float64(tap) - float64(sincKernelSize/2) - subsampleOffset)
			x := (float64(tap) - subsampleOffset) / sincKernelSize
			window := a0 - a1*math.Cos(2*math.Pi*x) + a2*math.Cos(4*math.Pi*x)
			sinc := scale
			if preSinc != 0 {
				sinc = math.Sin(scale*preSinc) / preSinc
			}
			resampler.kernel[index] = window * sinc
		}
	}
}

// flush resets the buffering state.
func (resampler *sincResampler) flush() {
	for index := range resampler.input {
		resampler.input[index] = 0
	}
	// r1 and r2 straddle the kernel centre and never move.
	resampler.r1 = 0
	resampler.r2 = sincKernelSize / 2
	resampler.virtualIndex = 0
	resampler.primed = false
	resampler.updateRegions(false)
}

// updateRegions recomputes the buffer region pointers.
func (resampler *sincResampler) updateRegions(secondLoad bool) {
	offset := sincKernelSize / 2
	if secondLoad {
		offset = sincKernelSize
	}
	resampler.r0 = offset
	resampler.r3 = resampler.r0 + resampler.requestSize - sincKernelSize
	resampler.r4 = resampler.r0 + resampler.requestSize - sincKernelSize/2
	resampler.blockSize = resampler.r4 - resampler.r2
}

// chunkSize is how many output frames a single buffered input block covers.
func (resampler *sincResampler) chunkSize() int {
	return int(float64(resampler.blockSize) / resampler.ioRatio)
}

// convolve computes one output sample from two neighbouring kernels.
func (resampler *sincResampler) convolve(inputIndex, offsetIndex int, factor float64) float64 {
	k1 := offsetIndex * sincKernelSize
	k2 := k1 + sincKernelSize
	var sum1, sum2 float64
	for tap := 0; tap < sincKernelSize; tap++ {
		sample := resampler.input[inputIndex+tap]
		sum1 += sample * resampler.kernel[k1+tap]
		sum2 += sample * resampler.kernel[k2+tap]
	}
	return sum1 + factor*(sum2-sum1)
}

// prime runs the discarded first pass that PushSincResampler uses to fill the
// buffer with the kernel's half-size delay.
func (resampler *sincResampler) prime() {
	if resampler.primed {
		return
	}
	discard := make([]float64, resampler.chunkSize())
	resampler.resample(discard, func(dst []float64) {
		for index := range dst {
			dst[index] = 0
		}
	})
}

// resample converts source into destination, pulling input through read.
// read must fill exactly requestSize frames.
func (resampler *sincResampler) resample(destination []float64, read func(dst []float64)) {
	remaining := len(destination)
	written := 0
	if !resampler.primed && remaining > 0 {
		read(resampler.input[resampler.r0 : resampler.r0+resampler.requestSize])
		resampler.primed = true
	}
	for remaining > 0 {
		count := int(math.Ceil((float64(resampler.blockSize) - resampler.virtualIndex) / resampler.ioRatio))
		for ; count > 0; count-- {
			sourceIndex := int(resampler.virtualIndex)
			subsampleRemainder := resampler.virtualIndex - float64(sourceIndex)
			virtualOffset := subsampleRemainder * sincKernelOffsetCount
			offsetIndex := int(virtualOffset)
			factor := virtualOffset - float64(offsetIndex)
			destination[written] = resampler.convolve(resampler.r1+sourceIndex, offsetIndex, factor)
			written++
			resampler.virtualIndex += resampler.ioRatio
			remaining--
			if remaining == 0 {
				return
			}
		}
		resampler.virtualIndex -= float64(resampler.blockSize)
		// Step (3): wrap the last input frames back to the start of the buffer.
		// r1 and r2 are fixed; only r0 moves between passes.
		copy(resampler.input[resampler.r1:resampler.r1+sincKernelSize],
			resampler.input[resampler.r3:resampler.r3+sincKernelSize])
		// Step (4): on the second load r0 slides right so the wrapped frames
		// are not overwritten.
		if resampler.r0 == resampler.r2 {
			resampler.updateRegions(true)
		}
		// Step (5): refresh the buffer.
		read(resampler.input[resampler.r0 : resampler.r0+resampler.requestSize])
	}
}
