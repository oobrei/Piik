package audio

import (
	"fmt"
	"math"
)

// spectral is the transform an Analyzer and a Synthesizer share: a forward
// half-spectrum and the inverse that returns a whole frame.
type spectral interface {
	forward(samples, bins []float32)
	inverse(bins, samples []float32)
}

var _ spectral = (*fftPlan)(nil)

// maxDirectSize bounds the short stage of a two-stage plan. A direct transform
// of that many points is cheap and needs no further decomposition.
const maxDirectSize = 32

// newSpectral prepares a transform for size. A power of two takes the radix-2
// plan; a size such as the reference pipeline's 1920 splits into one radix-2
// stage and one short direct stage.
func newSpectral(size int) (spectral, error) {
	if size <= 0 {
		return nil, fmt.Errorf("audio: transform size %d", size)
	}
	if size&(size-1) == 0 {
		return newFFTPlan(size), nil
	}
	stage, short, ok := splitSize(size)
	if !ok {
		return nil, fmt.Errorf("audio: no transform for size %d", size)
	}
	return newCompositePlan(stage, short), nil
}

// splitSize finds the largest power-of-two stage whose companion stage is small
// enough to transform directly.
func splitSize(size int) (stage, short int, ok bool) {
	for candidate := 1; candidate <= size; candidate <<= 1 {
		if size%candidate != 0 {
			continue
		}
		if rest := size / candidate; rest <= maxDirectSize {
			stage, short, ok = candidate, rest, true
		}
	}
	return stage, short, ok
}

// compositePlan applies a size-point DFT as a two-stage Cooley-Tukey
// decomposition: a short direct transform along one axis and the radix-2 plan
// along the other, joined by twiddles. It exists because 1920, the frame the
// denoiser masks, is not a power of two.
//
// Index maths: with size = big*short, sample n = big*shortIndex + bigIndex and
// bin k = shortBin + short*bigBin.
type compositePlan struct {
	stage    *fftPlan // the radix-2 axis
	size     int
	big      int
	short    int
	cos      []float64
	sin      []float64
	shortCos []float64
	shortSin []float64
	re       []float64 // natural-order input of the last transform
	im       []float64
	outRe    []float64 // natural-order output of the last transform
	outIm    []float64
	partRe   []float64 // the short stage's output
	partIm   []float64
}

func newCompositePlan(big, short int) *compositePlan {
	size := big * short
	plan := &compositePlan{
		stage:    newFFTPlan(big),
		size:     size,
		big:      big,
		short:    short,
		cos:      make([]float64, size),
		sin:      make([]float64, size),
		shortCos: make([]float64, short),
		shortSin: make([]float64, short),
		re:       make([]float64, size),
		im:       make([]float64, size),
		outRe:    make([]float64, size),
		outIm:    make([]float64, size),
		partRe:   make([]float64, size),
		partIm:   make([]float64, size),
	}
	for index := 0; index < size; index++ {
		angle := -2 * math.Pi * float64(index) / float64(size)
		plan.cos[index], plan.sin[index] = math.Cos(angle), math.Sin(angle)
	}
	for index := 0; index < short; index++ {
		angle := -2 * math.Pi * float64(index) / float64(short)
		plan.shortCos[index], plan.shortSin[index] = math.Cos(angle), math.Sin(angle)
	}
	return plan
}

func (plan *compositePlan) forward(samples, bins []float32) {
	for index := 0; index < plan.size; index++ {
		plan.re[index] = float64(samples[index])
		plan.im[index] = 0
	}
	plan.transform()
	for index := 0; index < plan.size/2+1; index++ {
		bins[2*index] = float32(plan.outRe[index])
		bins[2*index+1] = float32(plan.outIm[index])
	}
}

// inverse rebuilds the conjugate half of the spectrum, transforms the conjugate
// and scales, which is the inverse the radix-2 plan applies.
func (plan *compositePlan) inverse(bins, samples []float32) {
	half := plan.size/2 + 1
	for index := 0; index < half; index++ {
		plan.re[index] = float64(bins[2*index])
		plan.im[index] = -float64(bins[2*index+1])
	}
	for index := half; index < plan.size; index++ {
		mirror := plan.size - index
		plan.re[index] = float64(bins[2*mirror])
		plan.im[index] = float64(bins[2*mirror+1])
	}
	plan.transform()
	scale := 1 / float64(plan.size)
	for index := 0; index < plan.size; index++ {
		samples[index] = float32(plan.outRe[index] * scale)
	}
}

// transform runs both stages over re and im, leaving the natural-order spectrum
// in outRe and outIm.
func (plan *compositePlan) transform() {
	big, short := plan.big, plan.short
	// Stage one: a direct transform along the short axis for each big-axis
	// offset, which reads the input at a stride of big.
	for offset := 0; offset < big; offset++ {
		for bin := 0; bin < short; bin++ {
			var re, im float64
			for index := 0; index < short; index++ {
				twiddle := (index * bin) % short
				valueRe := plan.re[index*big+offset]
				valueIm := plan.im[index*big+offset]
				re += valueRe*plan.shortCos[twiddle] - valueIm*plan.shortSin[twiddle]
				im += valueRe*plan.shortSin[twiddle] + valueIm*plan.shortCos[twiddle]
			}
			plan.partRe[offset*short+bin] = re
			plan.partIm[offset*short+bin] = im
		}
	}
	// Stage two: twiddle each partial result, then transform along the radix-2
	// axis. The plan expects bit-reversed input.
	for bin := 0; bin < short; bin++ {
		for index := 0; index < big; index++ {
			position := index*short + bin
			twiddle := (index * bin) % plan.size
			re, im := plan.partRe[position], plan.partIm[position]
			slot := plan.stage.reverse[index]
			plan.stage.re[slot] = re*plan.cos[twiddle] - im*plan.sin[twiddle]
			plan.stage.im[slot] = re*plan.sin[twiddle] + im*plan.cos[twiddle]
		}
		plan.stage.transform()
		for index := 0; index < big; index++ {
			bin := bin + short*index
			plan.outRe[bin] = plan.stage.re[index]
			plan.outIm[bin] = plan.stage.im[index]
		}
	}
}
