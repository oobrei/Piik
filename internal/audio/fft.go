package audio

import "math"

// fftPlan is a radix-2 complex FFT for a fixed power-of-two size. It keeps
// precomputed twiddles and a bit-reversal table, so a frame costs one pass
// rather than a fresh plan.
type fftPlan struct {
	size    int
	cos     []float64
	sin     []float64
	reverse []int
	re      []float64
	im      []float64
}

// newFFTPlan prepares a plan for size, which must be a power of two.
func newFFTPlan(size int) *fftPlan {
	plan := &fftPlan{
		size:    size,
		cos:     make([]float64, size/2),
		sin:     make([]float64, size/2),
		reverse: make([]int, size),
		re:      make([]float64, size),
		im:      make([]float64, size),
	}
	for index := 0; index < size/2; index++ {
		angle := -2 * math.Pi * float64(index) / float64(size)
		plan.cos[index] = math.Cos(angle)
		plan.sin[index] = math.Sin(angle)
	}
	bits := 0
	for 1<<bits < size {
		bits++
	}
	for index := 0; index < size; index++ {
		reversed := 0
		for bit := 0; bit < bits; bit++ {
			if index&(1<<bit) != 0 {
				reversed |= 1 << (bits - 1 - bit)
			}
		}
		plan.reverse[index] = reversed
	}
	return plan
}

// forward transforms samples into interleaved real/imaginary bins.
func (plan *fftPlan) forward(samples, bins []float32) {
	for index := 0; index < plan.size; index++ {
		plan.re[index] = float64(samples[plan.reverse[index]])
		plan.im[index] = 0
	}
	plan.transform()
	for index := 0; index < plan.size/2+1; index++ {
		bins[2*index] = float32(plan.re[index])
		bins[2*index+1] = float32(plan.im[index])
	}
}

// inverse transforms interleaved bins back into samples. It takes the
// half-spectrum forward produced, rebuilds the conjugate half, and runs the
// same butterflies over the conjugated input.
func (plan *fftPlan) inverse(bins, samples []float32) {
	half := plan.size/2 + 1
	// Rebuild the full spectrum in natural order: the half forward produced,
	// then its conjugate mirror. transform expects bit-reversed input, so the
	// values go in at their reversed slots.
	for index := 0; index < half; index++ {
		slot := plan.reverse[index]
		plan.re[slot] = float64(bins[2*index])
		plan.im[slot] = -float64(bins[2*index+1])
	}
	for index := half; index < plan.size; index++ {
		mirror := plan.size - index
		slot := plan.reverse[index]
		plan.re[slot] = float64(bins[2*mirror])
		plan.im[slot] = float64(bins[2*mirror+1])
	}
	plan.transform()
	scale := 1 / float64(plan.size)
	for index := 0; index < plan.size; index++ {
		samples[index] = float32(plan.re[index] * scale)
	}
}

// transform runs the in-place Cooley-Tukey butterflies over re and im. The
// caller has already placed the input in bit-reversed order.
func (plan *fftPlan) transform() {
	size := plan.size
	for length := 2; length <= size; length <<= 1 {
		half := length / 2
		step := size / length
		for start := 0; start < size; start += length {
			for offset := 0; offset < half; offset++ {
				// The table is built for size, so a stage of length uses
				// every step-th entry: angle = -2*pi*offset/length.
				cosine := plan.cos[offset*step]
				sine := plan.sin[offset*step]
				a := start + offset
				b := a + half
				tr := plan.re[b]*cosine - plan.im[b]*sine
				ti := plan.re[b]*sine + plan.im[b]*cosine
				plan.re[b] = plan.re[a] - tr
				plan.im[b] = plan.im[a] - ti
				plan.re[a] += tr
				plan.im[a] += ti
			}
		}
	}
}
