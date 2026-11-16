package vad

const pi32 float32 = 3.14159265358979323846

// dftPlan is an exact real-input DFT of Frame20ms24k points.
//
// WebRTC transforms 480 samples with PFFFT, which is not a power of two, so a
// radix-2 plan cannot reproduce it: zero-padding to 512 moves every bin and the
// Opus band edges no longer land on the same frequencies. A direct transform is
// O(n^2), which is 230k multiply-adds per frame and comfortably fast at 100
// frames per second, and it is exactly the transform the reference uses.
type dftPlan struct {
	size    int
	cosines []float32
	sines   []float32
}

// newDFTPlan prepares the twiddle factors for a size-point transform.
func newDFTPlan(size int) *dftPlan {
	plan := &dftPlan{
		size:    size,
		cosines: make([]float32, size),
		sines:   make([]float32, size),
	}
	for index := 0; index < size; index++ {
		angle := -2 * pi32 * float32(index) / float32(size)
		plan.cosines[index], plan.sines[index] = cos32(angle), sin32(angle)
	}
	return plan
}

// forward returns the interleaved real/imaginary spectrum, keeping the
// Frame20ms24k/2 coefficient pairs WebRTC stores and dropping Nyquist.
func (plan *dftPlan) forward(samples []float32) []float32 {
	out := make([]float32, Frame20ms24k)
	half := Frame20ms24k / 2
	for bin := 0; bin < half; bin++ {
		var re, im float32
		for index := 0; index < plan.size; index++ {
			twiddle := (index * bin) % plan.size
			value := samples[index]
			re += value * plan.cosines[twiddle]
			im += value * plan.sines[twiddle]
		}
		out[2*bin] = re
		out[2*bin+1] = im
	}
	return out
}
