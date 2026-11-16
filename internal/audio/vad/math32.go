package vad

import "math"

// Float32 wrappers for the math functions WebRTC applies to float values.
// Keeping these in float32 avoids the mixed-type casts at every call site and
// matches WebRTC's arithmetic precision.
func cos32(x float32) float32    { return float32(math.Cos(float64(x))) }
func sin32(x float32) float32    { return float32(math.Sin(float64(x))) }
func sqrt32(x float32) float32   { return float32(math.Sqrt(float64(x))) }
func log10_32(x float32) float32 { return float32(math.Log10(float64(x))) }
func abs32(x float32) float32    { return float32(math.Abs(float64(x))) }
func copysign32(mag, sign float32) float32 {
	return float32(math.Copysign(float64(mag), float64(sign)))
}
func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
