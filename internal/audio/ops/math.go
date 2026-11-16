package ops

import "math"

// exp32 is math.Exp in float32, kept local so the hot loops avoid the
// float64 round trip on every element.
func exp32(value float32) float32 { return float32(math.Exp(float64(value))) }
