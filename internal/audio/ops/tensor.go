// Package ops implements the handful of ONNX operators Piik's denoiser graph
// uses. It is deliberately not a general ONNX runtime: the exported DFSMN needs
// eight ops, and a purpose-built executor is far smaller than adopting one.
package ops

import (
	"errors"
	"fmt"
)

// Tensor is a dense row-major float32 tensor.
type Tensor struct {
	Shape []int
	Data  []float32
}

// New allocates a zeroed tensor.
func New(shape ...int) *Tensor {
	size := 1
	for _, dim := range shape {
		if dim < 0 {
			panic("ops: negative dimension")
		}
		size *= dim
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: make([]float32, size)}
}

// From wraps existing data, checking it matches the shape.
func From(shape []int, data []float32) (*Tensor, error) {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	if size != len(data) {
		return nil, fmt.Errorf("ops: shape %v needs %d values, got %d", shape, size, len(data))
	}
	return &Tensor{Shape: append([]int(nil), shape...), Data: data}, nil
}

// Len reports the number of values.
func (t *Tensor) Len() int { return len(t.Data) }

// Clone copies the tensor.
func (t *Tensor) Clone() *Tensor {
	return &Tensor{Shape: append([]int(nil), t.Shape...), Data: append([]float32(nil), t.Data...)}
}

// Rank reports the number of dimensions.
func (t *Tensor) Rank() int { return len(t.Shape) }

// ErrShape reports an operator given tensors it cannot combine.
var ErrShape = errors.New("ops: incompatible shapes")

func shapeString(shape []int) string { return fmt.Sprint(shape) }

// MatMul multiplies 2-D tensors: a[m,k] * b[k,n] = out[m,n].
func MatMul(a, b *Tensor) (*Tensor, error) {
	if a.Rank() != 2 || b.Rank() != 2 {
		return nil, fmt.Errorf("%w: MatMul needs 2-D, got %s and %s",
			ErrShape, shapeString(a.Shape), shapeString(b.Shape))
	}
	m, k := a.Shape[0], a.Shape[1]
	k2, n := b.Shape[0], b.Shape[1]
	if k != k2 {
		return nil, fmt.Errorf("%w: MatMul %s x %s", ErrShape, shapeString(a.Shape), shapeString(b.Shape))
	}
	out := New(m, n)
	for row := 0; row < m; row++ {
		outRow := out.Data[row*n : (row+1)*n]
		aRow := a.Data[row*k : (row+1)*k]
		for inner := 0; inner < k; inner++ {
			weight := aRow[inner]
			if weight == 0 {
				continue
			}
			bRow := b.Data[inner*n : (inner+1)*n]
			for col := 0; col < n; col++ {
				outRow[col] += weight * bRow[col]
			}
		}
	}
	return out, nil
}

// Add sums two tensors, broadcasting a trailing dimension of size one.
func Add(a, b *Tensor) (*Tensor, error) {
	if len(a.Data) == len(b.Data) {
		out := New(a.Shape...)
		for index := range out.Data {
			out.Data[index] = a.Data[index] + b.Data[index]
		}
		return out, nil
	}
	// Bias add: b is a vector matching the last axis of a.
	if b.Rank() == 1 && a.Rank() >= 1 && a.Shape[a.Rank()-1] == b.Shape[0] {
		out := New(a.Shape...)
		width := b.Shape[0]
		for index := range out.Data {
			out.Data[index] = a.Data[index] + b.Data[index%width]
		}
		return out, nil
	}
	return nil, fmt.Errorf("%w: Add %s + %s", ErrShape, shapeString(a.Shape), shapeString(b.Shape))
}

// Relu applies max(0, x) in place and returns the tensor.
func Relu(t *Tensor) *Tensor {
	for index, value := range t.Data {
		if value < 0 {
			t.Data[index] = 0
		}
	}
	return t
}

// Sigmoid applies 1/(1+exp(-x)) in place and returns the tensor.
func Sigmoid(t *Tensor) *Tensor {
	for index, value := range t.Data {
		t.Data[index] = 1 / (1 + exp32(-value))
	}
	return t
}

// Transpose permutes axes.
func Transpose(t *Tensor, perm []int) (*Tensor, error) {
	if len(perm) != t.Rank() {
		return nil, fmt.Errorf("%w: Transpose perm %v for rank %d", ErrShape, perm, t.Rank())
	}
	shape := make([]int, len(perm))
	for index, axis := range perm {
		if axis < 0 || axis >= t.Rank() {
			return nil, fmt.Errorf("%w: Transpose axis %d", ErrShape, axis)
		}
		shape[index] = t.Shape[axis]
	}
	out := New(shape...)
	strides := stridesFor(t.Shape)
	outStrides := stridesFor(shape)
	index := make([]int, t.Rank())
	for position := 0; position < len(out.Data); position++ {
		// Decompose the output position into its axes, then read the input.
		remainder := position
		source := 0
		for axis := range index {
			index[axis] = remainder / outStrides[axis]
			remainder %= outStrides[axis]
			source += index[axis] * strides[perm[axis]]
		}
		out.Data[position] = t.Data[source]
	}
	return out, nil
}

// Unsqueeze inserts a dimension of size one at axis.
func Unsqueeze(t *Tensor, axis int) (*Tensor, error) {
	if axis < 0 || axis > t.Rank() {
		return nil, fmt.Errorf("%w: Unsqueeze axis %d for rank %d", ErrShape, axis, t.Rank())
	}
	shape := make([]int, 0, t.Rank()+1)
	shape = append(shape, t.Shape[:axis]...)
	shape = append(shape, 1)
	shape = append(shape, t.Shape[axis:]...)
	return &Tensor{Shape: shape, Data: t.Data}, nil
}

// Squeeze removes every dimension of size one.
func Squeeze(t *Tensor) (*Tensor, error) {
	shape := make([]int, 0, t.Rank())
	for _, dim := range t.Shape {
		if dim != 1 {
			shape = append(shape, dim)
		}
	}
	if len(shape) == 0 {
		shape = []int{1}
	}
	return &Tensor{Shape: shape, Data: t.Data}, nil
}

// ConvDepthwise applies a depthwise 2-D convolution with an optional bias.
// weight is [channels, 1, kernelH, kernelW]; input is [channels, height, width].
func ConvDepthwise(input, weight, bias *Tensor) (*Tensor, error) {
	if input.Rank() != 3 {
		return nil, fmt.Errorf("%w: Conv input rank %d, want 3", ErrShape, input.Rank())
	}
	if weight.Rank() != 4 || weight.Shape[1] != 1 {
		return nil, fmt.Errorf("%w: Conv weight %s, want [C,1,kH,kW]", ErrShape, shapeString(weight.Shape))
	}
	channels, height, width := input.Shape[0], input.Shape[1], input.Shape[2]
	if weight.Shape[0] != channels {
		return nil, fmt.Errorf("%w: Conv channels %d vs %d", ErrShape, weight.Shape[0], channels)
	}
	kernelH, kernelW := weight.Shape[2], weight.Shape[3]
	outH, outW := height-kernelH+1, width-kernelW+1
	if outH <= 0 || outW <= 0 {
		return nil, fmt.Errorf("%w: Conv output %dx%d", ErrShape, outH, outW)
	}
	out := New(channels, outH, outW)
	for channel := 0; channel < channels; channel++ {
		inBase := channel * height * width
		wBase := channel * kernelH * kernelW
		outBase := channel * outH * outW
		for y := 0; y < outH; y++ {
			for x := 0; x < outW; x++ {
				var sum float32
				for ky := 0; ky < kernelH; ky++ {
					inRow := inBase + (y+ky)*width + x
					wRow := wBase + ky*kernelW
					for kx := 0; kx < kernelW; kx++ {
						sum += input.Data[inRow+kx] * weight.Data[wRow+kx]
					}
				}
				if bias != nil {
					sum += bias.Data[channel]
				}
				out.Data[outBase+y*outW+x] = sum
			}
		}
	}
	return out, nil
}

func stridesFor(shape []int) []int {
	strides := make([]int, len(shape))
	step := 1
	for index := len(shape) - 1; index >= 0; index-- {
		strides[index] = step
		step *= shape[index]
	}
	return strides
}
