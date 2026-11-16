// Package dfsmn runs the DFSMN-ANS 48 kHz noise-suppression model in pure Go:
// 120 log-mel features per frame in, a real-valued mask over the 961 bins of a
// 1920-sample frame out.
//
// The graph is a chain of linear layers around nine UniDeepFsmn blocks, so the
// executor walks the parsed nodes and dispatches each to internal/audio/ops
// rather than unrolling the blocks by hand. That keeps one copy of the model
// structure, in the file the reader already parses, and refuses any construct
// this build cannot run instead of approximating it.
package dfsmn

import (
	"errors"
	"fmt"
	"os"

	"github.com/TNTcraftHIM/Piik/internal/audio/onnx"
	"github.com/TNTcraftHIM/Piik/internal/audio/ops"
)

// Features is the number of mel features the model consumes per frame.
const Features = 120

// Bins is the number of mask values the model emits per frame, matching the
// 1920-sample analysis frame the App uses.
const Bins = 961

// ErrUnsupported reports a graph construct this executor does not implement.
var ErrUnsupported = errors.New("dfsmn: unsupported graph construct")

// Model is a parsed graph plus the look-back state of its causal convolutions.
// It is not safe for concurrent use; one stream owns one Model.
type Model struct {
	graph   *onnx.Model
	weights map[string]*ops.Tensor
	history []*ops.Tensor // look-back rows per Conv node, indexed by node
}

// Load reads a model file.
func Load(path string) (*Model, error) {
	payload, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("dfsmn: %w", err)
	}
	return LoadBytes(payload)
}

// LoadBytes parses a model from memory, used by the embedded production graph.
func LoadBytes(payload []byte) (*Model, error) {
	graph, err := onnx.Parse(payload)
	if err != nil {
		return nil, fmt.Errorf("dfsmn: %w", err)
	}
	if len(graph.Inputs) != 1 || len(graph.Outputs) != 1 {
		return nil, fmt.Errorf("dfsmn: graph has %d inputs and %d outputs, want one of each",
			len(graph.Inputs), len(graph.Outputs))
	}
	model := &Model{
		graph:   graph,
		weights: make(map[string]*ops.Tensor, len(graph.Initializers)),
		history: make([]*ops.Tensor, len(graph.Nodes)),
	}
	for name, tensor := range graph.Initializers {
		weight, err := ops.From(tensor.Dims, tensor.Data)
		if err != nil {
			return nil, fmt.Errorf("dfsmn: initializer %q: %w", name, err)
		}
		model.weights[name] = weight
	}
	return model, nil
}

// Reset drops the convolution look-back, so the next sequence starts from a
// silent history. Masks for frames already processed are unaffected.
func (model *Model) Reset() {
	for index := range model.history {
		model.history[index] = nil
	}
}

// Forward runs features through the graph and returns one mask row per input
// frame. It is causal across calls: state from earlier calls is used and
// updated, so feeding one frame at a time gives the same masks as one
// whole-sequence call, which is what lets the stage run live.
func (model *Model) Forward(features [][]float32) ([][]float32, error) {
	if len(features) == 0 {
		return nil, nil
	}
	input := ops.New(1, len(features), Features)
	for row, values := range features {
		if len(values) != Features {
			return nil, fmt.Errorf("dfsmn: frame %d has %d features, want %d", row, len(values), Features)
		}
		copy(input.Data[row*Features:], values)
	}
	values := map[string]*ops.Tensor{model.graph.Inputs[0].Name: input}
	for index, node := range model.graph.Nodes {
		if len(node.Outputs) != 1 {
			return nil, fmt.Errorf("%w: %s has %d outputs, want 1", ErrUnsupported, node.OpType, len(node.Outputs))
		}
		out, err := model.apply(index, node, values)
		if err != nil {
			return nil, err
		}
		values[node.Outputs[0]] = out
	}
	mask, ok := values[model.graph.Outputs[0].Name]
	if !ok || len(mask.Data) != len(features)*Bins {
		return nil, fmt.Errorf("dfsmn: graph produced no %dx%d mask", len(features), Bins)
	}
	rows := make([][]float32, len(features))
	for row := range rows {
		rows[row] = append([]float32(nil), mask.Data[row*Bins:(row+1)*Bins]...)
	}
	return rows, nil
}

func (model *Model) apply(index int, node onnx.Node, values map[string]*ops.Tensor) (*ops.Tensor, error) {
	inputs := make([]*ops.Tensor, len(node.Inputs))
	for position, name := range node.Inputs {
		if name == "" {
			continue // an optional input the exporter left out, such as a bias
		}
		if value, ok := values[name]; ok {
			inputs[position] = value
			continue
		}
		weight, ok := model.weights[name]
		if !ok {
			return nil, fmt.Errorf("%w: %s reads %q, which nothing defines", ErrUnsupported, node.OpType, name)
		}
		inputs[position] = weight
	}
	switch node.OpType {
	case "MatMul":
		return matmul(inputs[0], inputs[1])
	case "Add":
		return ops.Add(inputs[0], inputs[1])
	case "Relu":
		// Relu and Sigmoid work in place; the copy keeps a value another node
		// may still read from being overwritten.
		return ops.Relu(inputs[0].Clone()), nil
	case "Sigmoid":
		return ops.Sigmoid(inputs[0].Clone()), nil
	case "Transpose":
		return transpose(node, inputs[0])
	case "Unsqueeze":
		return unsqueeze(node, inputs)
	case "Squeeze":
		return squeeze(node, inputs)
	case "Conv":
		return model.conv(index, node, inputs)
	}
	return nil, fmt.Errorf("%w: operator %s", ErrUnsupported, node.OpType)
}

// matmul folds a leading batch of one, which is the only batch the graph uses.
func matmul(a, b *ops.Tensor) (*ops.Tensor, error) {
	if a.Rank() != 3 || a.Shape[0] != 1 || b.Rank() != 2 {
		return ops.MatMul(a, b)
	}
	out, err := ops.MatMul(&ops.Tensor{Shape: a.Shape[1:], Data: a.Data}, b)
	if err != nil {
		return nil, err
	}
	out.Shape = []int{1, out.Shape[0], out.Shape[1]}
	return out, nil
}

func transpose(node onnx.Node, input *ops.Tensor) (*ops.Tensor, error) {
	attribute, ok := node.Attributes["perm"]
	axes := make([]int, 0, input.Rank())
	for _, axis := range attribute.Ints {
		axes = append(axes, int(axis))
	}
	if !ok {
		for axis := input.Rank() - 1; axis >= 0; axis-- {
			axes = append(axes, axis)
		}
	}
	return ops.Transpose(input, axes)
}

// axesOf reads an axis list, which ONNX exporters write either as an attribute
// or as a second tensor input.
func axesOf(node onnx.Node, inputs []*ops.Tensor, rank int) ([]int, error) {
	axes := []int{}
	if attribute, ok := node.Attributes["axes"]; ok {
		for _, axis := range attribute.Ints {
			axes = append(axes, int(axis))
		}
	} else if len(inputs) > 1 && inputs[1] != nil {
		for _, axis := range inputs[1].Data {
			axes = append(axes, int(axis))
		}
	}
	for index, axis := range axes {
		if axis < 0 {
			axes[index] = axis + rank
		}
	}
	return axes, nil
}

func unsqueeze(node onnx.Node, inputs []*ops.Tensor) (*ops.Tensor, error) {
	axes, err := axesOf(node, inputs, inputs[0].Rank()+1)
	if err != nil {
		return nil, err
	}
	out := inputs[0]
	for _, axis := range axes {
		if out, err = ops.Unsqueeze(out, axis); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// squeeze drops every dimension of size one, which is what this graph asks
// for; an explicit axis list would need a second reshaping path, so it is
// refused.
func squeeze(node onnx.Node, inputs []*ops.Tensor) (*ops.Tensor, error) {
	if len(inputs) > 1 || len(node.Attributes["axes"].Ints) > 0 {
		return nil, fmt.Errorf("%w: Squeeze with explicit axes", ErrUnsupported)
	}
	return ops.Squeeze(inputs[0])
}

// conv applies the graph's convolution: depthwise over channels, causal over
// time with kernel-1 rows of look-back, one bin wide. Look-back that an earlier
// call already saw is folded in here, so a one-frame call is as correct as a
// whole-sequence one. Any other convolution shape is refused rather than
// approximated.
func (model *Model) conv(index int, node onnx.Node, inputs []*ops.Tensor) (*ops.Tensor, error) {
	input, weight := inputs[0], inputs[1]
	var bias *ops.Tensor
	if len(inputs) > 2 {
		bias = inputs[2]
	}
	if input.Rank() != 4 || input.Shape[0] != 1 {
		return nil, fmt.Errorf("%w: Conv input %v, want [1,C,H,1]", ErrUnsupported, input.Shape)
	}
	channels, height, width := input.Shape[1], input.Shape[2], input.Shape[3]
	if weight.Rank() != 4 || weight.Shape[0] != channels || weight.Shape[1] != 1 {
		return nil, fmt.Errorf("%w: Conv weight %v for %d channels, want depthwise", ErrUnsupported, weight.Shape, channels)
	}
	if group := node.Attributes["group"].Int; group != 0 && group != int64(channels) {
		return nil, fmt.Errorf("%w: Conv group %d for %d channels", ErrUnsupported, group, channels)
	}
	for _, name := range []string{"strides", "dilations"} {
		if values := node.Attributes[name].Ints; len(values) > 0 && (values[0] != 1 || values[1] != 1) {
			return nil, fmt.Errorf("%w: Conv %s %v", ErrUnsupported, name, values)
		}
	}
	kernelH, kernelW := weight.Shape[2], weight.Shape[3]
	pads := node.Attributes["pads"].Ints
	if width != kernelW || len(pads) != 4 || pads[0] != int64(kernelH-1) || pads[1] != 0 || pads[2] != 0 || pads[3] != 0 {
		return nil, fmt.Errorf("%w: Conv pads %v, kernel %dx%d and width %d are not a causal depthwise row",
			ErrUnsupported, pads, kernelH, kernelW, width)
	}
	lookback := kernelH - 1
	history := model.history[index]
	if history == nil {
		history = ops.New(channels, lookback, width)
	}
	combined := ops.New(channels, lookback+height, width)
	row := (lookback + height) * width
	for channel := 0; channel < channels; channel++ {
		base := channel * row
		copy(combined.Data[base:base+lookback*width], history.Data[channel*lookback*width:(channel+1)*lookback*width])
		copy(combined.Data[base+lookback*width:base+row], input.Data[channel*height*width:(channel+1)*height*width])
	}
	model.history[index] = lastRows(combined, lookback)
	out, err := ops.ConvDepthwise(combined, weight, bias)
	if err != nil {
		return nil, err
	}
	return &ops.Tensor{Shape: []int{1, channels, out.Shape[1], out.Shape[2]}, Data: out.Data}, nil
}

// lastRows copies the newest count rows of a [channels, rows, width] tensor.
func lastRows(tensor *ops.Tensor, count int) *ops.Tensor {
	channels, rows, width := tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]
	out := ops.New(channels, count, width)
	for channel := 0; channel < channels; channel++ {
		base := channel * rows * width
		copy(out.Data[channel*count*width:(channel+1)*count*width], tensor.Data[base+(rows-count)*width:base+rows*width])
	}
	return out
}
