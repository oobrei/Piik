package onnx

import (
	"os"
	"path/filepath"
	"testing"
)

// The fixture is the real DFSMN export, so a passing test means the reader
// handles the graph Piik actually ships.
func loadFixture(t *testing.T) *Model {
	t.Helper()
	path := filepath.Join("testdata", "dfsmn_ans.onnx")
	payload, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("fixture missing: %v", err)
	}
	model, err := Parse(payload)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestParsesTheDenoiserGraph(t *testing.T) {
	model := loadFixture(t)
	if len(model.Nodes) == 0 {
		t.Fatal("no nodes")
	}
	if len(model.Initializers) == 0 {
		t.Fatal("no initializers")
	}
	ops := map[string]int{}
	for _, node := range model.Nodes {
		ops[node.OpType]++
	}
	for _, want := range []string{"MatMul", "Add", "Relu", "Sigmoid", "Transpose", "Conv"} {
		if ops[want] == 0 {
			t.Fatalf("op %s missing from %v", want, ops)
		}
	}
	// Anything outside the supported set would need a kernel we do not have.
	allowed := map[string]bool{"MatMul": true, "Add": true, "Relu": true, "Sigmoid": true,
		"Transpose": true, "Conv": true, "Unsqueeze": true, "Squeeze": true}
	for op := range ops {
		if !allowed[op] {
			t.Fatalf("graph uses unsupported op %q", op)
		}
	}
}

func TestReadsInputAndOutputShapes(t *testing.T) {
	model := loadFixture(t)
	if len(model.Inputs) != 1 || model.Inputs[0].Name != "fbank" {
		t.Fatalf("inputs = %+v", model.Inputs)
	}
	if shape := model.Inputs[0].Shape; len(shape) != 3 || shape[0] != 1 || shape[2] != 120 {
		t.Fatalf("input shape = %v, want [1,-1,120]", shape)
	}
	if len(model.Outputs) != 1 || model.Outputs[0].Name != "mask" {
		t.Fatalf("outputs = %+v", model.Outputs)
	}
	if shape := model.Outputs[0].Shape; len(shape) != 3 || shape[2] != 961 {
		t.Fatalf("output shape = %v, want [1,-1,961]", shape)
	}
}

func TestReadsWeightShapes(t *testing.T) {
	model := loadFixture(t)
	// The exporter folds the linear weights into MatMul constants, so they
	// appear under generated names; the conv kernels keep theirs.
	cases := []struct {
		name string
		dims []int
	}{
		{"linear1.linear.bias", []int{256}},
		{"linear2.linear.bias", []int{961}},
		{"deepfsmn.0.conv1.weight", []int{256, 1, 20, 1}},
		{"val_0", []int{120, 256}},
		{"val_85", []int{256, 961}},
	}
	for _, want := range cases {
		tensor, ok := model.Initializers[want.name]
		if !ok {
			t.Fatalf("%s missing", want.name)
		}
		if len(tensor.Dims) != len(want.dims) {
			t.Fatalf("%s dims = %v, want %v", want.name, tensor.Dims, want.dims)
		}
		for index := range want.dims {
			if tensor.Dims[index] != want.dims[index] {
				t.Fatalf("%s dims = %v, want %v", want.name, tensor.Dims, want.dims)
			}
		}
		if len(tensor.Data) == 0 {
			t.Fatalf("%s has no data", want.name)
		}
	}
}

func TestRejectsMalformedInput(t *testing.T) {
	if _, err := Parse([]byte{0xff, 0xff, 0xff}); err == nil {
		t.Fatal("malformed input was accepted")
	}
}
