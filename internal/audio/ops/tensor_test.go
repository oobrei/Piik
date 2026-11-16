package ops

import (
	"math"
	"testing"
)

// The expected values come from NumPy on the same inputs, so a passing test
// means the kernels agree with the reference the model was exported from.

func TestMatMulMatchesNumPy(t *testing.T) {
	a, _ := From([]int{3, 4}, []float32{1.74945474, -0.286072999, -0.484565139, -2.65331864,
		-0.00828462932, -0.319631368, -0.536629379, 0.315402657,
		0.421050727, -1.06560302, -0.886239648, -0.475733489})
	b, _ := From([]int{4, 5}, []float32{0.689682305, 0.561192155, -1.30554855, -1.11947525, 0.736837387,
		1.57463408, -0.0310750883, -0.683446646, 1.09562969, -0.309576631,
		0.725752234, 1.54907167, 0.630079806, 0.0734932348, 0.732271373,
		-0.642575383, -0.17809318, -0.573954582, -0.204375312, -0.486495107})
	want := []float32{2.10939074, 0.712581873, -0.87091279, -1.76524091, 2.31361866,
		-1.10114622, -0.882165134, -0.289879173, -0.444822341, -0.453554213,
		-1.72503948, -1.01871979, -0.106771648, -1.60676658, 0.22260581}
	out, err := MatMul(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Shape) != 2 || out.Shape[0] != 3 || out.Shape[1] != 5 {
		t.Fatalf("shape = %v", out.Shape)
	}
	assertClose(t, out.Data, want, 1e-5)
}

func TestAddBroadcastsBias(t *testing.T) {
	a, _ := From([]int{2, 3}, []float32{-0.185775325, -0.380536407, 0.0889776349,
		0.0636716634, 0.296347111, 1.40277112})
	b, _ := From([]int{3}, []float32{-1.5468626, 1.29561853, -0.237250447})
	want := []float32{-1.73263788, 0.915082097, -0.148272812,
		-1.48319089, 1.59196568, 1.16552067}
	out, err := Add(a, b)
	if err != nil {
		t.Fatal(err)
	}
	assertClose(t, out.Data, want, 1e-5)
}

func TestTransposeMatchesNumPy(t *testing.T) {
	in, _ := From([]int{2, 3, 4}, []float32{-1.23234618, -0.172419772, 0.0918383747, 1.06755841,
		-1.06163442, 0.217348203, 0.117819503, -1.68411088,
		-1.18575525, 0.600102007, 0.69556725, 1.08771086,
		0.533821702, 0.395212024, 0.122867532, 1.20910168,
		-0.843066096, -0.141893581, 0.385354131, -1.57749426,
		1.3109436, -0.792864978, -0.0717494115, 2.15667439})
	want := []float32{-1.23234618, -1.06163442, -1.18575525, 0.533821702, -0.843066096, 1.3109436,
		-0.172419772, 0.217348203, 0.600102007, 0.395212024, -0.141893581, -0.792864978,
		0.0918383747, 0.117819503, 0.69556725, 0.122867532, 0.385354131, -0.0717494115,
		1.06755841, -1.68411088, 1.08771086, 1.20910168, -1.57749426, 2.15667439}
	out, err := Transpose(in, []int{2, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if out.Shape[0] != 4 || out.Shape[1] != 2 || out.Shape[2] != 3 {
		t.Fatalf("shape = %v", out.Shape)
	}
	assertClose(t, out.Data, want, 1e-5)
}

func TestConvDepthwiseMatchesNumPy(t *testing.T) {
	in, _ := From([]int{2, 3, 5}, []float32{-0.829437256, -0.529372036, 1.56170368, -1.08019388, -0.432057619,
		0.516084015, 0.455392867, 0.593686223, 0.370506346, 1.34537804,
		1.0159421, 0.595278442, -0.682801604, -0.713559926, -1.90828955,
		0.687312365, -1.82300961, 0.879113793, 1.84636486, -1.06165445,
		-0.684484661, -0.476214468, 0.830310404, -0.863352537, -0.130541474,
		-0.523087621, -0.25127691, 1.29127479, -0.964204848, 0.0717597753})
	w, _ := From([]int{2, 1, 2, 2}, []float32{0.271606296, 0.858667195, -1.26407373, 1.11487031,
		0.434776992, 0.874272764, 0.0716520995, -1.63905168})
	want := []float32{-0.824499071, 1.28343582, -0.840755105, 0.36719358,
		-0.0893648863, -0.880244672, 0.546975315, 0.0303609371,
		-0.563484669, -1.41906083, 3.47101784, 0.0266845971,
		-0.339564055, -1.61559987, 1.27909827, -0.676199973}
	out, err := ConvDepthwise(in, w, nil)
	if err != nil {
		t.Fatal(err)
	}
	if out.Shape[0] != 2 || out.Shape[1] != 2 || out.Shape[2] != 4 {
		t.Fatalf("shape = %v", out.Shape)
	}
	assertClose(t, out.Data, want, 1e-5)
}

func TestReluAndSigmoid(t *testing.T) {
	in, _ := From([]int{4}, []float32{-2, -0.5, 0.5, 2})
	relu := Relu(in.Clone())
	assertClose(t, relu.Data, []float32{0, 0, 0.5, 2}, 1e-6)
	sig := Sigmoid(in.Clone())
	for index, value := range in.Data {
		want := float32(1 / (1 + math.Exp(float64(-value))))
		if diff := math.Abs(float64(sig.Data[index] - want)); diff > 1e-6 {
			t.Fatalf("sigmoid[%d] = %v, want %v", index, sig.Data[index], want)
		}
	}
}

func TestSqueezeAndUnsqueeze(t *testing.T) {
	in := New(1, 3, 1)
	up, err := Unsqueeze(in, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(up.Shape) != 4 || up.Shape[0] != 1 {
		t.Fatalf("unsqueeze shape = %v", up.Shape)
	}
	down, err := Squeeze(up)
	if err != nil {
		t.Fatal(err)
	}
	if len(down.Shape) != 1 || down.Shape[0] != 3 {
		t.Fatalf("squeeze shape = %v", down.Shape)
	}
}

func TestMatMulRejectsMismatchedShapes(t *testing.T) {
	a, _ := From([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6})
	b, _ := From([]int{4, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8})
	if _, err := MatMul(a, b); err == nil {
		t.Fatal("mismatched MatMul was accepted")
	}
}

func assertClose(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, want %d", len(got), len(want))
	}
	for index := range want {
		if diff := math.Abs(float64(got[index] - want[index])); diff > tolerance {
			t.Fatalf("[%d] = %v, want %v (diff %v)", index, got[index], want[index], diff)
		}
	}
}
