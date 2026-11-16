package voice

import (
	"math"
	"testing"
)

// The service must run the embedded DFSMN model and AGC2 over 128-sample
// blocks and return one sample per input sample.
func TestServiceProcessesBlocks(t *testing.T) {
	service := NewService()
	id, err := service.Open()
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if kind, _ := service.Describe(); kind == "closed" {
		t.Fatal("service reports closed after open")
	}

	block := make([]float32, BlockSamples)
	phase := 0
	peak := 0.0
	nonEmpty := 0
	// One second of a 440 Hz tone at the float S16 scale room voice uses.
	for n := 0; n < 48000/BlockSamples; n++ {
		for i := range block {
			block[i] = float32(math.Sin(2*math.Pi*440*float64(phase)/48000) * 3000)
			phase++
		}
		out, err := service.Process(id, block)
		if err != nil {
			t.Fatalf("process: %v", err)
		}
		if len(out) != len(block) {
			t.Fatalf("block %d: %d samples out, want %d", n, len(out), len(block))
		}
		for _, v := range out {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("block %d: non-finite output", n)
			}
			if a := math.Abs(float64(v)); a > peak {
				peak = a
			}
			if v != 0 {
				nonEmpty++
			}
		}
	}
	if peak == 0 || nonEmpty == 0 {
		t.Fatal("chain produced no output")
	}
	if peak >= 32768 {
		t.Fatalf("output peak %.1f exceeds the float S16 range", peak)
	}
	t.Logf("peak %.1f, %d non-zero samples", peak, nonEmpty)

	if err = service.Close(id); err != nil {
		t.Fatalf("close: %v", err)
	}
	if _, err = service.Process(id, block); err == nil {
		t.Fatal("process after close should fail")
	}
}

// A second open must be refused rather than silently replacing the session.
func TestServiceRejectsSecondOpen(t *testing.T) {
	service := NewService()
	if _, err := service.Open(); err != nil {
		t.Fatalf("first open: %v", err)
	}
	if _, err := service.Open(); err == nil {
		t.Fatal("second open should fail while a session is open")
	}
}
