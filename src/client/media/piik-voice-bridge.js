// Room-voice DSP bridge. The audio thread cannot await an IPC call, so this
// worklet posts its input blocks to the page, which runs the native Go chain
// and returns processed blocks.
//
// Audio and IPC run at different rates, so the worklet keeps a queue on both
// sides: input is sent as soon as a block is full and output is drained as the
// graph asks for it. When the chain falls behind, the worklet repeats the last
// processed block rather than emitting silence, which would click.

const BLOCK = 128;
// Hold a few blocks before starting, so ordinary scheduling jitter does not
// starve the output.
const PRIME_BLOCKS = 4;

class PiikVoiceBridge extends AudioWorkletProcessor {
  constructor() {
    super();
    this.input = new Float32Array(BLOCK);
    this.inputFill = 0;
    this.output = [];
    this.last = new Float32Array(BLOCK);
    this.ready = false;
    this.port.onmessage = event => {
      const data = event.data;
      if (data?.type === "block" && data.samples) {
        this.output.push(Float32Array.from(data.samples));
        if (this.output.length >= PRIME_BLOCKS) this.ready = true;
      }
    };
  }

  process(inputs, outputs) {
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!output) return true;

    // Accumulate input and hand whole blocks to the page.
    if (input) {
      for (let i = 0; i < input.length; i++) {
        this.input[this.inputFill++] = input[i];
        if (this.inputFill === BLOCK) {
          this.port.postMessage({ type: "block", samples: Array.from(this.input) });
          this.inputFill = 0;
        }
      }
    }

    // Drain processed output.
    if (this.ready && this.output.length > 0) {
      const block = this.output.shift();
      this.last.set(block);
      output.set(block);
    } else {
      // Before the first result, or on underrun, hold the last block so the
      // stream never drops to silence mid-word.
      output.set(this.last);
    }
    return true;
  }
}

registerProcessor("piik-voice-bridge", PiikVoiceBridge);
