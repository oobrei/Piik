// Typed access to the App's native room-voice chain.
//
// The chain (DFSMN denoise then AGC2) is Go code in internal/audio, exposed
// through mygo.Bind. Only the App can run it: in an ordinary Browser page
// globalThis.mygo is absent, so appVoice() reports unavailable and room voice
// keeps the Browser's own capture path.

import { call, isMyGo } from "mygo-runtime";

// BlockSamples is the block size the chain expects: the 128 samples an
// AudioWorklet renders per quantum at 48 kHz.
export const BlockSamples = 128;

interface VoiceRuntime {
  open(): Promise<string>;
  close(id: string): Promise<void>;
  reset(id: string): Promise<void>;
  process(id: string, samples: number[]): Promise<number[]>;
}

// appVoice reports whether this page can use the App's native chain.
export function appVoice(): boolean {
  return isMyGo();
}

// voiceRuntime is the bound service, resolved lazily so importing this module
// never touches the runtime.
function runtime(): VoiceRuntime {
  return {
    open: () => call("Service.Open") as Promise<string>,
    close: (id) => call("Service.Close", id) as Promise<void>,
    reset: (id) => call("Service.Reset", id) as Promise<void>,
    process: (id, samples) => call("Service.Process", id, Array.from(samples)) as Promise<number[]>,
  };
}

// VoiceSession is one microphone's chain. Opening it allocates the native
// streaming state; closing it releases that state, so one microphone never
// inherits another's gain or noise estimate.
export class VoiceSession {
  private id: string | null = null;
  private pending: Promise<void> | null = null;

  // start opens the native session. It resolves once the chain is ready.
  start(): Promise<void> {
    if (this.pending) return this.pending;
    this.pending = runtime().open().then(id => { this.id = id; });
    return this.pending;
  }

  // process runs one block through the chain. Before the session is open, or
  // if the App call fails, it returns the input unchanged so voice keeps
  // flowing rather than going silent.
  async process(block: Float32Array): Promise<Float32Array> {
    if (!this.id) return block;
    try {
      const out = await runtime().process(this.id, block.subarray(0, BlockSamples) as unknown as number[]);
      if (!out || out.length !== BlockSamples) return block;
      block.set(out);
      return block;
    } catch {
      return block;
    }
  }

  // reset clears streaming state for a device change that keeps the session.
  async reset(): Promise<void> {
    if (!this.id) return;
    try { await runtime().reset(this.id); } catch { /* a failed reset is not fatal */ }
  }

  // stop closes the native session.
  async stop(): Promise<void> {
    const id = this.id;
    this.id = null;
    this.pending = null;
    if (!id) return;
    try { await runtime().close(id); } catch { /* the App may already be gone */ }
  }
}
