import { appVoice, VoiceSession } from "./app-voice";
import workletURL from "./piik-voice-bridge.js?url";

// A single microphone capture owns its device, gain and output. Transports
// borrow the output track; replacing a device never changes room authority.
export class MicrophoneInput {
  readonly stream: MediaStream;
  private readonly gain: GainNode;
  private readonly analyser: AnalyserNode;
  private readonly samples = new Float32Array(256);
  private bridge: AudioWorkletNode | null = null;
  private session: VoiceSession | null = null;
  private closed = false;

  constructor(private readonly captured: MediaStream, private readonly context: AudioContext,
    volume: number, ended: () => void) {
    this.gain = this.context.createGain();
    this.gain.gain.value = volume;
    this.analyser = this.context.createAnalyser();
    this.analyser.fftSize = this.samples.length;
    const source = this.context.createMediaStreamSource(captured);
    const destination = this.context.createMediaStreamDestination();
    source.connect(this.gain); this.gain.connect(destination); this.gain.connect(this.analyser);
    this.stream = destination.stream;
    this.stream.getAudioTracks()[0]!.contentHint = "speech";
    captured.getAudioTracks()[0]!.onended = ended;
  }

  // attachDSP routes the microphone through the App's native chain: denoise
  // with DFSMN, then AGC2, both Go code bound with mygo. The chain sits before
  // the volume control, so the user's volume scales the processed voice.
  //
  // Only the App can run it. A Browser page reports unavailable and keeps its
  // own capture path, so this never changes how a plain page behaves.
  async attachDSP(): Promise<void> {
    if (this.bridge || this.closed || !appVoice()) return;
    try {
      const session = new VoiceSession();
      await session.start();
      if (this.closed) { await session.stop(); return; }
      await this.context.audioWorklet.addModule(workletURL);
      if (this.closed) { await session.stop(); return; }

      const node = new AudioWorkletNode(this.context, "piik-voice-bridge", {
        numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1,
        channelCountMode: "explicit", outputChannelCount: [1],
      });
      // The worklet posts blocks; this page runs the native chain for each.
      node.port.onmessage = async event => {
        if (event.data?.type !== "block" || !event.data.samples) return;
        const block = Float32Array.from(event.data.samples);
        const processed = await session.process(block);
        node.port.postMessage({ type: "block", samples: Array.from(processed) });
      };

      const source = this.context.createMediaStreamSource(this.captured);
      source.connect(node);
      node.connect(this.gain);
      this.bridge = node;
      this.session = session;
    } catch {
      // A missing App or a refused bind leaves plain capture in place; room
      // voice must never go silent because processing failed.
      await this.session?.stop();
      this.session = null;
      this.bridge = null;
    }
  }

  volume(value: number): void { this.gain.gain.setTargetAtTime(value, this.context.currentTime, .01); }
  level(): number {
    if (this.closed) return 0;
    this.analyser.getFloatTimeDomainData(this.samples);
    return Math.sqrt(this.samples.reduce((sum, value) => sum + value * value, 0) / this.samples.length);
  }
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.bridge?.disconnect();
    this.bridge = null;
    void this.session?.stop();
    this.session = null;
    this.captured.getTracks().forEach(track => { track.onended = null; track.stop(); });
    this.stream.getTracks().forEach(track => track.stop());
    void this.context.close().catch(() => {});
  }
}
