class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // 512 samples = 32ms @ 16kHz = exactly 1 Silero VAD window per chunk.
    // Small chunks let the backend VAD bridge brief intra-sentence pauses
    // instead of fragmenting speech into 300ms utterances (which produced
    // garbage like 'n.' / 'lô.'). Matches the proven test_backend_local path.
    this.buffer = new Float32Array(512);
    this.writeIdx = 0;
  }

  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;

    for (let i = 0; i < input.length; i++) {
      this.buffer[this.writeIdx++] = input[i];
      if (this.writeIdx >= this.buffer.length) {
        const int16 = new Int16Array(this.buffer.length);
        for (let j = 0; j < this.buffer.length; j++) {
          const s = Math.max(-1, Math.min(1, this.buffer[j]));
          int16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        this.port.postMessage(int16.buffer, [int16.buffer]);
        this.writeIdx = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
