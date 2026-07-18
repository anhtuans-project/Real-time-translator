import { useRef } from 'react';

export function useAudioCapture(onChunk: (buf: ArrayBuffer) => void) {
  const startRef = useRef<(() => void) | null>(null);

  const start = async () => {
    try {
      console.log('[AudioCapture] Requesting microphone...');
      // Raw mic: disable browser audio processing so we capture the true
      // signal (echoCancellation / noiseSuppression / autoGainControl can
      // squash speech on some setups). The translator app is used with
      // headphones, so echo cancellation isn't needed.
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      console.log('[AudioCapture] Microphone granted, creating AudioContext...');

      const ctx = new AudioContext({ sampleRate: 16000 });
      await ctx.audioWorklet.addModule('/worklets/pcm-processor.js');
      console.log('[AudioCapture] Worklet loaded, starting capture...');

      const source = ctx.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(ctx, 'pcm-processor');
      let chunkCount = 0;

      node.port.onmessage = (e) => {
        chunkCount++;
        if (chunkCount <= 3 || chunkCount % 100 === 0) {
          console.log(`[AudioCapture] Chunk #${chunkCount}, size=${e.data.byteLength} bytes`);
        }
        onChunk(e.data);
      };

      source.connect(node);
      console.log('[AudioCapture] Audio capture started successfully');

      startRef.current = () => {
        source.disconnect();
        ctx.close();
        stream.getTracks().forEach(t => t.stop());
      };
    } catch (err) {
      console.error("Failed to capture audio:", err);
      throw err;
    }
  };

  const stop = () => {
    if (startRef.current) {
      startRef.current();
      startRef.current = null;
    }
  };

  return { start, stop };
}
