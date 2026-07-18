import threading
import queue as sync_queue
import asyncio
import numpy as np
from typing import AsyncIterator
from .interfaces import TTSEngine

class VieNeuEngine(TTSEngine):
    def __init__(self, voice: str = "Trọng Hữu"):
        self.voice = voice
        self.sample_rate = 48000  # Unified output rate

    async def stream_pcm16(self, text: str) -> AsyncIterator[bytes]:
        # Use a thread-safe queue to bridge between the SDK's
        # synchronous generator and asyncio
        chunk_q: sync_queue.Queue = sync_queue.Queue()

        def _run():
            try:
                # Assuming the SDK is installed as `tieneu`
                import tieneu
                for f32 in tieneu.infer_stream(text, voice=self.voice):
                    # Convert float32 to PCM16
                    pcm = (f32 * 32767).astype(np.int16).tobytes()
                    chunk_q.put(pcm)
            except ImportError:
                # Fallback for missing SDK
                chunk_q.put(b"SDK_MISSING")
            except Exception as e:
                print(f"VieNeu Error: {e}")
            finally:
                chunk_q.put(None) # Poison pill

        threading.Thread(target=_run, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            chunk = await loop.run_in_executor(None, chunk_q.get)
            if chunk is None:
                break
            if chunk == b"SDK_MISSING":
                # Emit silence as fallback
                yield b"\x00" * (self.sample_rate * 2 // 10)
                break
            yield chunk
