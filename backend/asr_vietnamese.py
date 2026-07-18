import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Tuple
from .interfaces import ASREngine

logger = logging.getLogger("asr_vietnamese")

MODELS_DIR = Path("D:/VNAI/models/stt")

# Start transcription after this many bytes (~300ms at 16kHz mono)
INITIAL_TRANSCRIBE_BYTES = 4800

class VietnameseASR(ASREngine):
    def __init__(self):
        self.model = None
        self._mock = None
        self._audio_buffer = bytearray()
        self._pending_partial: str | None = None
        self._transcribe_future = None  # concurrent future from thread pool
        self._poll_task: asyncio.Task | None = None
        self._result_callback = None  # called with partial result when ready

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                str(MODELS_DIR / "whisper-medium"),
                device="cpu",
                compute_type="int8"
            )
            logger.info("Successfully loaded Vietnamese ASR model (whisper-medium)")
        except Exception as e:
            logger.error("Error loading Vietnamese ASR: %s", e)
            from .fakes import FakeASR
            self._mock = FakeASR()

    async def start_utterance(self) -> None:
        self._audio_buffer.clear()
        self._pending_partial = None
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        if self._transcribe_future is not None:
            self._transcribe_future.result()
            self._transcribe_future = None
        if self._mock:
            await self._mock.start_utterance()

    def _transcribe_in_thread(self, audio_bytes: bytes):
        """Runs in thread pool."""
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(
                audio_data,
                language="vi",
                task="transcribe",
                beam_size=1
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return text if text else None
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    async def _poll_loop(self, loop):
        """
        Poll the thread pool future until transcription completes,
        then fire callback and restart if there's more audio.
        """
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)

        while True:
            # Wait a bit before checking
            await asyncio.sleep(0.05)  # poll every 50ms

            if self._transcribe_future is None:
                # Start transcription if enough audio
                if len(self._audio_buffer) >= INITIAL_TRANSCRIBE_BYTES:
                    audio_copy = bytes(self._audio_buffer)
                    self._transcribe_future = loop.run_in_executor(
                        executor, self._transcribe_in_thread, audio_copy
                    )
                continue

            # Check if done
            try:
                done = self._transcribe_future.done()
            except Exception:
                self._transcribe_future = None
                continue

            if not done:
                continue

            # Transcription complete
            try:
                result = self._transcribe_future.result()
            except Exception as e:
                logger.error("Transcribe future error: %s", e)
                result = None
            self._transcribe_future = None

            if result:
                logger.info("[ASR] Partial result: %s", result)
                if self._result_callback:
                    try:
                        asyncio.create_task(self._result_callback(result))
                    except Exception as e:
                        logger.error("Callback error: %s", e)

            # Restart if there's still audio
            if len(self._audio_buffer) >= INITIAL_TRANSCRIBE_BYTES:
                audio_copy = bytes(self._audio_buffer)
                self._transcribe_future = loop.run_in_executor(
                    executor, self._transcribe_in_thread, audio_copy
                )

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        if self._mock:
            return await self._mock.feed_audio(pcm16_chunk)

        self._audio_buffer.extend(pcm16_chunk)

        # Start polling task on first chunk
        if self._poll_task is None:
            loop = asyncio.get_event_loop()
            self._poll_task = asyncio.create_task(self._poll_loop(loop))

        # feed_audio no longer returns partial — it goes through callback
        return None

    def set_result_callback(self, callback):
        """SessionState calls this to receive partial results."""
        self._result_callback = callback

    def snapshot_audio(self) -> bytes:
        """
        Atomically grab + clear the current utterance audio buffer (SYNC, race-safe).
        Called by SessionState at utterance_end / max-duration cap, BEFORE scheduling
        the async finalize task, so a new utterance's start_utterance() cannot clear
        the buffer out from under the in-flight finalize.
        """
        if self._mock:
            return b""
        b = bytes(self._audio_buffer)
        self._audio_buffer = bytearray()
        self._pending_partial = None
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        # Orphan any in-flight thread transcription; its (stale) result is discarded.
        self._transcribe_future = None
        return b

    async def finalize(self, audio_bytes: bytes = b"") -> Tuple[str, str]:
        """
        Transcribe the given (already snapshotted) utterance audio.
        Audio is snapshotted synchronously by snapshot_audio() to avoid races with
        the next utterance. beam_size=1 for low latency on short chunks (realtime).
        """
        if self._mock:
            return await self._mock.finalize()

        if not audio_bytes:
            return "", "vi"

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language="vi",
                task="transcribe",
                beam_size=1
            )
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        return text, "vi"
