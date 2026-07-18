import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
from .interfaces import ASREngine

logger = logging.getLogger("asr_english")

MODELS_DIR = Path("D:/VNAI/models/stt")

INITIAL_TRANSCRIBE_BYTES = 4800

class EnglishASR(ASREngine):
    def __init__(self):
        self.model = None
        self._mock = None
        self._audio_buffer = bytearray()
        self._pending_partial: str | None = None
        self._transcribe_future = None
        self._poll_task: asyncio.Task | None = None
        self._result_callback = None

        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                str(MODELS_DIR / "whisper-medium"),
                device="cpu",
                compute_type="int8"
            )
            logger.info("Successfully loaded English ASR model (faster-whisper medium)")
        except Exception as e:
            logger.error("Error loading English ASR: %s", e)
            from .fakes import FakeASR
            self._mock = FakeASR()

    async def start_utterance(self) -> None:
        self._audio_buffer.clear()
        self._pending_partial = None
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        if self._transcribe_future is not None:
            try:
                self._transcribe_future.result()
            except Exception:
                pass
            self._transcribe_future = None
        if self._mock:
            await self._mock.start_utterance()

    def _transcribe_in_thread(self, audio_bytes: bytes):
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _ = self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                beam_size=1
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return text if text else None
        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    async def _poll_loop(self, loop):
        executor = ThreadPoolExecutor(max_workers=1)

        while True:
            await asyncio.sleep(0.05)  # poll every 50ms

            if self._transcribe_future is None:
                if len(self._audio_buffer) >= INITIAL_TRANSCRIBE_BYTES:
                    audio_copy = bytes(self._audio_buffer)
                    self._transcribe_future = loop.run_in_executor(
                        executor, self._transcribe_in_thread, audio_copy
                    )
                continue

            try:
                done = self._transcribe_future.done()
            except Exception:
                self._transcribe_future = None
                continue

            if not done:
                continue

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

            if len(self._audio_buffer) >= INITIAL_TRANSCRIBE_BYTES:
                audio_copy = bytes(self._audio_buffer)
                self._transcribe_future = loop.run_in_executor(
                    executor, self._transcribe_in_thread, audio_copy
                )

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        if self._mock:
            return await self._mock.feed_audio(pcm16_chunk)

        self._audio_buffer.extend(pcm16_chunk)

        if self._poll_task is None:
            loop = asyncio.get_event_loop()
            self._poll_task = asyncio.create_task(self._poll_loop(loop))

        return None

    def set_result_callback(self, callback):
        self._result_callback = callback

    def snapshot_audio(self) -> bytes:
        """SYNC race-safe grab + clear of current utterance audio (see asr_vietnamese)."""
        if self._mock:
            return b""
        b = bytes(self._audio_buffer)
        self._audio_buffer = bytearray()
        self._pending_partial = None
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None
        self._transcribe_future = None
        return b

    async def finalize(self, audio_bytes: bytes = b"") -> Tuple[str, str]:
        """Transcribe snapshotted utterance audio, beam_size=1 for low latency."""
        if self._mock:
            return await self._mock.finalize()

        if not audio_bytes:
            return "", "en"

        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: self.model.transcribe(
                audio_data,
                language="en",
                task="transcribe",
                beam_size=1
            )
        )

        text = " ".join(s.text.strip() for s in segments).strip()
        return text, "en"
