import asyncio
import random
from typing import AsyncIterator
from .interfaces import ASREngine, MTEngine, TTSEngine, ASRFinal

FIXTURE_VI = "Chúng tôi đề xuất mức đầu tư hai triệu đô la cho dự án này."
FIXTURE_EN = "We propose a two million dollar investment for this project."

class FakeASR:
    def __init__(self):
        self._target_lang = random.choice(["vi", "en"])
        self._text = FIXTURE_VI if self._target_lang == "vi" else FIXTURE_EN
        self._words = self._text.split()
        self._idx = 0

    async def start_utterance(self) -> None:
        self._idx = 0

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        # Simulate some processing delay
        await asyncio.sleep(0.05)
        if self._idx >= len(self._words):
            return None

        # Feed 1-2 words per chunk to simulate streaming
        step = min(2, len(self._words) - self._idx)
        self._idx += step
        return " ".join(self._words[:self._idx])

    def set_result_callback(self, callback) -> None:
        pass

    def snapshot_audio(self) -> bytes:
        return b""

    async def finalize(self, audio_bytes: bytes = b"", prompt: str = "") -> ASRFinal:
        await asyncio.sleep(0.3)
        return ASRFinal(text=self._text, lang=self._target_lang,
                        avg_logprob=-0.3, no_speech_prob=0.05, compression_ratio=1.2)


class FakeMT:
    async def translate_stream(self, text: str, source_lang: str, target_lang: str,
                               context: list, glossary: dict | None) -> AsyncIterator[str]:
        out = FIXTURE_EN if source_lang == "vi" else FIXTURE_VI
        for word in out.split():
            await asyncio.sleep(0.1 + random.random() * 0.1)
            yield word + " "


class FakeTTS:
    def __init__(self):
        self.sample_rate = 48000
        self._sec_per_word = 0.35

    async def stream_pcm16(self, text: str) -> AsyncIterator[bytes]:
        # Simulate audio by generating silence bytes (PCM16)
        total_samples = int(self.sample_rate * len(text.split()) * self._sec_per_word)
        chunk_size = 9600  # ~200ms

        while total_samples > 0:
            n = min(chunk_size, total_samples)
            await asyncio.sleep(0.05)
            # Each sample is 2 bytes for int16
            yield b"\x00\x00" * n
            total_samples -= n
