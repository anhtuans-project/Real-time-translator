from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass
class ASRFinal:
    """Kết quả finalize ASR + metadata confidence (Phase 4) cho low-confidence filtering.

    - avg_logprob: trung bình (word-weighted) log-prob các segment (cao = tự tin).
    - no_speech_prob: max no_speech_prob các segment (cao = khả năng là silence).
    - compression_ratio: max compression_ratio các segment (cao = lặp/garble).
    - last_word_end: end-time của từ cuối (cho word-timing endpoint — Phase 3 defer).
    """
    text: str
    lang: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 1.0
    compression_ratio: float = 1.0
    last_word_end: float | None = None


def asr_final_from_segments(segments, lang: str) -> ASRFinal:
    """Build ASRFinal từ faster-whisper segments: text + confidence metadata.

    avg_logprob word-weighted (weight = số word trong segment); no_speech_prob /
    compression_ratio lấy max (1 segment xấu đủ để nghi ngờ).
    """
    text = " ".join(s.text.strip() for s in segments).strip()
    segs = list(segments)
    if not segs:
        return ASRFinal(text=text, lang=lang)
    total_words = 0
    weighted_lp = 0.0
    max_nsp = 0.0
    max_cr = 1.0
    for s in segs:
        wc = max(1, len(s.text.split()))
        total_words += wc
        weighted_lp += (getattr(s, "avg_logprob", 0.0) or 0.0) * wc
        max_nsp = max(max_nsp, getattr(s, "no_speech_prob", 0.0) or 0.0)
        max_cr = max(max_cr, getattr(s, "compression_ratio", 1.0) or 1.0)
    avg_lp = weighted_lp / total_words if total_words else 0.0
    return ASRFinal(
        text=text, lang=lang,
        avg_logprob=avg_lp, no_speech_prob=max_nsp, compression_ratio=max_cr,
    )


class ASREngine(Protocol):
    async def start_utterance(self) -> None:
        """Initialize ASR state for a new utterance."""
        ...

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        """
        Processes a PCM16 audio chunk.
        Returns a partial transcript if available, otherwise None.
        """
        ...

    def set_result_callback(self, callback) -> None:
        """Set a callback for partial transcript results."""
        ...

    def snapshot_audio(self) -> bytes:
        """
        Sync, race-safe: grab + clear the current utterance audio buffer.
        Called before scheduling the async finalize so the next utterance cannot
        clear the buffer out from under finalize.
        """
        ...

    async def finalize(self, audio_bytes: bytes = b"", prompt: str = "") -> ASRFinal:
        """
        Transcribe the snapshotted utterance audio. Returns ASRFinal (text + lang +
        confidence metadata). `prompt` (init_prompt) cho context coherence — ship
        OFF mặc định (rủi ro repetition bias), bật qua env ở SessionState.
        """
        ...

class MTEngine(Protocol):
    def translate_stream(
        self, text: str, source_lang: str, target_lang: str,
        context: list[tuple[str, str]], glossary: dict | None
    ) -> AsyncIterator[str]:
        """
        Streams translation deltas for the given text.
        """
        ...

class TTSEngine(Protocol):
    sample_rate: int  # The output sample rate of the engine

    async def stream_pcm16(self, text: str) -> AsyncIterator[bytes]:
        """
        Streams PCM16 audio bytes for the given text.
        """
        ...