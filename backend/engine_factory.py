import os
from dataclasses import dataclass
from .interfaces import ASREngine, MTEngine, TTSEngine
from .fakes import FakeASR, FakeMT, FakeTTS
from .vad import VADModel

@dataclass
class Engines:
    asr_vi: ASREngine   # Vietnamese ASR (PhoWhisper medium)
    asr_en: ASREngine   # English ASR (faster-whisper medium)
    mt: MTEngine
    tts: dict[str, TTSEngine]  # key: "vi" or "en"
    vad: VADModel

def build_engines() -> Engines:
    # Check environment variables for mock mode
    fake = os.getenv("USE_FAKE_PIPELINE", "").lower() in ("1", "true", "yes")

    if fake:
        return Engines(
            asr_vi=FakeASR(),
            asr_en=FakeASR(),
            mt=FakeMT(),
            tts={"vi": FakeTTS(), "en": FakeTTS()},
            vad=VADModel(),
        )

    # Remote ASR mode: offload ASR to a GPU service (e.g. Colab notebook
    # colab_asr_gpu.ipynb) via WebSocket. VAD/MT/TTS still run locally.
    remote_asr_url = os.getenv("ASR_REMOTE_URL")
    if remote_asr_url:
        from .asr_remote import RemoteASR
        from .llm_engine import OllamaTranslationEngine
        from .tts_vi import VieNeuEngine
        from .tts_en import PiperTTSEngine
        return Engines(
            asr_vi=RemoteASR(lang="vi", url=remote_asr_url),
            asr_en=RemoteASR(lang="en", url=remote_asr_url),
            mt=OllamaTranslationEngine(),
            tts={
                "vi": VieNeuEngine(),
                "en": PiperTTSEngine(),
            },
            vad=VADModel(),
        )

    # Real Implementations (local CPU)
    from .asr_vietnamese import VietnameseASR  # PhoWhisper medium
    from .asr_english import EnglishASR        # faster-whisper medium
    from .llm_engine import OllamaTranslationEngine
    from .tts_vi import VieNeuEngine
    from .tts_en import PiperTTSEngine

    return Engines(
        asr_vi=VietnameseASR(),
        asr_en=EnglishASR(),
        mt=OllamaTranslationEngine(),
        tts={
            "vi": VieNeuEngine(),
            "en": PiperTTSEngine(),
        },
        vad=VADModel(),
    )
