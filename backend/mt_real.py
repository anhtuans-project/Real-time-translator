import asyncio
import logging
from pathlib import Path
from typing import AsyncIterator
from .interfaces import MTEngine

logger = logging.getLogger("mt_real")

# All models live in backend/models/ (downloaded manually)
MODELS_DIR = Path(__file__).resolve().parent / "models"

class RealMT(MTEngine):
    def __init__(self, model_name: str = str(MODELS_DIR / "nllb-200-distilled-600M")):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._mock = None

        # NLLB uses specific language codes
        self.lang_map = {
            "vi": "vie_Latn",
            "en": "eng_Latn"
        }

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"Loading RealMT model: {model_name}... (This may take a few minutes)")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Successfully loaded RealMT model: {model_name}")
        except Exception as e:
            print(f"Error loading RealMT model {model_name}: {e}")
            print(f"Expected model files at: {MODELS_DIR / 'nllb-200-distilled-600M'}")
            print("Falling back to Mock MT.")
            from .fakes import FakeMT
            self._mock = FakeMT()

    async def translate_stream(
        self, text: str, source_lang: str, target_lang: str,
        context: list, glossary: dict | None
    ) -> AsyncIterator[str]:
        if self.model is None:
            async for delta in self._mock.translate_stream(text, source_lang, target_lang, context, glossary):
                yield delta
            return

        try:
            src_code = self.lang_map.get(source_lang, "eng_Latn")
            tgt_code = self.lang_map.get(target_lang, "vie_Latn")

            # Run translation in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._translate(text, src_code, tgt_code)
            )

            logger.info("MT result [%s→%s]: %s", source_lang, target_lang, result)

            for word in result.split():
                await asyncio.sleep(0.05)
                yield word + " "
        except Exception as e:
            logger.exception("Translation error")
            yield f"[Error: {e}] "

    def _translate(self, text: str, src_code: str, tgt_code: str) -> str:
        """Synchronous translation (runs in thread)."""
        # NLLB tokenizer must know the source language (prefixes input with src lang code)
        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt")

        # Get target language token ID for forced BOS
        # (NllbTokenizer has no lang_code_to_id in transformers v5)
        tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_code)

        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id
        )
        return self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
