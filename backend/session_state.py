import asyncio
import time
import uuid
import logging
from .engine_factory import Engines
from .connection_manager import ConnectionManager
from .vad import VAD

logger = logging.getLogger("session_state")

# Global state constants

# Max utterance duration: ép chốt utterance khi kéo dài quá N giây mà VAD chưa báo
# utterance_end (anti "câu dài vô tận" → finalize beam_size=5 trên cả đoạn = 7s+).
# Mỗi chunk ≤4s + finalize beam_size=1 (~1s) + MT (~1.3s) ≈ <6s/utterance.
MAX_UTT_S = 3.0


class SessionState:
    def __init__(self, session_id: str, manager: ConnectionManager, engines: Engines, languages: tuple[str, ...]):
        self.session_id = session_id
        self.manager = manager
        self.engines = engines
        self.languages = set(languages)

        self.source_lang = "vi"
        self.target_lang = "en"

        self.vad = VAD(self.engines.vad)
        self.context_history: list[tuple[str, str]] = []
        self.glossary: dict | None = None

        self._asr_started = False
        self._last_status = None
        self._utt_start_time: float | None = None  # for max-duration cap

        # Utterance tracking
        self.utterances: dict[str, dict] = {} # utt_id -> {text, translation, status}

        # Streaming state
        self._current_utt_id: str | None = None

        # Backpressure tracking
        self._dropped_chunks = 0

        # Pipeline serialization: chỉ 1 utterance finalize+MT tại lúc (shared ASR
        # engine + CPU contention + Ollama single-model queue). Nhiều utterance
        # concurrent làm mỗi cái chậm lại (test: F->tok utterance1 = 36s do contention).
        self._pipeline_lock = asyncio.Lock()

        self.audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)
        self.worker = asyncio.create_task(self._main_processing_loop())

    def _asr_engine(self):
        return self.engines.asr_vi if self.source_lang == "vi" else self.engines.asr_en

    async def enqueue_audio(self, chunk: bytes):
        try:
            self.audio_q.put_nowait(chunk)
        except asyncio.QueueFull:
            self._dropped_chunks += 1
            if self._dropped_chunks <= 5 or self._dropped_chunks % 100 == 0:
                logger.warning("[%s] Audio queue full, dropped %d chunks total",
                                self.session_id, self._dropped_chunks)

    async def on_control(self, msg: dict):
        msg_type = msg.get("type")
        if msg_type == "start_session":
            self.source_lang = msg.get("source_lang", "vi")
            self.target_lang = msg.get("target_lang", "en")
            logger.info("[%s] Session started: %s -> %s", self.session_id, self.source_lang, self.target_lang)
        elif msg_type == "end_session":
            await self.shutdown()
        elif msg_type == "ping":
            await self.manager.push(self.session_id, {"type": "pong"})

    async def _update_status(self, status: str):
        if self._last_status != status:
            self._last_status = status
            await self.manager.push(self.session_id, {"type": "status", "state": status})

    # Debounced translation removed in favor of utterance-based flow

    async def _main_processing_loop(self):
        """
        Audio -> VAD -> ASR (partial via callback)
        """
        logger.info("[%s] Processing pipeline started", self.session_id)
        # Pipeline được serialize bằng self._pipeline_lock bên trong
        # _finalize_and_pipeline (không cần lock ở đây nữa).

        # Register ASR partial callback
        asr = self._asr_engine()
        asr.set_result_callback(self._on_asr_partial)

        try:
            while True:
                chunk = await self.audio_q.get()
                if chunk is None:
                    break

                # Measure VAD latency
                t0 = time.perf_counter()
                vad_state = self.vad.process(chunk)
                vad_ms = (time.perf_counter() - t0) * 1000

                if vad_state != "silence":
                    logger.debug("[%s] VAD state: %s (%.2fms)", self.session_id, vad_state, vad_ms)

                if vad_state == "speech_ongoing":
                    if not self._asr_started:
                        try:
                            await self._asr_engine().start_utterance()
                        except Exception as e:
                            logger.exception("[%s] start_utterance failed: %s", self.session_id, e)
                            continue
                        self._asr_started = True
                        self._current_utt_id = str(uuid.uuid4())
                        self._utt_start_time = time.perf_counter()
                        logger.info("[%s] ASR started (%s)", self.session_id, self.source_lang)
                    elif self._utt_start_time is not None and \
                            (time.perf_counter() - self._utt_start_time) >= MAX_UTT_S:
                        # Max-duration cap: ép chốt utterance hiện tại, chunk này + kế tiếp
                        # thuộc utterance mới (giữ trải nghiệm realtime, không đợi cả câu dài).
                        logger.info("[%s] ASR max-duration cap (%.1fs)", self.session_id, MAX_UTT_S)
                        utt_id = self._current_utt_id
                        audio_bytes = self._asr_engine().snapshot_audio()
                        self._asr_started = False
                        asyncio.create_task(self._finalize_and_pipeline(utt_id, audio_bytes))
                        # Start utterance mới ngay để không mất chunk hiện tại
                        try:
                            await self._asr_engine().start_utterance()
                        except Exception as e:
                            logger.exception("[%s] start_utterance (cap) failed: %s", self.session_id, e)
                        self._asr_started = True
                        self._current_utt_id = str(uuid.uuid4())
                        self._utt_start_time = time.perf_counter()

                    await self._update_status("listening")

                    try:
                        # Measure ASR feed latency
                        t_asr = time.perf_counter()
                        await self._asr_engine().feed_audio(chunk)
                        asr_ms = (time.perf_counter() - t_asr) * 1000
                        if asr_ms > 10: # Log if it takes significant time
                            logger.debug("[%s] ASR feed: %.2fms", self.session_id, asr_ms)
                    except Exception as e:
                        logger.exception("[%s] Error in ASR feed_audio: %s", self.session_id, e)

                elif vad_state == "utterance_end" and self._asr_started:
                    await self._update_status("silence")
                    logger.info("[%s] ASR utterance_end", self.session_id)
                    # Snapshot utt_id + audio BEFORE scheduling finalize (race-safe):
                    # finalize runs async; next speech chunk's start_utterance() would
                    # otherwise clear the buffer out from under it.
                    utt_id = self._current_utt_id
                    audio_bytes = self._asr_engine().snapshot_audio()
                    self._asr_started = False
                    self._utt_start_time = None
                    asyncio.create_task(self._finalize_and_pipeline(utt_id, audio_bytes))

                elif vad_state == "silence":
                    await self._update_status("silence")

        except Exception as e:
            logger.exception("[%s] Critical error in processing loop: %s", self.session_id, e)

    async def _on_asr_partial(self, text: str):
        """Called by ASR whenever a partial transcript is ready."""
        logger.info("[%s] ASR partial: %s", self.session_id, text)
        await self.manager.push(self.session_id, {
            "type": "partial_transcript",
            "text": text
        })

    async def _finalize_and_pipeline(self, utt_id: str | None = None, audio_bytes: bytes = b""):
        """
        Finalize ASR, translate the full utterance.
        Utterances are tracked in self.utterances for UI history.

        utt_id + audio_bytes are snapshotted by the caller (VAD utterance_end or
        max-duration cap) to avoid races: the finalize task runs async, and a new
        utterance's start_utterance() would otherwise overwrite self._current_utt_id
        and clear the shared ASR buffer.

        Serialized by self._pipeline_lock: chỉ 1 utterance finalize+MT tại lúc
        (shared ASR engine + CPU contention + Ollama single-model queue).
        """
        async with self._pipeline_lock:
            try:
                logger.info("[%s] _finalize_and_pipeline starting (utt_id=%s, %d bytes)",
                            self.session_id, utt_id, len(audio_bytes))

                # 1. Get final transcript from ASR (transcribe the snapshotted audio)
                t_start = time.perf_counter()
                final_text, detected_lang = await self._asr_engine().finalize(audio_bytes)
                asr_finalize_ms = (time.perf_counter() - t_start) * 1000
                logger.info("[%s] ASR finalize: [%s] '%s' (%.2fms)", self.session_id, detected_lang, final_text, asr_finalize_ms)

                if not final_text or not final_text.strip():
                    return

                utt_id = utt_id or self._current_utt_id or str(uuid.uuid4())

                # Track utterance in state
                self.utterances[utt_id] = {
                    "text": final_text,
                    "translation": "",
                    "status": "transcribed"
                }

                # Send final transcript to UI
                await self.manager.push(self.session_id, {
                    "type": "final_transcript",
                    "lang": self.source_lang,
                    "text": final_text,
                    "utt_id": utt_id
                })

                # 2. Translate the finalized text
                self.utterances[utt_id]["status"] = "translating"

                t_mt_start = time.perf_counter()
                accumulated_translation = ""
                async for delta in self.engines.mt.translate_stream(
                    final_text, self.source_lang, self.target_lang, self.context_history, self.glossary
                ):
                    accumulated_translation += delta
                    await self.manager.push(self.session_id, {
                        "type": "translation_delta",
                        "utt_id": utt_id,
                        "text_delta": delta
                    })
                mt_ms = (time.perf_counter() - t_mt_start) * 1000
                logger.info("[%s] MT Translation complete (%.2fms)", self.session_id, mt_ms)

                # Update history and state
                self.context_history.append((final_text, accumulated_translation))
                self.context_history = self.context_history[-5:]
                self.utterances[utt_id]["translation"] = accumulated_translation
                self.utterances[utt_id]["status"] = "translated"

                await self.manager.push(self.session_id, {
                    "type": "translation_done",
                    "utt_id": utt_id,
                    "full_text": accumulated_translation
                })

            except Exception as e:
                logger.exception("[%s] Pipeline failure: %s", self.session_id, e)

    async def shutdown(self):
        if self._asr_started:
            logger.info("[%s] Shutdown: Finalizing pending utterance", self.session_id)
            utt_id = self._current_utt_id
            audio_bytes = self._asr_engine().snapshot_audio()
            self._asr_started = False
            self._utt_start_time = None
            try:
                await self._finalize_and_pipeline(utt_id, audio_bytes)
            except Exception as e:
                logger.exception("[%s] Error finalizing during shutdown: %s", self.session_id, e)

        await self.audio_q.put(None)
        self.worker.cancel()
        try:
            await self.worker
        except asyncio.CancelledError:
            pass
