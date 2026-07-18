import asyncio
import os
import time
import uuid
import logging
from .engine_factory import Engines
from .connection_manager import ConnectionManager
from .vad import VAD

logger = logging.getLogger("session_state")

# Global state constants

# Max utterance duration: ép chốt utterance khi kéo dài quá N giây mà VAD chưa báo
# utterance_end. Lớn hơn = câu dài giữ nguyên 1 khối (đỡ garble/mất ngữ cảnh),
# partial translation live vẫn cho cảm giác realtime. 6s = cân bằng.
MAX_UTT_S = 6.0

# Pre-roll ring buffer: luôn giữ ~300ms audio gần nhất (kể cả khi VAD chưa báo
# speech) -> flush vào ASR khi speech-start để không mất âm tiết đầu (Silero fire
# trễ ~100-200ms). RealtimeSTT dùng cùng cơ chế (pre_recording_buffer_duration).
PREROLL_BYTES = 16000 * 2 * 3 // 10   # ~300ms @ 16kHz int16 mono
# Overlap khi ép chốt MAX_UTT_S: mang ~100ms audio cuối của utterance cũ sang
# utterance mới -> không cắt đôi từ ở biên 6s. StreamSSN dùng 50-180ms.
OVERLAP_BYTES = 16000 * 2 * 1 // 10   # ~100ms

# Adaptive endpointing (Phase 3): silence_chunks_to_end thay đổi theo mật độ speech
# của utterance — câu dài/dày -> pause dài hơn (không cắt giữa câu); câu ngắn -> cắt
# gọn (nói nhanh tách utterance). Deepgram cũng fixed-ms, phần adaptive là dev-side.
# Override qua env để tuning không cần sửa code.
VAD_SILENCE_CHUNKS_DEFAULT = int(os.getenv("VAD_SILENCE_CHUNKS_DEFAULT", "30"))  # ~960ms
VAD_SILENCE_CHUNKS_LONG    = int(os.getenv("VAD_SILENCE_CHUNKS_LONG", "45"))     # ~1440ms
VAD_SILENCE_CHUNKS_SHORT   = int(os.getenv("VAD_SILENCE_CHUNKS_SHORT", "20"))    # ~640ms
VAD_UTT_CHUNKS_LONG  = int(os.getenv("VAD_UTT_CHUNKS_LONG", "150"))   # >~4.8s speech -> nhánh long
VAD_UTT_CHUNKS_SHORT = int(os.getenv("VAD_UTT_CHUNKS_SHORT", "20"))   # <~0.6s -> nhánh short
MIN_UTT_SPEECH_CHUNKS = int(os.getenv("MIN_UTT_SPEECH_CHUNKS", "8"))  # <~256ms + empty final -> drop

# Phase 4e: prompt reuse (init_prompt cho finalize kế -> context coherence). OFF mặc định
# vì rủi ro repetition bias; bật qua env ASR_PROMPT_REUSE=1 sau khi telemetry OK.
ASR_PROMPT_REUSE = os.getenv("ASR_PROMPT_REUSE", "") == "1"
ASR_PROMPT_MAX_WORDS = int(os.getenv("ASR_PROMPT_MAX_WORDS", "6"))

# Phase 5+: partial anti-hallucination. Drop partial có avg_logprob dưới ngưỡng (Whisper
# hallucinate boilerplate YouTube "subscribe/đăng ký/cảm ơn" trên silence/low-info audio —
# vd_filter=True cắt phần lớn, guard này là backstop). Local engine không gửi confidence
# (default 0.0 -> pass). Tune qua env sau khi xem log distribution.
ASR_PARTIAL_MIN_LOGPROB = float(os.getenv("ASR_PARTIAL_MIN_LOGPROB", "-1.0"))


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

        # Pre-roll ring buffer (luôn giữ ~300ms audio gần nhất; flush khi speech-start
        # để không mất âm đầu). Phase 2.
        self._preroll = bytearray()

        # Phase 3: adaptive endpointing — đếm speech chunks của utterance hiện tại để
        # chỉnh self.vad.silence_chunks_to_end; + min-utterance discard.
        self._utt_speech_chunks = 0

        # Phase 4e: prompt reuse hook (OFF mặc định — xem ASR_PROMPT_REUSE).
        self._last_final_text = ""

        # Partial translation preview: dịch partial transcript trong lúc nói
        # (debounce + cancel-in-flight) -> UI hiện bản dịch dần, chốt ở utterance_end.
        self._last_partial_text = ""
        self._partial_debounce_task: asyncio.Task | None = None
        self._partial_tr_task: asyncio.Task | None = None
        self._partial_debounce_s = 0.7

        # LocalAgreement-2 partial stabilization (ufal/whisper_streaming): commit LCP
        # của 2 partial liên tiếp làm "confirmed" (locked, chỉ grow trong 1 utterance);
        # phần còn lại là suffix mutable. UI render confirmed (solid) + suffix (italic)
        # -> không nhấp nháy/rewind. Reset ở mỗi start_utterance.
        self._prev_partial = ""
        self._confirmed_text = ""

        # Phase 5c: tách lock. _asr_lock chỉ wrap ASR finalize + guards + push
        # final_transcript (phần tạo transcript hiển thị). MT chạy KHÔNG lock —
        # Ollama có single-model queue riêng; frontend append theo utt_id nên order
        # chỉ quan trọng trong utt (đã preserve). Trước đây 1 lock bao cả ASR+MT
        # làm ASR finalize utterance N+1 phải chờ MT utterance N xong.
        self._asr_lock = asyncio.Lock()

        # Phase 5d: rolling latency metrics (keep last 50 mỗi loại) + push週期 5s.
        self._metric_asr: list[float] = []
        self._metric_mt: list[float] = []
        self._metric_tts: list[float] = []
        self._metric_partial_count = 0
        self._metrics_push_task: asyncio.Task | None = None

        self.audio_q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=500)
        self.worker = asyncio.create_task(self._main_processing_loop())

        # Phase 5b: register connection-state callback trên RemoteASR (engines shared
        # nên đăng ký cho cả asr_vi + asr_en; local engine không có set_state_callback).
        for _asr in (self.engines.asr_vi, self.engines.asr_en):
            _cb = getattr(_asr, "set_state_callback", None)
            if _cb:
                _cb(self._on_asr_connection_state)
        self._metrics_push_task = asyncio.create_task(self._metrics_push_loop())

    def _asr_engine(self):
        return self.engines.asr_vi if self.source_lang == "vi" else self.engines.asr_en

    async def _flush_preroll_into_asr(self):
        """Feed ~300ms pre-roll (audio gần nhất trước speech-start) vào ASR để không
        mất âm đầu (Silero fire trễ); cũng làm overlap khi ép chốt MAX_UTT_S. Preroll
        lags 1 chunk (extend ở cuối loop) nên không double-feed chunk hiện tại."""
        if self._preroll:
            try:
                await self._asr_engine().feed_audio(bytes(self._preroll))
            except Exception as e:
                logger.warning("[%s] preroll feed failed: %s", self.session_id, e)

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
        # Pipeline ASR được serialize bằng self._asr_lock bên trong
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
                        await self._flush_preroll_into_asr()   # pre-roll: không mất âm đầu
                        self._reset_partial_stabilizer()
                        self._utt_speech_chunks = 0
                        self.vad.silence_chunks_to_end = VAD_SILENCE_CHUNKS_DEFAULT
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
                        await self._flush_preroll_into_asr()   # overlap biên ~300ms
                        self._reset_partial_stabilizer()
                        self._utt_speech_chunks = 0
                        self.vad.silence_chunks_to_end = VAD_SILENCE_CHUNKS_DEFAULT
                        self._asr_started = True
                        self._current_utt_id = str(uuid.uuid4())
                        self._utt_start_time = time.perf_counter()

                    # Phase 3 adaptive endpointing: đếm speech chunks + chỉnh silence
                    # tolerance theo mật độ utterance (câu dài -> pause dài, ngắn -> gọn).
                    self._utt_speech_chunks += 1
                    if self._utt_speech_chunks > VAD_UTT_CHUNKS_LONG:
                        self.vad.silence_chunks_to_end = VAD_SILENCE_CHUNKS_LONG
                    elif self._utt_speech_chunks < VAD_UTT_CHUNKS_SHORT:
                        self.vad.silence_chunks_to_end = VAD_SILENCE_CHUNKS_SHORT
                    else:
                        self.vad.silence_chunks_to_end = VAD_SILENCE_CHUNKS_DEFAULT

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

                # Rolling pre-roll buffer: luôn giữ ~300ms audio gần nhất. Lags 1 chunk
                # (extend sau khi đã feed chunk hiện tại) nên khi speech-start flush ở
                # đầu iteration kế tiếp, không double-feed chunk hiện tại.
                self._preroll.extend(chunk)
                if len(self._preroll) > PREROLL_BYTES:
                    del self._preroll[:-PREROLL_BYTES]

        except Exception as e:
            logger.exception("[%s] Critical error in processing loop: %s", self.session_id, e)

    @staticmethod
    def _lcp(a: str, b: str) -> str:
        """Word-level longest common prefix của 2 partial hypothesis (LocalAgreement-2)."""
        aw, bw = a.split(), b.split()
        n = min(len(aw), len(bw))
        i = 0
        while i < n and aw[i] == bw[i]:
            i += 1
        return " ".join(aw[:i])

    def _reset_partial_stabilizer(self):
        """Xóa trạng thái stabilized partial — gọi ở mỗi start_utterance."""
        self._prev_partial = ""
        self._confirmed_text = ""

    def _init_prompt(self) -> str:
        """Init prompt cho finalize kế (context coherence, chống mất ngữ cảnh giữa utterance).
        OFF mặc định — chỉ trả prompt khi ASR_PROMPT_REUSE=1 và last final ngắn
        (<= ASR_PROMPT_MAX_WORDS) để tránh bias repetition. Hook sẵn cho follow-up."""
        if not ASR_PROMPT_REUSE or not self._last_final_text:
            return ""
        words = self._last_final_text.split()
        if len(words) > ASR_PROMPT_MAX_WORDS:
            return ""
        return self._last_final_text

    # ---- Phase 5d: metrics ----

    def _record_metric(self, bucket: list[float], ms: float):
        bucket.append(ms)
        if len(bucket) > 50:
            del bucket[:-50]

    @staticmethod
    def _avg(xs: list[float]) -> float | None:
        if not xs:
            return None
        return sum(xs) / len(xs)

    def _stale_drops(self) -> int:
        """Audio drops ở RemoteASR send queue (Phase 5a). Local engine không có."""
        asr = self._asr_engine()
        return getattr(asr, "stale_drop_count", 0)

    def _metrics_snapshot(self) -> dict:
        return {
            "type": "metrics",
            "asr_finalize_ms": self._avg(self._metric_asr),
            "mt_ms": self._avg(self._metric_mt),
            "tts_ms": self._avg(self._metric_tts),
            "dropped_chunks": self._dropped_chunks,
            "stale_drops": self._stale_drops(),
            "partial_count": self._metric_partial_count,
        }

    async def _metrics_push_loop(self):
        """Push metrics snapshot lên UI mỗi 5s (rolling avg, debounced)."""
        try:
            while True:
                await asyncio.sleep(5)
                try:
                    await self.manager.push(self.session_id, self._metrics_snapshot())
                except Exception as e:
                    logger.warning("[%s] metrics push failed: %s", self.session_id, e)
        except asyncio.CancelledError:
            pass

    async def _on_asr_connection_state(self, state: str):
        """Phase 5b: RemoteASR fire state ∈ {connected,disconnected,reconnecting};
        đẩy lên UI để hiển thị banner trạng thái ASR GPU."""
        try:
            await self.manager.push(self.session_id, {
                "type": "asr_connection", "state": state
            })
            if state != "connected":
                logger.info("[%s] ASR connection: %s", self.session_id, state)
        except Exception as e:
            logger.warning("[%s] asr_connection push failed: %s", self.session_id, e)

    async def _on_asr_partial(self, text: str, no_speech_prob: float = 0.0, avg_logprob: float = 0.0):
        """Called by ASR whenever a partial transcript is ready.

        LocalAgreement-2: confirmed = LCP(prev, cur) (locked, chỉ grow); suffix =
        phần cur sau confirmed (mutable). Push cả 3 field (confirmed/partial/text)
        — frontend dùng confirmed+partial render solid+italic; text giữ back-compat.

        Phase 5+: drop low-confidence partial (hallucination boilerplate trên silence).
        RemoteASR gửi avg_logprob/no_speech_prob; local engine dùng default 0.0 (pass).
        Bỏ qua partial thấp tự tin: không push UI, không update _prev_partial ->
        hallucinated partial không flicker và không bị LocalAgreement lock.
        """
        if avg_logprob < ASR_PARTIAL_MIN_LOGPROB:
            logger.debug("[%s] Drop low-conf partial (lp=%.2f nsp=%.2f): %r",
                         self.session_id, avg_logprob, no_speech_prob, text[:60])
            return
        if self._prev_partial:
            new_confirmed = self._lcp(self._prev_partial, text)
        else:
            new_confirmed = ""
        # confirmed chỉ grow trong utterance — không shrink khi 1 partial rewind
        if len(new_confirmed) >= len(self._confirmed_text):
            self._confirmed_text = new_confirmed
        cwords = self._confirmed_text.split()
        words = text.split()
        suffix = " ".join(words[len(cwords):]) if len(words) > len(cwords) else ""
        combined = (self._confirmed_text + " " + suffix).strip() if self._confirmed_text else suffix

        logger.info("[%s] ASR partial: %r | confirmed=%r suffix=%r",
                    self.session_id, text, self._confirmed_text, suffix)
        self._metric_partial_count += 1
        await self.manager.push(self.session_id, {
            "type": "partial_transcript",
            "confirmed": self._confirmed_text,
            "partial": suffix,
            "text": combined,   # back-compat cho frontend chưa nâng cấp
        })
        self._prev_partial = text
        # Schedule a live translation preview of the partial (replaceable).
        self._last_partial_text = combined
        self._schedule_partial_translation()

    def _schedule_partial_translation(self):
        """(Re)start the debounce timer for a partial translation preview."""
        if self._partial_debounce_task and not self._partial_debounce_task.done():
            self._partial_debounce_task.cancel()
        try:
            self._partial_debounce_task = asyncio.create_task(
                self._partial_debounce_then_translate())
        except RuntimeError:
            pass

    async def _partial_debounce_then_translate(self):
        try:
            await asyncio.sleep(self._partial_debounce_s)
        except asyncio.CancelledError:
            return
        text = self._last_partial_text
        if not text.strip():
            return
        # Cancel any in-flight partial translation before starting a new one.
        if self._partial_tr_task and not self._partial_tr_task.done():
            self._partial_tr_task.cancel()
        try:
            self._partial_tr_task = asyncio.create_task(self._do_partial_translate(text))
        except RuntimeError:
            pass

    async def _do_partial_translate(self, text: str):
        """Translate a partial transcript and stream a replaceable preview."""
        accumulated = ""
        try:
            async for delta in self.engines.mt.translate_stream(
                text, self.source_lang, self.target_lang,
                self.context_history, self.glossary
            ):
                accumulated += delta
                if accumulated.strip():
                    await self.manager.push(self.session_id, {
                        "type": "partial_translation",
                        "text": accumulated
                    })
        except asyncio.CancelledError:
            return
        except Exception as e:
            # Preview errors are non-fatal — the final translation is authoritative.
            logger.warning("[%s] partial translate failed: %s", self.session_id, e)

    def _cancel_partial_translation(self):
        """Cancel any pending/in-flight partial translation preview."""
        if self._partial_debounce_task and not self._partial_debounce_task.done():
            self._partial_debounce_task.cancel()
        if self._partial_tr_task and not self._partial_tr_task.done():
            self._partial_tr_task.cancel()

    async def _finalize_and_pipeline(self, utt_id: str | None = None, audio_bytes: bytes = b""):
        """
        Finalize ASR, translate the full utterance.
        Utterances are tracked in self.utterances for UI history.

        utt_id + audio_bytes are snapshotted by the caller (VAD utterance_end or
        max-duration cap) to avoid races: the finalize task runs async, and a new
        utterance's start_utterance() would otherwise overwrite self._current_utt_id
        and clear the shared ASR buffer.

        Phase 5c: lock tách làm 2 — _asr_lock chỉ wrap ASR finalize + guards + push
        final_transcript (phần tạo transcript hiển thị, ngắn). MT + TTS chạy KHÔNG
        lock: Ollama có single-model queue riêng, frontend append theo utt_id nên
        order chỉ quan trọng trong utt (đã preserve). Trước đây 1 lock bao cả ASR+MT
        làm ASR finalize utterance N+1 phải chờ MT utterance N xong → tăng latency.
        """
        # --- Phase 5c STAGE 1 (locked): ASR finalize + guards + push transcript ---
        final_text = ""
        final_utt_id = utt_id
        async with self._asr_lock:
            try:
                logger.info("[%s] _finalize_and_pipeline starting (utt_id=%s, %d bytes)",
                            self.session_id, utt_id, len(audio_bytes))

                # Stop the live preview translation; the final translation below
                # is authoritative. Clear the preview box in the UI.
                self._cancel_partial_translation()
                self._reset_partial_stabilizer()
                await self.manager.push(self.session_id, {
                    "type": "partial_translation", "text": ""
                })

                # 1. Get final transcript from ASR (transcribe the snapshotted audio)
                t_start = time.perf_counter()
                res = await self._asr_engine().finalize(audio_bytes, prompt=self._init_prompt())
                final_text, detected_lang = res.text, res.lang
                asr_finalize_ms = (time.perf_counter() - t_start) * 1000
                self._record_metric(self._metric_asr, asr_finalize_ms)
                logger.info("[%s] ASR finalize: [%s] '%s' (%.2fms, lp=%.2f nsp=%.2f cr=%.2f)",
                            self.session_id, detected_lang, final_text, asr_finalize_ms,
                            res.avg_logprob, res.no_speech_prob, res.compression_ratio)
                micro = self._utt_speech_chunks < MIN_UTT_SPEECH_CHUNKS
                self._utt_speech_chunks = 0   # reset cho utterance kế (dù drop hay giữ)

                if not final_text or not final_text.strip():
                    # Phase 3: log micro-utterance (cough/click) riêng để dễ phân biệt
                    # với empty final do bug. Empty final đã có Phase 0 fallback phía
                    # Colab; rỗng ở đây = không có partial nào cả (micro-burst).
                    if micro:
                        logger.info("[%s] Drop micro-utterance (empty final)", self.session_id)
                    return

                # Phase 4d: drop low-confidence final (silence / decode kém tự tin). Fallback
                # partial phía Colab set confidence neutral (nsp=0,lp=0) nên không bị drop ở đây.
                if res.no_speech_prob > 0.8 or res.avg_logprob < -1.2:
                    logger.info("[%s] Drop low-confidence final (nsp=%.2f lp=%.2f): %r",
                                self.session_id, res.no_speech_prob, res.avg_logprob, final_text[:80])
                    return

                # Drop Whisper trailing-audio hallucination: khi đoạn nhạc/junk ở cuối
                # bị VAD nhặt thành utterance mới, Whisper thường bidiện ra câu giống
                # utterance trước (decoder loop trên audio low-info). Bỏ qua final
                # trùng y hệt utterance liền trước — self-repetition hợp lệ liên tiếp
                # là cực hiếm nên an toàn.
                if self.utterances:
                    last_text = list(self.utterances.values())[-1].get("text", "")
                    if last_text and final_text.strip() == last_text.strip():
                        logger.info("[%s] Drop duplicate final (hallucination): %r",
                                    self.session_id, final_text[:80])
                        self._last_final_text = ""   # clear để không reuse prompt gây repetition
                        return
                self._last_final_text = final_text   # cho prompt reuse (OFF mặc định)

                utt_id = utt_id or self._current_utt_id or str(uuid.uuid4())
                final_utt_id = utt_id

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
                self.utterances[utt_id]["status"] = "translating"

            except Exception as e:
                logger.exception("[%s] Pipeline failure: %s", self.session_id, e)
                return

        # --- Phase 5c STAGE 2 (unlocked): MT streaming ---
        # Ollama có single-model queue riêng; context_history append/slice là atomic
        # (không await giữa), mỗi utt_id độc lập → concurrent MT an toàn.
        if not final_text.strip():
            return
        t_mt_start = time.perf_counter()
        accumulated_translation = ""
        try:
            async for delta in self.engines.mt.translate_stream(
                final_text, self.source_lang, self.target_lang, self.context_history, self.glossary
            ):
                accumulated_translation += delta
                await self.manager.push(self.session_id, {
                    "type": "translation_delta",
                    "utt_id": final_utt_id,
                    "text_delta": delta
                })
        except Exception as e:
            logger.exception("[%s] MT failure: %s", self.session_id, e)
            # Vẫn đẩy translation_done rỗng để UI không kẹt ở "đang dịch…".
            await self.manager.push(self.session_id, {
                "type": "translation_done", "utt_id": final_utt_id, "full_text": accumulated_translation
            })
            return
        mt_ms = (time.perf_counter() - t_mt_start) * 1000
        self._record_metric(self._metric_mt, mt_ms)
        logger.info("[%s] MT Translation complete (%.2fms)", self.session_id, mt_ms)

        # Update history and state
        self.context_history.append((final_text, accumulated_translation))
        self.context_history = self.context_history[-5:]
        self.utterances[final_utt_id]["translation"] = accumulated_translation
        self.utterances[final_utt_id]["status"] = "translated"

        await self.manager.push(self.session_id, {
            "type": "translation_done",
            "utt_id": final_utt_id,
            "full_text": accumulated_translation
        })

        # --- Phase 5c STAGE 3 (unlocked): TTS stream ---
        # Chạy ngoài lock để utterance kế không phải chờ synth xong; frontend tự
        # interrupt audio chồng nhau trên mỗi tts_start.
        if accumulated_translation.strip():
            await self._synthesize_and_stream_tts(final_utt_id, accumulated_translation)

    async def _synthesize_and_stream_tts(self, utt_id: str, text: str):
        """Synthesize the translation text with Piper and stream PCM16 to the client."""
        tts_engine = self.engines.tts.get(self.target_lang)
        if tts_engine is None:
            return
        try:
            await self.manager.push(self.session_id, {
                "type": "tts_start",
                "utt_id": utt_id,
                "sample_rate": tts_engine.sample_rate,
            })
            t_tts = time.perf_counter()
            n_chunks = 0
            async for pcm in tts_engine.stream_pcm16(text):
                await self.manager.push_bytes(self.session_id, pcm)
                n_chunks += 1
            tts_ms = (time.perf_counter() - t_tts) * 1000
            self._record_metric(self._metric_tts, tts_ms)
            logger.info("[%s] TTS complete (%.2fms, %d chunks, %d chars)",
                        self.session_id, tts_ms, n_chunks, len(text))
        except Exception as e:
            logger.exception("[%s] TTS failure: %s", self.session_id, e)
        finally:
            await self.manager.push(self.session_id, {"type": "tts_end", "utt_id": utt_id})

    async def shutdown(self):
        self._cancel_partial_translation()
        if self._metrics_push_task and not self._metrics_push_task.done():
            self._metrics_push_task.cancel()
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
