"""
RemoteASR — ASR engine that streams PCM16 audio to a remote GPU WebSocket ASR
service (e.g. the Colab notebook colab_asr_gpu.ipynb) and surfaces partial
transcripts via the same callback contract as the local faster-whisper engines.

DESIGN GOAL: never block the session processing loop on the network. The local
CPU engine decouples via a thread pool + local buffer; this engine decouples by
connecting in a BACKGROUND task and only ever put_nowait()-ing into an unbounded
send queue from start_utterance()/feed_audio()/snapshot_audio(). The single
writer task drains the queue over the WS when connected; if not yet connected,
chunks simply wait in the queue. finalize() (which runs in its own task, not the
loop) is the only place that may await the connection.

Protocol (JSON text + binary PCM16 chunks, all ordered by the WS):
  client -> server:
    {"action":"start","lang":"vi"}    # begin new utterance, clear server buffer
    <binary PCM16 16kHz chunk>        # append to current utterance buffer
    {"action":"finalize"}            # transcribe current buffer (final), then clear
  server -> client:
    {"type":"partial","text":"..."}  # unsolicited, while buffer grows
    {"type":"final","text":"...","lang":"vi"}  # in reply to finalize

Ordering: finalize requests and their responses stay FIFO (single WS). We pair
them with a deque of Futures — snapshot_audio() queues a Future + sends
finalize; the reader pops the oldest Future and sets the final result.
finalize() peeks deque[0] and awaits it. Because session_state serializes
utterance finalization (_pipeline_lock), finalize(A) completes (reader pops
F_A) before finalize(B) peeks, so deque[0] is always the right Future.
"""
import asyncio
import json
import logging
import os
import random
from collections import deque
from typing import Tuple

from .interfaces import ASREngine, ASRFinal

logger = logging.getLogger("asr_remote")

CONNECT_TIMEOUT = 15  # seconds
# beam_size=1 greedy: finalize ~2-5s với buffer 6s trên T4 (Colab transcribe cả buffer +
# fallback về partial nếu rỗng). 30s headroom an toàn; nếu hang thì block tối đa 30s.
FINALIZE_TIMEOUT = 30  # seconds

# Phase 5a: bound send queue để memory không grow vô hạn khi GPU stall. 600 chunks
# ~19s audio @ 32ms/chunk — đủ backlog cho blip ngắn, drop-oldest khi vượt (realtime
# > completeness). RealtimeSTT dùng cùng思路 (allowed_latency_limit drop).
SEND_QUEUE_MAX = int(os.getenv("ASR_SEND_QUEUE_MAX", "600"))
# Phase 5b: reconnect backoff cap. 2^attempts giây (jittered) tới 30s, thử tối đa 8 lần.
MAX_CONNECT_ATTEMPTS = int(os.getenv("ASR_CONNECT_MAX_ATTEMPTS", "8"))
RECONNECT_MAX_DELAY = 30


class RemoteASR(ASREngine):
    def __init__(self, lang: str = "vi", url: str | None = None):
        self.lang = lang
        self.url = url or os.getenv("ASR_REMOTE_URL")
        if not self.url:
            raise RuntimeError("ASR_REMOTE_URL not set")
        self._ws = None
        # Phase 5a: bounded send queue — drop-oldest khi đầy (realtime > completeness),
        # drain stale trên disconnect để không flush audio cũ sang server mới.
        self._send_q: asyncio.Queue = asyncio.Queue(maxsize=SEND_QUEUE_MAX)
        self._stale_drop_count = 0
        self._writer_task: asyncio.Task | None = None
        self._reader_task: asyncio.Task | None = None
        self._connect_task: asyncio.Task | None = None
        self._connected = False
        self._connected_event = asyncio.Event()
        self._result_callback = None
        self._pending_finals: deque = deque()
        # Phase 5b: connection-state callbacks (engines shared across sessions → list).
        # Mỗi SessionState đăng ký 1 async cb(state) để push asr_connection lên UI của nó.
        self._state_callbacks = []

    # ---- connection (background, never blocks the loop) ----

    def _ensure_connect_bg(self):
        """Start a connect task if not connected and none in flight. Returns
        immediately (non-blocking). Safe to call from sync context too, but we
        only call it from async loop methods."""
        if self._connected or self._connect_task is not None:
            return
        try:
            self._connect_task = asyncio.create_task(self._do_connect())
        except RuntimeError:
            # No running loop (shouldn't happen in our call sites)
            pass

    def set_state_callback(self, cb) -> None:
        """Register an async cb(state) for connection-state changes. Engines are
        shared across sessions, so multiple sessions may register (each pushes to
        its own UI). state ∈ {"connected","disconnected","reconnecting"}."""
        self._state_callbacks.append(cb)

    def _fire_state(self, state: str):
        for cb in list(self._state_callbacks):
            try:
                asyncio.create_task(cb(state))
            except Exception as e:
                logger.warning("RemoteASR state callback error: %s", e)

    async def _do_connect(self):
        # Phase 5b: reconnect với exponential backoff + jitter. Thử tối đa
        # MAX_CONNECT_ATTEMPTS lần; mỗi lần fail -> "reconnecting" + sleep; success ->
        # "connected" + reset; hết attempts -> "disconnected" (sẽ retry lần kế khi
        # feed_audio/start_utterance gọi _ensure_connect_bg lại).
        attempts = 0
        try:
            while True:
                try:
                    import websockets
                    logger.info("RemoteASR connecting to %s (attempt %d) ...",
                                self.url, attempts + 1)
                    ws = await asyncio.wait_for(
                        websockets.connect(self.url, max_size=None, ping_interval=20),
                        timeout=CONNECT_TIMEOUT,
                    )
                    self._ws = ws
                    self._connected = True
                    self._connected_event.set()
                    self._writer_task = asyncio.create_task(self._writer_loop())
                    self._reader_task = asyncio.create_task(self._reader_loop())
                    self._fire_state("connected")
                    logger.info("RemoteASR connected (lang=%s).", self.lang)
                    return
                except Exception as e:
                    attempts += 1
                    if attempts >= MAX_CONNECT_ATTEMPTS:
                        logger.error("RemoteASR connect gave up after %d attempts: %s",
                                     attempts, e)
                        self._fire_state("disconnected")
                        return
                    delay = min(2 ** attempts, RECONNECT_MAX_DELAY) * (0.5 + random.random() * 0.5)
                    logger.warning("RemoteASR connect failed (attempt %d): %s — retry in %.1fs",
                                   attempts, e, delay)
                    self._fire_state("reconnecting")
                    await asyncio.sleep(delay)
        finally:
            self._connect_task = None

    async def _ensure_connected_blocking(self):
        """Used by finalize() (runs in its own task, may await). Returns True if
        connected (now or after waiting for the in-flight connect)."""
        if self._connected:
            return True
        self._ensure_connect_bg()
        if self._connect_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._connect_task), timeout=CONNECT_TIMEOUT)
            except Exception:
                pass
        return self._connected

    def _handle_disconnect(self):
        if not self._connected:
            return
        self._connected = False
        self._connected_event.clear()
        # Fail any pending finals so finalize() doesn't hang forever.
        while self._pending_finals:
            fut = self._pending_finals.popleft()
            if not fut.done():
                fut.set_result(ASRFinal(text="", lang=self.lang))
        # Phase 5a: drain stale queued audio/control để reconnect sau không flush
        # 19s audio cũ sang server mới (gây transcribe sai/garble utterance đã qua).
        drained = 0
        while True:
            try:
                self._send_q.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break
        if drained:
            logger.info("RemoteASR drained %d stale queued msgs on disconnect.", drained)
        self._fire_state("disconnected")
        logger.warning("RemoteASR disconnected; will reconnect on next use.")

    def _put(self, msg):
        """put_nowait vào bounded _send_q; drop-oldest khi đầy (realtime > completeness).
        Audio cũ bị drop thay vì chặn producer hay grow memory khi GPU stall."""
        try:
            self._send_q.put_nowait(msg)
            return
        except asyncio.QueueFull:
            pass
        try:
            dropped = self._send_q.get_nowait()
            self._stale_drop_count += 1
            kind = "audio" if isinstance(dropped, (bytes, bytearray)) else "control"
            if self._stale_drop_count <= 5 or self._stale_drop_count % 100 == 0:
                logger.warning("RemoteASR send queue full, dropped %s msg (#%d total)",
                               kind, self._stale_drop_count)
        except asyncio.QueueEmpty:
            pass
        try:
            self._send_q.put_nowait(msg)
        except asyncio.QueueFull:
            self._stale_drop_count += 1
            logger.warning("RemoteASR send queue still full after drop; lost 1 msg (#%d)",
                           self._stale_drop_count)

    @property
    def stale_drop_count(self) -> int:
        return self._stale_drop_count

    # ---- writer / reader ----

    async def _writer_loop(self):
        await self._connected_event.wait()
        try:
            while True:
                msg = await self._send_q.get()
                if msg is None:
                    break
                try:
                    await self._ws.send(msg)
                except Exception as e:
                    logger.warning("RemoteASR send error: %s", e)
                    self._handle_disconnect()
                    return
        except Exception as e:
            logger.exception("RemoteASR writer loop error: %s", e)
            self._handle_disconnect()

    async def _reader_loop(self):
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                t = data.get("type")
                if t == "partial":
                    text = data.get("text", "")
                    if text and self._result_callback:
                        # Phase 5+: forward confidence (Colab gửi kèm) để backend filter
                        # hallucinated partial trên silence/low-info audio.
                        lp = float(data.get("avg_logprob", 0.0))
                        nsp = float(data.get("no_speech_prob", 0.0))
                        try:
                            asyncio.create_task(self._result_callback(text, nsp, lp))
                        except Exception as e:
                            logger.error("RemoteASR partial callback error: %s", e)
                elif t == "final":
                    try:
                        fut = self._pending_finals.popleft()
                    except IndexError:
                        logger.warning("RemoteASR final without pending future")
                        continue
                    if not fut.done():
                        # Phase 4: Colab server gửi kèm confidence fields (forward-compat:
                        # server cũ không gửi -> .get() default).
                        fut.set_result(ASRFinal(
                            text=data.get("text", ""),
                            lang=data.get("lang", self.lang),
                            avg_logprob=float(data.get("avg_logprob", 0.0)),
                            no_speech_prob=float(data.get("no_speech_prob", 1.0)),
                            compression_ratio=float(data.get("compression_ratio", 1.0)),
                        ))
        except Exception as e:
            logger.warning("RemoteASR reader loop ended: %s", e)
        self._handle_disconnect()

    # ---- ASREngine interface (all non-blocking from the loop's perspective) ----

    async def start_utterance(self) -> None:
        self._ensure_connect_bg()
        self._put(json.dumps({"action": "start", "lang": self.lang}))

    async def feed_audio(self, pcm16_chunk: bytes) -> str | None:
        if not self._connected:
            self._ensure_connect_bg()
        self._put(pcm16_chunk)
        return None

    def set_result_callback(self, callback) -> None:
        self._result_callback = callback

    def snapshot_audio(self) -> bytes:
        """Sync + race-safe. Audio lives on the server, so we return b"" (finalize
        ignores it). Queue a Future + send finalize in WS order so the server
        finalizes THIS utterance's buffer before the next start clears it."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending_finals.append(fut)
        self._put(json.dumps({"action": "finalize"}))
        return b""

    async def finalize(self, audio_bytes: bytes = b"", prompt: str = "") -> ASRFinal:
        # prompt (init_prompt) không được thread sang Colab server (protocol chưa có);
        # ship OFF mặc định nên prompt="" luôn. Hook sẵn cho follow-up.
        if not await self._ensure_connected_blocking():
            logger.error("RemoteASR not connected at finalize; returning empty.")
            return ASRFinal(text="", lang=self.lang)
        if not self._pending_finals:
            loop = asyncio.get_event_loop()
            fut = loop.create_future()
            self._pending_finals.append(fut)
            self._put(json.dumps({"action": "finalize"}))
        fut = self._pending_finals[0]
        try:
            return await asyncio.wait_for(fut, timeout=FINALIZE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("RemoteASR finalize timed out (%ss)", FINALIZE_TIMEOUT)
            try:
                self._pending_finals.popleft()
            except IndexError:
                pass
            return ASRFinal(text="", lang=self.lang)