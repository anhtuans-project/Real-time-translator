import { useEffect, useRef, useState, useCallback } from 'react';

type Utterance = {
  uttId: string;
  sourceLang: string;
  sourceText: string;
  targetText: string;
  targetReady: boolean;
};

const WS_BASE_URL = 'ws://localhost:8000';
const INITIAL_RETRY_DELAY_MS = 500;
const MAX_RETRY_DELAY_MS = 30000;

export function useTranslatorSocket(sessionId: string) {
  const wsRef = useRef<WebSocket | null>(null);
  const [utterances, setUtterances] = useState<Utterance[]>([]);
  // LocalAgreement-2: confirmed prefix (solid, locked) + unstable suffix (italic, mutable).
  const [confirmedPartial, setConfirmedPartial] = useState('');
  const [unstablePartial, setUnstablePartial] = useState('');
  const currentPartial = (confirmedPartial + ' ' + unstablePartial).trim();  // back-compat
  const [partialTranslation, setPartialTranslation] = useState('');
  const [status, setStatus] = useState<'listening' | 'processing' | 'silence'>('silence');
  const [wsConnected, setWsConnected] = useState(false);
  // Phase 5b: RemoteASR (Colab GPU) connection state — riêng với wsConnected (FE↔BE).
  const [asrConnection, setAsrConnection] = useState<'connected' | 'disconnected' | 'reconnecting' | null>(null);
  // Phase 5d: rolling latency/drop metrics pushed every 5s by backend.
  const [metrics, setMetrics] = useState<{
    asr_finalize_ms: number | null;
    mt_ms: number | null;
    tts_ms: number | null;
    dropped_chunks: number;
    stale_drops: number;
    partial_count: number;
  } | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const pendingMessages = useRef<string[]>([]);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const currentSessionIdRef = useRef(sessionId);

  // TTS audio playback (PCM16 streamed from backend).
  const audioCtxRef = useRef<AudioContext | null>(null);
  const nextStartRef = useRef(0);              // gapless scheduling time
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const ttsSampleRateRef = useRef(48000);


  // Keep refs up-to-date without triggering re-renders/reconnections


  const flushPending = useCallback(() => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      while (pendingMessages.current.length > 0) {
        ws.send(pendingMessages.current.shift()!);
      }
    }
  }, []);

  const clearRetryTimer = useCallback(() => {
    if (retryTimerRef.current !== null) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  // Lazily create/replace the AudioContext so it matches the TTS sample rate.
  const ensureCtx = useCallback((sr: number) => {
    if (!audioCtxRef.current || audioCtxRef.current.sampleRate !== sr) {
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = new AudioContext({ sampleRate: sr });
      nextStartRef.current = 0;
      activeSourcesRef.current = [];
    }
    return audioCtxRef.current;
  }, []);

  // Interrupt any currently-playing/pending TTS audio (called on each tts_start).
  const stopAllSources = useCallback(() => {
    activeSourcesRef.current.forEach(s => {
      try { s.stop(); } catch { /* already ended */ }
    });
    activeSourcesRef.current = [];
    nextStartRef.current = 0;
  }, []);

  // Decode Int16 PCM -> Float32, schedule gaplessly via AudioBufferSourceNode.
  const playPcm = useCallback((buf: ArrayBuffer) => {
    const sr = ttsSampleRateRef.current;
    const ctx = ensureCtx(sr);
    const i16 = new Int16Array(buf);
    if (i16.length === 0) return;
    const f32 = new Float32Array(i16.length);
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;
    const audioBuf = ctx.createBuffer(1, f32.length, sr);
    audioBuf.copyToChannel(f32, 0);
    const src = ctx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime, nextStartRef.current);
    src.start(start);
    nextStartRef.current = start + audioBuf.duration;
    activeSourcesRef.current.push(src);
    src.onended = () => {
      activeSourcesRef.current = activeSourcesRef.current.filter(s => s !== src);
    };
  }, [ensureCtx]);

  const connect = useCallback(() => {
    // Guard: if sessionId changed while retrying, don't connect with old ID
    if (currentSessionIdRef.current !== sessionId) return;

    clearRetryTimer();

    const wsUrl = `${WS_BASE_URL}/ws/${sessionId}`;
    console.log(`[WS] Connecting to ${wsUrl} (attempt ${retryCountRef.current + 1})`);

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;
    retryCountRef.current += 1;

    ws.onopen = () => {
      console.log('[WS] Connected');
      retryCountRef.current = 0;
      setWsConnected(true);
      setErrorMessage(null);
      flushPending();
    };

    ws.onclose = (ev) => {
      console.log(`[WS] Disconnected (code=${ev.code}, reason=${ev.reason})`);
      setWsConnected(false);
      setErrorMessage('Mất kết nối tới server, đang thử lại…');
      // Schedule reconnect with exponential backoff
      const delay = Math.min(
        INITIAL_RETRY_DELAY_MS * Math.pow(2, retryCountRef.current - 1),
        MAX_RETRY_DELAY_MS
      );
      console.log(`[WS] Reconnecting in ${delay}ms...`);
      retryTimerRef.current = setTimeout(() => {
        if (currentSessionIdRef.current === sessionId) {
          connect();
        }
      }, delay);
    };

    ws.onerror = (err) => {
      console.error('[WS] Error:', err);
      setErrorMessage('Lỗi kết nối tới server.');
    };

    ws.onmessage = (event) => {
      // Binary frame = PCM16 TTS audio from the backend.
      if (event.data instanceof ArrayBuffer) {
        playPcm(event.data);
        return;
      }
      try {
        const data = JSON.parse(event.data as string);
        switch (data.type) {
          case 'partial_transcript':
            // LocalAgreement-2: confirmed (locked prefix) + partial (unstable suffix).
            // Fall back to splitting data.text if backend hasn't sent the new fields.
            if (data.confirmed !== undefined || data.partial !== undefined) {
              setConfirmedPartial(data.confirmed ?? '');
              setUnstablePartial(data.partial ?? '');
            } else {
              setConfirmedPartial('');
              setUnstablePartial(data.text ?? '');
            }
            break;
          case 'final_transcript':
            setUtterances(prev => [...prev, {
              uttId: data.utt_id,
              sourceLang: data.lang,
              sourceText: data.text,
              targetText: '',
              targetReady: false,
            }]);
            setConfirmedPartial('');
            setUnstablePartial('');
            setPartialTranslation('');
            break;
          case 'partial_translation':
            // Live preview translation (replaceable) — cleared on final_transcript.
            setPartialTranslation(data.text);
            break;
          case 'translation_delta':
            setUtterances(prev => prev.map(u =>
              u.uttId === data.utt_id
                ? { ...u, targetText: u.targetText + data.text_delta }
                : u
            ));
            break;
          case 'translation_done':
            setUtterances(prev => prev.map(u =>
              u.uttId === data.utt_id ? { ...u, targetReady: true } : u
            ));
            break;
          case 'tts_start':
            // New utterance audio: interrupt any pending playback, (re)create ctx
            // at the engine sample rate, and resume (browsers suspend w/o gesture).
            ttsSampleRateRef.current = data.sample_rate || 48000;
            ensureCtx(ttsSampleRateRef.current);
            stopAllSources();
            audioCtxRef.current?.resume().catch(() => {});
            break;
          case 'tts_end':
            // Sources self-stop; nothing to do.
            break;
          case 'status':
            setStatus(data.state);
            break;
          case 'asr_connection':
            // Phase 5b: RemoteASR GPU connection state (connected/disconnected/reconnecting).
            setAsrConnection(data.state);
            break;
          case 'metrics':
            // Phase 5d: rolling avg latency + drop counters (every 5s).
            setMetrics(data);
            break;
        }
      } catch (e) {
        console.error("Error parsing WS message", e);
      }
    };
  }, [sessionId, flushPending, clearRetryTimer, ensureCtx, stopAllSources, playPcm]);

  useEffect(() => {
    currentSessionIdRef.current = sessionId;
    retryCountRef.current = 0;
    connect();

    return () => {
      clearRetryTimer();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounting');
        wsRef.current = null;
      }
    };
  }, [sessionId, connect, clearRetryTimer]);

  const sendAudioChunk = (buf: ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buf);
    }
  };

  const sendControl = (msg: any) => {
    const json = JSON.stringify(msg);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(json);
    } else {
      pendingMessages.current.push(json);
    }
  };

  return { utterances, currentPartial, confirmedPartial, unstablePartial, partialTranslation, status, wsConnected, asrConnection, metrics, errorMessage, sendAudioChunk, sendControl };
}
