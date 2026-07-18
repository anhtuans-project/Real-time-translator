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
  const [currentPartial, setCurrentPartial] = useState('');
  const [partialTranslation, setPartialTranslation] = useState('');
  const [status, setStatus] = useState<'listening' | 'processing' | 'silence'>('silence');
  const [wsConnected, setWsConnected] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const pendingMessages = useRef<string[]>([]);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryCountRef = useRef(0);
  const currentSessionIdRef = useRef(sessionId);


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
      try {
        const data = JSON.parse(event.data as string);
        switch (data.type) {
          case 'partial_transcript':
            setCurrentPartial(data.text);
            break;
          case 'final_transcript':
            setUtterances(prev => [...prev, {
              uttId: data.utt_id,
              sourceLang: data.lang,
              sourceText: data.text,
              targetText: '',
              targetReady: false,
            }]);
            setCurrentPartial('');
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
          case 'status':
            setStatus(data.state);
            break;
        }
      } catch (e) {
        console.error("Error parsing WS message", e);
      }
    };
  }, [sessionId, flushPending, clearRetryTimer]);

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

  return { utterances, currentPartial, partialTranslation, status, wsConnected, errorMessage, sendAudioChunk, sendControl };
}
