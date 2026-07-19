import { useState, useRef, useEffect } from 'react';
import { useAudioCapture } from './hooks/useAudioCapture';
import { useTranslatorSocket } from './hooks/useTranslatorSocket';
import './App.css';

type SourceLang = 'vi' | 'en';

const LANG_LABEL: Record<SourceLang, string> = { vi: 'Tiếng Việt', en: 'English' };
const LANG_FLAG: Record<SourceLang, string> = { vi: '🇻🇳', en: '🇺🇸' };

export default function App() {
  const [sessionId] = useState(`session-${Math.random().toString(36).slice(2, 11)}`);
  const {
    utterances,
    currentPartial,
    confirmedPartial,
    unstablePartial,
    partialTranslation,
    status,
    wsConnected,
    asrConnection,
    metrics,
    errorMessage,
    sendAudioChunk,
    sendControl,
  } = useTranslatorSocket(sessionId);

  const { start, stop } = useAudioCapture((buf) => {
    sendAudioChunk(buf);
  });

  const [isCapturing, setIsCapturing] = useState(false);
  const [sourceLang, setSourceLang] = useState<SourceLang>('vi');
  const [micError, setMicError] = useState<string | null>(null);
  const [faceToFace, setFaceToFace] = useState(false);

  const targetLang: SourceLang = sourceLang === 'vi' ? 'en' : 'vi';
  const sourceBottomRef = useRef<HTMLDivElement>(null);
  const targetBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sourceBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [utterances, currentPartial]);

  useEffect(() => {
    targetBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [utterances, partialTranslation]);

  const handleToggleCapture = async () => {
    setMicError(null);
    if (isCapturing) {
      stop();
      setIsCapturing(false);
    } else {
      try {
        await start();
        sendControl({
          type: 'start_session',
          source_lang: sourceLang,
          target_lang: targetLang,
        });
        setIsCapturing(true);
      } catch (err) {
        console.error('Mic start failed', err);
        setMicError(err instanceof Error ? err.message : 'Không truy cập được micro');
        setIsCapturing(false);
      }
    }
  };

  const onKeyToggle = (e: React.KeyboardEvent) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      handleToggleCapture();
    }
  };

  const dismissError = () => setMicError(null);
  const showError = micError || errorMessage;
  const errorText = micError ?? errorMessage;

  const listening = isCapturing && status === 'listening';
  const processing = isCapturing && status === 'processing';
  const micState: 'idle' | 'listening' | 'processing' =
    listening ? 'listening' : processing ? 'processing' : 'idle';

  return (
    <div className="app">
      {/* Status announced to screen readers (visually hidden). */}
      <div className="sr-only" role="status" aria-live="polite">
        {wsConnected ? 'Đã kết nối server.' : 'Mất kết nối server.'}
        {isCapturing ? ` Đang ${micState === 'listening' ? 'lắng nghe' : 'xử lý'}.` : ''}
        {asrConnection && asrConnection !== 'connected' ? ' ASR GPU đang kết nối lại.' : ''}
      </div>

      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden>🎙️</span>
          <div className="brand-text">
            <h1>VNAI</h1>
            <span className="brand-sub">Real-time Translator · Vi ↔ En</span>
          </div>
        </div>

        <div className="lang-switch" role="group" aria-label="Chiều dịch">
          <span className="lang-side">
            <span className="lang-flag" aria-hidden>{LANG_FLAG[sourceLang]}</span>
            <span className="lang-name">{LANG_LABEL[sourceLang]}</span>
          </span>
          <button
            className="lang-swap"
            onClick={() => setSourceLang((v) => (v === 'vi' ? 'en' : 'vi'))}
            disabled={isCapturing}
            aria-label="Đảo chiều dịch"
            title="Đảo chiều dịch"
          >
            <span className="lang-swap-icon" aria-hidden>⇄</span>
          </button>
          <span className="lang-side">
            <span className="lang-flag" aria-hidden>{LANG_FLAG[targetLang]}</span>
            <span className="lang-name">{LANG_LABEL[targetLang]}</span>
          </span>
        </div>

        <div className="status-pills">
          <span className={`pill ${wsConnected ? 'pill-ok' : 'pill-bad'}`} title="Kết nối tới backend">
            <span className="pill-dot" />
            {wsConnected ? 'Server' : 'Mất server'}
          </span>
          <span
            className={`pill ${asrConnection === 'connected' ? 'pill-ok' : asrConnection ? 'pill-warn' : 'pill-muted'}`}
            title="Kết nối tới GPU ASR"
          >
            <span className="pill-dot" />
            ASR GPU
          </span>
        </div>
      </header>

      {(showError || (asrConnection && asrConnection !== 'connected')) && (
        <div className="banners">
          {showError && (
            <div className="banner banner-error" role="alert">
              <span>⚠ {errorText}</span>
              {micError && (
                <button className="banner-close" onClick={dismissError}>Đóng</button>
              )}
            </div>
          )}
          {asrConnection && asrConnection !== 'connected' && (
            <div className="banner banner-warn">
              <span>⏳ {asrConnection === 'reconnecting'
                ? 'Mất kết nối ASR GPU, đang kết nối lại…'
                : 'ASR GPU đang ngắt — sẽ kết nối lại khi có audio.'}</span>
            </div>
          )}
        </div>
      )}

      <main className="panels">
        <section className="panel panel-source">
          <div className="panel-head">
            <span className="panel-flag" aria-hidden>{LANG_FLAG[sourceLang]}</span>
            <h2>Source · {LANG_LABEL[sourceLang]}</h2>
          </div>
          <div className="panel-body">
            {utterances.map((u) => (
              <article key={u.uttId} className="bubble bubble-source">
                <div className="bubble-lang">{u.sourceLang}</div>
                <div className="bubble-text">{u.sourceText}</div>
              </article>
            ))}
            {currentPartial && (
              <div className="bubble bubble-source bubble-live">
                <span className="bubble-confirmed">{confirmedPartial}</span>
                {unstablePartial && <span className="bubble-unstable"> {unstablePartial}</span>}
                <span className="bubble-cursor" aria-hidden>…</span>
              </div>
            )}
            <div ref={sourceBottomRef} />
          </div>
        </section>

        <section className={`panel panel-target ${faceToFace ? 'flip' : ''}`}>
          <div className="panel-head">
            <span className="panel-flag" aria-hidden>{LANG_FLAG[targetLang]}</span>
            <h2>Translation · {LANG_LABEL[targetLang]}</h2>
            <button
              className={`flip-btn ${faceToFace ? 'on' : ''}`}
              onClick={() => setFaceToFace((v) => !v)}
              aria-pressed={faceToFace}
              title="Xoay 180° cho người đối diện đọc"
            >
              🪞
            </button>
          </div>
          <div className="panel-body">
            {utterances.map((u) => (
              <article key={u.uttId} className={`bubble bubble-target ${u.targetReady ? 'ready' : 'pending'}`}>
                <div className="bubble-lang">{targetLang}</div>
                <div className="bubble-text">{u.targetText || (u.targetReady ? '' : '…')}</div>
                {!u.targetReady && <div className="typing" aria-label="đang dịch"><i /><i /><i /></div>}
              </article>
            ))}
            {partialTranslation && (
              <div className="bubble bubble-target bubble-preview">
                <span className="bubble-text">{partialTranslation}…</span>
              </div>
            )}
            <div ref={targetBottomRef} />
          </div>
        </section>
      </main>

      <footer className="dock">
        <div className="dock-mic">
          <button
            className={`mic mic-${micState} ${isCapturing ? 'on' : ''}`}
            onClick={handleToggleCapture}
            onKeyDown={onKeyToggle}
            aria-pressed={isCapturing}
            aria-label={isCapturing ? 'Dừng micro' : 'Bắt đầu micro'}
          >
            <span className="mic-icon" aria-hidden>{isCapturing ? '■' : '🎤'}</span>
            {listening && <span className="mic-ring" aria-hidden />}
          </button>
          <div className="dock-state">
            <div className={`state state-${micState}`}>
              <span className="state-dot" />
              <span className="state-label">
                {micState === 'listening' ? 'Listening' : micState === 'processing' ? 'Processing' : 'Idle'}
              </span>
            </div>
            {listening && (
              <div className="wave" aria-hidden>
                <i /><i /><i /><i /><i /><i /><i /><i /><i />
              </div>
            )}
          </div>
        </div>

        <div className="dock-meta">
          <span className="meta-label">Latency</span>
          <div className="metrics">
            <span>ASR {metrics?.asr_finalize_ms != null ? `${Math.round(metrics.asr_finalize_ms)}ms` : '—'}</span>
            <span>MT {metrics?.mt_ms != null ? `${Math.round(metrics.mt_ms)}ms` : '—'}</span>
            <span>TTS {metrics?.tts_ms != null ? `${Math.round(metrics.tts_ms)}ms` : '—'}</span>
            <span className="muted">partials {metrics?.partial_count ?? 0}</span>
            {!!metrics?.dropped_chunks && <span className="warn">⚠ dropped {metrics.dropped_chunks}</span>}
            {!!metrics?.stale_drops && <span className="warn">⚠ GPU stale {metrics.stale_drops}</span>}
          </div>
          <span className="session-id">session: {sessionId}</span>
        </div>
      </footer>
    </div>
  );
}