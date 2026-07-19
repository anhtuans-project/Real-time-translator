import { useState, useRef, useEffect } from 'react';
import { useAudioCapture } from './hooks/useAudioCapture';
import { useTranslatorSocket } from './hooks/useTranslatorSocket';

export default function App() {
  const [sessionId] = useState(`session-${Math.random().toString(36).substr(2, 9)}`);
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
    sendControl
  } = useTranslatorSocket(sessionId);

  const { start, stop } = useAudioCapture((buf) => {
    sendAudioChunk(buf);
  });

  const [isCapturing, setIsCapturing] = useState(false);
  const [sourceLang, setSourceLang] = useState<'vi' | 'en'>('vi');
  const [micError, setMicError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (localStorage.getItem('theme') as 'dark' | 'light') || 'dark');

  useEffect(() => {
    localStorage.setItem('theme', theme);
  }, [theme]);

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
          target_lang: sourceLang === 'vi' ? 'en' : 'vi'
        });
        setIsCapturing(true);
      } catch (err) {
        console.error('Mic start failed', err);
        setMicError(err instanceof Error ? err.message : 'Không truy cập được micro');
        setIsCapturing(false);
      }
    }
  };

  const dismissError = () => {
    setMicError(null);
  };

  const handleCopySessionId = () => {
    navigator.clipboard.writeText(sessionId);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const showError = micError || errorMessage;
  const errorText = micError ?? errorMessage;

  return (
    <div className={`app-container ${theme === 'light' ? 'light-theme' : ''}`}>
      {/* Background ambient glowing shapes */}
      <div className="ambient-glow ambient-glow-indigo" />
      <div className="ambient-glow ambient-glow-emerald" />

      {/* Main Glass Header */}
      <header className="glass-panel app-header">
        <div className="brand-section">
          <div className="brand-title-group">
            <h1 className="brand-title">VocalSync AI</h1>
            <div className="session-info">
              <span>ID: {sessionId}</span>
              <button
                onClick={handleCopySessionId}
                className="session-copy-btn"
                title="Copy Session ID"
              >
                {copied ? (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--color-emerald)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                ) : (
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        <div className="controls-group">
          {/* Language selection dropdown */}
          <div className="dropdown">
            <button 
              className="lang-dropdown" 
              style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'space-between',
                gap: '0.5rem',
                textAlign: 'left'
              }}
              disabled={isCapturing}
            >
              <span>{sourceLang === 'vi' ? '🇻🇳 Tiếng Việt → English' : '🇺🇸 English → Tiếng Việt'}</span>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.8 }}>
                <path d="m6 9 6 6 6-6"/>
              </svg>
            </button>
            <div className="dropdown-content">
              <a 
                href="#" 
                onClick={(e) => {
                  e.preventDefault();
                  if (!isCapturing) setSourceLang('vi');
                }}
              >
                🇻🇳 Tiếng Việt → English
              </a>
              <a 
                href="#" 
                onClick={(e) => {
                  e.preventDefault();
                  if (!isCapturing) setSourceLang('en');
                }}
              >
                🇺🇸 English → Tiếng Việt
              </a>
            </div>
          </div>

          {/* Connection Pills */}
          <div className="status-pill" title="Websocket Connection">
            <div className={`status-dot ${wsConnected ? 'active' : 'inactive'}`} />
            <span>Server</span>
          </div>

          <div className="status-pill" title="System state">
            <div className={`status-dot ${status === 'listening' ? 'active' : status === 'processing' ? 'pending' : 'inactive'
              }`} />
            <span>{status}</span>
          </div>

          {/* Waveform indicator when listening */}
          {isCapturing && status === 'listening' && (
            <div className="wave-indicator">
              <div className="wave-bar" />
              <div className="wave-bar" />
              <div className="wave-bar" />
              <div className="wave-bar" />
              <div className="wave-bar" />
            </div>
          )}

          {/* Theme Toggle Button */}
          <button 
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className={`btn-slide ${theme === 'light' ? 'light' : ''}`}
            title={theme === 'dark' ? 'Chuyển sang Giao diện Sáng' : 'Chuyển sang Giao diện Tối'}
          >
            <span className="circle">
              {theme === 'dark' ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="4" />
                  <path d="M12 2v2" />
                  <path d="M12 20v2" />
                  <path d="m4.93 4.93 1.41 1.41" />
                  <path d="m17.66 17.66 1.41 1.41" />
                  <path d="M2 12h2" />
                  <path d="M20 12h2" />
                  <path d="m6.34 17.66-1.41 1.41" />
                  <path d="m19.07 4.93-1.41 1.41" />
                </svg>
              )}
            </span>
          </button>

          {/* Beautiful pulsing microphone trigger */}
          <button
            onClick={handleToggleCapture}
            className={`mic-toggle-btn ${isCapturing ? 'active' : 'idle'}`}
          >
            {isCapturing ? (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="4" y="4" width="16" height="16" rx="2" />
                </svg>
                <span className="title">Stop Mic</span>
              </>
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" x2="12" y1="19" y2="22" />
                </svg>
                <span className="title">Start Mic</span>
              </>
            )}
          </button>
        </div>
      </header>

      {/* Error Toasts */}
      {showError && (
        <div className="alert-toast alert-toast-danger">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" x2="12" y1="8" y2="12" />
              <line x1="12" x2="12.01" y1="16" y2="16" />
            </svg>
            <span>{errorText}</span>
          </div>
          {micError && (
            <button onClick={dismissError} className="alert-close-btn">
              Đóng
            </button>
          )}
        </div>
      )}

      {/* Phase 5b: RemoteASR (GPU server) connection banner — riêng với FE↔BE ws. */}
      {asrConnection && asrConnection !== 'connected' && (
        <div className="alert-toast alert-toast-warning">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" x2="12" y1="9" y2="13" />
              <line x1="12" x2="12.01" y1="17" y2="17" />
            </svg>
            <span>
              {asrConnection === 'reconnecting'
                ? 'Mất kết nối ASR GPU, đang kết nối lại…'
                : 'ASR GPU đang ngắt — sẽ kết nối lại khi có audio.'}
            </span>
          </div>
        </div>
      )}

      {/* Main Workspace Panels */}
      <div className="workspace-grid">
        {/* Source Column */}
        <div className="glass-panel stream-column">
          <div className="column-header">
            <div className="column-icon">🎙</div>
            <h2 className="column-title">Nguồn (Source)</h2>
            <span className="column-subtitle">
              {sourceLang === 'vi' ? 'Tiếng Việt 🇻🇳' : 'English 🇺🇸'}
            </span>
          </div>

          <div className="message-feed">
            {utterances.map((u) => (
              <div key={u.uttId} className="utterance-card">
                <div className="utterance-card-header">
                  <span className={`lang-badge ${u.sourceLang}`}>
                    {u.sourceLang}
                  </span>
                </div>
                <div className="utterance-text">{u.sourceText}</div>
              </div>
            ))}

            {currentPartial && (
              <div className="partial-card">
                <span className="partial-text-confirmed">{confirmedPartial}</span>
                {unstablePartial && (
                  <span className="partial-text-unstable"> {unstablePartial}</span>
                )}
                <span className="partial-dots">
                  <span className="partial-dot" />
                  <span className="partial-dot" />
                  <span className="partial-dot" />
                </span>
              </div>
            )}
            <div ref={sourceBottomRef} />
          </div>
        </div>

        {/* Translation Column */}
        <div className="glass-panel stream-column">
          <div className="column-header">
            <div className="column-icon">🌐</div>
            <h2 className="column-title">Bản dịch (Translation)</h2>
            <span className="column-subtitle">
              {sourceLang === 'vi' ? 'English 🇺🇸' : 'Tiếng Việt 🇻🇳'}
            </span>
          </div>

          <div className="message-feed">
            {utterances.map((u) => (
              <div key={u.uttId} className="utterance-card">
                <div className="utterance-card-header">
                  <span className={`lang-badge ${sourceLang === 'vi' ? 'en' : 'vi'}`}>
                    {sourceLang === 'vi' ? 'en' : 'vi'}
                  </span>
                  {!u.targetReady && (
                    <span className="utterance-status translating">đang dịch…</span>
                  )}
                </div>
                <div className={`utterance-text ${u.targetReady ? '' : 'pending'}`}>
                  {u.targetText}
                </div>
              </div>
            ))}

            {partialTranslation && (
              <div className="partial-card">
                <span className="partial-text-unstable">{partialTranslation}</span>
                <span className="partial-dots">
                  <span className="partial-dot" />
                  <span className="partial-dot" />
                  <span className="partial-dot" />
                </span>
              </div>
            )}
            <div ref={targetBottomRef} />
          </div>
        </div>
      </div>

      {/* Telemetry Metrics Panel */}
      {metrics && (
        <footer className="glass-panel telemetry-panel">
          <div className="telemetry-header">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" x2="18" y1="20" y2="10" />
              <line x1="12" x2="12" y1="20" y2="4" />
              <line x1="6" x2="6" y1="20" y2="14" />
            </svg>
            <h3 className="telemetry-title">Hệ thống đo lường (Telemetry Metrics)</h3>
          </div>

          <div className="telemetry-grid">
            <div className="telemetry-item">
              <span className="telemetry-label">Độ trễ ASR</span>
              <span className={`telemetry-value ${metrics.asr_finalize_ms == null ? '' : metrics.asr_finalize_ms < 300 ? 'good' : 'warn'
                }`}>
                {metrics.asr_finalize_ms != null ? `${Math.round(metrics.asr_finalize_ms)}ms` : '—'}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Dịch thuật MT</span>
              <span className={`telemetry-value ${metrics.mt_ms == null ? '' : metrics.mt_ms < 200 ? 'good' : 'warn'
                }`}>
                {metrics.mt_ms != null ? `${Math.round(metrics.mt_ms)}ms` : '—'}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Phát âm TTS</span>
              <span className={`telemetry-value ${metrics.tts_ms == null ? '' : metrics.tts_ms < 400 ? 'good' : 'warn'
                }`}>
                {metrics.tts_ms != null ? `${Math.round(metrics.tts_ms)}ms` : '—'}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Partials</span>
              <span className="telemetry-value">{metrics.partial_count}</span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">Dropped Chunks</span>
              <span className={`telemetry-value ${metrics.dropped_chunks > 0 ? 'error' : ''}`}>
                {metrics.dropped_chunks}
              </span>
            </div>

            <div className="telemetry-item">
              <span className="telemetry-label">GPU Stale Drops</span>
              <span className={`telemetry-value ${metrics.stale_drops > 0 ? 'error' : ''}`}>
                {metrics.stale_drops}
              </span>
            </div>
          </div>
        </footer>
      )}
    </div>
  );
}
