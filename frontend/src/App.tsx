import { useState, useRef, useEffect } from 'react';
import { useAudioCapture } from './hooks/useAudioCapture';
import { useTranslatorSocket } from './hooks/useTranslatorSocket';

export default function App() {
  const [sessionId] = useState(`session-${Math.random().toString(36).substr(2, 9)}`);
  const {
    utterances,
    currentPartial,
    status,
    wsConnected,
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

  const sourceBottomRef = useRef<HTMLDivElement>(null);
  const targetBottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    sourceBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [utterances, currentPartial]);

  useEffect(() => {
    targetBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [utterances]);

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

  const showError = micError || errorMessage;
  const errorText = micError ?? errorMessage;

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100vh',
      backgroundColor: '#111827',
      color: 'white',
      padding: '2rem',
      fontFamily: 'sans-serif',
      boxSizing: 'border-box'
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '2rem'
      }}>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: 0 }}>Real-time Translator (React + Vite)</h1>
          <p style={{ color: '#9ca3af', fontSize: '0.875rem', margin: '4px 0 0 0' }}>Session: {sessionId}</p>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value as 'vi' | 'en')}
            disabled={isCapturing}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '0.5rem',
              border: '1px solid #4b5563',
              backgroundColor: '#374151',
              color: 'white',
              fontSize: '0.875rem',
              cursor: isCapturing ? 'not-allowed' : 'pointer',
              opacity: isCapturing ? 0.6 : 1
            }}
          >
            <option value="vi">🇻🇳 Tiếng Việt → English</option>
            <option value="en">🇺🇸 English → Tiếng Việt</option>
          </select>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: wsConnected ? '#22c55e' : '#ef4444',
              boxShadow: wsConnected ? '0 0 8px #22c55e' : '0 0 8px #ef4444'
            }} />
            <span style={{ textTransform: 'capitalize', fontSize: '0.875rem' }}>{wsConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{
              width: '12px',
              height: '12px',
              borderRadius: '50%',
              backgroundColor: status === 'listening' ? '#22c55e' : status === 'processing' ? '#eab308' : '#6b7280',
              boxShadow: status === 'listening' ? '0 0 8px #22c55e' : 'none'
            }} />
            <span style={{ textTransform: 'capitalize', fontSize: '0.875rem' }}>{status}</span>
          </div>
          <button
            onClick={handleToggleCapture}
            style={{
              padding: '0.5rem 1.5rem',
              borderRadius: '9999px',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 'medium',
              backgroundColor: isCapturing ? '#ef4444' : '#3b82f6',
              color: 'white',
              transition: 'background 0.2s'
            }}
          >
            {isCapturing ? 'Stop Mic' : 'Start Mic'}
          </button>
        </div>
      </div>

      {showError && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '1rem',
          backgroundColor: 'rgba(239, 68, 68, 0.15)',
          border: '1px solid #ef4444',
          color: '#fca5a5',
          padding: '0.75rem 1rem',
          borderRadius: '0.5rem',
          marginBottom: '1rem',
          fontSize: '0.875rem'
        }}>
          <span>⚠ {errorText}</span>
          {micError && (
            <button
              onClick={dismissError}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#fca5a5',
                cursor: 'pointer',
                fontSize: '0.8rem',
                textDecoration: 'underline'
              }}
            >
              Đóng
            </button>
          )}
        </div>
      )}

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '2rem',
        flex: 1,
        overflow: 'hidden'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#1f2937',
          borderRadius: '0.75rem',
          padding: '1.5rem',
          overflowY: 'auto'
        }}>
          <h2 style={{ fontSize: '1.125rem', fontWeight: 'semibold', marginBottom: '1rem', color: '#d1d5db', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
            Source ({sourceLang === 'vi' ? 'Tiếng Việt' : 'English'})
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {utterances.map((u) => (
              <div key={u.uttId} style={{ padding: '0.75rem', backgroundColor: '#374151', borderRadius: '0.5rem' }}>
                <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginBottom: '0.25rem', fontWeight: 'bold', textTransform: 'uppercase' }}>{u.sourceLang}</div>
                <div style={{ fontSize: '1.125rem' }}>{u.sourceText}</div>
              </div>
            ))}
            {currentPartial && (
              <div style={{ padding: '0.75rem', backgroundColor: 'rgba(55, 65, 81, 0.5)', borderRadius: '0.5rem', fontStyle: 'italic', color: '#9ca3af' }}>
                {currentPartial}...
              </div>
            )}
            <div ref={sourceBottomRef} />
          </div>
        </div>

        <div style={{
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: '#1f2937',
          borderRadius: '0.75rem',
          padding: '1.5rem',
          overflowY: 'auto'
        }}>
          <h2 style={{ fontSize: '1.125rem', fontWeight: 'semibold', marginBottom: '1rem', color: '#d1d5db', borderBottom: '1px solid #374151', paddingBottom: '0.5rem' }}>
            Translation ({sourceLang === 'vi' ? 'English' : 'Tiếng Việt'})
          </h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {utterances.map((u) => (
              <div key={u.uttId} style={{
                padding: '0.75rem',
                backgroundColor: 'rgba(30, 58, 138, 0.3)',
                border: '1px solid rgba(30, 64, 175, 0.5)',
                borderRadius: '0.5rem',
                opacity: u.targetReady ? 1 : 0.6
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                  <div style={{ fontSize: '0.75rem', color: '#60a5fa', fontWeight: 'bold', textTransform: 'uppercase' }}>{sourceLang === 'vi' ? 'en' : 'vi'}</div>
                  {!u.targetReady && (
                    <div style={{ fontSize: '0.7rem', color: '#93c5fd', fontStyle: 'italic' }}>đang dịch…</div>
                  )}
                </div>
                <div style={{ fontSize: '1.125rem', color: '#dbeafe' }}>{u.targetText}</div>
              </div>
            ))}
            <div ref={targetBottomRef} />
          </div>
        </div>
      </div>
    </div>
  );
}
