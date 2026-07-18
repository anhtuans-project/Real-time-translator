# VNAI — Dịch giọng nói thời gian thực (Tiếng Việt ↔ English) 🎙️🌐

Ứng dụng dịch giọng nói **2 chiều, thời gian thực**: nói tiếng Việt → ra text + bản dịch tiếng Anh + **phát audio tiếng Anh** (và ngược lại). Kiến trúc tách rời: phần nặng (**ASR + MT**) chạy trên **GPU Colab**, phần nhẹ (**VAD + TTS**) chạy **local CPU**, giao tiếp qua WebSocket + Cloudflare Tunnel. Không cần GPU trên máy local.

```
 🎙️ Mic (16kHz PCM16)
      │  (WebSocket binary)
      ▼
 ┌────────────── Local Backend (FastAPI, port 8000) ──────────────┐
 │  VAD (Silero, CPU) → ASR (Colab GPU, WS) → MT (Colab GPU, HTTP) │
 │                                            → TTS (Piper, CPU)  │
 └────────────────────────────┬───────────────────────────────────┘
                              │  partial_transcript / final_transcript
                              │  partial_translation (live preview)
                              │  translation_delta / translation_done
                              │  tts_start / PCM16 binary / tts_end
                              ▼
                    Frontend (React + Vite, port 5173)
                    ─ 2 cột Source / Translation
                    ─ phát audio TTS realtime (gapless, interrupt)
```

## ✨ Tính năng
- **ASR** — `faster-whisper` `large-v3` trên GPU Colab. Chống hallucinate (`condition_on_previous_text=False`, `hallucination_silence_threshold`, `no_speech_threshold`). Có partial transcript live + final `beam_size=1`.
- **MT** — `qwen2.5:7b` qua **Ollama** trên GPU Colab (proxy `/api/*` qua cùng tunnel). Streaming NDJSON, prompt dịch strict (chỉ dịch, không trả lời). Kèm **partial translation preview** trong lúc nói (debounce 700ms + cancel-in-flight).
- **TTS** — **Piper** (`piper1-gpl` / OHF-Voice fork) offline local CPU. Voice `vi_VN-vais1000-medium` (vi) + `en_US-ryan-medium` (en). Resample 48kHz, stream PCM16 về browser phát **gapless + interrupt** khi có câu mới.
- **VAD** — Silero VAD local CPU. Silence ~1s để chốt utterance, cap `MAX_UTT_S=6s` chống câu dài vô tận / garble.
- **Tunnel** — Cloudflare quick tunnel (`cloudflared`): free, không cần token, hỗ trợ WS + streaming ổn định.

## 📋 Yêu cầu
- **Python 3.10+** (test 3.11/3.12) + `venv`.
- **Node.js 18+** + npm (frontend).
- **Google Colab** runtime **T4 GPU** (free tier) để host ASR + MT. Không cần GPU local.
- Lần đầu tải ~250MB model Piper + ~3GB `whisper-large-v3` + ~4.5GB `qwen2.5:7b` trên Colab.

## 🗂 Cấu trúc
```
VNAI/
├── backend/
│   ├── main.py              # FastAPI WebSocket endpoint /ws/{session_id}
│   ├── session_state.py     # pipeline VAD→ASR→MT→TTS, partial translation preview
│   ├── engine_factory.py    # build engines theo .env (remote ASR / local / fake)
│   ├── asr_remote.py         # WS client → Colab ASR (non-blocking, send-queue)
│   ├── llm_engine.py         # OllamaTranslationEngine (qwen2.5:7b, strict prompt, timeout có bound)
│   ├── tts_en.py             # PiperTTSEngine (generic, dùng cho cả vi+en)
│   ├── vad.py                # Silero VAD (silence_chunks_to_end≈1s)
│   ├── models/               # voice Piper: *.onnx + *.onnx.json (vi + en)
│   └── ...
├── frontend/                # React + Vite + AudioWorklet (16kHz PCM16)
│   └── src/hooks/useTranslatorSocket.ts  # WS + phát audio TTS gapless
├── colab_asr_gpu.ipynb      # Notebook host ASR (large-v3) + Ollama proxy + cloudflared
├── .env.example             # template cấu hình
├── .env                     # (gitignored) cấu hình thực tế
└── requirements.txt
```

## 🚀 Cài đặt

### 1. Backend
```powershell
cd D:\VNAI
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # hoặc: .venv\Scripts\activate (cmd)
pip install -r requirements.txt
```

Model Piper TTS (vi+en) cần nằm trong `backend/models/`. Nếu thiếu, tải 4 file:
```powershell
cd backend\models
$BASE="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
curl -L -f -o en_US-ryan-medium.onnx          "$BASE/en/en_US/ryan/medium/en_US-ryan-medium.onnx"
curl -L -f -o en_US-ryan-medium.onnx.json      "$BASE/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
curl -L -f -o vi_VN-vais1000-medium.onnx       "$BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
curl -L -f -o vi_VN-vais1000-medium.onnx.json  "$BASE/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"
```

### 2. Frontend
```powershell
cd frontend
npm install
```

### 3. Cấu hình `.env`
```powershell
Copy-Item .env.example .env
```
Sửa `.env` (xem `.env.example` để biết ý nghĩa từng biến):
```env
# Để trống cả ASR_REMOTE_URL + MT_BASE_URL = chạy 100% local CPU.
# Điền URL Colab = offload ASR + MT lên GPU Colab (khuyến nghị).

ASR_REMOTE_URL=wss://xxxx-yyyy.trycloudflare.com/asr
MT_BASE_URL=https://xxxx-yyyy.trycloudflare.com
MT_MODEL=qwen2.5:7b

USE_FAKE_PIPELINE=
```

## ▶️ Cách chạy

### Bước 1 — Khởi động GPU Colab (ASR + MT)
Mở `colab_asr_gpu.ipynb` trên Colab, runtime **T4 GPU**, chạy tuần tự:
1. **Cell 1**: kiểm tra GPU.
2. **Cell 2**: cài deps (fastapi/uvicorn/faster-whisper/httpx).
3. **Cell 3**: ghi `asr_server.py` (whisper-large-v3 + proxy Ollama, header chống buffer).
4. **Cell 4**: khởi động uvicorn (kill uvicorn cũ nếu có).
5. **Cell 5**: xem log đến khi hiện `ASR ready (whisper-large-v3)`.
6. **Cell 6**: cài + chạy Ollama, `ollama pull qwen2.5:7b` (lần đầu ~2-3 phút).
7. **Cell 7**: cloudflared tunnel → in ra 3 giá trị:
   - `ASR_REMOTE_URL` → `wss://…trycloudflare.com/asr`
   - `MT_BASE_URL`  → `https://…trycloudflare.com`
   - `MT_MODEL`      → `qwen2.5:7b`

> URL Cloudflare đổi mỗi lần chạy cell 7 → cập nhật `.env` + restart backend local.

### Bước 2 — Cập nhật `.env`
Dán 3 giá trị từ cell 7 vào `.env` (như mẫu trên).

### Bước 3 — Khởi động backend local
```powershell
cd D:\VNAI
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
Log khi OK: `Successfully loaded Piper TTS model: …vi_VN-vais1000-medium.onnx` + `…en_US-ryan-medium.onnx`, rồi `AI engines loaded successfully.`

### Bước 4 — Khởi động frontend
```powershell
cd frontend
npm run dev
```
Mở `http://localhost:5173`, chọn chiều dịch (🇻🇳 Tiếng Việt → English hoặc 🇺🇸 English → Tiếng Việt), bấm **Start Mic**, cho phép micro, bắt đầu nói.

## 🧪 Các chế độ chạy

| Chế độ | ASR | MT | TTS | Khi nào dùng |
|---|---|---|---|---|
| **Colab GPU (khuyến nghị)** | large-v3 @ Colab | qwen2.5:7b @ Colab | Piper local | chạy thật, chất lượng cao |
| **Local CPU** | whisper-medium local | qwen2.5:1.5b local Ollama | Piper local | không có Colab (chậm) |
| **Fake** | giả lập | giả lập | giả lập | test UI/luồng, không cần model |

- **Colab GPU**: điền `ASR_REMOTE_URL` + `MT_BASE_URL` = URL Cloudflare (Bước 1-3). `MT_MODEL=qwen2.5:7b`.
- **Local CPU**: để `ASR_REMOTE_URL=` và `MT_BASE_URL=` trống. Cần Ollama local (`ollama serve`) + model `qwen2.5:1.5b` (`ollama pull qwen2.5:1.5b`).
- **Fake**: set `USE_FAKE_PIPELINE=1` (test UI không cần GPU/model).

## 🔧 Luồng pipeline (chi tiết)
1. **Capture**: browser `AudioWorklet` thu âm → 16kHz PCM16 → gửi binary chunk qua WS.
2. **VAD** (local): Silero chạy trên mỗi chunk (512 samples ≈ 32ms). Trạng thái `silence` / `speech_ongoing` / `utterance_end` (sau ~1s im lặng).
3. **ASR** (Colab): `speech_ongoing` → `RemoteASR` stream PCM lên Colab qua WS; Colab transcribe partial (beam_size=1) mỗi ~1.5s, đẩy về `partial_transcript`.
4. **Partial translation**: mỗi partial → debounce 700ms → dịch preview `partial_translation` (thay thế, cancel-in-flight).
5. **Finalize**: `utterance_end` (hoặc cap 6s) → cancel partial → `finalize` (beam_size=1) → `final_transcript` → dịch full → `translation_delta` (stream) → `translation_done`.
6. **TTS** (local, **ngoài `_pipeline_lock`**): `tts_start` (sample_rate) → stream PCM16 binary → `tts_end`. Browser phát gapless, interrupt khi có `tts_start` mới.

## 🩛 Khắc phục sự cố
- **`HTTP 502 Bad Gateway`** khi kết nối WS: server Colab chưa ready hoặc tunnel chết → chạy lại cell 5 (chờ `ASR ready`) + cell 7 (URL mới) → cập nhật `.env` → restart backend.
- **`đang dịch…` treo mãi**: Ollama trên Colab chưa lên / model chưa pull xong / tunnel buffer. Cell 3 đã thêm header `Cache-Control: no-cache` + `X-Accel-Buffering: no`. Backend có timeout (connect 10s, read 120s) nên không treo vĩnh viễn — sẽ báo `Error: …`.
- **Hallucinate ASR** ("hello", text chả liên quan từ đâu ra): đã chống bằng `condition_on_previous_text=False` + `hallucination_silence_threshold=2.0` + `no_speech_threshold=0.6`. Nếu vẫn còn → tăng `no_speech_threshold` trong `colab_asr_gpu.ipynb` cell 3.
- **Câu bị băm thành nhiều mảnh** ("đẩy xuống các thứ"): do `MAX_UTT_S` quá nhỏ. Hiện đã set `6.0`. Nếu vẫn bị → tăng trong `backend/session_state.py`.
- **Latency cao** ("đợi lâu mới thấy dịch"): `large-v3` chậm trên T4. Đổi cell 3 sang `WhisperModel("large-v3-turbo", …)` (nhanh hơn ~8×, cùng họ).
- **Audio TTS không phát**: browser suspend `AudioContext` → đã `ctx.resume()` trên `tts_start` (cần 1 user gesture trước — nút Start Mic đã là gesture). Kiểm tab DevTools Console lỗi decode PCM.
- **Giọng vi (vais1000) nghe robot**: đổi sang `vi_VN-vivos-x_low` (tải file tương ứng + sửa `VI_VOICE` trong `backend/engine_factory.py`), hoặc cân nhắc Edge-TTS cho vi (neural, chất lượng cao hơn, nhưng cần mạng).
- **URL Cloudflare đổi mỗi phiên**: bản chất của quick tunnel free. Mỗi lần Colab reconnect → copy URL mới vào `.env` → restart backend.

## 📦 Ghi chú
- `.env` gitignored (chứa URL tunnel, không phải secret nhưng nên giữ local).
- License Piper (`piper1-gpl`) = GPL v3.0; từng voice có license riêng (xem `MODEL_CARD` trên HF).
- ASR/MT trên Colab free tier: giới hạn thời gian session, GPU có thể bị recycle → giữ notebook mở + cell 7 chạy nền.
- Mọi xử lý AI trong `SessionState` chạy dưới dạng background task, không làm nghẽn vòng lặp `receive()` của WebSocket.