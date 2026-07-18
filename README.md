# Real-time Communication Layer (WebSocket) 🎙️🌐

Hệ thống truyền tải dữ liệu thời gian thực phục vụ dịch thuật AI: **Mic → Backend (ASR, MT, TTS) → Dual-panel hiển thị & Phát âm thanh**.

## 🚀 Hướng dẫn chạy nhanh

### 1. Yêu cầu hệ thống
- **Backend**: Python 3.10+
- **Frontend**: Node.js 18+ & npm
- **GPU (Khuyến khích)**: NVIDIA GPU (GTX 1650 4GB+) để chạy pipeline streaming.

### 2. Chạy Backend (FastAPI)

Hệ thống có 2 chế độ hoạt động:

**A. Chế độ Mock (Giả lập - Không cần GPU)**
Dùng để kiểm tra luồng dữ liệu, giao diện mà không cần tải model hay dùng GPU.
- **Windows (PowerShell)**: `$env:USE_FAKE_PIPELINE="1"; python -m backend.main`
- **Windows (CMD)**: `set USE_FAKE_PIPELINE=1 && python -m backend.main`  `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 `

**B. Chế độ Real (Sử dụng Model thật - Cần GPU)**
Dùng để chạy thực tế với các model đã tải về (Whisper, Qwen, VieNeu).
- **Windows (PowerShell)**: `$env:USE_FAKE_PIPELINE="0"; python -m backend.main`
- **Windows (CMD)**: `set USE_FAKE_PIPELINE=0 && python -m backend.main`

*Server sẽ chạy tại: `http://localhost:8000`*

### 3. Chạy Frontend (React + Vite)
```bash
# Di chuyển vào thư mục frontend
cd frontend

# Cài đặt dependencies
npm install

# Khởi chạy chế độ phát triển
npm run dev
```
*Giao diện sẽ chạy tại: `http://localhost:5173` (hoặc port do Vite cung cấp)*

---

## 🛠 Luồng hoạt động (Data Pipeline)

### 1. Pipeline chuẩn (High-Level)
1. **Capture**: Trình duyệt thu âm qua `AudioWorklet` $\rightarrow$ Downsample về **16kHz PCM16** $\rightarrow$ Gửi binary chunk qua WebSocket.
2. **Processing (Backend)**:
   - **VAD**: Phát hiện tiếng nói $\rightarrow$ Kích hoạt ASR.
   - **ASR**: Chuyển âm thanh thành văn bản (Partial $\rightarrow$ Final).
   - **MT**: Dịch văn bản theo luồng (Streaming Delta $\rightarrow$ Done).
   - **TTS**: Tổng hợp giọng nói từ văn bản dịch $\rightarrow$ Gửi binary **48kHz PCM16**.
3. **Render**: 
   - **Dual-panel**: Hiển thị văn bản gốc (trái) và văn bản dịch (phải).
   - **Audio Queue**: Phát âm thanh tuần tự, không chồng lấn.

### 2. Pipeline Streaming Tối ưu (GTX 1650 4GB)
Hệ thống hiện hỗ trợ cấu hình tối ưu cho VRAM thấp để đạt độ trễ 0.8s - 1.5s:

**Kiến trúc luồng:**
`Mic` $\rightarrow$ `Silero VAD` $\rightarrow$ `faster-whisper-small (Streaming)` $\rightarrow$ `Qwen2.5-1.5B GGUF (Streaming)` $\rightarrow$ `VieNeu V3 / Kokoro-82M`.

**Chi tiết Model:**
| Thành phần | Model | Chế độ | VRAM (Ước tính) |
| :--- | :--- | :--- | :--- |
| **VAD** | Silero VAD | CPU | ~100MB |
| **STT** | `faster-whisper-small` | GPU (FP16) | ~1GB |
| **Translation** | `Qwen2.5-1.5B-Instruct-Q4_K_M` | GPU (GGUF) | ~1.5-2GB |
| **TTS (VI)** | `VieNeu-TTS-v3-Turbo` | CPU | ~500MB |
| **TTS (EN)** | `Kokoro-82M` | CPU | ~200MB |

## ⚙️ Cấu hình kỹ thuật
| Thành phần | Thông số | Ghi chú |
|---|---|---|
| Audio Input | PCM16, Mono, 16kHz | Binary WebSocket frame |
| Audio Output | PCM16, 48kHz | Web Audio API Buffer |
| Chunk size | ~300ms | Tối ưu latency và băng thông |
| AI Engines | Interface-based | Dễ dàng swap giữa Mock và Real (Piper, VieNeu, Whisper) |

## 📝 Ghi chú cho Developer
- Để chuyển sang dùng model thật, hãy thay đổi `USE_FAKE_PIPELINE=0` và cài đặt các wrapper trong thư mục `backend/`.
- **Lưu ý về `llama-cpp-python`**: Cần cài đặt **Visual Studio Build Tools** (Desktop development with C++) để biên dịch thư viện này trên Windows.
- Mọi xử lý AI trong `SessionState` đều chạy dưới dạng background task, không làm nghẽn vòng lặp `receive()` của WebSocket.
